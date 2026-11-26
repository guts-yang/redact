"""SCOPE: Step-wise Scope-Consistent Equivalence-Class Unlearning.

Core reference implementation of proposal IDEA_06. In-scope CoT steps receive a
suppression loss; out-of-scope steps are anchored with a KL to the reference
model. The SCS metric measures the worst step over each equivalence class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .base import MethodOutput


@dataclass
class SCOPEConfig:
    lam: float = 1.0
    epsilon: float = 0.05
    step_temperature: float = 1.0


def step_entity_probability(step_logits: torch.Tensor,
                            entity_ids: torch.Tensor) -> torch.Tensor:
    """p(e_f | r_<=t) for step logits ``[B, T, V]``, returns ``[B, T]``."""
    if step_logits.dim() != 3:
        raise ValueError("step_logits must be [B, T, V]")
    probs = F.softmax(step_logits, dim=-1)
    ids = entity_ids.view(-1, 1, 1).expand(-1, step_logits.shape[1], -1)
    return probs.gather(2, ids).squeeze(2)


def scope_supervision_loss(step_logits: torch.Tensor,
                           entity_ids: torch.Tensor) -> torch.Tensor:
    """L_sup = -log(1 - p(e_f | r_<=t)), averaged over all steps."""
    p = step_entity_probability(step_logits, entity_ids)
    return -torch.log1p(-p.clamp(max=1.0 - 1e-6)).mean()


def scope_keep_kl(step_logits: torch.Tensor,
                  ref_step_logits: torch.Tensor) -> torch.Tensor:
    """Per-step KL(p_theta || p_ref), averaged over steps and batch."""
    logp = F.log_softmax(step_logits, dim=-1)
    logq = F.log_softmax(ref_step_logits, dim=-1)
    per = (logp.exp() * (logp - logq)).sum(dim=-1)
    return per.mean()


def scope_loss(
    step_logits: torch.Tensor,
    ref_step_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    scope_mask: torch.Tensor,
    retain_kl: torch.Tensor,
    config: Optional[SCOPEConfig] = None,
) -> MethodOutput:
    """Sum_t [g_t L_sup(r_t) + (1-g_t) L_keep(r_t)] + lam*KL_retain.

    ``scope_mask`` is ``[B, T]`` with 1 for in-scope steps.
    """
    cfg = config or SCOPEConfig()
    if step_logits.shape != ref_step_logits.shape:
        raise ValueError("step_logits and ref_step_logits must match")
    if scope_mask.shape != step_logits.shape[:2]:
        raise ValueError("scope_mask must be [B, T]")
    p = step_entity_probability(step_logits, entity_ids)
    sup = -torch.log1p(-p.clamp(max=1.0 - 1e-6))
    logp = F.log_softmax(step_logits, dim=-1)
    logq = F.log_softmax(ref_step_logits, dim=-1)
    keep = (logp.exp() * (logp - logq)).sum(dim=-1)
    g = scope_mask.to(step_logits.dtype)
    per_step = g * sup + (1.0 - g) * keep
    total = per_step.sum(dim=1).mean() + cfg.lam * retain_kl
    return MethodOutput(loss=total, diagnostics={
        "scope_sup_mean": sup.detach().mean(),
        "scope_keep_mean": keep.detach().mean(),
        "scope_inscope_fraction": g.detach().mean(),
    })


def step_scope_consistency(leak_matrix: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """SCS = mean over equivalence classes of min_t 1[leak_{e,t} <= eps].

    ``leak_matrix`` is ``[E, T]``; each row is one equivalent question.
    """
    if leak_matrix.dim() != 2:
        raise ValueError("leak_matrix must be [E, T]")
    ok = (leak_matrix <= epsilon).float()
    return ok.min(dim=1).values.mean()


def worst_step_leak(leak_matrix: torch.Tensor) -> torch.Tensor:
    return leak_matrix.max(dim=1).values.mean()


class SCOPE:
    def __init__(self, config: Optional[SCOPEConfig] = None):
        self.config = config or SCOPEConfig()

    def loss(self, step_logits, ref_step_logits, entity_ids, scope_mask, retain_kl):
        return scope_loss(step_logits, ref_step_logits, entity_ids, scope_mask,
                          retain_kl, self.config)

    @staticmethod
    def scs(leak_matrix, epsilon=0.05):
        return step_scope_consistency(leak_matrix, epsilon)
