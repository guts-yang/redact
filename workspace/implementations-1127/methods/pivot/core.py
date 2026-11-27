"""PIVOT core: de-pivoting cross-lingual CoT unlearning (IDEA_02_PIVOT-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  pivot transfer leak rho(Lq,Lp,Lo), DPC (de-pivot consistency),
  cross-lingual step-level scope loss, shared-layer consistency constraint.
Tensors are synthetic-friendly for CPU tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PivotConfig:
    lam: float = 1.0     # retain KL
    gamma: float = 0.5   # shared-layer consistency
    eps: float = 0.05    # leak threshold for DPC indicator


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def de_pivot_consistency(step_leak: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """DPC = mean over (Lq,Lp) of max_t 1[Leak(r_t) > eps]; step_leak [Q,P,T]."""
    if step_leak.dim() != 3:
        raise ValueError("step_leak must be [Q,P,T]")
    return (step_leak.max(dim=-1).values > eps).float().mean()


def step_scope_loss(
    step_logits: torch.Tensor,      # [N,T,V] current model logits per step
    ref_step_logits: torch.Tensor,  # [N,T,V] frozen theta_0 logits
    entity_ids: torch.Tensor,       # [N]
    g: torch.Tensor,                # [N,T] in-scope (1) / out-of-scope (0) labels
    config: PivotConfig | None = None,
) -> torch.Tensor:
    """L_sup = -log(1-p(entity|step)) on in-scope; L_keep = KL on out-of-scope."""
    cfg = config or PivotConfig()
    lp = F.log_softmax(step_logits, dim=-1)
    ent = lp.gather(-1, entity_ids.view(-1, 1, 1).expand(-1, step_logits.shape[1], 1)).squeeze(-1)
    p_ent = ent.exp()
    l_sup = -torch.log((1.0 - p_ent).clamp_min(1e-8))
    lq = F.log_softmax(ref_step_logits, dim=-1)
    l_keep = (lp.exp() * (lp - lq)).sum(-1)
    per = g * l_sup + (1.0 - g) * l_keep
    return per.mean()


def pivot_loss(
    step_logits: torch.Tensor,
    ref_step_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    g: torch.Tensor,
    retain_kl: torch.Tensor,
    shared_hidden: torch.Tensor,       # [N,H] mid-layer reps of current model
    ref_shared_hidden: torch.Tensor,   # [N,H] theta_0 mid-layer reps
    step_leak: torch.Tensor | None = None,
    config: PivotConfig | None = None,
) -> Out:
    cfg = config or PivotConfig()
    scope = step_scope_loss(step_logits, ref_step_logits, entity_ids, g, cfg)
    shared = (shared_hidden - ref_shared_hidden).pow(2).sum(-1).mean()
    loss = scope + cfg.lam * retain_kl + cfg.gamma * shared
    diag = {"pivot_scope": scope.detach(), "pivot_shared": shared.detach(),
            "pivot_retain_kl": retain_kl.detach()}
    if step_leak is not None:
        diag["pivot_DPC"] = de_pivot_consistency(step_leak, cfg.eps).detach()
    return Out(loss=loss, diagnostics=diag)


class PIVOT:
    def __init__(self, config: PivotConfig | None = None):
        self.config = config or PivotConfig()

    def loss(self, *args, **kwargs):
        return pivot_loss(*args, config=self.config, **kwargs)

    def dpc(self, step_leak):
        return de_pivot_consistency(step_leak, self.config.eps)
