"""DISCO: Decoding-Invariant Suppression of context-free revival.

Core reference implementation of proposal IDEA_04. The method treats decoding
policies as environments and minimizes the worst-policy entity margin while
also suppressing the bare-prompt (context-free) revival channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .base import MethodOutput, entity_leak_probability, entity_margin_loss


@dataclass
class DISCOConfig:
    margin: float = 0.0
    tau: float = 0.0
    lam: float = 1.0
    beta: float = 1.0
    gamma: float = 0.0


def policy_margin_losses(
    policy_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return per-policy ``max(0, logit_entity - margin)`` means, shape [P]."""
    if policy_logits.dim() != 3:
        raise ValueError("policy_logits must be [P, B, V]")
    losses = []
    for p in range(policy_logits.shape[0]):
        losses.append(entity_margin_loss(policy_logits[p], entity_ids, margin))
    return torch.stack(losses)


def worst_policy_margin(
    policy_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    return policy_margin_losses(policy_logits, entity_ids, margin).max()


def context_free_loss(
    bare_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    tau: float = 0.0,
) -> torch.Tensor:
    """-log sigmoid(-(logit_e - tau)) on the bare prompt, averaged over B."""
    if bare_logits.dim() != 2:
        raise ValueError("bare_logits must be [B, V]")
    ent = bare_logits.gather(1, entity_ids.view(-1, 1)).squeeze(1)
    return -F.logsigmoid(-(ent - tau)).mean()


def decoding_separation(leak_cot: torch.Tensor, leak_bare: torch.Tensor) -> torch.Tensor:
    """Delta = Leak_CoT - Leak_bare; both are scalars or [B]."""
    return leak_cot - leak_bare


def mcs_worst(leak_rates: torch.Tensor) -> torch.Tensor:
    """MCS_worst = min_p (1 - Leak_p)."""
    return (1.0 - leak_rates).min()


def leak_rates_from_logits(policy_logits: torch.Tensor,
                           entity_ids: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        entity_leak_probability(policy_logits[p], entity_ids).mean()
        for p in range(policy_logits.shape[0])
    ])


def disco_loss(
    policy_logits: torch.Tensor,
    bare_logits: torch.Tensor,
    entity_ids: torch.Tensor,
    retain_loss: torch.Tensor,
    config: Optional[DISCOConfig] = None,
) -> MethodOutput:
    """min_theta max_p L_leak(theta;p) + lam*retain + beta*L_param."""
    cfg = config or DISCOConfig()
    worst = worst_policy_margin(policy_logits, entity_ids, cfg.margin)
    bare = context_free_loss(bare_logits, entity_ids, cfg.tau)
    loss = worst + cfg.lam * retain_loss + cfg.beta * bare
    leak_rates = leak_rates_from_logits(policy_logits, entity_ids)
    if leak_rates.numel() > 1:
        separation = decoding_separation(leak_rates.max(), leak_rates.min())
    else:
        separation = torch.zeros((), dtype=leak_rates.dtype, device=leak_rates.device)
    return MethodOutput(loss=loss, diagnostics={
        "disco_worst_margin": worst.detach(),
        "disco_bare_loss": bare.detach(),
        "disco_mcs_worst": mcs_worst(leak_rates).detach(),
        "disco_policy_leak_max": leak_rates.max().detach(),
        "disco_policy_leak_min": leak_rates.min().detach(),
        "disco_separation": separation.detach(),
    })


class DISCO:
    def __init__(self, config: Optional[DISCOConfig] = None):
        self.config = config or DISCOConfig()

    def loss(self, policy_logits, bare_logits, entity_ids, retain_loss):
        return disco_loss(policy_logits, bare_logits, entity_ids, retain_loss,
                          self.config)

    @staticmethod
    def mcs_worst(leak_rates):
        return mcs_worst(leak_rates)
