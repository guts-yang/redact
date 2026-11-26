"""Shared, method-agnostic primitives for the ten LRM-unlearning ideas.

The modules in this package are deliberately decoupled: each method imports only
PyTorch, the standard library, and this small base module. They expose core
algorithmic losses/utilities that can be plugged into an R-TOFU training loop.
Nothing here creates a claim or touches an ARIS gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass
class MethodOutput:
    """A method's scalar objective plus optional detached diagnostics."""

    loss: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in self.diagnostics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    out[key] = float(value.detach().cpu())
                else:
                    out[key] = value.detach().float().cpu().tolist()
            elif isinstance(value, (int, float)):
                out[key] = float(value)
        return out


def as_scalar(value: torch.Tensor) -> torch.Tensor:
    """Return a scalar tensor, accepting [B] or [1] reductions."""
    if value.numel() == 1:
        return value.reshape(())
    return value.mean()


def safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return num / den.clamp_min(eps)


def l2_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean L2 distance over the last dimension for matching [B, ...] tensors."""
    return (x - y).flatten(start_dim=1).norm(dim=1)


def kl_divergence(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(p_theta || p_ref), reduced by mean."""
    logp = F.log_softmax(logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1).mean()


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two batches of logits, per sample."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-12))
    return 0.5 * (p * (torch.log(p.clamp_min(1e-12)) - logm)).sum(-1) + \
        0.5 * (q * (torch.log(q.clamp_min(1e-12)) - logm)).sum(-1)


def entity_margin_loss(
    logits: torch.Tensor,
    entity_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """max(0, logit(entity) - margin), averaged over the batch.

    ``logits`` is [B, V] and ``entity_ids`` is [B].
    """
    if logits.dim() != 2:
        raise ValueError("logits must be [B, V]")
    ent = logits.gather(1, entity_ids.view(-1, 1)).squeeze(1)
    return torch.clamp(ent - margin, min=0.0).mean()


def entity_leak_probability(logits: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
    """p(entity | context) for each row, returned as [B]."""
    if logits.dim() != 2:
        raise ValueError("logits must be [B, V]")
    return F.softmax(logits, dim=-1).gather(1, entity_ids.view(-1, 1)).squeeze(1)


def detach_mean(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().mean().cpu())
