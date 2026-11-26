"""EAS: Entanglement-Adaptive Scheduling for concept-level unlearning.

Core reference implementation of proposal IDEA_02. The sample-level gate
``alpha(x)`` chooses between a representation-level (heavy) objective and an
output-level (light) objective, with a budget penalty and an entanglement-
dependent retain anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import MethodOutput, as_scalar


@dataclass
class EASConfig:
    lambda0: float = 0.1
    beta: float = 0.1
    gamma: float = 0.1
    budget: float = 1.0
    c_repr: float = 1.0
    c_out: float = 0.2
    leak_epsilon: float = 0.05
    alpha_temperature: float = 1.0


def entanglement_alpha(
    kappa: torch.Tensor,
    depth: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """alpha(x) = sigmoid(w^T [kappa, depth, 1]).

    ``kappa`` and ``depth`` are ``[B]``. ``weights`` defaults to
    ``[1.0, 0.5, 0.0]``; the last weight is the bias.
    """
    if kappa.shape != depth.shape:
        raise ValueError("kappa and depth must have the same shape")
    if weights is None:
        weights = torch.tensor([1.0, 0.5, 0.0], dtype=kappa.dtype, device=kappa.device)
    if weights.numel() != 3:
        raise ValueError("weights must have three entries")
    logits = weights[0] * kappa + weights[1] * depth + weights[2]
    return torch.sigmoid(logits / max(temperature, 1e-6))


def intervention_cost(alpha: torch.Tensor, config: Optional[EASConfig] = None) -> torch.Tensor:
    cfg = config or EASConfig()
    return alpha * cfg.c_repr + (1.0 - alpha) * cfg.c_out


def eas_loss(
    alpha: torch.Tensor,
    repr_loss: torch.Tensor,
    out_loss: torch.Tensor,
    retain_kl: torch.Tensor,
    kappa: torch.Tensor,
    leak: Optional[torch.Tensor] = None,
    config: Optional[EASConfig] = None,
) -> MethodOutput:
    """Sample-weighted EAS objective with budget and leakage penalties.

    All per-sample losses are ``[B]``; ``retain_kl`` is ``[B]`` or scalar.
    """
    cfg = config or EASConfig()
    if alpha.shape != repr_loss.shape or alpha.shape != out_loss.shape:
        raise ValueError("alpha, repr_loss and out_loss must share [B]")
    if retain_kl.dim() == 0:
        retain_kl = retain_kl.expand_as(alpha)
    lam_kappa = cfg.lambda0 * (1.0 + kappa)
    weighted = alpha * repr_loss + (1.0 - alpha) * out_loss + lam_kappa * retain_kl
    cost = intervention_cost(alpha, cfg)
    budget_penalty = torch.clamp(cost.sum() - cfg.budget, min=0.0)
    loss = weighted.mean() + cfg.beta * budget_penalty
    if leak is not None:
        loss = loss + cfg.gamma * torch.clamp(as_scalar(leak) - cfg.leak_epsilon, min=0.0)
    return MethodOutput(loss=loss, diagnostics={
        "eas_alpha_mean": alpha.detach().mean(),
        "eas_alpha_min": alpha.detach().min(),
        "eas_alpha_max": alpha.detach().max(),
        "eas_cost": cost.detach().sum(),
        "eas_budget_penalty": budget_penalty.detach(),
        "eas_lambda_mean": lam_kappa.detach().mean(),
    })


class EAS:
    def __init__(self, config: Optional[EASConfig] = None,
                 weights: Optional[torch.Tensor] = None):
        self.config = config or EASConfig()
        self.weights = weights

    def alpha(self, kappa, depth):
        return entanglement_alpha(kappa, depth, self.weights,
                                  self.config.alpha_temperature)

    def loss(self, alpha, repr_loss, out_loss, retain_kl, kappa, leak=None):
        return eas_loss(alpha, repr_loss, out_loss, retain_kl, kappa, leak,
                        self.config)
