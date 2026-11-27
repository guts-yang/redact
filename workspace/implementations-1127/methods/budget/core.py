"""BUDGET core: budget-monotone continuous min-max unlearning (IDEA_01_BUDGET-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  rho(b) continuous leak field, monotonicity regularizer, budget-integral
  worst-budget min-max objective, and the BC certificate.
All inputs are tensors so the module is testable on CPU with synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class BudgetConfig:
    lam: float = 1.0            # weight on retain KL
    mu: float = 0.5            # weight on monotonicity
    nu: float = 0.1            # weight on step-structure preservation
    eps: float = 0.0           # slack in monotonicity hinge
    worst_weight: float = 1.0  # weight on worst-budget (inner max) term


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def monotonicity_loss(rho: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """L_mono = sum_k [rho(b_{k+1}) - rho(b_k) + eps]_+^2.

    Proposal §3.2 requires rho NON-INCREASING in budget b (leak must not grow
    with more reasoning budget). A violation is rho(k+1) > rho(k); that is the
    hinge we penalize. rho is in ascending budget order b_0=0 ... b_{K-1}=B.
    """
    if rho.dim() != 1 or rho.numel() < 2:
        return rho.new_zeros(())
    diff = rho[1:] - rho[:-1] + eps
    return torch.clamp(diff, min=0.0).pow(2).sum()


def budget_certificate(rho: torch.Tensor) -> torch.Tensor:
    """BC = max_b rho(b) - rho(0); ->0 means worst budget leaks no more than zero budget."""
    return rho.max() - rho[0]


def budget_loss(
    rho: torch.Tensor,               # [K] differentiable soft leak prob per budget grid
    retain_kl: torch.Tensor,        # scalar
    step_preserve: torch.Tensor,    # scalar: ||phi-phi0||^2 on non-forget steps
    config: BudgetConfig | None = None,
) -> Out:
    """min_theta int_0^B rho db + lam KL + mu L_mono + nu preserve + worst_budget max.

    `rho` is expected in ASCENDING budget order b_0=0 ... b_{K-1}=B so that
    monotonic non-increase is well defined. The worst-budget term implements the
    inner max_b by a softmax-weighted (temperature) surrogate of max(rho).
    """
    cfg = config or BudgetConfig()
    if rho.dim() != 1:
        raise ValueError("rho must be [K]")
    mono = monotonicity_loss(rho, cfg.eps)
    integral = rho.mean()
    # smooth max (LogSumExp) as differentiable inner-max surrogate
    worst = torch.logsumexp(rho * 10.0, dim=0) / 10.0
    loss = (integral + cfg.lam * retain_kl + cfg.mu * mono
            + cfg.nu * step_preserve + cfg.worst_weight * worst)
    return Out(loss=loss, diagnostics={
        "budget_integral": integral.detach(),
        "budget_mono": mono.detach(),
        "budget_worst": worst.detach(),
        "budget_BC": budget_certificate(rho).detach(),
        "budget_max_rho": rho.max().detach(),
        "budget_rho0": rho[0].detach(),
    })


class BUDGET:
    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()

    def loss(self, rho, retain_kl, step_preserve):
        return budget_loss(rho, retain_kl, step_preserve, self.config)

    def certificate(self, rho):
        return budget_certificate(rho)
