"""SEQU: Sequential Unlearning with CoT Drift Correction.

Core reference implementation of proposal IDEA_07. The stateful helper tracks
representation drift on retain CoT after each deletion request and raises the
retain-anchor strength when drift crosses a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import torch

from .base import MethodOutput, l2_per_sample


@dataclass
class SEQUConfig:
    lambda0: float = 0.1
    rho: float = 0.01
    delta: float = 0.1
    repair_lr: float = 1e-3
    max_repairs: int = 100


def representation_drift(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """D_n = E_x ||phi_theta_n(x) - phi_theta_0(x)||_2 on retain anchors."""
    if current.shape != reference.shape:
        raise ValueError("current and reference must match")
    return l2_per_sample(current, reference).mean()


def adaptive_lambda(lambda0: float, drift: torch.Tensor) -> torch.Tensor:
    return lambda0 * (1.0 + drift)


def parameter_proximity(params: Mapping[str, torch.Tensor],
                        anchor: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return sum((params[k] - anchor[k]).pow(2).sum() for k in params)


def sequ_loss(
    forget_loss: torch.Tensor,
    retain_kl: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    anchor_params: Mapping[str, torch.Tensor],
    drift_prev: torch.Tensor,
    config: Optional[SEQUConfig] = None,
) -> MethodOutput:
    """L_forget + lambda_n KL + rho ||theta - theta_{n-1}||^2."""
    cfg = config or SEQUConfig()
    lam = adaptive_lambda(cfg.lambda0, drift_prev)
    prox = parameter_proximity(params, anchor_params)
    loss = forget_loss + lam * retain_kl + cfg.rho * prox
    return MethodOutput(loss=loss, diagnostics={
        "sequ_lambda": lam.detach(),
        "sequ_drift_prev": drift_prev.detach(),
        "sequ_proximity": prox.detach(),
    })


def should_repair(drift: torch.Tensor, delta: float) -> bool:
    return bool((drift > delta).item())


def anchor_loss(current_cot: torch.Tensor, base_cot: torch.Tensor) -> torch.Tensor:
    """Step-level CoT anchor: ||phi_theta(r_t) - phi_theta0(r_t)||^2."""
    if current_cot.shape != base_cot.shape:
        raise ValueError("current_cot and base_cot must match")
    return (current_cot - base_cot).pow(2).flatten(start_dim=1).sum(dim=1).mean()


class SEQU:
    """Stateful sequential wrapper; one call per deletion request."""

    def __init__(self, config: Optional[SEQUConfig] = None):
        self.config = config or SEQUConfig()
        self.drift_history = []
        self.repair_count = 0

    def observe_drift(self, current_repr, base_repr):
        drift = representation_drift(current_repr, base_repr)
        self.drift_history.append(float(drift.detach().cpu()))
        return drift

    def loss(self, forget_loss, retain_kl, params, anchor_params, drift_prev):
        return sequ_loss(forget_loss, retain_kl, params, anchor_params,
                         drift_prev, self.config)

    def repair(self, params, current_cot, base_cot, lr: Optional[float] = None):
        """One CoT anchor gradient step; returns updated parameter dict."""
        lr = self.config.repair_lr if lr is None else lr
        loss = anchor_loss(current_cot, base_cot)
        grads = torch.autograd.grad(loss, [v for v in params.values() if v.requires_grad],
                                    allow_unused=True)
        updated = {}
        grads_iter = iter(grads)
        for name, value in params.items():
            if value.requires_grad:
                grad = next(grads_iter)
                updated[name] = value if grad is None else value - lr * grad
            else:
                updated[name] = value
        self.repair_count += 1
        return updated
