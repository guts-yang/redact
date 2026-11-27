"""REVERSE core: bounded reversibility radius (IDEA_05_REVERSE-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  reversibility radius R(eps), worst-direction power-iteration surrogate,
  de-low-rank suppression regularizer (nuclear-norm surrogate), min-max loss.
The steering injection itself (forward hook) is in runtime.model; this core is
the differentiable objective over leak values under a set of direction vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class ReverseConfig:
    lam: float = 1.0       # retain KL
    eta: float = 0.5       # weight on de-low-rank (high-rank suppression) reg
    eps: float = 0.05      # leak threshold defining the radius
    r_max: float = 1.0     # steering vector norm budget


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def worst_direction(grad: torch.Tensor, r_max: float = 1.0) -> torch.Tensor:
    """v* = r_max * g / ||g||, where g = d Leak / d v at v=0 (power iter step 1)."""
    g = grad.flatten()
    n = g.norm().clamp_min(1e-8)
    return (r_max * g / n).view_as(grad)


def reversibility_radius(leak_under_v: torch.Tensor, direction_norms: torch.Tensor,
                         eps: float = 0.05) -> torch.Tensor:
    """R(eps) = max ||v|| among directions whose Leak(theta;v) >= eps (softened)."""
    mask = torch.sigmoid((leak_under_v - eps) * 20.0)
    return (mask * direction_norms).max()


def low_rank_penalty(suppression: torch.Tensor) -> torch.Tensor:
    """Nuclear-norm surrogate ||Sigma_f||_* encourages DISPERSED (high-rank) suppression.

    We return a penalty that is LARGE when suppression is concentrated in a single
    direction: ratio of top singular value to sum of singular values. Minimizing
    its negative spread is equivalent; here we minimize (s1/sum) which pushes
    suppression to be spread (high-rank), shrinking single-direction recovery.
    """
    if suppression.dim() != 2:
        raise ValueError("suppression must be [d,k]")
    s = torch.linalg.svdvals(suppression)
    total = s.sum().clamp_min(1e-8)
    return (s[0] / total).pow(2)


def reverse_loss(
    leak_zero: torch.Tensor,        # scalar: Leak(theta;0)
    leak_under_v: torch.Tensor,    # [K] leak under sampled steering directions
    direction_norms: torch.Tensor, # [K] their norms
    retain_kl: torch.Tensor,
    suppression: torch.Tensor,     # [d,k] matrix of per-step suppression directions
    config: ReverseConfig | None = None,
) -> Out:
    cfg = config or ReverseConfig()
    excess = torch.clamp(leak_under_v.max() - leak_zero, min=0.0)
    radius = reversibility_radius(leak_under_v, direction_norms, cfg.eps)
    lowrank = low_rank_penalty(suppression)
    loss = excess + cfg.lam * retain_kl + cfg.eta * lowrank
    return Out(loss=loss, diagnostics={
        "reverse_worst_excess": excess.detach(),
        "reverse_radius": radius.detach(),
        "reverse_lowrank_ratio": lowrank.detach(),
        "reverse_leak_zero": leak_zero.detach(),
    })


class REVERSE:
    def __init__(self, config: ReverseConfig | None = None):
        self.config = config or ReverseConfig()

    def worst_dir(self, grad):
        return worst_direction(grad, self.config.r_max)

    def radius(self, leak_under_v, norms):
        return reversibility_radius(leak_under_v, norms, self.config.eps)

    def loss(self, *args, **kwargs):
        return reverse_loss(*args, config=self.config, **kwargs)
