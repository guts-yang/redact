"""DRIFT core: step-wise recoverable-drift decomposition (IDEA_04_DRIFT-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  Delta_t = phi_theta(r_t) - phi_theta0(r_t), split into load-bearing (c_t>=tau)
  and free (c_t<tau) parts; a low-rank restoration delta is trained to pull
  ONLY free steps back to theta_0, keeping load steps (and hence forgetting) fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class DriftConfig:
    tau: float = 0.5        # bearing-step threshold on c_t
    beta: float = 1.0        # weight that locks load-bearing steps
    reg: float = 1e-3        # ridge on restoration delta
    kappa: float = 0.1       # soft-mask temperature


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def drift_decomposition(current: torch.Tensor, base: torch.Tensor,
                        contributions: torch.Tensor, tau: float = 0.5):
    """Return Delta [T,H], free_mask [T], load_mask [T] (binary)."""
    if current.shape != base.shape:
        raise ValueError("current/base must match [T,H]")
    delta = current - base
    load = (contributions >= tau).to(delta.dtype).view(-1, 1)
    return delta, 1.0 - load, load


def restoration_loss(
    current: torch.Tensor,   # [T,H] phi_theta (requires grad via delta)
    base: torch.Tensor,      # [T,H] phi_theta0
    contributions: torch.Tensor,  # [T] detached c_t
    delta: torch.Tensor,     # [T,H] trainable restoration (applied to current)
    config: DriftConfig | None = None,
) -> Out:
    cfg = config or DriftConfig()
    _, free, load = drift_decomposition(current, base, contributions, cfg.tau)
    restored = current + delta
    free_term = (free * (restored - base).pow(2)).sum()
    load_term = (load * (restored - current).pow(2)).sum()
    ridge = cfg.reg * delta.pow(2).sum()
    loss = free_term + cfg.beta * load_term + ridge
    full = (current - base).pow(2).sum()
    free_ratio = (free * (current - base).pow(2)).sum() / full.clamp_min(1e-8)
    return Out(loss=loss, diagnostics={
        "drift_free_restore": free_term.detach(),
        "drift_load_lock": load_term.detach(),
        "drift_free_ratio": free_ratio.detach(),
        "drift_delta_norm": delta.detach().norm(),
    })


def recovery_ratio(util_before: float, util_after: float,
                   free_norm: float) -> float:
    """R = Delta_rec / ||Delta_free|| (reported, non-differentiable)."""
    return (util_after - util_before) / max(free_norm, 1e-8)


class DRIFT:
    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()

    def decompose(self, current, base, contributions):
        return drift_decomposition(current, base, contributions, self.config.tau)

    def loss(self, *args, **kwargs):
        return restoration_loss(*args, config=self.config, **kwargs)
