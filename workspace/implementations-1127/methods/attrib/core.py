"""ATTRIB core: step keyword attribution + recovery-destruction dilemma (IDEA_08_ATTRIB-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  KA_t keyword attention share, RR/DR rates, RD dilemma closure,
  step-level keyword-attention REDIRECTION (not global suppression).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class AttribConfig:
    lam: float = 1.0     # retain KL
    eta: float = 1.0     # weight on destructive-rate upper bound
    mu: float = 1.0      # weight on step KA redirection
    delta: float = 0.02  # allowed destructive rate
    kappa: float = 0.1   # soft mask temperature for bearing steps


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def keyword_attention_share(attention: torch.Tensor, keyword_positions: torch.Tensor,
                            steps: torch.Tensor) -> torch.Tensor:
    """KA_t = sum of attention mass on keyword tokens for each CoT step.

    attention [T,S] (step x source-token attention), keyword_positions [K] long,
    steps is unused placeholder kept for API symmetry. Returns [T].
    """
    if attention.dim() != 2:
        raise ValueError("attention must be [T,S]")
    k = keyword_positions.clamp(0, attention.shape[1] - 1)
    return attention.index_select(1, k).sum(-1)


def step_ka_redirect(ka_theta: torch.Tensor, ka_safe: torch.Tensor,
                     contributions: torch.Tensor, kappa: float = 0.1) -> torch.Tensor:
    """Weighted redirection: push high-attribution steps toward safe KA baseline."""
    w = torch.sigmoid((contributions - contributions.mean()) / kappa)
    return (w * (ka_theta - ka_safe).pow(2)).sum()


def attrib_loss(
    rho_emph: torch.Tensor,      # scalar: leak under keyword-emphasis rewrite
    rho_orig: torch.Tensor,      # scalar: leak under original question
    util_theta0: torch.Tensor,   # scalar: retain utility of theta_0
    util_theta: torch.Tensor,    # scalar: retain utility of theta
    ka_theta: torch.Tensor,      # [T] current step keyword attention share
    ka_safe: torch.Tensor,       # [T] retain CoT baseline
    contributions: torch.Tensor, # [T] detached bearing estimate
    retain_kl: torch.Tensor,
    config: AttribConfig | None = None,
) -> Out:
    cfg = config or AttribConfig()
    rr = torch.clamp(rho_emph - rho_orig, min=0.0)
    dr = torch.clamp(util_theta0 - util_theta, min=0.0)
    dr_excess = torch.clamp(dr - cfg.delta, min=0.0)
    redirect = step_ka_redirect(ka_theta, ka_safe, contributions, cfg.kappa)
    loss = rr + cfg.eta * dr_excess + cfg.lam * retain_kl + cfg.mu * redirect
    return Out(loss=loss, diagnostics={
        "attrib_RR": rr.detach(), "attrib_DR": dr.detach(),
        "attrib_RD": (rr + dr).detach(), "attrib_redirect": redirect.detach(),
    })


class ATTRIB:
    def __init__(self, config: AttribConfig | None = None):
        self.config = config or AttribConfig()

    def ka(self, attention, keyword_positions):
        return keyword_attention_share(attention, keyword_positions, None)

    def loss(self, *args, **kwargs):
        return attrib_loss(*args, config=self.config, **kwargs)
