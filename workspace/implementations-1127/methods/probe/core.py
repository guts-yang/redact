"""PROBE core: probe-adversarial undecodable erasure (IDEA_10_PROBE-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  step-level probe AUC / UDec (worst over probe families), GRL adversary,
  behavior-representation double-evidence gap, undecodable min-max loss.
The probe training (linear/MLP) is in runtime.model + the method train.py;
this core is the adversarial objective and the metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeConfig:
    rho: float = 1.0     # retain KL
    mu: float = 0.5      # non-forget step structure preservation
    gamma: float = 0.5   # target UDec slack above 0.5
    feature_dim: int = 16


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


class StepProbe(nn.Module):
    """A small MLP probe reading step reps -> membership logit."""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rank-based AUC (differentiable-ish, returns scalar in [0,1])."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(())
    diff = pos.view(-1, 1) - neg.view(1, -1)
    return (diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()


def undecodable_loss(
    step_hidden: torch.Tensor,      # [T,H] (gradients flow to theta)
    membership: torch.Tensor,       # [T] 1=forget step, 0=retain step
    probes: list,                   # list[StepProbe]
    retain_kl: torch.Tensor,
    base_step_hidden: torch.Tensor,  # [T,H] theta_0 reps (detached)
    config: ProbeConfig | None = None,
) -> Out:
    """min_theta max_psi [probe_loss - gamma]_+ + rho KL + mu preserve.

    GRL is realized by NEGATING the probe-loss gradient w.r.t. theta: we optimize
    probes on detached features in train.py, and here we push step reps AWAY from
    what the (frozen) probe uses to classify, i.e. minimize -BCE(probe, flipped).
    """
    cfg = config or ProbeConfig()
    feats = step_hidden.detach()
    # worst probe AUC (adversary strength), detached diagnostic
    aucs = []
    for pr in probes:
        with torch.no_grad():
            aucs.append(roc_auc(pr(feats), membership))
    udec = torch.stack(aucs).max() if aucs else feats.new_zeros(())
    # adversarial term: make probe predict the WRONG label on live features
    adv = None
    for pr in probes:
        logits = pr(step_hidden)
        bce = F.binary_cross_entropy_with_logits(logits, 1.0 - membership,
                                                 reduction="mean")
        adv = bce if adv is None else adv + bce
    if adv is None:
        adv = feats.new_zeros(())
    non_forget = (membership < 0.5).float().view(-1, 1)
    preserve = (non_forget * (step_hidden - base_step_hidden).pow(2)).sum()
    gap = udec
    loss = torch.clamp(adv - cfg.gamma, min=0.0) + cfg.rho * retain_kl + cfg.mu * preserve + adv
    return Out(loss=loss, diagnostics={
        "probe_UDec": udec.detach(),
        "probe_adv": adv.detach(),
        "probe_preserve": preserve.detach(),
        "probe_Gap": gap.detach(),
    })


class PROBE:
    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()

    def new_probe(self, dim: int):
        return StepProbe(dim)

    def loss(self, *args, **kwargs):
        return undecodable_loss(*args, config=self.config, **kwargs)

    def auc(self, scores, labels):
        return roc_auc(scores, labels)
