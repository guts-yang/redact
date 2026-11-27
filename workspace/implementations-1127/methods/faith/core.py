"""FAITH core: faithfulness-gated answer-level erasure (IDEA_06_FAITH-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  Faith(r_t) from control-patch answer-distribution shift,
  gated loss (faithful steps -> CoT representation suppression, unfaithful
  steps -> answer-level counterfactual dependency suppression), FaithGap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class FaithConfig:
    lam: float = 1.0      # retain KL
    tau: float = 0.5      # bearing-step threshold (for m_t)
    eta: float = 0.3      # unfaithful threshold for FaithGap
    kappa: float = 0.1


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def faithfulness(answer_logits_base: torch.Tensor,
                 answer_logits_patched: torch.Tensor) -> torch.Tensor:
    """Faith(r_t) = normalized distribution shift when step t is patched.

    Both inputs [T,V] (or [V]); returns [T] in [0,1] via JS -> 1-exp(-JS*scale).
    Detached by caller (faith is a fixed gate on theta_0, not trainable).
    """
    if answer_logits_base.dim() == 1:
        answer_logits_base = answer_logits_base.unsqueeze(0)
        answer_logits_patched = answer_logits_patched.unsqueeze(0)
    p = F.softmax(answer_logits_base, dim=-1)
    q = F.softmax(answer_logits_patched, dim=-1)
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-12))
    js = 0.5 * (p * (torch.log(p.clamp_min(1e-12)) - logm)).sum(-1) + \
         0.5 * (q * (torch.log(q.clamp_min(1e-12)) - logm)).sum(-1)
    return 1.0 - torch.exp(-js * 4.0)


def bearing_mask(contributions: torch.Tensor, tau: float = 0.5,
                 kappa: float = 0.1) -> torch.Tensor:
    return torch.sigmoid((contributions - tau) / kappa)


def faith_gated_loss(
    step_hidden: torch.Tensor,        # [T,H] current CoT step reps
    v_safe: torch.Tensor,            # [H] safe direction
    contributions: torch.Tensor,     # [T] detached c_t
    faith: torch.Tensor,             # [T] detached Faith(r_t) in [0,1]
    answer_entity_logp: torch.Tensor,  # scalar or [T]: log p(e_f | do(a<-a_cf))
    retain_kl: torch.Tensor,
    config: FaithConfig | None = None,
) -> Out:
    cfg = config or FaithConfig()
    m = bearing_mask(contributions, cfg.tau, cfg.kappa)
    cot_term = (faith * m * (step_hidden - v_safe).pow(2).sum(-1)).sum()
    # L_ans = -log(1 - p(e_f | counterfactual answer)); differentiable soft form
    p = answer_entity_logp.exp().clamp(0.0, 1.0 - 1e-6)
    ans_term = (-(1.0 - faith) * torch.log((1.0 - p).clamp_min(1e-8))).sum()
    loss = cot_term + ans_term + cfg.lam * retain_kl
    return Out(loss=loss, diagnostics={
        "faith_cot_term": cot_term.detach(),
        "faith_ans_term": ans_term.detach(),
        "faith_gap": (faith < cfg.eta).float().mean().detach(),
        "faith_mean": faith.mean().detach(),
    })


class FAITH:
    def __init__(self, config: FaithConfig | None = None):
        self.config = config or FaithConfig()

    def faith(self, base, patched):
        return faithfulness(base, patched)

    def loss(self, *args, **kwargs):
        return faith_gated_loss(*args, config=self.config, **kwargs)
