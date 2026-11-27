"""CONSENSUS core: majority-vote recovery + consensus-invariant objective (IDEA_03_CONSENSUS-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3.

Training-side convention (important): ``entity_scores`` are ENTITY PROBABILITIES
in (0,1) of shape [B, N_max], one column per sampled reasoning path. The soft
consensus uses a soft-MAXIMUM over the N sampled paths (log-sum-exp), which is
non-decreasing in N -- this is exactly the "test-time scaling amplifies residual
preference" effect the proposal requires. The greedy leak is a single path.
The resulting hinge ``[CLR(N) - rho_greedy]_+`` is therefore non-degenerate and
carries gradients into the model logits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class ConsensusConfig:
    lam: float = 1.0       # retain KL
    mu: float = 0.5        # step-structure preservation
    gamma: float = 0.1     # soft-consensus decision threshold
    temperature: float = 0.2
    beta: float = 8.0      # sharpness of the soft-max over samples
    n_grid: tuple = (1, 4, 8, 16)


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def soft_consensus(entity_probs: torch.Tensor, n: int,
                   gamma: float = 0.1, temperature: float = 0.2,
                   beta: float = 8.0) -> torch.Tensor:
    """Differentiable CLR(N): sigmoid(soft_max_over_N_samples(probs) - gamma).

    soft-max is ``logsumexp(beta*p)/beta``; it is non-decreasing in the number
    of sampled paths N, matching the majority-vote recovery amplification.
    """
    sub = entity_probs[:, :max(1, n)]
    soft_max = torch.logsumexp(beta * sub, dim=1) / beta - torch.log(
        torch.tensor(float(sub.shape[1]), dtype=sub.dtype, device=sub.device)) / beta
    # subtract log(N)/beta so that N=1 returns ~p (unbiased soft-max)
    return torch.sigmoid((soft_max - gamma) / temperature).mean()


def majority_vote_label(answer_ids: torch.Tensor) -> torch.Tensor:
    """True (non-differentiable) majority vote over [B,N] integer answer ids."""
    if answer_ids.dim() != 2:
        raise ValueError("answer_ids must be [B,N]")
    b = answer_ids.shape[0]
    out = torch.empty(b, dtype=answer_ids.dtype, device=answer_ids.device)
    for i in range(b):
        vals, counts = torch.unique(answer_ids[i], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def consensus_margin(entity_probs: torch.Tensor, cfg: ConsensusConfig) -> torch.Tensor:
    """CM = max_N [softCLR(N) - softCLR(1)] (should ->0 after training)."""
    base = soft_consensus(entity_probs, 1, cfg.gamma, cfg.temperature, cfg.beta)
    vals = [soft_consensus(entity_probs, n, cfg.gamma, cfg.temperature, cfg.beta) - base
            for n in cfg.n_grid]
    return torch.stack(vals).max()


def consensus_loss(
    entity_probs: torch.Tensor,  # [B, N_max] entity probability per sampled path
    greedy_leak: torch.Tensor,  # scalar / [B] single-path leak probability
    retain_kl: torch.Tensor,
    step_preserve: torch.Tensor,
    config: ConsensusConfig | None = None,
) -> Out:
    """min_theta max_{N,tau} [CLR(N)-rho_greedy]_+ + lam KL + mu preserve."""
    cfg = config or ConsensusConfig()
    worst = None
    for n in cfg.n_grid:
        v = soft_consensus(entity_probs, n, cfg.gamma, cfg.temperature, cfg.beta)
        worst = v if worst is None else torch.maximum(worst, v)
    # tau (decision threshold) is the second attack dimension: a sharper/looser
    # threshold yields a worse (larger) CLR, so take the max over it too.
    sharp = soft_consensus(entity_probs, max(cfg.n_grid), cfg.gamma * 0.5,
                            cfg.temperature, cfg.beta)
    worst = torch.maximum(worst, sharp)
    greedy = greedy_leak.mean() if greedy_leak.dim() else greedy_leak
    excess = torch.clamp(worst - greedy, min=0.0)
    cm = consensus_margin(entity_probs, cfg)
    # include a small direct worst term so the objective always pushes the
    # residual consensus preference down, even when the hinge momentarily closes
    loss = excess + 0.5 * worst + cfg.lam * retain_kl + cfg.mu * step_preserve
    return Out(loss=loss, diagnostics={
        "consensus_worst_clr": worst.detach(),
        "consensus_excess": excess.detach(),
        "consensus_CM": cm.detach(),
        "consensus_greedy": (greedy.detach() if isinstance(greedy, torch.Tensor)
                             else float(greedy)),
    })


class CONSENSUS:
    def __init__(self, config: ConsensusConfig | None = None):
        self.config = config or ConsensusConfig()

    def loss(self, *args, **kwargs):
        return consensus_loss(*args, config=self.config, **kwargs)

    def margin(self, entity_probs):
        return consensus_margin(entity_probs, self.config)

    def vote(self, answer_ids):
        return majority_vote_label(answer_ids)
