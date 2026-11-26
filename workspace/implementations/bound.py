"""BOUND: Commitment-Boundary Early Exit with Censoring-Aware Evaluation.

Core reference implementation of proposal IDEA_10. The boundary detector finds
the first step after which the answer distribution stops changing; the exit
objective closes the CoT safely at that boundary. The evaluation reports a
censored interval for truncated generations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .base import MethodOutput, js_divergence


@dataclass
class BOUNDConfig:
    delta_js: float = 0.01
    mu: float = 0.1
    zeta: float = 0.1
    max_width: float = 0.2
    max_lower: float = 0.05


def commitment_boundary(
    answer_logits: torch.Tensor,
    delta_js: float = 0.01,
) -> int:
    """Return the earliest t such that every later step has JS <= delta.

    ``answer_logits`` is ``[T, V]`` (one sequence) or ``[B, T, V]`` (take the
    first sequence). Returns ``T-1`` when no stable suffix is found.
    """
    if answer_logits.dim() == 3:
        answer_logits = answer_logits[0]
    if answer_logits.dim() != 2:
        raise ValueError("answer_logits must be [T, V] or [B, T, V]")
    t_steps = answer_logits.shape[0]
    if t_steps <= 1:
        return 0
    js = js_divergence(answer_logits[:-1], answer_logits[1:])
    for t in range(js.numel()):
        if bool((js[t:] <= delta_js).all()):
            return t
    return t_steps - 1


def entropy_boundary(
    step_logits: torch.Tensor,
    delta_entropy: float = 0.05,
) -> int:
    """Cheap monotone detector: first step where entropy change stabilizes."""
    if step_logits.dim() == 3:
        step_logits = step_logits[0]
    logp = F.log_softmax(step_logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    if ent.numel() <= 1:
        return 0
    changes = (ent[1:] - ent[:-1]).abs()
    for t in range(changes.numel()):
        if bool((changes[t:] <= delta_entropy).all()):
            return t
    return step_logits.shape[0] - 1


def safe_cot_exit(safe_cot: str, idk: str = "I don't know.") -> str:
    return f"{safe_cot.rstrip()}\n\n{idk}".strip()


def exit_loss(
    exit_logits: torch.Tensor,
    target_ids: torch.Tensor,
    forget_loss: torch.Tensor,
    mu: float = 0.1,
) -> torch.Tensor:
    """CE(y_exit | x) + mu * L_forget."""
    ce = F.cross_entropy(exit_logits, target_ids)
    return ce + mu * forget_loss


def bound_loss(
    budget_losses: Sequence[torch.Tensor],
    exit_losses: Sequence[torch.Tensor],
    zeta: float = 0.1,
) -> MethodOutput:
    """max_b [L_leak(theta;b) + zeta * L_exit(theta;b)]."""
    if len(budget_losses) != len(exit_losses):
        raise ValueError("budget_losses and exit_losses must align")
    if not budget_losses:
        raise ValueError("at least one budget is required")
    per_budget = [leak + zeta * ex for leak, ex in zip(budget_losses, exit_losses)]
    stacked = torch.stack([p.reshape(()) for p in per_budget])
    worst = stacked.max()
    return MethodOutput(loss=worst, diagnostics={
        "bound_worst_budget_loss": worst.detach(),
        "bound_mean_budget_loss": stacked.mean().detach(),
        "bound_n_budgets": len(budget_losses),
    })


def censored_interval(
    n: int,
    leaks: int,
    truncated: int,
) -> Tuple[float, float, float]:
    """Return (lo, hi, width) for the censored leakage interval."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= leaks <= n and 0 <= truncated <= n):
        raise ValueError("leaks/truncated must be in [0, n]")
    lo = leaks / n
    hi = min(1.0, (leaks + truncated) / n)
    return lo, hi, hi - lo


def bound_success(
    lo: float,
    width: float,
    max_width: float = 0.2,
    max_lower: float = 0.05,
) -> bool:
    return width <= max_width and lo <= max_lower


class BOUND:
    def __init__(self, config: Optional[BOUNDConfig] = None):
        self.config = config or BOUNDConfig()

    def boundary(self, answer_logits):
        return commitment_boundary(answer_logits, self.config.delta_js)

    def loss(self, budget_losses, exit_losses):
        return bound_loss(budget_losses, exit_losses, self.config.zeta)

    def evaluate(self, n, leaks, truncated):
        lo, hi, width = censored_interval(n, leaks, truncated)
        return {
            "lo": lo,
            "hi": hi,
            "width": width,
            "success": bound_success(lo, width, self.config.max_width,
                                     self.config.max_lower),
        }
