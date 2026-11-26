"""Full BOUND hook: multi-budget boundaries, exit loss, censored evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from ..base import MethodOutput
from ..bound import (BOUND, BOUNDConfig, bound_loss, censored_interval,
                     commitment_boundary, exit_loss)


@dataclass
class BOUNDHookConfig:
    budgets: Tuple[int, ...] = (96, 192, 512, 1024)
    delta_js: float = 0.01
    mu: float = 0.1
    zeta: float = 0.1
    max_width: float = 0.2
    max_lower: float = 0.05


class BOUNDHook:
    def __init__(self, config: Optional[BOUNDHookConfig] = None):
        self.config = config or BOUNDHookConfig()
        self.bound = BOUND(BOUNDConfig(delta_js=self.config.delta_js,
                                       mu=self.config.mu,
                                       zeta=self.config.zeta,
                                       max_width=self.config.max_width,
                                       max_lower=self.config.max_lower))

    def boundaries(self, logits_by_budget: Mapping[int, torch.Tensor]) -> Dict[int, int]:
        return {
            int(budget): commitment_boundary(logits, self.config.delta_js)
            for budget, logits in logits_by_budget.items()
        }

    def exit_losses(
        self,
        logits_by_budget: Mapping[int, torch.Tensor],
        labels: torch.Tensor,
        forget_loss: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        out = {}
        for budget, logits in logits_by_budget.items():
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            tau = min(max(commitment_boundary(logits, self.config.delta_js), 0),
                      logits.shape[1] - 1)
            targets = labels[:, tau]
            targets = torch.where(targets < 0, labels[:, -1], targets)
            out[int(budget)] = exit_loss(logits[:, tau, :], targets,
                                         forget_loss, self.config.mu)
        return out

    def loss(
        self,
        budget_losses: Sequence[torch.Tensor],
        exit_losses: Sequence[torch.Tensor],
    ) -> MethodOutput:
        return bound_loss(budget_losses, exit_losses, self.config.zeta)

    def evaluate(self, n: int, leaks: int, truncated: int) -> Dict[str, float]:
        lo, hi, width = censored_interval(n, leaks, truncated)
        return {
            "lo": lo,
            "hi": hi,
            "width": width,
            "success": bool(width <= self.config.max_width and lo <= self.config.max_lower),
        }
