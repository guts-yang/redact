"""Full SEQU hook: sequential request stream, drift tracking, repair trigger."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from ..base import MethodOutput
from ..sequ import SEQU, SEQUConfig, representation_drift, should_repair


@dataclass
class SEQUHookConfig:
    n_batches: int = 5
    lambda0: float = 0.1
    rho: float = 0.01
    delta: float = 0.1
    repair_lr: float = 1e-3
    seed: int = 1001


class SequentialRequestStream:
    def __init__(self, rows: Sequence[dict], n_batches: int = 5, seed: int = 1001):
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        n_batches = max(1, min(n_batches, len(rows)))
        self.batches = [
            rows[i::n_batches] for i in range(n_batches)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class SEQUHook:
    def __init__(self, config: Optional[SEQUHookConfig] = None):
        self.config = config or SEQUHookConfig()
        self.sequ = SEQU(SEQUConfig(lambda0=self.config.lambda0,
                                    rho=self.config.rho,
                                    delta=self.config.delta,
                                    repair_lr=self.config.repair_lr))
        self.drift_history: List[float] = []
        self.repair_count = 0
        self.prev_drift = torch.zeros(())

    def stream(self, rows: Sequence[dict]) -> SequentialRequestStream:
        return SequentialRequestStream(rows, self.config.n_batches, self.config.seed)

    def observe_drift(self, current_repr: torch.Tensor,
                      base_repr: torch.Tensor) -> torch.Tensor:
        drift = representation_drift(current_repr, base_repr)
        self.drift_history.append(float(drift.detach().cpu()))
        return drift

    def loss(
        self,
        forget_loss: torch.Tensor,
        retain_kl: torch.Tensor,
        params: Mapping[str, torch.Tensor],
        anchor_params: Mapping[str, torch.Tensor],
    ) -> MethodOutput:
        return self.sequ.loss(forget_loss, retain_kl, params, anchor_params,
                              self.prev_drift)

    def maybe_repair(
        self,
        params: Mapping[str, torch.Tensor],
        current_cot: torch.Tensor,
        base_cot: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        drift = self.observe_drift(current_cot, base_cot)
        self.prev_drift = drift.detach()
        if should_repair(drift, self.config.delta):
            self.repair_count += 1
            return self.sequ.repair(params, current_cot, base_cot)
        return params

    def summary(self) -> Dict[str, object]:
        return {
            "n_requests": len(self.drift_history),
            "drift_history": list(self.drift_history),
            "repair_count": self.repair_count,
        }
