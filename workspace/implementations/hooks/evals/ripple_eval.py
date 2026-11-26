"""RIPPLE evaluation hook: RippleBench-style MCQ and multi-hop reconstruction."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from ...ripple import reconstruction_rate, ripple_metric


def mcq_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not preds:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(preds)


class RippleBenchEvaluator:
    def __init__(self):
        pass

    def evaluate(
        self,
        neighbor_preds: Sequence[int],
        neighbor_labels: Sequence[int],
        unrelated_preds: Sequence[int],
        unrelated_labels: Sequence[int],
        recovered: torch.Tensor,
        current_utility: torch.Tensor,
        base_utility: torch.Tensor,
        neighbor_ids: Sequence[int],
    ) -> Dict[str, float]:
        return {
            "neighbor_mcq": mcq_accuracy(neighbor_preds, neighbor_labels),
            "unrelated_mcq": mcq_accuracy(unrelated_preds, unrelated_labels),
            "recon": float(reconstruction_rate(recovered).cpu()),
            "ripple": float(ripple_metric(current_utility, base_utility,
                                          neighbor_ids).cpu()),
        }
