"""RRBU evaluation hook: relearning curve RR(K) and recovery rate."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

import torch


def recovery_rate(leaks: Iterable[float], threshold: float = 0.5) -> float:
    vals = list(leaks)
    if not vals:
        return 0.0
    return sum(1 for x in vals if x >= threshold) / len(vals)


class RRCurveEvaluator:
    """Generic RR(K) runner.

    ``attack_fn(k)`` performs K relearning steps and returns any state.
    ``leak_fn(state)`` returns held-out forget leakage in [0, 1].
    """

    def __init__(self, ks: Sequence[int] = (0, 1, 3, 5, 10, 75)):
        self.ks = list(ks)

    def curve(
        self,
        attack_fn: Callable[[int], object],
        leak_fn: Callable[[object], float],
        threshold: float = 0.5,
    ) -> Dict[str, object]:
        points: List[Dict[str, float]] = []
        for k in self.ks:
            state = attack_fn(k)
            leak = float(leak_fn(state))
            points.append({"K": k, "leak": leak,
                           "recovered": float(leak >= threshold)})
        return {
            "points": points,
            "rr5": next((p["leak"] for p in points if p["K"] == 5), None),
            "rr75": next((p["leak"] for p in points if p["K"] == 75), None),
            "recovery_rate_5": next((p["recovered"] for p in points if p["K"] == 5), None),
        }
