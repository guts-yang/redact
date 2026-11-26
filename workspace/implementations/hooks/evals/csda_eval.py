"""CSDA evaluation hook: step-wise ROUGE-L/cosine and intervention sparsity."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return 0.0
    lcs = _lcs(p, g)
    prec, rec = lcs / len(p), lcs / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    if sim.numel() > 1:
        sim = sim.mean()
    return float(sim.cpu())


def step_wise_metrics(
    pred_steps: Sequence[str],
    gold_steps: Sequence[str],
    pred_reps: Sequence[torch.Tensor],
    gold_reps: Sequence[torch.Tensor],
) -> List[Dict[str, float]]:
    n = min(len(pred_steps), len(gold_steps))
    rows = []
    for i in range(n):
        rows.append({
            "step": i,
            "rouge_l": rouge_l(pred_steps[i], gold_steps[i]),
            "cosine": cosine(pred_reps[i], gold_reps[i]) if i < len(pred_reps) and i < len(gold_reps) else 0.0,
        })
    return rows


def intervention_sparsity(mask: torch.Tensor) -> float:
    return float(mask.float().mean().cpu())


class CSDAEvaluator:
    def evaluate(
        self,
        pred_steps: Sequence[str],
        gold_steps: Sequence[str],
        pred_reps: Sequence[torch.Tensor],
        gold_reps: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, object]:
        rows = step_wise_metrics(pred_steps, gold_steps, pred_reps, gold_reps)
        return {
            "steps": rows,
            "mean_rouge_l": sum(r["rouge_l"] for r in rows) / max(1, len(rows)),
            "mean_cosine": sum(r["cosine"] for r in rows) / max(1, len(rows)),
            "sparsity": intervention_sparsity(mask),
        }
