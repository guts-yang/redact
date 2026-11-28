"""Pure teacher-forced margin utilities used by proposal plugins."""
from __future__ import annotations

import torch


def teacher_forced_margin(safe_score: torch.Tensor, disclose_score: torch.Tensor) -> torch.Tensor:
    if safe_score.shape != disclose_score.shape:
        raise ValueError("safe/disclose scores must have identical shapes")
    return safe_score - disclose_score


def margin_violation(margins: torch.Tensor, threshold: float) -> torch.Tensor:
    """Positive part of threshold-margin; zero means the margin is satisfied."""
    return torch.clamp(threshold - margins, min=0.0)


def disclosure_risk(margins: torch.Tensor) -> torch.Tensor:
    if margins.numel() == 0:
        raise ValueError("margins must be non-empty")
    return (margins <= 0).to(margins.dtype).mean()


def worst_margin(margins: torch.Tensor) -> torch.Tensor:
    if margins.numel() == 0:
        raise ValueError("margins must be non-empty")
    return margins.min()


def robust_upper_bound(values: torch.Tensor, std_weight: float = 1.0) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    if values.numel() == 1:
        return values[0]
    return values.max() + std_weight * values.std(unbiased=False)
