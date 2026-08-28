"""Math tests for :func:`redact.training.compute_sample_weights`.

Covers:
- alpha=1.0 inverse-frequency schedule on imbalanced classes.
- alpha=0.0 collapses to uniform (recovers empirical sampling).
- alpha=0.5 sqrt-balance (intermediate schedule).
- Length preservation: returned tensor matches input length.
- Output dtype is float64 (required by WeightedRandomSampler at fp32
  underflow boundaries on rare classes).
- Within-class weights are uniform — every row of the same class gets
  the same weight, so WeightedRandomSampler treats them as a single
  multinomial bucket.
- Validation: empty list, non-finite alpha raise.
"""
from __future__ import annotations

import math

import pytest
import torch

from redact.training import compute_sample_weights


# ---------- alpha schedules ----------


def test_alpha_one_inverse_frequency_9_to_1():
    """9:1 class split at alpha=1 → rare class weight 9× the common class.

    Class A has 9 rows, class B has 1 row. At alpha=1.0 weight = 1/count,
    so w_A=1/9, w_B=1/1=1.  Ratio of expected mass = 9*(1/9) : 1*(1) = 1:1,
    which is the equal-mass-per-class property we want.
    """
    class_ids = [0] * 9 + [1] * 1
    w = compute_sample_weights(class_ids, alpha=1.0)
    assert len(w) == 10
    # Class A rows
    for i in range(9):
        assert w[i].item() == pytest.approx(1.0 / 9.0)
    # Class B row
    assert w[9].item() == pytest.approx(1.0)
    # Equal-mass property: sum over class A = sum over class B
    assert w[:9].sum().item() == pytest.approx(w[9:].sum().item())


def test_alpha_zero_collapses_to_uniform():
    """alpha=0 → every (1/count)^0 = 1, i.e. uniform weights (empirical sampling)."""
    class_ids = [0, 0, 0, 1, 2, 2]
    w = compute_sample_weights(class_ids, alpha=0.0)
    assert torch.allclose(w, torch.ones(len(class_ids), dtype=torch.float64))


def test_alpha_half_sqrt_balance():
    """alpha=0.5 → weight = (1/count)**0.5, the canonical sqrt schedule."""
    class_ids = [0] * 4 + [1] * 1  # 4:1 split
    w = compute_sample_weights(class_ids, alpha=0.5)
    expected_a = (1.0 / 4.0) ** 0.5
    expected_b = (1.0 / 1.0) ** 0.5  # = 1.0
    for i in range(4):
        assert w[i].item() == pytest.approx(expected_a)
    assert w[4].item() == pytest.approx(expected_b)
    # Sqrt softens the ratio: not equal mass, not empirical
    mass_ratio_a_over_b = w[:4].sum().item() / w[4:].sum().item()
    # Equal mass would be 1.0; empirical (alpha=0) would be 4.0; sqrt sits
    # between → 4 * (1/2) / (1 * 1) = 2.0
    assert mass_ratio_a_over_b == pytest.approx(2.0)


# ---------- shape / dtype invariants ----------


def test_returns_float64_tensor():
    """Float64 is required to avoid underflow on rare-class weights at fp32."""
    w = compute_sample_weights([0, 1, 2], alpha=1.0)
    assert w.dtype == torch.float64
    assert w.ndim == 1


def test_length_matches_input():
    class_ids = list(range(50))  # all distinct (singleton classes)
    w = compute_sample_weights(class_ids, alpha=1.0)
    assert len(w) == 50


def test_within_class_weights_uniform():
    """Two rows of the same class get identical weights."""
    class_ids = [5, 5, 5, 7, 7]
    w = compute_sample_weights(class_ids, alpha=1.0)
    assert w[0].item() == w[1].item() == w[2].item()
    assert w[3].item() == w[4].item()
    # Different classes have different weights here (different counts)
    assert w[0].item() != w[3].item()


# ---------- validation ----------


def test_empty_class_ids_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_sample_weights([], alpha=1.0)


def test_non_finite_alpha_raises():
    with pytest.raises(ValueError, match="finite"):
        compute_sample_weights([0, 1], alpha=math.nan)
    with pytest.raises(ValueError, match="finite"):
        compute_sample_weights([0, 1], alpha=math.inf)


def test_singleton_class_alpha_one():
    """A class with exactly 1 row gets weight 1.0 at alpha=1."""
    w = compute_sample_weights([42], alpha=1.0)
    assert w[0].item() == pytest.approx(1.0)
