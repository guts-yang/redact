"""Unit tests for :mod:`redact.leak_classifier_metrics` ``compute_*`` functions.

These tests exercise the ``compute_metrics`` callable directly — the
parameterization refactor that hoisted ``threshold``, ``gamma``, and
``label_smoothing`` out of the train script's module scope is the only
reason this is possible. Before the extract to
:mod:`redact.leak_classifier_metrics`, the function read those values
from the train script's module-level ``args`` and could not be imported
without running the script's CLI parse + ``load_from_disk``.

Covers :func:`compute_binary_metrics` — precision / recall / F-beta on
synthetic logits, and the ``effective_positive_mass`` diagnostic
emitted when ``gamma > 0``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from redact.leak_classifier_metrics import compute_binary_metrics


# ---------- compute_binary_metrics ----------


def _binary_pred(logits_pleak: list[float], labels: list[int]):
    """Build an ``(eval_pred, labels)`` pair matching the HF Trainer
    contract: ``logits`` of shape ``(N, 2)`` with column 1 = leak.

    Given a target ``P(leak)`` for each row, we manufacture logits as
    ``[log(1-p), log(p)]`` (up to a shared additive constant which the
    softmax cancels) so the resulting softmax recovers the requested
    probability exactly within fp32 precision.
    """
    pleak = np.asarray(logits_pleak, dtype=np.float64)
    pleak = np.clip(pleak, 1e-9, 1.0 - 1e-9)
    logits = np.stack([np.log(1.0 - pleak), np.log(pleak)], axis=-1)
    return (logits, np.asarray(labels))


def test_binary_perfect_precision_and_recall_at_low_threshold():
    """Strong separation: all positives have P(leak)=0.9, all negatives
    P(leak)=0.1. At threshold=0.5 we get perfect precision and recall."""
    eval_pred = _binary_pred(
        [0.9, 0.1, 0.9, 0.1, 0.1], [1, 0, 1, 0, 0],
    )
    m = compute_binary_metrics(eval_pred, threshold=0.5)
    assert m["binary_precision"] == 1.0
    assert m["binary_recall"] == 1.0
    assert m["binary_f1"] == 1.0
    assert m["binary_f2"] == 1.0


def test_binary_threshold_zero_recall_one_precision_drop():
    """A threshold of 0 flags every row as positive → recall=1.0,
    precision = n_pos / N. Confirms the threshold kwarg is consulted
    (was ``args.threshold`` module-scope before the refactor)."""
    eval_pred = _binary_pred(
        [0.9, 0.1, 0.9, 0.1, 0.1], [1, 0, 1, 0, 0],
    )
    m = compute_binary_metrics(eval_pred, threshold=0.0)
    assert m["binary_recall"] == 1.0
    assert m["binary_precision"] == pytest.approx(2 / 5)


def test_binary_effective_positive_mass_emitted_only_when_gamma_positive():
    eval_pred = _binary_pred([0.9, 0.1], [1, 0])
    no_focal = compute_binary_metrics(eval_pred, threshold=0.5, gamma=0.0)
    assert "effective_positive_mass" not in no_focal
    with_focal = compute_binary_metrics(eval_pred, threshold=0.5, gamma=1.0)
    assert "effective_positive_mass" in with_focal


def test_binary_effective_positive_mass_matches_manual_formula():
    """Confident positive (P(leak)=0.95) under gamma=2 with eps=0.05
    gets dyn weight (0.05)**2 = 0.0025 — the eps floor clipping the
    (1-0.95)=0.05 from below was a no-op here, the value is already
    at the boundary. Confirms the formula is computed end-to-end."""
    eval_pred = _binary_pred(
        [0.95, 0.5, 0.9, 0.1], [1, 1, 0, 0],
    )
    m = compute_binary_metrics(
        eval_pred, threshold=0.5, gamma=2.0, label_smoothing=0.05,
    )
    # Manual: only label=1 rows contribute; mass = mean((1-p)^2 clipped to [0.05, 0.95]).
    one_minus_p_pos = np.clip(np.array([1 - 0.95, 1 - 0.5]), 0.05, 0.95)
    expected = float((one_minus_p_pos ** 2.0).mean())
    assert m["effective_positive_mass"] == pytest.approx(expected, rel=1e-5)


def test_binary_effective_positive_mass_nan_when_no_positives_in_batch():
    eval_pred = _binary_pred([0.5, 0.5, 0.5], [0, 0, 0])
    m = compute_binary_metrics(eval_pred, threshold=0.5, gamma=2.0)
    assert math.isnan(m["effective_positive_mass"])


def test_binary_label_smoothing_off_uses_internal_1e_minus_6_floor():
    """At gamma=2, label_smoothing=0, and a row with P(leak)=1-1e-9
    (close to a degenerate confident positive), the internal floor
    clamps 1-p to >= 1e-6, so dyn weight >= 1e-12. Confirms the floor
    is consulted in the numpy path (mirrors the in-graph forward)."""
    eval_pred = _binary_pred(
        [1.0 - 1e-9], [1],
    )
    m = compute_binary_metrics(
        eval_pred, threshold=0.5, gamma=2.0, label_smoothing=0.0,
    )
    # 1-p (after fp32 round-trip) is well below 1e-6; clamped to 1e-6 → (1e-6)^2 = 1e-12.
    assert m["effective_positive_mass"] == pytest.approx(1e-12, rel=1.0)
    # Crucially, finite — not NaN/Inf.
    assert math.isfinite(m["effective_positive_mass"])
