"""Tests for the shared input-validation in
:func:`redact.leak_classifier_metrics._filter_eval_rows_retain_aux`.

The filter is the single entry point that all four metric primitives
(:func:`partial_auroc_retain_aux`,
:func:`partial_auroc_above_recall_retain_aux`,
:func:`fpr_at_recall_retain_aux`, :func:`gap_idk`) delegate to for
length and value validation. Bugs in shared input-validation tend to
surface as cryptic downstream errors (sklearn's ``roc_curve`` does not
handle NaN gracefully; numpy boolean indexing prefix-truncates on
length mismatch); the tests below pin the loud-failure messages so a
regression here doesn't silently degrade eval logs.

Covers:

- ``len(p_leak) != len(labels)`` raises (was missing before this
  validation was added).
- ``len(sources) != len(labels)`` raises (the historical check).
- ``p_leak`` containing NaN / Inf raises (was missing — silently
  propagated through the metrics).
- Each primitive (including ``gap_idk``) surfaces the new errors via
  the same filter, so a caller hits the same message regardless of
  which metric they invoked.
- A clean fixture still passes all the validation gates and produces
  numeric metrics (sanity-floor: validation didn't accidentally reject
  the happy path).
"""
from __future__ import annotations

import numpy as np
import pytest

from redact.leak_classifier_metrics import (
    _filter_eval_rows_retain_aux,
    fpr_at_recall_retain_aux,
    gap_idk,
    partial_auroc_above_recall_retain_aux,
    partial_auroc_retain_aux,
)


# ---------- direct filter tests ----------


def test_p_leak_length_mismatch_raises():
    """``len(p_leak) != len(labels)``: was previously silently
    prefix-truncating or producing a numpy IndexError; should now
    surface a clear ValueError naming both lengths."""
    p = np.array([0.9, 0.7, 0.3])             # length 3
    y = np.array([1, 1])                       # length 2
    sources = ["forget", "idk"]    # length 2 (matches y)
    with pytest.raises(ValueError, match=r"p_leak length \(3\)"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_sources_length_mismatch_raises():
    """Historical check — sources length must match labels."""
    p = np.array([0.9, 0.3])
    y = np.array([1, 0])
    sources = ["forget"]                 # length 1, not 2
    with pytest.raises(ValueError, match=r"sources length \(1\)"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_p_leak_with_nan_raises():
    """A single NaN in p_leak raises with a clear message — no silent
    propagation through downstream min/max/sklearn calls."""
    p = np.array([0.9, np.nan, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "idk"]
    with pytest.raises(ValueError, match=r"1 non-finite"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_p_leak_with_multiple_nans_reports_count():
    """The error message counts non-finite values to distinguish a
    single rogue NaN (likely a fixture bug) from widespread NaN
    (model divergence)."""
    p = np.array([np.nan, 0.7, np.nan, 0.3, np.nan])
    y = np.array([1, 1, 0, 0, 0])
    sources = ["forget"] * 2 + ["idk"] * 3
    with pytest.raises(ValueError, match=r"3 non-finite"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_p_leak_with_inf_raises():
    """``+inf`` and ``-inf`` also count as non-finite."""
    p = np.array([0.9, np.inf, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "idk"]
    with pytest.raises(ValueError, match=r"non-finite"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )

    p_neg_inf = np.array([0.9, -np.inf, 0.3])
    with pytest.raises(ValueError, match=r"non-finite"):
        _filter_eval_rows_retain_aux(
            p_neg_inf, y, sources, caller="unit_test",
        )


def test_caller_name_in_finite_error_message():
    """The ``caller`` kwarg propagates into the non-finite ValueError
    so a Trainer-eval failure points at the right metric (the same
    convention as the empty-pool messages)."""
    p = np.array([np.nan])
    y = np.array([1])
    sources = ["forget"]
    with pytest.raises(ValueError, match=r"^my_metric:"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="my_metric",
        )


def test_validation_order_p_leak_length_first():
    """When both p_leak length and sources length are wrong, the
    p_leak length error fires first (it appears first in the function
    body, and lengths are validated before content).

    Pins the order so a future refactor reordering the checks would
    surface a different message — useful for log-grepping during
    debugging."""
    # p_leak length=2, labels length=3, sources length=4 — all three
    # disagree. The expected error is the p_leak vs labels mismatch.
    p = np.array([0.9, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "idk", "aux"]
    with pytest.raises(ValueError, match=r"p_leak length"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_validation_order_lengths_before_finite():
    """A NaN in p_leak does NOT mask a length mismatch error — the
    length check runs first. A user who fixes the length mismatch
    will then see the non-finite error on the next attempt."""
    # p_leak length=2, labels length=3, but p_leak contains a NaN.
    p = np.array([np.nan, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "idk"]
    with pytest.raises(ValueError, match=r"p_leak length"):
        _filter_eval_rows_retain_aux(
            p, y, sources, caller="unit_test",
        )


def test_clean_input_passes_all_validation_gates():
    """Sanity-floor: a well-formed input still passes through the
    filter and returns a (positives + IDK-negatives) pair. Catches
    a regression where the new validation accidentally rejects
    everything."""
    p = np.array([0.9, 0.7, 0.2, 0.3])
    y = np.array([1, 1, 0, 0])
    sources = ["forget", "forget", "idk", "idk"]
    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p, y, sources, caller="unit_test",
        source_primaries=("idk",),
    )
    assert len(p_filt) == 4
    assert len(y_filt) == 4


# ---------- primitives surface the new errors ----------


@pytest.mark.parametrize("metric_fn", [
    partial_auroc_retain_aux,
    partial_auroc_above_recall_retain_aux,
    fpr_at_recall_retain_aux,
    gap_idk,
])
def test_each_primitive_raises_on_nan_p_leak(metric_fn):
    """Every metric in the family delegates to the same filter, so
    every primitive should raise the same NaN ValueError. Pins the
    integration: if a future metric is added that bypasses the
    filter, it would skip this test and quietly accept NaN."""
    p = np.array([0.9, np.nan, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "idk"]
    with pytest.raises(ValueError, match=r"non-finite"):
        metric_fn(p, y, sources)


@pytest.mark.parametrize("metric_fn", [
    partial_auroc_retain_aux,
    partial_auroc_above_recall_retain_aux,
    fpr_at_recall_retain_aux,
    gap_idk,
])
def test_each_primitive_raises_on_p_leak_length_mismatch(metric_fn):
    """Same shared-validation contract for the p_leak vs labels
    length check."""
    p = np.array([0.9, 0.3])             # length 2
    y = np.array([1, 1, 0])              # length 3
    sources = ["forget"] * 2 + ["idk"]
    with pytest.raises(ValueError, match=r"p_leak length"):
        metric_fn(p, y, sources)
