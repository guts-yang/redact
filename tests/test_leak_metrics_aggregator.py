"""Tests for :class:`redact.leak_classifier_metrics.LeakMetricsAggregator`.

The class is the HF Trainer-compatible ``compute_metrics`` callable
for the leak-classifier training pipeline; it layers cross-distribution
metrics (pAUC, IDK trio, focal-mass, rolling smoothing) on top of a
base per-branch metric function.

This file exercises the aggregator without spinning up an HF Trainer
or RoBERTa model. We construct synthetic ``eval_pred`` tuples and a
minimal stub ``SplitAwareTrainer``, then verify:

- All pooled headlines are present on every test-split eval.
- All 9 per-source diagnostic keys (3 metrics × 3 buckets) are present.
- When source-balance is on, the 3 ``*_balanced`` keys equal the
  macro-average of the per-source components.
- Empty source bucket → NaN for the affected per-source keys + a
  NaN-skipping macro-average. Does not crash the eval pass.
- Smoothing window emits the expected ``*_smoothed`` companions and
  obeys the rolling/NaN-skipping semantics.
- IDK focal mass diagnostic and the IDK-pool trio (gap_idk,
  min_p_pos, max_p_idk) follow their documented contracts.
- **Class-specific**: two instances have isolated state; the
  aggregator is a Callable; a fresh instance after "resume" starts
  with empty smoothing history.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from redact.leak_classifier_metrics import (
    PER_SOURCE_BUCKETS,
    LeakMetricsAggregator,
)


# ---------- helpers ----------

class _Args:
    """Minimal stand-in for the argparse namespace consumed by the
    aggregator. Only the fields the aggregator reads are required."""
    def __init__(
        self,
        source_balance: bool,
        *,
        idk_focal_gamma: float = 0.0,
        idk_focal_label_smoothing: float = 0.0,
    ):
        self.pauc_fpr_cap = 0.10
        self.pauc_recall_lower = 0.95
        self.fpr_at_recall_target = 0.95
        self.pauc_source_balance = source_balance
        self.idk_focal_gamma = idk_focal_gamma
        self.idk_focal_label_smoothing = idk_focal_label_smoothing


def _build_wrapper(test_sources, args, *, metric_prefix: str = "eval_test"):
    """Construct a :class:`LeakMetricsAggregator` with a stubbed
    trainer, returning the instance (which is callable as
    ``aggregator(eval_pred)``).

    The base ``compute_metrics`` is a no-op (returns an empty dict) so
    the tests inspect only the keys the aggregator adds. The
    p_leak/label extractors are identity adapters because the test
    fixtures pass ``p`` and ``y`` directly in their final
    ``(p_leak, binary_labels)`` form rather than going through a model.

    ``metric_prefix`` controls the stubbed trainer's
    ``_current_metric_prefix``; the aggregator short-circuits when it
    is not exactly ``"eval_test"``.
    """
    aggregator = LeakMetricsAggregator(
        base_compute_metrics=lambda _eval_pred: {},
        p_leak_extractor=lambda logits: np.asarray(logits, dtype=np.float64),
        label_extractor=lambda labels: np.asarray(labels, dtype=np.int64),
        test_sources=test_sources,
        pauc_fpr_cap=args.pauc_fpr_cap,
        pauc_recall_lower=args.pauc_recall_lower,
        fpr_at_recall_target=args.fpr_at_recall_target,
        pauc_source_balance=args.pauc_source_balance,
        idk_focal_gamma=args.idk_focal_gamma,
        idk_focal_label_smoothing=args.idk_focal_label_smoothing,
    )

    class _StubTrainer:
        _current_metric_prefix = metric_prefix

    aggregator.trainer = _StubTrainer()
    return aggregator


# ---------- core key contracts ----------

def test_pooled_headlines_always_present():
    """All pooled headlines exist regardless of the flag."""
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
        + ["idk"] * 10
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30 + [0.1] * 10)
    y = np.array([1] * 20 + [0] * 90)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    assert "pauc_retain_aux" in out
    assert "pauc_above_recall_retain_aux" in out
    assert "fpr_at_recall_retain_aux" in out
    # IDK-pool trio: gap and its two components.
    assert "gap_idk" in out
    assert "min_p_pos" in out
    assert "max_p_idk" in out


def test_gap_idk_pooled_value_matches_formula():
    """``gap_idk`` from the wrapper equals min(p_leak | label=1) -
    max(p_leak | clean_idk, label=0). Pins the wrapper's wiring against
    a hand-set fixture so refactors of the wrapper can't silently drop
    the new key or compute a different formula."""
    # Positives at 0.9, 0.7; IDK negatives at 0.3, 0.4; gap = 0.7 - 0.4 = 0.3.
    sources = (
        ["forget"] * 2
        + ["retain"] * 2  # high-scoring but ignored
        + ["idk"] * 2
    )
    p = np.array([0.9, 0.7, 0.95, 0.93, 0.3, 0.4])
    y = np.array([1, 1, 0, 0, 0, 0])
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    assert out["gap_idk"] == pytest.approx(0.3)


def test_gap_idk_nan_when_idk_pool_empty():
    """When the eval split has no clean_idk negatives, the wrapper
    swallows the IDK-empty ValueError from gap_idk and emits NaN —
    same graceful-degrade pattern as the per-source bucket NaN
    handling. The two component keys (min_p_pos, max_p_idk) follow
    the same NaN convention since they share the row filter."""
    # No clean_idk rows; the other pooled metrics still compute fine
    # because they cover retain ∪ aux ∪ idk and retain/aux are
    # populated.
    sources = (
        ["forget"] * 5
        + ["retain"] * 10
        + ["aux"] * 5
    )
    p = np.array([0.9] * 5 + [0.1] * 15)
    y = np.array([1] * 5 + [0] * 15)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    for key in ("gap_idk", "min_p_pos", "max_p_idk"):
        assert key in out
        assert math.isnan(out[key])


def test_min_p_pos_pooled_value_matches_formula():
    """``min_p_pos`` equals min(P(leak) | label=1) on the eval split.
    A high-confidence label=0 retain row should NOT pull the value
    down — the metric is positive-only."""
    sources = (
        ["forget"] * 2
        + ["retain"] * 2
        + ["idk"] * 2
    )
    # Positives at 0.6, 0.8; non-positives at 0.99, 0.95, 0.3, 0.2.
    p = np.array([0.6, 0.8, 0.99, 0.95, 0.3, 0.2])
    y = np.array([1, 1, 0, 0, 0, 0])
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    # min over [0.6, 0.8] = 0.6 (the 0.99/0.95 negatives are excluded).
    assert out["min_p_pos"] == pytest.approx(0.6)


def test_max_p_idk_pooled_value_matches_formula():
    """``max_p_idk`` equals max(P(leak) | label=0 AND clean_idk).
    A high-confidence retain or aux negative must NOT raise it."""
    sources = (
        ["forget"] * 2
        + ["retain"] * 1
        + ["aux"] * 1
        + ["idk"] * 2
    )
    # Positives at 0.85, 0.7; retain at 0.99; aux at 0.93; IDK at 0.3, 0.45.
    p = np.array([0.85, 0.7, 0.99, 0.93, 0.3, 0.45])
    y = np.array([1, 1, 0, 0, 0, 0])
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    # max over IDK = max(0.3, 0.45) = 0.45 (the 0.99/0.93 are NOT IDK)
    assert out["max_p_idk"] == pytest.approx(0.45)


def test_wrapper_gap_equals_min_minus_max_components():
    """Algebraic identity at the wrapper level: the gap_idk key equals
    the difference of its two component keys, bit-for-bit. Because
    the wrapper computes all three from a single shared row filter,
    this should hold exactly (no fp drift from two independent
    filter calls)."""
    sources = (
        ["forget"] * 3
        + ["idk"] * 3
    )
    p = np.array([0.9, 0.8, 0.65, 0.2, 0.35, 0.5])
    y = np.array([1, 1, 1, 0, 0, 0])
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    # gap = min_p_pos - max_p_idk = 0.65 - 0.5 = 0.15
    assert out["min_p_pos"] == pytest.approx(0.65)
    assert out["max_p_idk"] == pytest.approx(0.5)
    assert out["gap_idk"] == pytest.approx(0.15)
    # Bit-for-bit identity (single filter call → no fp drift).
    assert out["gap_idk"] == out["min_p_pos"] - out["max_p_idk"]


# ---------- robust (quantile) variant of gap_idk ----------


def test_gap_idk_robust_value_matches_quantile_formula():
    """``gap_idk_robust`` from the wrapper equals
    quantile(P_pos, 0.05) - quantile(P_idk, 0.95) under numpy's default
    ``method='linear'`` quantile. Pins the wrapper's wiring against a
    hand-computable fixture with >=10 samples per bucket so the
    linear-interpolation behavior is visible (rather than collapsing
    onto the bucket extremes as it would for very small samples).

    Positives: 10 values evenly spaced in [0.50, 0.95]:
        [0.50, 0.55, 0.60, ..., 0.95].
    IDK negatives: 10 values evenly spaced in [0.05, 0.50]:
        [0.05, 0.10, 0.15, ..., 0.50].
    np.quantile with method='linear' on a sorted array of length 10
    returns the value at index ``q * (n - 1)`` with linear
    interpolation between bracketing entries. For q=0.05, n=10:
    index = 0.45 → between sorted[0] and sorted[1].
    """
    sources = (
        ["forget"] * 10
        + ["idk"] * 10
    )
    pos_vals = np.linspace(0.50, 0.95, 10)  # ascending
    idk_vals = np.linspace(0.05, 0.50, 10)  # ascending
    p = np.concatenate([pos_vals, idk_vals])
    y = np.array([1] * 10 + [0] * 10)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))

    expected_min_q05 = float(np.quantile(pos_vals, 0.05, method="linear"))
    expected_max_q95 = float(np.quantile(idk_vals, 0.95, method="linear"))
    expected_gap = expected_min_q05 - expected_max_q95

    assert out["min_p_pos_q05"] == pytest.approx(expected_min_q05, abs=1e-6)
    assert out["max_p_idk_q95"] == pytest.approx(expected_max_q95, abs=1e-6)
    assert out["gap_idk_robust"] == pytest.approx(expected_gap, abs=1e-6)
    # Bit-for-bit identity from the shared filter call.
    assert out["gap_idk_robust"] == (
        out["min_p_pos_q05"] - out["max_p_idk_q95"]
    )


def test_gap_idk_robust_nan_when_idk_pool_empty():
    """When the eval split has no clean_idk negatives, the wrapper
    swallows the ValueError from the shared filter and emits NaN for
    all three keys of the robust trio — same graceful-degrade pattern
    as the strict gap_idk trio."""
    sources = (
        ["forget"] * 5
        + ["retain"] * 10
        + ["aux"] * 5
    )
    p = np.array([0.9] * 5 + [0.1] * 15)
    y = np.array([1] * 5 + [0] * 15)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    for key in ("gap_idk_robust", "min_p_pos_q05", "max_p_idk_q95"):
        assert key in out
        assert math.isnan(out[key])


def test_gap_idk_robust_resists_outlier_versus_strict():
    """The whole point of the robust variant: a single anomalous IDK
    P(leak) value drives strict ``gap_idk`` toward zero (catastrophic)
    while ``gap_idk_robust`` shrugs it off.

    Construct a clean base scenario where strict and robust both yield
    a clear positive gap (~0.4). Then inject one anomalous IDK row at
    P(leak)=0.95 and assert:
    - strict ``gap_idk`` collapses (becomes catastrophically small or
      negative — < 0.1) because ``max_p_idk`` jumps to 0.95.
    - ``gap_idk_robust`` stays near 0.4 because the 95th percentile is
      still drawn from the unscathed bulk of the IDK distribution.

    Sizes: >=20 rows per bucket so the 95th percentile clearly
    excludes any single outlier (q=0.95 on n=20 sits at index 18.05,
    interpolating between the 19th and 20th sorted values; pushing
    one row beyond the 19th moves the quantile only marginally)."""
    # Base scenario: 20 positives in [0.6, 0.95], 20 IDKs in [0.05, 0.20].
    # strict gap = 0.6 - 0.20 = 0.4
    # robust gap (q05 pos - q95 idk):
    #   q05(0.6..0.95 ascending, n=20) interpolates between idx 0 (0.6)
    #     and idx 1 ~ 0.618 → ~0.6018
    #   q95(0.05..0.20 ascending, n=20) interpolates between idx 18
    #     (~0.192) and idx 19 (0.20) → ~0.199
    # → robust gap ~ 0.40
    base_pos = np.linspace(0.60, 0.95, 20)
    base_idk = np.linspace(0.05, 0.20, 20)
    sources_base = ["forget"] * 20 + ["idk"] * 20
    p_base = np.concatenate([base_pos, base_idk])
    y_base = np.array([1] * 20 + [0] * 20)

    wrapped_base = _build_wrapper(sources_base, _Args(source_balance=False))
    out_base = wrapped_base((p_base, y_base))

    # Sanity: both strict and robust start near 0.4 on the clean batch.
    assert out_base["gap_idk"] == pytest.approx(0.4, abs=1e-6)
    assert out_base["gap_idk_robust"] == pytest.approx(0.4, abs=0.05)

    # Inject one anomalous IDK row at P(leak)=0.95. (Fresh aggregator
    # so the smoothing history doesn't carry into this comparison.)
    sources_outlier = sources_base + ["idk"]
    p_outlier = np.concatenate([p_base, np.array([0.95])])
    y_outlier = np.concatenate([y_base, np.array([0])])
    wrapped_outlier = _build_wrapper(
        sources_outlier, _Args(source_balance=False),
    )
    out_outlier = wrapped_outlier((p_outlier, y_outlier))

    # Strict gap: max_p_idk jumps to 0.95, min_p_pos still 0.60 →
    # gap = 0.60 - 0.95 = -0.35. Drops far below 0.1 (catastrophic).
    assert out_outlier["gap_idk"] < 0.1, (
        f"Strict gap_idk should collapse on outlier injection but got "
        f"{out_outlier['gap_idk']}"
    )

    # Robust gap: q95 on n=21 IDK with one outlier at 0.95 interpolates
    # between idx 19 and idx 20 of the sorted IDK array. The sorted
    # array is [0.05..0.20 (20 vals), 0.95], so q05*20 = 19 → exactly
    # at idx 19 (value 0.20). q05(pos) unchanged ~0.60. Robust gap
    # stays ~0.40 — clearly higher than the now-broken strict gap.
    assert out_outlier["gap_idk_robust"] == pytest.approx(0.4, abs=0.05), (
        f"Robust gap_idk should stay near 0.4 despite outlier but got "
        f"{out_outlier['gap_idk_robust']}"
    )
    # Direction sanity: robust > strict after the outlier injection,
    # by a wide margin.
    assert out_outlier["gap_idk_robust"] > out_outlier["gap_idk"] + 0.3


def test_gap_idk_robust_smoothed_companion_present():
    """``gap_idk_robust_smoothed`` is emitted alongside the raw robust
    key when the smoothing window is active, and equals the mean over
    the last ``window`` eval-test passes (since ``_add_smoothed_companions``
    iterates generically over all numeric keys, the robust trio gets
    smoothing for free)."""
    sources = ["forget"] * 10 + ["idk"] * 10
    y = np.array([1] * 10 + [0] * 10)
    args = _Args(source_balance=False)
    wrapped = _build_wrapper_with_window(sources, args, window=2)

    pos1 = np.linspace(0.60, 0.95, 10)
    idk1 = np.linspace(0.05, 0.20, 10)
    out1 = wrapped((np.concatenate([pos1, idk1]), y))
    # First eval — smoothed == unsmoothed (single-entry window).
    assert "gap_idk_robust_smoothed" in out1
    assert out1["gap_idk_robust_smoothed"] == pytest.approx(
        out1["gap_idk_robust"],
    )
    # Companion keys for the two components also auto-smooth.
    assert "min_p_pos_q05_smoothed" in out1
    assert "max_p_idk_q95_smoothed" in out1

    pos2 = np.linspace(0.45, 0.75, 10)
    idk2 = np.linspace(0.05, 0.25, 10)
    out2 = wrapped((np.concatenate([pos2, idk2]), y))
    # Second eval — smoothed = mean of the two raw values.
    assert out2["gap_idk_robust_smoothed"] == pytest.approx(
        (out1["gap_idk_robust"] + out2["gap_idk_robust"]) / 2,
    )
    assert out2["min_p_pos_q05_smoothed"] == pytest.approx(
        (out1["min_p_pos_q05"] + out2["min_p_pos_q05"]) / 2,
    )
    assert out2["max_p_idk_q95_smoothed"] == pytest.approx(
        (out1["max_p_idk_q95"] + out2["max_p_idk_q95"]) / 2,
    )


def test_gap_idk_robust_smoothed_skips_nan_history_via_nanmean():
    """A NaN entry in the robust gap history is skipped by
    ``np.nanmean`` in the smoothed companion — mirrors the existing
    ``test_smoothing_window_skips_nan_history_via_nanmean`` contract
    for strict ``gap_idk_smoothed``."""
    sources = (
        ["forget"] * 3 + ["retain"] * 3 + ["idk"] * 10
    )
    args = _Args(source_balance=False)
    wrapped = _build_wrapper_with_window(sources, args, window=3)

    # eval 1: IDK-prefixed rows labeled 1 → no IDK negatives → NaN.
    p1 = np.concatenate([
        np.array([0.9, 0.8, 0.7]),  # clean_forget (label=1, positives)
        np.array([0.05, 0.06, 0.04]),  # clean_retain (label=0, ignored by IDK filter)
        np.linspace(0.91, 0.95, 10),  # clean_idk rows mislabeled as 1
    ])
    y1 = np.array([1, 1, 1, 0, 0, 0] + [1] * 10)
    out1 = wrapped((p1, y1))
    assert math.isnan(out1["gap_idk_robust"])
    assert math.isnan(out1["gap_idk_robust_smoothed"])

    # eval 2: IDK rows now labeled 0 → robust gap finite. History
    # after: [NaN, finite]. nanmean ignores the NaN.
    p2 = np.concatenate([
        np.array([0.9, 0.8, 0.7]),
        np.array([0.05, 0.06, 0.04]),
        np.linspace(0.05, 0.20, 10),
    ])
    y2 = np.array([1, 1, 1, 0, 0, 0] + [0] * 10)
    out2 = wrapped((p2, y2))
    finite_gap = out2["gap_idk_robust"]
    assert math.isfinite(finite_gap)
    # Critical: nanmean over [NaN, finite_gap] = finite_gap.
    assert out2["gap_idk_robust_smoothed"] == pytest.approx(finite_gap)


def test_all_nine_per_source_keys_present():
    """3 metrics × 3 buckets = 9 diagnostic keys, always emitted."""
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
        + ["idk"] * 10
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30 + [0.1] * 10)
    y = np.array([1] * 20 + [0] * 90)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    expected_keys = {
        f"{metric}_{bucket}"
        for metric in (
            "pauc_retain_aux",
            "pauc_above_recall_retain_aux",
            "fpr_at_recall_retain_aux",
        )
        for bucket in ("retain", "aux", "idk")
    }
    assert expected_keys.issubset(out.keys())


def test_balanced_keys_only_when_flag_set():
    """The ``*_balanced`` headlines appear iff --pauc_source_balance."""
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
        + ["idk"] * 10
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30 + [0.1] * 10)
    y = np.array([1] * 20 + [0] * 90)
    balanced_keys = {
        "pauc_retain_aux_balanced",
        "pauc_above_recall_retain_aux_balanced",
        "fpr_at_recall_retain_aux_balanced",
    }
    wrapped_off = _build_wrapper(sources, _Args(source_balance=False))
    wrapped_on = _build_wrapper(sources, _Args(source_balance=True))
    out_off = wrapped_off((p, y))
    out_on = wrapped_on((p, y))
    assert balanced_keys.isdisjoint(out_off.keys())
    assert balanced_keys.issubset(out_on.keys())


def test_balanced_equals_macro_of_per_source():
    """``pauc_retain_aux_balanced`` is exactly the mean of the three
    per-source components."""
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
        + ["idk"] * 10
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30 + [0.1] * 10)
    y = np.array([1] * 20 + [0] * 90)
    wrapped = _build_wrapper(sources, _Args(source_balance=True))
    out = wrapped((p, y))
    for metric in (
        "pauc_retain_aux",
        "pauc_above_recall_retain_aux",
        "fpr_at_recall_retain_aux",
    ):
        components = [out[f"{metric}_{b}"] for b in ("retain", "aux", "idk")]
        assert out[f"{metric}_balanced"] == pytest.approx(
            float(np.nanmean(components)), abs=1e-12,
        )


# ---------- non-test prefix: short-circuit ----------

def test_non_test_split_short_circuits():
    """On eval_forget / eval_retain the wrapper writes no pAUC keys
    (those derived splits contain only one side of the positive/
    negative distribution, so pAUC is undefined there)."""
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
        + ["idk"] * 10
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30 + [0.1] * 10)
    y = np.array([1] * 20 + [0] * 90)
    wrapped = _build_wrapper(
        sources, _Args(source_balance=True), metric_prefix="eval_forget",
    )
    out = wrapped((p, y))
    # No pAUC / per-source / balanced keys written on a non-test split.
    pauc_keys = {k for k in out if "pauc" in k or "fpr_at_recall" in k}
    assert pauc_keys == set()


# ---------- empty-bucket graceful degradation ----------

def test_empty_bucket_yields_nan_not_crash():
    """If a source bucket is absent from the eval split, the
    per-source keys for that bucket are NaN (not missing, not
    exception) and the balanced macro skips NaN via nanmean."""
    # No clean_idk rows; retain + aux only.
    sources = (
        ["forget"] * 20
        + ["retain"] * 50
        + ["aux"] * 30
    )
    p = np.array([0.9] * 20 + [0.1] * 50 + [0.1] * 30)
    y = np.array([1] * 20 + [0] * 80)
    wrapped = _build_wrapper(sources, _Args(source_balance=True))
    out = wrapped((p, y))

    # IDK keys exist and are NaN — the externally-observable
    # graceful-degrade behavior. The wrapper's internal
    # ``warned_empty_buckets`` tracker (used for one-shot log
    # deduplication) is implementation detail and no longer inspected.
    for metric in (
        "pauc_retain_aux",
        "pauc_above_recall_retain_aux",
        "fpr_at_recall_retain_aux",
    ):
        assert f"{metric}_idk" in out
        assert math.isnan(out[f"{metric}_idk"])

    # retain / aux components are real numbers.
    assert not math.isnan(out["pauc_retain_aux_retain"])
    assert not math.isnan(out["pauc_retain_aux_aux"])

    # Balanced headline equals nanmean of (retain, aux, NaN) =
    # mean of (retain, aux).
    for metric in (
        "pauc_retain_aux",
        "pauc_above_recall_retain_aux",
        "fpr_at_recall_retain_aux",
    ):
        expected = (out[f"{metric}_retain"] + out[f"{metric}_aux"]) / 2
        assert out[f"{metric}_balanced"] == pytest.approx(expected, abs=1e-12)


def test_per_source_buckets_tied_to_eligible_primaries():
    """The module-level invariant: every primary in
    ``_ELIGIBLE_NEGATIVE_PRIMARIES`` must appear in exactly one
    ``PER_SOURCE_BUCKETS`` entry, and vice versa. The check runs at
    import time; this test just confirms the contents still agree.

    Post-refactor, ``PER_SOURCE_BUCKETS`` entries are ``(name, primary)``
    pairs (a single primary source value) rather than ``(name, prefixes)``.
    """
    from redact.leak_classifier_metrics import _ELIGIBLE_NEGATIVE_PRIMARIES
    bucket_primaries = frozenset(primary for _, primary in PER_SOURCE_BUCKETS)
    assert bucket_primaries == _ELIGIBLE_NEGATIVE_PRIMARIES


# ---------- effective_idk_mass ----------


def test_effective_idk_mass_absent_when_gamma_zero():
    """``idk_focal_gamma=0`` is the default; the diagnostic key should
    not appear in the wrapper output."""
    sources = (
        ["forget"] * 5
        + ["retain"] * 10
        + ["idk"] * 5
    )
    p = np.array([0.9] * 5 + [0.1] * 10 + [0.2] * 5)
    y = np.array([1] * 5 + [0] * 15)
    wrapped = _build_wrapper(sources, _Args(source_balance=False))
    out = wrapped((p, y))
    assert "effective_idk_mass" not in out


def test_effective_idk_mass_present_when_gamma_positive():
    """``idk_focal_gamma>0`` causes the diagnostic key to be emitted."""
    sources = (
        ["forget"] * 5
        + ["retain"] * 10
        + ["idk"] * 5
    )
    p = np.array([0.9] * 5 + [0.1] * 10 + [0.2] * 5)
    y = np.array([1] * 5 + [0] * 15)
    wrapped = _build_wrapper(
        sources, _Args(source_balance=False, idk_focal_gamma=1.0),
    )
    out = wrapped((p, y))
    assert "effective_idk_mass" in out


def test_effective_idk_mass_matches_manual_formula():
    """``effective_idk_mass`` equals mean(clamp(P(leak), eps, 1-eps)^gamma)
    over the IDK-negative rows in the eval split. Pins the wrapper's
    wiring against a hand-set fixture."""
    # 2 positives, 2 retain negatives, 3 IDK negatives.
    # IDK P(leak) values: [0.3, 0.5, 0.9].
    sources = (
        ["forget"] * 2
        + ["retain"] * 2
        + ["idk"] * 3
    )
    p = np.array([0.9, 0.8, 0.05, 0.06, 0.3, 0.5, 0.9])
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    wrapped = _build_wrapper(
        sources, _Args(
            source_balance=False,
            idk_focal_gamma=2.0,
            idk_focal_label_smoothing=0.05,
        ),
    )
    out = wrapped((p, y))
    p_idk = np.clip(np.array([0.3, 0.5, 0.9]), 0.05, 0.95)
    expected = float((p_idk ** 2.0).mean())
    assert out["effective_idk_mass"] == pytest.approx(expected, rel=1e-6)


def test_effective_idk_mass_nan_when_idk_pool_empty():
    """No IDK-negative rows → effective_idk_mass=NaN. Same graceful-
    degrade pattern as gap_idk and the per-source buckets."""
    sources = (
        ["forget"] * 5
        + ["retain"] * 10
        + ["aux"] * 5
    )
    p = np.array([0.9] * 5 + [0.1] * 15)
    y = np.array([1] * 5 + [0] * 15)
    wrapped = _build_wrapper(
        sources, _Args(source_balance=False, idk_focal_gamma=1.0),
    )
    out = wrapped((p, y))
    assert "effective_idk_mass" in out
    assert math.isnan(out["effective_idk_mass"])


def test_effective_idk_mass_excludes_idk_positives():
    """If an IDK row has label=1 (shouldn't happen by convention, but
    defensive), the wrapper excludes it from effective_idk_mass.
    The training script's _attach_is_idk_negative raises in this
    case anyway, but the diagnostic should never count IDK positives."""
    # Single IDK row with label=1 (anomaly) and one IDK row with label=0.
    sources = (
        ["forget"] * 2
        + ["idk"] * 2
    )
    p = np.array([0.9, 0.8, 0.7, 0.3])
    y = np.array([1, 1, 1, 0])  # IDK row 0 (idx 2) is label=1
    wrapped = _build_wrapper(
        sources, _Args(source_balance=False, idk_focal_gamma=1.0),
    )
    out = wrapped((p, y))
    # Only the label=0 IDK row (P(leak)=0.3) contributes:
    expected = float(np.clip(0.3, 1e-6, 1.0) ** 1.0)
    assert out["effective_idk_mass"] == pytest.approx(expected, rel=1e-6)


# ---------- smoothing_window ----------


def test_smoothing_window_one_emits_no_smoothed_keys():
    """Default ``smoothing_window=1`` is the no-op path — no ``_smoothed``
    companion keys appear in the output."""
    sources = ["forget"] * 5 + ["idk"] * 5
    p = np.array([0.9] * 5 + [0.2] * 5)
    y = np.array([1] * 5 + [0] * 5)
    args = _Args(source_balance=False)
    # _Args doesn't expose smoothing_window; the default is window=1
    # via LeakMetricsAggregator's kwarg default. So this is the
    # default path — assert no _smoothed keys appear.
    wrapped = _build_wrapper(sources, args)
    out = wrapped((p, y))
    assert not any(k.endswith("_smoothed") for k in out), (
        f"Default smoothing_window=1 should emit no _smoothed keys; "
        f"saw: {[k for k in out if k.endswith('_smoothed')]}"
    )


def _build_wrapper_with_window(test_sources, args, *, window: int):
    """Same builder as _build_wrapper but with explicit
    smoothing_window. Kept inline rather than threading through _Args
    so the additional tests are explicit about their window setting."""
    aggregator = LeakMetricsAggregator(
        base_compute_metrics=lambda _eval_pred: {},
        p_leak_extractor=lambda logits: np.asarray(logits, dtype=np.float64),
        label_extractor=lambda labels: np.asarray(labels, dtype=np.int64),
        test_sources=test_sources,
        pauc_fpr_cap=args.pauc_fpr_cap,
        pauc_recall_lower=args.pauc_recall_lower,
        fpr_at_recall_target=args.fpr_at_recall_target,
        pauc_source_balance=args.pauc_source_balance,
        idk_focal_gamma=args.idk_focal_gamma,
        idk_focal_label_smoothing=args.idk_focal_label_smoothing,
        smoothing_window=window,
    )

    class _StubTrainer:
        _current_metric_prefix = "eval_test"

    aggregator.trainer = _StubTrainer()
    return aggregator


def test_smoothing_window_two_emits_parallel_smoothed_keys():
    """``smoothing_window=2`` emits a ``<key>_smoothed`` for each
    numeric key. First eval: window has one entry, so
    ``<key>_smoothed == <key>``. Second eval with different values:
    ``<key>_smoothed`` is the mean of the two evals."""
    sources = ["forget"] * 3 + ["idk"] * 3
    p1 = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.4])
    p2 = np.array([0.5, 0.4, 0.3, 0.1, 0.15, 0.2])
    y = np.array([1, 1, 1, 0, 0, 0])
    args = _Args(source_balance=False)
    wrapped = _build_wrapper_with_window(sources, args, window=2)

    out1 = wrapped((p1, y))
    # First eval: smoothed == unsmoothed because window has 1 entry.
    assert out1["gap_idk_smoothed"] == pytest.approx(out1["gap_idk"])
    assert out1["min_p_pos_smoothed"] == pytest.approx(out1["min_p_pos"])
    assert out1["max_p_idk_smoothed"] == pytest.approx(out1["max_p_idk"])

    out2 = wrapped((p2, y))
    # Second eval: smoothed = (eval1_value + eval2_value) / 2.
    assert out2["gap_idk_smoothed"] == pytest.approx(
        (out1["gap_idk"] + out2["gap_idk"]) / 2,
    )
    assert out2["min_p_pos_smoothed"] == pytest.approx(
        (out1["min_p_pos"] + out2["min_p_pos"]) / 2,
    )
    assert out2["max_p_idk_smoothed"] == pytest.approx(
        (out1["max_p_idk"] + out2["max_p_idk"]) / 2,
    )


def test_smoothing_window_three_fills_progressively():
    """With ``window=3``, the smoothing window fills over 3 evals:
    eval 1 → mean of 1 value; eval 2 → mean of 2; eval 3 → mean of 3.
    A 4th eval drops the oldest (rolling)."""
    sources = ["forget"] * 3 + ["idk"] * 3
    # Hand-set p_leak per eval; min_p_pos is min over [a, b, c],
    # max_p_idk is max over [d, e, f]. Choose values so gap_idk
    # equals a known sequence per eval.
    y = np.array([1, 1, 1, 0, 0, 0])
    args = _Args(source_balance=False)
    wrapped = _build_wrapper_with_window(sources, args, window=3)

    # eval 1: min_pos=0.9, max_idk=0.1 → gap=0.8
    out1 = wrapped((np.array([0.95, 0.9, 0.92, 0.05, 0.1, 0.08]), y))
    assert out1["gap_idk_smoothed"] == pytest.approx(0.8)

    # eval 2: min_pos=0.7, max_idk=0.3 → gap=0.4
    out2 = wrapped((np.array([0.8, 0.7, 0.85, 0.2, 0.3, 0.25]), y))
    assert out2["gap_idk_smoothed"] == pytest.approx((0.8 + 0.4) / 2)

    # eval 3: min_pos=0.6, max_idk=0.5 → gap=0.1
    out3 = wrapped((np.array([0.7, 0.65, 0.6, 0.4, 0.5, 0.45]), y))
    assert out3["gap_idk_smoothed"] == pytest.approx((0.8 + 0.4 + 0.1) / 3)

    # eval 4: drops eval 1 from the window.
    # min_pos=0.55, max_idk=0.45 → gap=0.1
    out4 = wrapped((np.array([0.6, 0.58, 0.55, 0.4, 0.45, 0.42]), y))
    assert out4["gap_idk_smoothed"] == pytest.approx((0.4 + 0.1 + 0.1) / 3)


def test_smoothing_window_skips_nan_history_via_nanmean():
    """A NaN entry in the history is skipped by ``np.nanmean``. The
    smoothed value reflects only the finite history entries.

    Uses one wrapper with a stable ``test_sources`` reference. Flips
    the per-row labels between evals: when the IDK-prefixed rows are
    labeled 1, the filter sees zero IDK negatives → gap_idk=NaN;
    when they're labeled 0, the filter sees them as IDK negatives →
    gap_idk is finite. Mixes NaN + finite into the same closure
    history.
    """
    sources = ["forget"] * 3 + ["retain"] * 3 + ["idk"] * 3
    args = _Args(source_balance=False)
    wrapped = _build_wrapper_with_window(sources, args, window=3)

    # eval 1: IDK-prefixed rows ALL labeled 1 (no IDK negatives in
    # the filter) → gap_idk = NaN. History after: [NaN].
    p1 = np.array([0.9, 0.8, 0.7, 0.05, 0.06, 0.04, 0.95, 0.93, 0.91])
    y1 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
    out1 = wrapped((p1, y1))
    assert math.isnan(out1["gap_idk"])
    assert math.isnan(out1["gap_idk_smoothed"])  # all-NaN window

    # eval 2: IDK-prefixed rows now labeled 0 → gap_idk finite.
    # min_pos over clean_forget label=1 rows = 0.7;
    # max_idk over clean_idk label=0 rows = 0.5; gap = 0.2.
    # History after: [NaN, 0.2]. nanmean over that = 0.2.
    p2 = np.array([0.9, 0.8, 0.7, 0.05, 0.06, 0.04, 0.3, 0.5, 0.4])
    y2 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    out2 = wrapped((p2, y2))
    assert out2["gap_idk"] == pytest.approx(0.2)
    # CRITICAL: nanmean over [NaN, 0.2] = 0.2 (not NaN, not (NaN+0.2)/2).
    assert out2["gap_idk_smoothed"] == pytest.approx(0.2)

    # eval 3: another finite gap (0.4). History: [NaN, 0.2, 0.4].
    # Smoothed = (0.2 + 0.4) / 2 = 0.3.
    p3 = np.array([0.95, 0.9, 0.85, 0.05, 0.06, 0.04, 0.3, 0.45, 0.4])
    y3 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    out3 = wrapped((p3, y3))
    # min_pos = 0.85, max_idk = 0.45, gap = 0.4
    assert out3["gap_idk"] == pytest.approx(0.4)
    assert out3["gap_idk_smoothed"] == pytest.approx((0.2 + 0.4) / 2)


def test_smoothing_window_effective_idk_mass_smoothed():
    """``effective_idk_mass_smoothed`` is emitted alongside the raw
    diagnostic when both the focal gate and smoothing are active.
    Exercises the focal-mass key's smoothing path (which the
    primary gap_idk-focused tests don't cover)."""
    sources = ["forget"] * 3 + ["idk"] * 3
    y = np.array([1, 1, 1, 0, 0, 0])
    args = _Args(
        source_balance=False,
        idk_focal_gamma=1.0,
        idk_focal_label_smoothing=0.0,
    )
    wrapped = _build_wrapper_with_window(sources, args, window=2)

    # eval 1: IDK P(leak) ∈ [0.2, 0.4]. effective_idk_mass =
    # mean(clip(0.2, 1e-6, None), clip(0.3, ...), clip(0.4, ...))
    # ^ gamma=1 ⇒ identity, so eim1 = (0.2 + 0.3 + 0.4) / 3.
    p1 = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.4])
    out1 = wrapped((p1, y))
    eim1 = (0.2 + 0.3 + 0.4) / 3
    assert out1["effective_idk_mass"] == pytest.approx(eim1)
    # First eval: smoothed == unsmoothed.
    assert out1["effective_idk_mass_smoothed"] == pytest.approx(eim1)

    # eval 2: IDK P(leak) ∈ [0.1, 0.2]. eim2 = (0.1 + 0.15 + 0.2) / 3.
    p2 = np.array([0.9, 0.8, 0.7, 0.1, 0.15, 0.2])
    out2 = wrapped((p2, y))
    eim2 = (0.1 + 0.15 + 0.2) / 3
    assert out2["effective_idk_mass"] == pytest.approx(eim2)
    # Smoothed = (eim1 + eim2) / 2.
    assert out2["effective_idk_mass_smoothed"] == pytest.approx(
        (eim1 + eim2) / 2,
    )


def test_smoothing_window_rejects_zero_or_negative():
    """``smoothing_window`` must be >= 1 per the documented contract."""
    args = _Args(source_balance=False)
    sources = ["forget"] * 2 + ["idk"] * 2
    with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
        _build_wrapper_with_window(sources, args, window=0)
    with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
        _build_wrapper_with_window(sources, args, window=-1)


def test_smoothing_does_not_pollute_across_non_test_splits():
    """Only eval_test passes contribute to the smoothing history.
    A non-test-split eval (e.g. eval_forget) short-circuits before
    the smoothing logic, so it does NOT advance the window."""
    sources = ["forget"] * 3 + ["idk"] * 3
    args = _Args(source_balance=False)
    aggregator = LeakMetricsAggregator(
        base_compute_metrics=lambda _eval_pred: {},
        p_leak_extractor=lambda logits: np.asarray(logits, dtype=np.float64),
        label_extractor=lambda labels: np.asarray(labels, dtype=np.int64),
        test_sources=sources,
        pauc_fpr_cap=args.pauc_fpr_cap,
        pauc_recall_lower=args.pauc_recall_lower,
        fpr_at_recall_target=args.fpr_at_recall_target,
        pauc_source_balance=args.pauc_source_balance,
        idk_focal_gamma=args.idk_focal_gamma,
        idk_focal_label_smoothing=args.idk_focal_label_smoothing,
        smoothing_window=3,
    )

    class _StubTrainer:
        _current_metric_prefix = "eval_test"

    aggregator.trainer = _StubTrainer()

    y = np.array([1, 1, 1, 0, 0, 0])
    # eval_test call 1: min_pos=0.9, max_idk=0.1 → gap=0.8
    out1 = aggregator((np.array([0.95, 0.9, 0.92, 0.05, 0.1, 0.08]), y))
    assert out1["gap_idk_smoothed"] == pytest.approx(0.8)

    # Switch to a non-test split. The aggregator short-circuits before
    # smoothing, so this eval must NOT advance the window.
    aggregator.trainer._current_metric_prefix = "eval_forget"
    aggregator((np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9]), y))
    # No assertion on the forget-split output (the aggregator returns
    # base unchanged, which is the identity-base empty dict).

    # Back to eval_test, call 2: gap=0.4. If the forget-split eval had
    # advanced the window, smoothed would reflect 3 entries (with a
    # NaN-or-divergent middle one); instead it should be the mean of
    # just the 2 eval_test calls.
    aggregator.trainer._current_metric_prefix = "eval_test"
    out2 = aggregator((np.array([0.8, 0.7, 0.85, 0.2, 0.3, 0.25]), y))
    assert out2["gap_idk_smoothed"] == pytest.approx((0.8 + 0.4) / 2)


# ---------- class-specific regression tests ----------


def test_two_instances_have_isolated_smoothing_state():
    """Closure → class refactor risk: if any mutable state (the
    metric_history deque, the warned_empty_buckets set) accidentally
    lived on the class body instead of in __init__, two instances
    would share it.

    Construct two aggregators with the same sources but feed them
    different eval values. Verify each instance's smoothed value
    reflects only its own history.
    """
    sources = ["forget"] * 3 + ["idk"] * 3
    y = np.array([1, 1, 1, 0, 0, 0])
    args = _Args(source_balance=False)

    agg_a = _build_wrapper_with_window(sources, args, window=3)
    agg_b = _build_wrapper_with_window(sources, args, window=3)

    # Feed agg_a a gap=0.8 eval.
    out_a = agg_a((np.array([0.95, 0.9, 0.92, 0.05, 0.1, 0.08]), y))
    assert out_a["gap_idk_smoothed"] == pytest.approx(0.8)

    # Feed agg_b a gap=0.2 eval. If state were shared, the smoothed
    # value would be (0.8 + 0.2) / 2 = 0.5; with isolated state it's
    # just 0.2 (agg_b's only history entry).
    out_b = agg_b((np.array([0.7, 0.65, 0.6, 0.3, 0.4, 0.35]), y))
    assert out_b["gap_idk_smoothed"] == pytest.approx(0.2), (
        f"agg_b's smoothed gap should be 0.2 (only its own history) "
        f"but got {out_b['gap_idk_smoothed']} — state likely leaked "
        f"from agg_a (got {out_a['gap_idk_smoothed']})."
    )

    # Second eval on agg_a now sees its own first eval + this one,
    # not contaminated by agg_b's eval. eval2 gap = 0.4 → smoothed
    # = (0.8 + 0.4) / 2 = 0.6.
    out_a2 = agg_a((np.array([0.8, 0.7, 0.85, 0.2, 0.3, 0.25]), y))
    assert out_a2["gap_idk_smoothed"] == pytest.approx((0.8 + 0.4) / 2)


def test_aggregator_is_callable():
    """HF Trainer expects ``compute_metrics`` to be a callable. The
    class instance must satisfy the ``Callable`` protocol.

    Verifies both: (a) the bool callable() check passes, and (b) the
    instance produces a dict when invoked with an eval_pred tuple.
    """
    sources = ["forget"] * 2 + ["idk"] * 2
    args = _Args(source_balance=False)
    aggregator = _build_wrapper(sources, args)
    assert callable(aggregator)

    out = aggregator((np.array([0.9, 0.8, 0.2, 0.3]), np.array([1, 1, 0, 0])))
    assert isinstance(out, dict)
    # And the dict contains the expected keys (sanity-floor: not just
    # the base-empty dict — the eval_test branch ran).
    assert "gap_idk" in out


def test_fresh_instance_starts_with_empty_history():
    """Simulates resume_from_checkpoint: HF Trainer rebuilds the
    compute_metrics callable from scratch on resume, so the new
    aggregator's smoothing history starts empty even if a previous
    aggregator had accumulated state.

    Construct an aggregator, feed it a few evals, then construct a
    fresh instance and verify the fresh one's first eval has
    smoothed == unsmoothed (single-point window).
    """
    sources = ["forget"] * 3 + ["idk"] * 3
    y = np.array([1, 1, 1, 0, 0, 0])
    args = _Args(source_balance=False)

    # Pre-resume: accumulate state in the first aggregator.
    pre_resume = _build_wrapper_with_window(sources, args, window=3)
    pre_resume((np.array([0.95, 0.9, 0.92, 0.05, 0.1, 0.08]), y))
    pre_resume((np.array([0.8, 0.7, 0.85, 0.2, 0.3, 0.25]), y))

    # Post-resume: a fresh instance. Its first eval should see only
    # itself in the window, so smoothed == unsmoothed AND the
    # history deque has exactly one entry (this eval's snapshot).
    post_resume = _build_wrapper_with_window(sources, args, window=3)
    out = post_resume((np.array([0.7, 0.65, 0.6, 0.3, 0.4, 0.35]), y))
    # The explicit deque-length check pins "empty pre-state" vs a
    # hypothetical contamination mode that wiped state down to a
    # single entry — smoothed==unsmoothed alone is consistent with
    # either, so we assert the underlying history shape too.
    assert len(post_resume._metric_history) == 1, (
        f"Fresh instance should have exactly one history entry (the "
        f"eval just run) — got {len(post_resume._metric_history)}, "
        f"suggesting the deque inherited state from a prior "
        f"instance."
    )
    assert out["gap_idk_smoothed"] == pytest.approx(out["gap_idk"]), (
        f"Fresh instance's first eval should have smoothed == unsmoothed "
        f"(single-point window), but got smoothed="
        f"{out['gap_idk_smoothed']} vs unsmoothed={out['gap_idk']}."
    )
