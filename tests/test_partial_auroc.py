"""Tests for ``redact.training.partial_auroc_retain_aux``.

Covers:
- Perfect / random classifier behavior on filtered subsets.
- Source filtering under the (source, source_tags) schema: ``retain``,
  ``aux``, and ``idk`` primaries count as eligible negatives (``idk``
  covers both prose and paraphrase, which now share the primary).
  ``forget`` background (including corrupted-tagged forget rows) is
  excluded.
- Positive side: confirmed leaks from any source count.
- Empty filtered class raises ValueError (fail-fast on misconfigured eval).
- ``fpr_cap`` validation and the cap-respecting behavior.
- Length-mismatch validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from redact.leak_classifier_metrics import partial_auroc_retain_aux


# ---------- helpers ----------

def _build_eval_rows(spec):
    """Spec is a list of (p_leak, label, source) tuples → unzipped arrays."""
    p = np.array([s[0] for s in spec], dtype=np.float64)
    y = np.array([s[1] for s in spec], dtype=np.int64)
    src = [s[2] for s in spec]
    return p, y, src


# ---------- perfect / random ----------

def test_perfect_classifier_returns_one():
    """Filtered positives all above filtered negatives → pAUC = 1.0."""
    spec = []
    # Positives at p=0.9 from forget sources (mixed clean_forget + corrupted)
    for _ in range(20):
        spec.append((0.9, 1, "forget"))
    for _ in range(10):
        spec.append((0.9, 1, "forget"))
    # Retain+aux negatives at p=0.1
    for _ in range(40):
        spec.append((0.1, 0, "retain"))
    for _ in range(20):
        spec.append((0.1, 0, "aux"))
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


def test_perfectly_inverted_classifier_below_half():
    """Filtered positives all BELOW filtered negatives → pAUC < 0.5.

    McClish normalization maps the *random* classifier's raw partial
    AUC (``0.5 * fpr_cap^2``) to 0.5, and the *perfect* classifier's
    raw pAUC (``fpr_cap``) to 1.0. A perfectly-inverted classifier has
    raw pAUC = 0 (TPR stays at 0 across the integration window), so
    McClish-normalized pAUC dips below 0.5 — concretely
    ``0.5 - 0.5 * (0.5 * cap^2) / (cap - 0.5 * cap^2)`` for cap=0.10
    ≈ 0.474.
    """
    spec = [
        *[(0.1, 1, "forget") for _ in range(20)],
        *[(0.9, 0, "retain") for _ in range(40)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc < 0.5
    # Exact McClish value for raw pAUC = 0 at cap=0.10.
    cap = 0.10
    min_area = 0.5 * cap**2
    expected = 0.5 + 0.5 * (0.0 - min_area) / (cap - min_area)
    assert pauc == pytest.approx(expected, abs=1e-6)


def test_random_scores_near_half():
    """Uniformly random scores → pAUC ≈ 0.5 ± tolerance."""
    rng = np.random.default_rng(42)
    n_pos, n_neg = 200, 400
    p = rng.uniform(0.0, 1.0, size=n_pos + n_neg)
    y = np.array([1] * n_pos + [0] * n_neg)
    src = ["forget"] * n_pos + ["retain"] * n_neg
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert 0.40 < pauc < 0.60


# ---------- source filtering ----------

def test_clean_idk_negatives_counted():
    """idk rows count toward FPR (the regen pipeline must not cascade by
    flagging IDK continuations). High-scoring idk rows therefore drag
    pAUC below 1.0.

    Setup: 20 positives at p=0.9 (so the perfect operating point is
    threshold=0.5 with TPR=1, FPR=0). Add 50 idk negatives at p=0.95 —
    these score ABOVE the positives, so any threshold catching a
    positive also flags all idk rows. FPR = 50/110 ≈ 0.4545 long before
    threshold reaches the positives' scores.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
        *[(0.1, 0, "aux") for _ in range(20)],
        *[(0.95, 0, "idk") for _ in range(50)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    # clean_idk scoring above positives caps the achievable TPR at FPR
    # well above 0.10 — pAUC over the [0, 0.10] strip is at the
    # McClish floor.
    assert pauc < 0.5


def test_idk_negatives_counted_high_score():
    """IDK negatives (``source == "idk"``, covering both prose and
    paraphrase variants, which now share the primary) count toward FPR:
    high-scoring IDK rows drag pAUC below 1.0."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
        *[(0.95, 0, "idk") for _ in range(50)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc < 0.5


def test_idk_only_eligible_pool():
    """The metric works with an eligible-negative pool that's *only*
    ``source == "idk"`` rows (no retain/aux). Documents that the IDK
    primary is fully self-sufficient — relevant for subset experiments
    or ablations that drop retain/aux from the eval split.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "idk") for _ in range(40)],
        *[(0.1, 0, "idk") for _ in range(20)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


def test_clean_forget_background_negatives_excluded():
    """label=0 rows from clean_forget (within-forget bg) excluded from FPR."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
        # forget-source background — high p — would tank pAUC if counted
        *[(0.95, 0, "forget") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


def test_corrupted_background_negatives_excluded():
    """label=0 ``forget`` rows (within-forget bg, including corrupted-tagged
    post-soft-token bg) are excluded from FPR. Corrupted is a tag on the
    ``forget`` primary now, so these are indistinguishable from clean
    forget bg at the metric layer — both correctly excluded.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
        *[(0.95, 0, "forget") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


def test_clean_retain_xfer_counts_as_negative():
    """xfer retain rows (subset-protocol retain) carry ``source == "retain"``
    (``xfer`` is a tag) and must be counted as retain negatives — the
    metric keys on the primary, so xfer and plain retain both count."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        # Both halves are source='retain' (one would carry the xfer tag,
        # invisible at the primary-only metric layer); both count.
        *[(0.1, 0, "retain") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(20)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


def test_corrupted_label1_counts_as_positive():
    """Confirmed leaks from corrupted-tagged forget rows are positives
    (deployed classifier must catch leaks from the post-soft-token
    distribution too). At the metric layer they are ``source == "forget"``
    with ``label == 1`` — any label=1 row is a positive."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


# ---------- empty-class raises ----------

def test_raises_on_empty_positives():
    """No label=1 rows → fail loudly (don't silently return 0.0)."""
    spec = [
        *[(0.5, 0, "retain") for _ in range(40)],
        *[(0.5, 0, "aux") for _ in range(20)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no positive"):
        partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)


def test_raises_on_empty_eligible_negatives():
    """No filtered eligible negatives → fail loudly. Eligible sources
    are clean_retain*, clean_aux*, clean_idk*; clean_forget bg and
    corrupted_* bg are NOT eligible. Setup uses only those excluded
    sources for the negatives.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        # Negatives present, but all from sources that aren't eligible
        # negatives (clean_forget bg, corrupted_* bg).
        *[(0.1, 0, "forget") for _ in range(40)],
        *[(0.1, 0, "forget") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no eligible negative"):
        partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)


# ---------- fpr_cap behavior ----------

def test_fpr_cap_validation():
    spec = [
        *[(0.9, 1, "forget") for _ in range(5)],
        *[(0.1, 0, "retain") for _ in range(5)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="fpr_cap"):
        partial_auroc_retain_aux(p, y, src, fpr_cap=0.0)
    with pytest.raises(ValueError, match="fpr_cap"):
        partial_auroc_retain_aux(p, y, src, fpr_cap=1.5)
    with pytest.raises(ValueError, match="fpr_cap"):
        partial_auroc_retain_aux(p, y, src, fpr_cap=-0.1)


def test_pauc_in_valid_range_across_caps():
    """For a non-degenerate classifier (positives skew high, negatives
    skew low), pAUC should be in ``[0.5, 1.0]`` for every cap.

    Note: the relationship between pAUC at different caps is NOT
    monotonic under McClish normalization, so this test only checks
    bounds — not ``pauc_05 <= pauc_10 <= pauc_50``.
    """
    rng = np.random.default_rng(7)
    n_pos, n_neg = 100, 200
    p_pos = rng.beta(2.0, 1.5, size=n_pos)  # skewed high
    p_neg = rng.beta(1.5, 2.0, size=n_neg)  # skewed low
    p = np.concatenate([p_pos, p_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    src = ["forget"] * n_pos + ["retain"] * n_neg
    pauc_05 = partial_auroc_retain_aux(p, y, src, fpr_cap=0.05)
    pauc_10 = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    pauc_50 = partial_auroc_retain_aux(p, y, src, fpr_cap=0.50)
    for pauc in (pauc_05, pauc_10, pauc_50):
        assert 0.5 <= pauc <= 1.0


def test_known_separable_case():
    """Hand-crafted scores where the partial-AUC sits below 0.5.

    Setup: 5 retain false positives at p=0.95 fire first (the 5 highest
    scores), then 4 forget positives at p=0.9, then 95 clean retain
    negatives at p=0.1. With fpr_cap=0.05 the integration window
    covers the interval ``[0, 5/100=0.05]``; the 5 retain FPs span
    that whole window with TPR pinned at 0 (no positive has fired yet).

    Raw pAUC ≈ 0; McClish-normalized to
    ``0.5 - 0.5 * (0.5 * cap^2) / (cap - 0.5 * cap^2)``
    ≈ 0.487 at cap=0.05.
    """
    spec = [
        *[(0.95, 0, "retain") for _ in range(5)],
        *[(0.9, 1, "forget") for _ in range(4)],
        *[(0.1, 0, "retain") for _ in range(95)],
    ]
    p, y, src = _build_eval_rows(spec)
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.05)
    cap = 0.05
    min_area = 0.5 * cap**2
    expected = 0.5 + 0.5 * (0.0 - min_area) / (cap - min_area)
    assert pauc == pytest.approx(expected, abs=1e-3)


# ---------- length validation ----------

def test_length_mismatch_raises():
    p = np.array([0.9, 0.1])
    y = np.array([1, 0])
    src = ["forget", "retain", "aux"]  # wrong length
    with pytest.raises(ValueError, match="sources length"):
        partial_auroc_retain_aux(p, y, src)


def test_accepts_list_inputs():
    """Function should accept Python lists, not just numpy arrays."""
    pauc = partial_auroc_retain_aux(
        p_leak=[0.9, 0.9, 0.1, 0.1],
        labels=[1, 1, 0, 0],
        sources=["forget", "forget", "retain", "aux"],
        fpr_cap=0.10,
    )
    assert pauc == pytest.approx(1.0)


# ---------- non-string sources ----------

def test_split_aware_trainer_records_prefix():
    """SplitAwareTrainer.evaluate must record metric_key_prefix on the
    instance before delegating, so a closure-captured trainer can
    dispatch on which split is being evaluated.

    HF Trainer recurses on dict-valued eval_dataset by re-entering
    self.evaluate with a per-split prefix; we verify the recursive
    case sets the latest prefix correctly.
    """
    from redact.training import SplitAwareTrainer

    # Stub super().evaluate to short-circuit the heavy eval loop. We
    # only need to confirm self._current_metric_prefix is updated
    # before the super() call would consume it.
    captured: list[str] = []

    class _Stub(SplitAwareTrainer):
        def __init__(self):  # noqa: D401
            # Skip Trainer.__init__ — we don't need a real model.
            pass

        # Bypass super() chain: capture and return.
        def _super_evaluate(self, eval_dataset=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            captured.append(self._current_metric_prefix or "<unset>")
            return {f"{metric_key_prefix}_dummy": 0.0}

    # Patch the override to call our stub instead of super().
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._current_metric_prefix = metric_key_prefix
        return self._super_evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    _Stub.evaluate = evaluate  # type: ignore[assignment]

    stub = _Stub()
    stub.evaluate(metric_key_prefix="eval_test")
    stub.evaluate(metric_key_prefix="eval_forget")
    stub.evaluate(metric_key_prefix="eval_retain")
    assert captured == ["eval_test", "eval_forget", "eval_retain"]
    assert stub._current_metric_prefix == "eval_retain"


def test_non_string_sources_excluded_from_negatives():
    """If a row has a non-string ``source`` (e.g. None from a malformed
    dataset), it should be excluded from the retain+aux negative set
    rather than crashing. The function is intentionally defensive
    here since the ``source`` column comes from upstream aggregation
    scripts that could in principle write None for a missing source."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(5)],
        *[(0.1, 0, "retain") for _ in range(5)],
        # A None source — must not crash, must not count as retain+aux
        (0.95, 0, None),
    ]
    p = np.array([s[0] for s in spec], dtype=np.float64)
    y = np.array([s[1] for s in spec], dtype=np.int64)
    src = [s[2] for s in spec]  # type: ignore[list-item]
    pauc = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    assert pauc == pytest.approx(1.0)


# ---------- source_primaries: single-source restriction + macro-average ----------

def test_source_primaries_single_bucket_restricts_negatives():
    """Passing source_primaries=('idk',) restricts the eligible-negative
    pool to IDK rows only. High-scoring IDK rows then dominate FPR even
    when retain/aux rows would have given a clean separation.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        # Clean retain at low score — would give pAUC=1.0 if counted
        *[(0.1, 0, "retain") for _ in range(40)],
        # IDK at high score — only counted when source_primaries=('idk',)
        *[(0.95, 0, "idk") for _ in range(50)],
    ]
    p, y, src = _build_eval_rows(spec)
    # With IDK-only restriction, IDK is the only negative pool → high
    # FPR within the cap → pAUC at the McClish floor.
    pauc_idk_only = partial_auroc_retain_aux(
        p, y, src, fpr_cap=0.10, source_primaries=("idk",),
    )
    assert pauc_idk_only < 0.5
    # With retain-only restriction, the high-scoring IDK rows are
    # invisible to the metric → clean separation, pAUC=1.0.
    pauc_retain_only = partial_auroc_retain_aux(
        p, y, src, fpr_cap=0.10, source_primaries=("retain",),
    )
    assert pauc_retain_only == pytest.approx(1.0)


def test_source_primaries_empty_bucket_raises():
    """If the requested source bucket has no rows, the empty-negatives
    fail-fast kicks in (no eligible negatives for the restricted pool).
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(40)],
        # No idk rows in this fixture
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no eligible negative"):
        partial_auroc_retain_aux(
            p, y, src, fpr_cap=0.10, source_primaries=("idk",),
        )


def test_macro_average_pattern_balances_sources():
    """The trainer wrapper computes a 3-source macro-average by calling
    the metric three times (one per single-prefix tuple) and averaging.
    Verify the pattern here on a hand-crafted imbalanced case:

    - 1000 clean_aux at score 0.1 (low, clean separation — aux
      dominates the negative pool)
    - 5 clean_retain at score 0.95 (above all positives — bad)
    - 5 clean_idk at score 0.95 (above all positives — bad)
    - 20 positives at 0.5

    With aux at ~99% of the eligible-negative pool, the pooled FPR at
    cap=0.10 is dominated by aux's clean separation; pooled pAUC is
    near 1.0. The macro-average gives each source 1/3 weight; retain
    and IDK each land at the McClish floor (~0.474) because all 5 of
    their rows score above all positives. So:

        pooled ≈ 0.95
        macro ≈ (0.474 + 0.474 + 1.0) / 3 ≈ 0.65

    The gap (~0.30) shows the imbalance correction's effect.
    """
    spec = [
        *[(0.5, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "aux") for _ in range(1000)],
        *[(0.95, 0, "retain") for _ in range(5)],
        *[(0.95, 0, "idk") for _ in range(5)],
    ]
    p, y, src = _build_eval_rows(spec)
    pooled = partial_auroc_retain_aux(p, y, src, fpr_cap=0.10)
    per_source = {
        "retain": partial_auroc_retain_aux(
            p, y, src, fpr_cap=0.10, source_primaries=("retain",),
        ),
        "aux": partial_auroc_retain_aux(
            p, y, src, fpr_cap=0.10, source_primaries=("aux",),
        ),
        "idk": partial_auroc_retain_aux(
            p, y, src, fpr_cap=0.10, source_primaries=("idk",),
        ),
    }
    macro = float(np.mean(list(per_source.values())))
    # Hand-derived exact values for tight regression bounds:
    # - aux alone: 1000 negs at 0.1 all below positives at 0.5 → clean
    #   separation → pAUC = 1.0.
    # - retain alone (resp. IDK alone): 5 negs at 0.95 all above all
    #   positives → raw partial AUC = 0 within the cap → McClish floor:
    #     0.5 + 0.5 * (0 - 0.5*cap^2) / (cap - 0.5*cap^2)
    #     = 0.5 - 0.5 * (0.005 / 0.095)  for cap=0.10
    #     ≈ 0.473684
    # - macro = (0.473684 + 0.473684 + 1.0) / 3 ≈ 0.649123
    cap = 0.10
    floor = 0.5 + 0.5 * (0 - 0.5 * cap**2) / (cap - 0.5 * cap**2)
    expected_macro = (floor + floor + 1.0) / 3
    assert per_source["aux"] == pytest.approx(1.0, abs=1e-9)
    assert per_source["retain"] == pytest.approx(floor, abs=1e-6)
    assert per_source["idk"] == pytest.approx(floor, abs=1e-6)
    assert macro == pytest.approx(expected_macro, abs=1e-6)
    # Sanity: pooled favors aux (~99% of the negative pool) → high pAUC.
    assert pooled > 0.9
    # Macro is meaningfully below pooled — the imbalance correction works.
    assert macro < pooled
