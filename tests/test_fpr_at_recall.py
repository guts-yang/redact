"""Tests for ``redact.training.fpr_at_recall_retain_aux``.

Single operating-point companion diagnostic to
:func:`partial_auroc_above_recall_retain_aux`. Lower is better.

Covers:
- Perfect / random / anti-perfect classifier behavior.
- Same row-filtering invariants as the rest of the retain/aux family.
- recall_target validation and boundary cases (recall_target=1.0).
- Length-mismatch validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from redact.leak_classifier_metrics import fpr_at_recall_retain_aux


# ---------- helpers ----------

def _build_eval_rows(spec):
    p = np.array([s[0] for s in spec], dtype=np.float64)
    y = np.array([s[1] for s in spec], dtype=np.int64)
    src = [s[2] for s in spec]
    return p, y, src


# ---------- perfect / random / anti-perfect ----------

def test_perfect_classifier_returns_zero():
    """All positives above all negatives → 0 retain FPR achieves any recall."""
    spec = []
    for _ in range(40):
        spec.append((0.9, 1, "forget"))
    for _ in range(20):
        spec.append((0.9, 1, "forget"))
    for _ in range(80):
        spec.append((0.1, 0, "retain"))
    for _ in range(40):
        spec.append((0.1, 0, "aux"))
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_anti_perfect_classifier_returns_one():
    """All positives below all negatives → must flag every negative
    before catching any positive, so FPR = 1.0 at any recall_target > 0."""
    spec = [
        *[(0.1, 1, "forget") for _ in range(40)],
        *[(0.9, 0, "retain") for _ in range(80)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(1.0)


def test_random_scores_returns_roughly_recall_target():
    """For a random classifier, FPR ≈ recall_target (the ROC is the
    diagonal, so to reach TPR=0.95 you need FPR≈0.95).
    """
    rng = np.random.default_rng(42)
    n_pos, n_neg = 400, 800
    p = rng.uniform(0.0, 1.0, size=n_pos + n_neg)
    y = np.array([1] * n_pos + [0] * n_neg)
    src = ["forget"] * n_pos + ["retain"] * n_neg
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert 0.85 < fpr < 1.0


# ---------- source filtering ----------

def test_clean_idk_negatives_counted():
    """clean_idk* rows count toward FPR. With clean_idk* scoring above
    positives, the threshold that catches positives also flags every
    clean_idk row → FPR is dominated by the IDK contribution.

    Setup: 40 positives at 0.9; 80 clean_retain + 40 clean_aux at 0.1
    (below positives, won't fire); 60 clean_idk + 30 clean_idk_paraphrase
    at 0.95 (above positives). Eligible negatives = 80 + 40 + 60 + 30
    = 210. At threshold dropping to catch positives (just below 0.9),
    all 90 clean_idk* are already flagged → FPR ≈ 90/210 ≈ 0.43.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(40)],
        *[(0.1, 0, "retain") for _ in range(80)],
        *[(0.1, 0, "aux") for _ in range(40)],
        *[(0.95, 0, "idk") for _ in range(60)],
        *[(0.95, 0, "idk") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(90 / 210, abs=1e-9)


def test_clean_forget_background_negatives_excluded():
    spec = [
        *[(0.9, 1, "forget") for _ in range(40)],
        *[(0.1, 0, "retain") for _ in range(80)],
        *[(0.95, 0, "forget") for _ in range(50)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_corrupted_background_negatives_excluded():
    spec = [
        *[(0.9, 1, "forget") for _ in range(40)],
        *[(0.1, 0, "retain") for _ in range(80)],
        *[(0.95, 0, "forget") for _ in range(50)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_clean_retain_xfer_counts_as_negative():
    """clean_retain_xfer must count as a retain negative."""
    spec = [
        # 5 positives at 0.5; 5 retain at 0.9 (above positives); the
        # 5 xfer at 0.95 are EVEN higher. To catch any positive we
        # already crossed all 10 retain rows → FPR = 1.0 at any recall.
        *[(0.5, 1, "forget") for _ in range(5)],
        *[(0.9, 0, "retain") for _ in range(5)],
        *[(0.95, 0, "retain") for _ in range(5)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.2)
    assert fpr == pytest.approx(1.0)


def test_corrupted_label1_counts_as_positive():
    """Positives from corrupted_* count toward recall."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(40)],
        *[(0.1, 0, "retain") for _ in range(80)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


# ---------- empty-class raises ----------

def test_raises_on_empty_positives():
    spec = [
        *[(0.5, 0, "retain") for _ in range(40)],
        *[(0.5, 0, "aux") for _ in range(20)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no positive"):
        fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)


def test_raises_on_empty_eligible_negatives():
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        # Only non-eligible-negative sources for the negatives.
        *[(0.1, 0, "forget") for _ in range(30)],
        *[(0.1, 0, "forget") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no eligible negative"):
        fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)


# ---------- recall_target validation ----------

def test_recall_target_validation():
    spec = [
        *[(0.9, 1, "forget") for _ in range(5)],
        *[(0.1, 0, "retain") for _ in range(5)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="recall_target"):
        fpr_at_recall_retain_aux(p, y, src, recall_target=0.0)
    with pytest.raises(ValueError, match="recall_target"):
        fpr_at_recall_retain_aux(p, y, src, recall_target=1.5)
    with pytest.raises(ValueError, match="recall_target"):
        fpr_at_recall_retain_aux(p, y, src, recall_target=-0.1)


def test_recall_target_one_returns_min_fpr_at_full_recall():
    """recall_target=1.0 = "catch every positive". For perfect separation
    that's still FPR=0; for partial overlap it's the FPR at the
    lowest-scored positive's threshold.
    """
    # Positives at scores 0.9 and 0.4; negatives at 0.6.
    # To catch both positives (TPR=1.0), threshold must drop below 0.4.
    # At that threshold, all negatives at 0.6 are flagged → FPR = 1.0.
    spec = [
        (0.9, 1, "forget"),
        (0.4, 1, "forget"),
        (0.6, 0, "retain"),
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=1.0)
    assert fpr == pytest.approx(1.0)


# ---------- length / type validation ----------

def test_length_mismatch_raises():
    p = np.array([0.9, 0.1])
    y = np.array([1, 0])
    src = ["forget", "retain", "aux"]
    with pytest.raises(ValueError, match="sources length"):
        fpr_at_recall_retain_aux(p, y, src)


def test_accepts_list_inputs():
    fpr = fpr_at_recall_retain_aux(
        p_leak=[0.9, 0.9, 0.1, 0.1],
        labels=[1, 1, 0, 0],
        sources=["forget", "forget", "retain", "aux"],
        recall_target=0.95,
    )
    assert fpr == pytest.approx(0.0)


def test_non_string_sources_excluded_from_negatives():
    """A None source must be defensively excluded (not crash)."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(5)],
        *[(0.1, 0, "retain") for _ in range(5)],
        (0.95, 0, None),
    ]
    p = np.array([s[0] for s in spec], dtype=np.float64)
    y = np.array([s[1] for s in spec], dtype=np.int64)
    src = [s[2] for s in spec]  # type: ignore[list-item]
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_idk_only_eligible_pool():
    """The metric works with an eligible-negative pool that's *only*
    ``source == "idk"`` rows.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(40)],
        *[(0.1, 0, "idk") for _ in range(80)],
        *[(0.1, 0, "idk") for _ in range(40)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_unknown_primary_source_excluded_from_negatives():
    """Post-refactor the eligible-negative set is exactly the primaries
    ``{retain, aux, idk}`` (set membership, no prefix matching). An
    out-of-set source string is NOT an eligible negative — locks in the
    contract so it can't be accidentally widened.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "idk") for _ in range(40)],
        # Out-of-set primary string: must be ignored as a negative, so it
        # never raises retain FPR even at a high score.
        *[(0.95, 0, "some_other_source") for _ in range(30)],
    ]
    p, y, src = _build_eval_rows(spec)
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


def test_non_string_sources_various_types_excluded():
    """Defensive isinstance check rejects all non-string source types."""
    spec = [
        *[(0.9, 1, "forget") for _ in range(5)],
        *[(0.1, 0, "retain") for _ in range(5)],
        (0.95, 0, None),
        (0.95, 0, np.nan),
        (0.95, 0, ""),  # str but no prefix match
        (0.95, 0, {"src": "retain"}),
    ]
    p = np.array([s[0] for s in spec], dtype=np.float64)
    y = np.array([s[1] for s in spec], dtype=np.int64)
    src = [s[2] for s in spec]  # type: ignore[list-item]
    fpr = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    assert fpr == pytest.approx(0.0)


# ---------- source_primaries: single-source restriction ----------

def test_source_primaries_single_bucket_restricts_negatives():
    """Per-source FPR computed via source_prefixes matches a hand
    derivation. Setup: 20 positives at 0.9; 100 clean_retain at 0.1;
    10 clean_idk at 0.95.
    """
    spec = [
        *[(0.9, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "retain") for _ in range(100)],
        *[(0.95, 0, "idk") for _ in range(10)],
    ]
    p, y, src = _build_eval_rows(spec)
    # Retain-only: positives at 0.9, retain at 0.1 → FPR=0 at recall=0.95.
    fpr_retain = fpr_at_recall_retain_aux(
        p, y, src, recall_target=0.95, source_primaries=("retain",),
    )
    # IDK-only: all 10 IDK rows score above all positives → to reach
    # any positive (recall>0), threshold must drop below 0.9, flagging
    # all 10 IDK → FPR=1.0.
    fpr_idk = fpr_at_recall_retain_aux(
        p, y, src, recall_target=0.95, source_primaries=("idk",),
    )
    assert fpr_retain == pytest.approx(0.0)
    assert fpr_idk == pytest.approx(1.0)


def test_source_primaries_empty_bucket_raises():
    spec = [
        *[(0.9, 1, "forget") for _ in range(10)],
        *[(0.1, 0, "retain") for _ in range(40)],
    ]
    p, y, src = _build_eval_rows(spec)
    with pytest.raises(ValueError, match="no eligible negative"):
        fpr_at_recall_retain_aux(
            p, y, src, recall_target=0.95, source_primaries=("idk",),
        )


def test_macro_average_pattern_balances_sources():
    """fpr_at_recall is in the *opposite* direction from pAUC: lower
    is better. So pooled (aux-dominated, clean separation) gives a
    LOW value; macro should be HIGHER than pooled when retain/IDK are
    bad.

    Setup: 20 positives at 0.5; 1000 aux at 0.1 (below positives);
    5 retain at 0.95 (above positives); 5 IDK at 0.95.

    Per-source:
    - aux alone: positives above all aux → FPR at recall=0.95 is 0.
    - retain alone (resp. IDK): all 5 negs above positives → must
      flag all 5 to catch any positive → FPR = 1.0.
    - Macro = (1.0 + 0.0 + 1.0) / 3 ≈ 0.666667.

    Pooled at recall=0.95: TPR=0.95 reached at threshold catching
    19/20 positives. Need to drop threshold below 0.5 — but at that
    point we've already flagged all 10 retain+IDK rows above 0.95
    (since they score higher). Aux rows (1000 at 0.1) are not yet
    flagged. FPR = 10/1010 ≈ 0.0099. Pooled is near 0.

    So macro (0.67) >> pooled (0.01) — the inversion is the point.
    """
    spec = [
        *[(0.5, 1, "forget") for _ in range(20)],
        *[(0.1, 0, "aux") for _ in range(1000)],
        *[(0.95, 0, "retain") for _ in range(5)],
        *[(0.95, 0, "idk") for _ in range(5)],
    ]
    p, y, src = _build_eval_rows(spec)
    pooled = fpr_at_recall_retain_aux(p, y, src, recall_target=0.95)
    per_source = {
        bucket: fpr_at_recall_retain_aux(
            p, y, src, recall_target=0.95, source_primaries=(prefix,),
        )
        for bucket, prefix in (
            ("retain", "retain"),
            ("aux", "aux"),
            ("idk", "idk"),
        )
    }
    macro = float(np.mean(list(per_source.values())))
    assert per_source["aux"] == pytest.approx(0.0)
    assert per_source["retain"] == pytest.approx(1.0)
    assert per_source["idk"] == pytest.approx(1.0)
    assert macro == pytest.approx(2 / 3, abs=1e-9)
    # Pooled near 0 (aux dominates negative pool, FPR is small).
    assert pooled < 0.05
    # Macro is MUCH higher (lower is better; aux dominance hid retain/IDK).
    assert macro > pooled
