"""Unit tests for :func:`redact.leak_classifier_metrics.gap_idk`.

The metric:

    gap_idk = min(P_leak | label == 1)
              - max(P_leak | label == 0 ∧ source == "idk")

Tests cover:

- Basic formula on hand-set fixtures (positive and negative gap).
- Source-filter correctness: retain / aux / forget negatives are
  ignored; idk negatives count. Under the (source, source_tags) schema
  both prose and paraphrase IDK rows share ``source == "idk"`` (the
  paraphrase distinction moved to a tag), so a single primary check
  covers both.
- Length-mismatch / empty-positive / empty-IDK-negative validation.
- Tie at the boundary (gap == 0).
- Output range bounded by ``[-1, 1]``.
"""
from __future__ import annotations

import numpy as np
import pytest

from redact.leak_classifier_metrics import gap_idk, max_p_idk, min_p_pos


# ---------- basic formula ----------


def test_positive_gap_perfect_separation():
    """Positives all score above the highest IDK negative → positive
    gap equal to (min_pos − max_idk_neg).
    """
    # min(pos) = 0.7, max(idk_neg) = 0.4 → gap = 0.3
    p = np.array([0.9, 0.7, 0.8, 0.2, 0.3, 0.4])
    y = np.array([1, 1, 1, 0, 0, 0])
    sources = [
        "forget", "forget", "forget",
        "idk", "idk", "idk",
    ]
    g = gap_idk(p, y, sources)
    assert g == pytest.approx(0.3)


def test_negative_gap_when_idk_overlaps_positives():
    """An IDK negative scoring above the lowest positive → negative gap."""
    # min(pos) = 0.4; max(idk_neg) = 0.6; gap = -0.2.
    p = np.array([0.4, 0.7, 0.5, 0.6])
    y = np.array([1, 1, 0, 0])
    sources = ["forget", "forget", "idk", "idk"]
    g = gap_idk(p, y, sources)
    assert g == pytest.approx(-0.2)


def test_zero_gap_at_tie_boundary():
    """min(pos) == max(idk_neg) → gap == 0. Bounds the separating
    threshold to a single point on the deployable axis."""
    p = np.array([0.5, 0.9, 0.5, 0.2])
    y = np.array([1, 1, 0, 0])
    sources = ["forget", "forget", "idk", "idk"]
    g = gap_idk(p, y, sources)
    assert g == pytest.approx(0.0)


# ---------- source-filter correctness ----------


def test_only_clean_idk_negatives_count():
    """Negatives from sources other than ``clean_idk`` are ignored.
    Constructs a fixture where the highest-scoring negative overall is
    ``clean_retain`` (P(leak)=0.99) but the highest-scoring IDK
    negative is 0.3 — the metric must use the IDK max, not the global
    max."""
    p = np.array([0.6, 0.7, 0.99, 0.5, 0.3, 0.2])
    y = np.array([1, 1, 0, 0, 0, 0])
    sources = [
        "forget", "forget",
        "retain",     # high but ignored
        "aux",        # mid but ignored
        "idk",        # the max IDK negative
        "idk",
    ]
    # gap = min(0.6, 0.7) - max(0.3) = 0.3
    assert gap_idk(p, y, sources) == pytest.approx(0.3)


def test_idk_prose_and_paraphrase_both_count_via_primary():
    """Both prose and paraphrase IDK rows share ``source == "idk"`` under
    the (source, source_tags) schema (paraphrase is a tag, not a separate
    source), so both count toward the IDK negative pool."""
    p = np.array([0.7, 0.6, 0.2, 0.35])
    y = np.array([1, 1, 0, 0])
    sources = [
        "forget", "forget",
        "idk", "idk",  # prose + paraphrase both source='idk' → both count
    ]
    # gap = 0.6 - max(0.2, 0.35) = 0.25
    assert gap_idk(p, y, sources) == pytest.approx(0.25)


def test_clean_forget_label_zero_excluded_from_idk_pool():
    """`clean_forget` background rows (label=0) are *not* IDK
    negatives — they only count as eligible IDK negatives when the
    source starts with ``clean_idk``. Excluded here regardless of
    label."""
    p = np.array([0.6, 0.7, 0.95, 0.3])
    y = np.array([1, 1, 0, 0])
    sources = [
        "forget", "forget",
        "forget",   # label=0, but NOT an IDK negative
        "idk",      # the only IDK negative
    ]
    # gap = 0.6 - 0.3 = 0.3 (NOT 0.6 - 0.95)
    assert gap_idk(p, y, sources) == pytest.approx(0.3)


def test_positives_count_regardless_of_source():
    """Per the family convention, any ``label == 1`` row is a positive
    — including ``corrupted_*`` confirmed leaks if present in the
    eval split."""
    p = np.array([0.5, 0.6, 0.7, 0.2])
    y = np.array([1, 1, 1, 0])
    sources = [
        "forget",       # positive
        "forget",  # positive (was corrupted_garbage) — counts
        "forget",  # positive (was corrupted_random) — counts
        "idk",
    ]
    # min(pos) = 0.5, max(idk_neg) = 0.2 → gap = 0.3
    assert gap_idk(p, y, sources) == pytest.approx(0.3)


# ---------- validation ----------


def test_length_mismatch_raises():
    p = np.array([0.5, 0.6])
    y = np.array([1, 0])
    sources = ["forget"]   # length 1, not 2
    with pytest.raises(ValueError, match=r"sources length"):
        gap_idk(p, y, sources)


def test_no_positives_raises():
    """No label=1 rows → cannot compute min over an empty set; fail
    loud rather than return NaN, matching the rest of the family."""
    p = np.array([0.2, 0.3])
    y = np.array([0, 0])
    sources = ["idk", "idk"]
    with pytest.raises(ValueError, match="no positive"):
        gap_idk(p, y, sources)


def test_no_idk_negatives_raises():
    """No clean_idk negatives → cannot compute max over an empty set."""
    p = np.array([0.5, 0.6, 0.3])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "retain"]
    with pytest.raises(ValueError, match="no eligible negative"):
        gap_idk(p, y, sources)


def test_only_aux_negatives_raises():
    """An eval split with clean_aux negatives but no clean_idk
    negatives still raises — the filter is IDK-specific."""
    p = np.array([0.5, 0.3, 0.4])
    y = np.array([1, 0, 0])
    sources = ["forget", "aux", "aux"]
    with pytest.raises(ValueError, match="no eligible negative"):
        gap_idk(p, y, sources)


# ---------- range bound ----------


def test_output_bounded_in_minus_one_to_one():
    """Worst-case gap: positives all at p=0, IDK negatives all at p=1
    → gap = -1. Best-case gap: positives at 1, IDK negatives at 0 →
    gap = 1."""
    sources = ["forget", "idk"]
    # Worst case
    assert gap_idk(np.array([0.0, 1.0]), np.array([1, 0]), sources) == \
        pytest.approx(-1.0)
    # Best case
    assert gap_idk(np.array([1.0, 0.0]), np.array([1, 0]), sources) == \
        pytest.approx(1.0)


def test_accepts_list_inputs():
    """The signature accepts ``np.ndarray | list[float]`` (consistent
    with the rest of the metric family). Plain Python lists work."""
    p = [0.9, 0.7, 0.3, 0.4]
    y = [1, 1, 0, 0]
    sources = ["forget", "forget", "idk", "idk"]
    g = gap_idk(p, y, sources)
    assert g == pytest.approx(0.3)


# ---------- min_p_pos ----------


def test_min_p_pos_returns_smallest_positive_p_leak():
    """``min_p_pos`` returns ``min(P_leak | label == 1)``."""
    p = np.array([0.9, 0.7, 0.85, 0.3, 0.2])
    y = np.array([1, 1, 1, 0, 0])
    sources = ["forget"] * 3 + ["idk"] * 2
    # min over [0.9, 0.7, 0.85] = 0.7
    assert min_p_pos(p, y, sources) == pytest.approx(0.7)


def test_min_p_pos_ignores_negatives():
    """A high-confidence negative with P(leak)=0.99 must NOT lower
    ``min_p_pos``; only label=1 rows count."""
    p = np.array([0.6, 0.7, 0.99, 0.2])
    y = np.array([1, 1, 0, 0])
    sources = ["forget", "forget", "idk", "idk"]
    # min over positives [0.6, 0.7] = 0.6; the 0.99 negative is ignored.
    assert min_p_pos(p, y, sources) == pytest.approx(0.6)


def test_min_p_pos_positives_are_source_agnostic():
    """A label=1 row counts as a positive regardless of source — same
    convention as ``gap_idk`` and the rest of the family."""
    p = np.array([0.6, 0.5, 0.7, 0.3])
    y = np.array([1, 1, 1, 0])
    sources = [
        "forget",      # positive
        "forget",  # positive (was corrupted_random) — counts
        "forget",  # positive (was corrupted_garbage) — counts
        "idk",
    ]
    # min over [0.6, 0.5, 0.7] = 0.5
    assert min_p_pos(p, y, sources) == pytest.approx(0.5)


def test_min_p_pos_raises_on_no_positives():
    p = np.array([0.2, 0.3, 0.4])
    y = np.array([0, 0, 0])
    sources = ["idk"] * 3
    with pytest.raises(ValueError, match="no positive"):
        min_p_pos(p, y, sources)


def test_min_p_pos_raises_on_no_idk_negatives():
    """The shared row filter requires IDK negatives present even
    though ``min_p_pos`` is structurally independent of them. The
    wrapper catches this and emits NaN; standalone callers see the
    raise. Pinned here so the documented contract holds."""
    p = np.array([0.5, 0.6, 0.4])
    y = np.array([1, 1, 0])
    sources = ["forget", "forget", "retain"]
    with pytest.raises(ValueError, match="no eligible negative"):
        min_p_pos(p, y, sources)


# ---------- max_p_idk ----------


def test_max_p_idk_returns_largest_idk_negative_p_leak():
    """``max_p_idk`` returns ``max(P_leak | label == 0 ∧
    source == "idk")``."""
    p = np.array([0.9, 0.5, 0.2, 0.4, 0.3])
    y = np.array([1, 1, 0, 0, 0])
    sources = [
        "forget", "forget",
        "idk", "idk", "idk",
    ]
    # max over IDK negatives [0.2, 0.4, 0.3] = 0.4
    assert max_p_idk(p, y, sources) == pytest.approx(0.4)


def test_max_p_idk_ignores_non_idk_negatives():
    """A confident retain-source false positive (P(leak)=0.95) does NOT
    raise ``max_p_idk``; only IDK negatives count."""
    p = np.array([0.6, 0.7, 0.95, 0.5, 0.3])
    y = np.array([1, 1, 0, 0, 0])
    sources = [
        "forget", "forget",
        "retain",  # high but ignored
        "aux",     # mid but ignored
        "idk",     # the only IDK negative — drives max
    ]
    # max over IDK negatives [0.3] = 0.3 (NOT 0.95)
    assert max_p_idk(p, y, sources) == pytest.approx(0.3)


def test_max_p_idk_prose_and_paraphrase_both_count():
    """Both prose and paraphrase IDK rows share ``source == "idk"`` and
    count toward the IDK negative pool (paraphrase is a tag)."""
    p = np.array([0.7, 0.6, 0.2, 0.45])
    y = np.array([1, 1, 0, 0])
    sources = [
        "forget", "forget",
        "idk", "idk",
    ]
    # max over IDK negatives [0.2, 0.45] = 0.45
    assert max_p_idk(p, y, sources) == pytest.approx(0.45)


def test_max_p_idk_raises_on_no_idk_negatives():
    p = np.array([0.5, 0.6, 0.3, 0.4])
    y = np.array([1, 1, 0, 0])
    sources = ["forget", "forget", "retain", "aux"]
    with pytest.raises(ValueError, match="no eligible negative"):
        max_p_idk(p, y, sources)


def test_max_p_idk_raises_on_no_positives():
    p = np.array([0.2, 0.3])
    y = np.array([0, 0])
    sources = ["idk", "idk"]
    with pytest.raises(ValueError, match="no positive"):
        max_p_idk(p, y, sources)


# ---------- gap_idk = min_p_pos - max_p_idk consistency ----------


def test_gap_equals_min_minus_max():
    """The wrapper emits gap_idk as min_p_pos - max_p_idk derived from
    the same filtered set. Pin this algebraic identity on a fixture
    so a refactor of either component can't drift gap from its
    components silently."""
    p = np.array([0.85, 0.75, 0.2, 0.5, 0.3])
    y = np.array([1, 1, 0, 0, 0])
    sources = ["forget"] * 2 + ["idk"] * 3
    m_pos = min_p_pos(p, y, sources)
    m_idk = max_p_idk(p, y, sources)
    g = gap_idk(p, y, sources)
    assert g == pytest.approx(m_pos - m_idk, abs=1e-12)
