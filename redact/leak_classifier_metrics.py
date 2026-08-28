"""Metric helpers for the leak-classifier training pipeline.

Extracted from :mod:`redact.training` and :mod:`scripts.train_classifier`
so the metric/wrapper logic is importable without running the script's
module-level CLI parse + ``load_from_disk`` side effects.

What lives here:

- The eligible-negative pool definition (:data:`_ELIGIBLE_NEGATIVE_PRIMARIES`)
  and the :data:`PER_SOURCE_BUCKETS` mapping used by the
  source-balanced macro-average.
- The shared row filter (:func:`_filter_eval_rows_retain_aux`) and the
  three pAUC / FPR-at-recall metrics
  (:func:`partial_auroc_retain_aux`,
  :func:`partial_auroc_above_recall_retain_aux`,
  :func:`fpr_at_recall_retain_aux`).
- The ``(logits, labels) -> (p_leak, binary_labels)`` adapters
  (:func:`binary_p_leak_extractor`, :func:`binary_label_extractor`).
- The HF Trainer ``compute_metrics`` callable
  (:func:`compute_binary_metrics`). It accepts its config via
  keyword-only parameters; the train script wires the values via
  :func:`functools.partial`.
- The Trainer-eval orchestrator :class:`LeakMetricsAggregator` that
  bolts pAUC, IDK-pool, focal-mass, and rolling-smoothing components
  onto any base ``compute_metrics`` callable.

What does NOT live here: anything that touches the HF Trainer or
dataset loading. Those stay in :mod:`redact.training`.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve

from redact.utils import log_print


# ----------------------------------------------------------------------
# Eligible-negative pool: shared row filter + per-source bucket map
# ----------------------------------------------------------------------

# Primary source values that count as eligible negatives for the leak-
# classifier metrics. Defined as a constant on the module-level so the
# filter helper and any future diagnostic share the same definition
# (divergence here would silently make the metrics incomparable on the
# same eval pass). Under the (source, source_tags) schema, every
# retain/idk/aux row carries its primary source directly; paraphrase
# variants of idk are tag-distinguished and still match source=='idk'.
_ELIGIBLE_NEGATIVE_PRIMARIES = frozenset({"retain", "aux", "idk"})

# Per-source buckets for the source-balanced pAUC macro-average and
# the always-on per-source diagnostic logged by the leak-classifier
# Trainer wrapper. Each bucket is a (short_name, primary) pair where
# ``primary`` is passed to the metric functions' ``source_primaries=``
# kwarg to restrict the eligible-negative pool. Order matches the
# canonical reporting order (retain, aux, idk).
#
# Tied to :data:`_ELIGIBLE_NEGATIVE_PRIMARIES`: every primary in the
# eligible-negative set must appear in exactly one bucket here, and
# vice versa. If a new primary is added to ``_ELIGIBLE_NEGATIVE_PRIMARIES``
# without a matching entry here, the macro-average would silently
# cover a strict subset of the pooled FPR pool — equivalent to
# weighting the new source as zero. Asserted at module import.
PER_SOURCE_BUCKETS = (
    ("retain", "retain"),
    ("aux", "aux"),
    ("idk", "idk"),  # also matches idk+paraphrase via source==idk
)
if (
    frozenset(primary for _, primary in PER_SOURCE_BUCKETS)
    != _ELIGIBLE_NEGATIVE_PRIMARIES
):
    # Plain ``if/raise`` rather than ``assert`` so the invariant
    # survives ``python -O`` (which strips assertions). The docstring
    # above promises "Asserted at module import"; making that
    # invariant load-bearing under every Python invocation mode keeps
    # the promise honest.
    raise RuntimeError(
        "PER_SOURCE_BUCKETS primaries must match "
        "_ELIGIBLE_NEGATIVE_PRIMARIES in content — adding a new "
        "eligible-negative primary requires adding a matching "
        "bucket here."
    )


def _filter_eval_rows_retain_aux(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
    *,
    caller: str,
    source_primaries: tuple[str, ...] = tuple(sorted(_ELIGIBLE_NEGATIVE_PRIMARIES)),
) -> tuple[np.ndarray, np.ndarray]:
    """Filter eval rows to positives + eligible negatives.

    Shared row-selection logic for the retain/aux/IDK metric family
    (:func:`partial_auroc_retain_aux`,
    :func:`partial_auroc_above_recall_retain_aux`,
    :func:`fpr_at_recall_retain_aux`). Keeping all three on a single
    filtering definition is load-bearing: divergence would mean the
    three metrics are computed on subtly different row sets and become
    incomparable on the same eval pass.

    Selection rules:

    - **Positives**: any row with ``label == 1`` (any source —
      ``source=='forget'`` confirmed leaks across all variant tags
      count, since the deployed classifier must catch leaks from
      every forget-bucket distribution).
    - **Negatives**: ``label == 0`` AND ``source in source_primaries``.
      Default eligible primaries are :data:`_ELIGIBLE_NEGATIVE_PRIMARIES`
      (``retain``, ``aux``, ``idk`` — the last covers both
      ``idk + clean`` (prose) and ``idk + paraphrase`` since
      paraphrase is now a tag, not a source).
    - Drops ``source=='forget'`` background (label=0 forget rows).

    The ``source_primaries`` argument exists so callers can compute
    per-source metrics by restricting the eligible-negative pool to a
    single primary (e.g., ``("idk",)``). The macro-averaged
    "source-balanced" pAUC is computed by calling the metric three
    times — once per single-primary subset — and averaging.

    Why ``idk`` is counted as a negative: it is the post-regen
    "I don't know" continuation distribution. A false positive on an
    IDK continuation triggers a *second* regen on top of the first,
    which can cascade and harm both compute and convergence. So the
    classifier must not fire on IDK rows, and the metric must penalize
    it when it does.

    ``source=='forget'`` background (label=0 forget rows) are excluded:
    those are sub-classification decisions on forget-prompt CoTs
    (background sentences in a CoT that *does* eventually leak), not
    the retain/IDK discrimination this metric optimizes — counting
    them as FPs would conflate two different error modes.

    Parameters
    ----------
    p_leak, labels, sources
        Per-row arrays, all of equal length.
    caller
        Name of the calling function — included in raised
        ``ValueError`` messages so failures during Trainer eval point at
        the right metric.
    source_primaries
        Tuple of primary source values whose ``label == 0`` rows count
        as eligible negatives. Defaults to the full eligible pool; pass
        a single-element tuple to restrict to one source (used by the
        source-balanced metric variants).

    Returns
    -------
    (p_filt, y_filt) ndarray pair containing the rows that passed the
    filter, in their original positional order.

    Raises
    ------
    ValueError
        On any of: ``len(p_leak) != len(labels)``,
        ``len(sources) != len(labels)``, ``p_leak`` containing
        non-finite values (NaN / Inf), empty positives, or empty
        eligible negatives. Fail-fast on all of these: returning 0.0
        or a NaN here would silently elect the first checkpoint as
        "best" via Trainer tie-break.
    """
    p_leak_arr = np.asarray(p_leak, dtype=np.float64)
    labels_arr = np.asarray(labels)

    # Length checks first — the boolean mask ``keep`` is constructed
    # from ``labels_arr``, so a longer ``p_leak`` would be silently
    # prefix-truncated by the mask (no IndexError) and a shorter
    # ``p_leak`` would IndexError with a much less informative message
    # than the ValueError pattern below.
    if len(p_leak_arr) != len(labels_arr):
        raise ValueError(
            f"{caller}: p_leak length ({len(p_leak_arr)}) must equal labels "
            f"length ({len(labels_arr)})."
        )
    if len(sources) != len(labels_arr):
        raise ValueError(
            f"{caller}: sources length ({len(sources)}) must equal labels "
            f"length ({len(labels_arr)})."
        )

    # Finiteness: NaN / Inf in ``p_leak`` poisons ``min`` / ``max``
    # (gap_idk) and produces undefined behavior in sklearn's
    # ROC curve / AUC (silent garbage from ``roc_curve``; raise from
    # ``roc_auc_score``). Catch upstream divergence (NaN logits at a
    # recent training step) at the metric boundary with a clear
    # message rather than letting it surface as a confusing
    # "best checkpoint = NaN" outcome.
    if not np.isfinite(p_leak_arr).all():
        n_nonfinite = int((~np.isfinite(p_leak_arr)).sum())
        raise ValueError(
            f"{caller}: p_leak contains {n_nonfinite} non-finite value(s) "
            f"(NaN/Inf); the metric family assumes finite predictions. "
            f"Usually indicates upstream model divergence (NaN logits at "
            f"a recent training step)."
        )

    primaries_set = set(source_primaries)
    is_positive = labels_arr == 1
    # Defensive guard: non-string sources (None, NaN, dict, ...) never
    # match any primary. ``s in primaries_set`` would raise TypeError on
    # unhashable inputs (dict) and would coincidentally pass on float NaN
    # (np.nan is hashable). Gate on ``isinstance(s, str)`` first.
    is_eligible_negative_source = np.array(
        [isinstance(s, str) and s in primaries_set for s in sources],
        dtype=bool,
    )
    is_negative = (labels_arr == 0) & is_eligible_negative_source
    keep = is_positive | is_negative

    n_pos = int(is_positive.sum())
    n_neg = int(is_negative.sum())
    # The substrings "no positive" and "no eligible negative" below
    # are load-bearing: ``LeakMetricsAggregator._per_source_call`` and
    # the IDK-metrics try/except in ``_add_idk_metrics`` both sniff
    # them via ``in str(exc)`` to decide whether to swallow the
    # failure into a NaN entry (the graceful-degrade path for empty
    # per-source buckets / empty IDK pool). If you reword either
    # message, update both call sites.
    if n_pos == 0:
        raise ValueError(
            f"{caller}: no positive (label==1) rows in eval; cannot compute "
            f"metric."
        )
    if n_neg == 0:
        raise ValueError(
            f"{caller}: no eligible negative rows (label==0 AND "
            f"source in {source_primaries!r}) in eval; "
            f"cannot compute metric."
        )

    return p_leak_arr[keep], labels_arr[keep].astype(int)


# ----------------------------------------------------------------------
# pAUC / FPR-at-recall metric family
# ----------------------------------------------------------------------


def partial_auroc_retain_aux(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
    fpr_cap: float = 0.10,
    *,
    source_primaries: tuple[str, ...] = tuple(sorted(_ELIGIBLE_NEGATIVE_PRIMARIES)),
) -> float:
    """Partial AUROC over the eligible-negative distribution.

    Eligible negatives default to the full retain/aux/IDK pool but can
    be restricted to a single source bucket via ``source_primaries`` —
    the source-balanced macro-average wrapper passes a single-element
    tuple to compute one bucket at a time. Row filtering is shared
    with the rest of the family via
    :func:`_filter_eval_rows_retain_aux`: positives are any
    ``label == 1`` row; default negatives are ``label == 0`` rows from
    ``retain``, ``aux``, or ``idk`` (the last covers both prose and
    paraphrase via tag-distinguished variants under the same primary).
    ``source=='forget'`` background rows are excluded.

    Returns ``sklearn.metrics.roc_auc_score(y, p, max_fpr=fpr_cap)`` —
    McClish-normalized so 0.5 is random and 1.0 is perfect, on the same
    scale as full AUROC.

    The asymmetry this metric encodes is "average TPR while staying
    under ``fpr_cap`` FPR" — it rewards classifiers that pull recall up
    given a low-FPR constraint. For the dual asymmetry ("minimize FPR
    while staying above some recall floor"), see
    :func:`partial_auroc_above_recall_retain_aux` and
    :func:`fpr_at_recall_retain_aux`.

    Parameters
    ----------
    p_leak
        Per-row predicted leak probability (e.g.,
        ``softmax(binary_logits)[:, 1]``).
    labels
        Per-row binary labels (0 / 1).
    sources
        Per-row dataset source string. Length must equal ``labels``.
    fpr_cap
        Upper FPR bound for the partial AUC integration (default 0.10).
        Must lie in ``(0, 1]``.
    source_primaries
        Keyword-only override for the eligible-negative primary source
        list. Defaults to :data:`_ELIGIBLE_NEGATIVE_PRIMARIES` (the
        full retain/aux/IDK pool). Pass a single-element tuple to
        compute the metric over one primary only — the source-balanced
        wrapper uses this to average per-source pAUCs without changing
        this function's pooled semantics for direct callers.

    Raises
    ------
    ValueError
        If ``len(sources) != len(labels)``, ``fpr_cap`` is outside
        ``(0, 1]``, or the filtered positive / eligible-negative subset
        is empty (failing fast on a misconfigured eval split is
        preferred over returning 0.0, which would silently elect the
        first checkpoint as "best" via tie-break).
    """
    if not (0.0 < fpr_cap <= 1.0):
        raise ValueError(
            f"partial_auroc_retain_aux: fpr_cap must lie in (0, 1], got {fpr_cap!r}."
        )

    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="partial_auroc_retain_aux",
        source_primaries=source_primaries,
    )

    return float(roc_auc_score(y_filt, p_filt, max_fpr=fpr_cap))


def partial_auroc_above_recall_retain_aux(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
    tpr_lower: float = 0.95,
    *,
    source_primaries: tuple[str, ...] = tuple(sorted(_ELIGIBLE_NEGATIVE_PRIMARIES)),
) -> float:
    """Partial AUROC over the high-recall strip TPR ∈ [tpr_lower, 1].

    The dual asymmetry to :func:`partial_auroc_retain_aux`: instead of
    "average TPR while staying under a low-FPR cap" (which optimizes
    recall given a FPR constraint), this metric is "average (1 − FPR)
    while staying above a high-recall floor" (which optimizes FPR given
    a recall constraint).

    Concretely, integrates ``1 − FPR(TPR)`` along the TPR axis over
    ``TPR ∈ [tpr_lower, 1]`` via the trapezoid rule on the
    :func:`sklearn.metrics.roc_curve` step function, with linear
    interpolation at the ``tpr_lower`` boundary. McClish-normalizes:

        raw_min  = 0.5 * (1 - tpr_lower)**2   # random classifier
        raw_max  = 1 - tpr_lower              # perfect classifier
        result   = 0.5 + 0.5 * (raw - raw_min) / (raw_max - raw_min)

    so 0.5 is random and 1.0 is perfect, matching the scale of
    :func:`partial_auroc_retain_aux`.

    Filters eval rows identically to :func:`partial_auroc_retain_aux`
    via the shared :func:`_filter_eval_rows_retain_aux` helper —
    positives are any ``label == 1`` row; negatives are ``label == 0``
    rows from primary sources ``retain``, ``aux``, or ``idk``.

    Caveats to be aware of when reading the number:

    - **Recall integration window is averaged, not pinned.** A
      checkpoint that hits TPR=1.0 at FPR=0.30 can score above a
      checkpoint that hits TPR=0.95 at FPR=0.05, since pAUC averages
      ``1 − FPR`` across the strip. If you want the single operating-
      point view, use :func:`fpr_at_recall_retain_aux`.
    - **Strip width is narrow.** At ``tpr_lower=0.95`` the McClish
      denominator is ~0.049, so tie-induced jumps in the ROC swing the
      normalized value by a few hundredths. Pair with
      :func:`fpr_at_recall_retain_aux` to detect noise-floor cases.

    Parameters
    ----------
    p_leak
        Per-row predicted leak probability.
    labels
        Per-row binary labels (0 / 1).
    sources
        Per-row dataset source string. Length must equal ``labels``.
    tpr_lower
        Lower TPR bound for the partial AUC integration (default 0.95).
        Must lie in ``[0, 1)``. At ``tpr_lower=0`` the metric reduces
        exactly to full AUROC on the filtered subset under the
        trapezoid rule: the identity ``∫ TPR dFPR = ∫ (1 - FPR) dTPR``
        holds for any monotone non-decreasing ROC from (0, 0) to (1, 1),
        and McClish normalization at ``tpr_lower=0`` simplifies to the
        raw AUC value.
    source_primaries
        Keyword-only eligible-negative primary list override. See
        :func:`partial_auroc_retain_aux` for the full semantics — used
        by the source-balanced macro-average wrapper.

    Raises
    ------
    ValueError
        If ``len(sources) != len(labels)``, ``tpr_lower`` is outside
        ``[0, 1)``, or the filtered positive / eligible-negative subset
        is empty.
    """
    if not (0.0 <= tpr_lower < 1.0):
        raise ValueError(
            f"partial_auroc_above_recall_retain_aux: tpr_lower must lie in "
            f"[0, 1), got {tpr_lower!r}."
        )

    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="partial_auroc_above_recall_retain_aux",
        source_primaries=source_primaries,
    )

    fpr, tpr, _ = roc_curve(y_filt, p_filt)
    # roc_curve guarantees a sorted-ascending threshold sweep with
    # tpr[0]=0, tpr[-1]=1, fpr[0]=0, fpr[-1]=1 (plus an additional
    # leading inf-threshold row that yields the (0, 0) point). Both tpr
    # and fpr are non-decreasing in this order.

    # Truncate to tpr >= tpr_lower with linear interpolation at the
    # boundary, mirroring sklearn's max_fpr handling in roc_auc_score.
    start = int(np.searchsorted(tpr, tpr_lower, side="left"))
    if start == 0 or tpr[start] == tpr_lower:
        # Boundary already exactly on a step (or tpr_lower=0): take
        # the suffix directly.
        tpr_trunc = tpr[start:]
        fpr_trunc = fpr[start:]
    else:
        # tpr[start-1] < tpr_lower < tpr[start]; interpolate fpr.
        fpr_at_lower = float(
            np.interp(
                tpr_lower,
                [tpr[start - 1], tpr[start]],
                [fpr[start - 1], fpr[start]],
            )
        )
        tpr_trunc = np.concatenate([[tpr_lower], tpr[start:]])
        fpr_trunc = np.concatenate([[fpr_at_lower], fpr[start:]])

    # Raw partial AUC = ∫ (1 - FPR) d(TPR) over the strip. Compute via
    # auc on the parallel (tpr, 1-fpr) arrays with tpr on the x-axis.
    # Constant-TPR segments (where threshold crosses a negative's
    # score) contribute zero because their x-width is zero, which is
    # the correct ROC-step-function interpretation.
    raw = float(auc(tpr_trunc, 1.0 - fpr_trunc))

    strip_width = 1.0 - tpr_lower
    raw_min = 0.5 * strip_width**2  # random classifier
    raw_max = strip_width  # perfect classifier
    return 0.5 + 0.5 * (raw - raw_min) / (raw_max - raw_min)


def fpr_at_recall_retain_aux(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
    recall_target: float = 0.95,
    *,
    source_primaries: tuple[str, ...] = tuple(sorted(_ELIGIBLE_NEGATIVE_PRIMARIES)),
) -> float:
    """FPR at the threshold that first achieves TPR ≥ recall_target.

    Single operating-point quantity (companion diagnostic to
    :func:`partial_auroc_above_recall_retain_aux`). Directly answers
    "what's our retain/aux/IDK FPR when this classifier is catching at
    least ``recall_target`` of leaks?" — interpretable in deployment
    units, no integration or McClish normalization.

    **Smaller is better.** A perfect classifier returns 0.0 (catches
    all leaks at zero FPR on the eligible-negative pool); an
    anti-perfect classifier returns 1.0 (must flag every eligible
    negative row before catching any leaks).

    Filters eval rows identically to :func:`partial_auroc_retain_aux`
    via the shared :func:`_filter_eval_rows_retain_aux` helper.
    Uses :func:`sklearn.metrics.roc_curve` and picks the smallest
    threshold whose TPR first crosses ``recall_target`` from below — no
    interpolation. This corresponds to a fixed deployable threshold
    (rather than a randomized mixture of two thresholds, which would be
    the interpolated answer); for checkpoint selection on a single
    deployable threshold this is the more useful convention.

    Parameters
    ----------
    p_leak
        Per-row predicted leak probability.
    labels
        Per-row binary labels (0 / 1).
    sources
        Per-row dataset source string. Length must equal ``labels``.
    recall_target
        Recall floor to enforce (default 0.95). Must lie in ``(0, 1]``.
    source_primaries
        Keyword-only eligible-negative primary list override. See
        :func:`partial_auroc_retain_aux` for the full semantics — used
        by the source-balanced macro-average wrapper.

    Raises
    ------
    ValueError
        If ``len(sources) != len(labels)``, ``recall_target`` is
        outside ``(0, 1]``, or the filtered subset is empty.
    """
    if not (0.0 < recall_target <= 1.0):
        raise ValueError(
            f"fpr_at_recall_retain_aux: recall_target must lie in (0, 1], "
            f"got {recall_target!r}."
        )

    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="fpr_at_recall_retain_aux",
        source_primaries=source_primaries,
    )

    fpr, tpr, _ = roc_curve(y_filt, p_filt)
    # tpr is non-decreasing, tpr[-1] == 1. searchsorted with side='left'
    # returns the first index where tpr >= recall_target.
    idx = int(np.searchsorted(tpr, recall_target, side="left"))
    # idx <= len(tpr)-1 always, since tpr[-1] = 1 >= recall_target for
    # any recall_target in (0, 1].
    return float(fpr[idx])


def gap_idk(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
) -> float:
    """Worst-positive ``P(leak)`` minus best-IDK-negative ``P(leak)``.

    Concretely:

        gap_idk = min(P_leak | label == 1)
                  - max(P_leak | label == 0 ∧ source == "idk")

    Positive values mean a threshold exists that separates every
    positive from every IDK negative: pick any threshold in
    ``(max_idk_neg, min_pos]`` and you get perfect recall with zero
    FP on the IDK pool. Negative values mean at least one IDK
    negative scores at least as high as the lowest-scoring positive,
    so no single-threshold rule achieves the separation.

    Unlike :func:`partial_auroc_retain_aux` and friends, this metric
    is IDK-pool-specific by definition rather than parametrized over
    ``source_primaries`` — IDK is the pool where post-regen "I don't
    know" continuations live, and a false positive on an IDK row
    triggers a *cascading* second regen. The min/max gap is a tight
    bound on whether a deployable threshold can avoid that cascade
    while still catching every true leak in the eval split.

    Positives are any ``label == 1`` row regardless of source (mirrors
    :func:`_filter_eval_rows_retain_aux`: every ``source=='forget'``
    confirmed-leak row counts, across all variant tags). IDK negatives
    are ``label == 0`` rows whose ``source == 'idk'`` — covers both
    prose (``tags=['clean']``) and paraphrase
    (``tags=['clean','paraphrase']``) under the same primary.

    Range: ``[-1, 1]`` when ``p_leak`` lies in ``[0, 1]`` (the
    canonical case — produced by softmax in the trainer-side
    extractors). Higher is better. Empty positive or empty IDK
    negative subsets raise ``ValueError`` so a misconfigured eval
    split fails loud (consistent with the rest of the metric
    family); :class:`LeakMetricsAggregator` catches the IDK-empty
    case and emits NaN for the wrapper key. Non-finite
    ``p_leak`` values raise upfront via the shared
    :func:`_filter_eval_rows_retain_aux` validation rather than
    silently propagating through ``min`` / ``max``.

    Robust sibling: :class:`LeakMetricsAggregator` also emits
    ``gap_idk_robust = quantile(P_pos, 0.05) - quantile(P_idk, 0.95)``
    (with the matching ``min_p_pos_q05`` / ``max_p_idk_q95``
    components). Same family, same row filter, same direction —
    different reduction. It substitutes the 5th / 95th percentile
    for strict min / max so a single anomalous positive or IDK row
    can't drive the metric to zero by itself. For samples below
    ~20 rows per bucket the linear-interpolation quantile lands
    close to the extremes, so the robustness benefit shrinks (but
    isn't harmful). Selection criterion guidance: a comfortably
    positive ``gap_idk_robust`` is more deployment-safe than a
    tight ``gap_idk`` that one outlier could flip negative.

    Parameters
    ----------
    p_leak
        Per-row predicted leak probability.
    labels
        Per-row binary labels (0 / 1).
    sources
        Per-row dataset source string. Length must equal ``labels``.

    Raises
    ------
    ValueError
        If ``len(sources) != len(labels)``, no positives are present,
        or no eligible IDK negatives are present.
    """
    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="gap_idk",
        source_primaries=("idk",),
    )
    # _filter_eval_rows_retain_aux already raised on empty positives or
    # empty IDK negatives, so both masks below are non-empty.
    pos = y_filt == 1
    neg = y_filt == 0
    return float(p_filt[pos].min() - p_filt[neg].max())


def min_p_pos(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
) -> float:
    """Worst-confidence positive: ``min(P_leak | label == 1)``.

    The left operand of :func:`gap_idk` exposed as a standalone
    metric, so eval logs can show whether a regression in
    ``gap_idk`` came from the positive side (this metric dropped)
    versus the IDK side (:func:`max_p_idk` rose).

    Range: ``[0, 1]`` when ``p_leak`` is in ``[0, 1]``. Higher is
    better — a higher worst-positive P(leak) means the classifier
    is more confident about the hardest positive in the eval split.

    Shares the row filter with :func:`gap_idk` and
    :func:`max_p_idk` so all three are computed on the same rows
    (no risk of a positive included in ``min_p_pos`` but excluded
    from ``gap_idk``). The filter requires both positives and
    IDK negatives to be present; in practice the wrapper that
    emits these keys catches the empty-IDK case and emits NaN.

    Parameters
    ----------
    p_leak, labels, sources
        Per-row arrays of equal length; see
        :func:`_filter_eval_rows_retain_aux` for the validation
        contract.

    Raises
    ------
    ValueError
        See :func:`_filter_eval_rows_retain_aux`.
    """
    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="min_p_pos",
        source_primaries=("idk",),
    )
    return float(p_filt[y_filt == 1].min())


def max_p_idk(
    p_leak: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    sources: list[str],
) -> float:
    """Most-confident IDK false positive: ``max(P_leak | label == 0 ∧
    source == "idk")``.

    The right operand of :func:`gap_idk` exposed as a standalone
    metric. **Lower is better** — a lower max P(leak) on the IDK
    pool means the classifier is less likely to fire on any
    "I don't know" continuation at inference, attacking the
    cascading-second-regen failure mode that an IDK false positive
    triggers.

    Range: ``[0, 1]`` when ``p_leak`` is in ``[0, 1]``.

    Shares the row filter with :func:`gap_idk` and
    :func:`min_p_pos` so the components partition cleanly into
    positive vs. IDK-negative subsets of the same filtered set.

    Parameters
    ----------
    p_leak, labels, sources
        Per-row arrays of equal length; see
        :func:`_filter_eval_rows_retain_aux` for the validation
        contract.

    Raises
    ------
    ValueError
        See :func:`_filter_eval_rows_retain_aux`.
    """
    p_filt, y_filt = _filter_eval_rows_retain_aux(
        p_leak, labels, sources,
        caller="max_p_idk",
        source_primaries=("idk",),
    )
    return float(p_filt[y_filt == 0].max())


# ----------------------------------------------------------------------
# (logits, labels) -> (p_leak, binary_labels) adapters
# ----------------------------------------------------------------------


def binary_p_leak_extractor(logits):
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]


def binary_label_extractor(labels):
    return np.asarray(labels)


# ----------------------------------------------------------------------
# compute_metrics callable
# ----------------------------------------------------------------------


def compute_binary_metrics(
    eval_pred, *, threshold, gamma=0.0, label_smoothing=0.0,
):
    """Metrics for the binary-only model (single (B, 2) logits tensor).

    Parameters
    ----------
    eval_pred
        ``(logits, labels)`` from HF Trainer. ``logits`` shape ``(N, 2)``;
        ``labels`` are binary 0/1 (the train script renames the
        dataset's ``label`` column to ``labels``).
    threshold
        P(leak) cutoff for the precision/recall/F-beta family. Was
        ``args.threshold``.
    gamma
        Positive focal exponent — when > 0, additionally report
        ``effective_positive_mass`` = mean of the focal weight
        ``((1-P(leak)).clamp(eps, 1-eps))**gamma`` over label=1 rows.
        Was ``args.positive_focal_gamma`` via defensive ``getattr``.
    label_smoothing
        ε for the focal gate's lower clip; consulted only when
        ``gamma > 0``. Was ``args.positive_focal_label_smoothing`` via
        defensive ``getattr``.

    Notes
    -----
    The ``effective_positive_mass`` diagnostic mirrors the in-graph
    computation in ``BinaryHeadModel.forward`` so the eval-time value
    is comparable to the train-time loss reweighting. A run where the
    metric drops below ~0.05 indicates the focal gate is starving the
    head of positive gradient and the early-stopping checkpoint may
    be contaminated by a stalled-positive plateau.
    """
    logits, labels_a = eval_pred
    labels_a = np.asarray(labels_a)
    binary_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    binary_preds = np.where(binary_probs[:, 1] > threshold, 1, 0)

    tp = int(((binary_preds == 1) & (labels_a == 1)).sum())
    fp = int(((binary_preds == 1) & (labels_a == 0)).sum())
    fn = int(((binary_preds == 0) & (labels_a == 1)).sum())
    bp = tp / (tp + fp) if (tp + fp) else 0
    br = tp / (tp + fn) if (tp + fn) else 0

    def fbeta(p, r, beta):
        b2 = beta * beta
        d = b2 * p + r
        return (1 + b2) * p * r / d if d else 0

    metrics = {
        "binary_precision": bp,
        "binary_recall": br,
        "binary_f1": fbeta(bp, br, 1.0),
        "binary_f2": fbeta(bp, br, 2.0),
    }

    # Effective positive mass — only logged when the focal gate is
    # active. Computed in numpy from the already-softmaxed binary_probs.
    if gamma > 0.0:
        pos_mask = labels_a == 1
        if pos_mask.any():
            p_leak_pos = binary_probs[pos_mask, 1]
            one_minus_p = 1.0 - p_leak_pos
            if label_smoothing > 0.0:
                one_minus_p = np.clip(
                    one_minus_p, label_smoothing, 1.0 - label_smoothing,
                )
            else:
                one_minus_p = np.clip(one_minus_p, 1e-6, None)
            focal_w = one_minus_p ** gamma
            metrics["effective_positive_mass"] = float(focal_w.mean())
        else:
            metrics["effective_positive_mass"] = float("nan")

    return metrics


# ----------------------------------------------------------------------
# pAUC Trainer-eval wrapper
# ----------------------------------------------------------------------


class LeakMetricsAggregator:
    """HF Trainer-compatible ``compute_metrics`` callable.

    Composes the base metrics (precision/recall/F-beta) with
    cross-distribution metrics that need both positives and eligible
    negatives in the same call — pAUC on the union retain∪aux∪idk
    pool, the IDK-only trio (``gap_idk``, ``min_p_pos``, ``max_p_idk``),
    the IDK focal-mass diagnostic, per-source bucket components, and
    optional source-balanced macros plus rolling-window smoothed
    companions. Cross-distribution metrics are computed only on the
    merged ``test`` split (dispatched via ``self.trainer._current_metric_prefix``);
    derived ``forget`` / ``retain`` splits fall through to the base
    metrics unchanged.

    Instance state owned across calls:
    - ``_warned_empty_buckets``: one-shot log dedup so empty per-source
      buckets don't spam the log on every eval.
    - ``_metric_history``: bounded ``deque`` powering the smoothed
      companion keys; only the ``eval_test`` branch appends, so
      per-split pollution is prevented by construction.
    - ``trainer``: forward reference to the
      :class:`SplitAwareTrainer`. The Trainer takes ``compute_metrics``
      as a constructor argument, so the aggregator can't reference the
      trainer at ``__init__`` time. Set ``aggregator.trainer = trainer``
      after both objects exist.

    Resume semantics: instance state lives in memory and is lost on
    ``resume_from_checkpoint`` — HF Trainer rebuilds the
    ``compute_metrics`` object from scratch. The first post-resume
    ``*_smoothed`` value is therefore a single-point estimate
    (= unsmoothed) and the smoothing window re-fills over the next
    ``smoothing_window - 1`` evals.

    Emitted keys (eval_test split only):

    1. **Pooled headlines** — ``pauc_retain_aux`` (TPR over
       FPR ∈ [0, ``pauc_fpr_cap``]), ``pauc_above_recall_retain_aux``
       (1 − FPR over TPR ∈ [``pauc_recall_lower``, 1]),
       ``fpr_at_recall_retain_aux`` (FPR at the threshold first
       achieving TPR ≥ ``fpr_at_recall_target``), and the IDK-pool
       trio ``min_p_pos`` / ``max_p_idk`` / ``gap_idk``
       (= ``min_p_pos - max_p_idk``; positive values mean a deployable
       threshold separates every positive from every IDK negative).
       The first three are computed on the union of all eligible
       negatives; the IDK trio is IDK-only by definition and shares
       a single row filter so ``gap_idk`` equals the difference of
       its components bit-for-bit. The quantile-based robust
       sibling trio ``min_p_pos_q05`` / ``max_p_idk_q95`` /
       ``gap_idk_robust`` (= ``min_p_pos_q05 - max_p_idk_q95``)
       is emitted alongside; it uses the 5th / 95th percentile so
       a single outlier on either side can't drive the gap to zero.
       Higher ``gap_idk_robust`` is better, same direction as
       ``gap_idk``.

    2. **Per-source components** — the three pAUC / FPR-at-recall
       metrics restricted to one source bucket at a time with keys
       ``<metric>_retain`` / ``<metric>_aux`` / ``<metric>_idk``.
       Surfaces source-imbalance behavior. Empty bucket → NaN entry,
       does not abort the eval. ``gap_idk`` has no per-source variants —
       its definition is already IDK-specific.

    3. **Source-balanced headlines** (when ``pauc_source_balance``) —
       arithmetic ``np.nanmean`` across the three per-source buckets
       under keys ``<metric>_balanced``. Each source contributes 1/3
       regardless of row count.

    4. **IDK focal mass** (when ``idk_focal_gamma > 0``) —
       ``effective_idk_mass`` = mean of
       ``P(leak).clamp(eps, 1-eps)^gamma`` over IDK-negative rows.
       Mirrors the in-graph IDK reweight in
       :class:`BinaryHeadModel.forward`. Diagnostic only, NaN on empty
       IDK pool.

    5. **Smoothed companions** (when ``smoothing_window >= 2``) —
       parallel ``<key>_smoothed`` for each numeric key, computed as
       ``np.nanmean`` over the last ``smoothing_window`` eval_test
       passes (including the current one). NaN history entries are
       skipped via ``nanmean``. The window unit is eval-calls, not
       training-steps; changing ``--eval_steps`` mid-experiment
       changes the effective step-window.

    Parameters
    ----------
    base_compute_metrics
        Metric function called as ``base_compute_metrics(eval_pred)``.
        Any configuration (threshold, gamma, ...) must be
        partial-applied at the call site before passing in.
    p_leak_extractor, label_extractor : callable
        Numpy adapters that convert the model's logit / label shape
        into uniform ``p_leak`` and ``binary_labels (0/1)`` arrays.
    test_sources
        Per-row source string array captured before tokenization, in
        the same row order as the eval predictions. ``None`` disables
        the cross-distribution metrics and the aggregator returns
        base metrics unchanged.
    pauc_fpr_cap, pauc_recall_lower, fpr_at_recall_target
        Operating-point parameters for the three pAUC / FPR-at-recall
        headlines. See the CLI help on ``--pauc_fpr_cap``,
        ``--pauc_recall_lower``, ``--fpr_at_recall_target`` for
        semantics.
    pauc_source_balance
        Bool — when True, additionally emit ``<metric>_balanced`` keys
        with the macro-average across source buckets.
    idk_focal_gamma, idk_focal_label_smoothing
        When ``idk_focal_gamma > 0``, additionally emit
        ``effective_idk_mass``. Diagnostic only — not a selection
        target; ``train_classifier.py`` rejects
        ``--metric_for_best_model=test_effective_idk_mass`` explicitly.
    smoothing_window
        Rolling-window size for the smoothed companion metrics; ``1``
        (default) disables. See class docstring for resume semantics
        and the per-eval-call unit.
    """

    def __init__(
        self,
        *,
        base_compute_metrics,
        p_leak_extractor,
        label_extractor,
        test_sources,
        pauc_fpr_cap,
        pauc_recall_lower,
        fpr_at_recall_target,
        pauc_source_balance,
        idk_focal_gamma: float = 0.0,
        idk_focal_label_smoothing: float = 0.0,
        smoothing_window: int = 1,
    ):
        if smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be >= 1; got {smoothing_window}. "
                f"Default 1 disables smoothing (no ``*_smoothed`` keys "
                f"emitted)."
            )

        self._base_compute_metrics = base_compute_metrics
        self._p_leak_extractor = p_leak_extractor
        self._label_extractor = label_extractor
        self._test_sources = test_sources
        self._pauc_fpr_cap = pauc_fpr_cap
        self._pauc_recall_lower = pauc_recall_lower
        self._fpr_at_recall_target = fpr_at_recall_target
        self._pauc_source_balance = pauc_source_balance
        self._idk_focal_gamma = idk_focal_gamma
        self._idk_focal_label_smoothing = idk_focal_label_smoothing
        self._smoothing_window = smoothing_window

        # Forward reference to the SplitAwareTrainer. Assigned by the
        # caller after both the aggregator and the Trainer exist:
        # ``aggregator.trainer = trainer``. Until set, the
        # cross-distribution metrics short-circuit (the trainer is
        # needed to read ``_current_metric_prefix`` for split
        # dispatch). Typed as ``Any`` so the post-construction
        # rebinding doesn't trip narrow-type inference.
        self.trainer: Any = None

        # Per-instance mutable state. Each aggregator owns its own
        # set / deque — no class-level mutable defaults (which would
        # silently share state across instances).
        self._warned_empty_buckets: set[tuple[str, str]] = set()
        self._metric_history: deque[dict[str, float]] = deque(
            maxlen=smoothing_window,
        )

    def __call__(self, eval_pred):
        base = self._base_compute_metrics(eval_pred)
        # Dispatch: cross-distribution metrics are eval_test-only.
        if self._test_sources is None:
            return base
        if (
            self.trainer is None
            or self.trainer._current_metric_prefix != "eval_test"
        ):
            return base
        p_leak, binary_labels = self._extract_and_validate(eval_pred)
        self._add_pooled_pauc_family(base, p_leak, binary_labels)
        self._add_idk_metrics(base, p_leak, binary_labels)
        per_source = self._add_per_source_components(
            base, p_leak, binary_labels,
        )
        if self._pauc_source_balance:
            for metric_name, components in per_source.items():
                base[f"{metric_name}_balanced"] = float(
                    np.nanmean(list(components.values()))
                )
        self._add_smoothed_companions(base)
        return base

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_and_validate(self, eval_pred):
        """Apply the extractors and verify positional
        alignment with ``self._test_sources``.

        Length-mismatch is fail-loud: ``compute_metrics`` receives
        predictions in dataset row order (no shuffle on eval), and
        ``test_sources`` was captured from the same dataset before
        tokenization. If they disagree the metrics would be silently
        mis-aligned, so we abort the eval.
        """
        logits, labels_a = eval_pred
        p_leak = self._p_leak_extractor(logits)
        binary_labels = self._label_extractor(labels_a)
        if len(binary_labels) != len(self._test_sources):
            raise ValueError(
                f"LeakMetricsAggregator: test split eval predictions "
                f"have length {len(binary_labels)} but captured "
                f"test_sources has length {len(self._test_sources)}. "
                f"Did dataset['test'] get shuffled or filtered after "
                f"the source-column capture? Aborting to prevent "
                f"silently mis-aligned pAUC."
            )
        return p_leak, binary_labels

    def _per_source_call(
        self, metric_fn, p, y, src, threshold_arg, primaries,
        bucket_name, metric_name,
    ):
        """Call ``metric_fn`` restricted to one source bucket. Returns
        NaN on empty bucket (with one-shot log dedup via
        ``self._warned_empty_buckets``) rather than aborting the eval.
        """
        try:
            return metric_fn(
                p, y, src, threshold_arg, source_primaries=primaries,
            )
        except ValueError as exc:
            # Only swallow the "no eligible negative" / "no positive"
            # empty-class case; let other ValueErrors (length mismatch
            # etc.) bubble up.
            msg = str(exc)
            if "no eligible negative" in msg or "no positive" in msg:
                key = (metric_name, bucket_name)
                if key not in self._warned_empty_buckets:
                    self._warned_empty_buckets.add(key)
                    log_print(
                        f"LeakMetricsAggregator: per-source bucket "
                        f"{bucket_name!r} for metric {metric_name!r} "
                        f"is empty in this eval split — emitting NaN. "
                        f"Source-balanced macro-average will skip "
                        f"this bucket via np.nanmean."
                    )
                return float("nan")
            raise

    def _add_pooled_pauc_family(self, base, p_leak, binary_labels):
        """Pooled pAUC / FPR-at-recall headlines on the union retain∪aux∪idk
        negative pool. Always emitted; fail-loud on misconfigured
        eval split (no swallowing the empty-pool ValueError here)."""
        sources = self._test_sources
        base["pauc_retain_aux"] = partial_auroc_retain_aux(
            p_leak, binary_labels, sources, self._pauc_fpr_cap,
        )
        base["pauc_above_recall_retain_aux"] = (
            partial_auroc_above_recall_retain_aux(
                p_leak, binary_labels, sources, self._pauc_recall_lower,
            )
        )
        base["fpr_at_recall_retain_aux"] = fpr_at_recall_retain_aux(
            p_leak, binary_labels, sources, self._fpr_at_recall_target,
        )

    def _add_idk_metrics(self, base, p_leak, binary_labels):
        """IDK-only metrics computed in one block: the gap_idk trio
        (``min_p_pos`` / ``max_p_idk`` / ``gap_idk``), its robust
        quantile-based sibling trio (``min_p_pos_q05`` /
        ``max_p_idk_q95`` / ``gap_idk_robust``) via a shared
        row filter, plus the focal-mass diagnostic when the IDK
        focal gate is active. Both filter on ``source == 'idk'``
        rows and emit NaN on empty IDK pool.

        Surfacing the trio's components alongside the gap lets the
        eval log show whether a regression moved on the positive
        side, the IDK side, or both. The gap = min - max identity
        holds bit-for-bit because they share the filter call.

        The robust trio uses the 5th percentile of positives and the
        95th percentile of IDK negatives instead of strict min/max,
        so a single anomalous positive or IDK row can't drive
        ``gap_idk`` to zero by itself. Selection on
        ``gap_idk_robust`` is more deployment-safe than on
        ``gap_idk``.
        """
        sources = self._test_sources

        # gap_idk trio (strict + robust). Computed against the same
        # filtered row set so the robust variant is comparable to the
        # strict one on a bit-for-bit identical eval subset; the
        # filter already raises on non-finite p_leak, so np.quantile
        # (rather than np.nanquantile) is safe here.
        try:
            p_filt, y_filt = _filter_eval_rows_retain_aux(
                p_leak, binary_labels, sources,
                caller="gap_idk",
                source_primaries=("idk",),
            )
            p_pos = p_filt[y_filt == 1]
            p_idk = p_filt[y_filt == 0]
            min_pos = float(p_pos.min())
            max_idk = float(p_idk.max())
            base["min_p_pos"] = min_pos
            base["max_p_idk"] = max_idk
            base["gap_idk"] = min_pos - max_idk
            # Robust quantile-based variant. method="linear" is numpy's
            # default — on small samples it interpolates near the
            # extremes, so the robustness benefit is small (but never
            # harmful) below ~20 rows per bucket.
            min_pos_q05 = float(np.quantile(p_pos, 0.05, method="linear"))
            max_idk_q95 = float(np.quantile(p_idk, 0.95, method="linear"))
            base["min_p_pos_q05"] = min_pos_q05
            base["max_p_idk_q95"] = max_idk_q95
            base["gap_idk_robust"] = min_pos_q05 - max_idk_q95
        except ValueError as exc:
            if "no eligible negative" in str(exc):
                base["min_p_pos"] = float("nan")
                base["max_p_idk"] = float("nan")
                base["gap_idk"] = float("nan")
                base["min_p_pos_q05"] = float("nan")
                base["max_p_idk_q95"] = float("nan")
                base["gap_idk_robust"] = float("nan")
            else:
                raise

        # effective_idk_mass — only when the IDK focal gate is active
        # in the model. Same formula as the in-graph weight in
        # ``BinaryHeadModel.forward`` but aggregated at the eval-split
        # level. A value below ~0.05 indicates the focal gate is
        # starving the IDK pool. Wrapped in a positive guard (rather
        # than an early return) so any future post-focal-mass
        # diagnostic added to this helper still executes when the
        # gate is off.
        if self._idk_focal_gamma > 0.0:
            is_idk_neg = np.array(
                [isinstance(s, str) and s == "idk" for s in sources],
                dtype=bool,
            ) & (np.asarray(binary_labels) == 0)
            if not is_idk_neg.any():
                # Same graceful-degrade as the per-source bucket NaN.
                base["effective_idk_mass"] = float("nan")
            else:
                p_idk = np.asarray(p_leak, dtype=np.float64)[is_idk_neg]
                eps = self._idk_focal_label_smoothing
                if eps > 0.0:
                    p_idk = np.clip(p_idk, eps, 1.0 - eps)
                else:
                    # One-sided floor matching the in-graph
                    # ``clamp(min=1e-6)``. Softmax outputs never reach
                    # 1.0 so the upper side is unconstrained.
                    p_idk = np.clip(p_idk, 1e-6, None)
                base["effective_idk_mass"] = float(
                    (p_idk ** self._idk_focal_gamma).mean()
                )

    def _add_per_source_components(self, base, p_leak, binary_labels):
        """Per-source pAUC / FPR-at-recall components, one bucket
        per source. Returns the nested ``{metric: {bucket: value}}``
        dict so ``__call__`` can derive the source-balanced macros
        without re-stashing on ``self``.

        Returns
        -------
        dict[str, dict[str, float]]
            Keys: ``pauc_retain_aux`` / ``pauc_above_recall_retain_aux`` /
            ``fpr_at_recall_retain_aux``. Inner keys: ``retain`` /
            ``aux`` / ``idk``. Empty-bucket entries are NaN.
        """
        sources = self._test_sources
        per_source: dict[str, dict[str, float]] = {
            "pauc_retain_aux": {},
            "pauc_above_recall_retain_aux": {},
            "fpr_at_recall_retain_aux": {},
        }
        for bucket_name, primary in PER_SOURCE_BUCKETS:
            primaries = (primary,)
            per_source["pauc_retain_aux"][bucket_name] = self._per_source_call(
                partial_auroc_retain_aux,
                p_leak, binary_labels, sources, self._pauc_fpr_cap,
                primaries, bucket_name, "pauc_retain_aux",
            )
            per_source["pauc_above_recall_retain_aux"][bucket_name] = (
                self._per_source_call(
                    partial_auroc_above_recall_retain_aux,
                    p_leak, binary_labels, sources,
                    self._pauc_recall_lower,
                    primaries, bucket_name, "pauc_above_recall_retain_aux",
                )
            )
            per_source["fpr_at_recall_retain_aux"][bucket_name] = (
                self._per_source_call(
                    fpr_at_recall_retain_aux,
                    p_leak, binary_labels, sources,
                    self._fpr_at_recall_target,
                    primaries, bucket_name, "fpr_at_recall_retain_aux",
                )
            )
        # Flatten into base under <metric>_<bucket> keys.
        for metric_name, components in per_source.items():
            for bucket_name, value in components.items():
                base[f"{metric_name}_{bucket_name}"] = value
        return per_source

    def _add_smoothed_companions(self, base):
        """Rolling-window smoothed companions for each numeric key.

        Early-returns when ``smoothing_window < 2`` so the no-op
        default path is structurally distinct from the active path:
        no snapshot, no append, no key emission.

        Snapshot filter excludes ``bool`` explicitly because Python's
        ``isinstance(True, int) == True`` would otherwise let a
        hypothetical bool flag slip through and be silently averaged
        as 0/1 under ``nanmean`` (algebraically valid, semantically
        surprising).

        All-NaN window → NaN with an explicit ``isfinite`` guard to
        avoid the "Mean of empty slice" RuntimeWarning from
        ``np.nanmean``. Note that ``np.isfinite(+inf) == False``, so a
        ``+inf`` history entry (none of our metrics produce one today)
        would get silently converted to NaN here rather than passed
        through as ``+inf`` via nanmean. Deliberate simplification —
        no current metric is unbounded.
        """
        if self._smoothing_window < 2:
            return
        current_snapshot = {
            k: float(v) for k, v in base.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        self._metric_history.append(current_snapshot)
        # Union of keys across the history window: a key that appeared
        # in a past eval but not in this one (e.g. ``effective_idk_mass``
        # after toggling the focal gate off mid-run) still gets a
        # smoothed value derived from the prior evals where it was
        # present.
        all_keys: set[str] = set()
        for snap in self._metric_history:
            all_keys.update(snap.keys())
        for key in all_keys:
            values = [
                snap[key] for snap in self._metric_history if key in snap
            ]
            arr = np.asarray(values, dtype=np.float64)
            if not np.any(np.isfinite(arr)):
                base[f"{key}_smoothed"] = float("nan")
            else:
                base[f"{key}_smoothed"] = float(np.nanmean(arr))
