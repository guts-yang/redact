"""Tests for :class:`redact.data_samplers.StratifiedBatchSampler`.

Validates the composition / coverage / determinism / edge-case contracts
documented on the sampler. The downstream pairwise-hinge loss depends
on every batch carrying at least ``n_pos`` positives AND ``n_idk`` IDK
negatives — these tests are the regression suite for that contract.
"""
from __future__ import annotations

import pytest

from redact.data_samplers import StratifiedBatchSampler


def _make_inputs(n_pos: int, n_idk: int, n_other: int):
    """Build (sources, labels) for a synthetic training set.

    Layout: ``n_pos`` positives first (label=1, source='forget'),
    then ``n_idk`` IDK negatives (label=0, source='idk'), then
    ``n_other`` other negatives (label=0, source='retain'). Uses the
    post-refactor primary source values (the sampler bucketization
    keys on ``src == 'idk'``).
    """
    sources: list[str] = []
    labels: list[int] = []
    for _ in range(n_pos):
        sources.append("forget")
        labels.append(1)
    for _ in range(n_idk):
        sources.append("idk")
        labels.append(0)
    for _ in range(n_other):
        sources.append("retain")
        labels.append(0)
    return sources, labels


def _index_buckets(n_pos: int, n_idk: int, n_other: int):
    """Return (pos_set, idk_set, other_set) for the layout in ``_make_inputs``."""
    return (
        set(range(n_pos)),
        set(range(n_pos, n_pos + n_idk)),
        set(range(n_pos + n_idk, n_pos + n_idk + n_other)),
    )


# ----------------------------------------------------------------------
# 1. Composition: every batch has exactly n_pos / n_idk / n_other.
# ----------------------------------------------------------------------
def test_composition_every_batch_matches_requested_counts():
    n_pos, n_idk, n_other = 10, 10, 100
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    pos_set, idk_set, other_set = _index_buckets(n_pos, n_idk, n_other)
    batch_size = 12
    n_pos_per_batch = 2
    n_idk_per_batch = 2
    n_other_per_batch = batch_size - n_pos_per_batch - n_idk_per_batch

    sampler = StratifiedBatchSampler(
        sources=sources,
        labels=labels,
        batch_size=batch_size,
        n_pos=n_pos_per_batch,
        n_idk=n_idk_per_batch,
        seed=0,
    )
    batches = list(sampler)
    # ceil(100 / 8) = 13 batches under drop_last=False default.
    assert len(batches) == 13
    assert len(sampler) == 13

    seen_other = []
    for i, b in enumerate(batches):
        is_last = i == len(batches) - 1
        # n_pos and n_idk always satisfy the contract on every batch.
        n_p = sum(1 for j in b if j in pos_set)
        n_i = sum(1 for j in b if j in idk_set)
        n_o = sum(1 for j in b if j in other_set)
        assert n_p == n_pos_per_batch, (
            f"batch {i} expected {n_pos_per_batch} positives, got {n_p}"
        )
        assert n_i == n_idk_per_batch, (
            f"batch {i} expected {n_idk_per_batch} IDKs, got {n_i}"
        )
        if is_last:
            # Last batch is partial when n_other doesn't divide len(other_idx).
            # 100 % 8 = 4, so the last batch has 4 other slots.
            assert n_o == 100 - n_other_per_batch * (len(batches) - 1)
        else:
            assert n_o == n_other_per_batch, (
                f"batch {i} expected {n_other_per_batch} others, got {n_o}"
            )
        seen_other.extend(j for j in b if j in other_set)
    # No 'other' duplicates within one epoch.
    assert len(seen_other) == len(set(seen_other))


# ----------------------------------------------------------------------
# 2. Coverage: every 'other' index appears exactly once per epoch.
# ----------------------------------------------------------------------
def test_other_coverage_exactly_once_per_epoch():
    n_pos, n_idk, n_other = 8, 8, 100
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    _, _, other_set = _index_buckets(n_pos, n_idk, n_other)

    sampler = StratifiedBatchSampler(
        sources=sources,
        labels=labels,
        batch_size=12,
        n_pos=2,
        n_idk=2,
        seed=42,
    )
    batches = list(sampler)
    flattened = [j for b in batches for j in b]
    seen_other = [j for j in flattened if j in other_set]
    # Every 'other' index appears exactly once (no replacement on the spine).
    assert sorted(seen_other) == sorted(other_set), (
        f"Expected every other index exactly once; got "
        f"{len(seen_other)} draws, {len(set(seen_other))} unique."
    )


# ----------------------------------------------------------------------
# 3a. Determinism: same seed → same sequence.
# ----------------------------------------------------------------------
def test_determinism_same_seed_yields_same_sequence():
    sources, labels = _make_inputs(10, 10, 100)
    s1 = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=7,
    )
    s2 = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=7,
    )
    b1 = list(s1)
    b2 = list(s2)
    assert b1 == b2


# ----------------------------------------------------------------------
# 3b. Determinism: different seeds → different sequences.
# ----------------------------------------------------------------------
def test_different_seeds_yield_different_sequences():
    sources, labels = _make_inputs(10, 10, 100)
    s1 = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=7,
    )
    s2 = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=8,
    )
    b1 = list(s1)
    b2 = list(s2)
    assert b1 != b2, "Two different seeds happened to produce the same sequence."


# ----------------------------------------------------------------------
# 3c. Epoch progression: successive iter() calls on the same sampler
#    yield different orderings (seed advances internally).
# ----------------------------------------------------------------------
def test_successive_iter_calls_differ():
    sources, labels = _make_inputs(10, 10, 100)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=0,
    )
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    # Both epochs should still cover the 'other' spine fully, but in
    # a different order.
    _, _, other_set = _index_buckets(10, 10, 100)
    e0_other = [j for b in epoch0 for j in b if j in other_set]
    e1_other = [j for b in epoch1 for j in b if j in other_set]
    assert sorted(e0_other) == sorted(other_set)
    assert sorted(e1_other) == sorted(other_set)
    assert epoch0 != epoch1, (
        "Successive iter() calls produced identical batches; the "
        "internal epoch counter is not advancing."
    )


def test_incomplete_iteration_still_bumps_epoch():
    """Epoch counter must advance even when the consumer doesn't drain ``__iter__``.

    HF Trainer can early-stop on ``max_steps``, eval-driven patience,
    or a raised exception mid-epoch. If the counter only advanced
    after the ``while`` loop ran to completion, the next epoch
    would silently replay the same shuffle. Regression test for
    that bug: pull a single batch (incomplete consumption), then
    iterate a fresh epoch fully and assert it differs from a
    drained reference epoch starting at the same internal state.
    """
    sources, labels = _make_inputs(10, 10, 100)

    # Reference: full epoch 0, then full epoch 1, captured from a
    # fresh sampler.
    ref = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=0,
    )
    ref_epoch0 = list(ref)
    ref_epoch1 = list(ref)
    assert ref_epoch0 != ref_epoch1  # sanity — already covered above

    # Stressed sampler: pull only the first batch of epoch 0 (early
    # exit, never drains the generator). The epoch counter must
    # still advance so the next __iter__ call produces epoch 1.
    stressed = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=0,
    )
    first_batch_of_epoch_0 = next(iter(stressed))
    # The first batch of the un-drained epoch should match the first
    # batch of the reference epoch 0 (same seed, same data, same
    # internal state at iter() time).
    assert first_batch_of_epoch_0 == ref_epoch0[0]

    # Now iterate fully — this should be epoch 1, not a replay of epoch 0.
    next_full = list(stressed)
    assert next_full == ref_epoch1, (
        "Incomplete prior iteration left the epoch counter "
        "un-advanced; the second iter() call replayed epoch 0 "
        "instead of producing epoch 1."
    )


def test_exception_mid_iteration_still_bumps_epoch():
    """If the consumer raises inside the for-loop body, the next epoch must still differ.

    Equivalent in effect to ``test_incomplete_iteration_still_bumps_epoch``
    but exercises the exception-path: an HF training loop that
    raises (e.g. CUDA OOM) must not cause the resume to replay the
    same shuffle.
    """
    sources, labels = _make_inputs(10, 10, 100)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=0,
    )
    # Capture epoch 0's first batch via a raising consumer.
    first_batch: list[int] | None = None
    try:
        for batch in sampler:
            first_batch = batch
            raise RuntimeError("simulated early-exit")
    except RuntimeError:
        pass
    assert first_batch is not None
    # The epoch counter must have advanced. Iterate fresh and compare
    # to a reference epoch 1.
    ref = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2, seed=0,
    )
    list(ref)  # advance reference past epoch 0
    ref_epoch1 = list(ref)
    next_full = list(sampler)
    assert next_full == ref_epoch1, (
        "After a mid-iteration exception, the next __iter__ call "
        "replayed the same epoch instead of advancing."
    )


# ----------------------------------------------------------------------
# 4. Edge cases: validation errors at construction.
# ----------------------------------------------------------------------
def test_n_pos_plus_n_idk_at_least_batch_size_raises():
    sources, labels = _make_inputs(4, 4, 20)
    # n_pos + n_idk == batch_size leaves no room for others.
    with pytest.raises(ValueError, match="strictly less than"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=8, n_pos=4, n_idk=4,
        )
    # n_pos + n_idk > batch_size is even worse.
    with pytest.raises(ValueError, match="strictly less than"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=8, n_pos=5, n_idk=4,
        )


def test_empty_positives_pool_raises():
    sources, labels = _make_inputs(0, 4, 20)
    with pytest.raises(ValueError, match="positives pool is empty"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        )


def test_empty_idk_pool_raises():
    sources, labels = _make_inputs(4, 0, 20)
    with pytest.raises(ValueError, match="IDK negatives pool is empty"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        )


def test_empty_other_pool_raises():
    sources, labels = _make_inputs(4, 4, 0)
    with pytest.raises(ValueError, match="'other' negatives pool is"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        )


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="must equal"):
        StratifiedBatchSampler(
            sources=["a", "b", "c"], labels=[0, 1], batch_size=4,
            n_pos=1, n_idk=1,
        )


def test_batch_size_nonpositive_raises():
    sources, labels = _make_inputs(2, 2, 4)
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=0, n_pos=0, n_idk=0,
        )


def test_negative_n_counts_raise():
    sources, labels = _make_inputs(2, 2, 4)
    with pytest.raises(ValueError, match=">= 0"):
        StratifiedBatchSampler(
            sources=sources, labels=labels, batch_size=4, n_pos=-1, n_idk=1,
        )


# ----------------------------------------------------------------------
# 4d. Small positives pool: len(pos_idx) < n_pos still works (cycle
#     wraps with-replacement within a single take).
# ----------------------------------------------------------------------
def test_small_pos_pool_smaller_than_n_pos_still_works():
    # 1 positive, n_pos=2 per batch — the cycle buffer wraps and the
    # same positive index appears twice in the same batch's positives
    # slice. This is the documented degenerate fallback.
    n_pos, n_idk, n_other = 1, 4, 20
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=6, n_pos=2, n_idk=1, seed=0,
    )
    batches = list(sampler)
    pos_set, _, _ = _index_buckets(n_pos, n_idk, n_other)
    for b in batches:
        pos_in_batch = [j for j in b if j in pos_set]
        assert len(pos_in_batch) == 2
        # Only one unique positive in the data, so both slots must be that index.
        assert set(pos_in_batch) == pos_set


# ----------------------------------------------------------------------
# 5. Cycle behavior: with len(pos_idx)=4 and n_pos=2 over 4 batches,
#    each positive appears exactly twice. Validates "no repeat until
#    cycle exhausted".
# ----------------------------------------------------------------------
def test_pos_cycle_no_repeat_within_cycle():
    # 4 positives, 4 IDKs, 16 other. n_pos=2 per batch, n_other=2 per
    # batch → 8 batches per epoch, 8 * 2 = 16 positive draws total → 4
    # complete cycles. Each positive appears exactly 4 times across
    # the epoch (not 0 or 16).
    n_pos, n_idk, n_other = 4, 4, 16
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=6, n_pos=2, n_idk=2, seed=0,
    )
    batches = list(sampler)
    pos_set, _, _ = _index_buckets(n_pos, n_idk, n_other)
    # 8 batches × 2 positives = 16 positive draws.
    pos_draws = [j for b in batches for j in b if j in pos_set]
    assert len(pos_draws) == 16
    # Each of the 4 positives should appear 4 times.
    counts = {p: pos_draws.count(p) for p in pos_set}
    assert all(c == 4 for c in counts.values()), (
        f"Expected each positive to appear 4 times; got {counts}"
    )


def test_pos_cycle_4_batches_each_pos_twice():
    """The exact contract from the design doc.

    With len(pos_idx) = 4 and n_pos = 2, over 4 batches each positive
    should appear exactly twice (not 0 or 4). Validates the
    no-repeat-until-cycle-exhausted property — across 4 batches
    (= 8 positive draws = 2 full cycles), every positive is seen
    exactly twice.
    """
    n_pos, n_idk, n_other = 4, 4, 8  # ⇒ 4 batches at n_other_per_batch=2
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=6, n_pos=2, n_idk=2, seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 4
    pos_set, _, _ = _index_buckets(n_pos, n_idk, n_other)
    pos_draws = [j for b in batches for j in b if j in pos_set]
    assert len(pos_draws) == 8
    counts = {p: pos_draws.count(p) for p in pos_set}
    assert all(c == 2 for c in counts.values()), (
        f"Expected each positive to appear exactly 2x over 4 batches "
        f"(2 complete cycles of 4); got {counts}"
    )


def test_pos_cycle_first_cycle_window_has_no_repeats():
    """Strict no-repeat-within-a-cycle check.

    With ``len(pos_idx)=4`` and ``n_pos=2``, the first 4 positive
    draws (= first 2 batches = first cycle) must each be a distinct
    positive. A buggy implementation that reshuffles too eagerly or
    reuses indices within a cycle could still produce the right
    per-positive totals over 4 batches (8 draws) by coincidence —
    this test catches that by inspecting a single cycle window.
    """
    n_pos, n_idk, n_other = 4, 4, 8
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    pos_set, _, _ = _index_buckets(n_pos, n_idk, n_other)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=6, n_pos=2, n_idk=2, seed=0,
    )
    batches = list(sampler)
    pos_draws_per_batch = [
        [j for j in b if j in pos_set] for b in batches
    ]
    first_cycle = pos_draws_per_batch[0] + pos_draws_per_batch[1]
    assert sorted(first_cycle) == sorted(pos_set), (
        f"First cycle (2 batches, 4 draws) should contain every "
        f"positive exactly once; got {first_cycle}"
    )
    second_cycle = pos_draws_per_batch[2] + pos_draws_per_batch[3]
    assert sorted(second_cycle) == sorted(pos_set), (
        f"Second cycle (next 2 batches, 4 draws) should also contain "
        f"every positive exactly once; got {second_cycle}"
    )


# ----------------------------------------------------------------------
# 6. drop_last semantics — partial final batch is dropped, length adjusts.
# ----------------------------------------------------------------------
def test_drop_last_drops_partial_final_batch():
    # 100 'other', n_other_per_batch=8 → 12 full + 1 partial (of 4).
    n_pos, n_idk, n_other = 4, 4, 100
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    sampler_keep = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        seed=0, drop_last=False,
    )
    sampler_drop = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        seed=0, drop_last=True,
    )
    assert len(sampler_keep) == 13  # ceil(100/8)
    assert len(sampler_drop) == 12  # floor(100/8)
    batches_drop = list(sampler_drop)
    assert len(batches_drop) == 12
    # Every batch is full-size under drop_last.
    for b in batches_drop:
        assert len(b) == 12


# ----------------------------------------------------------------------
# 7. Assembled-batch shuffle: positives are not always at the start.
# ----------------------------------------------------------------------
def test_batch_layout_is_shuffled():
    n_pos, n_idk, n_other = 8, 8, 100
    sources, labels = _make_inputs(n_pos, n_idk, n_other)
    pos_set, _, _ = _index_buckets(n_pos, n_idk, n_other)
    sampler = StratifiedBatchSampler(
        sources=sources, labels=labels, batch_size=12, n_pos=2, n_idk=2,
        seed=0,
    )
    batches = list(sampler)
    # The probability that all positives sit at the leading two
    # positions of every batch is vanishingly small under shuffle. If
    # the shuffle were missing, the first two indices of every batch
    # would lie in pos_set.
    leading_two_all_pos = all(
        (b[0] in pos_set and b[1] in pos_set) for b in batches
    )
    assert not leading_two_all_pos, (
        "Every batch had its first two slots populated by positives — "
        "the post-assembly shuffle is missing or broken."
    )
