"""Custom :class:`torch.utils.data.Sampler` implementations for the leak
classifier training pipeline.

Today only one sampler lives here: :class:`StratifiedBatchSampler`. It
guarantees a fixed number of positives and IDK-negatives in every
minibatch while drawing the remaining slots from the "other negative"
pool without replacement (so every other-negative row appears exactly
once per epoch). Designed to be paired with a downstream pairwise hinge
loss that needs both positives AND IDKs co-resident in every batch — a
:class:`~torch.utils.data.WeightedRandomSampler` only matches *expected*
counts and can yield batches with zero positives or zero IDKs when the
classes are highly imbalanced (forget01 has ~32 positives in ~4.4k
training rows; a uniform-mass binary sampler still produces ~half of
batches with no positives at batch_size=32 simply by Poisson variance).

The sampler is deterministic given a seed: the same ``seed`` + same
input ``sources``/``labels`` yields the same batch sequence across
runs. ``__iter__`` advances a generator seeded with ``seed + epoch``
where ``epoch`` is an internal counter incremented on each call, so
successive epochs see different orderings without the caller needing
to re-instantiate the sampler.

Not distributed-safe — single-process only (matches
:class:`WeightedSplitAwareTrainer`).
"""
from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler[list[int]]):
    """Yield fixed-composition minibatches: ``n_pos`` positives, ``n_idk`` IDKs, rest "other".

    Categorization (computed once at ``__init__``):
      * ``pos_idx``    = indices where ``labels[i] == 1``.
      * ``idk_idx``    = indices where ``labels[i] == 0`` AND
        ``sources[i] == "idk"``.
      * ``other_idx``  = all other indices (label==0 AND not an IDK).

    Per-epoch iteration:
      * Walks ``other_idx`` exactly once (shuffled per epoch). This is the
        without-replacement spine of the epoch — batches are produced
        until the spine is exhausted.
      * Maintains independent shuffled cycle buffers for ``pos_idx`` and
        ``idk_idx``; advances each by ``n_pos`` / ``n_idk`` per batch
        and reshuffles when the buffer is exhausted, so no row repeats
        within a single cycle through its bucket (guaranteeing
        round-robin coverage when the bucket is smaller than
        ``n_per_batch * num_batches``).
      * Assembled batch is shuffled before yield so the trainer does
        not see a fixed positives-first / IDKs-next layout.

    Length contract:
      * Each batch has exactly ``batch_size`` indices
        (= ``n_pos + n_idk + n_other``).
      * With ``drop_last=False`` (default), the final batch is allowed
        to have fewer ``other`` slots if ``other_idx`` runs out mid-batch;
        positives and IDKs still fill their full count from the cycle
        buffers. ``__len__`` reports ``ceil(len(other_idx) / n_other)``.
      * With ``drop_last=True``, partial final batches are dropped;
        ``__len__`` reports ``len(other_idx) // n_other``.

    Determinism: a fresh :class:`torch.Generator` is constructed each
    ``__iter__`` call and seeded with ``seed + epoch``. Epoch is an
    internal counter (starts at 0, increments after each iter
    completes). This means two samplers built with the same ``seed`` on
    the same data emit the same batch sequence; two iter() calls on the
    same sampler emit different epochs' batches.

    Validation at construction time (all raise :class:`ValueError`):
      * ``len(sources) != len(labels)``.
      * ``batch_size <= 0``, ``n_pos < 0``, ``n_idk < 0``,
        ``n_pos + n_idk >= batch_size`` (no room for ``other`` slots).
      * Empty positives pool or empty IDK pool — the sampler is
        useless if either bucket is empty (cycle buffer cannot fill).

    A small (< ``n_pos``) positive pool is allowed: the cycle buffer
    naturally handles ``len(pos_idx) < n_pos`` by reshuffling
    immediately after consumption — the same positive can repeat
    within a single batch in that degenerate case, which is acceptable
    because (a) callers should have enough positives in any realistic
    setup, and (b) the alternative (raising) would break the contract
    on a 1-positive corner case that may legitimately occur on a
    very-low-data subset.

    Parameters
    ----------
    sources
        Per-row primary source values (``"forget"``, ``"retain"``,
        ``"aux"``, ``"idk"``). Length must match ``labels``.
    labels
        Per-row binary labels (0 = background/negative, 1 = leak).
    batch_size
        Number of indices yielded per batch.
    n_pos
        Number of positives (label==1 rows) to inject per batch.
    n_idk
        Number of IDK-negatives (label==0 AND source == "idk")
        to inject per batch.
    seed
        Base RNG seed. The actual generator seed at epoch ``e`` is
        ``seed + e``.
    drop_last
        If True, drop the final batch when it would have fewer than
        ``batch_size`` indices (only the ``other`` portion can come up
        short — positives and IDKs always fill their full count from
        the cycle buffers, so ``drop_last=True`` is equivalent to
        floor-divide on the ``other`` spine).
    """

    def __init__(
        self,
        sources: list[str],
        labels: list[int],
        batch_size: int,
        n_pos: int,
        n_idk: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if len(sources) != len(labels):
            raise ValueError(
                f"StratifiedBatchSampler: len(sources) ({len(sources)}) "
                f"must equal len(labels) ({len(labels)})."
            )
        if batch_size <= 0:
            raise ValueError(
                f"StratifiedBatchSampler: batch_size must be > 0; got "
                f"{batch_size}."
            )
        if n_pos < 0 or n_idk < 0:
            raise ValueError(
                f"StratifiedBatchSampler: n_pos and n_idk must be >= 0; "
                f"got n_pos={n_pos}, n_idk={n_idk}."
            )
        if n_pos + n_idk >= batch_size:
            raise ValueError(
                f"StratifiedBatchSampler: n_pos + n_idk ({n_pos} + "
                f"{n_idk} = {n_pos + n_idk}) must be strictly less than "
                f"batch_size ({batch_size}); otherwise no room for "
                f"'other' negatives."
            )

        # Categorize once at init. The categorization is deterministic
        # in the input order — pos_idx, idk_idx, other_idx are all
        # sorted ascending by virtue of the index walk below.
        pos_idx: list[int] = []
        idk_idx: list[int] = []
        other_idx: list[int] = []
        for i, (src, lbl) in enumerate(zip(sources, labels)):
            if lbl == 1:
                pos_idx.append(i)
                continue
            # label == 0 branch.
            if src == "idk":
                idk_idx.append(i)
            else:
                other_idx.append(i)

        if not pos_idx:
            raise ValueError(
                "StratifiedBatchSampler: positives pool is empty "
                "(no rows with label == 1). This sampler is intended "
                "for imbalanced binary training; if you have no "
                "positives, you don't need stratified sampling."
            )
        if not idk_idx:
            raise ValueError(
                "StratifiedBatchSampler: IDK negatives pool is empty "
                "(no rows with label == 0 AND source == 'idk'). The "
                "IDK pool is the hardest negative class; if it is "
                "genuinely empty, drop --use_stratified_sampler "
                "instead of injecting zero-IDK batches."
            )
        if not other_idx:
            raise ValueError(
                "StratifiedBatchSampler: 'other' negatives pool is "
                "empty (every label==0 row is an IDK). The spine of "
                "the without-replacement epoch walk is empty — "
                "sampler cannot enumerate any batches."
            )

        self._pos_idx: list[int] = pos_idx
        self._idk_idx: list[int] = idk_idx
        self._other_idx: list[int] = other_idx
        self._batch_size = int(batch_size)
        self._n_pos = int(n_pos)
        self._n_idk = int(n_idk)
        self._n_other = self._batch_size - self._n_pos - self._n_idk
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        # Internal epoch counter — advances on each __iter__ call so
        # successive iterations see different shuffles. Reset only
        # if the caller re-instantiates the sampler.
        self._epoch = 0

    # ------------------------------------------------------------------
    # Length contract
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        n_other_total = len(self._other_idx)
        if self._drop_last:
            return n_other_total // self._n_other
        return math.ceil(n_other_total / self._n_other)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[list[int]]:
        # Per-epoch generator, seeded deterministically. seed + epoch
        # gives each epoch a distinct (but reproducible across runs)
        # shuffle. torch.Generator is preferred over random.Random
        # because torch.randperm is what we'll use for shuffling — keeps
        # the RNG state in one place.
        #
        # Advance the epoch counter BEFORE yielding any batches so the
        # next __iter__ call sees a new epoch regardless of how the
        # consumer terminates. HF Trainer can early-stop on max_steps,
        # eval-driven patience, or a raised exception inside the
        # training loop — bumping the counter at the end of the while
        # loop would silently replay the same shuffle on the next
        # epoch. We snapshot the current epoch into a local so the
        # rest of __iter__ uses a stable value even if the consumer
        # somehow mutates self._epoch mid-iteration.
        epoch = self._epoch
        self._epoch = epoch + 1
        gen = torch.Generator()
        gen.manual_seed(self._seed + epoch)

        # Shuffle the spine (other_idx) once per epoch. This is the
        # without-replacement walk; we'll yield batches until it's
        # exhausted.
        other_perm = self._shuffle_indices(self._other_idx, gen)

        # Cycle buffers for positives and IDKs. The "buffer" is just a
        # shuffled list plus a pointer; when the pointer hits the end,
        # we reshuffle and reset. This guarantees no repeat within a
        # cycle through the bucket.
        pos_buf = self._refresh_cycle(self._pos_idx, gen)
        pos_ptr = 0
        idk_buf = self._refresh_cycle(self._idk_idx, gen)
        idk_ptr = 0

        # Walk the spine; emit one batch per chunk of n_other.
        n_other_total = len(other_perm)
        # Step size on the spine. The final chunk may be smaller when
        # drop_last=False and n_other doesn't divide len(other_idx).
        other_ptr = 0
        while other_ptr < n_other_total:
            other_end = other_ptr + self._n_other
            other_slice = other_perm[other_ptr:other_end]
            if len(other_slice) < self._n_other and self._drop_last:
                # Final partial batch under drop_last — stop without
                # yielding.
                break
            other_ptr = other_end

            # Pull n_pos from the positive cycle, advancing /
            # refreshing as needed.
            pos_slice, pos_buf, pos_ptr = self._take_from_cycle(
                self._pos_idx, pos_buf, pos_ptr, self._n_pos, gen,
            )
            idk_slice, idk_buf, idk_ptr = self._take_from_cycle(
                self._idk_idx, idk_buf, idk_ptr, self._n_idk, gen,
            )

            batch = list(pos_slice) + list(idk_slice) + list(other_slice)
            # Shuffle the assembled batch in-place so the trainer
            # doesn't see a fixed bucket layout. This shuffle uses the
            # same generator stream so it's still fully deterministic.
            batch = self._shuffle_indices(batch, gen)
            yield batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _shuffle_indices(
        indices: list[int], gen: torch.Generator,
    ) -> list[int]:
        """Return ``indices`` shuffled by a permutation drawn from ``gen``.

        Uses :func:`torch.randperm` to keep the entire sampler RNG on a
        single :class:`torch.Generator` — mixing Python's ``random``
        with torch's generator would break determinism across runs.
        """
        if not indices:
            return []
        perm = torch.randperm(len(indices), generator=gen).tolist()
        return [indices[i] for i in perm]

    @classmethod
    def _refresh_cycle(
        cls, source: list[int], gen: torch.Generator,
    ) -> list[int]:
        """Return a freshly shuffled copy of ``source`` (cycle buffer reset)."""
        return cls._shuffle_indices(source, gen)

    @classmethod
    def _take_from_cycle(
        cls,
        source: list[int],
        buf: list[int],
        ptr: int,
        n: int,
        gen: torch.Generator,
    ) -> tuple[list[int], list[int], int]:
        """Take ``n`` indices from a cycle buffer, refreshing when exhausted.

        Returns ``(slice, new_buf, new_ptr)``. The semantics: indices
        are popped in order from ``buf[ptr:]``; when the buffer is
        exhausted mid-take, it is reshuffled and the take continues
        from the fresh buffer. This guarantees no row repeats within a
        single cycle through ``source`` — once all items have been
        served, the next shuffle starts a new no-repeat cycle.

        Degenerate case: ``len(source) < n`` — we still serve ``n``
        indices, but the same row may repeat within a single take
        (one full cycle gets consumed before the reshuffle catches up).
        That's accepted as the lesser evil compared to raising at
        construction-time on small positive pools.
        """
        if n == 0:
            return [], buf, ptr
        out: list[int] = []
        while len(out) < n:
            remaining = n - len(out)
            available = len(buf) - ptr
            if available <= 0:
                # Refresh: new shuffle, pointer back to 0.
                buf = cls._refresh_cycle(source, gen)
                ptr = 0
                available = len(buf)
            take = min(remaining, available)
            out.extend(buf[ptr:ptr + take])
            ptr += take
        return out, buf, ptr
