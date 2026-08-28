"""Data collators for the leak-classifier training pipeline.

Currently houses :class:`RandomPriorDropoutCollator`, which is wired in
on the ``WeightedSplitAwareTrainer.get_train_dataloader`` train path to
randomly swap each example's ``input_ids`` / ``attention_mask`` with a
pre-computed ``input_ids_no_priors`` / ``attention_mask_no_priors``
variant. The dropout breaks the shortcut where the contextual_binary
classifier learns ``is_leak ≈ (question is about forget author) AND
(priors look like CoT opener pattern)`` and ignores the focal sentence
itself — by sometimes giving the model no priors at all, it is forced
to learn focal-as-leak-signal.

The collator wraps a *base* collator (typically
:class:`DataCollatorWithPadding`) and:

1. Decides per-example whether to swap to the no-priors variant (with
   probability ``dropout_rate``).
2. Strips the ``input_ids_no_priors`` / ``attention_mask_no_priors``
   columns AFTER deciding — they must not reach the base collator,
   which would try to pad them alongside ``input_ids`` and crash
   because the two have different lengths.
3. Delegates the swapped + stripped features to the base collator.

Determinism: a ``torch.Generator`` is seeded at construction. Two
collators with the same seed see the same swap sequence given the same
input order. The generator's state advances continuously across batch
calls within a single training run.

Resume contract: ``resume_from_checkpoint`` restarts the dropout RNG —
the trainer's accidental sampler resume safety (which relies on HF
``skip_first_batches`` iterating the sampler to advance state) does NOT
extend here, because ``accelerate.data_loader.SkipBatchSampler.__iter__``
only advances the inner sampler, not the dataloader's ``collate_fn``.
Two consequences worth knowing:

1. A run resumed from step N draws swaps from a fresh RNG stream
   (generator state = seed), not the state-after-N-batches stream the
   unbroken run would have. The aggregate per-example dropout
   frequency is statistically the same; the exact per-batch sequence
   differs.
2. Bit-exact resume would require routing the RNG through a step
   counter (e.g., a ``TrainerCallback`` that re-seeds from
   ``args.seed + state.global_step`` at the start of each step, or
   overriding the dataloader to track skipped batches). Out of scope
   for this iteration; the aggregate-rate property is what the
   downstream training objective depends on.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


_NO_PRIORS_COLUMNS = ("input_ids_no_priors", "attention_mask_no_priors")


class RandomPriorDropoutCollator:
    """Randomly swap each example's priors-on inputs for a priors-off variant.

    Wraps a base collator (typically :class:`DataCollatorWithPadding`).
    On each call, iterates over the input features and, with
    probability ``dropout_rate``, replaces an example's ``input_ids`` /
    ``attention_mask`` with its pre-computed ``input_ids_no_priors`` /
    ``attention_mask_no_priors`` variant. The no-priors columns are
    then stripped from every feature before delegation so the base
    collator never sees them — it would otherwise try to pad two
    different-length tensors under the same key and crash.

    Parameters
    ----------
    base_collator
        The downstream collator that actually does padding /
        tensor-stacking. The wrapper is intentionally agnostic to its
        type — anything that accepts a ``list[dict]`` and returns a
        batch dict works.
    dropout_rate
        Probability of swapping a given example. Must lie in
        ``[0.0, 1.0)``. ``0.0`` is a pass-through (strip-only); rates
        approaching 1.0 swap every example with vanishingly small
        slip-through probability. ``1.0`` is excluded — pass
        ``max_prior_sentences=0`` at tokenization time if you want
        priors-off always (matches the corresponding range on
        :class:`redact.training.WeightedSplitAwareTrainer`'s
        ``prior_dropout_rate`` kwarg).
    seed
        Seed for the internal :class:`torch.Generator`. Two collators
        with the same seed produce the same swap sequence given the
        same input order.
    """

    def __init__(
        self,
        base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
        dropout_rate: float,
        seed: int = 0,
    ):
        # Range matches WeightedSplitAwareTrainer.prior_dropout_rate
        # exactly so the trainer-level guard and the collator-level
        # guard never disagree. 1.0 is excluded — see class docstring.
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"RandomPriorDropoutCollator.dropout_rate must lie in "
                f"[0.0, 1.0); got {dropout_rate!r}. (1.0 is excluded "
                f"— pass max_prior_sentences=0 at tokenization time "
                f"if you want priors-off always.)"
            )
        self.base_collator = base_collator
        self.dropout_rate = float(dropout_rate)
        # Float64 so a future low-precision torch default doesn't quietly
        # shift the swap distribution. The generator runs on CPU because
        # the collator is invoked on the dataloader (main / worker)
        # process, not the GPU.
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed))

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Empty-batch contract: DataCollatorWithPadding raises on an
        # empty list in current HF (the padding routine indexes the
        # first feature). Mirror that by delegating immediately — we
        # do not synthesize a no-op batch shape; the caller's error
        # path is the right surface.
        if not features:
            return self.base_collator(features)

        if self.dropout_rate > 0.0:
            # Validate every example carries the no_priors columns
            # AND draw the swap decisions in one pass. If we let a
            # missing column slip through, the swap would silently
            # fall back to the priors-on variant for that example —
            # a determinism violation at best, a config-drift hazard
            # at worst. Single iteration over features avoids the
            # O(2N) double-loop the prior implementation did.
            for col in _NO_PRIORS_COLUMNS:
                for i, feat in enumerate(features):
                    if col not in feat:
                        raise ValueError(
                            f"RandomPriorDropoutCollator: dropout_rate="
                            f"{self.dropout_rate!r} requires column "
                            f"{col!r} on every input feature, but it "
                            f"is missing on feature at index {i}. Did "
                            f"the tokenization step emit the no_priors "
                            f"columns? Check _contextual_tokenize_binary."
                        )
            # Draw all swap decisions in a single op so the generator's
            # state advances by exactly one rand_call per batch — keeps
            # determinism easy to reason about.
            swap_probs = torch.rand(
                len(features), generator=self._generator,
            )
            swap_mask = swap_probs < self.dropout_rate
        else:
            # Rate-0 fast path: don't bother drawing or advancing the
            # generator. Strip the no_priors columns if present (the
            # caller may have left them on the dataset for symmetry
            # with rate > 0 runs) and pass through.
            swap_mask = None

        new_features: list[dict[str, Any]] = []
        for i, feat in enumerate(features):
            # Copy the dict so we don't mutate the dataset cache (the
            # HF Datasets layer hands us dict views; mutating in-place
            # would persist across epochs).
            new_feat = dict(feat)
            if swap_mask is not None and bool(swap_mask[i].item()):
                # Swap: replace input_ids / attention_mask with the
                # no_priors variants. The no_priors columns are
                # popped right after (regardless of swap outcome) so
                # the base collator never sees them.
                new_feat["input_ids"] = new_feat["input_ids_no_priors"]
                new_feat["attention_mask"] = new_feat[
                    "attention_mask_no_priors"
                ]
            # Strip the no_priors columns from every feature (swap or
            # not). DataCollatorWithPadding inspects all keys looking
            # for tokenizer-known names; an unknown variable-length
            # list under input_ids_no_priors raises a padding error,
            # and even if it didn't, the column would be torch.stack'd
            # at different lengths across the batch.
            for col in _NO_PRIORS_COLUMNS:
                new_feat.pop(col, None)
            new_features.append(new_feat)

        return self.base_collator(new_features)
