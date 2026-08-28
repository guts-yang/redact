"""Shared training helpers for the leak classifier.

Houses the pieces :mod:`scripts.train_classifier` reuses around the HF
:class:`Trainer`: the split-aware / rebalancing Trainer subclasses, the
:class:`LogPrintCallback` log bridge, class- and sample-weight
computation, and the per-source loss-weight parsing.  Callers retain
responsibility for dataset construction, metric computation, and CLI
argument parsing.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import datasets
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from transformers import (
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker

from redact.data_collators import RandomPriorDropoutCollator
from redact.data_samplers import StratifiedBatchSampler
from redact.utils import log_print


class LogPrintCallback(TrainerCallback):
    """Forward :class:`Trainer` log dicts through :func:`log_print`.

    Trainer writes its metrics (training loss, eval metrics) via
    ``self.log(...)`` which ultimately prints to stdout.  In the
    standard GPU-cluster launch pattern (``nohup ... >/dev/null
    2>>LOG_FILE``) stdout is discarded, so those metrics would never
    reach the log file.  This callback captures the same dicts and
    routes them through :func:`log_print` instead.  Training-loss logs
    and per-eval-split metric dicts both come through ``on_log`` — no
    need for a separate ``on_evaluate`` hook.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs:
            return
        log_print(f"[step {state.global_step}] {logs}")


class SplitAwareTrainer(Trainer):
    """:class:`Trainer` subclass that records the current eval-split prefix.

    HF ``Trainer.evaluate`` recurses on dict-valued ``eval_dataset``,
    calling ``self.evaluate(_split, metric_key_prefix=f"{prefix}_{name}")``
    for each split. Capturing ``metric_key_prefix`` on the instance lets
    a closed-over ``compute_metrics`` callable dispatch on which split
    is currently being evaluated (e.g., add a cross-distribution metric
    only on the merged ``test`` split, not on derived per-source splits
    like ``forget`` / ``retain``).

    Concretely: ``partial_auroc_retain_aux`` (and the rest of the
    retain/aux/IDK metric family, now in
    :mod:`redact.leak_classifier_metrics`) needs both forget positives and
    eligible negatives in the same call. Those only co-exist on the
    ``test`` split. Length-based dispatch
    (``len(eval_pred) == len(test_sources)``) was the previous proposal
    but breaks if any future eval split coincidentally matches the test
    split's length; this subclass-based dispatch is robust to that.

    Usage::

        trainer = SplitAwareTrainer(..., compute_metrics=fn_that_reads_prefix)
        # inside fn_that_reads_prefix:
        if trainer._current_metric_prefix == "eval_test":
            ...
    """

    _current_metric_prefix: str | None = None

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        # Set BEFORE delegating so the recursive per-split dispatch
        # (when eval_dataset is a dict) correctly updates the prefix on
        # each child call. The recursion goes through self.evaluate so
        # this override is invoked for both the outer dict call and each
        # inner per-split call.
        self._current_metric_prefix = metric_key_prefix
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


class WeightedSplitAwareTrainer(SplitAwareTrainer):
    """:class:`SplitAwareTrainer` that draws train batches via :class:`WeightedRandomSampler`.

    The HF default ``RandomSampler`` draws each train row with equal
    probability. This subclass replaces it with a
    :class:`WeightedRandomSampler` so the user can target an arbitrary
    distribution over classes (e.g., equal binary leak-vs-clean mass
    rather than the dataset's own skew). The per-row weights are
    supplied at construction time via *sample_weights* — see
    :func:`compute_sample_weights` for the canonical recipe.

    The sampler always runs with ``replacement=True``: with very
    unequal weights, ``replacement=False`` silently degrades to
    near-empirical sampling because the highest-weight rows are
    exhausted early and the remaining draws revert to the leftover
    pool. Replacement also makes the sampler robust to small classes
    (a 5-row class can supply many distinct batches per pseudo-epoch).

    Seeding: ``_get_train_sampler`` is called once at dataloader
    construction. We seed a fresh ``torch.Generator`` from
    ``args.seed`` there and pass it to the sampler, so the generator's
    state advances continuously across epochs. Each epoch re-iterates
    the sampler and produces a *different* draw (the generator's state
    has moved), but the sequence is fully determined by ``args.seed``
    and the order of consumption. Two consequences worth knowing:

    1. There is no per-epoch deterministic isolation — epoch N's
       batches are a function of "seed + having already drawn epochs
       1..N-1", not a function of (seed, N).
    2. Resume safety lands by accident, not design: on
       ``resume_from_checkpoint`` HF rebuilds the dataloader, this
       override re-seeds from ``args.seed``, and HF's
       ``skip_first_batches`` then iterates and discards the prefix —
       advancing the generator to the same state as the unbroken run.

    Single-process only: distributed training would need a
    ``DistributedSamplerWrapper`` around the weighted sampler so each
    rank sees a disjoint shard. Not implemented; raises explicitly.
    """

    def __init__(
        self,
        *args,
        sample_weights: torch.Tensor | list[float] | None = None,
        stratified_batch_sampler_kwargs: dict[str, Any] | None = None,
        prior_dropout_rate: float = 0.0,
        prior_dropout_seed: int = 0,
        **kwargs,
    ):
        # Set placeholders BEFORE super().__init__ so that if a
        # future HF Trainer version ever calls _get_train_sampler
        # during base __init__ (it doesn't in 5.4, but the override is
        # public-facing), the attributes exist. The base __init__ does
        # not currently invoke the sampler path, so the placeholders
        # are never used today.
        self._sample_weights: torch.Tensor | None = None
        self._stratified_batch_sampler_kwargs: dict | None = None
        # ``prior_dropout_rate`` plumbs through to a
        # :class:`RandomPriorDropoutCollator` wrapping the train-only
        # collator. Composes with both sample_weights and
        # stratified_batch_sampler_kwargs — see get_train_dataloader.
        if not (0.0 <= prior_dropout_rate < 1.0):
            raise ValueError(
                f"WeightedSplitAwareTrainer.prior_dropout_rate must lie "
                f"in [0.0, 1.0); got {prior_dropout_rate!r}. (1.0 is "
                f"excluded — use max_prior_sentences=0 at tokenization "
                f"time if you want priors-off always.)"
            )
        self._prior_dropout_rate: float = float(prior_dropout_rate)
        self._prior_dropout_seed: int = int(prior_dropout_seed)
        super().__init__(*args, **kwargs)
        # Mutual exclusion: WeightedRandomSampler (sample_weights) and
        # StratifiedBatchSampler (stratified_batch_sampler_kwargs) are
        # both input-level rebalancing mechanisms; combining them would
        # require resolving which one drives batch composition, and the
        # current design picks the stratified path when present. We
        # reject the combination at construction so callers don't
        # silently get one of the two ignored.
        if (
            sample_weights is not None
            and stratified_batch_sampler_kwargs is not None
        ):
            raise ValueError(
                "WeightedSplitAwareTrainer: sample_weights and "
                "stratified_batch_sampler_kwargs are mutually exclusive "
                "rebalancing mechanisms. Pass exactly one (or neither, "
                "and use SplitAwareTrainer for the HF default sampler)."
            )
        if (
            sample_weights is None
            and stratified_batch_sampler_kwargs is None
            and prior_dropout_rate == 0.0
        ):
            # Wording deliberately retains the "requires sample_weights"
            # prefix so existing pytest.raises(match=...) assertions on
            # the legacy weighted-only path continue to fire. Note that
            # the legal-shape check now also accepts a non-zero
            # prior_dropout_rate as a standalone reason to use this
            # subclass — it adds train-only collator behavior even
            # without input-level rebalancing.
            raise ValueError(
                "WeightedSplitAwareTrainer requires sample_weights or "
                "stratified_batch_sampler_kwargs (one of the two input-"
                "level rebalancing mechanisms) or a non-zero "
                "prior_dropout_rate; pass all three as None/0 and "
                "use the base SplitAwareTrainer instead."
            )

        # Multi-worker dataloaders fork the collator (and its
        # torch.Generator state) per worker, so every worker would
        # draw an identical swap mask on the same input order —
        # producing per-batch correlated dropout decisions that are
        # NOT i.i.d. across the batches a single epoch sees. HF's
        # default is 0 (single-process); we refuse to run with > 0
        # under prior dropout rather than silently producing
        # correlated swaps. The correct fix (per-worker re-seed via
        # worker_init_fn) is doable but adds plumbing complexity that
        # isn't required for the canonical single-process training
        # path used in this project.
        if prior_dropout_rate > 0.0 and getattr(
            self.args, "dataloader_num_workers", 0,
        ) > 0:
            raise NotImplementedError(
                f"WeightedSplitAwareTrainer with prior_dropout_rate="
                f"{prior_dropout_rate!r} and dataloader_num_workers="
                f"{self.args.dataloader_num_workers!r} is not "
                f"supported: forked dataloader workers share the "
                f"collator's torch.Generator state, producing "
                f"identical swap masks across workers. Set "
                f"dataloader_num_workers=0 (HF default) under prior "
                f"dropout, or extend the trainer with a "
                f"worker_init_fn that re-seeds the collator per "
                f"worker."
            )

        if sample_weights is not None:
            n_train = len(self.train_dataset)
            if len(sample_weights) != n_train:
                raise ValueError(
                    f"sample_weights length ({len(sample_weights)}) must "
                    f"equal len(train_dataset) ({n_train})."
                )
            # Materialize on CPU as float64 — WeightedRandomSampler is
            # sensitive to small-probability precision at fp32 (a class
            # with empirical frequency ~1e-5 underflows to zero weight
            # if we leave it on fp32). Keep on CPU because the sampler
            # runs in the dataloader (main process), not on the GPU.
            if not isinstance(sample_weights, torch.Tensor):
                sample_weights = torch.as_tensor(sample_weights)
            self._sample_weights = sample_weights.to(
                device="cpu", dtype=torch.float64,
            )

        if stratified_batch_sampler_kwargs is not None:
            # Defer construction until get_train_dataloader so the
            # sampler is created with the (now-finalized) seed off
            # self.args. The kwargs dict is stored as-is; we shallow-
            # copy to defend against the caller mutating it later.
            self._stratified_batch_sampler_kwargs = dict(
                stratified_batch_sampler_kwargs,
            )

    def _get_train_sampler(self, train_dataset=None) -> Sampler | None:
        # Refuse to silently override HF's non-default sampling
        # strategies. If a future caller wants weighted +
        # length-grouped sampling, that's a new sampler class, not a
        # silent drop of length grouping.
        strat = getattr(self.args, "train_sampling_strategy", "random")
        if strat != "random":
            raise ValueError(
                f"WeightedSplitAwareTrainer is incompatible with "
                f"train_sampling_strategy={strat!r}; expected 'random'."
            )
        if self.args.world_size > 1:
            raise NotImplementedError(
                "WeightedSplitAwareTrainer is single-process only. For "
                "distributed training, wrap a DistributedSamplerWrapper "
                "around the WeightedRandomSampler so each rank sees a "
                "disjoint shard."
            )
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None:
            return None
        # Stratified path takes over via get_train_dataloader; this
        # sampler method is bypassed entirely. Return None so any
        # stray caller (e.g. external code that pokes
        # _get_train_sampler directly) gets a clear signal.
        if self._stratified_batch_sampler_kwargs is not None:
            return None
        if self._sample_weights is None:
            # Standalone prior-dropout path: no rebalancing mechanism
            # is set, just prior_dropout_rate > 0. The dropout collator
            # wraps the data_collator on the train dataloader (see
            # get_train_dataloader); the sampler itself stays the HF
            # default RandomSampler, which we obtain by delegating to
            # the base Trainer's _get_train_sampler. Predicate the
            # delegation on the dropout rate so the legacy "neither
            # mechanism set" case (which __init__ rejects) still
            # surfaces as a clear constructor-time error rather than a
            # silent fall-through.
            if self._prior_dropout_rate > 0.0:
                return super()._get_train_sampler(train_dataset)
            # Unreachable: __init__ validates that at least one of
            # sample_weights / stratified_batch_sampler_kwargs /
            # prior_dropout_rate > 0 is set before storing.  Guarded
            # so a future refactor that bypasses __init__ surfaces
            # the issue here rather than as a confusing
            # WeightedRandomSampler error.
            raise RuntimeError(
                "WeightedSplitAwareTrainer._sample_weights is None, "
                "no stratified kwargs set, and prior_dropout_rate==0; "
                "the constructor must always require at least one "
                "of the three."
            )
        # Re-seed per construction. See class docstring for the
        # resume-safety contract this gives us. (Verified for HF 5.4 —
        # re-check if the Trainer dataloader build path changes.)
        gen = torch.Generator()
        gen.manual_seed(int(self.args.seed))
        return WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=len(ds),
            replacement=True,
            generator=gen,
        )

    # Sentinel guarding ``_remove_unused_columns`` against stripping
    # the no_priors columns. Set to True for the train DL build only;
    # the eval DL build runs with the flag back at False, so eval
    # batches don't carry the no_priors columns and the eval collator
    # (unwrapped :class:`DataCollatorWithPadding`) doesn't crash
    # trying to pad them.
    _preserve_no_priors_columns: bool = False

    def _remove_unused_columns(
        self,
        dataset: "datasets.Dataset",
        description: str | None = None,
    ) -> "datasets.Dataset":
        """Wrap HF's column-pruning to preserve no_priors variants on train.

        When the ``_preserve_no_priors_columns`` flag is active (set
        only inside :meth:`get_train_dataloader` when
        ``prior_dropout_rate > 0``), this method temporarily extends
        the parent's notion of "kept" columns to include the no_priors
        variants. Otherwise it delegates unchanged — eval batches
        continue to strip the no_priors columns automatically.

        Extending ``self._signature_columns`` permanently was the
        alternative; we don't do that because the same list is
        consulted by :meth:`get_eval_dataloader`, and leaving the
        no_priors columns on eval batches would crash the unwrapped
        :class:`DataCollatorWithPadding` (different-length tensors
        under the same key).
        """
        if not self._preserve_no_priors_columns:
            return super()._remove_unused_columns(
                dataset, description=description,
            )
        # Append no_priors columns to the signature for the duration
        # of this one call. self._signature_columns is a mutable list
        # so we use slice-assignment + a try/finally to restore the
        # exact prior state (handles the corner case where the list
        # was already populated with the columns, e.g. a future
        # refactor adds them upstream).
        self._set_signature_columns_if_needed()
        sig = self._signature_columns
        if sig is None:
            raise RuntimeError(
                "WeightedSplitAwareTrainer: _signature_columns is None "
                "after _set_signature_columns_if_needed(); the no_priors "
                "column injection cannot proceed."
            )
        added: list[str] = []
        for col in ("input_ids_no_priors", "attention_mask_no_priors"):
            if col not in sig:
                sig.append(col)
                added.append(col)
        try:
            return super()._remove_unused_columns(
                dataset, description=description,
            )
        finally:
            for col in added:
                try:
                    sig.remove(col)
                except ValueError:
                    # Defensive: a concurrent caller may have already
                    # popped the column. The contract is "best effort
                    # restore"; the eval-path crash this guards
                    # against requires the no_priors columns to be
                    # GONE from sig at eval time, so a stray double-
                    # remove is also fine.
                    pass

    def _get_collator_with_removed_columns(
        self,
        data_collator: Callable,
        description: str | None = None,
    ) -> Callable:
        """Wrap HF's collator-column-pruning to preserve no_priors on train.

        Same intent as the dataset-side override above, applied to the
        IterableDataset branch of :meth:`Trainer._get_dataloader`
        (which builds a :class:`RemoveColumnsCollator` instead of
        pruning the dataset directly). Our canonical contextual_binary
        path uses HF :class:`datasets.Dataset`, so the
        :meth:`_remove_unused_columns` branch is the hot one — this
        override is the symmetric guard for the IterableDataset
        branch.
        """
        if not self._preserve_no_priors_columns:
            return super()._get_collator_with_removed_columns(
                data_collator, description=description,
            )
        self._set_signature_columns_if_needed()
        sig = self._signature_columns
        if sig is None:
            raise RuntimeError(
                "WeightedSplitAwareTrainer: _signature_columns is None "
                "after _set_signature_columns_if_needed(); the no_priors "
                "column injection cannot proceed."
            )
        added: list[str] = []
        for col in ("input_ids_no_priors", "attention_mask_no_priors"):
            if col not in sig:
                sig.append(col)
                added.append(col)
        try:
            return super()._get_collator_with_removed_columns(
                data_collator, description=description,
            )
        finally:
            for col in added:
                try:
                    sig.remove(col)
                except ValueError:
                    pass

    def get_train_dataloader(self) -> DataLoader:
        """Build the training :class:`DataLoader`, branching on sampling mode.

        Three paths are reachable from a :class:`WeightedSplitAwareTrainer`:

        - **stratified** (``self._stratified_batch_sampler_kwargs`` is set):
          construct a :class:`StratifiedBatchSampler` and pass it via
          :class:`DataLoader`'s ``batch_sampler`` parameter. ``sampler``
          and ``batch_size`` must be omitted in that case (PyTorch
          enforces this — ``batch_sampler`` and ``batch_size`` are
          mutually exclusive).
        - **weighted** (``self._sample_weights`` is set): defer to the
          base :class:`Trainer.get_train_dataloader`, which calls our
          :meth:`_get_train_sampler` override and builds a
          :class:`DataLoader` with that sampler + ``self._train_batch_size``.

        Mutual exclusion between the two paths is enforced at construction
        time (see :meth:`__init__`), so exactly one of the two branches
        is reachable per instance.

        When ``self._prior_dropout_rate > 0``, both paths additionally
        wrap the final train collator in a
        :class:`RandomPriorDropoutCollator`. The
        ``input_ids_no_priors`` / ``attention_mask_no_priors`` columns
        are preserved through HF's column-pruning by setting
        ``_preserve_no_priors_columns = True`` for the train DL build,
        which the overridden :meth:`_remove_unused_columns` /
        :meth:`_get_collator_with_removed_columns` methods consume to
        extend the kept-columns set. The eval dataloader is
        intentionally NOT touched — eval always uses the natural
        priors-on inputs (the canonical :meth:`get_eval_dataloader`
        path, inherited from :class:`Trainer` and not overridden
        here, sees the flag back at False and strips the no_priors
        columns automatically).
        """
        if self._stratified_batch_sampler_kwargs is None:
            # Weighted-random-sampler path — defer entirely to the base
            # implementation, which already invokes our
            # _get_train_sampler override.
            if self._prior_dropout_rate > 0.0:
                # Temporarily swap self.data_collator for the
                # dropout-wrapped version so the base
                # get_train_dataloader picks it up. Set the
                # preserve-no-priors sentinel so the column-pruning
                # overrides keep the no_priors columns on the train
                # batch. Restore the PREVIOUS values of both in the
                # finally clause (rather than hard-coding False) so a
                # nested / re-entrant call doesn't clobber an outer
                # frame's state. The eval dataloader (built later via
                # get_eval_dataloader) sees the flag back at its
                # original False and strips the no_priors columns
                # from eval batches.
                original_collator = self.data_collator
                prev_preserve = self._preserve_no_priors_columns
                self.data_collator = RandomPriorDropoutCollator(
                    base_collator=original_collator,
                    dropout_rate=self._prior_dropout_rate,
                    seed=self._prior_dropout_seed,
                )
                self._preserve_no_priors_columns = True
                try:
                    return super().get_train_dataloader()
                finally:
                    self.data_collator = original_collator
                    self._preserve_no_priors_columns = prev_preserve
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.args.world_size > 1:
            raise NotImplementedError(
                "WeightedSplitAwareTrainer (stratified path) is single-"
                "process only. For distributed training, wrap a "
                "DistributedSamplerWrapper around the StratifiedBatchSampler "
                "so each rank sees a disjoint shard."
            )

        # Pull the dataset through HF's column-pruning + data_collator
        # adapter so the resulting batches match what the model's
        # forward() expects. Mirrors the logic in Trainer._get_dataloader
        # so adding stratified sampling doesn't accidentally change the
        # input pipeline (we still drop unused columns, still use the
        # configured collator, etc.).
        #
        # When prior dropout is active, the ``_preserve_no_priors_columns``
        # sentinel + overridden ``_remove_unused_columns`` /
        # ``_get_collator_with_removed_columns`` keep the no_priors
        # columns on the train dataset / collator. Restore the
        # PREVIOUS value (rather than hard-coding False) in the
        # finally clause so a nested / re-entrant call doesn't clobber
        # an outer frame's state — eval batches still strip them as
        # usual because the outermost caller's value is False.
        prev_preserve = self._preserve_no_priors_columns
        if self._prior_dropout_rate > 0.0:
            self._preserve_no_priors_columns = True
        try:
            dataset = self.train_dataset
            data_collator = self.data_collator
            if isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(
                    dataset, description="Training",
                )
            else:
                data_collator = self._get_collator_with_removed_columns(
                    self.data_collator, description="Training",
                )

            # When prior dropout is active, wrap the (post-prune)
            # collator in the dropout shim. Done HERE rather than via
            # the original-collator swap trick used on the weighted
            # path because the stratified path uses a local
            # ``data_collator`` variable (already post-prune) rather
            # than referring to ``self.data_collator`` directly.
            if self._prior_dropout_rate > 0.0:
                data_collator = RandomPriorDropoutCollator(
                    base_collator=data_collator,
                    dropout_rate=self._prior_dropout_rate,
                    seed=self._prior_dropout_seed,
                )
        finally:
            self._preserve_no_priors_columns = prev_preserve

        # Construct the stratified sampler. Defer construction to here
        # (rather than __init__) so we have access to the finalized
        # self.args.seed and the post-prune dataset length. The
        # sampler keys off the raw 'source' and 'label' columns from
        # the kwargs the caller supplied — the kwargs dict is the
        # contract surface, the trainer doesn't peek into the dataset
        # itself for source / label.
        kwargs = dict(self._stratified_batch_sampler_kwargs)
        # Re-seed: same contract as the weighted path (re-seed per
        # construction, generator advances continuously across epochs).
        # The caller's seed (if provided) takes precedence — they may
        # want a different RNG stream than args.seed.
        kwargs.setdefault("seed", int(self.args.seed))
        kwargs.setdefault(
            "drop_last", bool(self.args.dataloader_drop_last),
        )
        kwargs.setdefault("batch_size", int(self._train_batch_size))
        batch_sampler = StratifiedBatchSampler(**kwargs)

        should_fork = (
            torch.backends.mps.is_available()
            and self.args.dataloader_num_workers > 1
        )
        dataloader_params: dict[str, Any] = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "batch_sampler": batch_sampler,
            # worker_init_fn matches HF's per-worker seeding for
            # determinism across multi-worker dataloader runs.
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        # NOTE: batch_size / sampler / drop_last / prefetch_factor are
        # incompatible with batch_sampler. PyTorch raises if any of
        # batch_size/shuffle/sampler/drop_last is passed alongside
        # batch_sampler. prefetch_factor is fine in principle (it's a
        # DataLoader-level knob, not a sampler-level one), but is only
        # accepted when num_workers > 0.
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_prefetch_factor
            )

        return self.accelerator.prepare(
            DataLoader(dataset, **dataloader_params),
        )


def compute_sample_weights(
    class_ids: list[int],
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Per-row sampling weights for :class:`WeightedRandomSampler`.

    ``weight_i ∝ (1 / count[c(i)]) ** alpha`` where ``c(i)`` is the
    class assigned to row *i*.
    Returned as a 1D float64 tensor of length ``len(class_ids)``,
    suitable for direct construction of a
    :class:`torch.utils.data.WeightedRandomSampler`.

    The weights are *not* normalised to sum to 1 — ``WeightedRandomSampler``
    treats them as unnormalised multinomial probabilities and handles
    normalisation internally. Float64 is required because
    inverse-frequency on rare classes can underflow at fp32 (a class
    with 5 rows in a 10k-row dataset already produces weights near the
    fp32 quantisation floor at α=1).

    Parameters
    ----------
    class_ids
        Per-row integer class id. Length defines the returned tensor
        length.
    alpha
        Temperature on the inverse-frequency schedule. ``alpha=1.0``
        gives hard balance (equal expected mass per class).
        ``alpha=0.0`` gives uniform weights (recovers empirical
        sampling). ``alpha=0.5`` gives sqrt-balance — a common
        compromise that softens the rebalance without zeroing it.

    Raises
    ------
    ValueError
        If ``class_ids`` is empty, ``alpha`` is not finite, or any row
        maps to a class with count 0 — the last case is
        impossible by construction (a row's own class has count ≥ 1)
        but is guarded so that an upstream bug surfaces here rather
        than as a silent ``WeightedRandomSampler`` error.
    """
    if not class_ids:
        raise ValueError("compute_sample_weights: class_ids is empty.")
    if not np.isfinite(alpha):
        raise ValueError(
            f"compute_sample_weights: alpha must be finite, got {alpha!r}."
        )

    resolved = list(class_ids)

    counts: dict[int, int] = {}
    for c in resolved:
        counts[c] = counts.get(c, 0) + 1

    # Per-class weight: (1 / count) ** alpha.  alpha=0 collapses to 1.0
    # (uniform).  We compute in float64 throughout.
    class_w: dict[int, float] = {}
    for c, n in counts.items():
        if n <= 0:
            raise ValueError(
                f"compute_sample_weights: class {c!r} has count {n}; "
                "this should be unreachable (a row's own class is always "
                "counted)."
            )
        class_w[c] = (1.0 / float(n)) ** float(alpha)

    weights = np.asarray([class_w[c] for c in resolved], dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "compute_sample_weights: produced non-finite weights — check "
            "that alpha is reasonable and that no class has overflowing "
            "counts."
        )
    return torch.from_numpy(weights)


def compute_balanced_class_weights(
    labels: list[int], num_labels: int, device: torch.device
) -> torch.Tensor:
    """Inverse-frequency class weights for weighted cross-entropy.

    Equivalent to scikit-learn's ``class_weight='balanced'``: each class
    gets weight ``n_total / (n_classes * n_i)`` where ``n_i`` is that
    class's sample count in the training set.  Classes absent from the
    training split get weight 0 (they contribute nothing to the loss).
    """
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=float)
    for i in range(num_labels):
        if counts[i] > 0:
            weights[i] = total / (num_labels * counts[i])
    return torch.tensor(weights, dtype=torch.float32, device=device)


# Token used by ``--loss_weights_per_source`` to request the
# auto-computed default. Defined as a module-level constant so the parse
# function and the user-facing flag both refer to the same spelling.
TARGET_FRACTION_CAPPED_TOKEN = "target_fraction_capped"


def parse_per_source_loss_weights(
    spec: str | None,
    available_sources: list[str],
    available_labels: list[int],
    *,
    cap: float = 10.0,
    target_fraction: float = 1.0 / 3.0,
) -> dict[str, float]:
    """Resolve the ``--loss_weights_per_source`` spec to a per-source weight dict.

    The returned dict maps each unique source to a weight that the
    caller applies **only to label=0 rows** (label=1 rows always get
    weight 1.0 at application time — the positive class is the
    operating-point anchor and stays unmodulated). This label-gated
    semantics replaces the prior source-prefix gate so that
    ``source='forget'`` label=0 (pre-leak background) rows participate
    in the negative pool just like ``retain``, ``aux``, and ``idk``
    rows. Without this, the prior scheme drowned the
    within-forget-prompt content-discrimination signal in the
    cross-prompt retain/aux signal — forget01 v23/v24 ended up flagging
    100% / 85% of sent_idx=0 rows because the 44 pre-leak background
    rows contributed only ~1.3% of the negative gradient mass.

    Two parse modes:

    1. ``None`` / empty string → returns a uniform-1.0 dict covering
       every source in ``available_sources``.  Functionally equivalent
       to "no per-row reweighting; uniform per-row mean" — callers can
       materialize the loss_weight column from this dict and the BCE
       loss reduces to the unweighted mean.
    2. The literal token ``"target_fraction_capped"`` → auto-computes
       weights from rows where ``label == 0`` so each source's
       absolute negative loss mass equals ``target_fraction × N_neg``,
       capped at ``cap``. With K negative sources of equal size and no
       cap activating, each source receives ~``1/K`` of post-
       normalization negative mass — the ``target_fraction=1/3``
       default was chosen for the prior 3-source case before
       ``forget`` label=0 joined; with K=4 sources (the typical
       post-label-gate case), the per-source share becomes ``1/4`` but
       the absolute mass per source stays at ``N_neg/3``. The cap
       binds first on the smallest source (typically ``idk``).
    3. Comma-separated ``name=value`` pairs (e.g.,
       ``"idk=10.0,aux=0.4"``) → parses each pair as a
       float weight.  Sources not listed default to 1.0 (caller logs
       a warning if any source is missing).  Source-name matching is
       exact: a name in the spec must equal a source string in the
       training data verbatim.  No prefix or wildcard matching. Note
       that explicit weights still only apply to label=0 rows at
       application time — passing ``forget=5.0`` upweights pre-leak
       rows but does not modulate the label=1 first-leak row.

    ``available_sources`` and ``available_labels`` are parallel lists
    of per-row metadata from the training split. Used to validate
    user-supplied names exist in the data and to compute label-gated
    counts for the auto-computed default.

    Returns a ``{source_name: weight}`` dict covering EVERY source in
    ``available_sources`` (unlisted sources get 1.0).  The returned
    mapping is what callers later use to materialize the per-row
    ``loss_weight`` column — but **only after applying the label=1
    override** (label=1 rows always get 1.0 regardless of source
    weight).

    Raises ``ValueError`` if:
      - ``len(available_sources) != len(available_labels)``.
      - Any user-supplied source name does not appear in
        ``available_sources``.
      - A pair is malformed (missing ``=`` or non-numeric value).
      - A weight is non-finite or negative.
    """
    if len(available_sources) != len(available_labels):
        raise ValueError(
            f"parse_per_source_loss_weights: available_sources length "
            f"({len(available_sources)}) must equal available_labels "
            f"length ({len(available_labels)})."
        )
    unique_sources = sorted(set(available_sources))
    weights: dict[str, float] = {s: 1.0 for s in unique_sources}

    if spec is None or spec.strip() == "":
        return weights

    if spec.strip() == TARGET_FRACTION_CAPPED_TOKEN:
        # Count rows per source where label == 0 — those are the
        # negatives this metric rebalances. label == 1 rows always
        # get weight 1.0 at application time (caller enforces).
        from collections import Counter

        neg_counts: Counter[str] = Counter()
        for src, lbl in zip(available_sources, available_labels):
            if lbl == 0:
                neg_counts[src] += 1
        neg_sources = [s for s in unique_sources if neg_counts.get(s, 0) > 0]
        n_neg = sum(neg_counts.values())
        if n_neg == 0:
            raise ValueError(
                "target_fraction_capped requires at least one row with "
                "label == 0 in available_sources; saw only "
                f"{unique_sources!r}, none with label=0 rows."
            )
        for src in neg_sources:
            n_src = neg_counts[src]
            # Invariant: neg_sources excludes zero-count entries via the
            # `neg_counts.get(s, 0) > 0` filter above.
            assert n_src > 0, f"unreachable: neg_sources should exclude src={src!r}"
            raw = (n_neg * target_fraction) / n_src
            weights[src] = float(min(raw, cap))
        return weights

    # Comma-separated name=value parsing.
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Malformed --loss_weights_per_source pair {part!r}: "
                f"expected NAME=VALUE."
            )
        name, _, raw_value = part.partition("=")
        name = name.strip()
        raw_value = raw_value.strip()
        if not name:
            raise ValueError(
                f"Malformed --loss_weights_per_source pair {part!r}: "
                f"empty source name."
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Malformed --loss_weights_per_source value {raw_value!r} "
                f"for source {name!r}: not a number."
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"--loss_weights_per_source weight for {name!r} must be "
                f"finite and non-negative; got {value}."
            )
        if name not in weights:
            raise ValueError(
                f"--loss_weights_per_source source {name!r} not found in "
                f"the training data. Available sources: "
                f"{sorted(weights.keys())!r}."
            )
        weights[name] = value
    return weights


def compute_loss_weight_column(
    sources: list[str],
    labels: list[int],
    per_source_weights: dict[str, float],
) -> list[float]:
    """Materialize the per-row ``loss_weight`` column under label-gated semantics.

    Label=1 rows always receive weight 1.0 (the positive class is the
    operating-point anchor, unmodulated by per-source weighting). Label=0
    rows receive ``per_source_weights[source]``, defaulting to 1.0 when
    the source is missing from the map.

    Hoisted from an in-script closure so it can be unit-tested
    independently of HF Trainer's batched-map flow — the closure
    previously read a column name (``label``) that was renamed
    (``labels``) upstream, and the bug slipped through because the
    closure wasn't directly tested.

    Parameters
    ----------
    sources, labels
        Per-row arrays, parallel. Must have equal length.
    per_source_weights
        Output of :func:`parse_per_source_loss_weights`. Sources not in
        the map default to weight 1.0 at the label=0 branch.
    """
    if len(sources) != len(labels):
        raise ValueError(
            f"compute_loss_weight_column: sources length ({len(sources)}) "
            f"must equal labels length ({len(labels)})."
        )
    return [
        1.0 if lbl == 1 else float(per_source_weights.get(s, 1.0))
        for s, lbl in zip(sources, labels)
    ]
