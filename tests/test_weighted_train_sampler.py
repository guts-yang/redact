"""Tests for :class:`redact.training.WeightedSplitAwareTrainer`.

Covers:
- ``_get_train_sampler`` on the unweighted ``SplitAwareTrainer`` falls
  through to the HF base implementation (a ``RandomSampler``).
- ``_get_train_sampler`` on ``WeightedSplitAwareTrainer`` returns a
  ``WeightedRandomSampler`` carrying the supplied weights.
- Length mismatch between ``sample_weights`` and ``train_dataset``
  raises at construction.
- Non-"random" ``train_sampling_strategy`` raises at sampler build
  (we refuse to silently drop length-grouped or sequential sampling).
- ``world_size > 1`` raises at sampler build (single-process only).
- End-to-end empirical check: a 99:1 class imbalance with α=1.0
  produces ≈50/50 class frequencies after one pseudo-epoch of
  iteration.  This catches wiring bugs (e.g., weights passed in the
  wrong slot) that pure math tests on :func:`compute_sample_weights`
  cannot see.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from torch.utils.data import RandomSampler, WeightedRandomSampler
from transformers import TrainingArguments

from redact.training import (
    SplitAwareTrainer,
    WeightedSplitAwareTrainer,
    compute_sample_weights,
)


# ---------- minimal model + dataset fixtures ----------


class _TinyModel(torch.nn.Module):
    """Smallest possible HF-Trainer-compatible model.

    The trainer constructor only requires ``forward`` to accept the
    column names of the dataset (renamed to ``labels``) and return a
    ``(loss, logits)`` tuple or a dict.  We never actually train; the
    model is needed only to satisfy ``Trainer.__init__``.
    """

    def __init__(self, vocab_size: int = 8, num_labels: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 4)
        self.head = torch.nn.Linear(4, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids).mean(dim=1)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def _make_dataset(class_ids: list[int]) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[i % 8] for i in range(len(class_ids))],
            "attention_mask": [[1] for _ in class_ids],
            "labels": class_ids,
        }
    )


def _make_training_args(tmp_path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        do_train=True,
        do_eval=False,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=42,
    )


# ---------- base trainer falls through ----------


def test_split_aware_trainer_uses_random_sampler(tmp_path):
    """Without sample_weights, the unweighted SplitAwareTrainer should
    return HF's default RandomSampler."""
    dataset = _make_dataset([0, 1] * 10)
    trainer = SplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
    )
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, RandomSampler), (
        f"Expected RandomSampler, got {type(sampler).__name__}"
    )


# ---------- weighted trainer ----------


def test_weighted_trainer_returns_weighted_sampler(tmp_path):
    dataset = _make_dataset([0, 1] * 10)
    weights = compute_sample_weights(dataset["labels"], alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    sampler = trainer._get_train_sampler()
    assert sampler is not None
    assert isinstance(sampler, WeightedRandomSampler), (
        f"Expected WeightedRandomSampler, got {type(sampler).__name__}"
    )
    assert sampler.replacement is True
    assert sampler.num_samples == len(dataset)
    # The sampler holds the same weights we passed in (cast to fp64 on
    # CPU per the WeightedSplitAwareTrainer contract).
    assert sampler.weights.dtype == torch.float64
    assert torch.allclose(
        sampler.weights, weights.to(dtype=torch.float64, device="cpu")
    )


def test_weighted_trainer_rejects_length_mismatch(tmp_path):
    dataset = _make_dataset([0, 1] * 10)
    bad_weights = torch.ones(5, dtype=torch.float64)  # wrong length
    with pytest.raises(ValueError, match="sample_weights length"):
        WeightedSplitAwareTrainer(
            model=_TinyModel(),
            args=_make_training_args(tmp_path),
            train_dataset=dataset,
            sample_weights=bad_weights,
        )


def test_weighted_trainer_rejects_none_sample_weights(tmp_path):
    dataset = _make_dataset([0, 1] * 10)
    with pytest.raises(ValueError, match="requires sample_weights"):
        WeightedSplitAwareTrainer(
            model=_TinyModel(),
            args=_make_training_args(tmp_path),
            train_dataset=dataset,
            sample_weights=None,
        )


def test_weighted_trainer_accepts_list_input(tmp_path):
    """Plain Python list weights should be coerced to a float64 CPU tensor."""
    dataset = _make_dataset([0, 1] * 5)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=[1.0] * 10,
    )
    assert trainer._sample_weights is not None
    assert trainer._sample_weights.dtype == torch.float64
    assert trainer._sample_weights.device.type == "cpu"


# ---------- guards on the sampler build path ----------


def test_non_random_sampling_strategy_rejected(tmp_path):
    """If a user passes train_sampling_strategy='sequential' / 'group_by_length',
    we refuse to silently override it with weighted sampling."""
    dataset = _make_dataset([0, 1] * 10)
    weights = compute_sample_weights(dataset["labels"], alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    # Monkey-patch the arg after construction (the HF TrainingArguments
    # validator may reject non-default values at init time depending on
    # version, but our guard reads it dynamically so we can simulate
    # the conflict here).
    trainer.args.train_sampling_strategy = "group_by_length"
    with pytest.raises(ValueError, match="train_sampling_strategy"):
        trainer._get_train_sampler()


def test_distributed_world_size_rejected(tmp_path):
    """world_size > 1 → NotImplementedError (no DistributedSamplerWrapper)."""
    dataset = _make_dataset([0, 1] * 10)
    weights = compute_sample_weights(dataset["labels"], alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    # TrainingArguments.world_size is a @property; patch it on the class
    # for the duration of this test.
    with patch.object(
        type(trainer.args),
        "world_size",
        new_callable=lambda: property(lambda self: 4),
    ):
        with pytest.raises(NotImplementedError, match="single-process"):
            trainer._get_train_sampler()


# ---------- empirical end-to-end ----------


def test_empirical_class_balance_on_99_to_1_imbalance(tmp_path):
    """Iterate the sampler for one pseudo-epoch on a 99:1 imbalance and
    check the resulting class frequencies are ≈50/50 at α=1.0.

    Catches wiring bugs (e.g., trainer holding weights but passing the
    wrong tensor to WeightedRandomSampler) that the pure-math
    compute_sample_weights tests can't see.
    """
    # 99 class-0 rows, 1 class-1 row.
    class_ids = [0] * 99 + [1] * 1
    dataset = _make_dataset(class_ids)
    weights = compute_sample_weights(class_ids, alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    sampler = trainer._get_train_sampler()
    assert sampler is not None

    # Draw one full pseudo-epoch.
    drawn_indices = list(sampler)
    assert len(drawn_indices) == len(dataset)

    drawn_classes = [class_ids[i] for i in drawn_indices]
    n_class_0 = sum(1 for c in drawn_classes if c == 0)
    n_class_1 = sum(1 for c in drawn_classes if c == 1)

    # Equal-mass property: at α=1 we expect n_0 ≈ n_1 ≈ 50.
    # Use a generous tolerance (±20%) — multinomial draws with N=100
    # have ~5 stddev around 50.  Seed is fixed (via training_args)
    # so the draw is deterministic for the test.
    assert n_class_0 + n_class_1 == 100
    assert 30 <= n_class_0 <= 70, (
        f"Expected ≈50 class-0 draws under weighted balance, got {n_class_0}"
    )
    assert 30 <= n_class_1 <= 70, (
        f"Expected ≈50 class-1 draws under weighted balance, got {n_class_1}"
    )


def test_epoch_progression_advances_generator(tmp_path):
    """Calling `iter(sampler)` twice in a row produces different draws.

    Verifies the docstring claim that "each epoch re-iterates the sampler
    and produces a *different* draw (the generator's state has moved)".
    If a future regression re-seeded the generator inside
    ``_get_train_sampler`` per-iter rather than per-construction,
    consecutive epochs would draw identical sequences — this test
    catches that.
    """
    class_ids = [0] * 50 + [1] * 50
    dataset = _make_dataset(class_ids)
    weights = compute_sample_weights(class_ids, alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    sampler = trainer._get_train_sampler()
    assert sampler is not None
    draws1 = list(sampler)
    draws2 = list(sampler)
    assert draws1 != draws2, (
        "Consecutive epochs returned identical sequences — generator "
        "state did not advance. Did the sampler re-seed per __iter__?"
    )


def test_get_train_sampler_with_explicit_dataset_arg(tmp_path):
    """HF passes a non-None train_dataset in some paths; the override
    should accept and use it (not fall back to self.train_dataset)."""
    class_ids = [0] * 10 + [1] * 10
    dataset = _make_dataset(class_ids)
    weights = compute_sample_weights(class_ids, alpha=1.0)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=_make_training_args(tmp_path),
        train_dataset=dataset,
        sample_weights=weights,
    )
    # Pass an equal-length but distinct dataset object; the sampler's
    # num_samples should still be len(dataset) (length is the only
    # thing the override reads off the dataset).
    other_dataset = _make_dataset(class_ids)
    sampler = trainer._get_train_sampler(train_dataset=other_dataset)
    assert sampler is not None
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == len(other_dataset)


def test_seed_determinism(tmp_path):
    """Two trainers with the same seed produce the same draw sequence."""
    class_ids = [0] * 50 + [1] * 50
    weights = compute_sample_weights(class_ids, alpha=1.0)

    def _draw():
        dataset = _make_dataset(class_ids)
        trainer = WeightedSplitAwareTrainer(
            model=_TinyModel(),
            args=_make_training_args(tmp_path),
            train_dataset=dataset,
            sample_weights=weights,
        )
        sampler = trainer._get_train_sampler()
        assert sampler is not None
        return list(sampler)

    assert _draw() == _draw()
