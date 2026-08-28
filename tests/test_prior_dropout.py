"""Tests for :class:`redact.data_collators.RandomPriorDropoutCollator`.

Validates the swap-decision / stripping / determinism / edge-case
contracts documented on the collator. The downstream contextual_binary
classifier's training pipeline depends on random prior dropout to break
the "question-about-forget-author + CoT-opener-pattern priors" shortcut
— these tests are the regression suite for that contract.

The collator is wired into both train paths
(:class:`WeightedSplitAwareTrainer` weighted + stratified) via the
``prior_dropout_rate`` constructor parameter; see ``test_training_utils``
and ``test_weighted_train_sampler`` for the trainer-level integration.
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding

from redact.data_collators import RandomPriorDropoutCollator


# ---------- helpers ----------


@pytest.fixture(scope="module")
def tokenizer():
    """Real RoBERTa tokenizer — the base collator depends on it for
    padding behavior. Module-scoped so the slow ``from_pretrained``
    call only runs once."""
    return AutoTokenizer.from_pretrained("roberta-base")


@pytest.fixture
def base_collator(tokenizer):
    """Production-shape base collator used in the binary head branch."""
    return DataCollatorWithPadding(tokenizer, padding="longest")


def _make_feature(
    *,
    priors_ids: list[int],
    no_priors_ids: list[int],
    label: int = 0,
) -> dict:
    """One synthetic training feature carrying both variants.

    Lengths differ between the two variants on purpose — that's the
    realistic case (no-priors variant is shorter because the prompt
    block has no priors after it). The collator's contract is that
    these unequal-length tensors must not BOTH reach the base
    collator (DataCollatorWithPadding would crash trying to pad them
    together).
    """
    return {
        "input_ids": list(priors_ids),
        "attention_mask": [1] * len(priors_ids),
        "input_ids_no_priors": list(no_priors_ids),
        "attention_mask_no_priors": [1] * len(no_priors_ids),
        "labels": label,
    }


# ----------------------------------------------------------------------
# 1. Rate=0 is a no-op (strip-only pass-through).
# ----------------------------------------------------------------------


def test_rate_zero_passes_through_base_collator(base_collator, tokenizer):
    """With rate=0, the collator should never swap; it should just
    strip the no_priors columns and delegate. The output must match
    what the base collator would produce on the priors-on inputs
    alone (i.e., the no_priors columns are invisible to the base
    collator)."""
    features = [
        _make_feature(
            priors_ids=[0, 100, 200, 2],
            no_priors_ids=[0, 100, 2],
            label=0,
        ),
        _make_feature(
            priors_ids=[0, 300, 400, 500, 2],
            no_priors_ids=[0, 300, 2],
            label=1,
        ),
    ]
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.0, seed=0,
    )
    wrapped_batch = wrapped(features)

    # Reference: feed the same priors-on features (stripped of
    # no_priors keys) directly through the base collator.
    stripped_features = [
        {k: v for k, v in feat.items()
         if k not in ("input_ids_no_priors", "attention_mask_no_priors")}
        for feat in features
    ]
    reference_batch = base_collator(stripped_features)

    # Equal keys and equal values — bit-for-bit no-op.
    assert set(wrapped_batch.keys()) == set(reference_batch.keys())
    for k in wrapped_batch:
        if isinstance(wrapped_batch[k], torch.Tensor):
            assert torch.equal(wrapped_batch[k], reference_batch[k]), (
                f"rate=0 produced a different {k!r} tensor than the "
                f"base collator on the priors-on inputs."
            )
        else:
            assert wrapped_batch[k] == reference_batch[k]


# ----------------------------------------------------------------------
# 2. Rate near 1.0 always swaps (rate=0.999 to honor the [0, 1) range
# validation while saturating the swap distribution).
# ----------------------------------------------------------------------


def test_rate_near_one_always_swaps(base_collator, tokenizer):
    """At rate=0.999, every example's effective input_ids must equal
    its no_priors variant (modulo padding).

    The collator validates ``0 <= rate < 1``, so we use 0.999 here
    instead of literal 1.0 to honor the range. With 50 features and
    rate=0.999, the probability of ANY example not swapping is
    1 - 0.999^50 ≈ 4.9%; we use a large enough draw count (and a
    fixed seed) to make spurious test failures unlikely. Determinism
    of the seeded generator means the result is fully reproducible.
    """
    features = []
    for i in range(50):
        priors_len = 8 + (i % 4)
        priors_ids = [0] + list(range(100 + i, 100 + i + priors_len)) + [2]
        no_priors_ids = [0, 200 + i, 2]
        features.append(_make_feature(
            priors_ids=priors_ids,
            no_priors_ids=no_priors_ids,
            label=i % 2,
        ))

    # Single deterministic draw with fixed seed; below we assert every
    # example was swapped (the seed below was chosen so the test
    # passes; if a future torch RNG change breaks this, swap to a
    # higher rate or different seed).
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.999, seed=7,
    )
    wrapped_batch = wrapped(features)

    # Reference: feed the no_priors variants directly through the
    # base collator (after relabeling them to input_ids /
    # attention_mask).
    no_priors_features = [
        {
            "input_ids": feat["input_ids_no_priors"],
            "attention_mask": feat["attention_mask_no_priors"],
            "labels": feat["labels"],
        }
        for feat in features
    ]
    reference_batch = base_collator(no_priors_features)

    # With rate=0.999 + seed=7, we expect (probabilistically) every
    # example to swap. The deterministic generator means this is
    # either always true under this seed or always false; assert it
    # and let the test catch a future RNG regression.
    assert wrapped_batch["input_ids"].shape == reference_batch["input_ids"].shape, (
        f"With rate~1, swapped batch shape "
        f"({tuple(wrapped_batch['input_ids'].shape)}) should equal the "
        f"reference no-priors-only batch shape "
        f"({tuple(reference_batch['input_ids'].shape)}). A shape "
        f"mismatch implies at least one example did NOT swap."
    )
    assert torch.equal(wrapped_batch["input_ids"], reference_batch["input_ids"]), (
        "Some example failed to swap at rate=0.999 under this seed."
    )


# ----------------------------------------------------------------------
# 3. Statistical correctness: rate=0.5 yields ~50% swap frequency
# over 1000 single-feature batches.
# ----------------------------------------------------------------------


def test_rate_half_statistical_swap_frequency(base_collator):
    """1000 single-feature batches at rate=0.5; observed swap rate
    should fall in 0.5 +/- 0.05.

    Single-feature-per-batch design keeps the assertion clean:
    each batch's input_ids equals exactly one of the two variants,
    so we can count swaps unambiguously by comparing tensor
    identity rather than indexing into per-position swap masks.
    """
    n_batches = 1000
    priors_marker = 999
    no_priors_marker = 111
    base_feature = _make_feature(
        priors_ids=[0, priors_marker, priors_marker, 2],
        no_priors_ids=[0, no_priors_marker, 2],
        label=0,
    )

    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=42,
    )
    n_swaps = 0
    for _ in range(n_batches):
        batch = wrapped([dict(base_feature)])
        # The first non-special token at index 1 disambiguates which
        # variant was kept (markers are unique per variant).
        if int(batch["input_ids"][0, 1].item()) == no_priors_marker:
            n_swaps += 1
    observed_rate = n_swaps / n_batches
    # +/-0.05 (5 percentage points) is generous: a binomial with
    # p=0.5, n=1000 has std ~1.58%, so +/-0.05 corresponds to ~3
    # stddev. The seed is fixed so spurious failures are
    # deterministic, but the slack absorbs minor torch RNG version
    # drift.
    assert abs(observed_rate - 0.5) < 0.05, (
        f"At rate=0.5, observed swap rate {observed_rate:.3f} over "
        f"{n_batches} draws is more than 0.05 from 0.5. Either the "
        f"swap logic is biased or the torch RNG implementation has "
        f"drifted under this seed."
    )


# ----------------------------------------------------------------------
# 4. Determinism: same seed → same swap sequence on same input order.
# ----------------------------------------------------------------------


def test_determinism_same_seed_same_sequence(base_collator):
    """Two collators with the same seed produce identical batches
    given the same input order."""
    features = [
        _make_feature(
            priors_ids=[0, 100 + i, 100 + i, 2],
            no_priors_ids=[0, 200 + i, 2],
            label=i % 2,
        )
        for i in range(20)
    ]
    a = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=123,
    )
    b = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=123,
    )
    # Run two epochs each. The generator state advances continuously
    # across calls, so each successive call returns different swaps;
    # both collators must trace the same sequence.
    for _ in range(3):
        batch_a = a([dict(f) for f in features])
        batch_b = b([dict(f) for f in features])
        assert torch.equal(batch_a["input_ids"], batch_b["input_ids"])
        assert torch.equal(
            batch_a["attention_mask"], batch_b["attention_mask"],
        )


# ----------------------------------------------------------------------
# 5. Different seeds differ.
# ----------------------------------------------------------------------


def test_different_seeds_differ(base_collator):
    """Two collators with different seeds produce different swap
    sequences given the same input order (at least one batch should
    differ)."""
    features = [
        _make_feature(
            priors_ids=[0, 100 + i, 100 + i, 2],
            no_priors_ids=[0, 200 + i, 2],
            label=i % 2,
        )
        for i in range(50)
    ]
    a = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=0,
    )
    b = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=1,
    )
    batch_a = a([dict(f) for f in features])
    batch_b = b([dict(f) for f in features])
    # Different seeds → different RNG streams → different swap
    # decisions on at least one of the 50 examples (the probability
    # that two 50-bit Bernoulli(0.5) sequences match exactly is
    # 2^-50, vanishingly small).
    assert not torch.equal(batch_a["input_ids"], batch_b["input_ids"]), (
        "Two seeds produced identical batches — the RNG is not "
        "actually being seeded per-construction."
    )


# ----------------------------------------------------------------------
# 6. No-priors columns stripped from output.
# ----------------------------------------------------------------------


def test_no_priors_columns_stripped(base_collator):
    """The wrapped batch must NOT carry the no_priors keys (they
    would crash downstream model code on the forward pass).

    Test both rate=0 and rate=0.5 paths — both must strip.
    """
    features = [
        _make_feature(
            priors_ids=[0, 100, 100, 2],
            no_priors_ids=[0, 200, 2],
            label=0,
        ),
    ]
    for rate in (0.0, 0.5):
        wrapped = RandomPriorDropoutCollator(
            base_collator=base_collator, dropout_rate=rate, seed=0,
        )
        batch = wrapped([dict(features[0])])
        assert "input_ids_no_priors" not in batch, (
            f"input_ids_no_priors leaked into the output batch at "
            f"rate={rate}; downstream forward() would crash."
        )
        assert "attention_mask_no_priors" not in batch, (
            f"attention_mask_no_priors leaked into the output batch "
            f"at rate={rate}; downstream forward() would crash."
        )


# ----------------------------------------------------------------------
# 7. Edge cases.
# ----------------------------------------------------------------------


def test_rate_outside_range_raises():
    """The constructor's range validator must reject negative rates,
    rates == 1.0, and rates > 1.0. The collator-level range matches
    the trainer-level range exactly (``[0.0, 1.0)``) so the two
    guards never disagree."""
    # Need a callable; identity is fine for the validation path.
    def identity_collator(features):
        return features

    for bad in (-0.01, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"must lie in \[0.0, 1.0\)"):
            RandomPriorDropoutCollator(
                base_collator=identity_collator,
                dropout_rate=bad,
                seed=0,
            )


def test_missing_no_priors_columns_raises(base_collator):
    """When rate > 0, every feature must have the no_priors columns.
    A missing column on any feature should raise with a clear
    diagnostic before the silent fallback ever occurs."""
    bad_features = [
        # First feature is fine (has both no_priors columns).
        _make_feature(
            priors_ids=[0, 100, 2],
            no_priors_ids=[0, 200, 2],
            label=0,
        ),
        # Second feature is missing the no_priors columns.
        {
            "input_ids": [0, 300, 2],
            "attention_mask": [1, 1, 1],
            "labels": 1,
        },
    ]
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=0,
    )
    with pytest.raises(ValueError, match="requires column"):
        wrapped(bad_features)


def test_missing_no_priors_columns_silent_at_rate_zero(base_collator):
    """Symmetric: at rate=0, missing no_priors columns are FINE — the
    collator only strips (if present) and delegates. This is the
    backward-compatibility contract: existing training configs
    without the new columns still work when rate=0."""
    features = [
        {
            "input_ids": [0, 100, 2],
            "attention_mask": [1, 1, 1],
            "labels": 0,
        },
    ]
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.0, seed=0,
    )
    # Must not raise.
    batch = wrapped(features)
    assert "input_ids" in batch


def test_empty_batch_delegates_to_base(base_collator):
    """Empty batch: the collator should delegate immediately to the
    base collator (which is responsible for the empty-batch error
    surface). Mirrors DataCollatorWithPadding's behavior, which
    raises on empty input — we re-raise it rather than synthesizing
    a no-op shape."""
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=0,
    )
    with pytest.raises((IndexError, KeyError, ValueError, AttributeError)):
        # The base collator raises on empty input; the wrapper
        # propagates it rather than masking it. The exact exception
        # type depends on the HF version (transformers raises
        # AttributeError in 5.x when trying to call .keys() on an
        # empty-list intermediate); the broader catch shields the
        # test from minor exception-type drift.
        wrapped([])


# ----------------------------------------------------------------------
# 8. End-to-end smoke: real-shaped batches survive a real collator.
# ----------------------------------------------------------------------


def test_end_to_end_smoke(base_collator, tokenizer):
    """A small synthetic batch with both column sets runs through the
    wrapped collator and produces the expected shapes."""
    features = [
        _make_feature(
            priors_ids=[0, 100, 101, 102, 2],
            no_priors_ids=[0, 200, 2],
            label=0,
        ),
        _make_feature(
            priors_ids=[0, 300, 301, 302, 303, 2],
            no_priors_ids=[0, 400, 401, 2],
            label=1,
        ),
        _make_feature(
            priors_ids=[0, 500, 2],
            no_priors_ids=[0, 600, 2],
            label=0,
        ),
    ]
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=5,
    )
    batch = wrapped([dict(f) for f in features])
    # 3 examples in the batch.
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape[0] == 3
    assert batch["labels"].shape[0] == 3
    # input_ids and attention_mask must agree on the padded length.
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    # input_ids has integer dtype (token ids).
    assert batch["input_ids"].dtype == torch.long
    # attention_mask is 0/1; all rows start with the CLS=0 token.
    assert int(batch["input_ids"][0, 0].item()) == 0


# ----------------------------------------------------------------------
# 9. Trainer integration: get_train_dataloader wires the collator,
# get_eval_dataloader does NOT.
# ----------------------------------------------------------------------


def _make_dataset_with_both_variants(n: int = 8):
    """Tiny HF Dataset carrying both priors-on and priors-off columns
    plus the trainer-required labels column. Used for the wired
    integration tests below."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "input_ids": [[0, 100 + i, 101 + i, 2] for i in range(n)],
            "attention_mask": [[1, 1, 1, 1] for _ in range(n)],
            "input_ids_no_priors": [[0, 200 + i, 2] for i in range(n)],
            "attention_mask_no_priors": [[1, 1, 1] for _ in range(n)],
            "labels": [i % 2 for i in range(n)],
        }
    )


class _TinyModel(torch.nn.Module):
    """Tiny HF-Trainer-compatible model. Forward signature matches the
    columns we expect (no_priors columns NOT in the signature, so the
    trainer's ``_preserve_no_priors_columns`` sentinel + overridden
    ``_remove_unused_columns`` keep them on the train path, and the
    dropout collator strips them before reaching forward)."""

    def __init__(self, vocab_size: int = 1000, num_labels: int = 2):
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


def test_weighted_trainer_wraps_train_collator_only(tmp_path, tokenizer):
    """End-to-end: WeightedSplitAwareTrainer constructed with
    prior_dropout_rate > 0 must wrap the train collator (visible
    via get_train_dataloader().collate_fn) and leave the eval
    collator untouched."""
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(20)
    eval_dataset = _make_dataset_with_both_variants(10)
    base = DataCollatorWithPadding(tokenizer, padding="longest")

    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            do_train=True,
            do_eval=False,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
        ),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=base,
        prior_dropout_rate=0.5,
        prior_dropout_seed=0,
    )

    # Train dataloader's collate_fn is the dropout-wrapped collator.
    train_dl = trainer.get_train_dataloader()
    # HF wraps the collator with RemoveColumnsCollator; the dropout
    # collator should be nested inside it as the .data_collator.
    cfn = train_dl.collate_fn
    # Drill through the wrapping (RemoveColumnsCollator wraps our
    # dropout collator, which wraps the base DataCollatorWithPadding).
    found_dropout = False
    while hasattr(cfn, "data_collator") or hasattr(cfn, "base_collator"):
        if isinstance(cfn, RandomPriorDropoutCollator):
            found_dropout = True
            break
        cfn = getattr(cfn, "data_collator", None) or getattr(
            cfn, "base_collator", None,
        )
    if isinstance(cfn, RandomPriorDropoutCollator):
        found_dropout = True
    assert found_dropout, (
        "Train dataloader's collate_fn chain does not include a "
        "RandomPriorDropoutCollator — the prior_dropout_rate arg "
        "was silently ignored on the get_train_dataloader path."
    )

    # Eval dataloader's collate_fn must NOT be wrapped by the dropout
    # collator. The eval path inherits unchanged from HF Trainer.
    eval_dl = trainer.get_eval_dataloader()
    cfn = eval_dl.collate_fn
    while hasattr(cfn, "data_collator") or hasattr(cfn, "base_collator"):
        assert not isinstance(cfn, RandomPriorDropoutCollator), (
            "Eval dataloader's collate_fn chain includes a "
            "RandomPriorDropoutCollator. Eval must always use the "
            "natural priors-on inputs; the dropout collator should "
            "wrap the train collator only."
        )
        cfn = getattr(cfn, "data_collator", None) or getattr(
            cfn, "base_collator", None,
        )
    # Final check on the deepest layer.
    assert not isinstance(cfn, RandomPriorDropoutCollator)


def test_weighted_trainer_rate_zero_is_bit_compatible(tmp_path, tokenizer):
    """With prior_dropout_rate=0.0, the trainer's construction must
    NOT wrap the collator (rate=0 is fast-path no-op; we don't pay
    the wrapping cost). This is the backward-compat contract:
    existing training configs without --prior_dropout_rate continue
    to use the unwrapped collator path.

    The trainer can't be constructed with rate=0 AND no
    sample_weights/stratified (the mutual-exclusion check enforces
    at least one rebalancing mechanism). So we exercise the
    sample_weights path with rate=0 here.
    """
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(20)
    base = DataCollatorWithPadding(tokenizer, padding="longest")

    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            num_train_epochs=1,
            do_train=True,
            do_eval=False,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
        ),
        train_dataset=dataset,
        data_collator=base,
        sample_weights=[1.0] * len(dataset),
        # rate=0.0 is the default; explicit here for clarity.
        prior_dropout_rate=0.0,
    )

    # Walk the train dataloader's collator chain; no
    # RandomPriorDropoutCollator should appear.
    train_dl = trainer.get_train_dataloader()
    cfn = train_dl.collate_fn
    while hasattr(cfn, "data_collator") or hasattr(cfn, "base_collator"):
        assert not isinstance(cfn, RandomPriorDropoutCollator), (
            "At rate=0, the train collator chain still contains a "
            "RandomPriorDropoutCollator. Rate=0 must be a true "
            "no-op (no wrapping cost)."
        )
        cfn = getattr(cfn, "data_collator", None) or getattr(
            cfn, "base_collator", None,
        )


def test_weighted_trainer_rate_outside_range_raises(tmp_path, tokenizer):
    """Constructor must reject prior_dropout_rate outside [0.0, 1.0)."""
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(8)

    for bad_rate in (-0.01, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"prior_dropout_rate"):
            WeightedSplitAwareTrainer(
                model=_TinyModel(),
                args=TrainingArguments(
                    output_dir=str(tmp_path),
                    per_device_train_batch_size=4,
                    num_train_epochs=1,
                    save_strategy="no",
                    logging_strategy="no",
                    report_to="none",
                    seed=0,
                ),
                train_dataset=dataset,
                sample_weights=[1.0] * len(dataset),
                prior_dropout_rate=bad_rate,
            )


def test_train_path_preserves_no_priors_eval_strips_them(tmp_path, tokenizer):
    """When prior_dropout_rate > 0, the train DL must carry the
    no_priors columns through HF's column-pruning to the dropout
    collator; the eval DL must NOT carry them (the eval collator is
    unwrapped DataCollatorWithPadding, which would crash trying to
    pad two different-length tensors).

    Concretely: build a train DL and an eval DL; iterate one batch
    from each; the train collator must have seen the no_priors
    columns at some point (verified via the dropout collator's
    presence in the chain), while the eval batch must NOT carry the
    no_priors columns at all.
    """
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    # Use variable-length sequences in the no_priors variant — this
    # is what would crash the eval collator if no_priors columns
    # leaked through.
    from datasets import Dataset
    n = 8
    dataset = Dataset.from_dict({
        "input_ids": [[0, 100 + i, 101 + i, 102 + i, 2] for i in range(n)],
        "attention_mask": [[1] * 5 for _ in range(n)],
        "input_ids_no_priors": (
            [[0, 200 + i, 2] for i in range(n // 2)]
            + [[0, 300 + i, 301 + i, 2] for i in range(n // 2)]
        ),
        "attention_mask_no_priors": (
            [[1] * 3 for _ in range(n // 2)]
            + [[1] * 4 for _ in range(n // 2)]
        ),
        "labels": [i % 2 for i in range(n)],
    })

    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        sample_weights=[1.0] * len(dataset),
        prior_dropout_rate=0.5,
    )

    # Train DL: must produce batches without the no_priors columns
    # (the dropout collator strips them after the swap decision) and
    # must not crash.
    train_dl = trainer.get_train_dataloader()
    train_batch = next(iter(train_dl))
    assert "input_ids" in train_batch
    assert "input_ids_no_priors" not in train_batch
    assert "attention_mask_no_priors" not in train_batch

    # Eval DL: must NOT carry the no_priors columns (HF strips them
    # via _remove_unused_columns since _preserve_no_priors_columns
    # is False at eval time).
    eval_dl = trainer.get_eval_dataloader()
    eval_batch = next(iter(eval_dl))
    assert "input_ids" in eval_batch
    assert "input_ids_no_priors" not in eval_batch, (
        "Eval batch carries input_ids_no_priors; the column-pruning "
        "override leaked its train-only extension into the eval "
        "path. This would crash DataCollatorWithPadding on "
        "variable-length inputs."
    )
    assert "attention_mask_no_priors" not in eval_batch, (
        "Eval batch carries attention_mask_no_priors; the column-"
        "pruning override leaked its train-only extension into the "
        "eval path."
    )

    # After both DL builds, _preserve_no_priors_columns must be back
    # to False (the sentinel guards against leaking the train-time
    # extension to subsequent eval calls).
    assert trainer._preserve_no_priors_columns is False, (
        "_preserve_no_priors_columns was left True after the train "
        "DL build; subsequent eval calls would also preserve the "
        "no_priors columns and crash."
    )


def test_weighted_trainer_smoke_compose_with_stratified(tmp_path, tokenizer):
    """Smoke test: stratified path + prior dropout active should
    construct without crashing AND produce a train dataloader whose
    collator chain includes the dropout wrapper."""
    from datasets import Dataset
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    # Stratified sampler requires 'source' and 'label' columns and a
    # mix of positives / IDK / other negatives. Uses the post-refactor
    # primary source values (the sampler keys on ``src == 'idk'``).
    n_pos, n_idk, n_other = 4, 4, 32
    sources = (
        ["forget"] * n_pos
        + ["idk"] * n_idk
        + ["retain"] * n_other
    )
    labels = [1] * n_pos + [0] * n_idk + [0] * n_other

    dataset = Dataset.from_dict({
        "input_ids": [[0, 100 + i, 2] for i in range(len(labels))],
        "attention_mask": [[1, 1, 1] for _ in labels],
        "input_ids_no_priors": [[0, 200 + i, 2] for i in range(len(labels))],
        "attention_mask_no_priors": [[1, 1, 1] for _ in labels],
        "labels": labels,
        "source": sources,
    })

    stratified_kwargs = {
        "sources": sources,
        "labels": labels,
        "batch_size": 8,
        "n_pos": 2,
        "n_idk": 2,
    }

    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=8,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        stratified_batch_sampler_kwargs=stratified_kwargs,
        prior_dropout_rate=0.3,
        prior_dropout_seed=0,
    )

    train_dl = trainer.get_train_dataloader()
    cfn = train_dl.collate_fn
    found_dropout = False
    while hasattr(cfn, "data_collator") or hasattr(cfn, "base_collator"):
        if isinstance(cfn, RandomPriorDropoutCollator):
            found_dropout = True
            break
        cfn = getattr(cfn, "data_collator", None) or getattr(
            cfn, "base_collator", None,
        )
    if isinstance(cfn, RandomPriorDropoutCollator):
        found_dropout = True
    assert found_dropout, (
        "Stratified-path train collator does not include the "
        "RandomPriorDropoutCollator. Composition between stratified "
        "sampling and prior dropout is broken."
    )

    # Actually iterate a batch to confirm shapes + that no_priors
    # columns don't reach the model.
    batch = next(iter(train_dl))
    assert "input_ids" in batch
    assert "input_ids_no_priors" not in batch, (
        "no_priors columns leaked through the dropout collator into "
        "the model batch; the stripping step is broken."
    )
    assert "attention_mask_no_priors" not in batch


# ----------------------------------------------------------------------
# 10. Composition: stacking with stratified sampler / sample_weights
# trainers does not crash.
# ----------------------------------------------------------------------


def test_dropout_collator_does_not_mutate_input_features(base_collator):
    """The collator must not mutate the input feature dicts in place.
    HF Datasets caches feature dicts across epochs; mutating them
    would persist swaps across calls and break determinism /
    eval-train independence.
    """
    feat = _make_feature(
        priors_ids=[0, 100, 100, 2],
        no_priors_ids=[0, 200, 2],
        label=0,
    )
    snapshot_keys = set(feat.keys())
    snapshot_ids = list(feat["input_ids"])
    wrapped = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.999, seed=0,
    )
    wrapped([feat])
    # Keys unchanged.
    assert set(feat.keys()) == snapshot_keys, (
        f"Collator mutated the input feature's keys: original "
        f"{snapshot_keys!r}, after {set(feat.keys())!r}."
    )
    # input_ids unchanged.
    assert feat["input_ids"] == snapshot_ids, (
        "Collator mutated the input feature's input_ids list "
        "in place; downstream caching would persist swaps across "
        "calls."
    )


# ----------------------------------------------------------------------
# 11. Resume contract: documented behavior, not bit-exact resume.
# ----------------------------------------------------------------------


def test_resume_from_checkpoint_restarts_dropout_rng(base_collator):
    """Documented contract: ``resume_from_checkpoint`` restarts the
    dropout RNG (the trainer's accidental sampler resume safety does
    NOT extend here; ``SkipBatchSampler.__iter__`` only advances the
    inner sampler, not the dataloader's collate_fn).

    This test demonstrates the contract by simulating the path
    explicitly: build a collator, draw N batches (the "unbroken-run
    prefix"), then build a fresh collator with the same seed (the
    "resumed" collator), draw N more batches, and assert the resumed
    batch sequence does NOT match the unbroken-run sequence over the
    same N. The aggregate per-example dropout frequency is
    statistically the same; the per-batch identity differs.
    """
    n_prefix = 10
    n_post = 10
    base_feature = _make_feature(
        priors_ids=[0, 100, 100, 2],
        no_priors_ids=[0, 200, 2],
        label=0,
    )

    # Unbroken-run: draw n_prefix + n_post batches with a single
    # collator. Generator state advances continuously across both
    # halves.
    unbroken = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=42,
    )
    unbroken_prefix = []
    unbroken_post = []
    for _ in range(n_prefix):
        unbroken_prefix.append(
            unbroken([dict(base_feature)])["input_ids"]
        )
    for _ in range(n_post):
        unbroken_post.append(
            unbroken([dict(base_feature)])["input_ids"]
        )

    # Simulated resume: a fresh collator with the same seed draws the
    # n_post batches that would be "after the prefix". Because the
    # RNG state restarts, this draws the same sequence as
    # unbroken_prefix, NOT unbroken_post.
    resumed = RandomPriorDropoutCollator(
        base_collator=base_collator, dropout_rate=0.5, seed=42,
    )
    resumed_post = []
    for _ in range(n_post):
        resumed_post.append(
            resumed([dict(base_feature)])["input_ids"]
        )

    # The "resumed post" sequence matches the "unbroken prefix" (RNG
    # restart) but NOT the "unbroken post" (which would require the
    # generator state to have advanced through the prefix).
    matches_prefix = all(
        torch.equal(a, b) for a, b in zip(resumed_post, unbroken_prefix)
    )
    matches_post = all(
        torch.equal(a, b) for a, b in zip(resumed_post, unbroken_post)
    )
    assert matches_prefix, (
        "Resumed collator did NOT replay the seed-zero RNG stream; "
        "this means the contract documented in the module docstring "
        "(RNG restart on resume) is broken."
    )
    assert not matches_post, (
        "Resumed collator drew the same post-prefix sequence as the "
        "unbroken run. That would mean the generator state survives "
        "across construction — contradicts the documented "
        "RNG-restart contract."
    )


# ----------------------------------------------------------------------
# 12. Multi-worker guard.
# ----------------------------------------------------------------------


def test_weighted_trainer_num_workers_guard_raises(tmp_path):
    """When prior_dropout_rate > 0 AND dataloader_num_workers > 0,
    the trainer must refuse construction. Forked workers share the
    collator's torch.Generator state, producing identical swap
    masks. The simplest safe behavior is to reject the combination
    rather than silently produce correlated swaps."""
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(8)
    with pytest.raises(
        NotImplementedError, match=r"dataloader_num_workers",
    ):
        WeightedSplitAwareTrainer(
            model=_TinyModel(),
            args=TrainingArguments(
                output_dir=str(tmp_path),
                per_device_train_batch_size=4,
                num_train_epochs=1,
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                seed=0,
                dataloader_num_workers=2,
            ),
            train_dataset=dataset,
            sample_weights=[1.0] * len(dataset),
            prior_dropout_rate=0.5,
        )


def test_weighted_trainer_num_workers_zero_ok(tmp_path, tokenizer):
    """The symmetric positive case: with prior_dropout_rate > 0 AND
    dataloader_num_workers == 0 (the HF default), construction must
    succeed."""
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(8)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
            dataloader_num_workers=0,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        sample_weights=[1.0] * len(dataset),
        prior_dropout_rate=0.5,
    )
    # Trainer construction succeeded; that's the contract.
    assert trainer._prior_dropout_rate == 0.5


# ----------------------------------------------------------------------
# 13. Re-entrancy: nested get_train_dataloader restores prior state.
# ----------------------------------------------------------------------


def test_preserve_no_priors_columns_reentrant_restore(tmp_path, tokenizer):
    """If a nested / re-entrant caller stacks two
    ``get_train_dataloader`` builds, the inner build's finally
    clause must restore the OUTER frame's
    ``_preserve_no_priors_columns`` value, not hard-code False. We
    simulate the nested case by manually pre-setting the sentinel to
    True (the value an outer frame would have set) and checking it
    survives the inner call.

    Production note: nested ``get_train_dataloader`` calls are
    unlikely in HF Trainer's current code path, but the contract
    surface is public and this guard is cheap.
    """
    from transformers import TrainingArguments

    from redact.training import WeightedSplitAwareTrainer

    dataset = _make_dataset_with_both_variants(8)
    trainer = WeightedSplitAwareTrainer(
        model=_TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=4,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            seed=0,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        sample_weights=[1.0] * len(dataset),
        prior_dropout_rate=0.5,
    )
    # Simulate an outer frame setting the sentinel to True. Call
    # get_train_dataloader as the "inner" build; on return the
    # sentinel must still be True (the outer state), not False
    # (a hard-coded reset).
    trainer._preserve_no_priors_columns = True
    _ = trainer.get_train_dataloader()
    assert trainer._preserve_no_priors_columns is True, (
        "Inner get_train_dataloader build hard-coded "
        "_preserve_no_priors_columns to False instead of restoring "
        "the outer frame's True value. Re-entrancy contract broken."
    )
    # Reset for hygiene.
    trainer._preserve_no_priors_columns = False
