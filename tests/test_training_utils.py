"""Tests for batched soft-token training helpers.

Covers ``redact/regen/training_utils.py``: per-record tokenization with
the K-pad-budget truncation, batched-tensor assembly with per-row
left-padding and shifted-CE masks, the cross-pass validity filter, the
hook-schedule builder, and per-row masked CE.

The hook-application tests reuse the test pattern from
``tests/test_soft_token.py``: a tiny ``nn.Linear`` standing in for the
embedding module so we don't need a real LM in the test path.

Usage:
    conda run -n redact python -m pytest tests/test_training_utils.py -v
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from redact.regen.learned_hooks import SoftToken
from redact.regen.training_utils import (
    PassRecord,
    assemble_pass_batch,
    assert_clean_continuations_have_terminator,
    build_hook_schedule,
    cross_pass_filter,
    length_bucketed_shuffle,
    make_lr_scheduler,
    per_row_masked_ce,
    prepare_pass_record,
)


@pytest.fixture(scope="module")
def tokenizer():
    """GPT-2 tokenizer — small, fast, ships with the env. Pad token is
    None on GPT-2; tests pin pad_token_id=eos_token_id explicitly to
    mirror runtime behaviour on LLaMA-family tokenizers where pad and
    eos share an id."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    return tok


# -------------------------------------------------------------------------
# prepare_pass_record
# -------------------------------------------------------------------------


class TestPreparePassRecord:
    def test_normal_input_valid(self, tokenizer):
        rec = prepare_pass_record(
            "Tell me about Alice.",
            "Alice is a person.",
            "She lives in Paris and writes novels.",
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            n_tokens_per_block=1,
            max_seq_len=128,
            append_eos=False,
        )
        assert rec.valid
        assert len(rec.before_ids) > 0
        assert len(rec.pad_ids) == 1
        assert len(rec.after_ids) > 0

    def test_k_block_size(self, tokenizer):
        rec = prepare_pass_record(
            "p", "x", "y",
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            n_tokens_per_block=4,
            max_seq_len=128,
            append_eos=False,
        )
        assert len(rec.pad_ids) == 4
        assert all(t == tokenizer.eos_token_id for t in rec.pad_ids)

    def test_append_eos_keeps_eos_at_tail(self, tokenizer):
        rec = prepare_pass_record(
            "p", "x", "I don't know.",
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            n_tokens_per_block=1,
            max_seq_len=128,
            append_eos=True,
        )
        assert rec.valid
        assert rec.after_ids[-1] == tokenizer.eos_token_id

    def test_append_eos_eos_survives_truncation(self, tokenizer):
        # Long continuation, tight budget — truncation should preserve EOS.
        long_cont = " ".join(["word"] * 200)
        rec = prepare_pass_record(
            "prompt", "prefix", long_cont,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            n_tokens_per_block=1,
            max_seq_len=20,
            append_eos=True,
        )
        assert rec.valid
        assert rec.after_ids[-1] == tokenizer.eos_token_id

    def test_degenerate_returns_invalid(self, tokenizer):
        # Long prompt fills the budget — no room for continuation.
        long_prompt = " ".join(["word"] * 100)
        rec = prepare_pass_record(
            long_prompt, "", "anything",
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            n_tokens_per_block=1,
            max_seq_len=20,
            append_eos=False,
        )
        assert not rec.valid
        assert rec.after_ids == []

    def test_prompt_uses_add_special_tokens_true(self):
        """Prompt is tokenized with add_special_tokens=True so BOS is
        prepended for LLaMA/DeepSeek-family tokenizers, matching the
        inference path in inference_regen and generate_corrupted_cots.
        Continuations and prefixes use add_special_tokens=False (no BOS
        in the middle of a sequence)."""
        calls = []

        class _RecordingTokenizer:
            eos_token_id = 99

            def encode(self, text, add_special_tokens=False):
                calls.append({"text": text, "ast": add_special_tokens})
                return list(range(max(1, len(text))))

        prepare_pass_record(
            "PROMPT", "PREFIX", "CONTINUATION",
            tokenizer=_RecordingTokenizer(),
            pad_token_id=0,
            n_tokens_per_block=1,
            max_seq_len=128,
            append_eos=False,
            think_prefix="<think>\n",
        )

        prompt_calls = [c for c in calls if c["text"] == "PROMPT"]
        assert len(prompt_calls) == 1, (
            f"expected exactly one encode call for prompt, got {prompt_calls}"
        )
        assert prompt_calls[0]["ast"] is True, (
            "prompt must be encoded with add_special_tokens=True so BOS is "
            "prepended (matches inference path)"
        )
        # Lock prompt-first ordering: BOS at absolute position 0 is the
        # load-bearing property. A refactor that moved think before prompt
        # (e.g. think + prompt + prefix + cont) would still pass the
        # add_special_tokens=True check above but would put BOS at the
        # wrong absolute position vs inference.
        assert calls[0]["text"] == "PROMPT", (
            f"prompt must be the first encode call so BOS lands at "
            f"absolute position 0; first call was for {calls[0]['text']!r}"
        )
        non_prompt_calls = [c for c in calls if c["text"] != "PROMPT"]
        assert non_prompt_calls, "expected think/prefix/continuation encodes"
        for c in non_prompt_calls:
            assert c["ast"] is False, (
                f"non-prompt text {c['text']!r} encoded with "
                f"add_special_tokens={c['ast']!r}; must be False to avoid "
                f"injecting BOS mid-sequence"
            )

    def test_empty_prefix_skips_prefix_encode(self):
        """An empty prefix string must not trigger an encode call (mirrors
        the inference path which uses ``[] if not clean_prefix``)."""
        calls = []

        class _RecordingTokenizer:
            eos_token_id = 99

            def encode(self, text, add_special_tokens=False):
                calls.append(text)
                return [1, 2, 3]

        prepare_pass_record(
            "PROMPT", "", "CONT",
            tokenizer=_RecordingTokenizer(),
            pad_token_id=0,
            n_tokens_per_block=1,
            max_seq_len=128,
            append_eos=False,
        )
        assert "" not in calls, (
            f"empty prefix should not be passed to tokenizer.encode; "
            f"calls={calls}"
        )

    def test_append_eos_without_eos_token_raises(self, tokenizer):
        # Construct a tokenizer-like stub with no eos_token_id.
        class _NoEos:
            eos_token_id = None
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]
        with pytest.raises(ValueError, match="eos_token_id"):
            prepare_pass_record(
                "p", "x", "y",
                tokenizer=_NoEos(),
                pad_token_id=0,
                n_tokens_per_block=1,
                max_seq_len=128,
                append_eos=True,
            )


# -------------------------------------------------------------------------
# assemble_pass_batch
# -------------------------------------------------------------------------


def _rec(before, pad, after, valid=True):
    """Concise PassRecord constructor for tests."""
    return PassRecord(
        before_ids=before, pad_ids=pad, after_ids=after, valid=valid,
    )


class TestAssemblePassBatch:
    def test_b1_single_record(self):
        rec = _rec([10, 11, 12], [0, 0], [20, 21, 22])
        batch = assemble_pass_batch(
            [rec], pad_token_id=99, device="cpu",
        )
        assert batch.input_ids.shape == (1, 8)
        # All-real row, no left pad.
        assert batch.attention_mask.tolist() == [[1] * 8]
        assert batch.input_ids[0].tolist() == [10, 11, 12, 0, 0, 20, 21, 22]
        assert batch.soft_token_positions == [[3]]
        # cont_start=5, cont_end=8 -> shifted mask [4, 7) inclusive of 4..6.
        expected_mask = torch.zeros(1, 7)
        expected_mask[0, 4:7] = 1.0
        assert torch.equal(batch.loss_mask, expected_mask)

    def test_b2_mixed_lengths_left_pad(self):
        # Row 0: total_len=8.   Row 1: total_len=11. L_max=11.
        # left_pad: row 0=3, row 1=0.
        r0 = _rec([10, 11, 12], [0, 0], [20, 21, 22])
        r1 = _rec([10, 11, 12, 13, 14], [0, 0], [30, 31, 32, 33])
        batch = assemble_pass_batch(
            [r0, r1], pad_token_id=99, device="cpu",
        )
        assert batch.input_ids.shape == (2, 11)
        # Row 0 left-padded with 99s, then real tokens.
        expected_row0 = (
            [99, 99, 99]
            + [10, 11, 12]
            + [0, 0]
            + [20, 21, 22]
        )
        assert batch.input_ids[0].tolist() == expected_row0
        assert batch.attention_mask[0].tolist() == [0, 0, 0] + [1] * 8
        # Row 1 has no left pad.
        expected_row1 = [10, 11, 12, 13, 14, 0, 0, 30, 31, 32, 33]
        assert batch.input_ids[1].tolist() == expected_row1
        assert batch.attention_mask[1].tolist() == [1] * 11
        # soft_token_pos = left_pad + len(before_ids).
        # Row 0: 3 + 3 = 6.  Row 1: 0 + 5 = 5.
        assert batch.soft_token_positions == [[6], [5]]
        # loss_mask shape (2, 10).
        # Row 0: cont_start = 3+3+2 = 8, cont_end = 11.
        # Shifted mask: [7, 10). i.e. positions 7,8,9.
        # Row 1: cont_start = 0+5+2 = 7, cont_end = 11.
        # Shifted mask: [6, 10). i.e. positions 6,7,8,9.
        expected = torch.zeros(2, 10)
        expected[0, 7:10] = 1.0
        expected[1, 6:10] = 1.0
        assert torch.equal(batch.loss_mask, expected)

    def test_rejects_invalid_record(self):
        good = _rec([10], [0], [20])
        bad = _rec([10], [0], [], valid=False)
        with pytest.raises(ValueError, match="valid=False"):
            assemble_pass_batch(
                [good, bad], pad_token_id=99, device="cpu",
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="zero records"):
            assemble_pass_batch(
                [], pad_token_id=99, device="cpu",
            )

    def test_pad_block_position_consistent_with_left_pad(self):
        # The soft-token position is what a hook would write at; it
        # must point to the first padded slot, not at the left-pad
        # region (which shares the same token id as the K-block on
        # LLaMA-style tokenizers where pad_id == eos_id).
        # Row 0 has heavy left-pad; verify the position points to the
        # K-block, not the left-pad start.
        r0 = _rec([10, 11], [99], [20, 21])  # total 5
        r1 = _rec([10, 11, 12, 13, 14, 15, 16], [99], [20, 21])  # total 10
        batch = assemble_pass_batch(
            [r0, r1], pad_token_id=99, device="cpu",
        )
        # Row 0 has left_pad=5. soft_token_pos should be at index 5+2=7.
        assert batch.soft_token_positions[0] == [7]
        # The position should point to the K-pad token.
        assert batch.input_ids[0, 7].item() == 99
        # Row 1 has no left pad. soft_token_pos = 0+7=7.
        assert batch.soft_token_positions[1] == [7]
        assert batch.input_ids[1, 7].item() == 99


# -------------------------------------------------------------------------
# cross_pass_filter
# -------------------------------------------------------------------------


class TestCrossPassFilter:
    def test_all_valid_keeps_all(self):
        recs = {
            "forget": [_rec([1], [0], [2]), _rec([1], [0], [2])],
            "idk":    [_rec([1], [0], [2]), _rec([1], [0], [2])],
        }
        assert cross_pass_filter(recs) == [0, 1]

    def test_invalid_in_one_pass_drops_record(self):
        recs = {
            "forget": [_rec([1], [0], [2]), _rec([1], [0], [2])],
            "idk":    [_rec([1], [0], [2]), _rec([1], [0], [], valid=False)],
        }
        # Row 1 has invalid IDK -> drop from whole minibatch.
        assert cross_pass_filter(recs) == [0]

    def test_none_in_one_pass_drops_record(self):
        # None entries arise when the caller short-circuits on a
        # precondition (e.g. a record missing a field the pass needs).
        recs = {
            "forget": [None,                      _rec([1], [0], [2])],
            "idk":    [_rec([1], [0], [2]),       _rec([1], [0], [2])],
        }
        assert cross_pass_filter(recs) == [1]

    def test_only_forget_pass(self):
        recs = {"forget": [_rec([1], [0], [2]), _rec([1], [0], [], valid=False)]}
        assert cross_pass_filter(recs) == [0]

    def test_empty_passes_dict_raises(self):
        with pytest.raises(ValueError, match="no passes"):
            cross_pass_filter({})

    def test_inconsistent_lengths_raises(self):
        recs = {
            "forget": [_rec([1], [0], [2])],
            "idk":    [_rec([1], [0], [2]), _rec([1], [0], [2])],
        }
        with pytest.raises(ValueError, match="inconsistent lengths"):
            cross_pass_filter(recs)


# -------------------------------------------------------------------------
# build_hook_schedule
# -------------------------------------------------------------------------


class TestBuildHookSchedule:
    def test_passthrough(self):
        positions = [[5], [10]]
        sched = build_hook_schedule(positions)
        assert sched == [[5], [10]]

    def test_multiple_positions_per_row_preserved_in_order(self):
        positions = [[5, 7], [], [1]]
        sched = build_hook_schedule(positions)
        assert sched == [[5, 7], [], [1]]

    def test_inner_lists_are_copies(self):
        # The schedule must not alias the caller's
        # ``PassBatch.soft_token_positions`` — mutating the returned
        # schedule would otherwise corrupt the batch.
        positions = [[5]]
        sched = build_hook_schedule(positions)
        sched[0].append(9)
        assert positions == [[5]]

    def test_empty_batch(self):
        assert build_hook_schedule([]) == []


# -------------------------------------------------------------------------
# per_row_masked_ce
# -------------------------------------------------------------------------


class TestPerRowMaskedCE:
    def test_b1_matches_unweighted_mean_on_supervised_tokens(self):
        # B=1, L=5, V=4. Mask covers shifted positions [2, 4) — i.e.
        # supervises predicting input_ids[3] and input_ids[4].
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 4)
        input_ids = torch.tensor([[0, 1, 2, 3, 0]])
        loss_mask = torch.zeros(1, 4)
        loss_mask[0, 2:4] = 1.0
        ce = per_row_masked_ce(logits, input_ids, loss_mask)
        assert ce.shape == (1,)
        # Manual: shift logits to (5-1, 4); CE on positions 2,3 against
        # labels [3, 0]. Then mean.
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        ce_per_token = nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none",
        )
        manual = (ce_per_token[2] + ce_per_token[3]) / 2
        assert torch.isclose(ce[0], manual, atol=1e-6)

    def test_b2_independent_per_row(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 4)
        input_ids = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]])
        loss_mask = torch.zeros(2, 4)
        loss_mask[0, 2:4] = 1.0  # row 0: 2 supervised
        loss_mask[1, 0:3] = 1.0  # row 1: 3 supervised
        ce = per_row_masked_ce(logits, input_ids, loss_mask)
        assert ce.shape == (2,)
        # Verify each row independently.
        for k in range(2):
            shift_logits = logits[k, :-1, :]
            shift_labels = input_ids[k, 1:]
            per_tok = nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none",
            )
            mask = loss_mask[k]
            manual = (per_tok * mask).sum() / mask.sum()
            assert torch.isclose(ce[k], manual, atol=1e-6)

    def test_zero_supervised_row_raises(self):
        logits = torch.randn(2, 5, 4)
        input_ids = torch.zeros(2, 5, dtype=torch.long)
        loss_mask = torch.zeros(2, 4)
        loss_mask[0, 0:2] = 1.0  # row 1 has nothing supervised
        with pytest.raises(RuntimeError, match="zero supervised tokens"):
            per_row_masked_ce(logits, input_ids, loss_mask)

    def test_gradient_flows_to_logits(self):
        logits = torch.randn(1, 5, 4, requires_grad=True)
        input_ids = torch.tensor([[0, 1, 2, 3, 0]])
        loss_mask = torch.zeros(1, 4)
        loss_mask[0, 2:4] = 1.0
        ce = per_row_masked_ce(logits, input_ids, loss_mask)
        ce.sum().backward()
        assert logits.grad is not None
        # Only positions 2, 3 should have non-zero gradient on the
        # shifted logits. Positions 0, 1, 4 stay zero (mask is zero
        # there in the shifted frame; position 4 is dropped by shift).
        per_pos_grad_norm = logits.grad.norm(dim=-1).squeeze(0)
        assert per_pos_grad_norm[2].item() > 0
        assert per_pos_grad_norm[3].item() > 0
        assert per_pos_grad_norm[0].item() == 0
        assert per_pos_grad_norm[1].item() == 0
        assert per_pos_grad_norm[4].item() == 0


# -------------------------------------------------------------------------
# Hook integration: assemble + schedule + apply
# -------------------------------------------------------------------------


class TestHookOnAssembledBatch:
    """End-to-end: assemble a B=2 batch, build the hook schedule, register,
    and verify the soft-token writes hit the K-pad positions on each row."""

    def test_single_token_writes_per_row_position(self):
        embed_dim = 16
        st = SoftToken(embed_dim=embed_dim, n_tokens=2)
        module = nn.Linear(8, embed_dim)

        # Record 0 has K=2 pad. Record 1 has K=2 pad.
        r0 = _rec([1, 2], [0, 0], [3, 4])           # total 6
        r1 = _rec([1, 2, 3, 4, 5], [0, 0], [6])      # total 8
        batch = assemble_pass_batch(
            [r0, r1], pad_token_id=999, device="cpu",
        )
        schedule = build_hook_schedule(batch.soft_token_positions)
        assert schedule == batch.soft_token_positions

        handle = st.apply_hook(module, schedule)
        try:
            x = torch.randn(2, batch.input_ids.shape[1], 8)
            output = module(x)
            # Both rows' K-blocks (length 2) should carry st.embedding.
            for k in range(2):
                pos = batch.soft_token_positions[k][0]
                assert torch.allclose(
                    output[k, pos:pos + 2, :], st.embedding.detach()
                )
        finally:
            handle.remove()

class TestLengthBucketedShuffle:
    """Tests for the length-bucketed sampler. Returns a permutation of
    the input where each batch_size-window has minimal length variance,
    while preserving global randomness via initial + bucket-order
    shuffles."""

    def test_returns_permutation(self):
        import random as _r
        _r.seed(0)
        items = list(range(20))
        out = length_bucketed_shuffle(
            items, batch_size=4, length_key=lambda x: x,
        )
        assert sorted(out) == sorted(items)
        assert len(out) == len(items)

    def test_within_minibatch_lengths_are_close(self):
        """When items exactly fill one bucket, the within-minibatch
        spread is bounded exactly by ``batch_size - 1`` regardless of
        seed: the bucket-sort lays out 16 consecutive integers as
        [1..4], [5..8], [9..12], [13..16]. Single-bucket avoids the
        orphan-minibatch edge case where a partial second bucket has
        too few items to constrain the spread."""
        import random as _r
        items = list(range(1, 17))  # exactly one bucket at B=4, mult=4
        batch_size = 4
        for seed in range(10):  # robust across seeds
            _r.seed(seed)
            out = length_bucketed_shuffle(
                items, batch_size=batch_size, length_key=lambda x: x,
            )
            for i in range(0, len(out), batch_size):
                mb = out[i:i + batch_size]
                assert max(mb) - min(mb) == batch_size - 1, (
                    f"seed={seed}: minibatch at {i} = {mb} has spread "
                    f"{max(mb) - min(mb)}; expected {batch_size - 1}"
                )

    def test_multi_bucket_reduces_mean_spread_vs_random(self):
        """Across many seeds and multiple full buckets, bucketing
        produces meaningfully smaller mean within-minibatch length
        spread than plain random shuffling. Probabilistic bound, but
        aggregated over 20 seeds × 16 minibatches it's stable."""
        import random as _r
        items = list(range(1, 65))  # 64 items = 4 full buckets at B=4, mult=4
        batch_size = 4
        bucketed_spreads: list[int] = []
        random_spreads: list[int] = []
        for seed in range(20):
            _r.seed(seed)
            out = length_bucketed_shuffle(
                items, batch_size=batch_size, length_key=lambda x: x,
            )
            for i in range(0, len(out), batch_size):
                mb = out[i:i + batch_size]
                bucketed_spreads.append(max(mb) - min(mb))
            _r.seed(seed)
            shuffled = list(items)
            _r.shuffle(shuffled)
            for i in range(0, len(shuffled), batch_size):
                mb = shuffled[i:i + batch_size]
                random_spreads.append(max(mb) - min(mb))
        bucketed_mean = sum(bucketed_spreads) / len(bucketed_spreads)
        random_mean = sum(random_spreads) / len(random_spreads)
        # Random over 64 items, B=4: mean spread ~38.
        # Bucketed: ~6 (≈ 64/16 × (B-1)).
        assert bucketed_mean < random_mean / 3, (
            f"bucketed mean spread {bucketed_mean:.1f} not meaningfully "
            f"smaller than random {random_mean:.1f}"
        )

    def test_bucket_size_one_equals_random_shuffle(self):
        # batch_size=1 should short-circuit to plain shuffle.
        import random as _r
        _r.seed(7)
        items = list(range(20))
        out = length_bucketed_shuffle(
            items, batch_size=1, length_key=lambda x: x,
        )
        # Should be a valid permutation but NOT sorted
        # (almost certainly not at seed 7 with 20 items).
        assert sorted(out) == sorted(items)
        assert out != sorted(items)

    def test_seed_determinism(self):
        import random as _r
        items = list(range(30))
        _r.seed(123)
        out1 = length_bucketed_shuffle(
            items, batch_size=4, length_key=lambda x: x,
        )
        _r.seed(123)
        out2 = length_bucketed_shuffle(
            items, batch_size=4, length_key=lambda x: x,
        )
        assert out1 == out2

    def test_handles_partial_last_bucket(self):
        # 23 items, bucket_size = 4*4=16 -> buckets of [16, 7].
        import random as _r
        _r.seed(0)
        items = list(range(23))
        out = length_bucketed_shuffle(
            items, batch_size=4, length_key=lambda x: x,
        )
        assert sorted(out) == sorted(items)

    def test_rejects_invalid_args(self):
        with pytest.raises(ValueError, match="batch_size"):
            length_bucketed_shuffle([1, 2], 0, length_key=lambda x: x)
        with pytest.raises(ValueError, match="bucket_multiplier"):
            length_bucketed_shuffle(
                [1, 2], 2, length_key=lambda x: x, bucket_multiplier=0,
            )

    def test_uses_length_key_not_item_value(self):
        """Items are dicts; length_key extracts the relevant field.
        Single-bucket setup so the spread bound is exact regardless of
        seed (same reasoning as test_within_minibatch_lengths_are_close)."""
        import random as _r
        items = [{"id": i, "len": 100 - i} for i in range(16)]
        for seed in range(10):
            _r.seed(seed)
            out = length_bucketed_shuffle(
                items, batch_size=4, length_key=lambda x: x["len"],
            )
            for i in range(0, len(out), 4):
                mb_lens = [x["len"] for x in out[i:i + 4]]
                assert max(mb_lens) - min(mb_lens) == 3, (
                    f"seed={seed}: minibatch at {i} = {mb_lens} has "
                    f"spread {max(mb_lens) - min(mb_lens)}; expected 3 "
                    f"(length_key not used correctly)"
                )


class TestHookOnAssembledBatchGradients:
    """Gradient-flow integration test (kept separate from the forward-
    only tests above so each class has a focused intent)."""

    def test_soft_token_gradient_reaches_embedding(self):
        """End-to-end gradient flow through the full pipeline:
        assemble -> SoftToken.apply_hook -> forward (tiny dummy LM) ->
        per_row_masked_ce -> backward. Verifies that the soft token's
        embedding receives non-zero gradient while the frozen model
        parameters receive none.

        This is the load-bearing path the trainer relies on — if a hook
        clone-write breaks the autograd graph, the token wouldn't update.
        """
        embed_dim = 16
        vocab_size = 32
        soft_token = SoftToken(embed_dim=embed_dim, n_tokens=1)

        # Tiny dummy LM: embed -> linear -> logits. Standing in for the
        # real model.forward(input_ids) -> outputs.logits.
        embed_layer = nn.Embedding(vocab_size, embed_dim)
        lm_head = nn.Linear(embed_dim, vocab_size)
        # Freeze everything except the soft token to mirror the trainer.
        for p in embed_layer.parameters():
            p.requires_grad_(False)
        for p in lm_head.parameters():
            p.requires_grad_(False)

        r0 = _rec([1, 2], [0], [3, 4, 5])      # total 6
        r1 = _rec([1, 2, 3, 4], [0], [5, 6])   # total 7
        batch = assemble_pass_batch(
            [r0, r1], pad_token_id=0, device="cpu",
        )
        schedule = build_hook_schedule(batch.soft_token_positions)
        handle = soft_token.apply_hook(embed_layer, schedule)
        try:
            embedded = embed_layer(batch.input_ids)
            logits = lm_head(embedded)
            ce = per_row_masked_ce(logits, batch.input_ids, batch.loss_mask)
            ce.mean().backward()
        finally:
            handle.remove()

        grad = soft_token.embedding.grad
        assert grad is not None, "soft token received no gradient"
        assert grad.abs().sum().item() > 0, (
            "soft token has an all-zero gradient"
        )
        assert lm_head.weight.grad is None
        assert embed_layer.weight.grad is None


class TestAssertCleanContinuationsHaveTerminator:
    """Stale-data guard for the retain_clean answer-bundling format change.

    Before the change, ``clean_continuation`` was a CoT tail with no
    ``</think>`` and was tokenized with ``append_eos=False``. After,
    it is ``tail + "</think>\\n\\n" + answer`` with ``append_eos=True``.
    Old ``train.jsonl`` files run against the new code would silently
    train EOS at the end of a mid-CoT sentence — bad signal. The guard
    catches that loud.
    """

    def test_passes_when_all_continuations_have_terminator(self):
        examples = [
            {"clean_continuation": "tail.</think>\n\nanswer."},
            {"clean_continuation": "another tail</think>\n\nfine."},
        ]
        # Should not raise.
        assert_clean_continuations_have_terminator(examples, "fixture.jsonl")

    def test_raises_when_any_continuation_lacks_terminator(self):
        examples = [
            {"clean_continuation": "fine.</think>\n\nfoo"},
            {"clean_continuation": "stale tail with no terminator."},
        ]
        with pytest.raises(SystemExit, match="without `</think>`"):
            assert_clean_continuations_have_terminator(examples, "fixture.jsonl")

    def test_raises_when_continuation_is_missing(self):
        # A record without the key at all is treated as missing terminator.
        examples = [{"clean_continuation": "ok</think>\n\n"}, {}]
        with pytest.raises(SystemExit, match="without `</think>`"):
            assert_clean_continuations_have_terminator(examples, "fixture.jsonl")

    def test_raises_when_continuation_is_none(self):
        examples = [{"clean_continuation": None}]
        with pytest.raises(SystemExit, match="without `</think>`"):
            assert_clean_continuations_have_terminator(examples, "fixture.jsonl")

    def test_passes_on_empty_examples(self):
        # No records → nothing to validate. Caller decides what to do
        # with an empty training set elsewhere.
        assert_clean_continuations_have_terminator([], "fixture.jsonl")

    def test_error_message_includes_data_path(self):
        examples = [{"clean_continuation": "no terminator"}]
        with pytest.raises(SystemExit, match=r"path/to/train\.jsonl"):
            assert_clean_continuations_have_terminator(
                examples, "path/to/train.jsonl"
            )


class TestMakeLrScheduler:
    """LR scheduler factory for the soft-token training loop."""

    def _make_optimizer(self, lr=1e-3):
        # Tiny dummy parameter — the optimizer just needs SOMETHING with grad.
        param = torch.nn.Parameter(torch.zeros(2))
        return torch.optim.Adam([param], lr=lr)

    def test_none_returns_none(self):
        opt = self._make_optimizer()
        sched = make_lr_scheduler(opt, schedule="none", epochs=10,
                                  lr=1e-3, lr_min_factor=0.0)
        assert sched is None

    def test_cosine_returns_scheduler(self):
        opt = self._make_optimizer()
        sched = make_lr_scheduler(opt, schedule="cosine", epochs=10,
                                  lr=1e-3, lr_min_factor=0.0)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_starts_at_lr(self):
        # Before stepping, optimizer's lr is the initial value.
        opt = self._make_optimizer(lr=1e-3)
        make_lr_scheduler(opt, schedule="cosine", epochs=10,
                          lr=1e-3, lr_min_factor=0.0)
        assert opt.param_groups[0]["lr"] == 1e-3

    def test_cosine_decays_to_eta_min_at_T_max(self):
        # After T_max steps, lr should be exactly eta_min.
        opt = self._make_optimizer(lr=1e-3)
        sched = make_lr_scheduler(opt, schedule="cosine", epochs=10,
                                  lr=1e-3, lr_min_factor=0.1)
        assert sched is not None
        for _ in range(10):
            sched.step()
        assert torch.isclose(
            torch.tensor(opt.param_groups[0]["lr"]),
            torch.tensor(1e-3 * 0.1),
            atol=1e-9,
        )

    def test_cosine_decays_to_zero_when_factor_zero(self):
        opt = self._make_optimizer(lr=1e-3)
        sched = make_lr_scheduler(opt, schedule="cosine", epochs=10,
                                  lr=1e-3, lr_min_factor=0.0)
        assert sched is not None
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == 0.0

    def test_cosine_monotone_decreasing(self):
        # Cosine annealing without restart is monotone non-increasing.
        opt = self._make_optimizer(lr=1e-3)
        sched = make_lr_scheduler(opt, schedule="cosine", epochs=20,
                                  lr=1e-3, lr_min_factor=0.0)
        assert sched is not None
        prev_lr = opt.param_groups[0]["lr"]
        for _ in range(20):
            sched.step()
            cur_lr = opt.param_groups[0]["lr"]
            assert cur_lr <= prev_lr + 1e-12
            prev_lr = cur_lr

    def test_unknown_schedule_raises(self):
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            make_lr_scheduler(opt, schedule="exponential", epochs=10,
                              lr=1e-3, lr_min_factor=0.0)

    def test_zero_epochs_raises(self):
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="epochs must be > 0"):
            make_lr_scheduler(opt, schedule="cosine", epochs=0,
                              lr=1e-3, lr_min_factor=0.0)

    def test_min_factor_out_of_range_raises(self):
        opt = self._make_optimizer()
        with pytest.raises(ValueError, match="lr_min_factor must be"):
            make_lr_scheduler(opt, schedule="cosine", epochs=10,
                              lr=1e-3, lr_min_factor=-0.1)
        with pytest.raises(ValueError, match="lr_min_factor must be"):
            make_lr_scheduler(opt, schedule="cosine", epochs=10,
                              lr=1e-3, lr_min_factor=1.5)
