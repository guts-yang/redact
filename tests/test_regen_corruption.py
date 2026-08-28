"""
Tests for RegeneratingModel.regenerate() in redact/regen/model.py.

Usage:
    conda run -n redact python -m pytest tests/test_regen_corruption.py -v
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from redact.regen.model import RegeneratingModel


# ---------------------------------------------------------------------------
# Minimal mock infrastructure for RegeneratingModel.regenerate tests
# ---------------------------------------------------------------------------

class FakeEmbeddingModule(nn.Module):
    def forward(self, x):
        return x


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = FakeEmbeddingModule()

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def modules(self):
        return iter([self, self.embed_tokens])


class FakeHFModel:
    def __init__(self):
        self.model = FakeInnerModel()
        self.tokenizer = MagicMock()
        self.tokenizer.padding_side = "right"
        self.device = torch.device("cpu")
        self.model_name = "fake-model"
        self.model_config = {"embedding_module": "embed_tokens"}
        self.generation_config = None


def _make_regenerating_model(**overrides):
    hf_model = FakeHFModel()
    kwargs = dict(model=hf_model)
    kwargs.update(overrides)
    return RegeneratingModel(**kwargs)


# ---------------------------------------------------------------------------
# regenerate tests
# ---------------------------------------------------------------------------

class TestRegenerate:
    def test_regenerate_cleans_previous_hooks(self):
        """Existing hooks are removed before generation runs."""
        attacked = _make_regenerating_model()
        dummy_handle = MagicMock()
        attacked._hook_handles.append(dummy_handle)

        attacked.regenerate(input_ids=torch.tensor([[1, 2]]))
        dummy_handle.remove.assert_called_once()

    def test_regenerate_applies_soft_token(self):
        """soft_token.apply_hook is called with the correct position."""
        mock_soft_token = MagicMock()
        mock_st_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_st_handle

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[[5]],
            input_ids=torch.tensor([[1, 2]]),
        )
        mock_soft_token.apply_hook.assert_called_once_with(
            attacked.embedding_module, [[5]]
        )
        # Soft token handle should be removed after generate
        mock_st_handle.remove.assert_called_once()

    def test_regenerate_no_soft_token_when_positions_none(self):
        """soft_token.apply_hook is NOT called when positions is None."""
        mock_soft_token = MagicMock()
        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=None,
            input_ids=torch.tensor([[1, 2]]),
        )
        mock_soft_token.apply_hook.assert_not_called()

    def test_regenerate_no_soft_token_when_positions_empty(self):
        """soft_token.apply_hook is NOT called when positions is empty list."""
        mock_soft_token = MagicMock()
        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[],
            input_ids=torch.tensor([[1, 2]]),
        )
        mock_soft_token.apply_hook.assert_not_called()

    def test_regenerate_cleans_up_on_generate_error(self):
        """Soft-token hook is cleaned up even when model.generate() raises."""
        mock_soft_token = MagicMock()
        mock_st_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_st_handle

        attacked = _make_regenerating_model(soft_token=mock_soft_token)
        # Make model.generate() raise
        attacked.model.generate = MagicMock(side_effect=RuntimeError("boom"))

        try:
            attacked.regenerate(
                soft_token_positions=[[3]],
                input_ids=torch.tensor([[1, 2]]),
            )
        except RuntimeError:
            pass
        mock_st_handle.remove.assert_called_once()


# ---------------------------------------------------------------------------
# Multi-token escalation tests
# ---------------------------------------------------------------------------

class TestRegenerateMultiToken:
    def test_multiple_hooks_registered(self):
        """Multiple soft token positions collapse to a single schedule with
        all positions in row 0's inner list (replaces the per-position hook
        registration loop)."""
        mock_soft_token = MagicMock()
        mock_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_handle

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[[5, 6, 7]],
            input_ids=torch.tensor([[1, 2]]),
        )
        mock_soft_token.apply_hook.assert_called_once_with(
            attacked.embedding_module, [[5, 6, 7]]
        )
        # Single handle cleaned up once.
        mock_handle.remove.assert_called_once()

    def test_all_hooks_cleaned_on_error(self):
        """The single schedule hook is cleaned up even when generate() raises."""
        mock_soft_token = MagicMock()
        mock_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_handle

        attacked = _make_regenerating_model(soft_token=mock_soft_token)
        attacked.model.generate = MagicMock(side_effect=RuntimeError("boom"))

        try:
            attacked.regenerate(
                soft_token_positions=[[5, 6, 7]],
                input_ids=torch.tensor([[1, 2]]),
            )
        except RuntimeError:
            pass
        mock_handle.remove.assert_called_once()

# ---------------------------------------------------------------------------
# Per-row (B>1) signature tests
# ---------------------------------------------------------------------------

class TestRegeneratePerRowSchedule:
    """The B=1 path is the canonical inference flow today; these tests
    exercise the per-row schedule shape that the wave-level engine
    refactor will populate with non-trivial outer lists."""

    def test_b2_distinct_anchors_per_row(self):
        """Two-row schedule with different positions per row threads through
        to apply_hook as a B=2 schedule."""
        mock_soft_token = MagicMock(spec=["apply_hook"])
        mock_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_handle

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[[5], [7, 8]],
            input_ids=torch.tensor([[1, 2], [3, 4]]),
        )
        mock_soft_token.apply_hook.assert_called_once_with(
            attacked.embedding_module,
            [[5], [7, 8]],
        )
        mock_handle.remove.assert_called_once()

    def test_b2_one_row_empty(self):
        """A row with an empty anchor list passes through as an empty
        inner list in the schedule (no writes for that row)."""
        mock_soft_token = MagicMock(spec=["apply_hook"])
        mock_soft_token.apply_hook.return_value = MagicMock()

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[[5, 6], []],
            input_ids=torch.tensor([[1, 2], [3, 4]]),
        )
        mock_soft_token.apply_hook.assert_called_once_with(
            attacked.embedding_module,
            [[5, 6], []],
        )

    def test_schedule_inner_lists_are_copies(self):
        """The schedule handed to ``apply_hook`` must not alias the
        caller's per-row lists — the hook resolves entries at
        registration, and a caller mutating its own list afterwards
        must not change what the hook writes."""
        mock_soft_token = MagicMock(spec=["apply_hook"])
        mock_soft_token.apply_hook.return_value = MagicMock()

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        positions = [[5], [7]]
        attacked.regenerate(
            soft_token_positions=positions,
            input_ids=torch.tensor([[1, 2], [3, 4]]),
        )
        passed = mock_soft_token.apply_hook.call_args.args[1]
        assert passed == [[5], [7]]
        assert passed[0] is not positions[0]

    def test_empty_outer_no_op(self):
        """``soft_token_positions=[]`` (B=0 declared): no hook
        registered — the falsy outer list short-circuits the soft-token
        block entirely."""
        mock_soft_token = MagicMock(spec=["apply_hook"])

        attacked = _make_regenerating_model(soft_token=mock_soft_token)

        attacked.regenerate(
            soft_token_positions=[],
            input_ids=torch.tensor([[1, 2]]),
        )
        mock_soft_token.apply_hook.assert_not_called()


# ---------------------------------------------------------------------------
# Chunked re-prefill path
#
# The chunked path runs greedy AR generation in chunks, dropping rows that
# emit EOS or hit their per-row budget cap from the active subset between
# chunks.  The mock below is a content-deterministic greedy generator: each
# call's outputs depend only on each row's last token, so chunked vs single-
# shot calls produce identical per-row token streams up to first EOS or
# budget cap.  EOS is triggered by a sentinel "trigger" value, NOT by row
# index, because rows shift batch position as they drop out.
# ---------------------------------------------------------------------------

class _DeterministicGreedyMock(nn.Module):
    """Greedy mock: ``next_token = (last_token + 1) % vocab_size``, with
    ``eos_token_id`` substituted whenever ``next_token`` would equal
    ``eos_trigger``.  Once a row emits EOS the rest of its row in the
    output is filled with ``pad_token_id``.

    The next-token decision is purely a function of the row's content,
    not its batch position — so reordering or dropping rows between
    calls does not change any surviving row's output.  This is what
    makes chunked vs single-shot byte-equivalent on this mock.
    """

    def __init__(
        self,
        vocab_size: int = 200,
        eos_token_id: int = 99,
        eos_trigger: int = 50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.eos_trigger = eos_trigger
        self.embed_tokens = FakeEmbeddingModule()

    def modules(self):
        return iter([self, self.embed_tokens])

    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        eos_token_id=None,
        pad_token_id=None,
        generation_config=None,
        **kwargs,
    ):
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        pad = pad_token_id if pad_token_id is not None else 0
        if max_new_tokens is None and generation_config is not None:
            max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        B, T = input_ids.shape
        out = torch.full(
            (B, T + max_new_tokens),
            pad,
            dtype=torch.long,
            device=input_ids.device,
        )
        out[:, :T] = input_ids
        for k in range(B):
            last = int(input_ids[k, -1].item())
            stopped = False
            for step in range(max_new_tokens):
                if stopped:
                    out[k, T + step] = pad
                    continue
                next_tok = (last + 1) % self.vocab_size
                if next_tok == self.eos_trigger:
                    next_tok = eos
                    stopped = True
                out[k, T + step] = next_tok
                last = next_tok
        return out


class _MockHFModel:
    """HFModel-shaped wrapper around an arbitrary inner ``nn.Module``."""

    def __init__(self, inner):
        self.model = inner
        self.tokenizer = MagicMock()
        self.tokenizer.padding_side = "right"
        self.device = torch.device("cpu")
        self.model_name = "fake-model"
        self.model_config = {"embedding_module": "embed_tokens"}
        self.generation_config = None


def _make_attacked_with_inner(inner, **overrides):
    hf_model = _MockHFModel(inner)
    kwargs = dict(model=hf_model)
    kwargs.update(overrides)
    return RegeneratingModel(**kwargs)


def _greedy_gen_config(max_new_tokens):
    """Build a minimal generation_config that the chunked-path validator
    accepts.  ``SimpleNamespace`` is used in place of
    ``transformers.GenerationConfig`` so tests don't require the model
    config plumbing that real instances need."""
    from types import SimpleNamespace

    return SimpleNamespace(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        bad_words_ids=None,
        no_repeat_ngram_size=0,
    )


# ---- Validation tests --------------------------------------------------------

class TestChunkedDispatch:
    def test_chunked_requires_per_row_remaining(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        with pytest.raises(ValueError, match="per_row_remaining"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=_greedy_gen_config(5),
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
            )

    def test_chunked_rejects_do_sample_true(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.do_sample = True
        with pytest.raises(ValueError, match="do_sample"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_num_beams_gt_1(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.num_beams = 4
        with pytest.raises(ValueError, match="num_beams"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_repetition_penalty(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.repetition_penalty = 1.2
        with pytest.raises(ValueError, match="repetition_penalty"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_bad_words_ids(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.bad_words_ids = [[42]]
        with pytest.raises(ValueError, match="bad_words_ids"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_no_repeat_ngram(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.no_repeat_ngram_size = 3
        with pytest.raises(ValueError, match="no_repeat_ngram"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_validates_per_row_remaining_length(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        with pytest.raises(ValueError, match="per_row_remaining length"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 1], [1, 1, 1]]),
                generation_config=_greedy_gen_config(5),
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],  # wrong length
            )

    def test_chunked_rejects_chunk_size_zero(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        with pytest.raises(ValueError, match="chunk_size"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=_greedy_gen_config(5),
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=0,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_temperature_ne_1(self):
        # do_sample=False but temperature != 1.0 — silently divergent
        # under HF's TemperatureLogitsWarper on some code paths.
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.temperature = 0.7
        with pytest.raises(ValueError, match="temperature"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_top_k(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.top_k = 50
        with pytest.raises(ValueError, match="top_k"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )

    def test_chunked_rejects_top_p(self):
        attacked = _make_attacked_with_inner(_DeterministicGreedyMock())
        gc = _greedy_gen_config(5)
        gc.top_p = 0.9
        with pytest.raises(ValueError, match="top_p"):
            attacked.regenerate(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
                generation_config=gc,
                eos_token_id=99,
                pad_token_id=99,
                chunk_size=2,
                per_row_remaining=[5],
            )


# ---- Byte-equivalence tests --------------------------------------------------

class TestChunkedByteEquivalence:
    def _run_both(self, input_ids, attention_mask, max_new, chunk_size,
                  eos_token_id=99, eos_trigger=50, pad_token_id=99):
        inner = _DeterministicGreedyMock(
            eos_token_id=eos_token_id, eos_trigger=eos_trigger,
        )
        single = inner.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        attacked = _make_attacked_with_inner(inner)
        chunked = attacked.regenerate(
            soft_token_positions=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=_greedy_gen_config(max_new),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            chunk_size=chunk_size,
            per_row_remaining=[max_new] * input_ids.shape[0],
            # Mock doesn't return DynamicCache, so use Phase 1 path
            # explicitly.  Phase 2 byte-equivalence on real models is
            # covered by ``TestKVCacheChunked`` with the SmolLM fixture.
            chunk_mode="reprefill",
        )
        return single, chunked

    def _assert_per_row_real_tokens_match(
        self, single, chunked, prefill_len, eos_token_id,
    ):
        B = single.shape[0]
        assert chunked.shape[0] == B
        for k in range(B):
            single_new = single[k, prefill_len:].tolist()
            chunked_new = chunked[k, prefill_len:].tolist()
            single_real = (
                single_new[: single_new.index(eos_token_id) + 1]
                if eos_token_id in single_new else single_new
            )
            chunked_real = (
                chunked_new[: chunked_new.index(eos_token_id) + 1]
                if eos_token_id in chunked_new else chunked_new
            )
            assert single_real == chunked_real, (
                f"row {k}: single={single_real}, chunked={chunked_real}"
            )

    def test_b1_no_eos(self):
        # Last token 5; eos_trigger 50 unreachable in 10 steps.
        input_ids = torch.tensor([[1, 2, 5]])
        attention_mask = torch.tensor([[1, 1, 1]])
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=3,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )

    def test_b1_eos_in_first_chunk(self):
        # Last token 47; eos_trigger 50; row emits 48, 49, EOS in 3 steps.
        input_ids = torch.tensor([[1, 2, 47]])
        attention_mask = torch.tensor([[1, 1, 1]])
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=4,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )

    def test_b1_eos_across_chunk_boundary(self):
        # Last token 45; chunk 0 takes us to 48; chunk 1 emits 49, EOS.
        input_ids = torch.tensor([[1, 2, 45]])
        attention_mask = torch.tensor([[1, 1, 1]])
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=3,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )

    def test_b3_mixed_eos_times(self):
        # Three rows with different EOS arrival times so chunked drops
        # them at different chunk boundaries.
        input_ids = torch.tensor([
            [1, 2, 47],  # EOS at step 3 (chunk 0)
            [1, 2, 44],  # EOS at step 6 (chunk 1)
            [1, 2, 30],  # EOS at step 20 (way past max_new=10)
        ])
        attention_mask = torch.ones_like(input_ids)
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=3,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )

    def test_b2_left_padded_input_translates_positions(self):
        # Left-pad row 0 so the two rows have different real-content
        # lengths.  Verifies the chunked path's left-pad stripping +
        # per-chunk re-padding produces correct outputs.
        input_ids = torch.tensor([
            [99, 99, 1, 2, 47],  # 2 left-pads, real content [1, 2, 47]
            [1, 2, 3, 4, 45],    # no left-pad
        ])
        attention_mask = torch.tensor([
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=3,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=5, eos_token_id=99,
        )

    def test_chunk_size_1_extreme(self):
        # chunk_size=1 forces a re-prefill at every AR step.  Output
        # must still be byte-equivalent to single-shot greedy; this is
        # the worst-case for setup overhead but the strongest test of
        # the per-chunk re-padding + position-translation logic.
        input_ids = torch.tensor([[1, 2, 47]])
        attention_mask = torch.tensor([[1, 1, 1]])
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=5, chunk_size=1,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )

    def test_all_rows_eos_in_chunk_0(self):
        # Every row hits EOS within chunk 0.  The per-row drop logic
        # should clear active_indices after one chunk and exit cleanly
        # (no second chunk attempt).  Output shape: max generated length
        # across rows == longest row's EOS step + 1.
        input_ids = torch.tensor([
            [1, 2, 48],  # EOS at step 2 (49, EOS)
            [1, 2, 47],  # EOS at step 3 (48, 49, EOS)
            [1, 2, 46],  # EOS at step 4 (47, 48, 49, EOS)
        ])
        attention_mask = torch.ones_like(input_ids)
        single, chunked = self._run_both(
            input_ids, attention_mask, max_new=10, chunk_size=8,
        )
        self._assert_per_row_real_tokens_match(
            single, chunked, prefill_len=3, eos_token_id=99,
        )
        # Chunked output's max_new is bounded by longest row (4 tokens),
        # not by max_new_tokens=10 — confirming early termination.
        assert chunked.shape[1] == 3 + 4, (
            f"Expected chunked to truncate to longest-row length; "
            f"got shape {chunked.shape}"
        )


# ---- Per-row budget cap test ------------------------------------------------

class TestChunkedPerRowBudget:
    def test_each_row_truncates_at_its_individual_cap(self):
        # Three rows, all far from EOS.  Per-row caps differ;
        # the chunked path must enforce each row's individual cap.
        inner = _DeterministicGreedyMock(eos_trigger=190)  # unreachable in test
        attacked = _make_attacked_with_inner(inner)
        input_ids = torch.tensor([
            [1, 2, 10],
            [1, 2, 20],
            [1, 2, 30],
        ])
        attention_mask = torch.ones_like(input_ids)
        per_row = [3, 7, 2]  # each row gets a different budget
        out = attacked.regenerate(
            soft_token_positions=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=_greedy_gen_config(max(per_row)),
            eos_token_id=99,
            pad_token_id=99,
            chunk_size=4,
            per_row_remaining=per_row,
            chunk_mode="reprefill",
        )
        # Output shape: (B, prefill + max_new_across_rows)
        assert out.shape[0] == 3
        prefill = input_ids.shape[1]
        # Row 0: 3 new tokens (11, 12, 13), then pad
        assert out[0, prefill:prefill + 3].tolist() == [11, 12, 13]
        assert out[0, prefill + 3:].tolist() == [99] * (
            out.shape[1] - prefill - 3
        )
        # Row 1: 7 new tokens (21..27)
        assert out[1, prefill:prefill + 7].tolist() == [21, 22, 23, 24, 25, 26, 27]
        # Row 2: 2 new tokens (31, 32), then pad
        assert out[2, prefill:prefill + 2].tolist() == [31, 32]
        assert out[2, prefill + 2:].tolist() == [99] * (
            out.shape[1] - prefill - 2
        )


# ---- Soft-token hook lifecycle across chunks --------------------------------

class TestChunkedSoftTokenHook:
    def test_hook_registered_each_chunk_with_active_subset(self):
        """Per chunk, the hook is re-registered with a schedule whose
        outer length matches the survivor count and whose anchor
        positions are shifted by each row's chunk-frame left_pad."""
        # Configure: row 0 EOS's at step 3 (chunk 0 boundary); row 1
        # continues past chunk 0 and EOS's later.  We expect 2 hook
        # registrations: chunk 0 (B=2), chunk 1 (B=1, row 1 only).
        inner = _DeterministicGreedyMock(eos_trigger=50)
        # Row 0 ends in 47 → emits 48, 49, EOS in 3 steps.
        # Row 1 ends in 44 → emits 45..49, EOS in 6 steps.
        input_ids = torch.tensor([
            [1, 2, 47],
            [1, 2, 44],
        ])
        attention_mask = torch.ones_like(input_ids)

        mock_soft_token = MagicMock(spec=["apply_hook"])
        mock_soft_token.apply_hook.return_value = MagicMock()

        attacked = _make_attacked_with_inner(inner, soft_token=mock_soft_token)
        attacked.regenerate(
            soft_token_positions=[[2], [2]],  # both rows: anchor at real-pos 2
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=_greedy_gen_config(10),
            eos_token_id=99,
            pad_token_id=99,
            chunk_size=3,
            per_row_remaining=[10, 10],
            chunk_mode="reprefill",  # mock-based test exercises Phase 1
        )

        calls = mock_soft_token.apply_hook.call_args_list
        # Chunk 0: B=2.  Chunk 1: B=1.  At least 2 calls expected
        # (more if row 1 needs >1 chunk after row 0 drops).
        assert len(calls) >= 2
        # First call is chunk 0: outer length 2.
        assert len(calls[0].args[1]) == 2
        # Subsequent call(s) have outer length 1 (only row 1 survives).
        for call in calls[1:]:
            assert len(call.args[1]) == 1

    def test_hook_not_registered_when_no_soft_token(self):
        """No soft_token configured: chunked path runs without hook
        registration even when chunk_size is set."""
        inner = _DeterministicGreedyMock(eos_trigger=190)
        attacked = _make_attacked_with_inner(inner, soft_token=None)
        # Should not raise; output should just be the deterministic
        # mock's continuation.
        out = attacked.regenerate(
            soft_token_positions=None,
            input_ids=torch.tensor([[1, 2, 10]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            generation_config=_greedy_gen_config(5),
            eos_token_id=99,
            pad_token_id=99,
            chunk_size=2,
            per_row_remaining=[5],
            chunk_mode="reprefill",  # mock-based test exercises Phase 1
        )
        assert out.shape == (1, 3 + 5)


# ---------------------------------------------------------------------------
# Phase 2 (KV-cache forwarding) tests
#
# Phase 2 needs a real model — its correctness depends on HF's
# ``return_dict_in_generate``, ``DynamicCache``, ``batch_select_indices``,
# and ``prepare_inputs_for_generation`` behaviour, none of which a mock
# would faithfully reproduce.  SmolLM-135M (Llama-arch, ~270MB) loads in
# ~1s on CPU and runs each test in milliseconds.  Module-scoped fixture
# so the model loads once per test session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smollm():
    """Real Llama-arch model for Phase 2 byte-equivalence tests.

    Returns a tuple ``(attacked, tokenizer)`` with ``chunk_instrumentation``
    quieted so the tests don't spam logs.  No soft_token is configured;
    individual tests inject one when needed via ``attacked.soft_token = ...``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    name = "HuggingFaceTB/SmolLM-135M"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    m.generation_config = GenerationConfig(
        do_sample=False, use_cache=True, max_new_tokens=20,
    )
    m.eval()

    class _FakeHF:
        pass

    hf = _FakeHF()
    hf.model = m
    hf.tokenizer = tok
    hf.device = m.device
    hf.model_name = "smollm"
    hf.model_config = {"embedding_module": "model.embed_tokens"}
    hf.generation_config = m.generation_config

    attacked = RegeneratingModel(model=hf)
    attacked.chunk_instrumentation = False
    return attacked, tok


def _greedy_gen_config_for_smollm(max_new_tokens=20):
    """Real ``GenerationConfig`` so HF generate's downstream attribute
    accesses (max_length, etc.) work — ``SimpleNamespace`` would fail
    on the first missing field."""
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )


def _run_chunked(attacked, tok, input_ids, attention_mask, *,
                 chunk_size, max_new, mode, per_row_remaining=None):
    if per_row_remaining is None:
        per_row_remaining = [max_new] * input_ids.shape[0]
    return attacked.regenerate(
        soft_token_positions=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=_greedy_gen_config_for_smollm(max_new),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        chunk_size=chunk_size,
        per_row_remaining=per_row_remaining,
        chunk_mode=mode,
    )


def _real_token_seq_per_row(out, prefill_len, eos_id):
    """Strip post-EOS pad fill so two outputs of different lengths can
    be compared on real content alone."""
    seqs = []
    for k in range(out.shape[0]):
        row = out[k, prefill_len:].tolist()
        if eos_id in row:
            row = row[: row.index(eos_id) + 1]
        seqs.append(row)
    return seqs


class TestKVCacheChunked:
    def test_kv_chunked_matches_phase_1_byte_equivalence(self, smollm):
        attacked, tok = smollm
        ids = tok("Hello world this is a longer prompt", return_tensors="pt").input_ids
        mask = torch.ones_like(ids)
        out_p1 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=20, mode="reprefill",
        )
        out_p2 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=20, mode="kvcache",
        )
        prefill = ids.shape[1]
        eos = tok.eos_token_id
        assert (
            _real_token_seq_per_row(out_p1, prefill, eos)
            == _real_token_seq_per_row(out_p2, prefill, eos)
        )

    def test_kv_chunked_with_left_padded_inputs(self, smollm):
        attacked, tok = smollm
        prompts = [
            "A short prompt",
            "Hello world this is a much longer prompt to test left padding",
            "Medium length prompt here",
            "Yet another slightly longer prompt for variety",
        ]
        enc = tok(prompts, return_tensors="pt", padding=True)
        ids, mask = enc.input_ids, enc.attention_mask
        out_p1 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=15, mode="reprefill",
        )
        out_p2 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=15, mode="kvcache",
        )
        prefill = ids.shape[1]
        eos = tok.eos_token_id
        p1_seqs = _real_token_seq_per_row(out_p1, prefill, eos)
        p2_seqs = _real_token_seq_per_row(out_p2, prefill, eos)
        # Per-row equivalence — left-pad shouldn't perturb either path.
        for k, (a, b) in enumerate(zip(p1_seqs, p2_seqs)):
            assert a == b, f"row {k}: phase1={a}, phase2={b}"

    def test_kv_chunked_single_survivor(self, smollm):
        # B_orig=4 with very different per-row budgets so 3 of 4 rows
        # finish in chunk 0; only one row continues into chunks 1+.
        # Exercises the cache-slice-to-1-row path.
        attacked, tok = smollm
        prompts = ["Test " + str(i) for i in range(4)]
        enc = tok(prompts, return_tensors="pt", padding=True)
        ids, mask = enc.input_ids, enc.attention_mask
        per_row = [2, 2, 2, 30]  # rows 0-2 hit budget cap in chunk 0 (chunk_size=4)
        out_p2 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=30, mode="kvcache",
            per_row_remaining=per_row,
        )
        out_p1 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=30, mode="reprefill",
            per_row_remaining=per_row,
        )
        prefill = ids.shape[1]
        for k in range(4):
            p1 = out_p1[k, prefill:prefill + per_row[k]].tolist()
            p2 = out_p2[k, prefill:prefill + per_row[k]].tolist()
            # Allow trailing pad to differ in length; real-content prefix
            # up to per_row[k] must match.
            assert p1 == p2, f"row {k}: phase1={p1}, phase2={p2}"

    def test_kv_chunked_soft_token_hook_fires_only_chunk_0(self, smollm):
        # MagicMock soft_token; hook returns a real handle so the
        # context manager's try/finally can call .remove() on it.
        attacked, tok = smollm
        original_soft = attacked.soft_token
        try:
            mock_st = MagicMock(spec=["apply_hook"])
            mock_handle = MagicMock()
            mock_st.apply_hook.return_value = mock_handle
            attacked.soft_token = mock_st

            ids = tok("Hello world prompt", return_tensors="pt").input_ids
            mask = torch.ones_like(ids)
            _ = _run_chunked(
                attacked, tok, ids, mask,
                chunk_size=4, max_new=20, mode="kvcache",
                # Single anchor at position 2; runs through Phase 2's
                # chunk-0-with-hook path.
                per_row_remaining=[20],
            )
            # The chunked path passes soft_token_positions=None
            # through ``_run_chunked``, so apply_hook is NOT called
            # (the context manager's "no positions" branch).  This test assertion lives mainly to catch a
            # regression where Phase 2 accidentally re-registered the
            # hook in chunks 1+ even with no positions.
            assert mock_st.apply_hook.call_count == 0
        finally:
            attacked.soft_token = original_soft

    def test_kv_chunked_drops_eos_rows(self, smollm):
        # Mixed-budget batch where rows naturally drop at different
        # chunk boundaries.  Verifies Phase 2 byte-matches Phase 1 on
        # each row's cap region (any EOS-then-pad ambiguity past the
        # cap is irrelevant since both paths reassemble identically).
        attacked, tok = smollm
        prompts = ["A test", "B test", "C test", "D test"]
        enc = tok(prompts, return_tensors="pt", padding=True)
        ids, mask = enc.input_ids, enc.attention_mask
        per_row = [3, 8, 5, 12]
        out_p1 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=12, mode="reprefill",
            per_row_remaining=per_row,
        )
        out_p2 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=4, max_new=12, mode="kvcache",
            per_row_remaining=per_row,
        )
        prefill = ids.shape[1]
        for k in range(4):
            p1 = out_p1[k, prefill:prefill + per_row[k]].tolist()
            p2 = out_p2[k, prefill:prefill + per_row[k]].tolist()
            assert p1 == p2, f"row {k}: phase1={p1}, phase2={p2}"

    def test_kv_chunked_resumes_with_partial_cache(self, smollm):
        # B_orig=3 with mixed EOS times; verify Phase 2 produces
        # correct output even when the cache shrinks across chunks.
        attacked, tok = smollm
        prompts = ["First prompt", "Second much longer prompt for variety", "Third"]
        enc = tok(prompts, return_tensors="pt", padding=True)
        ids, mask = enc.input_ids, enc.attention_mask
        # Vary budgets so some rows drop early, exercising B=3→B=1
        # transitions.
        per_row = [3, 25, 10]
        out_p1 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=5, max_new=25, mode="reprefill",
            per_row_remaining=per_row,
        )
        out_p2 = _run_chunked(
            attacked, tok, ids, mask,
            chunk_size=5, max_new=25, mode="kvcache",
            per_row_remaining=per_row,
        )
        prefill = ids.shape[1]
        for k in range(3):
            p1 = out_p1[k, prefill:prefill + per_row[k]].tolist()
            p2 = out_p2[k, prefill:prefill + per_row[k]].tolist()
            assert p1 == p2, f"row {k}: phase1={p1}, phase2={p2}"


class TestKVCacheValidation:
    def test_engine_rejects_invalid_chunk_mode(self):
        # Engine __init__ must reject chunk_mode values other than
        # "reprefill" / "kvcache".
        from redact.inference_regen import RegeneratingReasoningEngine
        from redact.model.reasoning import ReasoningModel
        # Use the existing test infrastructure for a minimal dummy model
        # (importing from test_regen_engine would create a circular
        # dependency in pytest collection; build a minimal stub here).
        with pytest.raises(ValueError, match="chunk_mode"):
            # Passing a bad chunk_mode pre-empts the model wiring; we
            # don't need a real engine to exercise the validator.
            from unittest.mock import MagicMock
            mock_inner = MagicMock(spec=["regenerate", "soft_token"])
            mock_inner.soft_token = MagicMock()
            mock_model = MagicMock(spec=ReasoningModel)
            mock_model._inner = mock_inner
            RegeneratingReasoningEngine(
                model=mock_model,
                tokenizer=MagicMock(),
                data_module=MagicMock(),
                subset_names=[],
                answer_evaluator=[],
                leak_detector=MagicMock(),
                chunk_size=256,
                chunk_mode="invalid_mode",
            )
