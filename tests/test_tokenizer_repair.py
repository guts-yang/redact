"""Tests for the byte-level BPE tokenizer repair in HFModel.

Pinned to the LRM-target tokenizer because that's the production model
exhibiting the regression. Skips if the tokenizer can't be downloaded.
"""
from tokenizers.pre_tokenizers import Metaspace, Sequence as PreTokSequence
from tokenizers.decoders import ByteLevel as DecoderByteLevel
import pytest
from transformers import AutoTokenizer

from redact.model.model import _maybe_repair_byte_level_tokenizer


@pytest.fixture
def raw_tokenizer():
    """A freshly-loaded LRM-target tokenizer in its as-shipped (regressed)
    state. Function-scope so each test gets an unrepaired instance —
    repair mutates the backend in place, so a module-scope fixture would
    silently couple test order.
    """
    return AutoTokenizer.from_pretrained("sangyon/LRM-target")


@pytest.fixture
def repaired_tokenizer(raw_tokenizer):
    _maybe_repair_byte_level_tokenizer(raw_tokenizer)
    return raw_tokenizer


def test_detects_and_fires_on_regression(raw_tokenizer):
    backend = raw_tokenizer.backend_tokenizer
    assert isinstance(backend.pre_tokenizer, Metaspace), \
        "expected Metaspace pre_tokenizer in the regressed state"
    applied = _maybe_repair_byte_level_tokenizer(raw_tokenizer)
    assert applied is True
    assert isinstance(backend.pre_tokenizer, PreTokSequence)
    assert isinstance(backend.decoder, DecoderByteLevel)


def test_idempotent(raw_tokenizer):
    first = _maybe_repair_byte_level_tokenizer(raw_tokenizer)
    second = _maybe_repair_byte_level_tokenizer(raw_tokenizer)
    assert first is True
    assert second is False


def test_encode_round_trips_with_spaces(repaired_tokenizer):
    text = "Can you provide a brief about a book?"
    ids = repaired_tokenizer.encode(text, add_special_tokens=False)
    assert repaired_tokenizer.decode(ids) == text


def test_encode_matches_4_43_byte_for_byte(repaired_tokenizer):
    # IDs cross-validated against transformers==4.43.0 + tokenizers==0.19.1
    # in the rtofu environment.
    expected = [9906, 1917, 13, 1115, 374, 264, 1296, 13]
    actual = repaired_tokenizer.encode(
        "Hello world. This is a test.", add_special_tokens=False,
    )
    assert actual == expected


def test_decode_of_g_prefixed_tokens_yields_spaces(repaired_tokenizer):
    # 22691 = 'ĠHello', 1917 = 'Ġworld'
    assert repaired_tokenizer.decode([22691, 1917]) == " Hello world"


def test_special_tokens_unchanged(repaired_tokenizer):
    # Vocabulary IDs for chat-template specials must survive the repair.
    assert repaired_tokenizer.convert_tokens_to_ids("<｜begin▁of▁sentence｜>") == 128000
    assert repaired_tokenizer.convert_tokens_to_ids("<｜User｜>") == 128011
    assert repaired_tokenizer.convert_tokens_to_ids("<｜Assistant｜>") == 128012
    assert repaired_tokenizer.convert_tokens_to_ids("<think>") == 128013
    assert repaired_tokenizer.convert_tokens_to_ids("</think>") == 128014


def test_full_chat_prompt_round_trip(repaired_tokenizer):
    prompt = "<｜User｜>What is the full name of the author?<｜Assistant｜>"
    ids = repaired_tokenizer.encode(prompt, add_special_tokens=True)
    assert repaired_tokenizer.decode(ids, skip_special_tokens=False) == \
        "<｜begin▁of▁sentence｜>" + prompt


def test_tokenize_returns_g_prefixed_word_pieces(repaired_tokenizer):
    # TokenEntropy in redact.evaluator.token_entropy uses tokenizer.tokenize();
    # the regression silently corrupts those scores by returning despaced
    # multi-word fragments. This test guards against that hazard.
    tokens = repaired_tokenizer.tokenize("Hello world. This is a test.")
    assert tokens == [
        "Hello", "Ġworld", ".", "ĠThis", "Ġis", "Ġa", "Ġtest", ".",
    ]


def test_round_trip_with_leading_whitespace(repaired_tokenizer):
    # Edge case for the ByteLevel decoder's add_prefix_space=True option:
    # leading-whitespace inputs are the most likely boundary failure mode.
    text = "  Hello world"
    ids = repaired_tokenizer.encode(text, add_special_tokens=False)
    # Round-trip preserves the run of leading spaces.
    assert repaired_tokenizer.decode(ids) == text


def test_skips_legitimate_byte_level_tokenizer():
    # gpt2 ships with the canonical ByteLevel pre_tokenizer + decoder, so
    # the regression heuristic must skip it (no Metaspace pre_tokenizer).
    tok = AutoTokenizer.from_pretrained("gpt2")
    backend_before = tok.backend_tokenizer.to_str()
    applied = _maybe_repair_byte_level_tokenizer(tok)
    assert applied is False
    assert tok.backend_tokenizer.to_str() == backend_before


def test_skips_legitimate_sentencepiece_tokenizer():
    # A SentencePiece tokenizer has Metaspace pre_tokenizer but its vocab
    # uses ▁ markers, not Ġ — so the regression heuristic must skip it.
    # Using gpt2-medium would reuse the byte-level case above; instead
    # synthesize the negative case by stripping Ġ from a Metaspace
    # tokenizer so convert_tokens_to_ids("ĠHello") returns the unk id.
    # In practice any HF SentencePiece tokenizer hits this branch
    # (e.g. meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.1),
    # but they're gated; reproduce the discriminating condition here.
    tok = AutoTokenizer.from_pretrained("sangyon/LRM-target")
    # Pretend Ġ-prefixed tokens are missing from vocab (the sentinel
    # condition for a non-byte-level Metaspace tokenizer).
    original_convert = tok.convert_tokens_to_ids

    def fake_convert(tokens):
        if tokens == "ĠHello":
            return tok.unk_token_id
        return original_convert(tokens)

    tok.convert_tokens_to_ids = fake_convert  # type: ignore
    applied = _maybe_repair_byte_level_tokenizer(tok)
    assert applied is False
