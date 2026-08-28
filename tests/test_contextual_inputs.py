"""Unit tests for build_contextual_input (k-window context builder).

Uses a real roberta-base tokenizer with the three contextual special
tokens registered, since the builder's correctness hinges on
BPE-leading-space behavior, special-token id lookup, and pair encoding.
"""
from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from redact.regen.contextual_inputs import (
    CONTEXTUAL_SPECIAL_TOKENS,
    CTX_TRUNC,
    TGT_CLOSE,
    TGT_OPEN,
    build_contextual_input,
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("roberta-base")
    tok.add_special_tokens(
        {"additional_special_tokens": list(CONTEXTUAL_SPECIAL_TOKENS)}
    )
    return tok


def _ids(tok, text: str, leading_space: bool = False) -> list[int]:
    """Encode helper mirroring build_contextual_input's BPE handling."""
    if leading_space and text and not text.startswith(" "):
        text = " " + text
    return tok(text, add_special_tokens=False)["input_ids"]


def _layout(tok, ids: list[int]) -> list[str]:
    """Decode token-by-token for human-readable assertion failures."""
    return tok.convert_ids_to_tokens(ids)


# --- Empty / 1-prior / 2-prior: no truncation marker ----------------------

def test_zero_priors_no_marker(tokenizer):
    out = build_contextual_input(
        prompt="Who is Alice?",
        prior_sentences=[],
        target_sentence="Alice was born in 1990.",
        tokenizer=tokenizer,
    )
    assert not out.truncated_context
    ctx_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    assert ctx_id not in out.input_ids


def test_one_prior_kept_no_marker(tokenizer):
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=["First sentence."],
        target_sentence="Target sentence.",
        tokenizer=tokenizer,
    )
    assert not out.truncated_context
    ctx_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    assert ctx_id not in out.input_ids
    # Prior must appear before <TGT>.
    tgt_open_id = tokenizer.convert_tokens_to_ids(TGT_OPEN)
    tgt_close_id = tokenizer.convert_tokens_to_ids(TGT_CLOSE)
    open_idx = out.input_ids.index(tgt_open_id)
    close_idx = out.input_ids.index(tgt_close_id)
    assert open_idx < close_idx
    # Prior content sits between the pair-separator (</s></s>) and <TGT>.
    sep_id = tokenizer.sep_token_id
    sep_positions = [i for i, x in enumerate(out.input_ids) if x == sep_id]
    # First sep ends prompt segment; second is the duplicated pair-sep.
    assert len(sep_positions) >= 3, _layout(tokenizer, out.input_ids)
    # Tokens between second-sep and <TGT> are the prior content.
    prior_ids = out.input_ids[sep_positions[1] + 1 : open_idx]
    assert prior_ids == _ids(tokenizer, "First sentence.", leading_space=True)


def test_two_priors_kept_no_marker(tokenizer):
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=["S1.", "S2."],
        target_sentence="Target.",
        tokenizer=tokenizer,
    )
    assert not out.truncated_context
    ctx_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    assert ctx_id not in out.input_ids


# --- 3+ priors: only last 2 retained, marker fires -----------------------

def test_three_priors_keeps_last_two_with_marker(tokenizer):
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=["Earliest.", "Middle.", "Most recent."],
        target_sentence="Target.",
        tokenizer=tokenizer,
        max_prior_sentences=2,
    )
    assert out.truncated_context
    ctx_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    assert ctx_id in out.input_ids
    # Earliest must NOT appear; the kept 2 must.
    assert (
        _ids(tokenizer, "Earliest.", leading_space=True)[0]
        not in out.input_ids
    )
    middle_first = _ids(tokenizer, "Middle.", leading_space=True)[0]
    recent_first = _ids(tokenizer, "Most recent.", leading_space=True)[0]
    assert middle_first in out.input_ids
    assert recent_first in out.input_ids


def test_max_prior_sentences_zero_drops_all_context(tokenizer):
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=["S1.", "S2."],
        target_sentence="Target.",
        tokenizer=tokenizer,
        max_prior_sentences=0,
    )
    # max=0 means none of the priors are eligible — flagged as truncated
    # (we elided structurally) but no marker fits when budget is < 1?
    # The builder still emits the marker because budget >= 1; test that.
    assert out.truncated_context
    s1_first = _ids(tokenizer, "S1.", leading_space=True)[0]
    assert s1_first not in out.input_ids


def test_max_prior_sentences_four_keeps_all_when_below(tokenizer):
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=["S1.", "S2."],
        target_sentence="Target.",
        tokenizer=tokenizer,
        max_prior_sentences=4,
    )
    # 2 priors is below the cap; no truncation and no marker.
    assert not out.truncated_context


# --- BPE leading-space: priors carry Ġ (encoded as space prefix) ---------

def test_priors_encoded_with_leading_space(tokenizer):
    """Each kept prior should be tokenized with a leading space so its
    first BPE token is a Ġ-prefixed word piece, not a bare-token piece.
    The most reliable check: the prior's encoding here must equal what
    we'd get by encoding " <prior>" directly."""
    prior = "He was born in Tokyo."
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=[prior],
        target_sentence="Target.",
        tokenizer=tokenizer,
    )
    expected_prior_ids = _ids(tokenizer, prior, leading_space=True)
    # The prior content sits inside the input, identified as a contiguous
    # subsequence equal to expected_prior_ids.
    assert _is_subsequence(expected_prior_ids, out.input_ids), (
        f"prior tokens with leading space not found in input. "
        f"Expected: {_layout(tokenizer, expected_prior_ids)}; "
        f"Got input layout: {_layout(tokenizer, out.input_ids)}"
    )


def _is_subsequence(needle: list[int], haystack: list[int]) -> bool:
    if not needle:
        return True
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return True
    return False


# --- Special-tokens validation: missing tokens raise ---------------------

def test_missing_special_token_raises():
    bare = AutoTokenizer.from_pretrained("roberta-base")
    # Specials not added — builder must reject loudly.
    with pytest.raises(ValueError, match="Special token"):
        build_contextual_input(
            prompt="Q", prior_sentences=[], target_sentence="X.",
            tokenizer=bare,
        )


def test_negative_max_prior_sentences_raises(tokenizer):
    with pytest.raises(ValueError, match="max_prior_sentences"):
        build_contextual_input(
            prompt="Q", prior_sentences=[], target_sentence="X.",
            tokenizer=tokenizer, max_prior_sentences=-1,
        )


# --- Budget overflow: target dominant, prompt dominant -------------------

def test_target_clamped_to_budget(tokenizer):
    """Target longer than target_budget gets truncated; markers preserved."""
    long_target = " ".join(["word"] * 200)
    out = build_contextual_input(
        prompt="Q",
        prior_sentences=[],
        target_sentence=long_target,
        tokenizer=tokenizer,
        target_budget=32,
    )
    tgt_open_id = tokenizer.convert_tokens_to_ids(TGT_OPEN)
    tgt_close_id = tokenizer.convert_tokens_to_ids(TGT_CLOSE)
    assert tgt_open_id in out.input_ids
    assert tgt_close_id in out.input_ids
    open_idx = out.input_ids.index(tgt_open_id)
    close_idx = out.input_ids.index(tgt_close_id)
    target_block_len = close_idx - open_idx + 1  # includes both markers
    assert target_block_len <= 32  # hard target_budget cap


def test_prompt_clamped_when_overlong(tokenizer):
    """A prompt longer than the residual budget must be right-truncated."""
    huge_prompt = " ".join(["preamble"] * 600)
    out = build_contextual_input(
        prompt=huge_prompt,
        prior_sentences=[],
        target_sentence="T.",
        tokenizer=tokenizer,
        max_length=64,  # tight budget forces prompt clamp
        target_budget=8,
    )
    # Should still produce a valid sequence within max_length.
    assert len(out.input_ids) <= 64


# --- Round-trip: structural ordering --------------------------------------

def test_layout_order_is_canonical(tokenizer):
    """Verify the canonical layout: <s> prompt </s></s> [CTX_TRUNC?] priors <TGT> ... </TGT> </s>."""
    out = build_contextual_input(
        prompt="Question?",
        prior_sentences=["S1.", "S2.", "S3."],  # forces marker
        target_sentence="Target.",
        tokenizer=tokenizer,
    )
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    ctx_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    tgt_open_id = tokenizer.convert_tokens_to_ids(TGT_OPEN)
    tgt_close_id = tokenizer.convert_tokens_to_ids(TGT_CLOSE)

    assert out.input_ids[0] == cls_id
    assert out.input_ids[-1] == sep_id

    # Find structural marker positions.
    sep_positions = [i for i, x in enumerate(out.input_ids) if x == sep_id]
    open_idx = out.input_ids.index(tgt_open_id)
    close_idx = out.input_ids.index(tgt_close_id)
    ctx_idx = out.input_ids.index(ctx_id)

    # RoBERTa pair encoding emits </s></s> as the pair separator and a
    # single trailing </s>, so we expect three sep positions total: the
    # first two are consecutive (the pair sep), the last is the
    # sequence terminator.
    assert len(sep_positions) == 3, _layout(tokenizer, out.input_ids)
    pair_sep_first = sep_positions[0]
    pair_sep_second = sep_positions[1]
    final_sep = sep_positions[2]
    assert pair_sep_second == pair_sep_first + 1

    # Order: cls < pair_sep < ctx_marker < <TGT> < </TGT> < final_sep.
    assert 0 < pair_sep_first < ctx_idx < open_idx < close_idx < final_sep
