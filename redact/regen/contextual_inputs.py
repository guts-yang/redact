"""Token-level input construction for the contextual leak classifier.

Layout (RoBERTa pair encoding + target markers):

    <s> prompt </s></s> [CTX_TRUNC?] S_{i-k+1}...S_{i-1} <TGT> S_i </TGT> </s>

- Segment A holds the prompt (the R-TOFU question).
- Segment B holds the prior-sentence context followed by the target
  sentence wrapped in ``<TGT>``/``</TGT>`` markers.
- Up to ``max_prior_sentences`` (default 2) of the most recent prior
  sentences are kept, in document order. This matches the labeling
  protocol, where each focal sentence was labeled with at most the
  same number of prior sentences as context.
- ``[CTX_TRUNC]`` is prepended to the context block iff (a) the caller
  passed more priors than ``max_prior_sentences`` (so older context
  was sliced away), or (b) one of the kept priors had to drop due to
  the 512-token budget.

The three special tokens (``<TGT>``, ``</TGT>``, ``[CTX_TRUNC]``) must
be registered on the tokenizer before calling :func:`build_contextual_input`;
the training script adds them and calls ``resize_token_embeddings``.

This module is pure (no model/GPU dependencies) to keep the logic
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from redact.regen.utils import clean_question

TGT_OPEN = "<TGT>"
TGT_CLOSE = "</TGT>"
CTX_TRUNC = "[CTX_TRUNC]"
CONTEXTUAL_SPECIAL_TOKENS = [TGT_OPEN, TGT_CLOSE, CTX_TRUNC]


@dataclass
class ContextualInput:
    input_ids: list[int]
    attention_mask: list[int]
    truncated_context: bool  # True iff one or more prior sentences were dropped


def _encode(tokenizer, text: str, with_leading_space: bool) -> list[int]:
    """Tokenize ``text`` with RoBERTa BPE sensitivity to leading whitespace."""
    if with_leading_space and text and not text.startswith(" "):
        text = " " + text
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_contextual_input(
    prompt: str,
    prior_sentences: list[str],
    target_sentence: str,
    tokenizer,
    max_length: int = 512,
    target_budget: int = 128,
    max_prior_sentences: int = 2,
) -> ContextualInput:
    """Build input_ids and attention_mask for one contextual classifier forward.

    Parameters
    ----------
    prompt : str
        The user prompt / question (may be empty; caller decides).
    prior_sentences : list[str]
        Cumulative CoT (or answer) sentences preceding the target, in
        document order. The function slices to the last
        ``max_prior_sentences`` internally — callers can hand the full
        cumulative list and let the builder do the windowing, which keeps
        ``k`` configurable in one place.
    target_sentence : str
        The sentence being classified.
    tokenizer : PreTrainedTokenizer
        Must have ``<TGT>``, ``</TGT>``, ``[CTX_TRUNC]`` registered as
        special tokens. Must be a RoBERTa-family tokenizer (uses
        ``cls_token_id``, ``sep_token_id``, ``</s></s>`` pair separator).
    max_length : int
        Hard cap on the output length (RoBERTa's 512 by default).
    target_budget : int
        Token budget reserved for ``<TGT> target </TGT>`` (including the
        two marker tokens). Target content is truncated from the right
        if it exceeds ``target_budget - 2``.
    max_prior_sentences : int
        Hard cap on how many of the most recent prior sentences are
        kept (in document order). Default 2 matches the labeling
        protocol. Set to 0 to disable context entirely.

    Returns
    -------
    ContextualInput
        ``input_ids``, ``attention_mask`` (all ones since nothing pads
        inside this function), and ``truncated_context`` (True iff at
        least one prior sentence the labeling protocol would have shown
        was dropped — either by ``max_prior_sentences`` slicing or by
        token-budget eviction).
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError(
            "tokenizer must expose cls_token_id and sep_token_id "
            "(RoBERTa-family expected)."
        )
    tgt_open_id = tokenizer.convert_tokens_to_ids(TGT_OPEN)
    tgt_close_id = tokenizer.convert_tokens_to_ids(TGT_CLOSE)
    ctx_trunc_id = tokenizer.convert_tokens_to_ids(CTX_TRUNC)
    unk_id = tokenizer.unk_token_id
    for name, tid in (
        (TGT_OPEN, tgt_open_id),
        (TGT_CLOSE, tgt_close_id),
        (CTX_TRUNC, ctx_trunc_id),
    ):
        if tid is None or tid == unk_id:
            raise ValueError(
                f"Special token {name!r} not registered on tokenizer. "
                f"Add it via tokenizer.add_special_tokens({{'additional_special_tokens': [...]}}), "
                f"then call model.resize_token_embeddings(len(tokenizer))."
            )

    if max_prior_sentences < 0:
        raise ValueError(
            f"max_prior_sentences must be >= 0, got {max_prior_sentences}"
        )

    # --- Target segment (wrapped in markers, truncated to budget) ------------
    target_budget_content = max(target_budget - 2, 1)
    target_content_ids = _encode(tokenizer, target_sentence, with_leading_space=False)
    if len(target_content_ids) > target_budget_content:
        target_content_ids = target_content_ids[:target_budget_content]
    target_ids = [tgt_open_id] + target_content_ids + [tgt_close_id]

    # --- Prompt segment ------------------------------------------------------
    # Strip the LRM-target chat-template wrappers if the caller passed a
    # formatted prompt. The classifier was trained on bare prompts (the
    # labeling pipeline applies the same strip), and raw-encoding the
    # ``<｜User｜>``/``<｜Assistant｜>`` markers in RoBERTa-base shifts the
    # encoder distribution enough to depress borderline first-leak scores
    # below threshold (verified on forget01 q0: sent 9 p_leak drops from
    # 0.76 to 0.04 with chat-template tokens vs without). Idempotent on
    # already-bare prompts so training-side callers are unaffected.
    prompt = clean_question(prompt)
    prompt_ids = _encode(tokenizer, prompt, with_leading_space=False)

    # Overhead: <s> prompt </s></s> <target...> </s>
    # = 1 (cls) + 2 (pair sep) + 1 (final sep) = 4 tokens
    overhead = 4
    remaining_after_target = max_length - overhead - len(target_ids)
    if remaining_after_target < 1:
        # Target ate the whole budget — shrink target further to leave
        # room for at least one prompt token (empty prompt OK, but we
        # still need a valid encoding).
        shrink = 1 - remaining_after_target  # positive
        new_content_len = max(len(target_content_ids) - shrink, 1)
        target_content_ids = target_content_ids[:new_content_len]
        target_ids = [tgt_open_id] + target_content_ids + [tgt_close_id]
        remaining_after_target = max_length - overhead - len(target_ids)

    # Truncate prompt if it alone exceeds the remaining budget.
    if len(prompt_ids) > remaining_after_target:
        prompt_ids = prompt_ids[:remaining_after_target]

    context_budget = remaining_after_target - len(prompt_ids)

    # --- Slice priors to the protocol's k-window ----------------------------
    # The labeler saw at most ``max_prior_sentences`` priors (most recent),
    # so anything older is structurally truncated regardless of budget.
    if max_prior_sentences > 0:
        kept_prior_sents = list(prior_sentences[-max_prior_sentences:])
    else:
        kept_prior_sents = []
    truncated_by_slice = len(prior_sentences) > len(kept_prior_sents)

    # --- Pack kept priors right-to-left within ``context_budget`` -----------
    # ``reserved`` accounts for the [CTX_TRUNC] token slot when we know
    # we'll need it. If a kept prior drops during packing, we set
    # truncation flag and re-pack with reservation so the marker fits.
    reserved = 1 if truncated_by_slice else 0
    context_ids, dropped_during_packing = _pack_priors_rtl(
        kept_prior_sents,
        tokenizer,
        budget=context_budget - reserved,
    )
    if dropped_during_packing and reserved == 0:
        reserved = 1
        context_ids, _ = _pack_priors_rtl(
            kept_prior_sents,
            tokenizer,
            budget=context_budget - reserved,
        )

    truncated_context = truncated_by_slice or dropped_during_packing

    # Emit the marker only if the budget can hold it. If even one token
    # of context budget is unavailable, we still flag truncated_context
    # in the returned dataclass so callers can audit; the model just
    # won't see the marker token in input_ids.
    if truncated_context and context_budget >= 1:
        context_ids = [ctx_trunc_id] + context_ids

    # --- Assemble final input_ids -------------------------------------------
    # <s> prompt </s></s> [CTX_TRUNC?] context target </s>
    input_ids = (
        [cls_id]
        + prompt_ids
        + [sep_id, sep_id]
        + context_ids
        + target_ids
        + [sep_id]
    )

    # Defensive truncation (should only trigger if budget math was off).
    if len(input_ids) > max_length:
        input_ids = input_ids[: max_length - 1] + [sep_id]

    attention_mask = [1] * len(input_ids)
    return ContextualInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        truncated_context=truncated_context,
    )


def _pack_priors_rtl(
    kept_prior_sents: list[str],
    tokenizer,
    budget: int,
) -> tuple[list[int], bool]:
    """Pack ``kept_prior_sents`` into ``budget`` tokens, right-to-left.

    Most recent prior is added first (and kept if it fits); earliest
    drops first if they don't all fit. Returns ``(context_ids, dropped)``
    where ``dropped`` is True iff at least one of ``kept_prior_sents``
    did not fit.
    """
    if budget < 0:
        return [], len(kept_prior_sents) > 0
    out: list[int] = []
    dropped = False
    for sent in reversed(kept_prior_sents):
        # Prepend a leading space so BPE joins correctly — each prior
        # sentence is a distinct text block separated by whitespace when
        # printed, so its first token should carry the ``Ġ`` prefix.
        sent_ids = _encode(tokenizer, sent, with_leading_space=True)
        if len(out) + len(sent_ids) <= budget:
            out = sent_ids + out
        else:
            dropped = True
            break
    return out, dropped
