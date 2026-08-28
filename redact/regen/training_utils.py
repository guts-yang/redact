"""Pure-Python helpers for batched soft-token training.

Factored out of ``scripts/train_soft_token.py`` so the batched training
path can be unit-tested without loading the full LRM target.

The pipeline per minibatch is:

  1. ``prepare_pass_record`` — tokenize one (prompt, prefix, continuation)
     triple for one pass (forget / retain_clean / retain_ctx / idk),
     returning a :class:`PassRecord` with a ``valid`` flag for the
     cross-pass filter.
  2. **Cross-pass filter** (in the caller) — drop any record whose
     ``valid`` is False on any *active* pass, so per-pass ``(B,)`` loss
     tensors line up index-for-index for element-wise hinge math.
  3. ``assemble_pass_batch`` — left-pad surviving records into
     ``(B, L)`` tensors with per-row absolute soft-token positions and a
     ``(B, L-1)`` shifted continuation mask.
  4. ``build_hook_schedule`` — translate per-row positions into the
     schedule shape that ``SoftToken.apply_hook`` expects.
  5. ``per_row_masked_ce`` — apply the precomputed loss mask to the
     model's shifted logits and return one CE per row.

Per-epoch ordering is provided by ``length_bucketed_shuffle``: a
standard length-bucketed sampler that keeps similar-length records in
the same minibatch to reduce left-pad waste, while preserving global
randomness via a random initial shuffle and bucket-order shuffle.
"""
import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch
import torch.nn as nn

from redact.regen.learned_hooks import truncate_for_soft_token_budget

T = TypeVar("T")


@dataclass
class PassRecord:
    """Per-record per-pass tokenization output.

    Carries everything :func:`assemble_pass_batch` needs to build the
    batched tensors plus a ``valid`` flag for the cross-pass filter.
    """

    before_ids: list[int]
    pad_ids: list[int]
    after_ids: list[int]
    valid: bool

    @property
    def total_len(self) -> int:
        return len(self.before_ids) + len(self.pad_ids) + len(self.after_ids)


@dataclass
class PassBatch:
    """Batched tensors for a single pass over surviving records."""

    input_ids: torch.Tensor          # (B, L) long
    attention_mask: torch.Tensor     # (B, L) long
    soft_token_positions: list[list[int]]  # per row, [pos]
    loss_mask: torch.Tensor          # (B, L-1) float


def prepare_pass_record(
    prompt: str,
    prefix: str,
    continuation: str,
    *,
    tokenizer,
    pad_token_id: int,
    n_tokens_per_block: int,
    max_seq_len: int,
    append_eos: bool,
    think_prefix: str = "<think>\n",
) -> PassRecord:
    """Tokenize one (prompt, prefix, continuation) triple and apply the
    K-pad-budget truncation.

    The prompt is tokenized with ``add_special_tokens=True`` so the
    tokenizer prepends BOS for tokenizers that have one (LLaMA / DeepSeek
    family); ``think_prefix``, ``prefix`` and ``continuation`` are
    encoded separately with ``add_special_tokens=False`` and the four
    token lists are concatenated. This mirrors the inference paths in
    ``RegeneratingReasoningEngine._regenerate_sample`` and
    ``scripts/generate_corrupted_cots`` exactly, so the soft token sees
    the same absolute positions and leading BOS during training as at
    inference. Tokenizing the four pieces separately also avoids BPE
    drift at concatenation boundaries.

    ``valid`` is True iff at least one continuation token survives —
    a degenerate row (prompt+prefix+pad alone fills the budget) returns
    ``valid=False`` and the cross-pass filter at the call site drops it.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    think_ids = tokenizer.encode(think_prefix, add_special_tokens=False)
    prefix_ids = (
        tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    )
    before_ids = prompt_ids + think_ids + prefix_ids
    pad_ids = [pad_token_id] * n_tokens_per_block
    after_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "append_eos=True but tokenizer has no eos_token_id"
            )
        after_ids.append(tokenizer.eos_token_id)

    before_ids, after_ids = truncate_for_soft_token_budget(
        before_ids, pad_ids, after_ids, max_seq_len, append_eos
    )

    return PassRecord(
        before_ids=before_ids,
        pad_ids=pad_ids,
        after_ids=after_ids,
        valid=len(after_ids) > 0,
    )


def assemble_pass_batch(
    records: Sequence[PassRecord],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> PassBatch:
    """Left-pad valid records into a single batch.

    All records must have ``valid=True``; the cross-pass filter at the
    call site is responsible for dropping invalid rows. Including an
    invalid row would produce a row with zero supervised tokens, which
    :func:`per_row_masked_ce` rejects rather than silently returning
    NaN or a clamp-divided zero.

    Per-row absolute positions:

    * ``soft_token_pos = left_pad + len(before_ids)``
    * ``cont_start     = left_pad + len(before_ids) + len(pad_ids)``
    * ``cont_end       = left_pad + total_len``

    The shifted CE loss mask is ``(B, L-1)`` with row ``k`` having ones
    on positions ``[cont_start[k]-1 : cont_end[k]-1]`` (the standard
    next-token-prediction shift; clamped to non-negative).
    """
    if not records:
        raise ValueError("assemble_pass_batch called with zero records")
    for k, r in enumerate(records):
        if not r.valid:
            raise ValueError(
                f"row {k}: PassRecord.valid=False; the cross-pass filter "
                f"must drop invalid rows before assemble_pass_batch."
            )

    n_rows = len(records)
    L = max(r.total_len for r in records)

    input_ids = torch.full(
        (n_rows, L), pad_token_id, dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        (n_rows, L), dtype=torch.long, device=device,
    )
    soft_token_positions: list[list[int]] = []
    cont_ranges: list[tuple[int, int]] = []

    for k, r in enumerate(records):
        all_ids = r.before_ids + r.pad_ids + r.after_ids
        left_pad = L - len(all_ids)
        input_ids[k, left_pad:] = torch.tensor(
            all_ids, dtype=torch.long, device=device,
        )
        attention_mask[k, left_pad:] = 1
        st_pos = left_pad + len(r.before_ids)
        cont_start = left_pad + len(r.before_ids) + len(r.pad_ids)
        cont_end = left_pad + len(all_ids)
        soft_token_positions.append([st_pos])
        cont_ranges.append((cont_start, cont_end))

    loss_mask = torch.zeros(
        (n_rows, L - 1), dtype=torch.float32, device=device,
    )
    for k, (cs, ce) in enumerate(cont_ranges):
        # Predicting token at position p uses shifted-logit index p-1.
        ms = max(cs - 1, 0)
        me = ce - 1
        loss_mask[k, ms:me] = 1.0

    return PassBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        soft_token_positions=soft_token_positions,
        loss_mask=loss_mask,
    )


def cross_pass_filter(
    pass_records: dict[str, list[PassRecord | None]],
) -> list[int]:
    """Return record indices that survive every active pass.

    A record at index ``i`` survives iff, for every key in
    ``pass_records``, ``pass_records[key][i]`` is non-None *and*
    has ``valid=True``. The all-passes intersection is what makes
    per-pass ``(B,)`` loss tensors index-aligned for elementwise
    hinge math (e.g. ``contrast_term(l_forget, l_idk, m)``).

    A None entry is treated as "this record was never tokenized for
    this pass" — usually because the caller short-circuited on a
    precondition (e.g. a record missing the field a given pass needs)
    before invoking ``prepare_pass_record``. It contributes the same
    way as a degenerate (valid=False) tokenization: the record is
    dropped from the whole minibatch.

    Empty dict raises ``ValueError`` (the caller is supposed to have
    at least the forget pass).
    """
    if not pass_records:
        raise ValueError(
            "cross_pass_filter called with no passes; at least the "
            "forget pass should be present."
        )
    lengths = {len(recs) for recs in pass_records.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"cross_pass_filter: pass lists have inconsistent lengths "
            f"{sorted(lengths)}; every pass must have one entry per "
            f"input record (use None for records the caller skipped)."
        )
    n = lengths.pop()
    kept: list[int] = []
    for i in range(n):
        ok = True
        for recs in pass_records.values():
            r = recs[i]
            if r is None or not r.valid:
                ok = False
                break
        if ok:
            kept.append(i)
    return kept


def build_hook_schedule(
    soft_token_positions: list[list[int]],
) -> list[list[int]]:
    """Translate per-row positions into the hook schedule shape.

    Returns ``list[list[int]]``, the shape
    :meth:`SoftToken.apply_hook` expects: one inner list of absolute
    token positions per batch row.  Inner lists are copied, so the
    caller's ``PassBatch.soft_token_positions`` cannot be mutated
    through the returned schedule.
    """
    return [list(positions) for positions in soft_token_positions]


def length_bucketed_shuffle(
    items: Sequence[T],
    batch_size: int,
    length_key: Callable[[T], int],
    bucket_multiplier: int = 4,
) -> list[T]:
    """Shuffle ``items`` then sort within length-buckets to reduce
    per-minibatch left-pad waste.

    Pipeline (matches HuggingFace's ``LengthGroupedSampler`` semantics):

      1. Random shuffle the full pool — every record is equally likely
         to land in any bucket.
      2. Slice into buckets of ``batch_size * bucket_multiplier``
         consecutive records.
      3. Sort each bucket ascending by ``length_key``.
      4. Random shuffle the bucket *order* — consecutive minibatches
         see varied length regimes (short minibatch / long minibatch /
         medium / ...) instead of ramping monotonically through the
         epoch, which would bias optimizer warm-up.

    Caller then slices the returned list into ``batch_size`` minibatches
    in order. Within-minibatch length variance is bounded by the bucket's
    length spread, which is small after step 3, so left-pad per
    minibatch collapses to roughly the variance within `bucket_multiplier`
    consecutive records of the original shuffle.

    Randomness model: within a bucket, ordering is *deterministic* given
    that bucket's contents (it's just a sort by ``length_key``). The
    randomness lives in (a) which records co-occupy a bucket, set by
    the initial pool shuffle, and (b) which bucket runs when, set by
    the bucket-order shuffle. Across epochs every record sees diverse
    minibatch companions because both shuffles re-seed.

    ``bucket_multiplier=4`` follows the HF default; larger values give
    tighter length matching at the cost of less inter-minibatch
    randomness within a bucket.

    For ``batch_size=1`` this returns a plain shuffle (bucketing has
    no effect when minibatches are single-row, and the caller can skip
    this helper entirely in that case to save the bucket sort).

    All shuffles use the global ``random`` module so seeding via
    ``random.seed`` / ``seed_everything`` works as expected.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if bucket_multiplier < 1:
        raise ValueError(
            f"bucket_multiplier must be >= 1, got {bucket_multiplier}"
        )
    shuffled = list(items)
    random.shuffle(shuffled)
    if batch_size == 1:
        return shuffled
    bucket_size = batch_size * bucket_multiplier
    buckets: list[list[T]] = []
    for i in range(0, len(shuffled), bucket_size):
        bucket = shuffled[i:i + bucket_size]
        bucket.sort(key=length_key)
        buckets.append(bucket)
    random.shuffle(buckets)
    out: list[T] = []
    for bucket in buckets:
        out.extend(bucket)
    return out


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    lr: float,
    lr_min_factor: float,
):
    """Return a per-epoch LR scheduler for the soft-token training loop.

    Parameters
    ----------
    schedule
        ``"none"`` returns ``None`` (fixed lr — current behavior).
        ``"cosine"`` returns ``CosineAnnealingLR`` decaying from ``lr`` to
        ``lr * lr_min_factor`` over ``epochs`` epochs.
    epochs
        ``T_max`` for the cosine schedule. Caller is responsible for
        stepping the scheduler exactly once per training epoch (not the
        pre-train eval pass).
    lr_min_factor
        Floor for the cosine schedule, expressed as a fraction of ``lr``.
        ``0.0`` decays to zero; ``0.1`` leaves a 10% floor.

    Raises ``ValueError`` for unknown ``schedule`` strings; ``epochs <= 0``
    or ``lr_min_factor < 0`` or ``lr_min_factor > 1`` also raise.
    """
    if schedule == "none":
        return None
    if schedule != "cosine":
        raise ValueError(
            f"Unknown lr_schedule: {schedule!r}. Expected 'none' or 'cosine'."
        )
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0 for cosine schedule, got {epochs}")
    if not (0.0 <= lr_min_factor <= 1.0):
        raise ValueError(
            f"lr_min_factor must be in [0.0, 1.0], got {lr_min_factor}"
        )
    eta_min = lr * lr_min_factor
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min,
    )


def assert_clean_continuations_have_terminator(
    examples: list[dict],
    data_path: str,
) -> None:
    """Raise ``SystemExit`` if any record's ``clean_continuation`` lacks
    ``</think>``.

    The retain_clean pass tokenizes ``clean_continuation`` with
    ``append_eos=True``; the canonical target shape is
    ``retain_cot_tail + "</think>\\n\\n" + retain_answer``. Old
    ``train.jsonl`` files (CoT-tail-only, pre-answer-bundling) lack
    ``</think>`` and would silently train EOS at the end of a mid-CoT
    sentence — bad signal. Fail loud rather than rot the bank quietly.

    The check is structural (presence of ``</think>``); it does not
    validate the surrounding format. A record with a stray ``</think>``
    inside the CoT tail would pass, but that is rare in practice and
    the loose check is enough to catch the stale-format case the guard
    actually exists to flag.
    """
    n_missing = sum(
        1 for ex in examples
        if "</think>" not in (ex.get("clean_continuation") or "")
    )
    if n_missing > 0:
        raise SystemExit(
            f"ERROR: {n_missing}/{len(examples)} records in "
            f"{data_path} have a clean_continuation without `</think>`. The "
            f"retain_clean target now requires the answer terminator; "
            f"regenerate train.jsonl with the current "
            f"scripts.build_soft_token_data."
        )


def per_row_masked_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row mean cross-entropy on positions selected by ``loss_mask``.

    Shifts logits/labels for next-token prediction, computes per-token
    CE with ``reduction='none'``, multiplies by ``loss_mask``, and
    divides by per-row mask sum to produce a ``(B,)`` tensor.

    Raises ``RuntimeError`` if any row has zero supervised tokens — the
    cross-pass validity filter at the call site is responsible for
    dropping such rows. Choosing to raise rather than ``clamp(min=1)``
    makes a soft failure (silent zero gradient that drags the batch
    mean toward zero) loud.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).reshape(shift_labels.size())

    masked = per_token_loss * loss_mask
    denom = loss_mask.sum(dim=-1)
    if (denom == 0).any():
        raise RuntimeError(
            "per_row_masked_ce: row with zero supervised tokens — the "
            "cross-pass filter at the call site must drop empty rows "
            "before this function is reached."
        )
    return masked.sum(dim=-1) / denom
