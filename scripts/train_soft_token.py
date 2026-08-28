"""
Train learned soft token(s) for CoT truncation-point insertion.

Optimizes embedding vector(s) to maximize cross-entropy on forget-set
continuations while minimizing it on clean/retain continuations.

Two retain loss terms (independently weighted):
  - ``lambda_retain_clean``: CE on clean (retain-sourced) continuations
    appended after the *forget* prompt+prefix — keeps the token from
    destroying benign text after the same prompt.
  - ``lambda_retain_ctx``: CE on retain-set continuations with their own
    *retain* prompts — trains the token to be transparent on retain
    contexts (context-discriminative).

Trains one shared soft token (a block of ``--n_tokens`` consecutive
learned embedding vectors) for all examples.

Usage:
    python -m scripts.train_soft_token --split forget10 --epochs 100 --lr 1e-3
    python -m scripts.train_soft_token --split forget10 --lambda_retain_clean 1.0
    python -m scripts.train_soft_token --split forget10 --lambda_retain_ctx 0.1
    python -m scripts.train_soft_token --split forget10 --n_tokens 16
"""
import argparse
import contextlib
import json
import os
import random

import torch

from redact.regen.learned_hooks import (
    SoftToken, contrast_term, forget_term, retain_term,
)
from redact.regen.training_utils import (
    PassRecord, assemble_pass_batch,
    assert_clean_continuations_have_terminator, build_hook_schedule,
    cross_pass_filter, length_bucketed_shuffle, make_lr_scheduler,
    per_row_masked_ce, prepare_pass_record,
)
from redact.regen.utils import get_nested_attr, remove_hooks
from redact.evaluator.utils import split_sentences
from redact.model import HFModel
from redact.utils import log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="soft_token_data")
parser.add_argument("--raw_cots_dir", type=str, default="leak_detector_data")
parser.add_argument("--raw_cots_split", type=str, default=None,
                    help="Split name used to locate raw_cots.json "
                         "(defaults to --split).")
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_schedule", type=str, default="none",
                    choices=["none", "cosine"],
                    help="Learning-rate schedule. 'none' keeps lr fixed at "
                         "--lr (current behavior). 'cosine' uses "
                         "CosineAnnealingLR over --epochs to decay from --lr "
                         "down to --lr * --lr_min_factor; smooths late-"
                         "training oscillation in losses (esp. L_idk).")
parser.add_argument("--lr_min_factor", type=float, default=0.0,
                    help="Floor for the cosine schedule, expressed as a "
                         "fraction of --lr. 0.0 (default) decays to zero; "
                         "0.1 leaves a 10%% floor so the bank keeps slowly "
                         "updating through the end. Ignored when "
                         "--lr_schedule=none.")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lambda_retain_clean", type=float, default=0.0,
                    help="Weight for retain loss on clean continuations "
                         "(same forget prompt). The clean continuation now "
                         "carries `</think>\\n\\n` + answer with EOS "
                         "supervision, so this term teaches both retain-"
                         "content transparency AND clean termination — "
                         "previously the IDK term was the sole termination "
                         "signal. Re-sweep this weight when adopting the "
                         "new target shape; old λ values were tuned against "
                         "a CoT-tail-only objective.")
parser.add_argument("--retain_margin", type=float, default=0.0,
                    help="If >0, wrap the retain term in a one-sided hinge: "
                         "max(0, l_retain - margin). The penalty contributes "
                         "to the gradient only when L_ret_clean > margin "
                         "(pulls L_ret down toward margin); zero gradient "
                         "below margin (bank is left alone). Caps how far "
                         "the retain term drives the bank toward the "
                         "mismatched-author retain target, which empirically "
                         "produced inference-time fabrication artefacts at "
                         "low L_ret_clean. Typical values: 5-6 (matches the "
                         "natural peak of L_ret_clean during training). "
                         "0 disables (current behavior).")
parser.add_argument("--lambda_retain_ctx", type=float, default=0.0,
                    help="Weight for retain loss on retain-set prompts "
                         "(context-discriminative)")
parser.add_argument("--lambda_idk", type=float, default=0.0)
parser.add_argument("--lambda_l2", type=float, default=0.0,
                    help="L2 regularization on soft token embedding norm")
parser.add_argument("--forget_margin", type=float, default=0.0,
                    help="If >0, replace -l_forget with max(0, margin - l_forget). "
                         "Bounds the forget reward so the optimizer stops pushing "
                         "CE toward infinity once the forget continuation is "
                         "sufficiently suppressed. Typical value: base-model CE on "
                         "forget continuation + 3-5 nats. 0 disables (old behavior). "
                         "Mutually exclusive with --contrast_margin.")
parser.add_argument("--contrast_margin", type=float, default=0.0,
                    help="If >0, replace the forget + IDK terms with a single "
                         "preference-style ranking loss "
                         "max(0, margin - l_forget + l_idk). Wants the token to "
                         "prefer the IDK continuation over the leak continuation "
                         "by the given nat margin. Requires IDK data (reasoned "
                         "IDK fields or a static --idk_response). Mutually "
                         "exclusive with --forget_margin.")
parser.add_argument(
    "--idk_response",
    type=str,
    default="</think>\n\nI don't know.",
    help="Target response for IDK loss term (static mode fallback)",
)
parser.add_argument(
    "--idk_mode",
    type=str,
    default="static",
    choices=["static", "answer", "full"],
    help="IDK continuation: static=fixed string, answer=per-example refusal, "
         "full=per-example reasoned CoT + refusal",
)
parser.add_argument("--idk_max_seq_len", type=int, default=0,
                    help="Max seq len for IDK loss (0 = use --max_seq_len)")
parser.add_argument("--max_seq_len", type=int, default=2048)
parser.add_argument("--output_dir", type=str, default="soft_tokens")
parser.add_argument("--checkpoint_every", type=int, default=0,
                    help="Save checkpoint every N epochs (0 = disabled)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--n_tokens",
    type=int,
    default=1,
    help="Number of consecutive embedding vectors in the soft-token "
         "block. Default 1 reproduces the original single-vector "
         "behavior.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Minibatch size for soft-token training. Each minibatch runs "
         "one forward/backward per active loss component (forget / "
         "retain_clean / retain_ctx / idk) and a single optimizer step. "
         "Default 1 preserves the per-example SGD shape; larger values "
         "speed up sweeps at the cost of changing the gradient signal "
         "(see PIPELINE.md).",
)
args = parser.parse_args()
if args.batch_size < 1:
    parser.error("--batch_size must be >= 1")
if args.n_tokens < 1:
    parser.error("--n_tokens must be >= 1")

if args.forget_margin > 0 and args.contrast_margin > 0:
    parser.error("--forget_margin and --contrast_margin are mutually exclusive")
contrast_mode = args.contrast_margin > 0

seed_everything(args.seed)

# -------------------------------------------------------------------------
# Load model (frozen)
# -------------------------------------------------------------------------

log_print(f"Loading model: {args.model_name}")
hf_model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
)
model = hf_model.model
tokenizer = hf_model.tokenizer
model_config = hf_model.model_config

# Freeze all model parameters — gradients flow through the computation
# graph but only update the soft token embedding.
model.eval()
model.requires_grad_(False)

device = model.device
embed_dim = model_config["embedding_dim"]
embedding_module_path = model_config["embedding_module"]
embed_module = get_nested_attr(model, embedding_module_path)

log_print(f"Device: {device}, Embed dim: {embed_dim}")
log_print(f"Embedding module: {embedding_module_path}")

# -------------------------------------------------------------------------
# Load training data
# -------------------------------------------------------------------------

data_path = f"{args.data_dir}/{args.split}/train.jsonl"
log_print(f"Loading forget training data from {data_path}")
forget_examples = []
with open(data_path) as f:
    for line in f:
        forget_examples.append(json.loads(line))
log_print(f"Forget examples: {len(forget_examples)}")

if args.lambda_retain_clean > 0:
    assert_clean_continuations_have_terminator(forget_examples, data_path)

# Load retain examples for context-discriminative retain loss
retain_examples = []
if args.lambda_retain_ctx > 0:
    raw_cots_split = args.raw_cots_split or args.split
    raw_cots_path = f"{args.raw_cots_dir}/{raw_cots_split}/raw_cots.json"
    log_print(f"Loading retain CoTs from {raw_cots_path}")
    with open(raw_cots_path) as f:
        raw_data = json.load(f)

    retain_prompts = raw_data["retain_prompts"]
    retain_cots = raw_data["retain_cots"]
    log_print(f"Retain prompts: {len(retain_prompts)}, Retain CoTs: {len(retain_cots)}")

    for prompt, cot in zip(retain_prompts, retain_cots):
        sentences = split_sentences(cot)
        if len(sentences) < 2:
            continue
        retain_examples.append({"prompt": prompt, "sentences": sentences})
    log_print(f"Retain examples (with >=2 sentences): {len(retain_examples)}")

# -------------------------------------------------------------------------
# Initialize soft token(s) and optimizer
# -------------------------------------------------------------------------

soft_token = SoftToken(
    embed_dim=embed_dim, n_tokens=args.n_tokens
).to(device)
optimizer = torch.optim.Adam([soft_token.embedding], lr=args.lr)
log_print(
    f"Soft token initialized (dim={embed_dim}, "
    f"n_tokens={args.n_tokens})"
)

# Optional cosine LR decay. Steps once per epoch end (not per minibatch). The
# pre-train eval pass at epoch=-1 doesn't step the scheduler, so the schedule
# starts counting from epoch 1 and ends at epoch=args.epochs as expected.
scheduler = make_lr_scheduler(
    optimizer,
    schedule=args.lr_schedule,
    epochs=args.epochs,
    lr=args.lr,
    lr_min_factor=args.lr_min_factor,
)
if scheduler is not None:
    log_print(
        f"LR schedule: {args.lr_schedule}, T_max={args.epochs}, "
        f"start_lr={args.lr}, eta_min={args.lr * args.lr_min_factor}"
    )

THINK_PREFIX = "<think>\n"
pad_token_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)
n_tokens_per_block = args.n_tokens

# Precompute forget-pass token length for length bucketing. Forget is
# the dominant pass and the others (retain_clean, idk) share the same
# prompt+prefix prefix, so the forget length is a good correlate of
# overall minibatch padding waste. One-shot cost: ~0.3s for 293
# records on the LRM-target tokenizer. Stored on the example dicts
# under "_bucket_len" — the leading underscore signals "internal,
# do not serialize".
if args.batch_size > 1:
    log_print("Precomputing forget-pass lengths for length bucketing...")
    for ex in forget_examples:
        text = (
            ex["prompt"] + THINK_PREFIX + ex["prefix"]
            + ex["leaking_continuation"]
        )
        # add_special_tokens=True so the bucket length tracks
        # prepare_pass_record (which prepends BOS to the prompt). The
        # bucket length is just a sort key, so a small +/-1 BPE-boundary
        # drift vs the actual concatenated tokenization is harmless.
        ex["_bucket_len"] = len(
            tokenizer.encode(text, add_special_tokens=True)
        )
    lens = [ex["_bucket_len"] for ex in forget_examples]
    log_print(
        f"  forget length stats: min={min(lens)}, max={max(lens)}, "
        f"mean={sum(lens) / len(lens):.0f}"
    )

log_print(f"Optimizer: Adam, lr={args.lr}")
log_print(
    f"Lambda retain_clean: {args.lambda_retain_clean}, "
    f"Lambda retain_ctx: {args.lambda_retain_ctx}, "
    f"Lambda IDK: {args.lambda_idk}, Epochs: {args.epochs}"
)
if args.forget_margin > 0:
    log_print(f"Forget margin: {args.forget_margin} (hinge loss active)")
if contrast_mode:
    log_print(
        f"Contrast margin: {args.contrast_margin} "
        f"(preference-ranking loss active; l_idk computed regardless of lambda_idk)"
    )
if args.lambda_retain_ctx > 0:
    log_print("Retain ctx mode: context-discriminative (retain-set prompts)")
# EOS supervision now applies to both the IDK pass and retain_clean (after
# the answer-terminator change in build_soft_token_data). Guard once for
# whichever active pass needs it; canonical runs trigger this either way.
if args.lambda_idk > 0 or contrast_mode or args.lambda_retain_clean > 0:
    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer for {args.model_name!r} has no eos_token_id, but at "
            f"least one active pass (IDK and/or retain_clean) supervises an "
            f"EOS-terminated target. Either configure an EOS token in the "
            f"model config or drop the affected supervision (--lambda_idk, "
            f"--contrast_margin, --lambda_retain_clean)."
        )
if args.lambda_idk > 0 or contrast_mode:
    log_print(f"IDK mode: {args.idk_mode}")
    if args.idk_mode != "static":
        n_with_idk = sum(1 for ex in forget_examples if "reasoned_idk_answer" in ex)
        log_print(f"Forget examples with reasoned IDK: {n_with_idk}/{len(forget_examples)}")
        if n_with_idk == 0:
            log_print("WARNING: No examples have reasoned IDK fields; falling back to static")
        if args.idk_max_seq_len > 0:
            log_print(f"IDK max_seq_len override: {args.idk_max_seq_len}")
    else:
        idk_ids = tokenizer.encode(args.idk_response, add_special_tokens=False)
        log_print(f"IDK response: {args.idk_response!r} -> {len(idk_ids)} tokens: {idk_ids}")
        assert len(idk_ids) > 0, "IDK response tokenized to empty sequence"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


# Active-pass selection. Mirrors today's per-example branches; computed
# once at startup since these are governed by global flags, not per-record.
ACTIVE_FORGET = True
ACTIVE_RETAIN_CLEAN = args.lambda_retain_clean > 0
ACTIVE_RETAIN_CTX = args.lambda_retain_ctx > 0
ACTIVE_IDK = args.lambda_idk > 0 or contrast_mode


def _idk_continuation(example):
    """Build the IDK target continuation for one record.

    Mirrors today's per-record branch: prefer reasoned IDK when the mode
    selects it and the example carries the field; otherwise fall back to
    the static --idk_response.
    """
    if (args.idk_mode == "answer"
            and "reasoned_idk_answer" in example):
        return "</think>\n\n" + example["reasoned_idk_answer"]
    if (args.idk_mode == "full"
            and "reasoned_idk_cot" in example
            and "reasoned_idk_answer" in example):
        return (example["reasoned_idk_cot"]
                + "</think>\n\n" + example["reasoned_idk_answer"])
    return args.idk_response


def sample_retain_example():
    """Sample a retain (prompt, prefix, continuation) triple.

    Picks a random retain CoT and splits it at a random sentence boundary.
    """
    ex = random.choice(retain_examples)
    sentences = ex["sentences"]
    split_point = random.randint(1, len(sentences) - 1)
    prefix = " ".join(sentences[:split_point])
    continuation = " ".join(sentences[split_point:])
    return ex["prompt"], prefix, continuation


def _prepare(prompt, prefix, continuation, *, append_eos, max_seq_len_override=None):
    """Thin wrapper around prepare_pass_record that fills in the script's
    globals (tokenizer, pad_token_id, K, default max_seq_len)."""
    return prepare_pass_record(
        prompt, prefix, continuation,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        n_tokens_per_block=n_tokens_per_block,
        max_seq_len=(
            max_seq_len_override
            if max_seq_len_override is not None
            else args.max_seq_len
        ),
        append_eos=append_eos,
        think_prefix=THINK_PREFIX,
    )


def _build_minibatch_records(minibatch):
    """Tokenize every active pass for a minibatch and run the cross-pass
    validity filter.

    Returns ``(kept_indices, kept_examples, retain_triples,
    pass_records)``:

    * ``kept_indices`` — positions within ``minibatch`` of records that
      survive every active pass (all active passes' ``valid=True``).
    * ``kept_examples`` — the surviving forget records, sorted by total
      forget-pass length (ascending). Sorting reduces pad waste in the
      forget pass; other passes pad independently against their own
      max length.
    * ``retain_triples`` — per kept record, the sampled retain
      (prompt, prefix, cont) tuple aligned with that record. Empty list
      if retain_ctx is inactive.
    * ``pass_records`` — dict of pass-name → list of PassRecord, aligned
      with ``kept_examples`` (so passes index-line up for elementwise
      hinge math downstream).

    Records are sampled and tokenized in input order; only the
    surviving subset is sorted, so the retain-triple RNG sequence is
    deterministic at fixed seed within an epoch.
    """
    n = len(minibatch)
    pass_recs: dict[str, list[PassRecord | None]] = {
        "forget": [None] * n,
    }
    if ACTIVE_RETAIN_CLEAN:
        pass_recs["retain_clean"] = [None] * n
    if ACTIVE_RETAIN_CTX:
        pass_recs["retain_ctx"] = [None] * n
    if ACTIVE_IDK:
        pass_recs["idk"] = [None] * n

    # Sample retain triples per-record up-front (stays in input order so
    # RNG-replay is deterministic regardless of the cross-pass filter).
    retain_triples_full: list[tuple[str, str, str] | None] = [None] * n
    if ACTIVE_RETAIN_CTX:
        for i in range(n):
            retain_triples_full[i] = sample_retain_example()

    idk_max = (
        args.idk_max_seq_len if args.idk_max_seq_len > 0 else None
    )

    for i, ex in enumerate(minibatch):
        pass_recs["forget"][i] = _prepare(
            ex["prompt"], ex["prefix"], ex["leaking_continuation"],
            append_eos=False,
        )
        if ACTIVE_RETAIN_CLEAN:
            # append_eos=True: clean_continuation now carries the full
            # end-of-response (CoT tail + </think>\n\n + answer); supervising
            # the EOS token at the end teaches the soft token to allow clean
            # termination, which the IDK term was previously the sole source of.
            pass_recs["retain_clean"][i] = _prepare(
                ex["prompt"], ex["prefix"], ex["clean_continuation"],
                append_eos=True,
            )
        if ACTIVE_RETAIN_CTX:
            triple = retain_triples_full[i]
            assert triple is not None  # populated above when ACTIVE_RETAIN_CTX
            ret_prompt, ret_prefix, ret_cont = triple
            pass_recs["retain_ctx"][i] = _prepare(
                ret_prompt, ret_prefix, ret_cont,
                append_eos=False,
            )
        if ACTIVE_IDK:
            pass_recs["idk"][i] = _prepare(
                ex["prompt"], ex["prefix"], _idk_continuation(ex),
                append_eos=True,
                max_seq_len_override=idk_max,
            )

    # Cross-pass filter: any record None-or-invalid in any pass is
    # dropped from the whole minibatch.
    kept_indices = cross_pass_filter(pass_recs)

    if not kept_indices:
        return [], [], [], {}

    # Sort kept indices by forget-pass total length (ascending). Halves
    # worst-case wasted compute on the forget pass; other passes pad
    # against their own per-pass max independently. The cross-pass
    # filter above guarantees pass_recs["forget"][i] is non-None for
    # every i in kept_indices.
    def _forget_len(i: int) -> int:
        rec = pass_recs["forget"][i]
        assert rec is not None
        return rec.total_len
    kept_indices.sort(key=_forget_len)

    kept_examples = [minibatch[i] for i in kept_indices]
    if ACTIVE_RETAIN_CTX:
        retain_triples = []
        for i in kept_indices:
            t = retain_triples_full[i]
            assert t is not None
            retain_triples.append(t)
    else:
        retain_triples = []
    aligned_pass_records: dict[str, list[PassRecord]] = {}
    for kind, recs in pass_recs.items():
        aligned: list[PassRecord] = []
        for i in kept_indices:
            r = recs[i]
            assert r is not None
            aligned.append(r)
        aligned_pass_records[kind] = aligned
    return kept_indices, kept_examples, retain_triples, aligned_pass_records


@contextlib.contextmanager
def _hook_for_pass(positions):
    """Register the soft-token hook for one pass; remove on exit.

    ``positions`` is per-row [pos] (list[list[int]]), which
    ``build_hook_schedule`` turns into the ``list[list[int]]`` schedule
    that ``SoftToken.apply_hook`` expects.

    try/finally guarantees hook removal even if the forward raises
    (autograd NaN, OOM, shape mismatch) — without this, a stale hook
    would fire on the next pass with a stale batch-size assertion.
    """
    schedule = build_hook_schedule(positions)
    handle = soft_token.apply_hook(embed_module, schedule)
    try:
        yield
    finally:
        handle.remove()


def _run_pass(records, *, log_diagnostic=False, pass_name=""):
    """Run one batched pass: assemble → register hook → forward → CE.

    Returns a (B,) per-row CE tensor aligned with ``records``.
    """
    batch = assemble_pass_batch(
        records, pad_token_id=pad_token_id, device=device,
    )
    if log_diagnostic:
        _log_pass_diagnostic(pass_name, batch)
    with _hook_for_pass(batch.soft_token_positions):
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
    return per_row_masked_ce(
        outputs.logits, batch.input_ids, batch.loss_mask
    )


def _log_pass_diagnostic(pass_name, batch):
    """Eval-pass diagnostic: log row 0's tokenization for one pass.

    Replaces the per-example ``log_diagnostic=True`` flag the old
    tokenize_with_soft_token took. Fires exactly once per active pass on
    the first minibatch of the eval pass.
    """
    row = 0
    pad_count = int((batch.attention_mask[row] == 0).sum().item())
    # assemble_pass_batch always left-pads (real tokens occupy [pad_count:]);
    # this assert fails loud if anyone ever introduces a right-pad path.
    assert (batch.attention_mask[row, pad_count:] == 1).all(), (
        "diagnostic log assumes pure left-pad layout from assemble_pass_batch"
    )
    real_ids = batch.input_ids[row].tolist()[pad_count:]
    pos = batch.soft_token_positions[row][0] - pad_count
    n_supervised = int(batch.loss_mask[row].sum().item())
    log_print(f"\n[runtime check] pass={pass_name}")
    log_print(f"  ids[:30]: {real_ids[:30]}")
    log_print(
        f"  tokens[:30]: "
        f"{tokenizer.convert_ids_to_tokens(real_ids[:30])}"
    )
    log_print(f"  ids[-12:]: {real_ids[-12:]}")
    log_print(
        f"  tokens[-12:]: "
        f"{tokenizer.convert_ids_to_tokens(real_ids[-12:])}"
    )
    log_print(
        f"  soft_token_pos(real)={pos}, "
        f"supervised_tokens={n_supervised}"
    )


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

log_print("\n--- Starting training ---")

# epoch=-1 is a no-grad evaluation pass over the same loop body before any
# optimizer updates. Reports the loss components at the bank's initialization
# so we can see the natural CE baseline (i.e. how much of an apparent
# L_forget signal comes from the data alone vs. from bank-driven suppression).
for epoch in range(-1, args.epochs):
    is_pretrain_eval = (epoch == -1)
    torch.set_grad_enabled(not is_pretrain_eval)
    if not is_pretrain_eval:
        if args.batch_size > 1:
            # Length-bucket within batch_size×4 windows so each
            # minibatch's left-pad waste is bounded by the within-bucket
            # length variance (was bounded by the within-shuffle
            # variance, which made pad waste 2-3× the optimum).
            forget_examples = length_bucketed_shuffle(
                forget_examples,
                batch_size=args.batch_size,
                length_key=lambda ex: ex["_bucket_len"],
            )
        else:
            random.shuffle(forget_examples)
    epoch_forget_loss = 0.0
    epoch_retain_clean_loss = 0.0
    epoch_retain_ctx_loss = 0.0
    epoch_idk_loss = 0.0
    epoch_total_loss = 0.0
    n_examples = 0
    n_idk_examples = 0
    n_margin_saturated = 0
    n_retain_saturated = 0

    n_minibatches = (
        len(forget_examples) + args.batch_size - 1
    ) // args.batch_size

    for mb_idx in range(n_minibatches):
        start = mb_idx * args.batch_size
        end = min(start + args.batch_size, len(forget_examples))
        minibatch = forget_examples[start:end]

        kept_indices, kept_examples, _retain_triples, pass_records = (
            _build_minibatch_records(minibatch)
        )
        if not kept_indices:
            continue

        # Diagnostic gate: first minibatch of the eval pass, mirroring
        # today's "example 0 of the eval pass" semantic.
        diag = is_pretrain_eval and mb_idx == 0

        # --- Forget pass: maximize CE on leaking continuation ---
        l_forget = _run_pass(
            pass_records["forget"],
            log_diagnostic=diag, pass_name="forget",
        )

        # --- Retain-clean: minimize CE on clean continuation (same prompt) ---
        l_retain_clean = None
        if ACTIVE_RETAIN_CLEAN:
            l_retain_clean = _run_pass(
                pass_records["retain_clean"],
                log_diagnostic=diag, pass_name="retain_clean",
            )

        # --- Retain-ctx: minimize CE on retain-set continuation ---
        l_retain_ctx = None
        if ACTIVE_RETAIN_CTX:
            l_retain_ctx = _run_pass(
                pass_records["retain_ctx"],
                log_diagnostic=diag, pass_name="retain_ctx",
            )

        # --- IDK pass: minimize CE on refusal response ---
        l_idk = None
        if ACTIVE_IDK:
            l_idk = _run_pass(
                pass_records["idk"],
                log_diagnostic=diag, pass_name="idk",
            )

        # --- Combined per-row loss (kept (B,) until final .mean()) ---
        if contrast_mode:
            assert l_idk is not None, (
                "contrast_mode active but l_idk is None — IDK should be "
                "in active passes when contrast_mode is on."
            )
            c_term = contrast_term(l_forget, l_idk, args.contrast_margin)
            n_margin_saturated += int((c_term == 0).sum().item())
            loss_per_row = c_term
        else:
            f_term = forget_term(l_forget, args.forget_margin)
            if args.forget_margin > 0:
                n_margin_saturated += int((f_term == 0).sum().item())
            loss_per_row = f_term
            if l_idk is not None:
                loss_per_row = loss_per_row + args.lambda_idk * l_idk
        if l_retain_clean is not None:
            r_term = retain_term(l_retain_clean, args.retain_margin)
            if args.retain_margin > 0:
                n_retain_saturated += int((r_term == 0).sum().item())
            loss_per_row = (
                loss_per_row + args.lambda_retain_clean * r_term
            )
        if l_retain_ctx is not None:
            loss_per_row = (
                loss_per_row + args.lambda_retain_ctx * l_retain_ctx
            )

        loss = loss_per_row.mean()
        if args.lambda_l2 > 0:
            # Standard mini-batch SGD L2: added ONCE per optimizer step,
            # post-mean. Old per-example loop added L2 once per example
            # step, so at --batch_size > 1 the effective L2 strength
            # relative to CE is 1/B of what it was; re-sweep lambda_l2
            # alongside lr when changing batch size.
            l2 = (soft_token.embedding ** 2).sum() / soft_token.n_tokens
            loss = loss + args.lambda_l2 * l2

        if not is_pretrain_eval:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Epoch accumulators: per-row CE summed across kept rows so the
        # epoch-end division by n_examples yields the same per-row mean
        # the old per-example loop reported.
        epoch_forget_loss += l_forget.detach().sum().item()
        if l_retain_clean is not None:
            epoch_retain_clean_loss += l_retain_clean.detach().sum().item()
        if l_retain_ctx is not None:
            epoch_retain_ctx_loss += l_retain_ctx.detach().sum().item()
        if l_idk is not None:
            epoch_idk_loss += l_idk.detach().sum().item()
            n_idk_examples += len(kept_examples)
        # ``loss_per_row`` is the combined (forget/contrast + retain + idk)
        # per-row tensor before .mean(); summing across rows gives the
        # same total this loop's old per-example branch tracked.
        epoch_total_loss += loss_per_row.detach().sum().item()
        n_examples += len(kept_examples)

    # Clean up any stale hooks
    remove_hooks(model)

    if n_examples > 0:
        avg_forget = epoch_forget_loss / n_examples
        avg_retain_clean = epoch_retain_clean_loss / n_examples
        avg_retain_ctx = epoch_retain_ctx_loss / n_examples
        # NaN (not 0.0) when l_idk was never computed this epoch — disambiguates
        # "the bank collapsed l_idk to zero" (which is a real outcome) from
        # "we didn't supervise IDK at all" (which is a config choice).
        avg_idk = (
            epoch_idk_loss / n_idk_examples
            if n_idk_examples > 0
            else float("nan")
        )
        avg_total = epoch_total_loss / n_examples
    else:
        avg_forget = avg_retain_clean = avg_retain_ctx = avg_total = 0.0
        avg_idk = float("nan")
        if not is_pretrain_eval:
            # No surviving examples means optimizer.step() never ran this
            # epoch — make this loud so a misconfigured run (max_seq_len
            # too tight, etc.) doesn't masquerade as silent successful
            # training.
            log_print(
                f"WARNING: epoch {epoch + 1} had zero surviving examples — "
                f"no optimizer steps executed. Check max_seq_len budgets "
                f"vs. forget data lengths."
            )

    # Average per-vector L2 norm of the soft-token block.  Computed as
    # the mean of per-row norms within the block so the metric is
    # comparable across n_tokens settings.
    avg_norm = soft_token.embedding.norm(dim=-1).mean().item()

    epoch_label = (
        "Pre-train eval (no updates)"
        if is_pretrain_eval
        else f"Epoch {epoch + 1}/{args.epochs}"
    )
    log_line = (
        f"{epoch_label} | "
        f"L_forget: {avg_forget:.4f} | "
        f"L_ret_clean: {avg_retain_clean:.4f} | "
        f"L_ret_ctx: {avg_retain_ctx:.4f} | "
        f"L_idk: {avg_idk:.4f} | "
        f"L_total: {avg_total:.4f} | "
        f"norm: {avg_norm:.3f} | "
        f"examples: {n_examples}"
    )
    if (args.forget_margin > 0 or contrast_mode) and n_examples > 0:
        sat_frac = n_margin_saturated / n_examples
        label = "contrast_sat" if contrast_mode else "margin_sat"
        log_line += f" | {label}: {sat_frac:.2f}"
    if args.retain_margin > 0 and n_examples > 0:
        retain_sat_frac = n_retain_saturated / n_examples
        log_line += f" | retain_sat: {retain_sat_frac:.2f}"
    if scheduler is not None:
        # log the lr that was active for THIS epoch (not the post-step value)
        current_lr = optimizer.param_groups[0]["lr"]
        log_line += f" | lr: {current_lr:.2e}"
    log_print(log_line)

    # Step the LR scheduler at the end of training epochs (not the pre-train
    # eval pass). Stepping after the log line means the lr we logged matches
    # the lr that was active during the epoch's optimizer.step() calls.
    if scheduler is not None and not is_pretrain_eval:
        scheduler.step()

    # Periodic checkpoint (skipped on the pre-train eval pass)
    if (not is_pretrain_eval and args.checkpoint_every > 0
            and (epoch + 1) % args.checkpoint_every == 0):
        ckpt_dir = f"{args.output_dir}/{args.split}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/embedding_epoch{epoch + 1}.pt"
        soft_token.save(ckpt_path)
        log_print(f"  Checkpoint saved to {ckpt_path}")

# Defensive: re-enable autograd in case the loop's last iteration was the
# eval pass (only happens when --epochs == 0, currently a non-canonical run
# but cheap to guard).
torch.set_grad_enabled(True)

# -------------------------------------------------------------------------
# Save trained soft token
# -------------------------------------------------------------------------

output_path = f"{args.output_dir}/{args.split}"
os.makedirs(output_path, exist_ok=True)

save_path = f"{output_path}/embedding.pt"
soft_token.save(save_path)
log_print(f"\nSoft token saved to {save_path}")
