"""R2MU unlearning entrypoint for R-TOFU splits.

Runs R2MU (Wang et al. EMNLP 2025) on `sangyon/LRM-target` for a given R-TOFU
forget split. Writes a HuggingFace-compatible checkpoint to the output directory
that our existing R-TOFU evaluation pipeline can consume.

See `redact/baselines/r2mu.py` for method details and documented divergences
from the original paper / repo.

Example:
    python -m scripts.r2mu.unlearn \
        --split forget10 \
        --model-path sangyon/LRM-target \
        --extent-path r2mu_data/forget10/extent.jsonl \
        --output-dir results/r2mu/forget10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the repo root is importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from redact.baselines.r2mu import (  # noqa: E402
    R2MUConfig,
    format_extent_string,
    freeze_model,
    get_trainable_params,
    make_control_vector,
    r2mu_train_step,
)
from redact.dataset.rtofu import RTOFU  # noqa: E402
from redact.utils import load_yaml, log_print  # noqa: E402


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_extent_jsonl(path: Path) -> list[dict]:
    """Load the reformatted extent JSONL produced by `reformat_raw_cots_to_extent.py`.

    Each record: {"question": str, "cot": str}.
    """
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_s1k_streams(
    prompt_prefix: str,
    prompt_suffix: str,
    answer_prefix: str,
    eos_token: str,
    min_len: int = 50,
) -> tuple[list[str], list[str]]:
    """Load s1K for both CoT-preservation streams, wrapped in chat-formatting tokens.

    Returns (assistant_texts, reasoning_texts) where:
      - assistant = `{prompt_prefix}{question}{prompt_suffix}{answer_prefix}{attempt}{eos_token}`
                    (answer-only; fed to the answer-preservation term)
      - reasoning = `{prompt_prefix}{question}{prompt_suffix}{answer_prefix}<think>{thinking}</think>{attempt}{eos_token}`
                    (full reasoning trace + answer; fed to paper's β·ℓ_CoT term)

    Wrapping with prefix/suffix mirrors how the rest of our codebase formats
    R-TOFU prompts (see `redact/dataset/rtofu.py:245-247`); without it, the
    frozen-model preservation reference comes from inputs that don't match
    the model's training distribution.

    Schema note: the R2MU paper/repo used `deepseek_attempt` and
    `deepseek_thinking_trajectory` (singular string). The dataset was later
    renamed to `attempt` and `thinking_trajectories` (plural, a list of
    trajectories per record); we take the first trajectory.

    Records where any required field is missing or shorter than `min_len`
    chars are skipped to avoid degenerate inputs. The two lists are
    index-aligned.
    """
    from datasets import load_dataset

    ds = load_dataset("simplescaling/s1K", split="train")
    assistant_texts: list[str] = []
    reasoning_texts: list[str] = []
    n_total = 0
    n_dropped = 0
    for rec in ds:
        n_total += 1
        question = rec.get("question", "")
        attempt = rec.get("attempt", "")
        trajectories = rec.get("thinking_trajectories") or []
        if not isinstance(question, str) or not isinstance(attempt, str) or not trajectories:
            n_dropped += 1
            continue
        thinking = trajectories[0]
        if not isinstance(thinking, str):
            n_dropped += 1
            continue
        if len(question) < min_len or len(attempt) < min_len or len(thinking) < min_len:
            n_dropped += 1
            continue
        wrapped_q = f"{prompt_prefix}{question}{prompt_suffix}{answer_prefix}"
        assistant_texts.append(f"{wrapped_q}{attempt}{eos_token}")
        reasoning_texts.append(
            f"{wrapped_q}<think>{thinking}</think>{attempt}{eos_token}"
        )
    if not assistant_texts:
        raise RuntimeError("s1K returned no suitable streams")
    log_print(
        f"  s1K: kept {len(assistant_texts)}/{n_total} records "
        f"(dropped {n_dropped} for missing/short fields, min_len={min_len})"
    )
    return assistant_texts, reasoning_texts


def format_rtofu_sample(
    question: str,
    answer: str,
    prompt_prefix: str,
    prompt_suffix: str,
    answer_prefix: str,
    eos_token: str,
) -> str:
    """Text representation of an R-TOFU record for the forget/retain streams.

    Wraps the question in the model's chat prefix/suffix and appends the
    answer (with answer_prefix and eos_token), mirroring
    `RTOFU.load_dataset_for_eval` at `redact/dataset/rtofu.py:245-247`.
    CoT text is deliberately excluded to stay under the 512-token cap and
    keep the activation target concentrated on the factual leak surface
    (question + answer) rather than the verbose reasoning trace.
    """
    return f"{prompt_prefix}{question}{prompt_suffix}{answer_prefix}{answer}{eos_token}"


# ---------------------------------------------------------------------------
# Training-step batch construction
# ---------------------------------------------------------------------------

def cycle(iterable, seed: int):
    """Infinite cycle over a list, reshuffling each pass with the given seed.

    Each stream should be passed a distinct seed so that parallel streams don't
    accidentally iterate in lockstep (which would couple `batch["forget"][i]`
    to `batch["retain"][i]` for any splits of matching length).
    """
    data = list(iterable)
    rng = random.Random(seed)
    while True:
        rng.shuffle(data)
        for item in data:
            yield item


def cycle_pairs(items_a, items_b, seed: int):
    """Cycle two lists *together* with a shared permutation, preserving alignment.

    Use for the s1K assistant/reasoning streams, which are index-aligned: the
    i-th assistant sample is the answer of the same s1K record whose full
    think-trace is the i-th reasoning sample.

    Validation is eager so that a length mismatch raises at call time rather
    than on first `next()` (which would only fail deep inside the training loop).
    """
    if len(items_a) != len(items_b):
        raise ValueError(f"cycle_pairs: lists must have equal length, got {len(items_a)} vs {len(items_b)}")
    return _cycle_pairs_generator(items_a, items_b, seed)


def _cycle_pairs_generator(items_a, items_b, seed: int):
    indices = list(range(len(items_a)))
    rng = random.Random(seed)
    while True:
        rng.shuffle(indices)
        for idx in indices:
            yield (items_a[idx], items_b[idx])


def take(stream, n):
    return [next(stream) for _ in range(n)]


def build_batch(
    forget_stream, retain_stream, extent_stream, s1k_pair_stream, batch_size: int
) -> dict:
    """Draw `batch_size` items from each stream.

    The s1K pair stream yields `(assistant, reasoning)` tuples so that within a
    batch, position i of both lists refers to the same underlying s1K record.
    """
    s1k_pairs = [next(s1k_pair_stream) for _ in range(batch_size)]
    return {
        "forget": take(forget_stream, batch_size),
        "retain": take(retain_stream, batch_size),
        "extent": take(extent_stream, batch_size),
        "assistant": [a for a, _ in s1k_pairs],
        "reasoning": [r for _, r in s1k_pairs],
    }


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(updated_model, tokenizer, output_dir: Path, subdir: str = "checkpoint-last") -> None:
    """Save the updated model + tokenizer under `output_dir/<subdir>/`."""
    out = output_dir / subdir
    out.mkdir(parents=True, exist_ok=True)
    # Save full state dict (safetensors sharded). Non-trained params fall back to
    # their initial values, which equal the frozen reference's weights.
    updated_model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    log_print(f"Checkpoint saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["forget01", "forget05", "forget10"])
    p.add_argument("--model-path", default="sangyon/LRM-target")
    p.add_argument(
        "--model-config-path",
        type=Path,
        default=_REPO_ROOT / "config" / "rtofu_model_config" / "LRM-target.yaml",
        help="YAML model config to source formatting_tokens (prompt_prefix/suffix etc.) from.",
    )
    p.add_argument("--extent-path", type=Path, required=True,
                   help="Reformatted extent JSONL (from reformat_raw_cots_to_extent.py).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-id", type=int, default=1,
                   help="R-TOFU task_id (author-group offset) to unlearn. R-TOFU baselines use 1. "
                        "Matches the filter R-TOFU's forget.py applies: forget records with this task_id "
                        "become the forget set, records with other task_ids join the retain set.")
    # R2MU hyperparameters (defaults match the repo's run.sh).
    p.add_argument("--layer-id", type=int, default=7)
    p.add_argument("--trainable-layers", default="5,6,7")
    p.add_argument("--trainable-param-substring", default="mlp.down_proj.weight")
    p.add_argument("--steering-coeff", type=float, default=6.5)
    p.add_argument("--alpha", type=float, default=1.4,
                   help="Retain-loss weight (paper's γ). Repo default 1.4.")
    p.add_argument("--assist-coeff", type=float, default=1.0,
                   help="Weight on s1K answer-only preservation term.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight on s1K full-reasoning CoT-preservation term (paper's β). "
                        "The repo's bugged reasoning_loss, fixed here to actually contribute to the backward.")
    p.add_argument("--lr", type=float, default=7.5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-num-batches", type=int, default=None,
                   help="Override computed default. Default is min(500, ceil(|forget|/batch)).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=5)
    # Optional smoke-test aid: limit steps for a 5-min sanity run.
    p.add_argument("--smoke-test-steps", type=int, default=None,
                   help="If set, overrides max_num_batches and runs only this many steps (no checkpoint save).")
    args = p.parse_args()

    config = R2MUConfig(
        layer_id=args.layer_id,
        trainable_layer_ids=tuple(int(x) for x in args.trainable_layers.split(",")),
        trainable_param_substring=args.trainable_param_substring,
        steering_coeff=args.steering_coeff,
        alpha=args.alpha,
        assist_coeff=args.assist_coeff,
        beta=args.beta,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ---------------- model config + R-TOFU loading ----------------
    if not args.model_config_path.exists():
        sys.exit(f"Model config not found: {args.model_config_path}")
    if not args.extent_path.exists():
        sys.exit(f"Extent file not found: {args.extent_path}")
    model_config = load_yaml(args.model_config_path)
    formatting_tokens = model_config.get("formatting_tokens") or {}
    prompt_prefix = formatting_tokens.get("prompt_prefix", "")
    prompt_suffix = formatting_tokens.get("prompt_suffix", "")
    answer_prefix = formatting_tokens.get("answer_prefix", "")
    log_print(
        f"Formatting tokens from {args.model_config_path.name}: "
        f"prompt_prefix={prompt_prefix!r} prompt_suffix={prompt_suffix!r} "
        f"answer_prefix={answer_prefix!r}"
    )
    if not prompt_prefix and not prompt_suffix:
        log_print(
            "  WARNING: prompt_prefix and prompt_suffix are both empty. "
            "Forget/retain/s1K inputs will be unwrapped raw text — OOD for any "
            "chat-tuned model and the original cause of the R2MU collapse bug. "
            "Confirm this is intentional (e.g., training a base model)."
        )

    # RTOFU class handles task_id filtering and protocol-correct retain
    # construction (retain(100−n×XX) ∪ forgetXX[task_id≠K]) internally; see
    # redact/dataset/rtofu.py and CLAUDE.md "R-TOFU data loading and retain-split
    # naming" for the formula.
    log_print(f"Loading R-TOFU split={args.split} task_id={args.task_id}")
    rtofu = RTOFU(
        formatting_tokens=formatting_tokens,
        eos_token="",  # eos_token is appended manually below per-stream
        task_id=args.task_id,
    )
    rtofu.download()
    # We read the raw datasets directly (NOT via load_dataset_for_eval) so that
    # format_rtofu_sample can append eos_token itself per stream — this is why
    # eos_token="" was passed to RTOFU above. Don't switch to load_dataset_for_eval
    # without revisiting eos handling, or the trainer will see double-EOS inputs.
    forget_records = list(rtofu.dataset[args.split])
    retain_records = list(rtofu.dataset[RTOFU.match_retain[args.split]])
    extent_all = load_extent_jsonl(args.extent_path)

    # Extent: filter to records whose question is in the forget task's question set.
    # Our raw_cots covers forget10 task_id=1 (400 Qs about 20 authors); for
    # forget01/forget05 at task_id=1 the intersection is 40/200 respectively.
    forget_qs = {r["question"] for r in forget_records}
    extent_records = [r for r in extent_all if r["question"] in forget_qs]
    if not extent_records:
        sys.exit(
            f"Extent file {args.extent_path} contains 0 records matching the forget "
            f"task's {len(forget_qs)} questions. Re-generate extent for --task-id {args.task_id}."
        )
    extent_coverage = len({r["question"] for r in extent_records}) / max(len(forget_qs), 1)
    log_print(
        f"  |forget|={len(forget_records)} |retain|={len(retain_records)} "
        f"|extent|={len(extent_records)} (coverage {extent_coverage:.0%} of forget qs)"
    )
    if extent_coverage < 1.0:
        log_print(
            f"  WARNING: extent covers only {extent_coverage:.0%} of forget questions. "
            f"Unthink term's distribution will be narrower than the forget stream's. "
            f"Regenerate extent to match --task-id {args.task_id} for full coverage."
        )

    if args.max_num_batches is None:
        # One pass through forget, capped at paper's 500 batches.
        computed = min(500, math.ceil(len(forget_records) / config.batch_size))
        config.max_num_batches = computed
    else:
        config.max_num_batches = args.max_num_batches
    if args.smoke_test_steps is not None:
        config.max_num_batches = args.smoke_test_steps
    log_print(f"  max_num_batches={config.max_num_batches}")

    # ---------------- tokenizer ----------------
    log_print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # transformers >=5 mutates the byte-level BPE config on load — restore
    # the canonical Sequence(Split, ByteLevel) pre_tokenizer + ByteLevel
    # decoder so tokenizer.encode produces Ġ-prefixed tokens that round-trip
    # through decode with proper spaces.
    from redact.model.model import _maybe_repair_byte_level_tokenizer
    repaired = _maybe_repair_byte_level_tokenizer(tokenizer)
    log_print(f"  byte-level BPE repair fired: {repaired}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---------------- models ----------------
    log_print("Loading updated model (trainable)...")
    updated_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    log_print("Loading frozen reference model...")
    frozen_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    freeze_model(frozen_model)

    # Diagnostic: surface the named parameters of one block so the substring filter
    # hits something that exists. Failure modes in downstream training are otherwise
    # opaque ("no parameters matched...").
    first_block = updated_model.model.layers[config.trainable_layer_ids[0]]
    matches = [n for n, _ in first_block.named_parameters() if config.trainable_param_substring in n]
    log_print(f"Trainable-param matches in layer {config.trainable_layer_ids[0]}: {matches}")

    trainable_params = get_trainable_params(
        updated_model,
        config.trainable_layer_ids,
        config.trainable_param_substring,
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    log_print(f"Trainable parameter count: {n_trainable:,} across {len(trainable_params)} tensors")

    optimizer = AdamW(trainable_params, lr=config.lr)

    # ---------------- control vector ----------------
    hidden_dim = updated_model.config.hidden_size
    control_vec = make_control_vector(
        hidden_dim,
        config.steering_coeff,
        device="cuda",
        dtype=torch.bfloat16,
        seed=config.seed,
    )
    log_print(f"Control vector: shape={tuple(control_vec.shape)} steering_coeff={config.steering_coeff}")

    # ---------------- data streams ----------------
    eos_token = tokenizer.eos_token or ""
    forget_texts = [
        format_rtofu_sample(
            r["question"], r["answer"],
            prompt_prefix, prompt_suffix, answer_prefix, eos_token,
        )
        for r in forget_records
    ]
    retain_texts = [
        format_rtofu_sample(
            r["question"], r["answer"],
            prompt_prefix, prompt_suffix, answer_prefix, eos_token,
        )
        for r in retain_records
    ]
    # Extent is intentionally NOT wrapped in prompt_prefix/suffix: the trigger
    # string `"Let's reason this step by step.\n<think>"` already serves as the
    # in-context cue (the literal `<think>` is one of LRM-target's special tokens,
    # token id 128013). Matches upstream R2MU. The frozen-vs-updated MSE on the
    # extent stream is therefore in a slightly different distributional regime
    # than every other stream — a known asymmetry preserved for baseline parity.
    extent_texts = [format_extent_string(r["cot"], tokenizer, config) for r in extent_records]
    log_print("Loading s1K assistant + reasoning streams...")
    assistant_texts, reasoning_texts = load_s1k_streams(
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        answer_prefix=answer_prefix,
        eos_token=eos_token,
    )
    log_print(f"  |assistant|={len(assistant_texts)} |reasoning|={len(reasoning_texts)}")

    # Per-stream distinct seeds derived from config.seed so that streams don't
    # iterate in lockstep. s1K uses paired cycling to preserve assistant/reasoning
    # alignment (both come from the same underlying s1K record).
    base = config.seed
    forget_stream = cycle(forget_texts, seed=base + 1)
    retain_stream = cycle(retain_texts, seed=base + 2)
    extent_stream = cycle(extent_texts, seed=base + 3)
    s1k_pair_stream = cycle_pairs(assistant_texts, reasoning_texts, seed=base + 4)

    # ---------------- training loop ----------------
    log_print(f"Starting training: {config.max_num_batches} steps at batch_size={config.batch_size}")
    t0 = time.time()
    interrupted = False
    try:
        for step in range(config.max_num_batches):
            batch = build_batch(
                forget_stream, retain_stream, extent_stream,
                s1k_pair_stream, config.batch_size,
            )
            losses = r2mu_train_step(
                updated_model, frozen_model, tokenizer, batch, control_vec, optimizer, config
            )
            if step % args.log_every == 0 or step == config.max_num_batches - 1:
                elapsed = time.time() - t0
                mem_gb = torch.cuda.memory_allocated() / 1e9
                log_print(
                    f"step {step}/{config.max_num_batches} "
                    f"unlearn={losses['unlearn']:.4f} retain={losses['retain']:.4f} "
                    f"extent={losses['extent']:.4f} assistant={losses['assistant']:.4f} "
                    f"reasoning={losses['reasoning']:.4f} total={losses['total']:.4f} "
                    f"mem_gb={mem_gb:.2f} elapsed={elapsed:.1f}s"
                )
    except KeyboardInterrupt:
        interrupted = True
        log_print("\nInterrupted; attempting to save partial checkpoint.")

    # ---------------- save ----------------
    if args.smoke_test_steps is not None:
        log_print("Smoke-test run; skipping checkpoint save.")
    elif interrupted:
        save_checkpoint(updated_model, tokenizer, args.output_dir, subdir="checkpoint-interrupted")
    else:
        save_checkpoint(updated_model, tokenizer, args.output_dir)

    log_print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
