"""
Generate model responses to R-TOFU dataset queries.

Unified generation script for the forget, retain, and TOFU auxiliary splits
(``real_authors`` / ``world_facts``). Output CSVs are consumed by
``scripts.evaluate_rtofu_from_csv`` for AFE / CFE / MU scoring.

Supports two inference-time modes on the canonical target model
(``sangyon/LRM-target``):

  * no corruption (baseline path used for retain / aux / baseline checkpoints);
  * canonical detect-and-regenerate (``--leak_classifier_path`` +
    ``--soft_token_path`` — soft-token corruption is the only
    inference-time corruption mode after the window-mode removal).

For baseline unlearning methods (e.g. R2MU) whose unlearning is baked into
the checkpoint itself, set ``baseline: true`` in the model's YAML config and
pass ``--model_name <baseline>``. The script then refuses any of the
inference-time defense flags so that double-unlearning runs are caught at
launch instead of silently misreported.

Default sampling mirrors ``evaluate_rtofu.py`` so CSVs line up row-for-row
with what that script would have generated in-process: protocol retain
splits are shuffled with ``--seed`` (default 0) and capped at 400 rows (the
MU subsample); every other split generates on all rows in file order. Pass
``--num_examples 10`` for quick inspection or ``--no-shuffle`` to force
in-order slicing on retain.

Usage:
    # Full forget generation (AFE / CFE inputs). Baseline of the frozen
    # target, or a pre-unlearned checkpoint whose YAML has ``baseline: true``.
    python -m scripts.generate_rtofu_responses --split forget10 \\
        --output results/target/forget10.csv
    python -m scripts.generate_rtofu_responses --model_name GA3+GD-forget10 \\
        --split forget10 --output results/baselines/GA3+GD/forget10.csv

    # MU inputs: protocol retain is shuffled + capped to 400 by default,
    # aux splits generate on all rows.
    python -m scripts.generate_rtofu_responses --split forget10_retain \\
        --output results/mu/forget10_retain.csv
    python -m scripts.generate_rtofu_responses --split real_authors \\
        --output results/mu/real_authors.csv
    python -m scripts.generate_rtofu_responses --split world_facts \\
        --output results/mu/world_facts.csv

    # Quick inspection (override the default full-split sampling).
    python -m scripts.generate_rtofu_responses --split forget10 --num_examples 10

    # Regeneration (canonical detect-and-regenerate). --leak_classifier_path
    # is the implicit "regen on" signal; the contextual_binary detector is
    # paired with a single shared SoftToken via --soft_token_path.
    python -m scripts.generate_rtofu_responses --split forget10 --num_examples 4 \\
        --leak_classifier_path leak_classifiers/forget10/contextual_labeled_full_contextual_binary \\
        --soft_token_path artifacts/soft_tokens/forget10/embedding.pt \\
        --show_cot
"""
import argparse
import csv
import os

from transformers import GenerationConfig

from redact.regen import RegeneratingModel
from redact.dataset.rtofu import RTOFU
from redact.inference import ReasoningGenerationEngine
from redact.model import HFModel, ReasoningModel
from redact.utils import log_print


_DEFENSE_FLAGS = (
    "leak_classifier_path",
    "soft_token_path",
)

# evaluate_rtofu.py's MU protocol subsamples the protocol retain to 400 rows
# after a seeded shuffle; no other split is subsampled. Mirror that here.
_PROTOCOL_RETAIN_NAMES = frozenset(RTOFU.match_retain.values())
_MU_RETAIN_SUBSAMPLE = 400


def _resolve_sample_defaults(args):
    """Fill in split-dependent defaults for --num_examples and --shuffle.

    Mirrors evaluate_rtofu.py: protocol retain splits (forgetXX_retain)
    default to a seeded shuffle capped at _MU_RETAIN_SUBSAMPLE rows; every
    other split defaults to full, in-file-order generation. Values the user
    set explicitly on the CLI are preserved. Mutates args in place and
    returns it for convenience.
    """
    is_protocol_retain = args.split in _PROTOCOL_RETAIN_NAMES
    if args.shuffle is None:
        args.shuffle = is_protocol_retain
    if args.num_examples is None:
        args.num_examples = _MU_RETAIN_SUBSAMPLE if is_protocol_retain else 0
    return args


def _validate_baseline_guard(model_config, args):
    """Reject inference-time defense flags on a ``baseline: true`` model config.

    Baseline unlearning methods (e.g. R2MU) apply the unlearning at training
    time; running them through the detect-and-regenerate stack or prompt
    corruption would silently double-unlearn and make results incomparable.
    Returns an error message string when the combination is invalid, or
    ``None`` when safe to proceed.
    """
    if not model_config.get("baseline", False):
        return None
    conflicting = [f"--{f}" for f in _DEFENSE_FLAGS if getattr(args, f) is not None]
    if not conflicting:
        return None
    return (
        "Model YAML has baseline=true (unlearning is baked into the "
        "checkpoint); inference-time unlearning flags are not allowed: "
        f"{', '.join(conflicting)}"
    )


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="LRM-target",
        help="Model YAML basename under --model_config_dir (default: LRM-target). "
             "For baseline unlearning methods, point at that method's YAML (which "
             "should set hf_name to the unlearned checkpoint and baseline: true).",
    )
    parser.add_argument(
        "--model_config_dir", type=str, default="./config/rtofu_model_config",
        help="Directory holding model YAML configs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="forget10",
        choices=RTOFU.subsets,
        help="R-TOFU split to query. Includes forget/retain, protocol retain "
             "(forgetXX_retain), and TOFU auxiliary splits (real_authors / "
             "world_facts) used for MU. Aux splits fetch from HuggingFace "
             "lazily on first use.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=1,
        help="R-TOFU task_id for the forget split (1 is canonical; see "
             "data/rtofu/PROVENANCE.md). Threads into RTOFU so the protocol "
             "retain (forgetXX_retain) and aux splits are built for this "
             "task_id; non-int values raise TypeError.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=None,
        help="Number of examples to generate. Default is split-dependent to "
             "match evaluate_rtofu.py's sampling: 400 for protocol retain "
             "splits (forgetXX_retain; MU subsample), all rows for every "
             "other split. 0 = all rows. Override (e.g. 10) for quick "
             "inspection.",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Start from this row. Applied after --shuffle, so offset=400 "
             "takes the next 400 rows of the seeded shuffle.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for --shuffle (default matches evaluate_rtofu.py --seed).",
    )
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=None,
        help="Shuffle the split with --seed before slicing. Default is "
             "split-dependent: enabled for protocol retain splits (matches "
             "evaluate_rtofu.py's MU shuffle-then-subsample), disabled "
             "otherwise. Use --no-shuffle to force in-file-order slicing.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--show_cot", action="store_true",
                        help="Also print the gold and generated chain-of-thought to stdout. "
                             "CoT columns (gold_cot, model_cot) are always written to the "
                             "output CSV regardless, so downstream AFE/CFE scoring works.")
    parser.add_argument("--output", type=str, default=None, help="Save results to CSV file")
    parser.add_argument("--classifier_threshold", type=float, default=0.99, help="Prompt classifier confidence threshold")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty for generation")
    parser.add_argument("--sample", action="store_true",
                        help="Use sampling instead of greedy. LRM-target's own "
                             "generation_config specifies do_sample=True, "
                             "temperature=0.6, top_p=0.95; greedy decoding on "
                             "this reasoning model degenerates into repeated "
                             "'Let me think...' loops that never emit </think>. "
                             "Default remains greedy for reproduction compatibility.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    # Leak detector
    parser.add_argument("--leak_classifier_path", type=str, default=None,
                        help="Path to trained leak classifier. Omit for plain "
                             "generation; pass to enable detect-and-regenerate.")
    parser.add_argument("--classifier_type", type=str, default="contextual_binary",
                        choices=["contextual_binary"],
                        help="Classifier type: contextual_binary (binary head "
                             "only, with prompt + k-window prior context; the "
                             "canonical leak detector, paired with a single "
                             "shared SoftToken).")
    parser.add_argument(
        "--leak_classifier_threshold", type=float, default=None,
        help="Binary-head confidence threshold for the leak classifier. "
             "Default None means 'use the threshold persisted on the "
             "checkpoint' (BinaryHeadConfig.binary_threshold), which is the "
             "operating point the training sweep settled on. Override only "
             "when intentionally probing a different operating point.",
    )
    # Regeneration. Enabled implicitly by --leak_classifier_path; soft-token
    # corruption is the only inference-time corruption mode (window mode was
    # removed). --soft_token_path is required when --leak_classifier_path
    # is set.
    parser.add_argument("--regen_max_attempts", type=int, default=3)
    parser.add_argument("--soft_token_path", type=str, default=None, help="Path to trained soft token embedding")
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help=(
            "Chunked regen: drops EOS'd / budget-capped rows between "
            "chunks of this many tokens.  Eliminates the slow-tail tax "
            "where a non-terminating row holds the batch open while "
            "EOS'd rows generate wasted padding.  Greedy-only (the "
            "project default GenerationConfig satisfies this).  Default "
            "unset = single-shot generate per wave iteration."
        ),
    )
    parser.add_argument(
        "--chunk_mode", type=str, default="kvcache",
        choices=["reprefill", "kvcache"],
        help=(
            "Which chunked path to use when --chunk_size is set.  "
            "'kvcache' (Phase 2, default) keeps the KV cache live "
            "across chunks: chunk 0 prefills under the soft-token "
            "hook, chunks 1+ are pure AR-decode steps with no "
            "re-prefill and no hook re-fire.  Byte-equivalent to "
            "single-shot on bf16 + flash_attention_2.  'reprefill' "
            "(Phase 1) re-prefills each chunk's full accumulated "
            "input — simple but pays prefill cost N times per "
            "slow-tail row, AND drifts from single-shot on bf16+FA2 "
            "(FA2 tile reduction order depends on input length).  "
            "Kept for diagnostic comparison only."
        ),
    )
    # Safety net: if someone renames or adds a defense-related argparse dest
    # without updating _DEFENSE_FLAGS, the baseline guard would silently miss
    # the new flag. Fail loudly at parser construction instead.
    registered_dests = {a.dest for a in parser._actions}
    missing = [f for f in _DEFENSE_FLAGS if f not in registered_dests]
    assert not missing, f"_DEFENSE_FLAGS references unknown argparse dests: {missing}"
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _resolve_sample_defaults(args)

    # Validate
    if args.leak_classifier_path is not None and args.soft_token_path is None:
        parser.error("--leak_classifier_path requires --soft_token_path")
    if args.soft_token_path is not None and args.leak_classifier_path is None:
        parser.error("--soft_token_path requires --leak_classifier_path")

    # Load model
    log_print(f"Loading model: {args.model_name}")
    generation_config = GenerationConfig(
        do_sample=args.sample,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    if args.sample:
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
    if args.repetition_penalty is not None:
        generation_config.repetition_penalty = args.repetition_penalty
    model = HFModel(
        model_name=args.model_name,
        config_path=args.model_config_dir,
        generation_config=generation_config,
    )

    # Baseline-YAML guard: reject inference-time defense flags on pre-unlearned
    # checkpoints. Must run after HFModel load (we read model_config["baseline"])
    # but before RegeneratingModel wrapping.
    guard_error = _validate_baseline_guard(model.model_config, args)
    if guard_error is not None:
        parser.error(guard_error)

    # Wrap with RegeneratingModel when regen is configured.
    needs_regen = args.leak_classifier_path is not None
    if needs_regen:
        soft_token = None
        if args.soft_token_path is not None:
            from redact.regen.learned_hooks import SoftToken
            soft_token = SoftToken.load(args.soft_token_path, embed_dim=model.model_config["embedding_dim"])
            log_print(f"Loaded soft token from {args.soft_token_path}")

        model = RegeneratingModel(
            model=model,
            soft_token=soft_token,
        )

    model = ReasoningModel(model)

    # Load dataset and select examples
    log_print(f"Loading R-TOFU split: {args.split}")
    # eos_token="" avoids appending the tokenizer's printable EOS
    # representation (e.g. "<｜end▁of▁sentence｜>") to gold answers.
    # The append is a legacy training-loop convention that doesn't
    # belong in eval — gen-side decode uses skip_special_tokens=True,
    # so the literal EOS string would only ever appear in gold, biasing
    # ROUGE recall, entailment, and cosine against gen.
    data_module = RTOFU(
        formatting_tokens=model.model_config.get("formatting_tokens"),
        eos_token="",
        task_id=args.task_id,
    )
    data_module.download()
    # download() populates forget/retain/protocol-retain splits from the
    # vendored JSONLs. TOFU auxiliary splits (real_authors / world_facts)
    # are fetched lazily; calling load_dataset_for_eval triggers the fetch
    # and populates self.dataset[args.split] (its return value is discarded
    # — the engine will call it again to produce the formatted dataset).
    if args.split in data_module.tofu_aux_subsets and args.split not in data_module.dataset:
        data_module.load_dataset_for_eval(args.split)
    if args.split not in data_module.dataset:
        # Guards the task_id gap: RTOFU.subsets (used for argparse choices)
        # lists every forgetXX_retain unconditionally, but download() omits
        # the split when the instance's task_id is invalid for that forget
        # series (e.g. --split forget10 / forget10_retain with --task_id 10,
        # since forget10 only carries task_ids 1..9). Error out with a clear
        # message rather than a bare KeyError.
        parser.error(
            f"Split {args.split!r} is not available in the loaded dataset. "
            f"Available: {sorted(data_module.dataset.keys())}"
        )

    ds = data_module.dataset[args.split]
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
        log_print(f"Shuffled split with seed={args.seed}")
    end = len(ds) if args.num_examples == 0 else min(args.offset + args.num_examples, len(ds))
    data_module.dataset[args.split] = ds.select(range(args.offset, end))

    # Generate via engine
    if needs_regen:
        from redact.regen.build_classifier import build_leak_classifier
        from redact.regen.leak_detector import CoTLeakDetector
        from redact.inference_regen import RegeneratingReasoningEngine

        leak_classifier = build_leak_classifier(
            classifier_type=args.classifier_type,
            classifier_path=args.leak_classifier_path,
            classifier_threshold=args.leak_classifier_threshold,
            batch_size=args.batch_size,
        )
        # CoTLeakDetector.__init__ resolves None → classifier.threshold
        # internally, so passing the user's flag through directly is enough.
        leak_detector = CoTLeakDetector(
            classifier=leak_classifier,
            classifier_threshold=args.leak_classifier_threshold,
        )
        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=data_module,
            subset_names=[args.split],
            answer_evaluator=[],
            batch_size=args.batch_size,
            leak_detector=leak_detector,
            regen_max_attempts=args.regen_max_attempts,
            chunk_size=args.chunk_size,
            chunk_mode=args.chunk_mode,
        )
        log_print(
            f"Regeneration enabled: max_attempts={args.regen_max_attempts}, "
            f"chunk_size={args.chunk_size}, chunk_mode={args.chunk_mode}"
        )
    else:
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=model.tokenizer,
            data_module=data_module,
            subset_names=[args.split],
            answer_evaluator=[],
            batch_size=args.batch_size,
        )

    log_print(f"\nGenerating on {end - args.offset} examples from split '{args.split}'")
    log_print("=" * 80)

    generations = engine._generate()
    data = generations[args.split]

    # Flatten batch lists
    prompts = [item for batch in data["prompt"] for item in batch]
    gold_answers = [item for batch in data["gold_answer"] for item in batch]
    gold_cots = [item for batch in data["gold_cot"] for item in batch]
    generated_cots = [item for batch in data["generated_cot"] for item in batch]
    generated_answers = [item for batch in data["generated_answer"] for item in batch]

    # Display results
    results = []
    for i in range(len(prompts)):
        idx = args.offset + i
        log_print(f"\n[{idx}] QUESTION:\n{prompts[i]}\n")
        log_print(f"GOLD ANSWER:\n{gold_answers[i]}\n")
        if args.show_cot:
            if gold_cots[i]:
                log_print(f"GOLD COT:\n{gold_cots[i]}\n")
            if generated_cots[i]:
                log_print(f"MODEL THINKING:\n{generated_cots[i].strip()}\n")
        log_print(f"MODEL RESPONSE:\n{generated_answers[i]}\n")
        log_print("-" * 80)

        # Always include CoT columns so evaluate_rtofu_from_csv's AFE/CFE
        # path (which reads r["gold_cot"] / r["model_cot"] unconditionally)
        # works for any CSV this script produces. Aux splits legitimately
        # carry empty gold_cot; MU-only consumers ignore these columns.
        row = {
            "index": idx,
            "question": prompts[i],
            "gold_answer": gold_answers[i],
            "response": generated_answers[i],
            "gold_cot": gold_cots[i],
            "model_cot": generated_cots[i],
        }
        results.append(row)

    # Save to CSV if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fieldnames = ["index", "question", "gold_answer", "response", "gold_cot", "model_cot"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar="\\",
            )
            writer.writeheader()
            writer.writerows(results)
        log_print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
