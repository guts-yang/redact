"""Build soft-token training data from LLM-as-judge label data.

For each clean-forget CoT, selects the first sentence labelled ``leak``
that is attributed to a specific knowledge-bank claim (i.e.
``matched_claim_idx >= 0``; unattributed "uncertain leak" positives are
skipped). Assembles a
``(prompt, prefix, leaking_continuation, clean_continuation)``
triple per such CoT, with an optional reasoned-IDK companion.

Each ``clean_continuation`` is built as
``retain_cot_tail + "</think>\\n\\n" + retain_answer`` so the soft-token
retain term supervises a coherent end-of-response (CoT-tail plus
terminator plus answer; EOS is appended at tokenization time in
``train_soft_token``). Retain rows whose ``response`` is empty (model
never emitted ``</think>`` before max_new_tokens) are dropped — they
have no termination target.

Inputs:
  - ``--label_dataset``   HF DatasetDict with per-sentence label rows.
  - ``--forget_cots CSV`` clean-forget CoTs (from generate_rtofu_responses).
    cot_idx in the label dataset indexes into CSV row order.
  - ``--retain_cots CSV`` clean-retain CoTs — sampled for random
    ``clean_continuation`` tails. Both ``model_cot`` (the CoT) and
    ``response`` (the post-``</think>`` answer) columns are read.
    Retain CoTs whose labels contain any label=1 sentence are filtered
    out: using them as clean continuations would teach the soft token
    to suppress legitimate retain-prompt content where the model
    accidentally discussed a forget-set fact (about 1% of retain CoTs
    in practice).
  - ``--reasoned_idk``    Optional IDK JSON; entries keyed by forget idx.

Output:
  - ``<output>/train.jsonl`` — one JSON record per forget CoT with a
    confident leak, containing ``prompt``, ``prefix``,
    ``leaking_continuation``, ``clean_continuation``, ``forget_idx``,
    and optional ``reasoned_idk_{cot,answer}`` fields.

Usage:
    python -m scripts.build_soft_token_data \\
        --label_dataset leak_detector_data/forget10/labeled \\
        --forget_cots leak_detector_data/forget10/forget_cots.csv \\
        --retain_cots leak_detector_data/forget10/retain_cots.csv \\
        --reasoned_idk data/reasoned_idk/reasoned_idk_forget10.json \\
        --output soft_token_data/forget10
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

from datasets import concatenate_datasets, load_from_disk

from redact.evaluator.utils import split_sentences
from redact.utils import log_print, seed_everything


def _read_cot_csv(path: str) -> tuple[list[str], list[str], list[str]]:
    """Read a generate_rtofu_responses CSV.

    Returns parallel ``(prompts, cots, answers)`` lists in CSV row order.
    ``question`` is the formatted prompt; ``model_cot`` is the pre-``</think>``
    CoT; ``response`` is the post-``</think>`` answer (empty string if the
    model never emitted ``</think>`` before max_new_tokens).
    """
    prompts: list[str] = []
    cots: list[str] = []
    answers: list[str] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            prompts.append(row.get("question") or "")
            cots.append(row.get("model_cot") or "")
            answers.append(row.get("response") or "")
    return prompts, cots, answers


def collect_clean_forget_leaks(label_ds):
    """Collect ``cot_idx -> list[sent_idx]`` for CoT-phase leaks.

    Filters applied:
    - ``source == 'forget'`` AND ``source_tags == {'clean'}`` (skip
      reword_s0 / answer / corrupted / xfer variants AND
      retain/aux/idk rows).
    - ``phase == 'cot'`` (contextual schema labels both CoT and answer phases
      with independent sent_idx ranges; soft-token training data is CoT-
      redirection only). Legacy datasets without a ``phase`` column are
      treated as CoT-only by defaulting to ``"cot"``.
    - ``label == 1``
    - ``matched_claim_idx`` is not None and ``>= 0`` (the leak is attributed
      to a specific knowledge-bank claim).

    The ``matched_claim_idx`` predicate replaces the historical
    ``0 <= cluster_id < n_clusters`` cluster-range filter, which existed to
    skip the background sentinel (``n_clusters``) and the "uncertain leak"
    sentinel (``n_clusters + 1``). The two predicates were empirically
    verified set-identical over every ``label=1`` clean-forget CoT row of the
    canonical forget10 label dataset (0 mismatched rows, 0 first-leak-index
    differences, 399 CoTs under both). Do NOT "simplify" this filter away:
    dropping it yields 400 CoTs and breaks reproducibility of the published
    soft tokens.

    Returns a dict suitable for picking the first leak per CoT via
    ``min(leaks)``.

    Logs the count of rows dropped per non-cot phase value so a future schema
    addition (e.g., a hypothetical ``phase='thinking'``) is loudly visible
    rather than silently shrinking the dataset.
    """
    leaks = defaultdict(list)
    dropped_phases = Counter()
    for row in label_ds:
        # Exclude every forget-bucket variant (reword_s0, answer, corrupted)
        # and every non-forget primary. Soft-token training assumes the row
        # is a raw clean forget positive on the canonical cot_idx space.
        if not (
            row["source"] == "forget"
            and set(row["source_tags"]) == {"clean"}
        ):
            continue
        phase = row.get("phase", "cot")
        if phase != "cot":
            dropped_phases[phase] += 1
            continue
        if row["label"] != 1:
            continue
        matched_claim_idx = row["matched_claim_idx"]
        if matched_claim_idx is None or matched_claim_idx < 0:
            continue
        leaks[row["cot_idx"]].append(int(row["sent_idx"]))
    if dropped_phases:
        log_print(
            "Phase filter dropped clean forget rows: "
            + ", ".join(f"{p}={n}" for p, n in dropped_phases.most_common())
        )
    return leaks


def build_clean_continuations(
    retain_cots: list[str],
    retain_answers: list[str],
    leaky_retain_cot_idxs: set[int],
) -> tuple[list[str], dict[str, int]]:
    """Build the pool of clean continuations for the soft-token retain term.

    Each retain row contributes one continuation: a random tail of its CoT
    (sliced at a sentence boundary) concatenated with ``</think>\\n\\n``
    and the row's own ``response``. The terminator is what the retain
    term needs to teach the soft token — without it, retain_clean only
    pushes the model to keep producing reasoning text and never to
    finish, and the IDK term is the only termination signal.

    A row is dropped when:

    * its ``cot_idx`` is in ``leaky_retain_cot_idxs`` (retain CoT where
      the model accidentally mentioned a forget-set fact — using its
      tail would teach the bank to suppress legitimate retain content);
    * its CoT splits into fewer than 2 sentences (no valid split point);
    * its ``response`` is empty (model never emitted ``</think>`` before
      max_new_tokens — there is no termination target on this row).

    Returns ``(continuations, drop_stats)``. ``drop_stats`` carries the
    per-reason counts the caller logs.
    """
    if len(retain_cots) != len(retain_answers):
        raise ValueError(
            f"retain_cots ({len(retain_cots)}) and retain_answers "
            f"({len(retain_answers)}) must be parallel."
        )
    continuations: list[str] = []
    drop_stats = {"leaky": 0, "short": 0, "empty_answer": 0}
    for cot_idx, (cot, answer) in enumerate(zip(retain_cots, retain_answers)):
        if cot_idx in leaky_retain_cot_idxs:
            drop_stats["leaky"] += 1
            continue
        if not answer.strip():
            drop_stats["empty_answer"] += 1
            continue
        sentences = split_sentences(cot)
        if len(sentences) < 2:
            drop_stats["short"] += 1
            continue
        split_point = random.randint(1, len(sentences) - 1)
        tail = " ".join(sentences[split_point:])
        continuations.append(f"{tail}</think>\n\n{answer}")
    return continuations, drop_stats


def collect_leaky_retain_cot_idxs(label_ds) -> set[int]:
    """Return the set of ``cot_idx`` values for retain CoTs with >=1 leak label.

    Any row with ``source=='retain'``, ``"xfer" not in source_tags``, and
    ``label==1`` contributes its ``cot_idx`` to the set. xfer rows
    (out-of-subset forget prompts repurposed as retain signal) carry a
    DIFFERENT cot_idx index space (they index into the forget CoT pool,
    not the retain pool); folding them in here would collide with
    forget cot_idxs and misroute the continuation pool. Used to filter
    the clean-continuation pool so a retain CoT where the model
    accidentally discussed a forget-set fact doesn't end up teaching
    the soft token to suppress legitimate retain content.
    """
    leaky: set[int] = set()
    leaky_rows: list[dict] = []
    for row in label_ds:
        if (
            row["source"] == "retain"
            and "xfer" not in row["source_tags"]
            and row["label"] == 1
        ):
            leaky.add(int(row["cot_idx"]))
            leaky_rows.append(row)
    # Belt-and-suspenders: the predicate above already excludes xfer rows.
    # The assert catches a future regression where the xfer clause is dropped.
    assert all("xfer" not in r["source_tags"] for r in leaky_rows), (
        "xfer rows must never enter leaky_retain "
        "(different cot_idx index space)"
    )
    return leaky


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dataset", type=str, required=True,
                        help="Path to HF DatasetDict with per-sentence "
                             "label rows (e.g. leak_detector_data/forget10/labeled)")
    parser.add_argument("--forget_cots", type=str, required=True,
                        help="CSV of clean forget CoTs. "
                             "cot_idx in label_dataset must index into CSV row order.")
    parser.add_argument("--retain_cots", type=str, required=True,
                        help="CSV of clean retain CoTs for random clean_continuation tails.")
    parser.add_argument("--reasoned_idk", type=str, default=None,
                        help="Optional path to reasoned IDK JSON. "
                             "Entries are matched by 'idx' against forget cot_idx.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory; train.jsonl is written inside.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Load label dataset — concatenate train+test. The train/test split is a
    # classifier-training concern (held-out eval); soft-token training has no
    # held-out concept and using only the train split silently drops 20% of
    # forget CoTs from the soft-token training corpus.
    log_print(f"Loading label dataset from {args.label_dataset}")
    ds = load_from_disk(args.label_dataset)
    if "test" in ds:
        label_ds = concatenate_datasets([ds["train"], ds["test"]])  # type: ignore[arg-type]
        log_print(f"Label dataset rows: {len(label_ds)} (train+test concatenated)")
    else:
        label_ds = ds["train"]
        log_print(f"Label dataset rows: {len(label_ds)} (train only)")

    clean_forget_leaks = collect_clean_forget_leaks(label_ds)

    log_print(f"Forget CoTs with >=1 confident leak: {len(clean_forget_leaks)}")

    # Load forget prompts + CoTs from CSV
    log_print(f"Loading forget CoTs from {args.forget_cots}")
    forget_prompts, forget_cots, _ = _read_cot_csv(args.forget_cots)
    log_print(f"Forget CoTs: {len(forget_cots)} (prompts: {len(forget_prompts)})")

    # Load retain CoTs from CSV — answers needed so we can append the
    # `</think>\n\n` + answer terminator onto each clean_continuation, giving
    # the soft-token retain term a coherent end-of-response target rather
    # than just a CoT tail to keep flowing.
    log_print(f"Loading retain CoTs from {args.retain_cots}")
    _, retain_cots, retain_answers = _read_cot_csv(args.retain_cots)
    log_print(f"Retain CoTs: {len(retain_cots)}")

    # Load reasoned IDK (optional)
    idk_data: dict[int, dict] | None = None
    if args.reasoned_idk and os.path.exists(args.reasoned_idk):
        with open(args.reasoned_idk) as f:
            idk_list = json.load(f)
        idk_data = {e["idx"]: e for e in idk_list}
        log_print(f"Loaded reasoned IDK: {len(idk_data)} entries from {args.reasoned_idk}")
    elif args.reasoned_idk:
        log_print(f"No reasoned IDK at {args.reasoned_idk}; IDK fields omitted")
    else:
        log_print("No --reasoned_idk passed; IDK fields omitted")

    # Build forget triples — first leak per CoT
    log_print("\n--- Building forget triples ---")
    forget_triples = []
    out_of_range_cot_idxs: list[int] = []
    sentence_mismatches: list[tuple[int, int, int]] = []  # (cot_idx, sent_idx, n_sentences)
    for cot_idx, leaks in clean_forget_leaks.items():
        if cot_idx >= len(forget_cots):
            out_of_range_cot_idxs.append(cot_idx)
            continue
        first_sent_idx = min(leaks)
        sentences = split_sentences(forget_cots[cot_idx])
        if first_sent_idx >= len(sentences):
            # Sentence splitter disagreement between classifier build and now.
            sentence_mismatches.append((cot_idx, first_sent_idx, len(sentences)))
            continue
        prefix = " ".join(sentences[:first_sent_idx])
        leaking_continuation = " ".join(sentences[first_sent_idx:])
        triple = {
            "prompt": forget_prompts[cot_idx],
            "prefix": prefix,
            "leaking_continuation": leaking_continuation,
            "forget_idx": cot_idx,
        }
        if idk_data and cot_idx in idk_data:
            triple["reasoned_idk_cot"] = idk_data[cot_idx]["reasoned_idk_cot"]
            triple["reasoned_idk_answer"] = idk_data[cot_idx]["reasoned_idk_answer"]
        forget_triples.append(triple)

    n_total = len(clean_forget_leaks)
    n_skipped = len(out_of_range_cot_idxs) + len(sentence_mismatches)
    log_print(
        f"Forget triples: {len(forget_triples)} "
        f"(skipped {n_skipped}/{n_total}: "
        f"{len(out_of_range_cot_idxs)} out-of-range cot_idx, "
        f"{len(sentence_mismatches)} sentence-count mismatches)"
    )

    # Silent row drops here are almost always a sign the CSV and the label
    # dataset were generated against different CoT orderings (e.g. a shuffled
    # or offset generate_rtofu_responses invocation). Fail loud past a small
    # threshold so the user sees the problem rather than training on near-empty
    # forget triples.
    MAX_SKIP_FRAC = 0.05
    if n_total > 0 and n_skipped / n_total > MAX_SKIP_FRAC:
        oor_sample = out_of_range_cot_idxs[:5]
        sent_sample = [c for c, _, _ in sentence_mismatches[:5]]
        raise SystemExit(
            f"ERROR: {n_skipped}/{n_total} "
            f"({100 * n_skipped / n_total:.1f}%) of labeled forget CoTs could "
            f"not be matched against {args.forget_cots} "
            f"(tolerance: {100 * MAX_SKIP_FRAC:.0f}%). "
            f"This usually means the CSV row order does not match the cot_idx "
            f"used when the labels were built (did generate_rtofu_responses "
            f"run with --shuffle or --offset?). "
            f"Out-of-range cot_idx sample: {oor_sample}. "
            f"Sentence-count-mismatch cot_idx sample: {sent_sample}."
        )

    if not forget_triples:
        log_print("ERROR: No forget triples built.")
        raise SystemExit(1)

    # Filter retain CoTs that contain any label=1 sentence. A retain CoT
    # where the model spontaneously mentioned a forget-set fact is useless
    # (and actively harmful) as a clean_continuation: training the soft
    # token on it would push the model to suppress legitimate retain-prompt
    # content. About 1% of retain CoTs fall into this bucket in practice.
    # If no clean_retain labels are in the dataset (e.g. a forget-only
    # labeled dataset), this is a no-op.
    leaky_retain_cot_idxs = collect_leaky_retain_cot_idxs(label_ds)
    if leaky_retain_cot_idxs:
        log_print(
            f"Filtering {len(leaky_retain_cot_idxs)} retain CoTs with >=1 "
            f"label=1 sentence from the clean_continuation pool"
        )

    # Clean continuations from retain CoTs
    log_print("\n--- Splitting retain CoTs at random positions ---")
    n_total_empty = sum(1 for a in retain_answers if not a.strip())
    log_print(
        f"Retain rows with empty `response` (any reason): "
        f"{n_total_empty}/{len(retain_answers)}"
    )
    clean_continuations, drop_stats = build_clean_continuations(
        retain_cots, retain_answers, leaky_retain_cot_idxs,
    )
    log_print(
        f"Clean continuations: {len(clean_continuations)} "
        f"(dropped: {drop_stats['leaky']} leaky retain, "
        f"{drop_stats['short']} <2-sentence CoTs, "
        f"{drop_stats['empty_answer']} empty-response rows)"
    )

    if not clean_continuations:
        raise SystemExit(
            f"ERROR: No clean continuations. Dropped: "
            f"{drop_stats['leaky']} leaky / "
            f"{drop_stats['short']} <2-sentence / "
            f"{drop_stats['empty_answer']} empty-response. "
            f"If empty-response dominates, the retain CoTs were generated "
            f"with an inadequate --max_new_tokens budget — the model never "
            f"emitted `</think>`, so there is no termination target on any "
            f"row. Re-run scripts.generate_rtofu_responses with a larger "
            f"--max_new_tokens before rebuilding."
        )

    # Pair and save
    log_print("\n--- Pairing and saving ---")
    training_examples = []
    n_idk = 0
    for triple in forget_triples:
        ex = {
            "prompt": triple["prompt"],
            "prefix": triple["prefix"],
            "leaking_continuation": triple["leaking_continuation"],
            "clean_continuation": random.choice(clean_continuations),
            "forget_idx": triple["forget_idx"],
        }
        if "reasoned_idk_cot" in triple:
            ex["reasoned_idk_cot"] = triple["reasoned_idk_cot"]
            ex["reasoned_idk_answer"] = triple["reasoned_idk_answer"]
            n_idk += 1
        training_examples.append(ex)

    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "train.jsonl")
    with open(output_file, "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")

    log_print(f"\nWrote {len(training_examples)} examples ({n_idk} with IDK) to {output_file}")


if __name__ == "__main__":
    main()
