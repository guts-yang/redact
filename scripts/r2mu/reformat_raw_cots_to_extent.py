"""Convert pre-generated forget-set CoTs into R2MU's extent JSONL schema.

Two source formats are supported:

* `--raw-cots PATH`: legacy `leak_detector_data/<split>/raw_cots.json` (parallel
  list schema with `forget_prompts`, `forget_cots`, `retain_prompts`, `retain_cots`),
  produced by a retired leak-label generation script.
* `--csv PATH`: newer `leak_detector_data_promptfmt/<split>/forget_responses.csv`
  schema with columns `index, question, gold_answer, response, gold_cot, model_cot`.
  Questions in this CSV are wrapped with the chat-format tokens
  `<｜User｜>...<｜Assistant｜>`; we strip the wrap so the output JSONL question
  matches the R-TOFU dataset's plain question field.

Output schema (JSONL, one object per line) is identical regardless of source:
    {"question": "...", "cot": "..."}

This script does NOT do the truncate-to-N-tokens + trigger-prefix transform;
that happens at training time inside `r2mu.format_extent_string` so the trigger
prefix stays coupled to the tokenizer actually in use during training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# csv module's default field cap is 128 KiB. Largest observed CoTs are ~6 KiB,
# so a smaller bump would suffice — 16 MiB is overkill but cheap insurance against
# longer future generations. Set process-globally at import; relevant for tests
# (`test_load_csv_handles_large_fields`) only because the import-time side
# effect runs before any test does.
csv.field_size_limit(2**24)

_CHAT_PREFIX = "<｜User｜>"
_CHAT_SUFFIX = "<｜Assistant｜>"


def _strip_chat_wrap(question: str) -> str:
    """Remove `<｜User｜>...<｜Assistant｜>` chat wrapping if present."""
    q = question
    if q.startswith(_CHAT_PREFIX):
        q = q[len(_CHAT_PREFIX):]
    if q.endswith(_CHAT_SUFFIX):
        q = q[: -len(_CHAT_SUFFIX)]
    return q


def _load_raw_cots(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["forget_prompts"]
    cots = data["forget_cots"]
    if len(prompts) != len(cots):
        raise ValueError(
            f"forget_prompts and forget_cots length mismatch: {len(prompts)} vs {len(cots)}"
        )
    return list(zip(prompts, cots))


def _load_csv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "question" not in rdr.fieldnames or "model_cot" not in rdr.fieldnames:
            raise ValueError(
                f"CSV {path} missing required columns; got fieldnames={rdr.fieldnames}. "
                "Expected at least `question` and `model_cot`."
            )
        for row in rdr:
            q = _strip_chat_wrap(row["question"])
            cot = row["model_cot"]
            pairs.append((q, cot))
    return pairs


def reformat(
    pairs: list[tuple[str, str]],
    output_path: Path,
    filter_questions: set[str] | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for q, c in pairs:
            if filter_questions is not None and q not in filter_questions:
                n_skipped += 1
                continue
            out.write(json.dumps({"question": q, "cot": c}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} extent records to {output_path}", file=sys.stderr)
    if filter_questions is not None:
        print(f"Filtered out {n_skipped} records not in the requested question set", file=sys.stderr)
    return n_written


def load_question_set(rtofu_split_json_path: Path, task_id: str | None = None) -> set[str]:
    """Load a set of question strings from an R-TOFU split JSONL.

    If `task_id` is provided, restrict to records with that task_id (stored as
    a string in the JSONL even when the user passes an int). For forget01/05/10
    each task_id is a non-overlapping author-group; use task_id="1" for the
    canonical baselines."""
    questions: set[str] = set()
    with rtofu_split_json_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if task_id is not None and rec.get("task_id") != task_id:
                continue
            questions.add(rec["question"])
    if task_id is not None and not questions:
        raise ValueError(
            f"No records with task_id={task_id!r} in {rtofu_split_json_path}; "
            "task_id is stored as a string in the vendored JSONLs."
        )
    return questions


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--raw-cots", type=Path,
        help="Path to a legacy leak_detector_data/<split>/raw_cots.json.",
    )
    src.add_argument(
        "--csv", type=Path,
        help="Path to a leak_detector_data_promptfmt/<split>/forget_responses.csv.",
    )
    p.add_argument("--output", required=True, type=Path,
                   help="Output extent.jsonl path.")
    p.add_argument("--filter-split-json", type=Path, default=None,
                   help="Optional R-TOFU split JSONL (e.g. data/rtofu/forget01.json). "
                        "If provided, only records whose question appears in this file are kept. "
                        "Useful for deriving forget01/forget05 extent data from forget10's source. "
                        "Requires --filter-task-id.")
    p.add_argument("--filter-task-id", default=None,
                   help="Restrict --filter-split-json to records with this task_id "
                        "(string match against the JSONL's task_id field). Required when "
                        "--filter-split-json is given; use \"1\" for the canonical R-TOFU baselines.")
    args = p.parse_args()
    if (args.filter_split_json is None) != (args.filter_task_id is None):
        # Forbid one without the other. Without --filter-task-id, --filter-split-json would
        # silently mix all 10 task_ids' questions and produce 10× the expected row count;
        # without --filter-split-json, --filter-task-id has nothing to filter.
        p.error("--filter-split-json and --filter-task-id must be provided together")

    if args.raw_cots is not None:
        pairs = _load_raw_cots(args.raw_cots)
        src_label = str(args.raw_cots)
    else:
        pairs = _load_csv(args.csv)
        src_label = str(args.csv)
    print(f"Loaded {len(pairs)} (question, cot) pairs from {src_label}", file=sys.stderr)

    filter_questions = None
    if args.filter_split_json is not None:
        filter_questions = load_question_set(args.filter_split_json, task_id=args.filter_task_id)
        print(
            f"Filter set: {len(filter_questions)} questions from "
            f"{args.filter_split_json} task_id={args.filter_task_id}",
            file=sys.stderr,
        )

    n = reformat(pairs, args.output, filter_questions=filter_questions)
    if n == 0:
        sys.exit("No records written. Check inputs.")


if __name__ == "__main__":
    main()
