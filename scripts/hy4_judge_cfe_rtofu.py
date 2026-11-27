"""CFE judge (graded CoT leak score) via Hy4.

Same prompt, score parser, uid convention, and JSONL schema as
``scripts.judge_cfe_rtofu``; only the transport differs. Hy4 is this project's
reviewer/judge backend (see ``reviewer-override.md``), not a cross-family
substitute for the executor.

Usage:
    # Hy4_TOKEN is read from the environment or /usr/local/workspace/.env
    python -m scripts.hy4_judge_cfe_rtofu \\
        --csv results/redact/forget10.csv --split forget10 \\
        --output_jsonl rtofu_judge_scores/forget10/cfe.hy4.jsonl --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os

from redact.utils import log_print

from scripts.hy4_client import Hy4Error, complete_with_retries, require_token
from scripts.judge_cfe_rtofu import (
    SPLIT_CHOICES,
    SYSTEM_PROMPT,
    build_prompt,
    build_uid,
    derive_treatment_config,
    load_existing_ok_uids,
    parse_score,
)

DEFAULT_MODEL = "hy4"


def judge_one(
    uid: str,
    question: str,
    gold_answer: str,
    model_cot: str,
    model: str,
    max_retries: int,
    timeout: float,
) -> dict:
    prompt = build_prompt(question, gold_answer, model_cot)
    raw, err, attempts = complete_with_retries(
        prompt, model=model, timeout=timeout, max_retries=max_retries,
        system_prompt=SYSTEM_PROMPT,
    )
    score = parse_score(raw) if raw is not None else None
    if score is not None:
        return {
            "uid": uid,
            "score": score,
            "raw_output": raw,
            "model": model,
            "cost_usd": 0.0,
            "session_id": None,
            "status": "ok",
            "attempts": attempts,
        }
    status = "parse_failed" if raw is not None else "error"
    return {
        "uid": uid,
        "score": None,
        "raw_output": raw,
        "model": model,
        "cost_usd": 0.0,
        "session_id": None,
        "status": status,
        "attempts": attempts,
        "error": err if raw is None else "parse_failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        require_token()
    except Hy4Error as exc:
        raise SystemExit(str(exc)) from exc

    log_print(f"Model: {args.model} | workers: {args.workers}")
    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log_print(f"Loaded {len(rows)} rows from {args.csv} (split={args.split})")
    required_cols = {"question", "gold_answer", "model_cot"}
    if rows and (missing := required_cols - rows[0].keys()):
        raise SystemExit(
            f"CSV missing required columns: {sorted(missing)}; "
            f"got: {sorted(rows[0].keys())}"
        )

    treatment, config = derive_treatment_config(args.csv, args.treatment, args.config)
    log_print(f"Uid construction: treatment={treatment!r} config={config!r}")
    work: list[dict] = []
    for i, row in enumerate(rows):
        try:
            row_index = int(row.get("index", i))
        except (TypeError, ValueError):
            row_index = i
        work.append({
            "uid": build_uid(treatment, config, row_index),
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_cot": row["model_cot"],
        })
    if args.limit is not None:
        work = work[:args.limit]
        log_print(f"--limit: capped to {len(work)} samples")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    done = load_existing_ok_uids(args.output_jsonl)
    todo = [item for item in work if item["uid"] not in done]
    log_print(f"To do: {len(todo)} samples")
    if not todo:
        log_print("Nothing to do; exiting.")
        return

    n_ok = n_parse_fail = n_error = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                judge_one,
                item["uid"], item["question"], item["gold_answer"], item["model_cot"],
                args.model, args.max_retries, args.timeout,
            ): item["uid"] for item in todo
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            rec = fut.result()
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if rec["status"] == "ok":
                n_ok += 1
            elif rec["status"] == "parse_failed":
                n_parse_fail += 1
            else:
                n_error += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(todo):
                log_print(
                    f"  [{i + 1}/{len(todo)}] ok={n_ok} "
                    f"parse_fail={n_parse_fail} err={n_error}"
                )
    log_print(
        f"Done. {n_ok} scored / {n_parse_fail} parse-failed / {n_error} errored. "
        f"Output: {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
