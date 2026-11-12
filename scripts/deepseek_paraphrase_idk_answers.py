"""Paraphrase IDK refusal answers with DeepSeek.

Same prompt, parser, and JSONL schema as ``scripts.paraphrase_idk_answers``.

Usage:
    # DEEPSEEK_API_KEY is read from the environment or the repo-root .env
    python -m scripts.deepseek_paraphrase_idk_answers --split forget10 --limit 1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time

from redact.utils import log_print

from scripts.deepseek_client import DeepSeekError, complete_with_retries, require_api_key
from scripts.paraphrase_idk_answers import (
    SPLIT_CHOICES,
    build_prompt,
    build_uid,
    load_existing_ok_uids,
    parse_paraphrases,
)


def paraphrase_one(
    uid: str,
    question: str,
    answer: str,
    k: int,
    model: str,
    max_retries: int,
    timeout: float,
) -> dict:
    prompt = build_prompt(question, answer, k)
    raw, err, attempts = complete_with_retries(
        prompt, model=model, timeout=timeout, max_retries=max_retries,
    )
    paraphrases = parse_paraphrases(raw, k) if raw is not None else None
    if paraphrases is not None:
        return {
            "uid": uid,
            "paraphrases": paraphrases,
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
        "paraphrases": None,
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
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument(
        "--input_idk", default=None,
        help="Override IDK JSON path. Default: "
             "data/reasoned_idk/reasoned_idk_<split>.json",
    )
    parser.add_argument(
        "--output_jsonl", default=None,
        help="Destination JSONL. Default: paraphrases/<split>.deepseek.jsonl",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    try:
        require_api_key()
    except DeepSeekError as exc:
        raise SystemExit(str(exc)) from exc

    output_jsonl = args.output_jsonl or f"paraphrases/{args.split}.deepseek.jsonl"
    input_idk = args.input_idk or f"data/reasoned_idk/reasoned_idk_{args.split}.json"
    log_print(f"Model: {args.model} | workers: {args.workers}")
    log_print(f"Loading IDK records from {input_idk}")
    with open(input_idk) as f:
        idk = json.load(f)
    log_print(f"  loaded {len(idk)} records")

    if args.indices is not None:
        idx_set = set(args.indices)
        idk = [r for r in idk if r["idx"] in idx_set]
        log_print(f"  filtered to --indices: {len(idk)}")
    if args.limit is not None:
        idk = idk[:args.limit]
        log_print(f"  limited to --limit: {len(idk)}")

    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    existing = load_existing_ok_uids(output_jsonl)
    log_print(f"Existing OK records: {len(existing)} (resuming)")
    to_process = [
        r for r in idk if build_uid(args.split, r["idx"]) not in existing
    ]
    log_print(f"Records to process: {len(to_process)}")
    if not to_process:
        log_print("Nothing to do.")
        return

    def _work(record: dict) -> dict:
        uid = build_uid(args.split, record["idx"])
        rec = paraphrase_one(
            uid=uid,
            question=record["question"],
            answer=record["reasoned_idk_answer"],
            k=args.k,
            model=args.model,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        rec["idx"] = record["idx"]
        rec["split"] = args.split
        rec["question"] = record["question"]
        rec["original_answer"] = record["reasoned_idk_answer"]
        return rec

    n_ok = n_fail = 0
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(output_jsonl, "a", encoding="utf-8") as out_f:
        futures = {ex.submit(_work, r): r for r in to_process}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            done = n_ok + n_fail
            if done % 10 == 0 or done == len(to_process):
                log_print(
                    f"  progress: {done}/{len(to_process)} "
                    f"(ok={n_ok}, fail={n_fail}, "
                    f"elapsed={time.perf_counter() - t0:.1f}s)"
                )
    log_print(f"\nDONE. ok={n_ok}, fail={n_fail}")
    log_print(f"Wrote to {output_jsonl}")


if __name__ == "__main__":
    main()
