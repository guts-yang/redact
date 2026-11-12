"""First-leak labeling with DeepSeek.

Same prompt template, validation, and JSONL schema as
``scripts.label_first_leak_rtofu``.

Usage:
    # DEEPSEEK_API_KEY is read from the environment or the repo-root .env
    python -m scripts.deepseek_label_first_leak_rtofu --split forget10 --limit 1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
from pathlib import Path

from redact.dataset.contextual_window import clean_question
from redact.evaluator.utils import split_sentences
from redact.utils import log_print

from scripts.deepseek_client import DeepSeekError, complete_with_retries, require_api_key
from scripts.label_first_leak_rtofu import (
    PRIMARY_FIELDS,
    REPO,
    REQUIRED_CSV_COLS,
    SPLIT_CHOICES,
    build_prompt,
    load_existing_ok_uids,
    load_template,
    parse_label,
    render_author_claims_indexed,
    render_cot_indexed,
)


def label_one(
    uid: str,
    row_index: int,
    author: str,
    prompt: str,
    n_sentences: int,
    n_claims: int,
    model: str,
    max_retries: int,
    timeout: float,
) -> dict:
    raw, err, attempts = complete_with_retries(
        prompt, model=model, timeout=timeout, max_retries=max_retries,
    )
    parsed, reason = (
        parse_label(raw, n_sentences=n_sentences, n_claims=n_claims)
        if raw is not None else (None, err)
    )
    if parsed is not None:
        return {
            "uid": uid,
            "row_index": row_index,
            "author": author,
            "n_sentences": n_sentences,
            **{k: parsed[k] for k in PRIMARY_FIELDS},
            "raw_output": raw,
            "model": model,
            "cost_usd": 0.0,
            "session_id": None,
            "status": "ok",
            "attempts": attempts,
        }
    status = "validation_failed" if raw is not None else "error"
    last_err = f"parse/validation failed: {reason}" if raw is not None else err
    return {
        "uid": uid,
        "row_index": row_index,
        "author": author,
        "n_sentences": n_sentences,
        **{k: None for k in PRIMARY_FIELDS},
        "raw_output": raw,
        "model": model,
        "cost_usd": 0.0,
        "session_id": None,
        "status": status,
        "attempts": attempts,
        "error": last_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--output_jsonl", default=None,
        help="Default: leak_detector_data_promptfmt/<split>/first_leak_labels.deepseek.jsonl",
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        require_api_key()
    except DeepSeekError as exc:
        raise SystemExit(str(exc)) from exc

    split = args.split
    csv_path = Path(args.csv) if args.csv else (
        REPO / f"leak_detector_data_promptfmt/{split}/forget_responses.csv"
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else (
        REPO / f"leak_detector_data_promptfmt/{split}/first_leak_labels.deepseek.jsonl"
    )
    log_print(f"Model: {args.model} | workers: {args.workers}")

    template = load_template()
    with open(REPO / f"config/authors/{split}.json") as f:
        forget_data = json.load(f)
    with open(REPO / f"knowledge_banks/{split}/claims.json") as f:
        all_claims = json.load(f)
    n_claims = len(all_claims)

    author_claims_by_idx: dict[int, str] = {}
    author_name_by_idx: dict[int, str] = {}
    for author_idx, author in enumerate(forget_data["authors"]):
        author_claims_by_idx[author_idx] = render_author_claims_indexed(
            author["claim_ranges"], all_claims,
        )
        author_name_by_idx[author_idx] = author["name"]

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"CSV is empty: {csv_path}")
    missing = REQUIRED_CSV_COLS - set(rows[0].keys())
    if missing:
        sys.exit(f"CSV missing required columns {sorted(missing)}: {csv_path}")
    log_print(f"Loaded {len(rows)} rows from {csv_path}")

    work: list[dict] = []
    for row in rows:
        row_index = int(row["index"])
        author_idx = row_index // 20
        if author_idx not in author_name_by_idx:
            log_print(
                f"  WARN: row_index={row_index} maps to author_idx={author_idx} "
                f"which is not in config/authors/{split}.json; skipping"
            )
            continue
        cot = (row.get("model_cot") or "").strip()
        cot_sents = split_sentences(cot) if cot else []
        if not cot_sents:
            work.append({
                "uid": f"{split}_q{row_index:03d}",
                "row_index": row_index,
                "author": author_name_by_idx[author_idx],
                "empty_cot": True,
            })
            continue
        prompt = build_prompt(
            template=template,
            author_name=author_name_by_idx[author_idx],
            question=clean_question(row["question"]),
            author_claims_indexed=author_claims_by_idx[author_idx],
            cot_indexed=render_cot_indexed(cot_sents),
        )
        work.append({
            "uid": f"{split}_q{row_index:03d}",
            "row_index": row_index,
            "author": author_name_by_idx[author_idx],
            "prompt": prompt,
            "n_sentences": len(cot_sents),
        })

    if args.limit is not None:
        work = work[:args.limit]
        log_print(f"--limit: capped to {len(work)} samples")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing_ok_uids(str(output_jsonl))
    if done:
        log_print(f"Resuming: {len(done)} uids already scored OK")
    todo = [w for w in work if w["uid"] not in done]

    empty_records = []
    real_todo = []
    for item in todo:
        if item.get("empty_cot"):
            empty_records.append({
                "uid": item["uid"],
                "row_index": item["row_index"],
                "author": item["author"],
                "n_sentences": 0,
                "first_leak_idx": None,
                "matched_claim_idx": None,
                "claim_snippet": None,
                "subject_resolved": None,
                "rationale": "empty CoT — no sentences to label",
                "raw_output": None,
                "model": args.model,
                "cost_usd": 0.0,
                "session_id": None,
                "status": "ok",
                "attempts": 0,
            })
        else:
            real_todo.append(item)

    log_print(
        f"To do: {len(real_todo)} samples to judge; "
        f"{len(empty_records)} empty-CoT rows recorded directly."
    )
    if not real_todo and not empty_records:
        log_print("Nothing to do; exiting.")
        return

    n_ok = n_validation_fail = n_error = 0
    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for rec in empty_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
        out_f.flush()
        if real_todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        label_one,
                        item["uid"], item["row_index"], item["author"], item["prompt"],
                        item["n_sentences"], n_claims,
                        args.model, args.max_retries, args.timeout,
                    ): item["uid"] for item in real_todo
                }
                for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                    rec = fut.result()
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if rec["status"] == "ok":
                        n_ok += 1
                    elif rec["status"] == "validation_failed":
                        n_validation_fail += 1
                    else:
                        n_error += 1
                    if (i + 1) % 10 == 0 or (i + 1) == len(real_todo):
                        log_print(
                            f"  [{i + 1}/{len(real_todo)}] ok={n_ok} "
                            f"validation_fail={n_validation_fail} err={n_error}"
                        )
    log_print(
        f"Done. {n_ok} labeled / {n_validation_fail} validation-failed / "
        f"{n_error} errored. Output: {output_jsonl}"
    )


if __name__ == "__main__":
    main()
