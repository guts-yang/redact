"""Ensure the Hy4 AFE/CFE JSONLs cover every CSV row before aggregation.

Hy4 occasionally returns HTTP 500 or times out. The judge scripts are
resumable (``load_existing_ok_uids`` skips rows already scored OK), but
``evaluate_rtofu_from_csv`` requires ALL uids resolved and otherwise raises,
leaving no eval.json. Evaluation wrappers historically ignored the judge
return code, so chains continued and even wrote completion markers.

Usage:
    python -m scripts.ensure_hy4_judges \
        --csv <forget10.csv> --split forget10 \
        --afe <afe.hy4.jsonl> --cfe <cfe.hy4.jsonl> \
        [--rounds 3] [--workers 4] [--timeout 180]

Exit 0 only when both JSONLs have an OK record for every expected uid.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.evaluate_rtofu_from_csv import load_rows
from scripts.judge_afe_rtofu import build_uid, derive_treatment_config


def expected_uids(csv_path: str) -> set[str]:
    rows = load_rows(csv_path)
    treatment, config = derive_treatment_config(csv_path, None, None)
    uids = set()
    for i, row in enumerate(rows):
        try:
            idx = int(row.get("index", i))
        except (TypeError, ValueError):
            idx = i
        uids.add(build_uid(treatment, config, idx))
    return uids


def ok_uids(jsonl_path: str) -> set[str]:
    """Most-recent-wins, matching JudgeAFETuple / LLMJudgeRtofuScore."""
    latest: dict[str, dict] = {}
    p = Path(jsonl_path)
    if not p.is_file():
        return set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        uid = rec.get("uid")
        if uid is not None:
            latest[uid] = rec
    return {u for u, r in latest.items() if r.get("status") == "ok"}


def missing(jsonl_path: str, want: set[str]) -> set[str]:
    return want - ok_uids(jsonl_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--afe", required=True)
    ap.add_argument("--cfe", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    want = expected_uids(args.csv)
    if not want:
        print("[ensure] CSV produced no uids", flush=True)
        return 2

    jobs = [
        ("AFE", "scripts.hy4_judge_afe_rtofu", args.afe),
        ("CFE", "scripts.hy4_judge_cfe_rtofu", args.cfe),
    ]
    for label, module, path in jobs:
        for attempt in range(1, args.rounds + 1):
            gaps = missing(path, want)
            if not gaps:
                print(f"[ensure] {label}: {len(want)}/{len(want)} uids OK", flush=True)
                break
            print(f"[ensure] {label}: {len(gaps)} uids missing "
                  f"(round {attempt}/{args.rounds}); retrying", flush=True)
            rc = subprocess.call([
                sys.executable, "-m", module,
                "--csv", args.csv, "--split", args.split,
                "--output_jsonl", path,
                "--workers", str(args.workers),
                "--max_retries", "3",
                "--timeout", str(args.timeout),
            ])
            if rc != 0:
                print(f"[ensure] {label}: judge exited {rc}", flush=True)
            time.sleep(5)
        else:
            gaps = missing(path, want)
            print(f"[ensure] FAILED {label}: {len(gaps)} uids still unresolved "
                  f"after {args.rounds} rounds; first: {sorted(gaps)[:5]}", flush=True)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
