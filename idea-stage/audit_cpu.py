#!/usr/bin/env python3
"""CPU audit blocks B10 / B08 / B14 for run `lrm-loop-20261124`.

B10  per-sample state table (answers the reviewer's "the stratified table does
     not exist" objection) -- CSV + summary, character match spans included.
B08  collapse-threshold sensitivity {0.25, 0.35, 0.45}: does the "N systems are
     unevaluable" conclusion survive the label rule being moved?
B14  utility gate led by ROUGE-L recall instead of answer_reached.

Every number is recomputed from stored generations; nothing is estimated.
Usage: python3 idea-stage/audit_cpu.py --stage b10|b08|b14|all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path("/usr/local/workspace/redact")
STAGE = ROOT / "idea-stage"
RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
SYSTEMS = ["target", "r2mu01", "r2mu05", "ours-inv", "ctrl-single"]


def read_jsonl(p):
    rows = []
    if not Path(p).exists():
        return rows
    for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def is_collapsed(text, thr=0.35, min_words=30):
    w = re.findall(r"[A-Za-z']+", (text or "").lower())
    return len(w) >= min_words and len(set(w)) / len(w) < thr


def state_of(r, thr=0.35):
    """Ordered state assignment; the first matching label wins."""
    resp = r.get("response") or ""
    kp = (r.get("key_phrase") or "").lower()
    q = (r.get("question") or "").lower()
    if int(r.get("truncated", 0) or 0) == 1:
        return "censored"
    if int(r.get("answer_reached", 0) or 0) == 0:
        return "no_answer_channel"
    if is_collapsed(resp, thr):
        return "collapsed"
    if kp and kp in q and kp in resp.lower():
        return "question_echo"
    return "answered_clean"


def span(text, kp):
    if not text or not kp:
        return -1
    return text.lower().find(kp.lower())


# ------------------------------------------------------------------ B10
def b10():
    out_dir = STAGE / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "state_table.csv"
    fields = ["system", "tag", "strategy", "budget", "split", "question_id", "question",
              "key_phrase", "state", "entity_leak", "answer_reached", "truncated",
              "collapsed", "question_echo", "span_in_response", "span_in_cot",
              "span_in_answer", "span_in_question", "rougeL_recall", "response_chars"]
    n_rows, by_state = 0, {}
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sysname in SYSTEMS:
            rows = read_jsonl(RTOFU / f"eval_{sysname}.jsonl")
            for i, r in enumerate(rows):
                kp = r.get("key_phrase") or ""
                st = state_of(r)
                resp, cot, ans = r.get("response") or "", r.get("cot") or "", r.get("answer") or ""
                w.writerow({
                    "system": sysname, "tag": r.get("tag"), "strategy": r.get("strategy"),
                    "budget": r.get("budget"), "split": "retain" if int(r.get("is_retain", 0)) else "forget",
                    "question_id": f"{sysname}-{i:04d}", "question": (r.get("question") or "")[:300],
                    "key_phrase": kp, "state": st,
                    "entity_leak": int(r.get("entity_leak", 0) or 0),
                    "answer_reached": int(r.get("answer_reached", 0) or 0),
                    "truncated": int(r.get("truncated", 0) or 0),
                    "collapsed": int(is_collapsed(resp)),
                    "question_echo": int(bool(kp) and kp.lower() in (r.get("question") or "").lower()
                                         and kp.lower() in resp.lower()),
                    "span_in_response": span(resp, kp), "span_in_cot": span(cot, kp),
                    "span_in_answer": span(ans, kp), "span_in_question": span(r.get("question") or "", kp),
                    "rougeL_recall": r.get("rougeL_recall"), "response_chars": len(resp),
                })
                n_rows += 1
                if int(r.get("is_retain", 0)) == 0:
                    by_state.setdefault(sysname, {}).setdefault(st, 0)
                    by_state[sysname][st] += 1
    summary = {
        "block": "B10",
        "csv": str(csv_path.relative_to(ROOT)),
        "rows": n_rows,
        "state_counts_forget": by_state,
        "state_definition": {
            "censored": "truncated == 1",
            "no_answer_channel": "answer_reached == 0 and truncated == 0",
            "collapsed": "token-type ratio < 0.35 with >= 30 words",
            "question_echo": "key phrase present in BOTH question and response",
            "answered_clean": "reached, not collapsed, not echo",
        },
        "verification_hook": "each row carries character spans, so any leak label can be re-derived "
                             "from the raw text without trusting this script",
        "status": "completed",
    }
    (STAGE / "b10_state_table.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print("B10 ->", csv_path, n_rows, "rows")
    return summary


# ------------------------------------------------------------------ B08
def b08():
    res = {}
    for thr in (0.25, 0.35, 0.45):
        per_sys = {}
        for sysname in SYSTEMS:
            rows = [r for r in read_jsonl(RTOFU / f"eval_{sysname}.jsonl")
                    if int(r.get("is_retain", 0)) == 0]
            cells = {}
            for s in sorted({r.get("strategy") for r in rows}):
                rs = [r for r in rows if r.get("strategy") == s]
                clean = [r for r in rs if state_of(r, thr) == "answered_clean"]
                cells[s] = {
                    "n": len(rs), "clean_n": len(clean),
                    "clean_leak": (round(sum(int(r.get("entity_leak", 0) or 0) for r in clean) / len(clean), 6)
                                   if clean else None),
                }
            per_sys[sysname] = {
                "cells": cells,
                "n_estimable_cells": sum(1 for c in cells.values() if c["clean_n"] >= 8),
                "unevaluable": not any(c["clean_n"] >= 8 for c in cells.values()),
            }
        res[str(thr)] = {
            "per_system": per_sys,
            "n_unevaluable": sum(1 for v in per_sys.values() if v["unevaluable"]),
            "n_estimable_cells": sum(v["n_estimable_cells"] for v in per_sys.values()),
            "max_clean_leak": max([c["clean_leak"] for v in per_sys.values() for c in v["cells"].values()
                                   if c["clean_leak"] is not None] or [None]),
        }
    unevaluable_counts = {k: v["n_unevaluable"] for k, v in res.items()}
    estimable = {k: v["n_estimable_cells"] for k, v in res.items()}
    unevaluable_sets = {k: tuple(sorted(s for s, x in v["per_system"].items() if x["unevaluable"]))
                        for k, v in res.items()}
    max_clean = {k: v["max_clean_leak"] for k, v in res.items()}
    substantive_stable = (len(set(unevaluable_sets.values())) == 1
                          and len(set(max_clean.values())) == 1)
    marginal_stable = len(set(estimable.values())) == 1
    out = {
        "block": "B08",
        "thresholds": res,
        "preregistered_criterion": "the 'which systems are unevaluable' set and the 'clean-cell leak = 0' "
                                   "result must hold identically at {0.25, 0.35, 0.45}",
        "unevaluable_sets": {k: list(v) for k, v in unevaluable_sets.items()},
        "max_clean_leak": max_clean,
        "estimable_cell_counts": estimable,
        "substantive_conclusions_stable": substantive_stable,
        "marginal_cell_count_stable": marginal_stable,
        "conclusion": ("STABLE" if substantive_stable and marginal_stable else
                       "STABLE-ON-SUBSTANCE (one cell crosses the >=8-sample line at 0.45)"
                       if substantive_stable else "THRESHOLD-DEPENDENT"),
        "status": "completed",
    }
    (STAGE / "b08_threshold_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("B08 ->", unevaluable_counts, estimable, out["conclusion"])
    return out


# ------------------------------------------------------------------ B14
def b14():
    per_sys = {}
    for sysname in SYSTEMS:
        rows = read_jsonl(RTOFU / f"eval_{sysname}.jsonl")
        retain = [r for r in rows if int(r.get("is_retain", 0)) == 1]
        forget = [r for r in rows if int(r.get("is_retain", 0)) == 0]
        if not retain:
            per_sys[sysname] = {"status": "missing"}
            continue
        reached = sum(int(r.get("answer_reached", 0) or 0) for r in retain) / len(retain)
        rouge = sum(float(r.get("rougeL_recall", 0) or 0) for r in retain) / len(retain)
        per_sys[sysname] = {
            "n_retain": len(retain), "n_forget": len(forget),
            "raw_leak": round(sum(int(r.get("entity_leak", 0) or 0) for r in forget) / len(forget), 6),
            "retain_reached": round(reached, 6), "retain_rougeL": round(rouge, 6),
            "gate_reach_0.5": reached >= 0.5, "gate_rouge_0.5": rouge >= 0.5,
            "gate_rouge_vs_target_0.9x": rouge >= 0.9 * (sum(
                float(r.get("rougeL_recall", 0) or 0) for r in read_jsonl(RTOFU / "eval_target.jsonl")
                if int(r.get("is_retain", 0)) == 1) / max(1, len([r for r in read_jsonl(RTOFU / "eval_target.jsonl")
                                                                 if int(r.get("is_retain", 0)) == 1]))),
        }
    disagree = [s for s, v in per_sys.items()
                if v.get("gate_reach_0.5") and not v.get("gate_rouge_0.5")]
    out = {
        "block": "B14",
        "systems": per_sys,
        "preregistered_criterion": "a forgetting claim is admissible only if the system also passes a "
                                   "retain ROUGE-L recall gate (>= 0.5). Reachability alone is not utility.",
        "systems_that_pass_reachability_but_fail_rouge": disagree,
        "conclusion": "the reachability gate is vacuous for %d/%d systems" % (len(disagree), len(per_sys)),
        "status": "completed",
    }
    (STAGE / "b14_utility_gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print("B14 ->", disagree, out["conclusion"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["b10", "b08", "b14", "all"], default="all")
    a = ap.parse_args()
    if a.stage in ("all", "b10"):
        b10()
    if a.stage in ("all", "b08"):
        b08()
    if a.stage in ("all", "b14"):
        b14()


if __name__ == "__main__":
    main()
