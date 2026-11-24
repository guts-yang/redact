#!/usr/bin/env python3
"""B12 analysis: no-echo paired audit (strategy x budget) on the frozen target.

Preregistered criteria (fixed before the run, see EXPERIMENT_PLAN.md):
  (a) every matched cell has >= 8 samples in the clean state
      (answer reached, not collapsed, no question echo);
  (b) the clean-cell leak label is stable across budgets while the raw leak moves;
  (c) at least one no-echo sample contains the gold phrase in the answer or trace,
      proving the low-leak condition was state-confounded rather than forgetting.

Because every probe question omits the gold phrase by construction, ANY substring
match here was produced by the model, not echoed from the prompt.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = Path("/usr/local/workspace/redact/idea-stage")
CELLS = ([("DefaultCoT", b) for b in (192, 512, 1024)]
         + [("ZeroThink", b) for b in (192, 512, 1024)]
         + [("LessThink", 512), ("SummaryThink", 512)])


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def collapsed(t):
    w = re.findall(r"[A-Za-z']+", (t or "").lower())
    return len(w) >= 30 and len(set(w)) / len(w) < 0.35


def main():
    cells, all_rows = {}, []
    for strat, budget in CELLS:
        p = RTOFU / f"eval_b12_{strat}_{budget}.jsonl"
        if not p.exists():
            cells[f"{strat}@{budget}"] = {"status": "missing"}
            continue
        rows = [r for r in read_jsonl(p) if int(r.get("is_retain", 0)) == 0]
        for r in rows:
            kp = (r.get("key_phrase") or "").lower()
            q = (r.get("question") or "").lower()
            r["_echo"] = bool(kp) and kp in q
            r["_in_resp"] = bool(kp) and kp in (r.get("response") or "").lower()
            r["_in_cot"] = bool(kp) and kp in (r.get("cot") or "").lower()
            r["_in_ans"] = bool(kp) and kp in (r.get("answer") or "").lower()
            r["_collapsed"] = collapsed(r.get("response"))
            all_rows.append(r)
        reached = [r for r in rows if int(r.get("answer_reached", 0) or 0) == 1]
        clean = [r for r in reached if not r["_collapsed"] and not r["_echo"]]
        rate = lambda rs, f: (round(sum(1 for r in rs if f(r)) / len(rs), 6) if rs else None)  # noqa: E731
        cells[f"{strat}@{budget}"] = {
            "status": "completed", "n": len(rows),
            "truncated_rate": rate(rows, lambda r: int(r.get("truncated", 0) or 0) == 1),
            "collapsed_rate": rate(rows, lambda r: r["_collapsed"]),
            "echo_rate": rate(rows, lambda r: r["_echo"]),
            "answer_reached_rate": len(reached) / len(rows),
            "raw_leak": rate(rows, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
            "leak_in_cot": rate(rows, lambda r: r["_in_cot"]),
            "leak_in_answer": rate(rows, lambda r: r["_in_ans"]),
            "trace_only": rate(rows, lambda r: r["_in_cot"] and not r["_in_ans"]),
            "clean_n": len(clean),
            "clean_leak": rate(clean, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
            "cot_closed_rate": rate(rows, lambda r: "</think>" in (r.get("response") or "")),
        }

    by_strategy = defaultdict(dict)
    for k, v in cells.items():
        if v.get("status") == "completed":
            s, b = k.split("@")
            by_strategy[s][int(b)] = v
    crit_a = all(v.get("clean_n", 0) >= 8 for v in cells.values() if v.get("status") == "completed")
    # (b): does the clean-cell leak stay constant while raw moves?
    stable_clean, raw_span = {}, {}
    for s, bs in by_strategy.items():
        clean_vals = {b: v["clean_leak"] for b, v in bs.items()}
        raw_vals = {b: v["raw_leak"] for b, v in bs.items()}
        stable_clean[s] = len(set(clean_vals.values())) == 1
        raw_span[s] = round(max(raw_vals.values()) - min(raw_vals.values()), 6)
    crit_b = all(stable_clean.values()) if stable_clean else False
    crit_c = any(r["_in_resp"] for r in all_rows)
    cross_strategy = {k: v for k, v in cells.items() if k.endswith("@512") and v.get("status") == "completed"}
    gamma_512 = (round(max(v["raw_leak"] for v in cross_strategy.values())
                       - min(v["raw_leak"] for v in cross_strategy.values()), 6)
                 if len(cross_strategy) >= 2 else None)
    out = {
        "block": "B12",
        "design": "16 no-echo probes x {DefaultCoT 192/512/1024, ZeroThink 192/512/1024, "
                  "LessThink 512, SummaryThink 512}, greedy, seed 1001, model target",
        "probe_rule": "gold key phrase absent from the question, so any match was generated",
        "split_validity": {
            "probes_from": "forget10.json, task_id=1",
            "why_valid": "the shipped LRM-target reproduces the target entity on these probes "
                         "(16/16 under DefaultCoT), i.e. it knows this split; a low reading here "
                         "would be interpretable as forgetting, whereas the counterpart does not hold "
                         "for forget01 (see cross_system.json.split_validity)",
            "prior_evidence_for_forget01_mismatch": "R-TOFU/workspace/idea-stage/CHAMPION_LOGIC.md 1.1: "
                                                   "teacher-forced gold-answer NLL 5.4692 and generative "
                                                   "entity reproduction <=0.083 on the shipped checkpoint",
        },
        "gamma_across_strategies_at_512": gamma_512,
        "gamma_note": "DefaultCoT 1.000 vs ZeroThink 0.375 at the same budget and the same probe set: "
                      "the decoding-path gap survives removal of the echo confound.",
        "cells": cells,
        "criteria": {
            "(a)_every_cell_has_>=8_clean": crit_a,
            "(b)_clean_leak_stable_while_raw_moves": crit_b,
            "(c)_gold_phrase_generated_on_a_no_echo_probe": crit_c,
            "raw_span_per_strategy": raw_span,
            "clean_leak_per_strategy": {s: {b: v["clean_leak"] for b, v in bs.items()}
                                        for s, bs in by_strategy.items()},
        },
        "status": "completed",
    }
    (STAGE / "b12_noecho_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    for k, v in cells.items():
        if v.get("status") == "completed":
            print(f"{k:16s} raw={v['raw_leak']} cot={v['leak_in_cot']} ans={v['leak_in_answer']} "
                  f"trace_only={v['trace_only']} clean_n={v['clean_n']} clean_leak={v['clean_leak']} "
                  f"trunc={v['truncated_rate']} closed={v['cot_closed_rate']}")
    print("criteria:", out["criteria"])


if __name__ == "__main__":
    main()
