#!/usr/bin/env python3
"""Metric / protocol parity check against the original R-TOFU baseline.

Rule (from the user, 2026-11-24): training and evaluation must stay on the
original baseline protocol and report the original metrics. New metrics may only
be ADDED, and only where they are the innovation claim -- never substituted for
the baseline ones.

Baseline protocol (code/paths.py, code/strategies.py, code/decode_eval.py):
  forget  : <split>.json, task_id = 1, n_forget = 40
  retain  : retain90.json, n_retain = 20
  decode  : do_sample=False, seed 1001, fp32, batch 8
  budget  : DefaultCoT 512, LessThink / ZeroThink / SummaryThink 256
  metrics : leak_rate, answer_leak_rate, rouge_mean, truncated_rate,
            answer_reached_rate            <- reported verbatim from the summary

Anything else is a DEVIATION and must be labelled as such in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = Path("/usr/local/workspace/redact/idea-stage")

BASE = {
    "n_forget": 40, "n_retain": 20, "seed": 1001, "do_sample": False,
    "budget": {"DefaultCoT": 512, "LessThink": 256, "ZeroThink": 256, "SummaryThink": 256},
    "n_strategies": 4,
}
# tags whose probe set is NOT the baseline forget file (recorded here because the
# runner does not persist the --forget-file path; this is itself a metadata defect)
NON_BASELINE_PROBES = ("b12_", "b16_", "b19_", "b13_")
INTENTIONAL_EXTRAS = {"b12_": "no-echo probe set + extra budgets (new metric experiment)",
                      "b13_": "no-echo probe set, seed sweep",
                      "b16_": "no-echo probe set, post-unlearning panel",
                      "b19_": "no-echo probe set, 5% panel"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def check(tag, s):
    cfg = s.get("config", {})
    dev, notes = [], []
    if tag.startswith(NON_BASELINE_PROBES):
        dev.append("probe_set")
        notes.append(INTENTIONAL_EXTRAS.get(tag[:4], "non-baseline probe file"))
    if cfg.get("n_forget") != BASE["n_forget"]:
        dev.append("n_forget")
    if cfg.get("n_retain") not in (BASE["n_retain"], 0):
        dev.append("n_retain")
    if cfg.get("seed") != BASE["seed"]:
        dev.append("seed")
    budgets = cfg.get("budgets") or []
    strat = cfg.get("strategies") or []
    if budgets:
        dev.append("budget_grid(DefaultCoT only)")
    else:
        ov = cfg.get("max_new_tokens_override") or 0
        for st in strat:
            want = BASE["budget"].get(st)
            got = ov or want
            if want is not None and got != want:
                dev.append(f"budget:{st}={got}(want {want})")
    if len(strat) < BASE["n_strategies"]:
        dev.append(f"strategies={len(strat)}/{BASE['n_strategies']}")
    return dev, notes


def main():
    rows = []
    for p in sorted(RESULTS.glob("eval_*.summary.json")):
        s = load(p)
        tag = s.get("tag") or p.stem.replace("eval_", "").replace(".summary", "")
        dev, notes = check(tag, s)
        groups = s.get("groups", {})
        baseline_metrics = {}
        for gname, g in groups.items():
            f = g.get("forget", {})
            r = g.get("retain", {})
            baseline_metrics[gname] = {
                "leak_rate": f.get("leak_rate"), "answer_leak_rate": f.get("answer_leak_rate"),
                "truncated_rate": f.get("truncated_rate"), "answer_reached_rate": f.get("answer_reached_rate"),
                "retain_rouge_mean": r.get("rouge_mean"),
            }
        rows.append({
            "tag": tag, "system": s.get("system"), "n_rows": s.get("n_rows"),
            "config": s.get("config"), "baseline_metrics": baseline_metrics,
            "parity": "ALIGNED" if not dev else "DEVIATION",
            "deviations": dev, "notes": notes,
        })
    aligned = [r for r in rows if r["parity"] == "ALIGNED"]
    deviated = [r for r in rows if r["parity"] == "DEVIATION"]
    out = {
        "block": "parity",
        "baseline_protocol": BASE,
        "rule": "baseline metrics are reported verbatim; new metrics are additive and must be "
                "labelled; a deviating run may not be compared with a baseline run without saying so",
        "n_runs": len(rows), "n_aligned": len(aligned), "n_deviated": len(deviated),
        "deviating_tags": [{"tag": r["tag"], "why": r["deviations"]} for r in deviated],
        "runs": rows,
        "metadata_defect": "decode_eval.py does not persist --forget-file in the summary, so the probe "
                           "set is reconstructed from the tag prefix. Fix: log the probe path hash.",
        "status": "completed",
    }
    (STAGE / "metric_parity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"runs={len(rows)} aligned={len(aligned)} deviated={len(deviated)}")
    for r in deviated:
        print(f"  DEVIATION {r['tag']}: {', '.join(r['deviations'])}")


if __name__ == "__main__":
    main()
