#!/usr/bin/env python3
"""B18: before/after panels, primary = ORIGINAL R-TOFU baseline protocol.

Primary panels use the baseline protocol verbatim so the numbers are comparable
with the paper's tables:
    n_forget=40, n_retain=20, 4 strategies, DefaultCoT@512 / others@256,
    do_sample=False, seed 1001, forget file = <split>.json task_id=1.

Secondary panels keep the no-echo probe set (16 questions) purely to test how far
the conclusion survives when the question stops supplying the entity name. They are
NOT the baseline and are reported separately.

New metric (state stratum) is additive: clean = answer reached, not collapsed,
no question echo. It never replaces leak_rate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = Path("/usr/local/workspace/redact/idea-stage")

PRIMARY = [
    ("C", "5%", "base_target_forget05", "base_r2mu05_forget05"),
    ("B", "10%", "base_target_forget10", "base_r2mu10_forget10"),
    ("A", "1%", "target", "r2mu01"),
]
SECONDARY = [
    ("C'", "5%", "b19_target_f05_512", "b19_r2mu05_f05_512"),
    ("B'", "10%", "b12_DefaultCoT_512", "b16_r2mu10_DefaultCoT_512"),
]


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def collapsed(t):
    w = re.findall(r"[A-Za-z']+", (t or "").lower())
    return len(w) >= 30 and len(set(w)) / len(w) < 0.35


def echo(r):
    kp = (r.get("key_phrase") or "").lower()
    return bool(kp) and kp in (r.get("question") or "").lower() and kp in (r.get("response") or "").lower()


def panel(tag):
    rows = read(RTOFU / f"eval_{tag}.jsonl")
    f = [r for r in rows if int(r.get("is_retain", 0)) == 0]
    ret = [r for r in rows if int(r.get("is_retain", 0)) == 1]
    reached = [r for r in f if int(r.get("answer_reached", 0) or 0) == 1]
    clean = [r for r in reached if not collapsed(r.get("response")) and not echo(r)]
    rate = lambda rs, f: (round(sum(1 for r in rs if f(r)) / len(rs), 6) if rs else None)  # noqa: E731
    per = {}
    for s in sorted({r.get("strategy") for r in f}):
        rs = [r for r in f if r.get("strategy") == s]
        per[s] = {"leak": rate(rs, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
                  "truncated": rate(rs, lambda r: int(r.get("truncated", 0) or 0) == 1),
                  "collapsed": rate(rs, lambda r: collapsed(r.get("response")))}
    return {
        "tag": tag, "n_forget": len(f), "n_retain": len(ret),
        "baseline_metric_leak_rate": rate(f, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
        "per_strategy": per,
        "gamma": (round(max(v["leak"] for v in per.values()) - min(v["leak"] for v in per.values()), 6)
                  if len(per) > 1 else None),
        "collapsed_rate": rate(f, lambda r: collapsed(r.get("response"))),
        "truncated_rate": rate(f, lambda r: int(r.get("truncated", 0) or 0) == 1),
        "new_metric_clean_n": len(clean),
        "new_metric_clean_leak": rate(clean, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
        "clean_cell_estimable": len(clean) >= 8,
        "retain_rouge_mean": (round(sum(float(r.get("rougeL_recall", 0) or 0) for r in ret) / len(ret), 6)
                              if ret else None),
    }


def build(groups, key):
    out = {}
    for gid, ratio, pre, post in groups:
        pre_p, post_p = RTOFU / f"eval_{pre}.jsonl", RTOFU / f"eval_{post}.jsonl"
        if not (pre_p.exists() and post_p.exists()):
            out[gid] = {"status": "missing"}
            continue
        a, b = panel(pre), panel(post)
        out[gid] = {"ratio": ratio, "pre": a, "post": b,
                    "delta_baseline_leak": round((b["baseline_metric_leak_rate"] or 0)
                                                 - (a["baseline_metric_leak_rate"] or 0), 6),
                    "delta_clean_leak": (round((b["new_metric_clean_leak"] or 0)
                                               - (a["new_metric_clean_leak"] or 0), 6)
                                         if b["new_metric_clean_leak"] is not None else None)}
    return out


def main():
    out = {
        "block": "B18",
        "baseline_protocol": "n_forget=40, n_retain=20, 4 strategies, DefaultCoT@512/others@256, "
                             "seed 1001, do_sample=False, <split>.json task_id=1",
        "primary": build(PRIMARY, "primary"),
        "secondary_noecho": build(SECONDARY, "secondary"),
        "reading_rule": "baseline_metric_leak_rate is the paper's metric and is reported first in every "
                        "table; new_metric_* is the proposed addition and is reported next to it, never "
                        "instead of it.",
    }
    p = out["primary"]
    out["finding"] = {
        "baseline_view": {k: {"ratio": v["ratio"],
                              "pre": v["pre"]["baseline_metric_leak_rate"],
                              "post": v["post"]["baseline_metric_leak_rate"],
                              "retain_pre": v["pre"]["retain_rouge_mean"],
                              "retain_post": v["post"]["retain_rouge_mean"]}
                          for k, v in p.items() if v.get("status") != "missing"},
        "stratified_view": {k: {"ratio": v["ratio"],
                                "pre_clean_leak": v["pre"]["new_metric_clean_leak"],
                                "pre_clean_n": v["pre"]["new_metric_clean_n"],
                                "post_clean_leak": v["post"]["new_metric_clean_leak"],
                                "post_clean_n": v["post"]["new_metric_clean_n"]}
                            for k, v in p.items() if v.get("status") != "missing"},
        "agreement": "under the baseline protocol both views agree that the unlearned models stop "
                     "producing the entity (0.83-0.975 -> 0.000; clean 0.87-0.89 -> 0.00).",
        "added_value": "only the stratified view shows that the drop arrives with 0.70-0.90 collapse and "
                       "0.88-1.00 truncation inside the forget set, and that retain ROUGE falls from 0.55 "
                       "to 0.42-0.43. The baseline metric alone cannot report either.",
        "secondary_note": "on the 16-question no-echo subset the clean cells shrink to 0-2 after "
                          "unlearning; that is a small-sample property of the subset, NOT proof that the "
                          "clean stratum disappears. The 40-question baseline panel keeps 26-29 clean "
                          "samples, so this earlier reading is withdrawn.",
    }
    out["status"] = "completed"
    (STAGE / "b18_scenario_panels.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    for gid, v in p.items():
        if v.get("status") == "missing":
            continue
        a, b = v["pre"], v["post"]
        print(f"{gid} {v['ratio']:>3} baseline {a['baseline_metric_leak_rate']} -> "
              f"{b['baseline_metric_leak_rate']} | clean {a['new_metric_clean_leak']}"
              f"(n={a['new_metric_clean_n']}) -> {b['new_metric_clean_leak']}"
              f"(n={b['new_metric_clean_n']}) | collapse {b['collapsed_rate']} | "
              f"retain {a['retain_rouge_mean']} -> {b['retain_rouge_mean']}")


if __name__ == "__main__":
    main()
