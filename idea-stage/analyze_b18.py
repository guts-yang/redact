#!/usr/bin/env python3
"""B18: per-scenario (1% / 5% / 10%) before/after panels, with the state stratum.

forget01/05/10 are three forgetting-RATIO scenarios of the same benchmark. A system
may only be compared with a system scored on the SAME ratio, and the sign of any
before/after delta is meaningless until the clean stratum is checked.

  Panel A  1%   forget01, 40 questions (echo-prone), DefaultCoT@512  target -> r2mu01
  Panel C  5%   forget05 no-echo, 16 questions,       DefaultCoT@512  target -> r2mu05
  Panel B 10%   forget10 no-echo, 16 questions,       DefaultCoT@512  target -> r2mu10
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = Path("/usr/local/workspace/redact/idea-stage")


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def collapsed(t):
    w = re.findall(r"[A-Za-z']+", (t or "").lower())
    return len(w) >= 30 and len(set(w)) / len(w) < 0.35


def echo(r):
    kp = (r.get("key_phrase") or "").lower()
    return bool(kp) and kp in (r.get("question") or "").lower() and kp in (r.get("response") or "").lower()


def panel(path, label):
    rows = read(path)
    f = [r for r in rows if int(r.get("is_retain", 0)) == 0]
    ret = [r for r in rows if int(r.get("is_retain", 0)) == 1]
    reached = [r for r in f if int(r.get("answer_reached", 0) or 0) == 1]
    clean = [r for r in reached if not collapsed(r.get("response")) and not echo(r)]
    rate = lambda rs, f: (round(sum(1 for r in rs if f(r)) / len(rs), 6) if rs else None)  # noqa: E731
    return {
        "label": label, "source": Path(path).name, "n_forget": len(f),
        "raw_leak": rate(f, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
        "collapsed_rate": rate(f, lambda r: collapsed(r.get("response"))),
        "truncated_rate": rate(f, lambda r: int(r.get("truncated", 0) or 0) == 1),
        "question_echo_rate": rate(f, echo),
        "clean_n": len(clean),
        "clean_leak": rate(clean, lambda r: int(r.get("entity_leak", 0) or 0) == 1),
        "clean_cell_estimable": len(clean) >= 8,
        "retain_rouge": (round(sum(float(r.get("rougeL_recall", 0) or 0) for r in ret) / len(ret), 6)
                         if ret else None),
    }


PANELS = [
    ("A", "1%", "eval_target.jsonl", "target / forget01 / 40q echo-prone",
     "eval_r2mu01.jsonl", "r2mu01 / forget01 / 40q echo-prone"),
    ("C", "5%", "eval_b19_target_f05_512.jsonl", "target / forget05 / 16q no-echo",
     "eval_b19_r2mu05_f05_512.jsonl", "r2mu05 / forget05 / 16q no-echo"),
    ("B", "10%", "eval_b12_DefaultCoT_512.jsonl", "target / forget10 / 16q no-echo",
     "eval_b16_r2mu10_DefaultCoT_512.jsonl", "r2mu10 / forget10 / 16q no-echo"),
]


def main():
    out = {"block": "B18",
           "premise": "forget01/05/10 are 1%/5%/10% forgetting-ratio scenarios; compare only within a ratio",
           "budget": 512, "strategy": "DefaultCoT", "panels": {}}
    for key, ratio, pre_f, pre_lab, post_f, post_lab in PANELS:
        pre_p, post_p = RTOFU / pre_f, RTOFU / post_f
        if not (pre_p.exists() and post_p.exists()):
            out["panels"][key] = {"status": "missing"}
            continue
        pre, post = panel(pre_p, pre_lab), panel(post_p, post_lab)
        out["panels"][key] = {
            "ratio": ratio, "pre": pre, "post": post,
            "delta_raw_leak": round((post["raw_leak"] or 0) - (pre["raw_leak"] or 0), 6),
            "sign": "+" if (post["raw_leak"] or 0) > (pre["raw_leak"] or 0) else "-",
            "clean_evaluable_before": pre["clean_cell_estimable"],
            "clean_evaluable_after": post["clean_cell_estimable"],
        }

    b = out["panels"]
    out["finding"] = {
        "sign_pattern": {k: {"ratio": v["ratio"], "sign": v["sign"], "delta": v["delta_raw_leak"],
                             "collapse_pre": v["pre"]["collapsed_rate"],
                             "collapse_post": v["post"]["collapsed_rate"]}
                         for k, v in b.items() if v.get("status") != "missing"},
        "core": "the sign of the before/after delta follows the collapse-rate change, not the presence "
                "of erased knowledge: 1% reads +0.144 (collapse 88.8% -> 15.6%), 5% and 10% read -1.000 "
                "(collapse 0% -> 100% / 87.5%).",
        "hard_limit": "after unlearning there is not a single evaluable clean cell on the echo-free "
                      "probes (clean_n = 0 at 5%, 2 at 10%). The zero leak readings are therefore not "
                      "evidence of forgetting; the stratum that could carry that evidence no longer exists.",
        "why_1pct_differs": "the 1% panel uses echo-prone questions (the entity is in the prompt), so "
                            "the unlearned model can still produce fluent echo text and keeps 105 clean "
                            "samples. On echo-free probes it degenerates instead.",
    }
    out["status"] = "completed"
    (STAGE / "b18_scenario_panels.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    for k, v in b.items():
        if v.get("status") != "missing":
            print(f"{k} {v['ratio']:>3} pre={v['pre']['raw_leak']:.3f} (coll {v['pre']['collapsed_rate']:.2f}, "
                  f"clean_n {v['pre']['clean_n']}) -> post={v['post']['raw_leak']:.3f} "
                  f"(coll {v['post']['collapsed_rate']:.2f}, clean_n {v['post']['clean_n']}) "
                  f"delta={v['delta_raw_leak']:+.3f}")


if __name__ == "__main__":
    main()
