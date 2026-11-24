#!/usr/bin/env python3
"""B18: per-scenario before/after panels.

forget01/05/10 are three forgetting-RATIO scenarios, not three checkpoints' moods.
A system may only be compared with a system evaluated on the SAME ratio, and the
same ratio must be read through the same generation state before the sign of a
before/after delta means anything.

Panel A: forget01, 40 questions, DefaultCoT@512   target -> r2mu01
Panel B: forget10, 16 no-echo probes, DefaultCoT@512 target -> r2mu10
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


def panel(rows, label):
    f = [r for r in rows if int(r.get("is_retain", 0)) == 0]
    ret = [r for r in rows if int(r.get("is_retain", 0)) == 1]
    return {
        "label": label, "n_forget": len(f),
        "raw_leak": round(sum(int(r.get("entity_leak", 0) or 0) for r in f) / len(f), 6),
        "collapsed_rate": round(sum(1 for r in f if collapsed(r.get("response"))) / len(f), 6),
        "truncated_rate": round(sum(int(r.get("truncated", 0) or 0) for r in f) / len(f), 6),
        "question_echo_rate": round(sum(
            1 for r in f if r.get("key_phrase") and r["key_phrase"].lower() in (r.get("question") or "").lower()
            and r["key_phrase"].lower() in (r.get("response") or "").lower()) / len(f), 6),
        "retain_rouge": (round(sum(float(r.get("rougeL_recall", 0) or 0) for r in ret) / len(ret), 6)
                         if ret else None),
        "questions": sorted({r.get("question") for r in f}),
    }


def main():
    a_pre = panel(read(RTOFU / "eval_target.jsonl"), "target / forget01 / 40q")
    a_post = panel(read(RTOFU / "eval_r2mu01.jsonl"), "r2mu01 / forget01 / 40q")
    b_pre = panel([r for r in read(RTOFU / "eval_b12_DefaultCoT_512.jsonl") if int(r.get("is_retain", 0)) == 0],
                  "target / forget10 no-echo / 16q")
    b_post = panel([r for r in read(RTOFU / "eval_b16_r2mu10_DefaultCoT_512.jsonl")
                    if int(r.get("is_retain", 0)) == 0],
                   "r2mu10 / forget10 no-echo / 16q")

    same_questions_a = set(a_pre["questions"]) == set(a_post["questions"])
    out = {
        "block": "B18",
        "premise": "forget01/05/10 are three forgetting-ratio scenarios of the same benchmark; "
                   "a system is only comparable with another system scored on the same ratio.",
        "panel_A_forget01": {"pre": a_pre, "post": a_post,
                             "delta_leak": round(a_post["raw_leak"] - a_pre["raw_leak"], 6),
                             "same_question_set": same_questions_a},
        "panel_B_forget10": {"pre": b_pre, "post": b_post,
                             "delta_leak": round(b_post["raw_leak"] - b_pre["raw_leak"], 6)},
        "finding": {
            "signs_are_opposite": "on forget01 the unlearned system reads HIGHER (0.025 -> 0.169); "
                                  "on forget10 it reads ZERO (1.000 -> 0.000)",
            "mechanism": "the forget01 base collapses (97.5% degenerate, 97.5% truncated), so the "
                         "unlearning gain is invisible while the unlearned system's lower collapse rate "
                         "raises the raw reading. A metric whose before/after sign is set by the base's "
                         "collapse level is not measuring forgetting.",
            "what_this_does_not_say": "it does not say R2MU failed on forget01; it says this metric "
                                      "cannot adjudicate it without the state stratum.",
        },
        "status": "completed",
    }
    (STAGE / "b18_scenario_panels.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    for k in ("panel_A_forget01", "panel_B_forget10"):
        p = out[k]
        print(k, "pre", p["pre"]["raw_leak"], "collapse", p["pre"]["collapsed_rate"],
              "| post", p["post"]["raw_leak"], "collapse", p["post"]["collapsed_rate"],
              "| delta", p["delta_leak"])


if __name__ == "__main__":
    main()
