#!/usr/bin/env python3
"""Final cross-family adjudication of the idea set, on baseline-protocol evidence.

Sends the complete baseline table (three ratios, pre/post, clean stratum, judge
agreement, censoring interval, known defects) to Hy4 and asks which ideas are
trustworthy enough to write up, which need more evidence, and which must be
rejected. Every idea gets a verdict; the trace is archived.

This is a REVIEW call, separate from the judge calls used as a metric.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/usr/local/workspace/redact")
STAGE = ROOT / "idea-stage"
TRACES = ROOT / ".aris/traces"
ENV_FILE = Path("/usr/local/workspace/.env")
ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
MODEL = "hy4"


def token():
    for k in ("Hy4_TOKEN", "HY4_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?Hy4_TOKEN\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("no token")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build():
    rnd = jload(STAGE / "idea_loop_latest.json")
    scores = jload(STAGE / "idea_scores.json")
    i05 = jload(STAGE / "i05_baseline_judge.json") if (STAGE / "i05_baseline_judge.json").exists() else None
    i04 = jload(STAGE / "i04_baseline_intervals.json") if (STAGE / "i04_baseline_intervals.json").exists() else None
    par = jload(STAGE / "metric_parity.json")

    rows = []
    for label, p in rnd["panels"].items():
        pre, post = p["pre"], p["post"]
        if pre.get("status") != "completed" or post.get("status") != "completed":
            continue
        cps = {k: v["leak"] for k, v in (pre.get("clean_leak_per_strategy") or {}).items()}
        rows.append(
            f"- {label} scenario (split {p['split']}): baseline leak {pre['leak_rate']} -> {post['leak_rate']}; "
            f"per-strategy pre {pre['per_strategy_leak']}, post {post['per_strategy_leak']}; "
            f"clean-cell leak {pre['clean_leak']} (n={pre['clean_n']}) -> {post['clean_leak']} (n={post['clean_n']}); "
            f"clean-cell per-strategy pre {cps}; collapse {pre['collapsed_rate']} -> {post['collapsed_rate']}; "
            f"truncated {pre['truncated_rate']} -> {post['truncated_rate']}; "
            f"retain ROUGE {pre['retain_rouge']} -> {post['retain_rouge']}")

    idea_lines = [f"- {r['id']} {r['title']}: score {r['score']}, D4={r['D4']}, accepted={r['accepted']}"
                  for r in scores["ranked"]]
    cov = [f"- {k}: {v['status']} | evidence={v['evidence']} | value={json.dumps(v['value'], ensure_ascii=False)[:300]}"
           for k, v in rnd["idea_coverage"].items()]
    anom = [f"- {a['id']} {a.get('panel','-')}: {a['kind']} | cause={a['cause']} | "
            f"resolution={a.get('resolution','open')}" for a in rnd["anomalies"]]

    system = ("You are the cross-family senior reviewer for a machine-unlearning evaluation paper. "
              "You did not run these experiments. Judge only what is supplied. Be adversarial but exact. "
              "Do not invent numbers or citations. Answer in English and end with one fenced JSON object.")
    user = f"""## Setting
Machine unlearning for large reasoning models. Target checkpoint `sangyon/LRM-target`
(DeepSeek-R1-Distill-Llama-8B fine-tuned on R-TOFU). Baselines are R2MU checkpoints trained on the same splits.
Metric baseline (verbatim from the repo): `entity_leak` = substring containment of the gold key phrase
anywhere in the response; reported as leak_rate per strategy, plus truncated_rate, answer_reached_rate,
retain ROUGE-L. Protocol: 40 forget / 20 retain / 4 strategies / DefaultCoT 512, others 256 / greedy / seed 1001.

## Baseline-protocol results (all numbers computed from stored generations)
{chr(10).join(rows)}

## Proposed (additive) metrics
clean-cell leak = leak rate inside the subset where the answer channel was reached, the text was not
degenerate, and the key phrase was not echoed from the question; censoring interval = [optimistic,
pessimistic] with truncated samples right-censored and the upper bound clamped to 1.0; coverage =
answer_reached_rate.

## Judge cross-check (Hy4 as a metric, not as a reviewer)
{json.dumps(i05['systems'], ensure_ascii=False) if i05 else 'not run'}

## Censoring interval on baseline rows
{json.dumps(((i04 or {}).get('intervals') or {}), ensure_ascii=False) if i04 else 'not run'}

## Idea set and frozen-rubric scores
{chr(10).join(idea_lines)}

## Coverage under the baseline protocol
{chr(10).join(cov)}

## Open anomalies
{chr(10).join(anom)}

## Detector-contract defects (I10 source)
{chr(10).join(f"- {d['id']}: {d['finding']} | verified={d['verified']}" for d in jload(STAGE / 'code_diagnosis.json')['defects'])}

## Disclosure: the clean stratum is conditioned on a post-treatment variable
Clean-cell leak conditions on "answer reached, not degenerate, no echo". After unlearning, collapse and
truncation rise sharply, so this conditioning may be a collider. Please state explicitly whether the
clean-cell comparison is valid evidence or should be replaced (and by what).

## Known defects
- ours-inv / ctrl-single adapters are mounted on TARGET_MODEL but were trained on a knowledge-injected base;
  they are excluded from every comparison above.
- The shipped paraphrase bank contains refusal paraphrases, not question paraphrases, so I09 cannot be reproduced.
- 28 of 37 runs deviate from the baseline protocol; they are labelled and excluded from the tables above.
- Docs disagree on whether the base knows forget01 (local NLL 1.885 vs a prior record of 5.4692).

## What I need
1. `overall_verdict`: one of TRUSTWORTHY, TRUSTWORTHY_WITH_CAVEATS, NOT_TRUSTWORTHY.
2. `ideas`: one object per idea id I01..I10 with {{"verdict":"TRUSTWORTHY|NEEDS_MORE_EVIDENCE|REJECT","why":"<=25 words"}}.
3. `strongest_result`: the single result you would put in the paper first, with the number.
4. `fatal_issue`: the one thing most likely to sink the paper.
5. `reviewer_identity`: the model you are and what you cannot verify.
Fenced JSON at the end."""
    return system, user


def call(system, user, tok, timeout=600):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "stream": False}).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    last = None
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(5 * (i + 1))
    raise SystemExit(f"review call failed: {last}")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    system, user = build()
    resp = call(system, user, token())
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    parsed = None
    for b in reversed(m):
        try:
            parsed = json.loads(b)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        mm = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(mm.group(0)) if mm else {}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    TRACES.mkdir(parents=True, exist_ok=True)
    tp = TRACES / f"{ts}-hy4-final-adjudication.json"
    trace = {"tier": "final-adjudication", "model_requested": MODEL, "model_returned": resp.get("model"),
             "response_id": resp.get("id"), "request": {"system": system, "user": user},
             "response_content": content, "parsed": parsed,
             "note": "authorization header omitted on purpose"}
    raw = json.dumps(trace, ensure_ascii=False, indent=2).encode()
    tp.write_bytes(raw)
    out = {
        "run_id": "lrm-loop-20261124", "tier": "final-adjudication",
        "reviewer_model": resp.get("model") or MODEL, "response_id": resp.get("id"),
        "trace_path": str(tp.relative_to(ROOT)), "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "overall_verdict": parsed.get("overall_verdict"),
        "ideas": parsed.get("ideas"), "strongest_result": parsed.get("strongest_result"),
        "fatal_issue": parsed.get("fatal_issue"), "reviewer_identity": parsed.get("reviewer_identity"),
    }
    (STAGE / f"idea_final_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (STAGE / "idea_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("overall_verdict", "strongest_result", "fatal_issue")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
