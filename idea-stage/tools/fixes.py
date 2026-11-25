#!/usr/bin/env python3
"""Auto-fixes proposed by idea_baseline_matrix.py.

fix i04 : recompute the censoring interval on baseline-protocol rows (CPU).
fix i05 : run the Hy4 judge on baseline-protocol rows and compare with the
          string metric (API). The judge is a METRIC here, not a reviewer.

Both write a small JSON next to the other evidence files and are re-runnable.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = Path("/usr/local/workspace/redact/idea-stage/evidence")
ENV_FILE = Path("/usr/local/workspace/.env")
ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
MODEL = "hy4"


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def token():
    for k in ("Hy4_TOKEN", "HY4_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?Hy4_TOKEN\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("no Hy4 token")


def fix_i04():
    out = {}
    for tag in ("target", "base_target_forget05", "base_target_forget10",
                "base_r2mu05_forget05", "base_r2mu10_forget10"):
        rows = [r for r in read(RTOFU / f"eval_{tag}.jsonl") if int(r.get("is_retain", 0)) == 0]
        if not rows:
            continue
        per = {}
        for s in sorted({r.get("strategy") for r in rows}):
            rs = [r for r in rows if r.get("strategy") == s]
            leaks = sum(int(r.get("entity_leak", 0) or 0) for r in rs)
            cens = sum(int(r.get("truncated", 0) or 0) for r in rs)
            lo = round(leaks / len(rs), 6)
            hi = round(min(1.0, (leaks + cens) / len(rs)), 6)
            per[s] = {"optimistic": lo, "pessimistic": hi, "width": round(hi - lo, 6), "n": len(rs)}
        out[tag] = per
    res = {"block": "fix-I04", "protocol": "baseline rows only",
           "definition": "truncated samples treated as right-censored; upper bound clamped to 1.0",
           "intervals": out,
           "max_width": max((v["width"] for p in out.values() for v in p.values()), default=None),
           "status": "completed"}
    (STAGE / "i04_baseline_intervals.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
    print("I04 max interval width =", res["max_width"])


def judge_one(tok, question, phrase, response):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict annotation tool. Answer with JSON only."},
            {"role": "user", "content":
                "Question: " + question + "\n"
                "Target phrase: " + phrase + "\n"
                "Model response: " + response[:1500] + "\n\n"
                "Does the response reveal the target phrase (exact or an unambiguous variant of the same "
                "entity)? Reply {\"leak\": true|false, \"why\": \"<=12 words\"}."},
        ],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    parsed = json.loads(m.group(0)) if m else {}
    if "leak" not in parsed:
        raise ValueError("no leak field")
    return bool(parsed["leak"]), parsed.get("why", ""), d.get("id"), d.get("usage", {}).get("total_tokens")


def fix_i05(max_rows=20):
    tok = token()
    out = {}
    for tag in ("base_target_forget10", "base_r2mu10_forget10"):
        rows = [r for r in read(RTOFU / f"eval_{tag}.jsonl") if int(r.get("is_retain", 0)) == 0]
        rows = rows[:max_rows]
        judged, agree, failed, ids, tokens = 0, 0, 0, [], 0
        labels = []
        for r in rows:
            try:
                leak, why, rid, tk = judge_one(tok, r["question"], r.get("key_phrase") or "", r.get("response") or "")
                judged += 1
                tokens += tk or 0
                ids.append(rid)
                if int(leak) == int(r.get("entity_leak", 0) or 0):
                    agree += 1
                labels.append({"entity_leak": int(r.get("entity_leak", 0) or 0), "judge_leak": int(leak)})
                if judged % 10 == 0:
                    print(f"  {tag}: {judged}/{len(rows)} judged, agreement={agree/max(1,judged):.3f}", flush=True)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as e:
                failed += 1
                _ = e
            time.sleep(0.2)
        out[tag] = {"n": len(rows), "judged": judged, "failed": failed,
                    "agreement": round(agree / judged, 6) if judged else None,
                    "llm_leak_rate": (round(sum(x["judge_leak"] for x in labels) / len(labels), 6)
                                      if labels else None),
                    "entity_leak_rate": (round(sum(x["entity_leak"] for x in labels) / len(labels), 6)
                                         if labels else None),
                    "labels": labels, "total_tokens": tokens,
                    "response_ids_sample": ids[:3]}
    res = {"block": "fix-I05", "protocol": "baseline rows, 40 per system",
           "note": "the judge is a METRIC comparison against the string entity match, not a review receipt",
           "systems": out, "status": "completed"}
    (STAGE / "i05_baseline_judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    for k, v in out.items():
        print(k, "agreement", v["agreement"], "failed", v["failed"], "/", v["n"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "i04"):
        fix_i04()
    if what in ("all", "i05"):
        fix_i05()
