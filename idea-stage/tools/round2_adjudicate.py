#!/usr/bin/env python3
"""Per-idea novelty + feasibility adjudication for the round-2 pool (Hy4)."""
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
STAGE = ROOT / "idea-stage/evidence"
TRACES = ROOT / ".aris/traces"
ENV = Path("/usr/local/workspace/.env")
ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
MODEL = "hy4"


def token():
    for k in ("Hy4_TOKEN", "HY4_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    for line in ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?Hy4_TOKEN\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("no token")


def main():
    base = json.loads((STAGE / "round2_scores_strict.json").read_text())
    lines = []
    for r in base["ranked"]:
        lines.append(
            f"- {r['id']} [{r['q']}] {r['title']} | measured={json.dumps(r['measured'], ensure_ascii=False)} "
            f"| threshold={r['threshold']} crossed={r['threshold_crossed']} | novelty_note={r['novelty'][:60]}")
    system = ("You are the cross-family senior reviewer for a machine-unlearning evaluation paper. "
              "You did not run these experiments. Judge novelty and feasibility strictly, only from "
              "what is supplied. Do not invent citations. Answer in English, end with one fenced JSON.")
    user = f"""## Context
LRM (reasoning-model) machine unlearning. Baseline metric: entity_leak (substring of the gold key
phrase), reported per decoding strategy; AFE = 1 - LEAK from a 4-way judge (LEAK/HALLUC/COLLAPSE/REFUSE);
CFE = 1 - mean graded CoT-leak score; MU = harmonic mean of 12 component means.
Measured facts in this repo: target leak 0.87-0.89 -> 0.00 after unlearning, but truncation rises to
95% and collapse to 80%; retain ROUGE falls 0.546 -> 0.425/0.433; judge parse failures 7.5-15%;
censoring intervals reach width 1.0; the shipped paraphrase bank holds refusals, not question paraphrases.

## Known external work (from live search)
- STaR (arXiv:2601.9281): Multi-Decoding Consistency Score over Default/Zero/LessThink for unlearning.
- UWC: "Towards Effective Evaluations and Comparisons for LLM Unlearning" (robust metrics framework).
- "Linguistic Loopholes in LLM Unlearning" (coverage-aware unlearning, language coverage).
- Judge-harness truncation bias write-ups (26-point swing when truncation handling changes).
- REDaCT ("Collapse-Free Unlearning"), K-Bench (six agent channels), ShadowBench (lexical anchoring).

## Candidates
{chr(10).join(lines)}

## What I need
For EACH idea id N01..N12:
  "verdict": SUPPORTED | WEAKENED | REFUTED  -- on NOVELTY, given the known external work
  "feasible": true|false  -- can it be reproduced with the assets described (stored generations +
              AFE label cache + file audit), without new model training
  "why": <= 25 words
Also: "top8": the 8 ids you would keep, ordered; "weakest_link": the single idea most likely to be
rejected for novelty, and why.
Fenced JSON at the end, keys: ideas, top8, weakest_link."""
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "stream": False}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token()}",
                                          "Content-Type": "application/json"})
    last = None
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read().decode())
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(5 * (i + 1))
    else:
        raise SystemExit(f"call failed: {last}")

    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    parsed = {}
    for b in reversed(blocks):
        try:
            parsed = json.loads(b)
            break
        except json.JSONDecodeError:
            continue
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    TRACES.mkdir(parents=True, exist_ok=True)
    tp = TRACES / f"{ts}-hy4-round2-novelty.json"
    raw = json.dumps({"model_returned": resp.get("model"), "response_id": resp.get("id"),
                      "request": {"system": system, "user": user}, "response_content": content,
                      "parsed": parsed}, ensure_ascii=False, indent=2).encode()
    tp.write_bytes(raw)
    out = {"reviewer_model": resp.get("model"), "response_id": resp.get("id"),
           "trace_path": str(tp.relative_to(ROOT)), "trace_sha256": hashlib.sha256(raw).hexdigest(),
           "ideas": parsed.get("ideas"), "top8": parsed.get("top8"),
           "weakest_link": parsed.get("weakest_link")}
    (STAGE / "cross_review_round2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({"top8": out["top8"], "weakest": out["weakest_link"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
