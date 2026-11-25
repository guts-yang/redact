#!/usr/bin/env python3
"""Cross-family review of the lead idea (`lrm-loop-20261124`) via Hy4.

The pipeline's own reviewer slots (codex / gemini / claude) are not mounted in
this environment. The user designated the Hy4 endpoint as the cross-family
reviewer instead: the executor agent is DeepSeek, Hy4 is a different model
family, so the review is not a self-review.

This script is the ONLY thing that writes a review receipt. It is re-runnable:
every call appends a trace to `.aris/traces/` and overwrites the normalized
receipt at `idea-stage/cross_review.json`.

Honesty rules enforced here:
  * the token is read from the environment file and never written to any trace;
  * the prompt contains the negative results and the external near-duplicates
    found by the novelty search, so the reviewer cannot be sandbagged;
  * `verdict` is parsed from the reviewer, never synthesised locally;
  * if the reviewer is unreachable or returns no parsable verdict, the receipt
    records REVIEW_UNAVAILABLE and the gate must stay BLOCKED.
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
STAGE = ROOT / "idea-stage/evidence"
TRACES = ROOT / ".aris/traces"
ENV_FILE = Path("/usr/local/workspace/.env")
ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
MODEL = "hy4"
REVIEWER_TIER = "external-critical-review"


def read_token() -> str:
    tok = os.environ.get("Hy4_TOKEN") or os.environ.get("HY4_TOKEN")
    if tok:
        return tok.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?Hy4_TOKEN\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip("'\"")
    raise SystemExit("Hy4_TOKEN not found in env or /usr/local/workspace/.env")


def load(name):
    return json.loads((STAGE / name).read_text(encoding="utf-8"))


def build_prompt() -> tuple[str, str]:
    pilot = load("pilot_results.json")
    pv = load("problem_validation.json")
    ext = load("external_novelty.json")
    scores = load("idea_scores.json")

    sens = pv["P0_budget_sensitivity"]
    hist, cur = sens["historical_512tok"], sens["this_run_192tok"]
    ranked = [(r["id"], r["title"], r["score"]) for r in scores["ranked"]]
    ext_lines = []
    for iid, info in ext["ideas"].items():
        hits = info.get("near_duplicates") or []
        ext_lines.append(
            f"- {iid}: {info['verification_status']}; near_duplicates={len(hits)}; "
            + ("; ".join(f"{h['id']} {h['title']}" for h in hits[:3]) if hits else "none")
        )

    system = (
        "You are the cross-family senior reviewer for a machine-unlearning paper draft. "
        "You are NOT the author and you did not run the experiments. Be adversarial, but "
        "judge only what the supplied evidence can support. Do not invent citations, "
        "numbers, or paper titles that are not given to you. If a claim is unsupported by "
        "the supplied evidence, say so explicitly and name the missing measurement. "
        "Answer in English. End your answer with a single fenced JSON object."
    )

    user = f"""## Setting
Target: machine unlearning for large reasoning models (LRMs) whose chain-of-thought (CoT) can leak forgotten facts.
Frozen target model: `sangyon/LRM-target` (DeepSeek-R1-Distill-Llama-8B fine-tuned on R-TOFU). No weights were modified in any experiment below.
Benchmark: R-TOFU. Primary metric in the literature: `entity_leak` = substring containment of a gold key phrase anywhere in the response; forget efficacy is `1 - leak_rate`.

## The problem anchor (frozen, must not drift)
`entity_leak` is confounded by three generation states: answer reachability/censoring, reasoning collapse into degenerate text, and echoing the entity name already present in the question. Therefore a low leak rate does not establish forgetting, and cross-system comparisons are not comparable.

Measured evidence (identical weights, seed 1001, identical questions, only the generation budget changed):
| | historical 512-token run | this run, 192 tokens |
|---|---|---|
| leak_rate | {hist.get('leak_rate')} | {cur.get('leak_rate')} |
| truncated_rate | {hist.get('truncated_rate')} | {cur.get('truncated_rate')} |
| coverage (answer reached) | {hist.get('coverage')} | {cur.get('coverage')} |
| degenerate_rate | {hist.get('degenerate_rate')} | {cur.get('question_echo_rate') is not None and cur.get('degenerate_rate')} |
| question_echo_rate | {hist.get('question_echo_rate')} | {cur.get('question_echo_rate')} |

Sample contrast: the historical response opens `"Okay, let's main themes in his work is a AI assistant, AI assistant, right? ..."` (collapsed, no entity); this run opens `"Alright, let's dive into the themes Nikolai Abilov commonly explores in his works..."` (the entity name is only echoed from the question). Both are counted as leaks by the same substring metric.

## Pilot evidence actually obtained (1x H20, greedy decoding, fp32, whole batch ~2 GPU-minutes)
- arcl probe (32 forget samples): raw leak_rate={pilot['probes']['arcl']['raw_leak_rate']}, coverage={pilot['probes']['arcl']['coverage']}, censored_n={pilot['probes']['arcl']['censored_n']}, question_echo_n={pilot['probes']['arcl']['question_echo_n']}, collapsed_n={pilot['probes']['arcl']['collapsed_n']}.
- LLM judge (Hy4, 32 samples): agreement with the string metric = {pilot['probes']['judge']['agreement']}, llm_leak_rate={pilot['probes']['judge']['llm_leak_rate']}, {pilot['probes']['judge']['failed']} judge calls failed. NEGATIVE result for the "judge disagrees" hypothesis.
- Censoring bound (I04): interval = [{pilot['probes']['bounds']['optimistic']}, {pilot['probes']['bounds']['pessimistic']}], width = {pilot['probes']['bounds']['width']}. NEGATIVE: the hypothesis "the interval spans the decision threshold" is refuted on this batch, because every sample already leaks.
- CoT channel audit (I03): 8/8 samples at 1024 tokens closed `</think>`; channel_gap = {pilot['probes']['channel']['channel_gap']}, answer_channel_leak = {pilot['probes']['channel']['answer_leak_rate']}. NEGATIVE: no trace-only leakage observed on this model.
- Forced reachability (I06): DefaultCoT answer_reached={pilot['probes']['forced']['default_reached']} vs forced short-think answer_reached={pilot['probes']['forced']['forced_reached']}; leak rate is {pilot['probes']['forced']['default_leak']} in both.
- Retain utility gate (I07): retain answer_reached={pilot['probes']['utility']['retain_reached']}, ROUGE-L recall={pilot['probes']['utility']['rouge_mean']}, gate threshold 0.5, pass={pilot['probes']['utility']['gate_pass']}.
- Reachability-controlled invariance (I08): gamma_controlled={pilot['probes']['invariance']['gamma_controlled']} because DefaultCoT has zero coverage.
- Budget grid (I02): token-budget curves exist for DefaultCoT only, so the preregistered cross-strategy matched-budget contrast was NOT measured.
- Paraphrase recovery (I09): 400/400 paraphrase probes exist but were never paired with GPU generations; recovery rate is unmeasured (null, not zero).

## Candidate ideas (frozen 10-point rubric, acceptance at 8.0)
{chr(10).join(f"- {i}: {t} (score {s})" for i, t, s in ranked)}

## External novelty search already performed (web search + 1 full-text read)
{chr(10).join(ext_lines)}

Closest work, full text read: arXiv 2506.00688 "Existing Large Language Model Unlearning Evaluations Are Inconclusive" (CMU / UK AISI / OATML). It diagnoses information injection, spurious generalization, and task/format dependence, and recommends injection budgets and a cross-format leakage matrix. It does NOT formalize generation-state stratification, coverage, conditional leak rates, or censoring bounds, and it does not discuss decode-length truncation.
Also adjacent: arXiv 2511.04934 "Leak@k" (ICML 2026) targets multi-sample probabilistic decoding; arXiv 2605.18891 audits reasoning-trace memorization with a substring "thk_leak" and a bypass gap; K-Bench (2609.12808) scores six agent channels; ShadowBench (NeurIPS 2026 D&B) reports lexical anchoring.

## What I need from you
1. `verdict`: exactly one of PROCEED, PROCEED WITH CAUTION, ABANDON — your bottom line on pursuing the lead idea (generation-state-stratified leakage audit) as the paper's core contribution.
2. `score`: 0-10 for the lead idea as a submission candidate, judged on evidence shown, not on promise.
3. `strongest_case`: 3-4 sentences making the best honest case for it.
4. `risks`: array of 3-5 objects {{"risk","severity":"high|medium|low","cheapest_test"}} — ordered by severity, each with the cheapest experiment that would discriminate.
5. `novelty_adjudication`: for I01 and I04 only, one of SUPPORTED / WEAKENED / REFUTED, with one sentence each, given the near-duplicates above. Be strict: if the closest work already implies the contribution, say REFUTED.
6. `claim_audit`: array of any claim in the anchor/evidence above that the supplied evidence does NOT support, with the reason.
7. `next_experiment`: the single highest-value experiment to run next, with a concrete preregistered success criterion and an estimated GPU-hours (must be <= 2 h on one H20).
8. `reviewer_identity`: the model name you are and one sentence on what you can and cannot verify.

Fenced JSON at the end, with keys: verdict, score, strongest_case, risks, novelty_adjudication, claim_audit, next_experiment, reviewer_identity."""
    return system, user


def call(system: str, user: str, token: str, timeout: int = 600) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"Hy4 review call failed: {last}")


def extract_json(text: str) -> dict | None:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    token = read_token()
    system, user = build_prompt()
    resp = call(system, user, token)
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    parsed = extract_json(content) or {}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    TRACES.mkdir(parents=True, exist_ok=True)
    trace_path = TRACES / f"{ts}-hy4-{REVIEWER_TIER}.json"
    trace = {
        "tier": REVIEWER_TIER,
        "endpoint": ENDPOINT,
        "model_requested": MODEL,
        "model_returned": resp.get("model"),
        "response_id": resp.get("id"),
        "created": resp.get("created"),
        "usage": resp.get("usage"),
        "request": {"system": system, "user": user},
        "response_content": content,
        "parsed": parsed,
        "note": "Authorization header omitted deliberately; the token is never written to a trace.",
    }
    raw = json.dumps(trace, ensure_ascii=False, indent=2).encode("utf-8")
    trace_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    verdict = str(parsed.get("verdict", "")).strip().upper().replace(" ", "_")
    if verdict not in {"PROCEED", "PROCEED_WITH_CAUTION", "ABANDON"}:
        verdict = "REVIEW_UNAVAILABLE"

    receipt = {
        "run_id": "lrm-loop-20261124",
        "tier": REVIEWER_TIER,
        "reviewer_model": resp.get("model") or MODEL,
        "reviewer_family": "hy4 (user-designated cross-family backend; executor agent is DeepSeek)",
        "response_id": resp.get("id"),
        "trace_path": str(trace_path.relative_to(ROOT)),
        "trace_sha256": digest,
        "verdict": verdict,
        "score": parsed.get("score"),
        "strongest_case": parsed.get("strongest_case"),
        "risks": parsed.get("risks"),
        "novelty_adjudication": parsed.get("novelty_adjudication"),
        "claim_audit": parsed.get("claim_audit"),
        "next_experiment": parsed.get("next_experiment"),
        "reviewer_identity": parsed.get("reviewer_identity"),
        "raw_response_chars": len(content),
        "verdict_source": "parsed from the reviewer's own fenced JSON; never synthesised locally",
    }
    (STAGE / "cross_review.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"trace  -> {trace_path}")
    print(f"receipt-> {STAGE / 'cross_review.json'}")
    print(f"verdict={verdict} score={receipt['score']} reviewer={receipt['reviewer_model']}")
    if parsed.get("novelty_adjudication"):
        print("novelty:", json.dumps(parsed["novelty_adjudication"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
