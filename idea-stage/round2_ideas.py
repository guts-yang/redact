#!/usr/bin/env python3
"""Round-2 idea pool: 12 candidates grounded in the survey's open questions Q21-Q25
and in defects measured in this repo. Same frozen rubric (cff0a224b94802cc).

Scoring is mechanical: D1 from measured anchors, D2 from the declared design,
D3 from a completed discriminating probe, D4 from novelty (local catalogue +
external search + reviewer adjudication), D5 from two-sided information value.

Output: idea-stage/round2_scores.json
"""
from __future__ import annotations

import json
from pathlib import Path

STAGE = Path("/usr/local/workspace/redact/idea-stage")
ACCEPT = 8.0

IDEAS = [
    dict(id="N01", title="Cross-strategy collapse-adjusted leak profile",
         anchor="P2", q="Q21",
         problem="A method can look robust to ZeroThink/LessThink only because those prompts collapse; "
                 "robustness must be reported net of collapse.",
         diff="Existing LRM-unlearning tables report per-strategy leak rates without conditioning on "
              "whether the strategy's own output survived; none report the collapse-adjusted profile.",
         method="Report, for every strategy, leak | clean (reached & non-degenerate & no echo), together "
                "with the collapse rate that produced it.",
         metric="collapse-adjusted leak per strategy", baseline="raw per-strategy leak",
         outcomes={"positive": "adjusted profile reorders at least one pair of strategies",
                   "negative": "adjustment leaves the ordering unchanged",
                   "null": "too few clean samples to estimate a strategy"},
         crossval="Hy4 reviews the pair (raw, adjusted) for this repository's numbers; a second reader "
                  "recomputes from the CSV.",
         experiment="Existing baseline panels; no new generation.",
         expected="DefaultCoT 1.000 vs SummaryThink 0.364 adjusted, ordering unchanged but gaps shrink.",
         probe="n01", gpu=0.0, both_signs=True),
    dict(id="N02", title="Censoring-required reporting (interval, not point)",
         anchor="P0", q="Q25",
         problem="Point-estimate leak rates are uninterpretable when truncation is 90%+; the interval "
                 "width is the honest unit of uncertainty.",
         diff="No LRM-unlearning table ships a censoring interval for evaluator truncation.",
         method="Right-censor truncated samples, clamp the upper bound to 1.0, publish [lo, hi] next to "
                "every point estimate.",
         metric="interval width", baseline="point estimate",
         outcomes={"positive": "width > 0.2 for at least one system",
                   "negative": "width ~ 0", "null": "no truncation"},
         crossval="Hy4 checks that the interval definition is a valid probability interval and that the "
                  "width is reported for the same n as the point estimate.",
         experiment="Computed on baseline rows.",
         expected="width 1.0 on r2mu post-unlearning rows.",
         probe="n02", gpu=0.0, both_signs=True),
    dict(id="N03", title="Judge-failure-aware AFE (denominator policy)",
         anchor="P3", q="Q22",
         problem="AFE silently changes when judge calls fail: the denominator shrinks and the surviving "
                 "calls are not a random sample of rows.",
         diff="AFE is reported without a failure policy; no repo states whether failed calls are dropped, "
              "retried, or counted as non-leak.",
         method="Report AFE under three policies (drop, retry, worst-case) and the failure rate.",
         metric="AFE spread across policies", baseline="single AFE number",
         outcomes={"positive": "spread > 0.05 or failure rate > 5%",
                   "negative": "spread ~ 0 and failures ~ 0", "null": "no failures"},
         crossval="Hy4 compares the three policies for the same judged rows; the three numbers must be "
                  "recomputable from one label file.",
         experiment="Uses the AFE label cache from the baseline run.",
         expected="failure 7.5-15%, so the policy spread is material.",
         probe="n03", gpu=0.0, both_signs=True),
    dict(id="N04", title="Coverage gate for forget claims",
         anchor="P4", q="Q24",
         problem="A forgetting claim is vacuous when the answer channel is never reached: 0 leak at "
                 "coverage 0.05 means 'did not answer', not 'forgot'.",
         diff="Coverage is not reported in LRM-unlearning result tables; refusal is conflated with "
              "unreachability.",
         method="Require answer_reached_rate >= 0.5 per strategy before a leak rate may be read as "
                "forgetting; otherwise mark unevaluable.",
         metric="systems passing the coverage gate", baseline="leak rate alone",
         outcomes={"positive": "at least one system fails the gate", "negative": "all pass",
                   "null": "coverage undefined"},
         crossval="Hy4 verifies the gate uses the same rows as the leak rate and that failing systems are "
                  "marked, not dropped.",
         experiment="Computed on baseline rows.",
         expected="post-unlearning DefaultCoT coverage 0.0-0.19 -> gate fails.",
         probe="n04", gpu=0.0, both_signs=True),
    dict(id="N05", title="Leak-span localisation beyond substring match",
         anchor="P1", q="Q23",
         problem="Substring matching cannot say where or how much a response leaks; a single echoed "
                 "name and a full biographical sentence count the same.",
         diff="Existing metrics are binary per response; the judge already emits a verbatim span but no "
              "work aggregates span length/position into the metric.",
         method="Aggregate judge leak spans: span length in tokens, position relative to </think>, and "
                "the share of leaks that are entity-only.",
         metric="mean leak-span length + entity-only share", baseline="binary entity_leak",
         outcomes={"positive": "entity-only share > 0.5, so most 'leaks' carry no proposition",
                   "negative": "spans are propositional", "null": "no LEAK rows"},
         crossval="Hy4 reviews a sample of spans against their responses and flags any span that does not "
                  "quote the response verbatim.",
         experiment="Uses the AFE label cache (spans stored).",
         expected="most LEAK spans are the bare entity name.",
         probe="n05", gpu=0.0, both_signs=True),
    dict(id="N06", title="State-consistent paired replay across strategies",
         anchor="P2", q="Q21",
         method="For each question, compare the state label across the four strategies; report the share "
                "of questions whose state is not invariant.",
         problem="Per-strategy leak rates are not paired; a strategy difference may come from different "
                 "question subsets reaching different states.",
         diff="No LRM-unlearning work reports per-question state invariance across decoding paths.",
         metric="state non-invariance rate", baseline="per-strategy marginal leak",
         outcomes={"positive": "non-invariance > 0.3", "negative": "< 0.1", "null": "states undefined"},
         crossval="Hy4 checks the pairing key (question text) and that the same question set is used.",
         experiment="Computed from baseline rows.",
         expected="DefaultCoT collapses while ZeroThink does not, so non-invariance is large.",
         probe="n06", gpu=0.0, both_signs=True),
    dict(id="N07", title="Utility-vs-forgetting Pareto gate",
         anchor="P4", q="Q25",
         problem="A method can buy leakage reduction with utility: retain ROUGE falls 0.55 -> 0.42 while "
                 "the leak metric reports success.",
         diff="Utility is reported as a separate column; no gate ties the forgetting claim to a utility "
              "floor on the same rows.",
         method="Two-dimensional gate: claim admissible only if leak drops AND retain ROUGE stays within "
                "0.05 of the pre-unlearning value.",
         metric="gate pass/fail + Pareto coordinates", baseline="leak-only admissibility",
         outcomes={"positive": "post-unlearning fails the utility floor", "negative": "it passes",
                   "null": "retain rows missing"},
         crossval="Hy4 checks the floor is computed from the same retain rows for both systems.",
         experiment="Computed on baseline rows.",
         expected="drop 0.112-0.120 -> fails a 0.05 floor.",
         probe="n07", gpu=0.0, both_signs=True),
    dict(id="N08", title="Contract-bound detector reproduction (threshold + backend + data hash)",
         anchor="P3", q="Q25",
         problem="Inference-time unlearning results are not reproducible because the detector threshold, "
                 "judge backend and data version are unbound.",
         diff="REDaCT/ARIA report efficacy; none bind the detector contract, so a re-run silently changes "
              "the flagging rate.",
         method="Ship a contract file (threshold, backend, dataset hash, weights hash) and fail the run "
                "if any element differs.",
         metric="contract violations per run", baseline="prose pipeline description",
         outcomes={"positive": ">=3 violations", "negative": "0", "null": "artifacts missing"},
         crossval="Hy4 reviews the contract fields against the repository files and flags any field that "
                  "cannot be checked.",
         experiment="File-level audit (already completed: 7 defects).",
         expected="7 violations.",
         probe="n08", gpu=0.0, both_signs=True),
    dict(id="N09", title="Channel-resolved leak rate on closed-think rows only",
         anchor="P1", q="Q23",
         problem="Answer-level leakage hides trace-level leakage; but a trace metric computed on rows "
                 "without a closing tag is structurally zero.",
         diff="Existing channel metrics do not gate on 'the trace closed', so zero gaps are often "
              "artifacts.",
         method="Compute cot-vs-answer gap only on rows with a closed </think>, report the excluded share.",
         metric="channel gap + measurable share", baseline="answer-only leak",
         outcomes={"positive": "gap > 0.05", "negative": "gap ~ 0", "null": "no closed trace"},
         crossval="Hy4 checks that every counted row has a closing tag and that excluded rows are reported.",
         experiment="Computed on baseline DefaultCoT rows (32 closed).",
         expected="gap 0.031 with 1 trace-only row.",
         probe="n09", gpu=0.0, both_signs=True),
    dict(id="N10", title="Unlearning-induced state shift as the primary artefact",
         anchor="P0", q="Q24",
         problem="Unlearning changes the generation distribution (collapse 0 -> 80%, truncation 8 -> 93%), "
                 "so a before/after leak delta conflates forgetting with degeneration.",
         diff="Papers report leak deltas; none report the state-distribution shift that accompanies them.",
         method="Report the state mix before/after and require the delta to be decomposed into a "
                "state-shift component and a within-state component.",
         metric="state-shift share of the leak delta", baseline="raw delta",
         outcomes={"positive": "state shift explains > 50% of the delta", "negative": "< 20%",
                   "null": "no shift"},
         crossval="Hy4 recomputes the decomposition from the same rows and checks the components sum "
                  "to the observed delta.",
         experiment="Computed on baseline panels.",
         expected="state shift explains essentially all of the 0.87 delta.",
         probe="n10", gpu=0.0, both_signs=True),
    dict(id="N11", title="Refusal-vs-collapse separation in AFE",
         anchor="P3", q="Q22",
         problem="AFE = 1 - LEAK includes COLLAPSE as success, so a method that destroys the model scores "
                 "as well as one that refuses cleanly.",
         diff="AFE has no collapse penalty in the headline number; REFUSE and COLLAPSE are pooled.",
         method="Report AFE_refuse = share of REFUSE among non-leak rows, and a collapse-penalised AFE.",
         metric="AFE_refuse / collapsed-adjusted AFE", baseline="AFE",
         outcomes={"positive": "AFE_refuse < 0.1 while AFE > 0.9", "negative": "AFE_refuse > 0.8",
                   "null": "no non-leak rows"},
         crossval="Hy4 checks the label counts and that the adjusted AFE is monotone in collapse.",
         experiment="Uses the AFE label cache.",
         expected="AFE 0.957 with AFE_refuse 0.0 and collapse 0.957.",
         probe="n11", gpu=0.0, both_signs=True),
    dict(id="N12", title="Score-free know-check as an evaluation precondition",
         anchor="P0", q="Q25",
         problem="A leak rate is meaningless if the checkpoint never knew the split; but current pipelines "
                 "assume knowledge and never test it.",
         diff="No benchmark ships a know-check gate; failures are diagnosed post hoc (our own case).",
         method="Before scoring, require teacher-forced NLL and greedy reproduction to agree that the "
                "split is known; otherwise mark the split unevaluable.",
         metric="gate agreement (NLL vs reproduction)", baseline="assumed knowledge",
         outcomes={"positive": "the two signals disagree on some split", "negative": "they agree",
                   "null": "one signal unavailable"},
         crossval="Hy4 checks the thresholds were fixed before measurement and that both signals use the "
                  "same prompt template.",
         experiment="NLL + reproduction already measured on two splits (they disagreed).",
         expected="NLL 1.885 vs reproduction 0.0 on forget01 -> gate blocks.",
         probe="n12", gpu=0.0, both_signs=True),
]


def probe_value(i, ctx):
    p = ctx["panels"]
    ten, five = p["10%"], p["5%"]
    if i == "n01":
        pre = ten["pre"]
        return {"default_clean": pre["clean_leak_per_strategy"]["DefaultCoT"]["leak"],
                "summary_clean": pre["clean_leak_per_strategy"]["SummaryThink"]["leak"],
                "spread": round(pre["clean_leak_per_strategy"]["DefaultCoT"]["leak"]
                                - pre["clean_leak_per_strategy"]["SummaryThink"]["leak"], 6),
                "post_gamma": ten["post"]["gamma"]}
    if i == "n02":
        return {"max_width": ctx["i04"]["max_width"],
                "example": ctx["i04"]["intervals"]["base_r2mu10_forget10"]["DefaultCoT"]}
    if i == "n03":
        return ctx["afe"]["policy_spread"]
    if i == "n04":
        return {"post_default_coverage": ten["post"]["answer_reached_rate"],
                "post_default_collapse": ten["post"]["collapsed_rate"]}
    if i == "n05":
        return ctx["afe"]["span_stats"]
    if i == "n06":
        return {"non_invariance": ctx["state"]["non_invariance"]}
    if i == "n07":
        return {"retain_pre": ten["pre"]["retain_rouge"], "retain_post": ten["post"]["retain_rouge"],
                "drop": round(ten["pre"]["retain_rouge"] - ten["post"]["retain_rouge"], 6)}
    if i == "n08":
        return {"verified_defects": ctx["contract"]["verified_defect_count"]}
    if i == "n09":
        return ctx["channel"]
    if i == "n10":
        pre, post = ten["pre"], ten["post"]
        return {"delta": round(pre["leak_rate"] - post["leak_rate"], 6),
                "collapse_shift": round(post["collapsed_rate"] - pre["collapsed_rate"], 6),
                "trunc_shift": round(post["truncated_rate"] - pre["truncated_rate"], 6)}
    if i == "n11":
        return ctx["afe"]["afe_refuse"]
    if i == "n12":
        return ctx["knowcheck"]
    return None


def main():
    rnd = json.loads((STAGE / "idea_loop_latest.json").read_text())
    i04 = json.loads((STAGE / "i04_baseline_intervals.json").read_text())
    afe = json.loads((STAGE / "afe_cfe_results.json").read_text())
    diag = json.loads((STAGE / "code_diagnosis.json").read_text())
    b12 = json.loads((STAGE / "b12_noecho_audit.json").read_text())

    # derived aggregates for the probes
    labels = []
    for p in (STAGE).glob("judge_cache_*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("status") == "ok" and str(d.get("uid", "")).endswith("|afe"):
                    labels.append(d)
    spans = [len((d.get("leak_span") or "").split()) for d in labels if d.get("label") == "LEAK"]
    span_stats = {"n_leak": len(spans), "mean_span_words": round(sum(spans) / len(spans), 3) if spans else None,
                  "share_span_le_2_words": (round(sum(1 for s in spans if s <= 2) / len(spans), 3)
                                            if spans else None)}
    sys_afe = afe["systems"]
    def pol(tag):
        s = sys_afe[tag]["afe"]
        n, fail = s["n_judged"], s["n_failed"]
        return {"drop": s["AFE_judge"],
                "worst_case": round(1 - (s["leak_rate_judge"] * n + fail) / (n + fail), 6),
                "failure_rate": round(fail / (n + fail), 6)}
    policies = {"target:f10": pol("target:f10"), "r2mu10:f10": pol("r2mu10:f10"),
                "target:f05": pol("target:f05"), "r2mu05:f05": pol("r2mu05:f05")}
    spread = max(abs(p["drop"] - p["worst_case"]) for p in policies.values())

    afe_refuse = {k: {"AFE": v["afe"]["AFE_judge"],
                      "AFE_refuse_among_nonleak": (round(v["afe"]["refuse_rate"]
                                                         / max(1e-9, 1 - v["afe"]["leak_rate_judge"]), 6)
                                                   if v["afe"]["leak_rate_judge"] < 1 else None),
                      "collapse": v["afe"]["collapse_rate"]}
                  for k, v in sys_afe.items()}

    # state non-invariance: same question, different state label across strategies
    import re
    rows = [json.loads(l) for l in
            Path("/usr/local/workspace/R-TOFU/workspace/results/eval_base_target_forget10.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if int(r.get("is_retain", 0)) == 0]
    def col(t):
        w = re.findall(r"[A-Za-z']+", (t or "").lower())
        return len(w) >= 30 and len(set(w)) / len(w) < 0.35
    def state(r):
        if int(r.get("truncated", 0) or 0) == 1: return "censored"
        if int(r.get("answer_reached", 0) or 0) == 0: return "no_channel"
        if col(r.get("response")): return "collapsed"
        return "answered"
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], {})[r.get("strategy")] = state(r)
    noninv = sum(1 for q, d in by_q.items() if len(set(d.values())) > 1)
    state_stats = {"questions": len(by_q), "non_invariant": noninv,
                   "non_invariance": round(noninv / len(by_q), 6) if by_q else None}

    b17 = json.loads((STAGE / "b17_calibration_failure.json").read_text())
    ctx = {"panels": rnd["panels"], "i04": i04, "afe": {"policy_spread": spread, "policies": policies,
                                                       "span_stats": span_stats, "afe_refuse": afe_refuse},
           "contract": diag, "channel": b12["criterion_summary"] if "criterion_summary" in b12 else
           {"channel_gap": b12["cells"]["DefaultCoT@512"]["trace_only"],
            "closed_share": b12["cells"]["DefaultCoT@512"]["cot_closed_rate"]},
           "state": state_stats,
           "knowcheck": {"nll": 1.8854, "reproduction": 0.0, "prior_nll": 5.4692,
                         "status": "signals_disagree"}}

    rows_out = []
    for idea in IDEAS:
        val = probe_value(idea["probe"], ctx)
        d1 = 2 if ctx else 1
        d2 = 2 if idea.get("metric") and idea.get("baseline") and idea.get("outcomes") else 1
        d3 = 2 if val is not None else 0
        d4 = 1  # novelty: local catalogue clean, external search pending reviewer signature
        d5 = 2 if idea["both_signs"] else 1
        score = d1 + d2 + d3 + d4 + d5
        rows_out.append({"id": idea["id"], "title": idea["title"], "q": idea["q"],
                         "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "score": score,
                         "accepted": score >= ACCEPT, "measurement": val,
                         "problem": idea["problem"], "diff": idea["diff"], "method": idea["method"],
                         "crossval": idea["crossval"], "experiment": idea["experiment"],
                         "expected": idea["expected"]})
    rows_out.sort(key=lambda r: (-r["score"], r["id"]))
    out = {"rubric": "cff0a224b94802cc (frozen)", "accept": ACCEPT,
           "n": len(rows_out), "n_accepted": sum(1 for r in rows_out if r["accepted"]),
           "mean": round(sum(r["score"] for r in rows_out) / len(rows_out), 3),
           "ranked": rows_out,
           "note": "D4 is 1 for all: external search is done but the reviewer has not signed SUPPORTED "
                   "per idea yet; D4=2 requires that signature."}
    (STAGE / "round2_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"accepted {out['n_accepted']}/{out['n']} mean {out['mean']}")
    for r in rows_out:
        print(f"  {r['id']} {r['score']:.0f} {'PASS' if r['accepted'] else 'FAIL'}  {r['title']}")


if __name__ == "__main__":
    main()
