#!/usr/bin/env python3
"""Strict re-scoring of the round-2 idea pool.

Adds two things the first pass lacked:
  * D3 is granted 2 only when the measurement crosses a THRESHOLD DECLARED IN THE
    IDEA'S OWN `outcomes` block; a single descriptive number earns 1.
  * D4 uses the local 34-paper scan AND an external overlap map built from live
    searches (recorded below with URLs), and can only reach 2 if the reviewer
    signs SUPPORTED for that specific idea.

Output: idea-stage/round2_scores_strict.json
"""
from __future__ import annotations

import json
from pathlib import Path

STAGE = Path("/usr/local/workspace/redact/idea-stage")
ACCEPT = 8.0

# external overlaps found by live search on 2026-11-25 (recorded with source)
EXTERNAL = {
    "N01": {"overlap": True, "why": "STaR (arXiv:2601.9281) already defines a Multi-Decoding Consistency "
                                     "Score over DefaultThink/ZeroThink/LessThink for unlearning",
            "url": "https://alphaxiv.org/abs/2601.9281"},
    "N06": {"overlap": False, "why": "MCS scores leak consistency; N06 pairs the per-question STATE label, "
                                     "which is a different object, but the adjacency is real",
            "url": "https://alphaxiv.org/abs/2601.9281"},
    "N03": {"overlap": False, "why": "UWC proposes robust evaluation metrics but not a denominator policy "
                                     "for failed judge calls", "url": "https://www.themoonlight.io/tw/review/"
                                                                     "towards-effective-evaluations-and-comparisons-for-llm-unlearning-methods"},
    "N04": {"overlap": False, "why": "'coverage-aware unlearning' exists for LANGUAGE coverage, not for "
                                     "answer-channel reachability", "url": "https://airebao.com/item/38670"},
    "N02": {"overlap": False, "why": "truncation handling appears in judge harnesses (26-point swing), but "
                                     "no censoring interval is published for leak rates",
            "url": "https://hivebook.wiki/wiki/llm-as-a-judge-bias-is-a-harness-artifact-not-model-skill-truncation-length-position-self-preference"},
    "N09": {"overlap": True, "why": "11 local near-neighbours incl. REDaCT and K-Bench already split CoT vs "
                                    "answer channels", "url": "local catalogue"},
    "N11": {"overlap": True, "why": "REDaCT is literally 'Collapse-Free' and the baseline AFE table already "
                                    "reports a collapse column", "url": "local catalogue"},
    "N05": {"overlap": False, "why": "span aggregation is not published; nearest local hit is LEMUR "
                                     "entropy-aware unlearning", "url": "local catalogue"},
    "N07": {"overlap": False, "why": "utility-aware unlearning is common (MUBENCH over-safety), but a "
                                     "hard gate on the same rows is not", "url": "https://news.sciencenet.cn/htmlpaper/2026/10/20261026113415880156978.shtm"},
}

# strict thresholds taken verbatim from each idea's `outcomes.positive`
THRESH = {
    "N01": ("spread", ">=", 0.10, "adjusted profile must reorder at least one pair; we measured spread"),
    "N02": ("max_width", ">", 0.2, "width > 0.2"),
    "N03": ("policy_spread", ">", 0.05, "AFE spread across policies > 0.05"),
    "N04": ("post_coverage", "<", 0.5, "coverage gate fails when answer_reached < 0.5"),
    "N05": ("entity_only_share", ">", 0.5, "entity-only share > 0.5"),
    "N06": ("non_invariance", ">", 0.3, "state non-invariance > 0.3"),
    "N07": ("retain_drop", ">", 0.05, "retain drop > 0.05 utility floor"),
    "N08": ("defects", ">=", 3, ">= 3 contract violations"),
    "N09": ("channel_gap", ">", 0.05, "channel gap > 0.05"),
    "N10": ("state_shift_share", ">", 0.5, "state shift explains > 50% of the delta"),
    "N11": ("afe_refuse", "<", 0.1, "AFE_refuse < 0.1 while AFE > 0.9"),
    "N12": ("signals_disagree", "==", True, "NLL and reproduction disagree"),
}


def pick(idea, meas):
    def g(obj, key, default=None):
        if isinstance(obj, dict):
            v = obj.get(key, default)
            return g(v, key, default) if isinstance(v, dict) else v
        return default
    if not isinstance(meas, dict):
        return meas if idea == "N03" else None
    if idea == "N01":
        return meas.get("spread")
    if idea == "N02":
        return meas.get("max_width")
    if idea == "N03":
        return meas.get("policy_spread")
    if idea == "N04":
        return meas.get("post_default_coverage")
    if idea == "N05":
        return meas.get("share_span_le_2_words")
    if idea == "N06":
        v = meas.get("non_invariance")
        return v.get("non_invariance") if isinstance(v, dict) else v
    if idea == "N07":
        return meas.get("drop")
    if idea == "N08":
        return meas.get("verified_defects")
    if idea == "N09":
        return meas.get("channel_gap")
    if idea == "N10":
        v = meas.get("state_shift_share")
        return v if v is not None else meas.get("collapse_shift")
    if idea == "N11":
        v = meas.get("afe_refuse") or {}
        for k in ("r2mu10:f10", "r2mu05:f05"):
            if isinstance(v, dict) and isinstance(v.get(k), dict):
                return v[k].get("AFE_refuse_among_nonleak")
        return None
    if idea == "N12":
        return True if meas.get("status") == "signals_disagree" else meas.get("signals_disagree")
    return None


def cmp(x, op, thr):
    if x is None:
        return None
    return {"==": x == thr, ">": x > thr, ">=": x >= thr, "<": x < thr,
            "<=": x <= thr}.get(op, False)


def main():
    base = json.loads((STAGE / "round2_scores.json").read_text())
    recv = json.loads((STAGE / "cross_review_round2.json").read_text()) if \
        (STAGE / "cross_review_round2.json").exists() else {}
    adj = (recv.get("ideas") or {})
    rows = []
    for r in base["ranked"]:
        iid = r["id"]
        meas = r["measurement"] or {}
        key, op, thr, desc = THRESH[iid]
        val = pick(iid, meas)
        crossed = cmp(val, op, thr)
        d3 = 2 if crossed is True else (1 if val is not None else 0)
        ext = EXTERNAL.get(iid, {"overlap": False})
        if ext.get("overlap"):
            d4, d4why = 0, "external overlap: " + ext["why"]
        else:
            signed = str((adj.get(iid) or {}).get("verdict", "")).upper()
            if signed == "SUPPORTED":
                d4, d4why = 2, "reviewer signed SUPPORTED"
            else:
                d4, d4why = 1, ("reviewer " + signed if signed else "no per-idea reviewer signature")
        d1, d2, d5 = r["D1"], r["D2"], r["D5"]
        score = d1 + d2 + d3 + d4 + d5
        rows.append({"id": iid, "title": r["title"], "q": r["q"], "D1": d1, "D2": d2, "D3": d3,
                     "D4": d4, "D5": d5, "score": score, "accepted": score >= ACCEPT,
                     "threshold": desc, "measured": val, "threshold_crossed": crossed,
                     "novelty": d4why, "measurement": meas})
    rows.sort(key=lambda r: (-r["score"], r["id"]))
    out = {"rubric": "cff0a224b94802cc (frozen)", "accept": ACCEPT, "strict_d3": True,
           "n": len(rows), "n_accepted": sum(1 for r in rows if r["accepted"]),
           "mean": round(sum(r["score"] for r in rows) / len(rows), 3),
           "ranked": rows,
           "rules": ["D3=2 only if the idea's own preregistered threshold is crossed",
                     "D4=0 on any external overlap, =2 only with a per-idea SUPPORTED signature"]}
    (STAGE / "round2_scores_strict.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"accepted {out['n_accepted']}/{out['n']} mean {out['mean']}")
    for r in rows:
        print(f"  {r['id']} {r['score']:.0f} {'PASS' if r['accepted'] else 'FAIL'} "
              f"val={r['measured']} thr={r['threshold']} crossed={r['threshold_crossed']} | {r['novelty'][:48]}")


if __name__ == "__main__":
    main()
