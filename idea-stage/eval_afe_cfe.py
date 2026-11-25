#!/usr/bin/env python3
"""AFE / CFE (and the computable part of MU) on baseline-protocol generations.

Definition parity is guaranteed by importing the repo's own prompts and parsers:
  * AFE : `scripts.judge_afe_rtofu.PROMPT_TEMPLATE` (4-way LEAK > HALLUC > COLLAPSE > REFUSE)
  * CFE : `scripts.judge_cfe_rtofu.build_prompt` / `parse_score` (R-TOFU graded score)

Only the API transport changes: the judge backend is the user-provided Hy4
endpoint instead of DeepSeek. That substitution is recorded in the output.

AFE_judge = 1 - LEAK_rate; CFE_judge = 1 - mean(graded score); MU needs
retain / real_authors / world_facts with four metrics each, and only the retain
ROUGE-L component exists locally, so MU is reported as PARTIAL and never as a
full number.

Usage: python3 idea-stage/eval_afe_cfe.py --systems target:f10 r2mu10:f10 ...
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/usr/local/workspace/redact")
RTOFU = Path("/usr/local/workspace/R-TOFU/workspace/results")
STAGE = ROOT / "idea-stage"
ENV_FILE = Path("/usr/local/workspace/.env")
ENDPOINT = "https://tokenlink.woa.com/openapi/v1/chat/completions"
MODEL = "hy4"
sys.path.insert(0, str(ROOT))

from scripts.judge_afe_rtofu import PROMPT_TEMPLATE as AFE_PROMPT  # noqa: E402
from scripts.judge_cfe_rtofu import build_prompt as cfe_prompt, parse_score  # noqa: E402

TAGS = {
    "target:f10": "base_target_forget10", "r2mu10:f10": "base_r2mu10_forget10",
    "target:f05": "base_target_forget05", "r2mu05:f05": "base_r2mu05_forget05",
}


def token():
    for k in ("Hy4_TOKEN", "HY4_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(?:export\s+)?Hy4_TOKEN\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("no token")


def ask(prompt: str, tok: str, timeout=120) -> str:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def parse_afe(raw: str):
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    lab = str(d.get("label", "")).strip().upper()
    if lab not in ("LEAK", "HALLUC", "COLLAPSE", "REFUSE"):
        return None, ""
    span = d.get("leak_span", "") if lab == "LEAK" else ""
    return lab, span


def rows_for(tag: str, n_per_strategy: int):
    out = []
    for i, line in enumerate((RTOFU / f"eval_{tag}.jsonl").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("is_retain", 0)) == 1:
            continue
        out.append((i, r))
    picked, seen = [], {}
    for i, r in out:
        s = r.get("strategy")
        if seen.get(s, 0) < n_per_strategy:
            seen[s] = seen.get(s, 0) + 1
            picked.append((i, r))
    return picked


def run_system(name: str, tag: str, tok: str, n_per_strategy=20, workers=4):
    picked = rows_for(tag, n_per_strategy)
    cache = STAGE / f"judge_cache_{name.replace(':', '_')}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["uid"]] = d
    todo = [x for x in picked if f"{name}|{x[0]}|afe" not in done]
    print(f"[{name}] rows={len(picked)} cached={len(done)} todo={len(todo)}", flush=True)

    def one(item):
        i, r = item
        ans = r.get("answer") or r.get("response") or ""
        prompt = AFE_PROMPT.format(question=r.get("question"), gold_answer=r.get("gold_answer") or "",
                                   generated_answer=ans)
        try:
            raw = ask(prompt, tok)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return {"uid": f"{name}|{i}|afe", "status": "error"}
        lab, span = parse_afe(raw)
        return {"uid": f"{name}|{i}|afe", "status": "ok" if lab else "parse_failed",
                "label": lab, "leak_span": span, "strategy": r.get("strategy"), "index": i,
                "entity_leak": int(r.get("entity_leak", 0) or 0), "raw": raw[:400]}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, todo):
            done[res["uid"]] = res
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    recs = [done[k] for k in done if k.startswith(f"{name}|") and k.endswith("|afe")]
    ok = [x for x in recs if x.get("status") == "ok"]
    labels = [x["label"] for x in ok]
    n = len(labels)
    leak = labels.count("LEAK")
    res = {
        "system": name, "tag": tag, "n_judged": n, "n_failed": len(recs) - n,
        "AFE_judge": round(1 - leak / n, 6) if n else None,
        "leak_rate_judge": round(leak / n, 6) if n else None,
        "halluc_rate": round(labels.count("HALLUC") / n, 6) if n else None,
        "collapse_rate": round(labels.count("COLLAPSE") / n, 6) if n else None,
        "refuse_rate": round(labels.count("REFUSE") / n, 6) if n else None,
        "agreement_with_string": (round(sum(1 for x in ok
                                           if int(x["label"] == "LEAK") == int(x["entity_leak"] == 1)) / n, 6)
                                  if n else None),
    }
    print(f"[{name}] AFE={res['AFE_judge']} LEAK={res['leak_rate_judge']} "
          f"HALLUC={res['halluc_rate']} COLLAPSE={res['collapse_rate']} REFUSE={res['refuse_rate']}", flush=True)
    return res


def cfe_for(name: str, tag: str, tok: str, workers=4, max_rows=20):
    picked = [(i, r) for i, r in rows_for(tag, 999) if r.get("strategy") == "DefaultCoT"][:max_rows]
    cache = STAGE / f"judge_cache_cfe_{name.replace(':', '_')}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["uid"]] = d
    todo = [x for x in picked if f"{name}|{x[0]}|cfe" not in done]

    def one(item):
        i, r = item
        cot = r.get("cot") or ""
        try:
            raw = ask(cfe_prompt(r.get("question"), r.get("gold_answer") or "", cot), tok)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return {"uid": f"{name}|{i}|cfe", "status": "error"}
        score = parse_score(raw) if raw else None
        return {"uid": f"{name}|{i}|cfe", "status": "ok" if score is not None else "parse_failed",
                "score": score, "raw": (raw or "")[:300]}

    if todo:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(one, todo):
                done[res["uid"]] = res
                with cache.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
    recs = [done[k] for k in done if k.endswith("|cfe")]
    ok = [x for x in recs if x.get("status") == "ok" and x.get("score") is not None]
    res = {"system": name, "n_judged": len(ok), "n_failed": len(recs) - len(ok),
           "CFE_judge": (round(1 - sum(float(x["score"]) for x in ok) / len(ok), 6) if ok else None),
           "mean_graded_score": (round(sum(float(x["score"]) for x in ok) / len(ok), 6) if ok else None)}
    print(f"[{name}] CFE={res['CFE_judge']} (mean score {res['mean_graded_score']}, n={res['n_judged']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["target:f10", "r2mu10:f10", "target:f05", "r2mu05:f05"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-cfe", action="store_true")
    a = ap.parse_args()
    tok = token()
    out = {"block": "AFE/CFE/MU", "judge_backend": f"hy4 ({ENDPOINT})",
           "definition_source": "scripts/judge_afe_rtofu.PROMPT_TEMPLATE + scripts/judge_cfe_rtofu.build_prompt",
           "deviation": "transport only: DeepSeek -> Hy4; prompts and parsers unchanged",
           "sampling": f"AFE on {a.n} rows per strategy (4 strategies); CFE on up to 20 DefaultCoT rows",
           "systems": {}}
    for name in a.systems:
        tag = TAGS[name]
        out["systems"][name] = {"afe": run_system(name, tag, tok, a.n, a.workers)}
        if not a.skip_cfe:
            out["systems"][name]["cfe"] = cfe_for(name, tag, tok, a.workers)
    (STAGE / "afe_cfe_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print("written ->", STAGE / "afe_cfe_results.json")


if __name__ == "__main__":
    main()
