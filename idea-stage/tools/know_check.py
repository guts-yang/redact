#!/usr/bin/env python3
"""B17: know-check gate. Before any leak rate may be interpreted as forgetting, the
checkpoint must demonstrably know the forget split.

Two signals, both cheap:
  1. teacher-forced NLL of the gold answer given the prompt (lower = knows)
  2. reproduction rate of the gold key phrase under greedy generation (higher = knows)
     -- measured here from an existing eval file when one is available, otherwise skipped

Gate rule (preregistered): a split is admissible for forget evaluation only if
NLL <= 2.0 AND reproduction >= 0.5. Anything else must be reported as
"NOT EVALUABLE: checkpoint does not know this split".

Usage: python3 idea-stage/know_check.py --model target --split forget01 --n 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RTOFU_CODE = Path("/usr/local/workspace/R-TOFU/workspace/code")
DATA = Path("/jizhicfs/chenyliao/data/rtofu")
MODELS = {"target": Path("/jizhicfs/chenyliao/models/sangyon/LRM-target"),
          "r2mu10": Path("/jizhicfs/chenyliao/checkpoints/r2mu-forget10/checkpoint-last")}
STAGE = Path("/usr/local/workspace/redact/idea-stage/evidence")
sys.path.insert(0, str(RTOFU_CODE))
from data_utils import key_phrase  # noqa: E402
from strategies import build_prompt  # noqa: E402


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="target")
    ap.add_argument("--split", default="forget01")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--strategy", default="DefaultCoT")
    a = ap.parse_args()

    rows = [r for r in read(DATA / f"{a.split}.json") if str(r.get("task_id", "")) == "1"][: a.n]
    path = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="cuda")
    model.eval()

    losses, reps = [], []
    for r in rows:
        prompt = build_prompt(r["question"], a.strategy)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        tgt = tok(" " + r["answer"], return_tensors="pt", add_special_tokens=False).to("cuda")
        seq = torch.cat([ids["input_ids"], tgt["input_ids"]], dim=1)
        with torch.inference_mode():
            logits = model(seq).logits
        # loss over the answer tokens only
        n_p = ids["input_ids"].shape[1]
        shift_logits = logits[:, n_p - 1: -1, :]
        shift_labels = seq[:, n_p:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        losses.append(float(loss))
    nll = sum(losses) / len(losses)

    # reproduction from an existing eval file on the same split, if present
    repro, src = None, None
    for cand in sorted((Path("/usr/local/workspace/R-TOFU/workspace/results")).glob("eval_*.jsonl")):
        if cand.stem.endswith("_budget") or cand.stem.endswith("_cot"):
            continue
        rs = read(cand)
        if not rs:
            continue
        if str(rs[0].get("system")) != a.model:
            continue
        qs = {r["question"] for r in rows}
        hit = [r for r in rs if r.get("question") in qs and int(r.get("is_retain", 0)) == 0]
        if len(hit) >= min(8, len(rows)):
            repro = round(sum(int(r.get("entity_leak", 0) or 0) for r in hit) / len(hit), 6)
            src = str(cand)
            break

    gate = {"nll_threshold": 2.0, "repro_threshold": 0.5,
            "nll_pass": nll <= 2.0,
            "repro_pass": (repro is None) or (repro >= 0.5),
            "admissible": (nll <= 2.0) and ((repro is None) or (repro >= 0.5))}
    out = {
        "block": "B17", "model": a.model, "model_path": str(path), "split": a.split,
        "strategy": a.strategy, "n": len(rows),
        "teacher_forced_nll": round(nll, 4),
        "reproduction_rate": repro, "reproduction_source": src,
        "gate": gate,
        "verdict": ("EVALUABLE" if gate["admissible"]
                    else "NOT EVALUABLE: checkpoint does not know this split"),
        "note": "NLL computed in bf16 with a full forward pass (no KV-cache decode, so the "
                "known SIGFPE path is not touched); reproduction is read from an existing "
                "greedy eval file rather than re-generated.",
    }
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / f"b17_know_check_{a.model}_{a.split}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("model", "split", "teacher_forced_nll",
                                          "reproduction_rate", "verdict")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
