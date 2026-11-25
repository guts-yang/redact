#!/usr/bin/env python3
"""Build no-echo probes for B12: forget questions that never name the target entity.

The reviewer's cheapest discriminating test requires a cell where a leak cannot be
produced by echoing the question. We therefore keep only questions whose text does
NOT contain the gold key phrase (the same phrase the leak metric matches on).

Output: idea-stage/probes/noecho_forget10.jsonl with {question, answer, task_id}.
Also writes a manifest so the probe set is auditable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RTOFU_CODE = Path("/usr/local/workspace/R-TOFU/workspace/code")
DATA = Path("/jizhicfs/chenyliao/data/rtofu")
OUT = Path("/usr/local/workspace/redact/idea-stage/evidence/probes")

sys.path.insert(0, str(RTOFU_CODE))
from data_utils import key_phrase  # noqa: E402


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "forget10"
    src = DATA / f"{split}.json"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, dropped = [], []
    for r in rows:
        q, a = r.get("question", ""), r.get("answer", "")
        kp = key_phrase(a)
        if not kp:
            dropped.append({"question": q, "why": "no key phrase extractable"})
            continue
        if kp.lower() in q.lower():
            dropped.append({"question": q, "key_phrase": kp, "why": "entity present in question (echo-prone)"})
            continue
        kept.append({"question": q, "answer": a, "task_id": "1", "key_phrase": kp})
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"noecho_{split}.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("question", "answer", "task_id")}, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(src),
        "probe_file": str(dst),
        "n_source": len(rows),
        "n_kept": len(kept),
        "n_dropped_echo": sum(1 for d in dropped if d.get("why", "").startswith("entity")),
        "n_dropped_no_phrase": sum(1 for d in dropped if d["why"].startswith("no key")),
        "rule": "keep a row only if key_phrase(gold_answer) does not appear in the question",
        "sample": kept[:3],
        "note": "This is the only probe set in this run where a substring leak cannot be an echo "
                "of the question wording.",
    }
    (OUT.parent / f"b12_probe_manifest_{split}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_source", "n_kept", "n_dropped_echo", "n_dropped_no_phrase")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
