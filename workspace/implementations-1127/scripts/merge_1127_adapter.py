"""Merge a 1127 LoRA adapter into the frozen LRM-target base on CPU.

The runner saves a PEFT adapter under ``$UNL1127_RESULTS/<run_id>/adapter``.
``scripts/generate_rtofu_responses.py`` loads a full HF checkpoint from a YAML,
so adapters must be merged before the Hy4-judge / MU pipeline can use them.

Merging on CPU (fp32, 8B ~= 32 GB RAM) deliberately avoids touching GPU1,
which may still be occupied by an unrelated training job.

Usage:
    python scripts/merge_1127_adapter.py \
        --adapter $UNL1127_RESULTS/unl1127_budget_decisive_s1001/adapter \
        --output  $UNL1127_RESULTS/unl1127_budget_decisive_s1001/merged
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "/jizhicfs/chenyliao/models/sangyon/LRM-target"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without loading weights")
    args = ap.parse_args()

    if not args.adapter.is_dir():
        raise SystemExit(f"adapter dir missing: {args.adapter}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a PEFT adapter (no adapter_config.json): {args.adapter}")
    if (args.output / "config.json").is_file():
        print(f"[merge] already merged: {args.output}", flush=True)
        return
    if args.dry_run:
        print(f"[merge] would merge {args.adapter} onto {args.base} -> {args.output}")
        return

    t0 = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(str(args.output), safe_serialization=True)
    tok.save_pretrained(str(args.output))

    summary = {
        "run_id": args.output.parent.name,
        "adapter": str(args.adapter),
        "base": args.base,
        "merged": str(args.output),
        "dtype": "float32",
        "device": "cpu",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.output / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[merge] done {args.output} in {summary['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
