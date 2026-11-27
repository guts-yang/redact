#!/usr/bin/env python3
"""Pre-flight for the GPU cluster: check deps, paths, data, model, CUDA.

Run FIRST after pulling the repo:
    python scripts/check_env.py
It NEVER loads the model (cheap). Reports what is missing with exact UNL1127_*
env vars to set. Exit code 0 = ready for `run_all.py --dry-run`.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ok = True


def chk(label, cond, fix=""):
    global ok
    print(f"  [{'OK' if cond else 'MISS'}] {label}" + ("" if cond else f"  -> {fix}"))
    if not cond:
        ok = False


def main():
    print("implementations-1127 environment pre-flight")
    print(f"  root: {ROOT}")
    import torch
    print(f"  torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print("\n[1] core dependency")
    chk("torch importable", True)
    print("\n[2] runtime dependencies (GPU cluster only)")
    for m in ["transformers", "peft", "accelerate", "sentencepiece"]:
        try:
            importlib.import_module(m)
            chk(m, True)
        except Exception as e:
            chk(m, False, f"pip install -r requirements.txt ({type(e).__name__})")
    print("\n[3] runtime package")
    from runtime import paths
    chk("runtime.paths importable", True)
    print("\n[4] data / model paths (override with UNL1127_* env vars)")
    for label, p in [("TARGET_MODEL", paths.TARGET_MODEL),
                     ("FORGET_FILE", paths.FORGET_FILE),
                     ("RETAIN_FILE", paths.RETAIN_FILE),
                     ("REFUSAL_FILE", paths.REFUSAL_FILE)]:
        chk(f"{label}: {p}", p.exists(),
            f"set UNL1127_{label}=/abs/path  (or UNL1127_ASU_ROOT / UNL1127_DATA)")
    print("\n[5] CUDA")
    if torch.cuda.is_available():
        chk(f"GPU: {torch.cuda.get_device_name(0)}", True)
    else:
        print("  [--] no CUDA here (this is fine on a laptop; training needs a GPU)")
    print("\n[6] results dir")
    writable = True
    try:
        paths.ensure_dirs()
    except (PermissionError, OSError) as e:
        writable = False
        print(f"  [--] cannot create default RESULTS here ({type(e).__name__}); "
              f"set UNL1127_RESULTS=<writable abs path> on the cluster")
    chk(f"RESULTS reachable: {paths.RESULTS}", writable or paths.RESULTS.exists(),
        "set UNL1127_RESULTS=/writable/abs/path")

    print("\n" + ("READY: run `python scripts/run_all.py --all --dry-run` next."
                  if ok else "NOT READY: fix the MISS items above (set env vars / install deps)."))
    print("Reminder: proposal scores are MECHANISM scores; this package creates NO claim.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
