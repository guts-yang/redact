#!/usr/bin/env python3
"""Sequential reproduction driver for the ten 1127 methods.

On the GPU cluster, run ONE method at a time (each is fully decoupled):
    python scripts/run_all.py --only budget
    python scripts/run_all.py --all
    python scripts/run_all.py --all --dry-run        # CPU-only preflight, no GPU

`--dry-run` runs scripts/selftest.py (CPU smoke) and prints the exact per-method
GPU command WITHOUT launching training. Use it to verify the environment before
committing GPU hours. Each method writes its own adapter + train_summary.json
under UNL1127_RESULTS/<run_id>/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ranked order from IDEA_RANKING-1127.json (strongest first). Each entry is
# independent; changing one never touches another.
ORDER = [
    ("budget",    "IDEA_01_BUDGET-1127",    "7.80"),
    ("pivot",     "IDEA_02_PIVOT-1127",     "7.60"),
    ("consensus", "IDEA_03_CONSENSUS-1127", "7.60"),
    ("drift",     "IDEA_04_DRIFT-1127",     "7.60"),
    ("reverse",   "IDEA_05_REVERSE-1127",   "7.40"),
    ("faith",     "IDEA_06_FAITH-1127",     "7.40"),
    ("graph",     "IDEA_07_GRAPH-1127",     "7.40"),
    ("attrib",    "IDEA_08_ATTRIB-1127",    "7.40"),
    ("layer",     "IDEA_09_LAYER-1127",     "7.20"),
    ("probe",     "IDEA_10_PROBE-1127",     "7.20"),
]


def cmd(slug: str, steps: int = 8, n_forget: int = 4, n_retain: int = 4,
        max_len: int = 256, lr: float = 1e-4, seed: int = 1001) -> str:
    return (f"python -m methods.{slug}.train --run-id unl1127_{slug} "
            f"--steps {steps} --n-forget {n_forget} --n-retain {n_retain} "
            f"--max-len {max_len} --lr {lr} --seed {seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run all ten in ranked order")
    ap.add_argument("--only", default="", help="run a single method slug")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU preflight + print commands, no GPU training")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-forget", type=int, default=4)
    ap.add_argument("--n-retain", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--from", dest="start", default="", help="start from this slug")
    a = ap.parse_args()

    slugs = [s for s, _, _ in ORDER]
    if a.only:
        if a.only not in slugs:
            print(f"unknown method {a.only}; choose from {slugs}"); sys.exit(2)
        targets = [a.only]
    elif a.all:
        targets = slugs
        if a.start and a.start in slugs:
            targets = slugs[slugs.index(a.start):]
    else:
        ap.print_help(); sys.exit(0)

    if a.dry_run:
        print("=== CPU preflight (scripts/selftest.py) ===")
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "selftest.py")])
        if rc != 0:
            print("preflight FAILED; fix before using GPU"); sys.exit(rc)
        print("\n=== GPU commands (ranked order) ===")
        for slug, idea, score in ORDER:
            if slug in targets:
                print(f"  # {idea}  (mechanism score {score}, NOT a claim)")
                print(f"  {cmd(slug, a.steps, a.n_forget, a.n_retain, a.max_len, a.lr, a.seed)}")
        print("\n(dry-run: nothing launched)")
        return

    for slug, idea, _ in ORDER:
        if slug not in targets:
            continue
        print(f"\n========== {slug}  ({idea}) ==========", flush=True)
        c = cmd(slug, a.steps, a.n_forget, a.n_retain, a.max_len, a.lr, a.seed)
        print(f"$ {c}", flush=True)
        rc = subprocess.call(c, shell=True, cwd=str(ROOT))
        if rc != 0:
            print(f"[{slug}] exited {rc}; continuing with next method", flush=True)
    print("\nAll requested methods attempted. Each wrote its own train_summary.json.", flush=True)


if __name__ == "__main__":
    main()
