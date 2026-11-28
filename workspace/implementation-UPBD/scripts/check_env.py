#!/usr/bin/env python3
"""Read-only environment and path preflight for the isolated implementation."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upbd_impl.registry import list_methods  # noqa: E402
from upbd_impl.runtime.paths import Paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="verify the CPU-only path")
    parser.add_argument("--cuda", action="store_true", help="require explicit CUDA availability")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.cpu and args.cuda:
        parser.error("--cpu and --cuda are mutually exclusive")

    errors = []
    for package in ("torch", "yaml"):
        if importlib.util.find_spec(package) is None:
            errors.append(f"missing required package: {package}")
    for package in ("transformers", "peft"):
        status = "present" if importlib.util.find_spec(package) else "GPU-runtime-only missing"
        print(f"{package}: {status}")

    if args.cuda:
        try:
            from upbd_impl.runtime.device import resolve_device
            print("resolved device:", resolve_device("cuda"))
        except Exception as exc:
            errors.append(str(exc))
    else:
        from upbd_impl.runtime.device import resolve_device
        print("resolved device:", resolve_device("cpu"))

    print("methods:", ",".join(spec.slug for spec in list_methods()))
    paths = Paths.from_env()
    print("results root:", paths.results)
    if args.check_inputs:
        from upbd_impl.runtime.data import default_csv_paths
        for label, path in {
            "target_model": paths.target_model,
            "unlearned_model": paths.unlearned_model,
            **default_csv_paths(paths),
        }.items():
            exists = Path(path).exists()
            print(f"{label}: {'present' if exists else 'missing'} ({path})")
            if not exists:
                errors.append(f"missing input {label}")

    if errors:
        for error in errors:
            print("ERROR:", error)
        return 1
    print("PREFLIGHT PASS (read-only; nothing launched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
