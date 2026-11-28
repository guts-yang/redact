"""One-way adapter for the legacy UPBD reference implementation.

Nothing here is imported by method cores. The legacy script is never edited or
used as a base class; this module exists for controls and semantic parity only.
"""
from __future__ import annotations

from pathlib import Path

from upbd_impl.runtime.data import behavior_target


def compare_behavior_targets(rows: list[dict]) -> list[dict]:
    """Compare against the legacy pure helper without changing ``sys.path``."""
    import importlib.util

    from upbd_impl.runtime.paths import Paths

    path = Paths.from_env().redact_root / "scripts" / "upbd" / "train_upbd.py"
    spec = importlib.util.spec_from_file_location("_upbd_legacy_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy UPBD module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    legacy_target = module.behavior_target

    return [
        {
            "index": index,
            "matches": behavior_target(row) == legacy_target(row),
        }
        for index, row in enumerate(rows)
    ]


def legacy_train_command(*, python: str, output_dir: Path, merged_dir: Path,
                         extra_args: list[str] | None = None) -> list[str]:
    """Build, but never execute, an isolated legacy reference command."""
    return [
        python, "-m", "scripts.upbd.train_upbd",
        "--output-dir", str(output_dir),
        "--save-merged", str(merged_dir),
        *(extra_args or []),
    ]
