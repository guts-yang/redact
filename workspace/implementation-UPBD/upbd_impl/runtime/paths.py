"""Independent UPBD implementation paths; no writes occur at import time."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PREFIX = "UPBD_IMPL_"
PKG_ROOT = Path(__file__).resolve().parents[2]
REDACT_ROOT = Path(__file__).resolve().parents[4]


def env(name: str, default: str = "") -> str:
    return os.environ.get(PREFIX + name, default)


@dataclass(frozen=True)
class Paths:
    asu_root: Path
    target_model: Path
    unlearned_model: Path
    results: Path
    redact_root: Path = REDACT_ROOT

    @classmethod
    def from_env(cls) -> "Paths":
        asu = Path(env("ASU_ROOT", "/jizhicfs/chenyliao"))
        results = Path(env(
            "RESULTS", str(asu / "lrm-unlearning" / "results" / "upbd_impl")))
        return cls(
            asu_root=asu,
            target_model=Path(env("TARGET_MODEL", str(asu / "models" / "sangyon" / "LRM-target"))),
            unlearned_model=Path(env(
                "UNLEARNED_MODEL", str(asu / "checkpoints" / "r2mu-forget10" / "checkpoint-last"))),
            results=results,
        )


def run_dir(paths: Paths, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single non-empty path component")
    root = paths.results.resolve()
    candidate = (root / run_id).resolve()
    if candidate.parent != root:
        raise ValueError("run directory escapes UPBD_IMPL_RESULTS")
    return candidate
