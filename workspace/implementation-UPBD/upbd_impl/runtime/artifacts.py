"""Run-scoped, non-overwriting artifact writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from upbd_impl.runtime.paths import Paths, run_dir


class RunArtifacts:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def create(cls, paths: Paths, run_id: str, *, allow_existing_empty: bool = True):
        root = run_dir(paths, run_id)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {root}")
        if not root.exists() and not allow_existing_empty:
            raise FileNotFoundError(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @classmethod
    def existing(cls, root: str | Path):
        path = Path(root).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return cls(path)

    def path(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise ValueError("artifact path must be a relative, non-empty path")
        candidate = (self.root / relative).resolve()
        if self.root.resolve() not in candidate.parents:
            raise ValueError("artifact path escapes run root")
        return candidate

    def write_text(self, relative: str, text: str) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            handle.write(text)
            temp = Path(handle.name)
        os.replace(temp, target)
        return target

    def write_json(self, relative: str, payload: dict[str, Any]) -> Path:
        return self.write_text(
            relative, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def write_summary(self, payload: dict[str, Any]) -> Path:
        enriched = dict(payload)
        enriched["diagnostic_only"] = True
        enriched["claim_level"] = "non-claim"
        return self.write_json("train_summary.json", enriched)
