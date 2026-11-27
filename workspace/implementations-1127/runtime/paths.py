"""All filesystem paths for the 1127 reproduction run.

Everything is overridable via environment variables (prefix ``UNL1127_``) so the
same code runs on a laptop, a GPU cluster, or the R-TOFU server without edits.
No local (macOS) absolute path is hardcoded as a requirement.
"""
from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get("UNL1127_" + name, default)


# Where this package lives (implementations-1127/). Used to locate method.yaml,
# core.py, and to write results when no absolute results dir is given.
PKG_ROOT = Path(__file__).resolve().parents[1]

# Cluster defaults mirror the R-TOFU server layout. Override with UNL1127_* envs.
WORKSPACE = Path(env("WORKSPACE", "/usr/local/workspace/tracks/lrm-unlearning/R-TOFU/workspace"))
REDACT_ROOT = Path(env("REDACT_ROOT", "/usr/local/workspace/tracks/lrm-unlearning/redact"))
RTOFU_REPO = Path(env("RTOFU_REPO", "/usr/local/workspace/R-TOFU"))
ASU_ROOT = Path(env("ASU_ROOT", "/jizhicfs/chenyliao"))

DATA = Path(env("DATA", str(ASU_ROOT / "data" / "rtofu")))
MODELS = Path(env("MODELS", str(ASU_ROOT / "models")))
RESULTS = Path(env("RESULTS", str(WORKSPACE / "results" / "unl1127")))
LOGS = Path(env("LOGS", str(WORKSPACE / "logs" / "unl1127")))

TARGET_MODEL = Path(env("TARGET_MODEL", str(MODELS / "sangyon" / "LRM-target")))
FORGET_FILE = Path(env("FORGET_FILE", str(DATA / "forget10.json")))
RETAIN_FILE = Path(env("RETAIN_FILE", str(DATA / "retain90.json")))
REFUSAL_FILE = Path(env("REFUSAL_FILE", str(RTOFU_REPO / "data" / "idontknow.jsonl")))
CLAIMS_FILE = Path(env("CLAIMS_FILE", str(REDACT_ROOT / "knowledge_banks" / "forget10" / "claims.json")))

# Frozen base + optional pre-unlearned checkpoint (for DRIFT / REVERSE / LAYER / PROBE
# which need a theta_0 reference or a second checkpoint).
BASE_ADAPTER = env("BASE_ADAPTER", "")            # optional PEFT adapter on target
UNLEARNED_ADAPTER = env("UNLEARNED_ADAPTER", "")  # a checkpoint that already forgot (DRIFT source)

SEED = int(env("SEED", "1001"))

# Decoding strategies (mirrors R-TOFU strategies.py) and continuous budget grid.
STRATEGIES = ["DefaultCoT", "LessThink", "ZeroThink", "SummaryThink"]
BUDGET_GRID = [int(x) for x in env("BUDGET_GRID", "0,64,128,256,512").split(",") if x.strip()]
LANGUAGES = env("LANGUAGES", "EN,ZH,DE,RU,TR").split(",")


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against REDACT_ROOT or WORKSPACE."""
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (PKG_ROOT, REDACT_ROOT, WORKSPACE, Path.cwd()):
        cand = base / p
        if cand.exists():
            return cand
    return p
