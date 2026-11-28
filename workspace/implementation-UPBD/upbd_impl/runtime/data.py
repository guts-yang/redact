"""Run-local data adapters with explicit split manifests."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from upbd_impl.contracts import Probe
from upbd_impl.core.probes import split_probes
from upbd_impl.runtime.paths import Paths, env


CHANNELS = ("forget", "retain", "authors", "world")


@dataclass
class TrainingData:
    forget: list[dict]
    retain: list[dict]
    authors: list[dict]
    world: list[dict]
    calibration: dict[str, list[dict]] = field(default_factory=dict)
    test: dict[str, list[dict]] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def behavior_target(row: dict) -> str:
    """Match the legacy UPBD target text without importing its GPU module."""
    cot = (row.get("model_cot") or "").strip()
    answer = (row.get("response") or "").strip()
    if not answer:
        answer = (row.get("gold_answer") or "").strip()
    return cot + "\n</think>\n\n" + answer if cot else answer


def default_csv_paths(paths: Paths) -> dict[str, Path]:
    root = paths.asu_root / "runs"
    return {
        "forget": Path(env("FORGET_CSV", str(root / "r2mu" / "forget10" / "eval" / "forget10.csv"))),
        "retain": Path(env("RETAIN_CSV", str(root / "target" / "forget10" / "eval_tf443" / "forget10_retain.csv"))),
        "authors": Path(env("AUTHORS_CSV", str(root / "target" / "forget10" / "eval_tf443" / "real_authors.csv"))),
        "world": Path(env("WORLD_CSV", str(root / "target" / "forget10" / "eval_tf443" / "world_facts.csv"))),
    }


def normalize_question(text: str) -> str:
    return (text or "").replace("<｜User｜>", "").replace("<｜Assistant｜>", "").strip()


def load_entity_map() -> dict[str, str]:
    path = env("ENTITY_MAP_JSON")
    if not path:
        raise FileNotFoundError(
            "entity split requires UPBD_IMPL_ENTITY_MAP_JSON: "
            "a JSON object mapping normalized questions to entity keys")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in payload.items()
    ):
        raise ValueError("entity map must be a JSON object of string -> non-empty string")
    return {normalize_question(key): value for key, value in payload.items()}


def row_entity_key(row: dict, entity_map: dict[str, str] | None = None,
                   *, allow_question: bool = False) -> str:
    for key in ("entity_key", "entity", "subject", "author", "name"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    question = normalize_question(row.get("question") or "")
    if entity_map is not None:
        if question not in entity_map:
            raise KeyError(f"no entity mapping for question: {question[:120]}")
        return entity_map[question]
    if allow_question:
        if not question:
            raise ValueError("question-level split requires a non-empty question")
        return "question:" + question
    raise ValueError(
        "strict splitting requires an entity column or UPBD_IMPL_ENTITY_MAP_JSON")


def _probe_manifest(probes: tuple[Probe, ...]) -> list[dict]:
    return [
        {
            "probe_id": probe.probe_id,
            "entity_key": probe.entity_key,
            "question": normalize_question(probe.question),
        }
        for probe in probes
    ]


def split_rows(rows: list[dict], *, mode: str, seed: int,
               entity_map: dict[str, str] | None = None) -> dict:
    if mode not in {"entity", "question", "holdout"}:
        raise ValueError("split mode must be entity, question, or holdout")
    if mode == "holdout":
        raise ValueError("holdout mode is constructed by load_training_data")
    probes = [
        Probe(
            probe_id=str(index),
            entity_key=row_entity_key(
                row, entity_map, allow_question=(mode == "question")),
            question=row.get("question") or "",
        )
        for index, row in enumerate(rows)
    ]
    partitions = split_probes(probes, seed)
    by_id = {str(index): row for index, row in enumerate(rows)}
    return {
        "train": [by_id[probe.probe_id] for probe in partitions.train],
        "calibration": [by_id[probe.probe_id] for probe in partitions.calibration],
        "test": [by_id[probe.probe_id] for probe in partitions.test],
        "manifest": {
            "train": _probe_manifest(partitions.train),
            "calibration": _probe_manifest(partitions.calibration),
            "test": _probe_manifest(partitions.test),
        },
    }


def load_training_data(paths: Paths, holdout: int = 0, *, split_mode: str = "entity",
                       seed: int = 1001) -> TrainingData:
    if holdout < 0:
        raise ValueError("holdout must be non-negative")
    files = default_csv_paths(paths)
    raw = {name: read_csv_rows(path) for name, path in files.items()}
    entity_map = load_entity_map() if split_mode == "entity" else None
    partitioned: dict[str, dict] = {}
    for name, rows in raw.items():
        if split_mode == "holdout":
            heldout = rows[:holdout]
            partitioned[name] = {
                "train": rows[holdout:],
                "calibration": heldout,
                "test": heldout,
                "manifest": {
                    "train": [{"probe_id": str(i), "question": normalize_question(rows[i].get("question") or "")} for i in range(holdout, len(rows))],
                    "calibration": [{"probe_id": str(i), "question": normalize_question(rows[i].get("question") or "")} for i in range(holdout)],
                    "test": [{"probe_id": str(i), "question": normalize_question(rows[i].get("question") or "")} for i in range(holdout)],
                },
            }
        else:
            partitioned[name] = split_rows(
                rows, mode=split_mode, seed=seed, entity_map=entity_map)
    return TrainingData(
        **{name: partitioned[name]["train"] for name in CHANNELS},
        calibration={name: partitioned[name]["calibration"] for name in CHANNELS},
        test={name: partitioned[name]["test"] for name in CHANNELS},
        manifest={
            "mode": split_mode,
            "scientific_valid": split_mode == "entity",
            "holdout_test_overlap": split_mode == "holdout",
            "channels": {name: partitioned[name]["manifest"] for name in CHANNELS},
        },
    )


def entity_disjoint_split(rows: list[dict], seed: int = 1001):
    probes = [
        Probe(
            probe_id=str(row.get("uid") or index),
            entity_key=row_entity_key(row),
            question=row.get("question") or "",
        )
        for index, row in enumerate(rows)
    ]
    return split_probes(probes, seed)
