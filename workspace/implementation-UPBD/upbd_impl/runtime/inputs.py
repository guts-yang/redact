"""Explicit JSON inputs for proposal-specific state used by real GPU runs."""
from __future__ import annotations

import json
from pathlib import Path

from upbd_impl.contracts import BoundaryAsset, RequestSpec
from upbd_impl.runtime.paths import Paths, env


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prepare_inputs(paths: Paths, *, hidden_dim: int, device):
    import torch

    stats_path = env("BOUNDARY_STATS_JSON")
    stats = _read_json(stats_path) if stats_path else {}
    basis = stats.get("representation_sensitivity")
    weights = stats.get("parameter_weights")
    boundary = BoundaryAsset(
        checkpoint=str(paths.unlearned_model),
        probe_scope=("direct", "paraphrase", "alias", "multi-hop", "decoding"),
        representation_sensitivity=(
            torch.tensor(basis, dtype=torch.float32, device=device)
            if basis is not None else None),
        parameter_weights=(
            torch.tensor(weights, dtype=torch.float32, device=device)
            if weights is not None else None),
        metadata={"stats_path": stats_path or "synthetic-default"},
    )
    extras = {
        "hidden_dim": hidden_dim,
        "benign_operator_drops": stats.get("benign_operator_drops"),
        "outside_operator_family": bool(stats.get("outside_operator_family", False)),
        "aliases": stats.get("aliases") or {},
        "allow_concession": bool(stats.get("allow_concession", False)),
    }
    reference_path = env("REFERENCE_HIDDEN_JSON")
    if reference_path:
        extras["reference_hidden"] = torch.tensor(
            _read_json(reference_path), dtype=torch.float32, device=device)

    requests_path = env("REQUESTS_JSON")
    raw_requests = _read_json(requests_path) if requests_path else []
    if not isinstance(raw_requests, list):
        raise ValueError("UPBD_IMPL_REQUESTS_JSON must contain a list")
    requests = tuple(
        RequestSpec(
            request_id=str(item["request_id"]),
            forget=tuple(item.get("forget", ())),
            protect=tuple(item.get("protect", ())),
            aux=tuple(item.get("aux", ())),
            risk_budget=float(item.get("risk_budget", 0.05)),
            min_utility=float(item.get("min_utility", 0.0)),
        )
        for item in raw_requests
    )
    return boundary, requests, extras


def validate_decisive_inputs(method: str, boundary: BoundaryAsset,
                             requests: tuple[RequestSpec, ...], extras: dict) -> None:
    if method == "tcr" and boundary.representation_sensitivity is None:
        raise ValueError("decisive TCR requires BOUNDARY_STATS_JSON representation_sensitivity")
    if method == "bless" and not extras.get("benign_operator_drops"):
        raise ValueError("decisive BLESS requires non-empty benign_operator_drops")
    if method == "tmot" and extras.get("reference_hidden") is None:
        raise ValueError("decisive TMOT requires REFERENCE_HIDDEN_JSON")
    if method in {"stream", "crown"} and not requests:
        raise ValueError(f"decisive {method} requires REQUESTS_JSON")
