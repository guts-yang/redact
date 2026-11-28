"""Side-effect-free contracts shared by all UPBD proposal plugins."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class DecisionKind(str, Enum):
    ACCEPT = "ACCEPT"
    ROLLBACK = "ROLLBACK"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    idea_dir: str
    summary: str
    version: str = "0.1.0"


@dataclass(frozen=True)
class BoundaryAsset:
    checkpoint: str
    probe_scope: tuple[str, ...]
    representation_sensitivity: Any = None
    parameter_weights: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    forget: tuple[str, ...] = ()
    protect: tuple[str, ...] = ()
    aux: tuple[str, ...] = ()
    risk_budget: float = 0.05
    min_utility: float = 0.0


@dataclass(frozen=True)
class Probe:
    probe_id: str
    entity_key: str
    question: str
    transformation: str = "direct"
    decoding: str = "DefaultCoT"


@dataclass(frozen=True)
class ProbeSet:
    train: tuple[Probe, ...] = ()
    calibration: tuple[Probe, ...] = ()
    test: tuple[Probe, ...] = ()


@dataclass(frozen=True)
class EvidenceBatch:
    disclosure_risk: float
    utility: float
    collapse_rate: float
    structure_valid_rate: float = 1.0
    scope: str = "calibration"
    n: int = 0

    def __post_init__(self) -> None:
        for name in ("disclosure_risk", "utility", "collapse_rate", "structure_valid_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.scope != "calibration":
            raise ValueError("training acceptance only supports calibration evidence")
        if self.n < 0:
            raise ValueError("n must be non-negative")


@dataclass(frozen=True)
class Candidate:
    run_id: str
    step: int
    seed: int
    adapter_path: Path | None = None
    config_fingerprint: str = ""


@dataclass
class MethodState:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrepareContext:
    boundary: BoundaryAsset | None = None
    requests: tuple[RequestSpec, ...] = ()
    probes: ProbeSet = field(default_factory=ProbeSet)
    model: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepContext:
    model: Any = None
    tokenizer: Any = None
    batch: Any = None
    logits: Any = None
    hidden: Any = None
    nll: Any = None
    forget_rows: list[dict] = field(default_factory=list)
    retain_rows: list[dict] = field(default_factory=list)
    safe_texts: list[str] = field(default_factory=list)
    entity_ids: Any = None
    step: int = 0
    seed: int = 1001
    device: str = "cpu"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    loss: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def float_diagnostics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, value in self.diagnostics.items():
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            result[key] = float(value)
        return result


@dataclass
class ParamGroup:
    name: str
    params: list[Any]
    lr: float | None = None


@dataclass
class Update:
    gradients: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodDiagnostics:
    values: dict[str, Any] = field(default_factory=dict)


class Method(Protocol):
    spec: MethodSpec

    def prepare(self, ctx: PrepareContext) -> MethodState: ...
    def parameter_groups(self, state: MethodState) -> list[ParamGroup]: ...
    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput: ...
    def constrain_update(self, update: Update, state: MethodState) -> Update: ...
    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision: ...
    def finalize(self, state: MethodState) -> MethodDiagnostics: ...


class BaseMethod:
    """No-op lifecycle defaults; proposal plugins override only what they need."""

    spec: MethodSpec

    def prepare(self, ctx: PrepareContext) -> MethodState:
        return MethodState()

    def parameter_groups(self, state: MethodState) -> list[ParamGroup]:
        return []

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.nll is None:
            raise ValueError("base method requires ctx.nll")
        return LossOutput(loss=ctx.nll.mean(), diagnostics={"nll": ctx.nll.mean()})

    def constrain_update(self, update: Update, state: MethodState) -> Update:
        return update

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        return Decision(DecisionKind.ACCEPT, "default acceptance", ("calibration",))

    def finalize(self, state: MethodState) -> MethodDiagnostics:
        return MethodDiagnostics()
