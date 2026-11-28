"""CROWN: request-level composition, conflicts, and minimal concession."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodDiagnostics,
    MethodSpec, MethodState, PrepareContext, RequestSpec, StepContext,
)
from upbd_impl.core.decisions import accept, infeasible, rollback
from upbd_impl.core.margins import margin_violation


@dataclass
class CROWNConfig:
    lambda_margin: float = 0.1
    risk_budget: float = 0.05
    min_utility: float = 0.0


@dataclass(frozen=True)
class ConcessionSolution:
    feasible: bool
    concessions: dict[str, float]
    cost: float
    reason: str = ""


def normalize_request(request: RequestSpec, aliases: dict[str, str] | None = None) -> RequestSpec:
    aliases = aliases or {}

    def normalize(values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({aliases.get(value.strip(), value.strip()) for value in values if value.strip()}))

    return RequestSpec(
        request_id=request.request_id,
        forget=normalize(request.forget),
        protect=normalize(request.protect),
        aux=normalize(request.aux),
        risk_budget=request.risk_budget,
        min_utility=request.min_utility,
    )


def compose_requests(requests: list[RequestSpec] | tuple[RequestSpec, ...]) -> RequestSpec:
    if not requests:
        raise ValueError("at least one request is required")
    return RequestSpec(
        request_id="joint",
        forget=tuple(sorted({value for request in requests for value in request.forget})),
        protect=tuple(sorted({value for request in requests for value in request.protect})),
        aux=tuple(sorted({value for request in requests for value in request.aux})),
        risk_budget=min(request.risk_budget for request in requests),
        min_utility=max(request.min_utility for request in requests),
    )


def find_conflicts(requests: list[RequestSpec] | tuple[RequestSpec, ...]) -> list[tuple[str, str, str]]:
    conflicts = set()
    for request in requests:
        for value in set(request.forget) & set(request.protect):
            conflicts.add((request.request_id, request.request_id, value))
        for other in requests:
            if other.request_id == request.request_id:
                continue
            for value in set(request.forget) & set(other.protect):
                conflicts.add((request.request_id, other.request_id, value))
    return sorted(conflicts)


def minimum_concession(required: dict[str, float], weights: dict[str, float]) -> ConcessionSolution:
    if set(required) != set(weights):
        return ConcessionSolution(False, {}, 0.0, "required/weight request keys differ")
    if any((not isfinite(value)) or value == float("inf") for value in required.values()):
        return ConcessionSolution(False, {}, 0.0, "an unbounded concession is required")
    if any(weight < 0 or not isfinite(weight) for weight in weights.values()):
        return ConcessionSolution(False, {}, 0.0, "weights must be finite and non-negative")
    concessions = {key: max(0.0, value) for key, value in required.items()}
    return ConcessionSolution(
        True, concessions, sum(concessions[key] * weights[key] for key in concessions))


class CROWN(BaseMethod):
    spec = MethodSpec("crown", "IDEA_06_CROWN", "Conflict-aware request composition")

    def __init__(self, config: CROWNConfig | None = None):
        self.config = config or CROWNConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        aliases = ctx.extras.get("aliases") or {}
        requests = [normalize_request(request, aliases) for request in ctx.requests]
        joint = compose_requests(requests) if requests else None
        return MethodState(values={
            "requests": requests,
            "joint": joint,
            "conflicts": find_conflicts(requests),
            "allow_concession": bool(ctx.extras.get("allow_concession", False)),
            "concession": None,
        })

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.logits is None or ctx.nll is None:
            raise ValueError("CROWN requires logits and NLL")
        n_forget = len(ctx.forget_rows)
        if n_forget == 0 or ctx.entity_ids is None:
            raise ValueError("CROWN requires forget rows and entity ids")
        logp = F.log_softmax(ctx.logits[:n_forget, -1, :], dim=-1)
        disclose = logp.gather(1, ctx.entity_ids.view(-1, 1)).squeeze(1)
        margins = -disclose
        margin_loss = margin_violation(margins, 1.0).mean()
        loss = ctx.nll.mean() + self.config.lambda_margin * margin_loss
        return LossOutput(loss=loss, diagnostics={
            "nll": ctx.nll.mean().detach(),
            "crown_margin_loss": margin_loss.detach(),
            "crown_conflict_count": torch.tensor(float(len(state.values["conflicts"])), device=loss.device),
            "disclosure_risk": torch.sigmoid(1.0 - margins).mean().detach(),
            "utility": torch.sigmoid(-ctx.nll.mean().detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        conflicts = state.values["conflicts"]
        if conflicts and not state.values["allow_concession"]:
            return infeasible("CROWN forget/protect conflict requires explicit concession")
        joint = state.values["joint"]
        budget = joint.risk_budget if joint else self.config.risk_budget
        if conflicts:
            required = {"joint": max(0.0, evidence.disclosure_risk - budget)}
            solution = minimum_concession(required, {"joint": 1.0})
            if not solution.feasible:
                return infeasible("CROWN concession problem is infeasible")
            state.values["concession"] = solution
        if evidence.disclosure_risk > budget and not conflicts:
            return rollback("CROWN calibration risk exceeded without a declared conflict")
        if evidence.utility < self.config.min_utility:
            return rollback("CROWN utility constraint violated")
        return accept("CROWN request-level constraints satisfied", "request-specification")

    def finalize(self, state: MethodState) -> MethodDiagnostics:
        concession = state.values.get("concession")
        return MethodDiagnostics(values={
            "conflicts": state.values["conflicts"],
            "concession": None if concession is None else {
                "concessions": concession.concessions,
                "cost": concession.cost,
            },
            "scope": "request-specification; parameter commutativity is not claimed",
        })
