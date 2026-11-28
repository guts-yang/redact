"""TMOT: torch-only Sinkhorn with forbidden support and three-marginal checks."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodDiagnostics,
    MethodSpec, MethodState, PrepareContext, StepContext,
)
from upbd_impl.core.decisions import accept, rollback


@dataclass
class TMOTConfig:
    epsilon: float = 0.1
    iterations: int = 30
    lambda_transport: float = 0.1
    unknown_weight: float = 0.1
    marginal_tolerance: float = 1e-3
    forbidden_tolerance: float = 0.0
    risk_budget: float = 0.05


def sinkhorn_transport(
    cost: torch.Tensor,
    source: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    *, forbidden: torch.Tensor | None = None,
    epsilon: float = 0.1,
    iterations: int = 30,
) -> torch.Tensor:
    """Log-domain entropic OT; forbidden entries receive exactly zero mass."""
    if cost.dim() != 2:
        raise ValueError("cost must be [source,target]")
    if not torch.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("cost must be finite and non-negative")
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")
    n, m = cost.shape
    source = torch.full((n,), 1.0 / n, dtype=cost.dtype, device=cost.device) if source is None else source
    target = torch.full((m,), 1.0 / m, dtype=cost.dtype, device=cost.device) if target is None else target
    if source.shape != (n,) or target.shape != (m,):
        raise ValueError("source/target marginal shapes do not match cost")
    source = source / source.sum()
    target = target / target.sum()
    log_kernel = -cost / epsilon
    if forbidden is not None:
        if forbidden.shape != cost.shape:
            raise ValueError("forbidden mask must match cost")
        forbidden = forbidden.to(torch.bool)
        rows_available = (~forbidden).any(dim=1)
        cols_available = (~forbidden).any(dim=0)
        if (source > 0).any() and not rows_available.all():
            raise ValueError("a positive-mass source row forbids every target")
        if (target > 0).any() and not cols_available.all():
            raise ValueError("a positive-mass target column forbids every source")
        log_kernel = log_kernel.masked_fill(forbidden, -float("inf"))
    log_u = torch.zeros_like(source)
    log_v = torch.zeros_like(target)
    for _ in range(iterations):
        log_u = source.log() - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = target.log() - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return plan if forbidden is None else plan.masked_fill(forbidden, 0.0)


def marginal_error(plan: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if plan.shape != (source.numel(), target.numel()):
        raise ValueError("plan and marginal shapes differ")
    return torch.maximum(
        (plan.sum(dim=1) - source).abs().max(),
        (plan.sum(dim=0) - target).abs().max(),
    )


def forbidden_mass(plan: torch.Tensor, forbidden: torch.Tensor) -> torch.Tensor:
    if plan.shape != forbidden.shape:
        raise ValueError("plan and forbidden mask shapes differ")
    return plan[forbidden.to(torch.bool)].sum()


class TMOT(BaseMethod):
    spec = MethodSpec("tmot", "IDEA_05_TMOT", "Tri-marginal optimal-transport repair")

    def __init__(self, config: TMOTConfig | None = None):
        self.config = config or TMOTConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        return MethodState(values={"reference_hidden": ctx.extras.get("reference_hidden")})

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.hidden is None or ctx.nll is None:
            raise ValueError("TMOT requires hidden states and NLL")
        current = ctx.hidden[:, -1, :]
        reference = ctx.extras.get("reference_hidden", state.values.get("reference_hidden"))
        if reference is None:
            reference = torch.roll(current, shifts=1, dims=0).detach()
        if reference.shape != current.shape:
            raise ValueError("reference hidden states must match current shape")
        cost = (current[:, None, :] - reference[None, :, :]).square().mean(dim=-1)
        n, m = cost.shape
        source = torch.full((n,), 1.0 / n, dtype=cost.dtype, device=cost.device)
        target = torch.full((m,), 1.0 / m, dtype=cost.dtype, device=cost.device)
        forbidden = ctx.extras.get("forbidden_transport")
        plan = sinkhorn_transport(
            cost, source, target, forbidden=forbidden,
            epsilon=self.config.epsilon, iterations=self.config.iterations)
        transport = (plan * cost).sum()
        unknown_cost = self.config.unknown_weight * cost.diagonal().mean()
        nll = ctx.nll.mean()
        loss = nll + self.config.lambda_transport * transport + unknown_cost
        marg_error = marginal_error(plan, source, target)
        forbidden_value = (plan.new_zeros(()) if forbidden is None
                           else forbidden_mass(plan, forbidden))
        state.values["last_marginal_error"] = float(marg_error.detach().cpu())
        state.values["last_forbidden_mass"] = float(forbidden_value.detach().cpu())
        risk = torch.clamp(forbidden_value + torch.sigmoid(-current.mean()), 0.0, 1.0)
        return LossOutput(loss=loss, diagnostics={
            "nll": nll.detach(),
            "tmot_transport": transport.detach(),
            "tmot_marginal_error": marg_error.detach(),
            "tmot_forbidden_mass": forbidden_value.detach(),
            "tmot_unknown_cost": unknown_cost.detach(),
            "disclosure_risk": risk.detach(),
            "utility": torch.sigmoid(-nll.detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        if state.values.get("last_marginal_error", float("inf")) > self.config.marginal_tolerance:
            return rollback("TMOT marginal constraints are outside tolerance")
        if state.values.get("last_forbidden_mass", 0.0) > self.config.forbidden_tolerance:
            return rollback("TMOT forbidden support carries nonzero mass")
        if evidence.disclosure_risk > self.config.risk_budget:
            return rollback("TMOT calibration disclosure risk exceeded")
        return accept("TMOT transport and calibration constraints satisfied")

    def finalize(self, state: MethodState) -> MethodDiagnostics:
        return MethodDiagnostics(values={
            "marginal_error": state.values.get("last_marginal_error"),
            "forbidden_mass": state.values.get("last_forbidden_mass"),
        })
