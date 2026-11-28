"""TCR: tri-layer contract projection and calibration-gated acceptance."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from upbd_impl.contracts import (
    BaseMethod, Candidate, Decision, EvidenceBatch, LossOutput, MethodSpec,
    MethodState, PrepareContext, StepContext, Update,
)
from upbd_impl.core.decisions import accept, rollback
from upbd_impl.core.margins import margin_violation


@dataclass
class TCRConfig:
    margin_threshold: float = 1.0
    lambda_margin: float = 0.1
    lambda_representation: float = 0.01
    parameter_budget: float = 1.0
    risk_budget: float = 0.05
    min_utility: float = 0.0
    max_collapse: float = 0.05
    min_structure_valid: float = 0.95


def orthonormal_basis(basis: torch.Tensor) -> torch.Tensor:
    if basis.dim() != 2:
        raise ValueError("sensitive basis must be [hidden, rank]")
    return torch.linalg.qr(basis, mode="reduced").Q


def sensitive_component(update: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project the last update dimension onto the orthonormal sensitive basis."""
    q = orthonormal_basis(basis)
    if update.shape[-1] != q.shape[0]:
        raise ValueError("update and sensitive basis hidden dimensions differ")
    flat = update.reshape(-1, update.shape[-1])
    projected = (flat @ q) @ q.transpose(0, 1)
    return projected.reshape_as(update)


def project_repair_update(update: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return update - sensitive_component(update, basis)


def parameter_energy(update: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if update.shape != weights.shape:
        raise ValueError("parameter weights must match update shape")
    return (weights * update.square()).sum()


class TCR(BaseMethod):
    spec = MethodSpec("tcr", "IDEA_01_TCR", "Tri-layer contract repair")

    def __init__(self, config: TCRConfig | None = None):
        self.config = config or TCRConfig()

    def prepare(self, ctx: PrepareContext) -> MethodState:
        boundary = ctx.boundary
        return MethodState(values={
            "sensitive_basis": (
                boundary.representation_sensitivity if boundary else None),
            "parameter_weights": boundary.parameter_weights if boundary else None,
        })

    def compute_loss(self, ctx: StepContext, state: MethodState) -> LossOutput:
        if ctx.logits is None or ctx.nll is None:
            raise ValueError("TCR requires logits and NLL")
        n_forget = len(ctx.forget_rows)
        if n_forget == 0 or ctx.entity_ids is None:
            raise ValueError("TCR requires forget rows and entity ids")
        forget_logits = ctx.logits[:n_forget, -1, :]
        logp = F.log_softmax(forget_logits, dim=-1)
        disclose = logp.gather(1, ctx.entity_ids.view(-1, 1)).squeeze(1)
        margins = -disclose
        margin_loss = margin_violation(margins, self.config.margin_threshold).mean()

        basis = state.values.get("sensitive_basis")
        representation_loss = margins.new_zeros(())
        if basis is not None and ctx.hidden is not None:
            component = sensitive_component(ctx.hidden, basis.to(ctx.hidden.device))
            representation_loss = component.square().mean()
        loss = (ctx.nll.mean() + self.config.lambda_margin * margin_loss
                + self.config.lambda_representation * representation_loss)
        risk = torch.sigmoid(self.config.margin_threshold - margins).mean()
        state.values["last_margin"] = margins.detach().min()
        return LossOutput(loss=loss, diagnostics={
            "nll": ctx.nll.mean().detach(),
            "tcr_margin_loss": margin_loss.detach(),
            "tcr_representation_loss": representation_loss.detach(),
            "tcr_worst_margin": margins.detach().min(),
            "disclosure_risk": risk.detach(),
            "utility": torch.sigmoid(-ctx.nll.mean().detach()),
            "collapse_rate": torch.zeros((), device=loss.device),
        })

    def constrain_update(self, update: Update, state: MethodState) -> Update:
        basis = state.values.get("sensitive_basis")
        weights = state.values.get("parameter_weights")
        projected = {}
        for name, gradient in update.gradients.items():
            result = gradient
            if (basis is not None and gradient is not None and gradient.dim() > 0
                    and gradient.shape[-1] == basis.shape[0]):
                result = project_repair_update(gradient, basis.to(gradient.device))
            if weights is not None and result is not None and result.shape == weights.shape:
                energy = parameter_energy(result, weights.to(result.device))
                if float(energy.detach().cpu()) > self.config.parameter_budget > 0:
                    scale = (self.config.parameter_budget / energy.detach().clamp_min(1e-12)).sqrt()
                    result = result * scale
            projected[name] = result
        return Update(gradients=projected)

    def accept(self, candidate: Candidate, evidence: EvidenceBatch,
               state: MethodState) -> Decision:
        if evidence.disclosure_risk > self.config.risk_budget:
            return rollback("TCR behavior contract exceeded")
        if evidence.utility < self.config.min_utility:
            return rollback("TCR utility contract violated")
        if evidence.collapse_rate > self.config.max_collapse:
            return rollback("TCR collapse contract violated")
        if evidence.structure_valid_rate < self.config.min_structure_valid:
            return rollback("TCR structure contract violated")
        return accept("TCR calibration contracts satisfied")
