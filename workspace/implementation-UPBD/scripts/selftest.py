#!/usr/bin/env python3
"""CPU-only synthetic loss/backward checks for base + six proposal plugins."""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upbd_impl.contracts import (  # noqa: E402
    BoundaryAsset, PrepareContext, RequestSpec, StepContext,
)
from upbd_impl.registry import list_methods, load_method  # noqa: E402


def make_context() -> tuple[PrepareContext, StepContext]:
    batch_size, length, vocab, hidden_dim, n_forget = 4, 16, 32, 8, 2
    logits = torch.randn(batch_size, length, vocab, requires_grad=True)
    hidden = torch.randn(batch_size, length, hidden_dim, requires_grad=True)
    nll = torch.rand(batch_size, requires_grad=True)
    ids = torch.randint(0, vocab, (batch_size, length))
    labels = torch.full((batch_size, length), -100)
    mask = torch.ones(batch_size, length)
    boundary = BoundaryAsset(
        checkpoint="synthetic",
        probe_scope=("direct",),
        representation_sensitivity=torch.randn(hidden_dim, 2),
    )
    requests = (RequestSpec("r1", forget=("x",), protect=("y",)),)
    prepare = PrepareContext(
        boundary=boundary,
        requests=requests,
        device="cpu",
        extras={
            "hidden_dim": hidden_dim,
            "benign_operator_drops": torch.tensor([0.01, 0.02]),
        },
    )
    step = StepContext(
        batch=(ids, labels, mask), logits=logits, hidden=hidden, nll=nll,
        forget_rows=[{"question": "q", "gold_answer": "x"}] * n_forget,
        retain_rows=[{"question": "q", "answer": "y"}] * (batch_size - n_forget),
        safe_texts=["safe"] * n_forget,
        entity_ids=torch.randint(0, vocab, (n_forget,)),
        step=0, seed=1001, device="cpu",
    )
    return prepare, step


def run_one(slug: str) -> tuple[float, dict[str, float]]:
    method = load_method(slug)
    prepare, ctx = make_context()
    state = method.prepare(prepare)
    groups = method.parameter_groups(state)
    output = method.compute_loss(ctx, state)
    loss = output.loss
    assert isinstance(loss, torch.Tensor), f"{slug}: loss is not a tensor"
    assert loss.dim() == 0, f"{slug}: loss is not scalar"
    assert torch.isfinite(loss), f"{slug}: loss is non-finite"
    assert loss.requires_grad, f"{slug}: loss has no gradient path"
    loss.backward()
    params = [ctx.logits, ctx.hidden, ctx.nll]
    params += [param for group in groups for param in group.params]
    grads = [param.grad for param in params
             if isinstance(param, torch.Tensor) and param.grad is not None]
    assert grads, f"{slug}: no gradients produced"
    assert all(torch.isfinite(grad).all() for grad in grads), f"{slug}: non-finite gradient"
    diagnostics = output.float_diagnostics()
    assert diagnostics, f"{slug}: no diagnostics"
    for key, value in diagnostics.items():
        assert isinstance(value, float) and math.isfinite(value), f"{slug}: bad diagnostic {key}"
    return float(loss.detach()), diagnostics


def main() -> int:
    failures = []
    for spec in list_methods():
        try:
            loss, diagnostics = run_one(spec.slug)
            print(f"{spec.slug:10s} PASS loss={loss:.6f} diagnostics={len(diagnostics)}")
        except Exception as exc:
            failures.append(spec.slug)
            print(f"{spec.slug:10s} FAIL {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL base + six proposal plugins PASS synthetic CPU checks")
    print("Synthetic tensors only; no model, network, scientific claim, or ARIS gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
