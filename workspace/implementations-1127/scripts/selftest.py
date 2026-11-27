#!/usr/bin/env python3
"""CPU-only self-test: verify every method's compute_loss runs + backprops.

Runs on a laptop with ONLY torch + stdlib (no transformers/peft/datasets/GPU).
For each method we build a synthetic runtime.Context with the exact tensor
shapes the GPU runner produces, call compute_loss(ctx), assert:
  - loss is a finite scalar tensor with requires_grad
  - backward() succeeds and yields at least one finite parameter gradient
  - float_diagnostics() returns finite floats
This proves the core + train.py bridge is correct BEFORE uploading to the GPU
cluster. It does NOT prove scientific claims (no real model / data involved).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.runner import Context  # noqa: E402

METHODS = ["budget", "pivot", "consensus", "drift", "reverse",
           "faith", "graph", "attrib", "layer", "probe"]

B, L, V, H = 4, 64, 128, 16
NF = 4  # n_forget


def make_ctx():
    """Synthetic Context mirroring runner's real tensor contract (CPU)."""
    logits = torch.randn(B, L, V, requires_grad=True)
    hidden = torch.randn(B, L, H, requires_grad=True)
    nll = torch.rand(B, requires_grad=True)
    entity_ids = torch.randint(0, V, (NF,))
    ids = torch.randint(0, V, (B, L)); labels = torch.full((B, L), -100); mask = torch.ones(B, L)
    model = SimpleNamespace()  # setup() may attach parameters here
    return Context(
        model=model, ref_model=None, tok=None,
        forget_rows=[{"question": "q", "answer": "A Fictional Person", "cot": ""}] * NF,
        retain_rows=[{"question": "q", "answer": "ok"}] * (B - NF),
        safe_texts=["I don't know"] * NF,
        batch=(ids, labels, mask), logits=logits, hidden=hidden, nll=nll,
        n_forget=NF, entity_ids=entity_ids, seed=1001, step=0, max_len=L,
        device="cpu", extras={},
    )


def run_one(slug):
    import importlib
    mod = importlib.import_module(f"methods.{slug}.train")
    ctx = make_ctx()
    if mod.setup is not None:
        mod.setup(ctx, None)
    # collect any parameters setup attached to ctx.model / extras
    params = []
    for obj in (ctx.model, ctx.extras):
        vals = vars(obj).values() if hasattr(obj, "__dict__") else (obj.values() if isinstance(obj, dict) else [])
        for v in vals:
            if isinstance(v, torch.Tensor) and v.requires_grad:
                params.append(v)
    # core inputs often derive from ctx tensors without params; include base tensors
    base_params = [ctx.logits, ctx.hidden, ctx.nll]
    out = mod.compute_loss(ctx)
    loss = out.loss
    assert isinstance(loss, torch.Tensor), f"{slug}: loss not a Tensor"
    assert loss.dim() == 0, f"{slug}: loss not scalar ({tuple(loss.shape)})"
    assert torch.isfinite(loss), f"{slug}: loss non-finite"
    assert loss.requires_grad, f"{slug}: loss has no grad"
    loss.backward()
    grads = [p.grad for p in (params + base_params)
             if isinstance(p, torch.Tensor) and p.grad is not None]
    finite_grads = [g for g in grads if torch.isfinite(g).all()]
    assert len(finite_grads) > 0, f"{slug}: no finite gradients produced"
    diag = out.float_diagnostics()
    assert isinstance(diag, dict), f"{slug}: diagnostics not a dict"
    for k, v in diag.items():
        assert isinstance(v, float) and math.isfinite(v), f"{slug}: diag {k}={v} not finite float"
    return float(loss.detach()), diag


def main():
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | device: CPU")
    print(f"selftest root: {ROOT}")
    print("-" * 72)
    failures = []
    for slug in METHODS:
        try:
            loss, diag = run_one(slug)
            keys = ",".join(sorted(diag)[:3])
            print(f"  {slug:10s} OK   loss={loss:9.4f}  diag[{len(diag)}]={keys}...")
        except Exception as e:
            import traceback
            failures.append(slug)
            print(f"  {slug:10s} FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
    print("-" * 72)
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print(f"ALL {len(METHODS)} METHODS PASS (finite loss + backward + diagnostics)")
    print("NOTE: CPU synthetic tensors only. No claim, no ARIS gate, no real model/data.")
    sys.exit(0)


if __name__ == "__main__":
    main()
