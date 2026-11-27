"""PROBE (IDEA_10_PROBE-1127): probe-adversarial undecodable erasure.

GPU-cluster entry: bridges runtime.Context -> core.undecodable_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import PROBE, ProbeConfig, undecodable_loss


def setup(ctx, cfg):
    m = PROBE(ProbeConfig())
    ctx.extras["method"] = m
    ctx.extras["probes"] = None


def compute_loss(ctx):
    m = ctx.extras.get("method") or PROBE(ProbeConfig())
    if ctx.hidden is not None:
        step_hidden = ctx.hidden[:ctx.n_forget].reshape(-1, ctx.hidden.shape[-1])[:8]
    else:
        step_hidden = torch.zeros(8, 16, device=ctx.logits.device)
    h = step_hidden.shape[-1]
    probes = ctx.extras.get("probes")
    if not probes:
        probes = [m.new_probe(h), m.new_probe(h)]
        ctx.extras["probes"] = probes
    membership = torch.cat([torch.ones(4), torch.zeros(4)]).to(step_hidden.device)[:step_hidden.shape[0]]
    base = step_hidden.detach().clone()
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else step_hidden.new_zeros(())
    return m.loss(step_hidden, membership, probes, retain_kl, base)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_probe")
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
    rcfg = RunConfig(method="probe", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
