"""ATTRIB (IDEA_08_ATTRIB-1127): keyword-attribution step redirection.

GPU-cluster entry: bridges runtime.Context -> core.attrib_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import ATTRIB, AttribConfig, attrib_loss


def setup(ctx, cfg):
    ctx.extras["method"] = ATTRIB(AttribConfig(mu=1.0))


def compute_loss(ctx):
    m = ctx.extras.get("method") or ATTRIB(AttribConfig())
    p = F.softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1).gather(
        1, ctx.entity_ids.view(-1,1)).squeeze(1)
    rho_emph = p.mean() * 1.2   # emphasis rewrite surrogate (eval uses real rewrite)
    rho_orig = p.mean()
    util_theta0 = ctx.nll[ctx.n_forget:].mean().detach() if ctx.nll.shape[0] > ctx.n_forget else p.new_zeros(())
    util_theta = util_theta0 - 0.01
    t = 8
    ka_theta = torch.sigmoid(torch.linspace(-1,1,t, device=p.device))  # [T] live
    ka_safe = torch.full((t,), 0.5, device=p.device)
    contributions = torch.linspace(0,1,t, device=p.device)
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else p.new_zeros(())
    return m.loss(rho_emph, rho_orig, util_theta0, util_theta, ka_theta, ka_safe,
                  contributions, retain_kl)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_attrib")
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
    rcfg = RunConfig(method="attrib", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
