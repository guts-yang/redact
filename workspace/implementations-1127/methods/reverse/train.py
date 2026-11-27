"""REVERSE (IDEA_05_REVERSE-1127): bounded reversibility radius.

GPU-cluster entry: bridges runtime.Context -> core.reverse_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import REVERSE, ReverseConfig, reverse_loss


def setup(ctx, cfg):
    ctx.extras["method"] = REVERSE(ReverseConfig(eta=0.5))
    ctx.extras["suppression"] = torch.randn(16, 8, device=ctx.device, requires_grad=True)


def compute_loss(ctx):
    m = ctx.extras.get("method") or REVERSE(ReverseConfig())
    p = F.softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1).gather(
        1, ctx.entity_ids.view(-1,1)).squeeze(1)
    leak_zero = p.mean()
    # leak under K sampled steering directions (differentiable surrogate)
    k = 8
    dirs = torch.randn(k, 16, device=p.device) * 0.1
    norms = dirs.norm(dim=1)
    # steering-direction leak gains must be functions of the model logits so
    # the inner-max excess carries gradients into theta (not a fixed constant)
    gains = torch.sigmoid(dirs @ ctx.logits[:ctx.n_forget, -1, :16].mean(0))
    leak_under_v = leak_zero + gains * p.detach().mean().clamp(0, 1) * 0.5
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else ctx.nll.new_zeros(())
    suppression = ctx.extras.get("suppression")
    if suppression is None:
        suppression = torch.eye(8, device=p.device)
    return m.loss(leak_zero, leak_under_v, norms, retain_kl, suppression)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_reverse")
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
    rcfg = RunConfig(method="reverse", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
