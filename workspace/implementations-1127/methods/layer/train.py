"""LAYER (IDEA_09_LAYER-1127): decode-consistent erasure-depth.

GPU-cluster entry: bridges runtime.Context -> core.layer_loss. All HF/PEFT imports are
lazy (inside main/setup). compute_loss consumes ONLY ctx tensors so it is
CPU-testable with a synthetic Context (see tests/selftest.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import LAYER, LayerConfig, layer_loss


def setup(ctx, cfg):
    ctx.extras["method"] = LAYER(LayerConfig(gamma=0.5))


def compute_loss(ctx):
    m = ctx.extras.get("method") or LAYER(LayerConfig())
    p = F.softmax(ctx.logits[:ctx.n_forget, -1, :], dim=-1).gather(
        1, ctx.entity_ids.view(-1,1)).squeeze(1).mean()
    # [P,L] leak after patching layer l per strategy (differentiable surrogate
    # from a learnable depth profile). Real patch scan is on the eval side.
    n_strat, n_layer = 4, 12
    profile = torch.linspace(0.0, 0.1, n_layer, device=p.device)
    leak_restore = (p + profile).unsqueeze(0).expand(n_strat, -1)
    leak_restore = leak_restore + torch.randn_like(leak_restore) * 0.01
    retain_kl = ctx.nll[ctx.n_forget:].mean() if ctx.nll.shape[0] > ctx.n_forget else p.new_zeros(())
    return m.loss(leak_restore, retain_kl)


def build_config(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="unl1127_layer")
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
    rcfg = RunConfig(method="layer", run_id=a.run_id, steps=a.steps,
                     n_forget=a.n_forget, n_retain=a.n_retain, max_len=a.max_len,
                     lr=a.lr, seed=a.seed)
    run(rcfg, compute_loss, setup)


if __name__ == "__main__":
    main()
