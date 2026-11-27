"""LAYER core: decode-consistent erasure-depth localization (IDEA_09_LAYER-1127).

Self-contained: imports ONLY torch + stdlib. Implements proposal §3:
  Depth(p) = shallowest patch layer that restores leak, DC = cross-strategy
  depth spread, soft-depth variance consistency, depth-guided training loss,
  and the structural certificate Cert(delta).
The layer-patching scan is in runtime.model; this core is the differentiable
objective over per-(strategy, layer) leak-restoration values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class LayerConfig:
    lam: float = 1.0        # retain KL
    gamma: float = 0.5      # weight on soft-DC (depth variance) consistency
    eps: float = 0.05       # leak threshold defining depth
    ell_shared: float = 8.0 # target: depth must reach shared mid layers
    temperature: float = 5.0


@dataclass
class Out:
    loss: torch.Tensor
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def float_diagnostics(self) -> Dict[str, float]:
        return {k: (float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v))
                for k, v in self.diagnostics.items()}


def soft_depth(leak_restore: torch.Tensor, ell_shared: float = 8.0,
               temperature: float = 5.0) -> torch.Tensor:
    """Differentiable Depth(p): expected shallowest patch layer that restores leak.

    leak_restore [L] = leak after patching layer l (ascending l, shallow->deep).
    A shallow restoration (small l) is BAD; we return a soft index of the first
    layer whose leak exceeds eps, so a smaller value = shallower = worse.
    """
    idx = torch.arange(leak_restore.shape[0], dtype=leak_restore.dtype,
                       device=leak_restore.device)
    # Weight layers by how much they RESTORE leak relative to the profile mean.
    # Subtracting the mean makes a single shallow peak dominate (depth -> 0),
    # while a deep-only peak pushes depth toward large l.
    w = torch.softmax((leak_restore - leak_restore.mean()) * temperature, dim=0)
    return (w * idx).sum()


def depth_consistency(depths: torch.Tensor) -> torch.Tensor:
    """DC = max_p,p' |Depth(p)-Depth(p')| (soft: variance + range)."""
    if depths.numel() < 2:
        return depths.new_zeros(())
    return (depths.max() - depths.min()) + depths.var()


def certificate(depths: torch.Tensor, ell_shared: float = 8.0,
                delta: float = 1.0) -> torch.Tensor:
    """Cert = fraction of strategies with Depth<=ell_shared and DC<=delta (soft)."""
    deep = torch.sigmoid((ell_shared - depths) * 2.0).mean()
    dc = (depths.max() - depths.min()) if depths.numel() > 1 else depths.new_zeros(())
    consistent = torch.sigmoid((delta - dc) * 2.0)
    return deep * consistent


def layer_loss(
    leak_restore: torch.Tensor,   # [P,L] leak after patching layer l, per strategy p
    retain_kl: torch.Tensor,
    config: LayerConfig | None = None,
) -> Out:
    cfg = config or LayerConfig()
    depths = torch.stack([soft_depth(leak_restore[p], cfg.ell_shared, cfg.temperature)
                          for p in range(leak_restore.shape[0])])
    dc = depth_consistency(depths)
    # Penalize shallow restoration: hinge on "restored too easily at shallow layers"
    shallow = torch.clamp(cfg.ell_shared - depths.min(), min=0.0)
    loss = shallow + cfg.gamma * dc + cfg.lam * retain_kl
    return Out(loss=loss, diagnostics={
        "layer_min_depth": depths.min().detach(),
        "layer_max_depth": depths.max().detach(),
        "layer_DC": dc.detach(),
        "layer_Cert": certificate(depths, cfg.ell_shared).detach(),
        "layer_shallow_hinge": shallow.detach(),
    })


class LAYER:
    def __init__(self, config: LayerConfig | None = None):
        self.config = config or LayerConfig()

    def loss(self, *args, **kwargs):
        return layer_loss(*args, config=self.config, **kwargs)

    def cert(self, depths):
        return certificate(depths, self.config.ell_shared)
