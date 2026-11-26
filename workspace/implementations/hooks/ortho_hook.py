"""Full ORTHO hook: per-layer reasoning subspace bank and gradient projection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from ..base import MethodOutput
from ..ortho import (ORTHO, ORTHOConfig, estimate_reasoning_subspace,
                     orthogonalize_gradient)
from .common import collect_step_hidden, split_cot_steps


@dataclass
class ORTHOHookConfig:
    layers: Tuple[int, ...] = (-1,)
    k: int = 8
    mu: float = 0.1
    lam: float = 1.0
    max_len: int = 512
    device: str = "cuda"


class ORTHOSubspaceBank:
    def __init__(self, config: Optional[ORTHOHookConfig] = None):
        self.config = config or ORTHOHookConfig()
        self.subspaces: Dict[int, torch.Tensor] = {}

    def collect(
        self,
        model,
        tokenizer,
        examples: Sequence[Tuple[str, str]],
    ) -> Dict[int, torch.Tensor]:
        """examples: (prompt, cot) retain/math CoT pairs."""
        for layer in self.config.layers:
            reps = []
            for prompt, cot in examples:
                steps = split_cot_steps(cot, max_steps=12)
                if len(steps) < 2:
                    continue
                h, _, _ = collect_step_hidden(
                    model, tokenizer, prompt, steps, layer=layer,
                    max_len=self.config.max_len, device=self.config.device)
                reps.append(h.float())
            if not reps:
                raise ValueError("no ORTHO activations collected")
            flat = torch.cat(reps, dim=0)
            self.subspaces[layer] = estimate_reasoning_subspace(
                flat, k=self.config.k, center=True).to(flat.device)
        return self.subspaces

    def get(self, layer: int) -> torch.Tensor:
        key = layer if layer in self.subspaces else -1
        if key not in self.subspaces:
            raise KeyError(f"no subspace for layer {layer}")
        return self.subspaces[key]


class ORTHOHook:
    def __init__(self, config: Optional[ORTHOHookConfig] = None,
                 bank: Optional[ORTHOSubspaceBank] = None):
        self.config = config or ORTHOHookConfig()
        self.bank = bank or ORTHOSubspaceBank(self.config)
        self.ortho = ORTHO(ORTHOConfig(k=self.config.k, mu=self.config.mu,
                                       lam=self.config.lam))

    def collect(self, model, tokenizer, examples):
        return self.bank.collect(model, tokenizer, examples)

    def loss(
        self,
        forget_loss: torch.Tensor,
        current_hidden: torch.Tensor,
        base_hidden: torch.Tensor,
        layer: int,
        retain_kl: torch.Tensor,
    ) -> MethodOutput:
        U = self.bank.get(layer)
        return self.ortho.loss(forget_loss, current_hidden, base_hidden, U,
                               retain_kl)

    def project_gradient(self, grad: torch.Tensor, layer: int) -> torch.Tensor:
        U = self.bank.get(layer)
        if grad.shape[0] == U.shape[0]:
            return orthogonalize_gradient(grad, U)
        if grad.shape[-1] == U.shape[0]:
            return orthogonalize_gradient(grad.T.contiguous(), U).T.contiguous()
        raise ValueError(f"gradient shape {tuple(grad.shape)} does not match H={U.shape[0]}")

    def project_named_gradients(
        self,
        named_grads: Mapping[str, torch.Tensor],
        layer_key: str = "layers",
    ) -> Dict[str, torch.Tensor]:
        """Project gradients whose parameter name contains ``layers.<idx>.``."""
        out: Dict[str, torch.Tensor] = {}
        pattern = re.compile(rf"{re.escape(layer_key)}\.(\d+)\.")
        for name, grad in named_grads.items():
            match = pattern.search(name)
            if not match:
                out[name] = grad
                continue
            layer = int(match.group(1))
            key = layer if layer in self.bank.subspaces else -1
            if key not in self.bank.subspaces:
                out[name] = grad
                continue
            try:
                out[name] = self.project_gradient(grad, key)
            except ValueError:
                out[name] = grad
        return out
