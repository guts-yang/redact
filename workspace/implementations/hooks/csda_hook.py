"""Full CSDA training hook: CoT steps -> layer patching -> step-level loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..csda import CSDA, CSDAConfig
from ..base import MethodOutput
from .common import (collect_step_hidden, encode_step_spans, patch_layer,
                     split_cot_steps)


@dataclass
class CSDAHookConfig:
    layer: int = -1
    max_steps: int = 24
    max_len: int = 512
    device: str = "cuda"
    diag_cov: bool = True


def _key_phrase(answer: str) -> str:
    import re
    candidates = re.findall(r"\b[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)+\b",
                            answer or "")
    return max(candidates, key=lambda p: (len(p.split()), len(p))) if candidates else ""


def entity_token_id(tokenizer, answer: str) -> int:
    """First token of the longest capitalized phrase in the gold answer."""
    phrase = _key_phrase(answer)
    ids = tokenizer(phrase, add_special_tokens=False)["input_ids"] if phrase else []
    return int(ids[0]) if ids else 0


def entity_leak_loss(logits: torch.Tensor, entity_id: int) -> torch.Tensor:
    """-log p(entity | context) at the last position."""
    return -F.log_softmax(logits[:, -1, :], dim=-1)[:, entity_id].mean()


class CSDAHook:
    """Activation-patching adapter for an HF/PEFT causal LM."""

    def __init__(self, model, tokenizer, config: Optional[CSDAHookConfig] = None,
                 csda_config: Optional[CSDAConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CSDAHookConfig()
        self.csda = CSDA(csda_config or CSDAConfig())
        self.retain_mean: Optional[torch.Tensor] = None
        self.retain_cov: Optional[torch.Tensor] = None

    def steps(self, cot: str) -> List[str]:
        return split_cot_steps(cot, self.config.max_steps)

    def collect_retain_stats(
        self,
        examples: Sequence[Tuple[str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate retain step-statistics from (prompt, cot) pairs."""
        reps = []
        for prompt, cot in examples:
            steps = self.steps(cot)
            if not steps:
                continue
            reps.append(collect_step_hidden(
                self.model, self.tokenizer, prompt, steps,
                layer=self.config.layer, max_len=self.config.max_len,
                device=self.config.device)[0])
        if not reps:
            raise ValueError("no retain step representations collected")
        flat = torch.cat(reps, dim=0).float()
        mean = flat.mean(dim=0)
        centered = flat - mean
        if self.config.diag_cov:
            cov = torch.diag(centered.pow(2).mean(dim=0) + 1e-6)
        else:
            cov = centered.T @ centered / max(flat.shape[0] - 1, 1)
            cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        self.retain_mean = mean
        self.retain_cov = cov
        return mean, cov

    def _forward_logits(self, input_ids: torch.Tensor, attention_mask=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    def contributions(
        self,
        prompt: str,
        cot: str,
        entity_id: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return c_t and the step hidden states, using real layer patching."""
        if self.retain_mean is None or self.retain_cov is None:
            raise RuntimeError("call collect_retain_stats first")
        steps = self.steps(cot)
        ids, spans, _ = encode_step_spans(self.tokenizer, prompt, steps,
                                          self.config.max_len)
        if not spans:
            raise ValueError("no token spans for CSDA steps")
        input_ids = torch.tensor([ids], device=self.config.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
            base_logits = out.logits
            hidden = out.hidden_states[self.config.layer][0]
        base = entity_leak_loss(base_logits, entity_id).detach()
        reps = torch.stack([hidden[s:e].mean(dim=0) for s, e in spans])
        mean = self.retain_mean.to(reps.dtype)
        cov = self.retain_cov.to(reps.dtype)
        gen = generator or torch.Generator(device=mean.device).manual_seed(0)
        n_samples = max(1, self.csda.config.n_patch_samples)
        chol = torch.linalg.cholesky(cov)
        contributions = []
        for start, end in spans:
            vals = []
            for _ in range(n_samples):
                eps = torch.randn(1, mean.numel(), device=mean.device,
                                  dtype=mean.dtype, generator=gen)
                replacement = mean.view(1, -1) + eps @ chol.T
                with patch_layer(self.model, self.config.layer,
                                 list(range(start, end)), replacement):
                    with torch.inference_mode():
                        patched_logits = self._forward_logits(input_ids)
                vals.append((entity_leak_loss(patched_logits, entity_id).detach() - base).reshape(()))
            contributions.append(torch.stack(vals).mean())
        return torch.stack(contributions), reps

    def loss(
        self,
        prompt: str,
        cot: str,
        entity_id: int,
        retain_kl: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> MethodOutput:
        """One CSDA objective for a forget CoT. Patches O(T) times.

        Contributions are detached attribution statistics; the step
        representations used in the loss are recomputed with gradients so the
        hook can be used inside a normal training loop.
        """
        c, _ = self.contributions(prompt, cot, entity_id, generator)
        steps = self.steps(cot)
        ids, spans, _ = encode_step_spans(self.tokenizer, prompt, steps,
                                          self.config.max_len)
        input_ids = torch.tensor([ids], device=self.config.device)
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[self.config.layer][0]
        current = torch.stack([hidden[s:e].mean(dim=0) for s, e in spans])
        v_safe = self.retain_mean.to(current.dtype)
        base = current.detach()
        return self.csda.loss(current.unsqueeze(1), base.unsqueeze(1),
                              v_safe, c, retain_kl)
