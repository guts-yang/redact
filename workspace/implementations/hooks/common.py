"""Shared model/data plumbing for the full method hooks.

These helpers are intentionally small: layer discovery, a layer-patching
context manager, functional forward, CoT step splitting, and token-span
collection. They are used by the CSDA/RRBU/CFA/RIPPLE hook adapters.
"""
from __future__ import annotations

import contextlib
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Find the decoder-layer ModuleList in a HF/PEFT model."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.endswith("layers") and len(module) > 0:
            candidates.append((len(name), module))
    if not candidates:
        raise ValueError("could not find a decoder-layer ModuleList")
    return max(candidates, key=lambda item: item[0])[1]


@contextlib.contextmanager
def patch_layer(
    model: nn.Module,
    layer_idx: int,
    positions: Sequence[int],
    replacement: torch.Tensor,
):
    """Replace hidden states at ``positions`` in one decoder layer.

    ``replacement`` is ``[len(positions), H]`` or ``[H]``. The hook is removed
    on exit. This is the activation-patching primitive used by CSDA.
    """
    layers = get_transformer_layers(model)
    if layer_idx < 0:
        layer_idx += len(layers)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range")
    pos = list(positions)
    if not pos:
        yield
        return

    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.dim() != 3:
            return output
        rep = replacement.to(dtype=tensor.dtype, device=tensor.device)
        if rep.dim() == 1:
            rep = rep.view(1, 1, -1).expand(1, len(pos), -1)
        elif rep.dim() == 2 and rep.shape[0] == 1:
            rep = rep.view(1, 1, -1).expand(1, len(pos), -1)
        elif rep.dim() == 2:
            rep = rep.view(1, len(pos), -1)
        patched = tensor.clone()
        patched[:, pos, :] = rep
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def functional_forward(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_hidden_states: bool = False,
):
    """Forward with an explicit parameter dict (used by RRBU).

    ``params`` may contain only trainable parameters; unspecified parameters
    keep their module values. The caller is responsible for passing names that
    exist in ``model.named_parameters()``.
    """
    kwargs = {"output_hidden_states": output_hidden_states}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return torch.func.functional_call(
        model, params, args=(input_ids,), kwargs=kwargs, strict=False)


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample next-token NLL with -100 ignored labels."""
    tok_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100,
    ).view(labels.shape[0], -1)
    valid = (labels[:, 1:] != -100).float()
    return (tok_loss * valid).sum(1) / valid.sum(1).clamp(min=1.0)


def split_cot_steps(
    cot: str,
    max_steps: int = 24,
    min_chars: int = 20,
) -> List[str]:
    """Split a CoT into paragraph/sentence steps for CSDA/SCOPE.

    This is a deterministic, model-free splitter. It is not a semantic parser;
    it exists so the hook has explicit ``r_t`` boundaries.
    """
    text = (cot or "").strip()
    if not text:
        return []
    chunks = re.split(r"\n\s*\n", text)
    steps: List[str] = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= min_chars:
                steps.append(sent)
    if not steps:
        steps = [text]
    if len(steps) > max_steps:
        stride = max(1, len(steps) // max_steps)
        steps = steps[::stride][:max_steps]
    return steps


def encode_step_spans(
    tokenizer,
    prompt: str,
    steps: Sequence[str],
    max_len: int = 512,
) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
    """Tokenize prompt + steps and return token spans for each kept step."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ids = list(prompt_ids)
    spans: List[Tuple[int, int]] = []
    kept: List[str] = []
    for step in steps:
        step_ids = tokenizer(step, add_special_tokens=False)["input_ids"]
        if not step_ids:
            continue
        if len(ids) + len(step_ids) > max_len:
            break
        start = len(ids)
        ids.extend(step_ids)
        spans.append((start, len(ids)))
        kept.append(step)
    return ids, spans, kept


def collect_step_hidden(
    model: nn.Module,
    tokenizer,
    prompt: str,
    steps: Sequence[str],
    layer: int = -1,
    max_len: int = 512,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[str]]:
    """Return step-level hidden states ``[S, H]`` and token spans."""
    ids, spans, kept = encode_step_spans(tokenizer, prompt, steps, max_len)
    input_ids = torch.tensor([ids], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]
    reps = []
    for start, end in spans:
        reps.append(hidden[start:end].mean(dim=0))
    if not reps:
        raise ValueError("no step spans collected")
    return torch.stack(reps), spans, kept


def as_float_features(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()
