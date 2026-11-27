"""Model loading + forward helpers + LoRA wiring + generation utilities.

All HF/PEFT imports are LAZY (inside functions) so that `python selftest.py`
and every core unit test run on a machine with only torch+stdlib. On the GPU
cluster these resolve against transformers/peft installed in the venv.
"""
from __future__ import annotations

import contextlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import paths


def _resolve_dtype():
    """Training/eval dtype. Frozen protocol is fp32 (configs/repro_order.yaml).

    bf16 is accepted as an explicit opt-in but is NOT the default: on H20 with
    torch 2.5.1 it has triggered SIGFPE during generation (see
    R-TOFU/workspace/code/decode_eval.py).
    """
    name = (os.environ.get("UNL1127_DTYPE") or "float32").strip().lower()
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported UNL1127_DTYPE={name!r}; use fp32 or bf16")
    return mapping[name]


def load_model(attn_implementation: str = "sdpa", torch_dtype=None, tokenizer_only: bool = False):
    """Load target LRM + tokenizer; optionally attach a base PEFT adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch_dtype is None:
        torch_dtype = _resolve_dtype()
    tok = AutoTokenizer.from_pretrained(str(paths.TARGET_MODEL))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if tokenizer_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(
        str(paths.TARGET_MODEL), torch_dtype=torch_dtype,
        attn_implementation=attn_implementation).to("cuda")
    model.config.use_cache = False
    if paths.BASE_ADAPTER:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, paths.BASE_ADAPTER, is_trainable=True)
    return tok, model


def add_lora(model, r: int = 8, alpha: int = 32):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none"))


def load_unlearned_reference(attn_implementation: str = "sdpa"):
    """A second checkpoint that has already forgotten (used by DRIFT/REVERSE/LAYER/PROBE).

    Accepts either a PEFT adapter directory (``adapter_config.json`` present)
    or a full merged checkpoint (``config.json`` + weights). The R2MU
    ``checkpoint-last`` on the cluster is a merged model, not an adapter.
    """
    from transformers import AutoModelForCausalLM
    if not paths.UNLEARNED_ADAPTER:
        return None
    ref = str(paths.UNLEARNED_ADAPTER)
    torch_dtype = _resolve_dtype()
    if os.path.isfile(os.path.join(ref, "adapter_config.json")):
        model = AutoModelForCausalLM.from_pretrained(
            str(paths.TARGET_MODEL), torch_dtype=torch_dtype,
            attn_implementation=attn_implementation).to("cuda")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ref)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ref, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation).to("cuda")
    model.config.use_cache = False
    model.eval()
    return model


def collate(tok, examples: Sequence[Tuple[List[int], List[int], List[int]]], max_len: int):
    """Pad a list of (ids, labels, mask) to a fixed max_len."""
    import torch
    ids = torch.full((len(examples), max_len), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, (ii, ll, mm) in enumerate(examples):
        n = min(len(ii), max_len)
        ids[i, :n] = torch.tensor(ii[:n], dtype=torch.long)
        labels[i, :n] = torch.tensor(ll[:n], dtype=torch.long)
        mask[i, :n] = torch.tensor(mm[:n], dtype=torch.long)
    return ids, labels, mask


def encode_example(tok, prompt: str, target: str, max_len: int):
    """Encode prompt+target with loss only on the target tokens."""
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    mask = [1] * len(ids)
    return ids, labels, mask


def forward(model, ids, labels, mask, output_hidden_states: bool = True):
    """Return (logits [B,L,V], hidden [B,L,H] last layer, per-sample NLL)."""
    ids = ids.to("cuda"); labels = labels.to("cuda"); mask = mask.to("cuda")
    out = model(input_ids=ids, attention_mask=mask,
                output_hidden_states=output_hidden_states)
    logits = out.logits
    hidden = out.hidden_states[-1] if output_hidden_states else None
    tok_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100,
    ).view(labels.shape[0], -1)
    valid = (labels[:, 1:] != -100).float()
    nll = (tok_loss * valid).sum(1) / valid.sum(1).clamp(min=1.0)
    return logits, hidden, nll


def entity_leak_loss(logits: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
    """-log p(entity | context) at the last position, batched."""
    lp = F.log_softmax(logits[:, -1, :], dim=-1)
    return -lp.gather(1, entity_ids.view(-1, 1)).squeeze(1).mean()


def get_layers(model) -> "torch.nn.ModuleList":
    import torch.nn as nn
    cands = [(len(n), m) for n, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and n.endswith("layers") and len(m) > 0]
    if not cands:
        raise ValueError("no decoder-layer ModuleList found")
    return max(cands, key=lambda x: x[0])[1]


@contextlib.contextmanager
def patch_layer(model, layer_idx: int, positions: Sequence[int], replacement: torch.Tensor):
    """Activation patching at one decoder layer (used by LAYER/REVERSE/FAITH/DRIFT)."""
    layers = get_layers(model)
    if layer_idx < 0:
        layer_idx += len(layers)
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
        patched = tensor.clone(); patched[:, pos, :] = rep
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def steer_layer(model, layer_idx: int, v: torch.Tensor, alpha: float = 1.0):
    """Add a steering vector to all positions of one layer (REVERSE attack)."""
    layers = get_layers(model)
    if layer_idx < 0:
        layer_idx += len(layers)

    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.dim() != 3:
            return output
        add = (alpha * v).to(dtype=tensor.dtype, device=tensor.device).view(1, 1, -1)
        patched = tensor + add
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_budget(model, tok, prompt: str, budget: int, max_len: int = 512) -> str:
    """Generate with a fixed reasoning budget (BUDGET). budget==0 -> ZeroThink-like."""
    ids = tok(prompt, return_tensors="pt").to("cuda")
    n = max(1, min(int(budget), max_len))
    if budget == 0:
        # force immediate answer region: close the think tag first
        prompt = prompt + "</think>\n\n"
        ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def sample_answers(model, tok, prompt: str, n: int, temperature: float = 1.0,
                   max_new_tokens: int = 128) -> List[str]:
    """N sampled generations for CONSENSUS majority-vote recovery."""
    ids = tok(prompt, return_tensors="pt").to("cuda")
    outs = model.generate(**ids, max_new_tokens=max_new_tokens,
                          do_sample=True, temperature=temperature, top_p=0.95,
                          num_return_sequences=n, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in outs]


@torch.no_grad()
def collect_step_hidden(model, tok, prompt: str, steps: Sequence[str],
                        layer: int = -1, max_len: int = 512):
    """Step-level hidden states [S,H] + token spans (step == CoT sentence)."""
    from .data import split_cot_steps  # local, stdlib only
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    ids = list(p_ids); spans: List[Tuple[int, int]] = []
    for s in steps:
        sids = tok(s, add_special_tokens=False)["input_ids"]
        if not sids or len(ids) + len(sids) > max_len:
            break
        start = len(ids); ids.extend(sids); spans.append((start, len(ids)))
    inp = torch.tensor([ids], device="cuda")
    out = model(input_ids=inp, output_hidden_states=True)
    h = out.hidden_states[layer][0]
    reps = torch.stack([h[s:e].mean(0) for s, e in spans]) if spans else None
    return reps, spans
