"""Lazy HF/PEFT model helpers with explicit device placement."""
from __future__ import annotations

from typing import Sequence

from upbd_impl.runtime.device import resolve_device, resolve_dtype


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(path, *, device="cpu", dtype="fp32", attn_implementation="sdpa"):
    from transformers import AutoModelForCausalLM

    target_device = resolve_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=resolve_dtype(dtype),
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if target_device.type == "cuda":
        model = model.to(target_device)
    return model


def add_lora(model, *, rank: int = 8, alpha: int | None = None):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ))


def encode(tokenizer, prompt: str, target: str, max_len: int):
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [tokenizer.eos_token_id]
    if len(target_ids) >= max_len:
        target_ids = target_ids[:max_len - 1]
    if len(prompt_ids) + len(target_ids) > max_len:
        prompt_ids = prompt_ids[-(max_len - len(target_ids)):]
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def collate(tokenizer, items: Sequence[tuple[str, str]], max_len: int, device="cpu"):
    import torch

    encoded = [encode(tokenizer, prompt, target, max_len) for prompt, target in items]
    width = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id
    input_ids, attention, labels = [], [], []
    for ids, labs in encoded:
        padding = width - len(ids)
        input_ids.append(ids + [pad] * padding)
        attention.append([1] * len(ids) + [0] * padding)
        labels.append(labs + [-100] * padding)
    target = resolve_device(device)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=target),
        torch.tensor(attention, dtype=torch.long, device=target),
        torch.tensor(labels, dtype=torch.long, device=target),
    )


def forward(model, ids, attention, labels, *, output_hidden_states: bool = False):
    import torch.nn.functional as F

    out = model(
        input_ids=ids, attention_mask=attention, labels=labels,
        output_hidden_states=output_hidden_states,
    )
    logits = out.logits
    token_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape[0], -1)
    valid = (labels[:, 1:] != -100).to(token_loss.dtype)
    nll = (token_loss * valid).sum(1) / valid.sum(1).clamp(min=1.0)
    hidden = out.hidden_states[-1] if output_hidden_states else None
    return out, logits, hidden, nll
