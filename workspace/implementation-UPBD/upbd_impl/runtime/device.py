"""Explicit device resolution. No GPU is selected implicitly."""
from __future__ import annotations

import os


def resolve_device(name: str = "cpu"):
    import torch

    normalized = name.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized not in {"cuda", "cuda:0"}:
        raise ValueError("device must be 'cpu' or 'cuda' (logical device zero)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or visible.strip() == "-1":
        raise RuntimeError(
            "CUDA requires an explicit non-empty CUDA_VISIBLE_DEVICES; "
            "no GPU is selected automatically")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device("cuda", 0)


def resolve_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized = name.strip().lower()
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; use fp32 or bf16")
    return mapping[normalized]
