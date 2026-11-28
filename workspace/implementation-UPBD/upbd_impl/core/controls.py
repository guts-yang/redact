"""Matched attribution controls shared by every proposal."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlConfig:
    name: str
    start_checkpoint: str
    include_forget_channel: bool
    use_proposed_mechanism: bool
    steps: int
    lr: float
    seed: int
    max_len: int
    lora_r: int


def make_matched_controls(
    *, unlearned_checkpoint: str, target_checkpoint: str, steps: int,
    lr: float, seed: int, max_len: int, lora_r: int,
) -> dict[str, ControlConfig]:
    common = dict(steps=steps, lr=lr, seed=seed, max_len=max_len, lora_r=lora_r)
    return {
        "retain_only": ControlConfig(
            "retain_only", unlearned_checkpoint, False, False, **common),
        "target_start": ControlConfig(
            "target_start", target_checkpoint, True, True, **common),
        "full": ControlConfig(
            "full", unlearned_checkpoint, True, True, **common),
    }


def assert_matched(controls: dict[str, ControlConfig]) -> None:
    required = {"retain_only", "target_start", "full"}
    if set(controls) != required:
        raise ValueError(f"controls must be exactly {sorted(required)}")
    full = controls["full"]
    for control in controls.values():
        for field_name in ("steps", "lr", "seed", "max_len", "lora_r"):
            if getattr(control, field_name) != getattr(full, field_name):
                raise ValueError(f"{control.name}: mismatched {field_name}")
    if controls["retain_only"].include_forget_channel:
        raise ValueError("retain-only must exclude the forget channel")
    if not controls["target_start"].include_forget_channel:
        raise ValueError("target-start must use the full behavior recipe")
