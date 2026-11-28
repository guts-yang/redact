"""Static method registry with lazy imports and no algorithm side effects."""
from __future__ import annotations

from importlib import import_module

from upbd_impl.contracts import Method, MethodSpec


_SPECS = {
    "upbd_base": MethodSpec("upbd_base", "reference", "Legacy UPBD behavior repair control"),
    "tcr": MethodSpec("tcr", "IDEA_01_TCR", "Tri-layer contract repair"),
    "bless": MethodSpec("bless", "IDEA_02_BLESS", "Benign-learning-stable repair"),
    "rcbf": MethodSpec("rcbf", "IDEA_03_RCBF", "Robust control-barrier forgetting"),
    "stream": MethodSpec("stream", "IDEA_04_STREAM", "Feasibility-aware streaming repair"),
    "tmot": MethodSpec("tmot", "IDEA_05_TMOT", "Tri-marginal optimal-transport repair"),
    "crown": MethodSpec("crown", "IDEA_06_CROWN", "Conflict-aware request composition"),
}

PROPOSAL_ORDER = ("tcr", "bless", "rcbf", "stream", "tmot", "crown")


def list_methods(*, include_base: bool = True) -> list[MethodSpec]:
    names = list(PROPOSAL_ORDER)
    if include_base:
        names.insert(0, "upbd_base")
    return [_SPECS[name] for name in names]


def get_spec(slug: str) -> MethodSpec:
    if slug not in _SPECS:
        raise ValueError(f"unknown method {slug!r}; choose from {sorted(_SPECS)}")
    return _SPECS[slug]


def load_method(slug: str) -> Method:
    get_spec(slug)
    module = import_module(f"upbd_impl.methods.{slug}.train")
    method = module.build_method()
    if method.spec.slug != slug:
        raise ValueError(f"registry mismatch: requested {slug}, got {method.spec.slug}")
    return method
