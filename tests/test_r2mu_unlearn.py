"""Tests for scripts/r2mu/unlearn.py helper functions.

Usage:
    conda run -n redact python -m pytest tests/test_r2mu_unlearn.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "r2mu"))

from unlearn import (  # noqa: E402
    build_batch,
    cycle,
    cycle_pairs,
    format_rtofu_sample,
)


def test_cycle_uses_seed_for_ordering():
    items = list(range(10))
    stream_a = cycle(items, seed=1)
    stream_b = cycle(items, seed=2)
    first_a = [next(stream_a) for _ in range(10)]
    first_b = [next(stream_b) for _ in range(10)]
    # Different seeds -> different first-pass permutations (probability of
    # collision on 10 elements is 1/10! ≈ vanishingly small).
    assert first_a != first_b


def test_cycle_same_seed_gives_same_sequence():
    items = list(range(10))
    stream_a = cycle(items, seed=42)
    stream_b = cycle(items, seed=42)
    for _ in range(20):
        assert next(stream_a) == next(stream_b)


def test_cycle_covers_every_element_per_pass():
    items = list(range(10))
    stream = cycle(items, seed=0)
    got = [next(stream) for _ in range(10)]
    # First pass must be a permutation — every element exactly once.
    assert sorted(got) == items


def test_cycle_infinite():
    items = [1, 2]
    stream = cycle(items, seed=0)
    out = [next(stream) for _ in range(10)]
    # Each element appears 5 times (10 draws / 2 elements).
    assert out.count(1) == 5
    assert out.count(2) == 5


def test_cycle_pairs_preserves_alignment():
    a = ["a1", "a2", "a3", "a4", "a5"]
    b = ["b1", "b2", "b3", "b4", "b5"]
    # Expected: the i'th element of a should be paired with the i'th element of b
    # across any permutation.
    stream = cycle_pairs(a, b, seed=0)
    for _ in range(50):
        x, y = next(stream)
        suffix = x[1:]  # "1", "2", ...
        assert y == f"b{suffix}", f"Alignment broken: got ({x}, {y})"


def test_cycle_pairs_different_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        cycle_pairs(["a", "b"], ["x", "y", "z"], seed=0)


def test_cycle_pairs_different_seeds_give_different_orderings():
    a = list(range(10))
    b = [f"b{i}" for i in range(10)]
    stream_1 = cycle_pairs(a, b, seed=1)
    stream_2 = cycle_pairs(a, b, seed=2)
    first_1 = [next(stream_1)[0] for _ in range(10)]
    first_2 = [next(stream_2)[0] for _ in range(10)]
    assert first_1 != first_2


def test_build_batch_pulls_correct_sizes():
    forget = cycle(["f1", "f2", "f3", "f4"], seed=1)
    retain = cycle(["r1", "r2", "r3", "r4"], seed=2)
    extent = cycle(["e1", "e2", "e3", "e4"], seed=3)
    s1k = cycle_pairs(["a1", "a2"], ["R1", "R2"], seed=4)

    batch = build_batch(forget, retain, extent, s1k, batch_size=2)

    assert len(batch["forget"]) == 2
    assert len(batch["retain"]) == 2
    assert len(batch["extent"]) == 2
    assert len(batch["assistant"]) == 2
    assert len(batch["reasoning"]) == 2


def test_build_batch_s1k_alignment_within_batch():
    """assistant[i] and reasoning[i] must come from the same underlying s1K record."""
    s1k = cycle_pairs(["a1", "a2", "a3"], ["R1", "R2", "R3"], seed=5)
    forget = cycle(["f"] * 10, seed=1)
    retain = cycle(["r"] * 10, seed=2)
    extent = cycle(["e"] * 10, seed=3)

    batch = build_batch(forget, retain, extent, s1k, batch_size=3)

    for a, r in zip(batch["assistant"], batch["reasoning"]):
        # a is "a{k}"; r must be "R{k}" with the same k.
        assert a[0] == "a" and r[0] == "R"
        assert a[1:] == r[1:], f"s1K pair misaligned: ({a}, {r})"


def test_build_batch_parallel_streams_not_lockstep():
    """With distinct seeds, parallel streams shouldn't produce correlated indices."""
    items = list(range(20))
    forget = cycle([f"f{i}" for i in items], seed=1)
    retain = cycle([f"r{i}" for i in items], seed=2)
    extent = cycle([f"e{i}" for i in items], seed=3)
    s1k = cycle_pairs(["a"] * 20, ["R"] * 20, seed=4)

    batch = build_batch(forget, retain, extent, s1k, batch_size=20)

    # Extract the integer index from each "f{i}" / "r{i}" / "e{i}" string.
    f_idx = [int(s[1:]) for s in batch["forget"]]
    r_idx = [int(s[1:]) for s in batch["retain"]]
    e_idx = [int(s[1:]) for s in batch["extent"]]
    # Streams should produce different orderings. Exact match probability is 1/20!.
    assert f_idx != r_idx
    assert f_idx != e_idx
    assert r_idx != e_idx


def test_format_rtofu_sample_wraps_question_with_prefix_suffix():
    got = format_rtofu_sample(
        question="Who is X?",
        answer="A famous person.",
        prompt_prefix="<|U|>",
        prompt_suffix="<|A|>",
        answer_prefix="",
        eos_token="<eos>",
    )
    assert got == "<|U|>Who is X?<|A|>A famous person.<eos>"


def test_format_rtofu_sample_includes_answer_prefix_and_eos():
    got = format_rtofu_sample(
        question="Q?",
        answer="A.",
        prompt_prefix="<U>",
        prompt_suffix="<A>",
        answer_prefix="ANS:",
        eos_token="</s>",
    )
    assert got == "<U>Q?<A>ANS:A.</s>"


def test_format_rtofu_sample_empty_tokens_degrade_gracefully():
    """All formatting tokens default to empty strings (e.g. for models without
    chat templates); the result should be just q + a with no separator added."""
    got = format_rtofu_sample(
        question="Who is X?",
        answer="A famous person.",
        prompt_prefix="",
        prompt_suffix="",
        answer_prefix="",
        eos_token="",
    )
    assert got == "Who is X?A famous person."
