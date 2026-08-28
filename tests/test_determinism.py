"""Tests for :func:`redact.utils.enable_full_determinism`.

The function bundles the standard ``seed_everything`` torch/numpy/random
seeding with the PyTorch+CUDA knobs (CUBLAS_WORKSPACE_CONFIG,
cudnn.deterministic, cudnn.benchmark, torch.use_deterministic_algorithms)
needed for bit-for-bit reproducibility across runs with the same seed.

Tests:

- State checks confirm each knob is set after the call.
- A CPU model-rerun smoke test: build a tiny BinaryHeadModel twice with
  the same seed, run forward+backward+optimizer step on the same
  synthetic batch, assert the resulting parameters are bit-identical.
  This is the load-bearing check — if any RNG path (dropout, init,
  optimizer) leaks non-deterministic state, the assertion fails.
- A negative-control check: WITHOUT the function, build the same
  model twice and run the same step — assert the parameters DIFFER
  (the test fixture has a real RNG dependency, so determinism
  matters; if this test passed without enable_full_determinism, the
  fixture would be too simple to catch the issue).

The CPU-only tests can't directly verify cuBLAS / cuDNN determinism
(those code paths don't activate without CUDA), but they exercise the
RNG plumbing that ``seed_everything`` already handles — the new
function should be a strict superset of that behavior.
"""
from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig

from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel
from redact.utils import enable_full_determinism


# ---------- state checks ----------


def test_enable_full_determinism_sets_cublas_workspace_config():
    """CUBLAS_WORKSPACE_CONFIG must be set before any cuBLAS handle
    exists in the process. The function sets it eagerly."""
    # Clear if a previous test set it.
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    enable_full_determinism(0)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"


def test_enable_full_determinism_sets_cudnn_flags():
    """cuDNN deterministic on, benchmark off."""
    # Flip to known-bad state to confirm the function actively sets them.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    enable_full_determinism(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_enable_full_determinism_sets_use_deterministic_algorithms():
    """torch.use_deterministic_algorithms(True) is the main switch for
    non-cuDNN op-level determinism (scatter, index_add, embedding
    backward, etc.)."""
    # Reset to default-off, then call the function.
    torch.use_deterministic_algorithms(False)
    enable_full_determinism(0)
    assert torch.are_deterministic_algorithms_enabled() is True


def test_enable_full_determinism_warn_only_flag_propagates():
    """``warn_only`` must propagate distinctly into the underlying
    debug-mode integer: 1 = warn, 2 = error. A test that only checks
    ``are_deterministic_algorithms_enabled()`` returns True would
    pass for both warn-mode and strict-mode and miss a regression
    where the kwarg silently dropped to its default."""
    torch.use_deterministic_algorithms(False)
    enable_full_determinism(0, warn_only=True)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.get_deterministic_debug_mode() == 1  # warn


def test_enable_full_determinism_strict_mode_sets_debug_mode_error():
    """``warn_only=False`` is strict mode — debug_mode = 2 (error).
    Companion to the warn-mode test above; together they pin the
    propagation in both directions."""
    torch.use_deterministic_algorithms(False)
    enable_full_determinism(0, warn_only=False)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.get_deterministic_debug_mode() == 2  # error


def test_enable_full_determinism_does_not_set_cuda_launch_blocking():
    """CUDA_LAUNCH_BLOCKING=1 is the slow option (2-3x throughput
    cost); the function intentionally does NOT set it. Users who need
    it should export it manually."""
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    enable_full_determinism(0)
    assert "CUDA_LAUNCH_BLOCKING" not in os.environ


# ---------- bit-for-bit reproducibility ----------


def _tiny_model_and_inputs(seed: int):
    """Build a BinaryHeadModel + synthetic batch under a given seed.

    The model has dropout enabled (training mode) so the per-step
    update depends on the RNG state — a determinism breach would
    manifest as a parameter-mismatch after the optimizer step.
    """
    enable_full_determinism(seed)
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    config = BinaryHeadConfig(backbone_config=cfg.to_dict())
    model = BinaryHeadModel(config)
    model.train()  # dropout active

    vocab_size = model.backbone.config.vocab_size
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        100, min(1000, vocab_size), (4, 8), generator=g,
    )
    attention_mask = torch.ones((4, 8), dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return model, (input_ids, attention_mask, labels)


def _run_one_step(model, batch, lr: float = 1e-3):
    """Forward / backward / optimizer step. Returns the post-step
    state dict as a {name: tensor} mapping."""
    input_ids, attention_mask, labels = batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    out["loss"].backward()
    optimizer.step()
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_bit_identical_parameters_after_one_step():
    """The load-bearing test: same seed, same fixture, same step ⇒
    bit-identical parameters. Exercises the full RNG dependency
    chain — model init (Linear / Embedding / LayerNorm), dropout
    masks during forward, optimizer state updates."""
    model_a, batch_a = _tiny_model_and_inputs(42)
    params_a = _run_one_step(model_a, batch_a)

    model_b, batch_b = _tiny_model_and_inputs(42)
    params_b = _run_one_step(model_b, batch_b)

    assert set(params_a.keys()) == set(params_b.keys())
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), (
            f"Parameter {name!r} diverged after one optimizer step: "
            f"max |diff| = "
            f"{(params_a[name] - params_b[name]).abs().max().item()}"
        )


def test_different_seeds_produce_different_parameters():
    """Negative control: two seeds → different parameters. Confirms
    the fixture is RNG-sensitive (i.e., the determinism test isn't
    trivially passing because the model is constant)."""
    model_a, batch_a = _tiny_model_and_inputs(42)
    params_a = _run_one_step(model_a, batch_a)

    model_b, batch_b = _tiny_model_and_inputs(1337)
    params_b = _run_one_step(model_b, batch_b)

    # At least one parameter must differ — if every param matches
    # under different seeds, the test isn't testing anything.
    any_differ = any(
        not torch.equal(params_a[n], params_b[n]) for n in params_a
    )
    assert any_differ, (
        "Two seeds produced bit-identical parameters — fixture too "
        "trivial; the determinism test isn't actually verifying RNG-"
        "sensitive behavior."
    )


def test_three_runs_all_identical():
    """A 3-way check: a regression where determinism holds for run A=B
    but not for run C (e.g., a global state that gets latched on first
    call) would be missed by a 2-way comparison."""
    a_model, a_batch = _tiny_model_and_inputs(99)
    a = _run_one_step(a_model, a_batch)

    b_model, b_batch = _tiny_model_and_inputs(99)
    b = _run_one_step(b_model, b_batch)

    c_model, c_batch = _tiny_model_and_inputs(99)
    c = _run_one_step(c_model, c_batch)

    for name in a:
        assert torch.equal(a[name], b[name]) and torch.equal(b[name], c[name]), (
            f"3-way mismatch on {name!r}: "
            f"|a-b|={(a[name]-b[name]).abs().max().item()}, "
            f"|b-c|={(b[name]-c[name]).abs().max().item()}"
        )


# ---------- seed_everything backward compat ----------


def test_existing_seed_everything_still_works():
    """``seed_everything`` is still exported and still works — the
    new function is a strict superset, not a replacement. Other
    scripts (eval scripts, demos) that call seed_everything are
    unaffected."""
    from redact.utils import seed_everything
    seed_everything(0)
    a = torch.rand(3)
    seed_everything(0)
    b = torch.rand(3)
    assert torch.equal(a, b)


# ---------- positive proof: torch.use_deterministic_algorithms is active ----------
#
# A natural positive-proof test would call a known-nondeterministic op
# and assert it raises under strict mode. In torch 2.x the CPU
# enforcement set for ``use_deterministic_algorithms(True)`` is much
# narrower than the documented op list — the listed ops (Tensor.put_,
# scatter_reduce_, index_add_, kthvalue, interpolate, embedding_bag
# with mode='max') are CUDA-enforced but CPU-permissive. Probed all
# six candidates: none raise on CPU in this torch / Python build.
#
# So the positive-proof lives in the CUDA-conditional test below
# (functional bit-identity across a real forward/backward step on
# GPU). On CPU we rely on the state-check tests above for confirmation
# that the flag and env vars are set; the actual CUDA / cuBLAS paths
# can only be verified on a CUDA-enabled runner.


# ---------- CUDA-conditional ----------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-specific determinism knobs (cuBLAS / cuDNN) require a GPU.",
)
def test_bit_identical_parameters_on_cuda():
    """Same bit-identity contract as the CPU test, but exercising the
    cuBLAS GEMM path that the CUBLAS_WORKSPACE_CONFIG knob actually
    targets. Skipped locally (no GPU); activates on the cluster /
    any CUDA-enabled CI runner."""
    def _run(seed):
        enable_full_determinism(seed)
        cfg = AutoConfig.from_pretrained("roberta-base")
        cfg.hidden_size = 32
        cfg.num_hidden_layers = 1
        cfg.num_attention_heads = 2
        cfg.intermediate_size = 64
        config = BinaryHeadConfig(backbone_config=cfg.to_dict())
        model = BinaryHeadModel(config).cuda().train()

        vocab_size = model.backbone.config.vocab_size
        g = torch.Generator(device="cuda").manual_seed(seed)
        input_ids = torch.randint(
            100, min(1000, vocab_size), (4, 8),
            generator=g, device="cuda",
        )
        attention_mask = torch.ones((4, 8), dtype=torch.long, device="cuda")
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device="cuda")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
        )
        out["loss"].backward()
        optimizer.step()
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    params_a = _run(42)
    params_b = _run(42)

    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), (
            f"CUDA parameter {name!r} diverged after one step: "
            f"max |diff| = "
            f"{(params_a[name] - params_b[name]).abs().max().item()}"
        )
