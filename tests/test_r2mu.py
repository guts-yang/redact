"""Tests for R2MU baseline implementation.

Usage:
    conda run -n redact python -m pytest tests/test_r2mu.py -v

Uses a tiny LlamaForCausalLM (hidden=64, 4 layers) as a fixture so every test
exercises the real block structure (model.model.layers[i]) without the cost of
loading an 8B model. The module paths and parameter names we filter on are the
same as for DeepSeek-R1-Distill-Llama-8B.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from redact.baselines.r2mu import (
    R2MUConfig,
    forward_with_cache,
    format_extent_string,
    freeze_model,
    get_trainable_params,
    make_control_vector,
    masked_mse_loss,
    r2mu_train_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_llama_config(hidden_size: int = 64, num_layers: int = 4, vocab: int = 200) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=vocab - 1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    config = _tiny_llama_config()
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_frozen_model() -> LlamaForCausalLM:
    config = _tiny_llama_config()
    torch.manual_seed(0)  # same seed → same weights as tiny_model at init
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class DummyTokenizer:
    """Minimal tokenizer mimic for unit tests. Encodes characters to codepoints,
    truncated to config.vocab_size - 1 to avoid the pad id and clamp range."""

    def __init__(self, vocab_size: int = 200, pad_id: int = 199):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.pad_token_id = pad_id

    def _encode_str(self, s: str) -> list[int]:
        # Character ordinals, clamped into [0, vocab_size-2] to steer clear of pad_id.
        return [min(ord(c), self.vocab_size - 2) for c in s][:32]  # cap len for tests

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return self._encode_str(s)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Reverse of _encode_str (lossy, but faithful enough for substring tests).
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_str(t) for t in texts]
        if max_length is not None and truncation:
            encoded = [seq[:max_length] for seq in encoded]
        target_len = max(len(seq) for seq in encoded) if padding else max(len(seq) for seq in encoded)
        ids = torch.full((len(encoded), target_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(encoded), target_len), dtype=torch.long)
        for i, seq in enumerate(encoded):
            # Left-padding (matches our unlearn.py setting) — put real tokens at the END.
            start = target_len - len(seq)
            ids[i, start:] = torch.tensor(seq, dtype=torch.long)
            attn[i, start:] = 1

        class _Batch(dict):
            def to(self, device):
                return _Batch({k: v.to(device) for k, v in self.items()})

        return _Batch({"input_ids": ids, "attention_mask": attn})


@pytest.fixture(scope="module")
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


# ---------------------------------------------------------------------------
# make_control_vector
# ---------------------------------------------------------------------------

def test_make_control_vector_shape_and_dtype():
    vec = make_control_vector(hidden_dim=128, steering_coeff=1.0, device="cpu", dtype=torch.float32, seed=0)
    assert vec.shape == (1, 1, 128)
    assert vec.dtype == torch.float32


def test_make_control_vector_norm_equals_steering_coeff():
    """Critical: the random vector must be normalized to unit length BEFORE scaling
    by steering_coeff (matches upstream unlearn_wmdp.py:210). Without normalization,
    ||vec|| ≈ coeff*sqrt(H/3) ≈ 240 for H=4096 c=6.5 — ~37x the intended magnitude —
    which over-drives the unlearn-stage MSE and collapses the model. Regression test."""
    for hidden_dim in [128, 256, 4096]:
        for coeff in [1.0, 6.5, 20.0]:
            vec = make_control_vector(hidden_dim, coeff, "cpu", torch.float32, seed=0)
            norm = vec.norm().item()
            assert abs(norm - coeff) < 1e-4, (
                f"||vec||={norm:.4f} for H={hidden_dim} coeff={coeff}; expected ≈{coeff}"
            )


def test_make_control_vector_deterministic():
    v1 = make_control_vector(128, 1.0, "cpu", torch.float32, seed=42)
    v2 = make_control_vector(128, 1.0, "cpu", torch.float32, seed=42)
    v3 = make_control_vector(128, 1.0, "cpu", torch.float32, seed=43)
    assert torch.equal(v1, v2)
    assert not torch.equal(v1, v3)


# ---------------------------------------------------------------------------
# masked_mse_loss
# ---------------------------------------------------------------------------

def test_masked_mse_loss_all_ones_matches_plain_mse():
    torch.manual_seed(0)
    acts = torch.randn(2, 5, 8)
    tgt = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.long)
    got = masked_mse_loss(acts, tgt, mask)
    expected = (acts - tgt).pow(2).mean(dim=-1).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_mse_loss_excludes_padding():
    # Design: two sequences, one with trailing pads. Verify the loss is computed
    # only over the unpadded positions — concretely, add noise at pad positions
    # and show the loss doesn't change.
    torch.manual_seed(0)
    acts_clean = torch.zeros(1, 4, 8)
    tgt = torch.zeros(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    acts_noisy = acts_clean.clone()
    acts_noisy[0, 2, :] = 100.0  # enormous activation at a pad position
    acts_noisy[0, 3, :] = -100.0

    loss_clean = masked_mse_loss(acts_clean, tgt, mask)
    loss_noisy = masked_mse_loss(acts_noisy, tgt, mask)
    assert torch.allclose(loss_clean, loss_noisy), "Padding positions must not affect the loss"


def test_masked_mse_loss_broadcasts_target():
    # Target shape [1, 1, H] should broadcast over batch and seq dims.
    acts = torch.ones(3, 4, 8) * 2.0
    target_vec = torch.ones(1, 1, 8) * 5.0
    mask = torch.ones(3, 4, dtype=torch.long)
    got = masked_mse_loss(acts, target_vec, mask)
    # (2 - 5)^2 = 9; mean over hidden is 9; mean over tokens is 9.
    assert torch.allclose(got, torch.tensor(9.0), atol=1e-6)


def test_masked_mse_loss_zero_mask_safeguard():
    # Defensive: all-zero mask must not div-by-zero. The returned value is
    # effectively meaningless, but shouldn't crash.
    acts = torch.randn(1, 3, 8)
    tgt = torch.randn(1, 3, 8)
    mask = torch.zeros(1, 3, dtype=torch.long)
    got = masked_mse_loss(acts, tgt, mask)
    assert torch.isfinite(got)


# ---------------------------------------------------------------------------
# forward_with_cache
# ---------------------------------------------------------------------------

def test_forward_with_cache_captures_layer_output(tiny_model, dummy_tokenizer):
    inputs = dummy_tokenizer(["hello world", "short"])
    module = tiny_model.model.layers[2]
    act = forward_with_cache(tiny_model, inputs, module, no_grad=True)
    # Llama block returns (hidden [B, T, H], ...) tuple; hook takes [0].
    assert act.shape[0] == inputs["input_ids"].shape[0]
    assert act.shape[1] == inputs["input_ids"].shape[1]
    assert act.shape[2] == tiny_model.config.hidden_size


def test_forward_with_cache_no_grad_path(tiny_model, dummy_tokenizer):
    inputs = dummy_tokenizer(["abc"])
    module = tiny_model.model.layers[0]
    act = forward_with_cache(tiny_model, inputs, module, no_grad=True)
    assert not act.requires_grad


def test_forward_with_cache_grad_path(tiny_model, dummy_tokenizer):
    # For grad to flow, some param must require_grad. Default for eval mode is still
    # requires_grad=True unless explicitly disabled.
    for p in tiny_model.parameters():
        p.requires_grad_(True)
    inputs = dummy_tokenizer(["abc"])
    module = tiny_model.model.layers[0]
    act = forward_with_cache(tiny_model, inputs, module, no_grad=False)
    assert act.requires_grad


def test_forward_with_cache_removes_hook(tiny_model, dummy_tokenizer):
    before = len(tiny_model.model.layers[0]._forward_hooks)
    inputs = dummy_tokenizer(["abc"])
    _ = forward_with_cache(tiny_model, inputs, tiny_model.model.layers[0], no_grad=True)
    after = len(tiny_model.model.layers[0]._forward_hooks)
    assert before == after, "Hook handle should be removed by forward_with_cache"


# ---------------------------------------------------------------------------
# get_trainable_params / freeze_model
# ---------------------------------------------------------------------------

def test_get_trainable_params_selects_only_matches(tiny_model):
    params = get_trainable_params(tiny_model, layer_ids=[1, 2], name_substring="mlp.down_proj.weight")
    # Exactly one match per layer in Llama MLP: layers[i].mlp.down_proj.weight.
    assert len(params) == 2
    for p in params:
        assert p.requires_grad


def test_get_trainable_params_freezes_everything_else(tiny_model):
    _ = get_trainable_params(tiny_model, layer_ids=[0, 1], name_substring="mlp.down_proj.weight")
    # Parameters in layer 3 (not selected) must be frozen.
    for _, param in tiny_model.model.layers[3].named_parameters():
        assert not param.requires_grad, "Non-selected layer params must be frozen"
    # Embedding params must be frozen.
    assert not tiny_model.model.embed_tokens.weight.requires_grad


def test_get_trainable_params_raises_on_no_match(tiny_model):
    with pytest.raises(ValueError):
        get_trainable_params(tiny_model, layer_ids=[0], name_substring="nonexistent.weight")


def test_freeze_model_freezes_all(tiny_frozen_model):
    # Un-freeze first, then freeze, verify.
    for p in tiny_frozen_model.parameters():
        p.requires_grad_(True)
    freeze_model(tiny_frozen_model)
    assert all(not p.requires_grad for p in tiny_frozen_model.parameters())
    assert not tiny_frozen_model.training


# ---------------------------------------------------------------------------
# format_extent_string
# ---------------------------------------------------------------------------

def test_format_extent_string_truncates_and_prepends(dummy_tokenizer):
    """`extent_max_gen_tokens` is a token-count cap; the dummy tokenizer encodes
    1 char → 1 id, so a 50-char input truncated to 4 tokens produces 4 chars
    after decode."""
    config = R2MUConfig(extent_max_gen_tokens=4)
    raw = "a" * 50
    out = format_extent_string(raw, dummy_tokenizer, config)
    assert out.startswith(config.extent_trigger_prefix)
    truncated_portion = out[len(config.extent_trigger_prefix):]
    assert len(truncated_portion) == 4, (
        f"truncated to {len(truncated_portion)} chars; expected 4 tokens (1 char/token in dummy)"
    )


def test_format_extent_string_short_input_preserved(dummy_tokenizer):
    config = R2MUConfig(extent_max_gen_tokens=100)
    raw = "hi"
    out = format_extent_string(raw, dummy_tokenizer, config)
    assert out == config.extent_trigger_prefix + "hi"


# ---------------------------------------------------------------------------
# End-to-end regression with the real LRM-target tokenizer.
# Network-dependent: downloads tokenizer files (~1 MB) on first run, then cached.
# ---------------------------------------------------------------------------

def test_extent_string_survives_lrm_target_tokenization():
    """Regression for the original R2MU collapse bug. `format_extent_string`
    does encode→truncate→decode internally; with the byte-level tokenizer
    repair applied, the roundtrip is lossless for word-internal spaces
    (`'step by step'` stays intact, not collapsed to `'stepbystep'`). Use a
    `max_gen_tokens` cap larger than the input's token count so the test
    exercises the full string verbatim — pre-repair this would have
    silently mutated the output."""
    from transformers import AutoTokenizer

    from redact.model.model import _maybe_repair_byte_level_tokenizer

    tok = AutoTokenizer.from_pretrained("sangyon/LRM-target", trust_remote_code=True)
    _maybe_repair_byte_level_tokenizer(tok)

    config = R2MUConfig(extent_max_gen_tokens=200)  # well over raw's token count
    raw = "Let's reason about this step by step. The model will think carefully."
    out = format_extent_string(raw, tok, config)
    assert out == config.extent_trigger_prefix + raw, (
        f"format_extent_string mutated the text: "
        f"expected trigger+{raw!r}, got {out!r}"
    )


def test_format_rtofu_sample_preserves_chat_tokens_through_tokenizer():
    """Regression: the chat tokens <｜User｜> and <｜Assistant｜> use halfwidth-pipe
    codepoints (U+FF5C). A tokenizer that doesn't carry them as known tokens
    would split them into byte pieces and the model would never see the chat
    framing. With the byte-level repair, the full sample encode→decode is
    exact, so we assert full equality (special-token recognition + word-
    internal whitespace both)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r2mu"))
    from unlearn import format_rtofu_sample
    from transformers import AutoTokenizer

    from redact.model.model import _maybe_repair_byte_level_tokenizer

    tok = AutoTokenizer.from_pretrained("sangyon/LRM-target", trust_remote_code=True)
    _maybe_repair_byte_level_tokenizer(tok)

    text = format_rtofu_sample(
        question="Who is X?",
        answer="A famous person.",
        prompt_prefix="<｜User｜>",
        prompt_suffix="<｜Assistant｜>",
        answer_prefix="",
        eos_token=tok.eos_token or "",
    )
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text, (
        f"encode→decode mutated the sample: {text!r} -> {decoded!r}"
    )


# ---------------------------------------------------------------------------
# r2mu_train_step (end-to-end on tiny model)
# ---------------------------------------------------------------------------

def test_r2mu_train_step_returns_all_keys(dummy_tokenizer):
    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1, 2),
        batch_size=2,
        max_forget_len=8,
        max_retain_len=8,
        max_extent_len=8,
        max_assistant_len=8,
        max_reasoning_len=8,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)

    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)

    batch = {
        "forget": ["forget sample one", "forget sample two"],
        "retain": ["retain one", "retain two"],
        "extent": ["extent one", "extent two"],
        "assistant": ["assistant one", "assistant two"],
        "reasoning": ["reasoning one", "reasoning two"],
    }

    losses = r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)

    assert set(losses.keys()) == {"unlearn", "retain", "extent", "assistant", "reasoning", "total"}
    for k, v in losses.items():
        assert isinstance(v, float)
        assert v == v  # not NaN
        assert v >= 0  # all MSE-based losses are non-negative (including weighted)


def test_r2mu_train_step_advances_trainable_params(dummy_tokenizer):
    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1,),
        batch_size=2,
        max_forget_len=8,
        max_retain_len=8,
        max_extent_len=8,
        max_assistant_len=8,
        max_reasoning_len=8,
        lr=1e-2,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)

    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    before = [p.detach().clone() for p in trainable]
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
    batch = {
        "forget":    ["forget " * 3] * 2,
        "retain":    ["retain " * 3] * 2,
        "extent":    ["extent " * 3] * 2,
        "assistant": ["assist " * 3] * 2,
        "reasoning": ["reason " * 3] * 2,
    }

    r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)

    after = [p.detach().clone() for p in trainable]
    for b, a in zip(before, after):
        assert not torch.equal(b, a), "Trainable params should change after an optimizer step"


def test_r2mu_train_step_leaves_frozen_params_untouched(dummy_tokenizer):
    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1,),
        batch_size=2,
        max_forget_len=8,
        max_retain_len=8,
        max_extent_len=8,
        max_assistant_len=8,
        max_reasoning_len=8,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)

    # Snapshot a frozen-model param.
    frozen_snap = frozen.model.layers[1].mlp.down_proj.weight.detach().clone()

    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
    batch = {k: ["xx"] * 2 for k in ("forget", "retain", "extent", "assistant", "reasoning")}

    r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)

    assert torch.equal(frozen_snap, frozen.model.layers[1].mlp.down_proj.weight)


def test_r2mu_train_step_total_equals_sum_of_components(dummy_tokenizer):
    """Guards against a future bug where someone adds a weight to one term but
    forgets to propagate it into the sum. The logged `total` must equal the
    arithmetic sum of the other five logged values."""
    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1,),
        batch_size=2,
        max_forget_len=8,
        max_retain_len=8,
        max_extent_len=8,
        max_assistant_len=8,
        max_reasoning_len=8,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)

    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
    batch = {k: ["xx yy zz"] * 2 for k in ("forget", "retain", "extent", "assistant", "reasoning")}

    losses = r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)

    expected_total = sum(losses[k] for k in ("unlearn", "retain", "extent", "assistant", "reasoning"))
    assert abs(losses["total"] - expected_total) < 1e-5


def test_r2mu_train_step_accumulates_gradients_across_stages(dummy_tokenizer):
    """Gradients must accumulate (not overwrite) across the five staged backwards.
    Verify by comparing the post-step update against what a single-stage run would
    produce: if gradients were being overwritten, only the last stage's gradient
    would affect the update, and the total parameter delta would equal the
    reasoning-stage-only delta. Here we ensure it does NOT."""
    def run_step(active_stages):
        torch.manual_seed(0)
        updated = LlamaForCausalLM(_tiny_llama_config())
        torch.manual_seed(0)
        frozen = LlamaForCausalLM(_tiny_llama_config())
        freeze_model(frozen)
        config = R2MUConfig(
            layer_id=1,
            trainable_layer_ids=(1,),
            batch_size=2,
            max_forget_len=8, max_retain_len=8, max_extent_len=8,
            max_assistant_len=8, max_reasoning_len=8,
            lr=1e-2,
        )
        trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
        optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
        # For "dropped" stages, substitute a trivially-matching input so loss≈0.
        # (That's the closest we can get to "skip" without changing the implementation.)
        same = ["x"] * 2
        batch = {k: (same if k not in active_stages else ["xx yy zz"] * 2)
                 for k in ("forget", "retain", "extent", "assistant", "reasoning")}
        r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)
        return updated.model.layers[1].mlp.down_proj.weight.detach().clone()

    w_full = run_step({"forget", "retain", "extent", "assistant", "reasoning"})
    w_reasoning_only = run_step({"reasoning"})
    # If gradients were overwritten instead of accumulated, both weights would equal
    # the "reasoning only" version. The full step must diverge.
    assert not torch.allclose(w_full, w_reasoning_only)


def test_r2mu_train_step_raises_on_non_finite_loss(dummy_tokenizer, monkeypatch):
    """Inject a NaN in masked_mse_loss and verify the step raises rather than
    silently corrupting the accumulated gradient."""
    import redact.baselines.r2mu as r2mu_mod

    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1,),
        batch_size=2,
        max_forget_len=8, max_retain_len=8, max_extent_len=8,
        max_assistant_len=8, max_reasoning_len=8,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)
    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
    batch = {k: ["xx"] * 2 for k in ("forget", "retain", "extent", "assistant", "reasoning")}

    # Monkeypatch to return NaN from masked_mse_loss.
    def _bad_mse(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(r2mu_mod, "masked_mse_loss", _bad_mse)
    with pytest.raises(RuntimeError, match="non-finite"):
        r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)


def test_forward_with_cache_removes_hook_on_exception(tiny_model):
    """If the forward pass throws, the hook must still be removed (try/finally)."""
    before = len(tiny_model.model.layers[0]._forward_hooks)

    # Pass an out-of-range token ID (vocab_size=200, id=9999 → IndexError in embedding).
    bad = {
        "input_ids": torch.tensor([[9999]], dtype=torch.long),
        "attention_mask": torch.ones((1, 1), dtype=torch.long),
    }

    with pytest.raises(Exception):
        forward_with_cache(tiny_model, bad, tiny_model.model.layers[0], no_grad=True)

    after = len(tiny_model.model.layers[0]._forward_hooks)
    assert before == after, "Hook must be removed even when forward raises"


def test_fixture_frozen_matches_updated_at_init():
    """Sanity: `tiny_model` and `tiny_frozen_model` fixtures share the same seed,
    so their weights must be identical at init. Future regressions that change
    the seeding would silently invalidate every test that compares activations
    against a 'frozen reference'."""
    torch.manual_seed(0)
    m1 = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    m2 = LlamaForCausalLM(_tiny_llama_config())
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Param {n1} differs at init — tests relying on matched init will be wrong"


def test_r2mu_train_step_leaves_non_trainable_updated_params_untouched(dummy_tokenizer):
    """Layer 3 wasn't in trainable_layer_ids — its weights must not move."""
    config = R2MUConfig(
        layer_id=1,
        trainable_layer_ids=(1,),
        batch_size=2,
        max_forget_len=8,
        max_retain_len=8,
        max_extent_len=8,
        max_assistant_len=8,
        max_reasoning_len=8,
    )
    torch.manual_seed(0)
    updated = LlamaForCausalLM(_tiny_llama_config())
    torch.manual_seed(0)
    frozen = LlamaForCausalLM(_tiny_llama_config())
    freeze_model(frozen)

    non_trainable_snap = updated.model.layers[3].mlp.down_proj.weight.detach().clone()

    trainable = get_trainable_params(updated, config.trainable_layer_ids, config.trainable_param_substring)
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    control_vec = make_control_vector(updated.config.hidden_size, config.steering_coeff, "cpu", torch.float32, seed=0)
    batch = {k: ["xx"] * 2 for k in ("forget", "retain", "extent", "assistant", "reasoning")}

    r2mu_train_step(updated, frozen, dummy_tokenizer, batch, control_vec, optimizer, config)

    assert torch.equal(non_trainable_snap, updated.model.layers[3].mlp.down_proj.weight)
