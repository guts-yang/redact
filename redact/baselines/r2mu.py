"""R2MU (Reasoning-aware Representation Misdirection for Unlearning) baseline.

Implements the method from Wang et al. (EMNLP 2025, arXiv:2506.12963),
https://github.com/OPTML-Group/Unlearn-R2MU.

Design decisions (documented divergences from paper / repo):

- **Extent stream: match the repo, not the paper.** The paper describes splitting
  each forget sample into N chunks and interleaving generated CoTs. The repo
  doesn't implement that; it uses a single pre-generated CoT per forget sample,
  truncated to `extent_max_gen_tokens` and prepended with the trigger string
  "Let's reason this step by step.\\n<think>". We follow the repo — it is what
  the authors actually ran and reported.

- **Loss assembly: diverge from the repo to match the paper.** The repo computes
  `reasoning_loss` (β·ℓ_CoT on s1K `<think>thinking</think>answer`) but excludes
  it from the backward graph — a bug given the paper's stated objective. We
  include it. Final backward'd sum:
      unlearn + γ·retain + α·extent + assist_coeff·assistant + β·reasoning
  where γ ≈ `alpha` (repo default 1.4; paper range [1,10]), α implicit = 1 on
  extent, assist_coeff = 1, β = 1 (paper-optimal from Fig. A3).
  `user_loss` (the other repo-computed-but-unused term) is genuinely unused in
  both paper and in practice; we don't compute it.

- **Pad-position masking.** The repo's `forward_with_cache` + `F.mse_loss` averages
  over every position including pads. For R-TOFU forget prompts ("Who is X?"),
  left-padding dominates the 512-token window and the MSE would be computed
  mostly over pad activations. We mask pad positions via the attention mask.
  Caveat: combined with the unit-norm control vector, this concentrates
  per-real-token gradient pull more strongly than upstream's averaging-
  across-pads regime intended; if smoke tests show over-aggression on
  short R-TOFU prompts, reconsider this divergence.

- **Trainable-param selection by named substring** (`mlp.down_proj.weight`),
  not the repo's ordinal `param_ids=[6]`. Same parameter on Llama, safer across
  architectures.

- **Staged backwards** (one `.backward()` per loss term, optimizer step at end)
  instead of a single summed-loss backward. Equivalent gradients, lower peak
  activation memory because each forward graph is released before the next.

- **Safer module resolution.** The repo uses `eval(module_str.format(...))`; we
  index `model.model.layers[layer_id]` directly.

Known limitations (inherited from the repo; documented not fixed):

- **Single control vector per run.** The random target vector is built once
  from `config.seed`; every forget and extent sample misdirects to the same
  vector. Across runs sharing `seed`, the control vector is identical. Matches
  repo; change `seed` for independent instantiations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class R2MUConfig:
    """Hyperparameters for R2MU. Defaults match the repo's `run.sh`."""

    layer_id: int = 7                 # target transformer block for hidden-state probing
    trainable_layer_ids: tuple[int, ...] = (5, 6, 7)
    trainable_param_substring: str = "mlp.down_proj.weight"
    steering_coeff: float = 6.5       # scale of random control vector
    alpha: float = 1.4                # retain-loss weight (paper's γ)
    assist_coeff: float = 1.0         # assistant-stream (s1K answer-only) preservation weight
    beta: float = 1.0                 # reasoning-stream (s1K <think>...</think>answer) preservation weight; paper-optimal
    lr: float = 7.5e-5
    batch_size: int = 4
    max_forget_len: int = 512
    max_retain_len: int = 512
    max_extent_len: int = 512
    max_assistant_len: int = 150      # matches repo's asymmetric cap on reasoning-style inputs
    max_reasoning_len: int = 150      # matches repo
    max_num_batches: int = 500
    seed: int = 42
    extent_trigger_prefix: str = "Let's reason this step by step.\n<think>"
    extent_max_gen_tokens: int = 100  # truncate pre-generated CoT to first N tokens before prepending trigger


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def forward_with_cache(model, inputs, module, no_grad: bool):
    """Run model forward; capture the output of `module` via a forward hook.

    Returns a `[B, T, H]` activation tensor from the single hook call.

    If `no_grad=True`, the forward is wrapped in `torch.no_grad()` and the returned
    activation has no grad_fn (suitable for use as a fixed MSE target).
    """
    cache: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # Transformer blocks return a tuple; the hidden state is output[0].
        cache.append(output[0] if isinstance(output, tuple) else output)

    handle = module.register_forward_hook(hook)
    try:
        if no_grad:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        handle.remove()

    if not cache:
        raise RuntimeError("forward_with_cache: hook did not fire")
    return cache[0]


def masked_mse_loss(activations: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error over non-pad positions only.

    Shapes:
        activations: [B, T, H]
        target:      [B, T, H]  or any shape broadcastable to activations
        attention_mask: [B, T] (1 for real tokens, 0 for pad)

    The hidden-dim mean is computed first; then we average over real tokens using
    the attention mask. If the batch has no real tokens (shouldn't happen), we
    fall back to a denominator of 1 to avoid div-by-zero.
    """
    diff = activations - target                   # broadcasts target against [B, T, H]
    per_token = diff.pow(2).mean(dim=-1)          # [B, T]
    mask = attention_mask.to(per_token.dtype)     # [B, T]
    num = (per_token * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return num / denom


def make_control_vector(hidden_dim: int, steering_coeff: float, device, dtype, seed: int) -> torch.Tensor:
    """Build the random target vector RMU misdirects forget activations toward.

    Shape: [1, 1, H]. Broadcasts against any [B, T, H] activation tensor.
    Matches upstream `unlearn_wmdp.py:209-210`: uniform random vector normalized
    to unit norm, then scaled by steering_coeff, one fixed vector per run.
    The normalization is load-bearing: without it, ||vec|| ≈ steering_coeff *
    sqrt(H/3) ≈ 240 for H=4096, c=6.5 — ~37× the intended magnitude — and the
    unlearn-stage MSE pull on layer-7 activations becomes catastrophic.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    vec = torch.rand((1, 1, hidden_dim), generator=g)
    vec = vec / vec.norm() * steering_coeff
    return vec.to(device=device, dtype=dtype)


def get_trainable_params(model, layer_ids, name_substring: str):
    """Unfreeze only parameters whose (layer-scoped) name contains `name_substring`.

    Freezes every parameter first, then selectively unfreezes. Returns the list of
    unfrozen parameters (for use with an optimizer).
    """
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    trainable: list[torch.nn.Parameter] = []
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        for name, param in layer.named_parameters():
            if name_substring in name:
                param.requires_grad_(True)
                trainable.append(param)

    if not trainable:
        raise ValueError(
            f"No parameters matched substring {name_substring!r} in layers {list(layer_ids)}. "
            f"Check the model's block layout: model.model.layers[i].named_parameters()."
        )
    return trainable


def freeze_model(model):
    """Freeze every parameter in `model` (for the frozen reference copy)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def r2mu_train_step(
    updated_model,
    frozen_model,
    tokenizer,
    batch: dict,
    control_vec: torch.Tensor,
    optimizer,
    config: R2MUConfig,
) -> dict:
    """One R2MU step with staged backwards. Returns a dict of scalar losses for logging.

    `batch` keys (each a list of strings):
        forget    - forget-set inputs (raw text; we tokenize here)
        retain    - retain-set inputs
        extent    - pre-formatted extent strings (trigger + truncated CoT already applied)
        assistant - s1K `deepseek_attempt` (answer-only) for answer-preservation term
        reasoning - s1K `<think>{thinking}</think>{answer}` for CoT-preservation term (paper's β·ℓ_CoT)

    Stages:
        1. forget    -> push activations toward control_vec
        2. retain    -> preserve updated activations against frozen activations (weighted by alpha)
        3. extent    -> push activations toward control_vec (the "unthink" term)
        4. assistant -> preserve updated activations against frozen activations (weighted by assist_coeff)
        5. reasoning -> preserve updated activations against frozen activations (weighted by beta)

    Each stage does forward -> loss -> backward; graphs are released between stages
    to keep peak memory low. Optimizer step happens once at the end.
    """
    updated_module = updated_model.model.layers[config.layer_id]
    frozen_module = frozen_model.model.layers[config.layer_id]
    device = next(updated_model.parameters()).device

    def tok(text_list, max_len):
        return tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

    losses = {}
    optimizer.zero_grad(set_to_none=True)

    def _backward_checked(stage_name: str, loss_tensor: torch.Tensor) -> float:
        """Assert loss is finite before backprop. A NaN/Inf in any stage silently
        corrupts the accumulated gradient across all stages, and the optimizer.step()
        would propagate it everywhere. Fail fast."""
        val = loss_tensor.detach().item()
        if not torch.isfinite(loss_tensor).all().item():
            raise RuntimeError(f"R2MU stage {stage_name!r} produced non-finite loss: {val}")
        loss_tensor.backward()
        return val

    # Stage 1: forget -> random
    inputs = tok(batch["forget"], config.max_forget_len)
    act = forward_with_cache(updated_model, inputs, updated_module, no_grad=False)
    loss = masked_mse_loss(act, control_vec, inputs["attention_mask"])
    losses["unlearn"] = _backward_checked("unlearn", loss)
    del act, loss

    # Stage 2: retain preservation
    inputs = tok(batch["retain"], config.max_retain_len)
    updated_act = forward_with_cache(updated_model, inputs, updated_module, no_grad=False)
    frozen_act = forward_with_cache(frozen_model, inputs, frozen_module, no_grad=True)
    loss = config.alpha * masked_mse_loss(updated_act, frozen_act, inputs["attention_mask"])
    losses["retain"] = _backward_checked("retain", loss)
    del updated_act, frozen_act, loss

    # Stage 3: extent -> random (the "unthink" term)
    inputs = tok(batch["extent"], config.max_extent_len)
    act = forward_with_cache(updated_model, inputs, updated_module, no_grad=False)
    loss = masked_mse_loss(act, control_vec, inputs["attention_mask"])
    losses["extent"] = _backward_checked("extent", loss)
    del act, loss

    # Stage 4: assistant (s1K answer-only) preservation
    inputs = tok(batch["assistant"], config.max_assistant_len)
    updated_act = forward_with_cache(updated_model, inputs, updated_module, no_grad=False)
    frozen_act = forward_with_cache(frozen_model, inputs, frozen_module, no_grad=True)
    loss = config.assist_coeff * masked_mse_loss(updated_act, frozen_act, inputs["attention_mask"])
    losses["assistant"] = _backward_checked("assistant", loss)
    del updated_act, frozen_act, loss

    # Stage 5: reasoning (s1K <think>...</think>answer) preservation — paper's β·ℓ_CoT
    inputs = tok(batch["reasoning"], config.max_reasoning_len)
    updated_act = forward_with_cache(updated_model, inputs, updated_module, no_grad=False)
    frozen_act = forward_with_cache(frozen_model, inputs, frozen_module, no_grad=True)
    loss = config.beta * masked_mse_loss(updated_act, frozen_act, inputs["attention_mask"])
    losses["reasoning"] = _backward_checked("reasoning", loss)
    del updated_act, frozen_act, loss

    optimizer.step()

    losses["total"] = sum(losses.values())
    return losses


def format_extent_string(raw_cot: str, tokenizer, config: R2MUConfig) -> str:
    """Truncate a pre-generated CoT to the first `extent_max_gen_tokens`
    tokens and prepend the trigger. Matches upstream
    `get_dataset_with_generated`.

    Precondition: `tokenizer` must round-trip encode→decode losslessly. On
    transformers ≥5 with LRM-target this requires
    `_maybe_repair_byte_level_tokenizer` (redact/model/model.py); without it,
    encode→decode collapses word-internal spaces and the extent stream
    becomes despaced gibberish.
    """
    token_ids = tokenizer.encode(raw_cot, add_special_tokens=False)[: config.extent_max_gen_tokens]
    truncated = tokenizer.decode(token_ids, skip_special_tokens=True)
    return config.extent_trigger_prefix + truncated
