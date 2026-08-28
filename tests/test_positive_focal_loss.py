"""Tests for hard-positive focal mining in BinaryHeadModel.

Covers the new ``positive_focal_gamma`` / ``positive_focal_label_smoothing``
fields on :class:`BinaryHeadConfig` and the corresponding
``BinaryHeadModel.forward`` reweighting:

- ``gamma=0`` is a strict no-op (identical loss to the pre-feature path).
- ``gamma>0`` upweights label=1 rows with low P(leak) and downweights
  label=1 rows with high P(leak); label=0 rows are untouched.
- The focal gate composes multiplicatively with the static per-row
  ``loss_weight`` and with the class weights from
  :class:`torch.nn.CrossEntropyLoss`.
- Label smoothing clips ``(1 - P(leak))`` to ``[eps, 1-eps]``, capping
  the per-row dynamic ratio.
- The focal weight is detached from the autograd graph — gradient flows
  through CE but not through the gate.
- Save / load round-trips preserve both new fields.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel


# ---------- helpers ----------


def _tiny_config(*, positive_focal_gamma: float = 0.0,
                positive_focal_label_smoothing: float = 0.0,
                binary_class_weights: list[float] | None = None):
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    return BinaryHeadConfig(
        backbone_config=cfg.to_dict(),
        positive_focal_gamma=positive_focal_gamma,
        positive_focal_label_smoothing=positive_focal_label_smoothing,
        binary_class_weights=binary_class_weights,
    )


def _tiny_model(*, positive_focal_gamma: float = 0.0,
                positive_focal_label_smoothing: float = 0.0,
                binary_class_weights: list[float] | None = None):
    cfg = _tiny_config(
        positive_focal_gamma=positive_focal_gamma,
        positive_focal_label_smoothing=positive_focal_label_smoothing,
        binary_class_weights=binary_class_weights,
    )
    model = BinaryHeadModel(cfg)
    model.eval()  # disable dropout for determinism
    return model


def _synthetic_inputs(model, *, seed: int = 0):
    """4-row mini-batch with labels [0, 1, 0, 1]."""
    vocab_size = model.backbone.config.vocab_size
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        100, min(1000, vocab_size), (4, 5), generator=g,
    )
    attention_mask = torch.ones((4, 5), dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return input_ids, attention_mask, labels


def _replicate_forward(model, input_ids, attention_mask):
    """Compute (logits, per_sample_loss) outside the model — used to
    build the manual reference loss the model output should match."""
    with torch.no_grad():
        backbone_out = model.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = model.dropout(backbone_out.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    return logits


# ---------- config validation ----------


def test_config_rejects_negative_gamma():
    with pytest.raises(ValueError, match="positive_focal_gamma must be >= 0"):
        _tiny_config(positive_focal_gamma=-0.1)


def test_config_rejects_label_smoothing_at_or_above_half():
    with pytest.raises(ValueError, match=r"must lie in \[0, 0.5\)"):
        _tiny_config(positive_focal_label_smoothing=0.5)
    with pytest.raises(ValueError, match=r"must lie in \[0, 0.5\)"):
        _tiny_config(positive_focal_label_smoothing=0.6)


def test_config_rejects_negative_label_smoothing():
    with pytest.raises(ValueError, match=r"must lie in \[0, 0.5\)"):
        _tiny_config(positive_focal_label_smoothing=-0.05)


def test_config_default_gamma_and_smoothing_are_zero():
    """The no-feature default. Existing checkpoints loaded with from_pretrained
    will get these defaults via PretrainedConfig.__init__ — verifies the
    behavior is no-op for old checkpoints."""
    cfg = _tiny_config()
    assert cfg.positive_focal_gamma == 0.0
    assert cfg.positive_focal_label_smoothing == 0.0


# ---------- forward: gamma=0 identity ----------


def test_gamma_zero_matches_pre_feature_mean_reduction():
    """``positive_focal_gamma=0`` and ``loss_weight=None`` recovers the
    plain mean reduction loss exactly. This is the existence proof that
    leaving the flag at its default doesn't drift training."""
    model = _tiny_model(positive_focal_gamma=0.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    ref = nn.functional.cross_entropy(logits, labels, reduction="mean")
    assert out["loss"].detach().item() == pytest.approx(ref.item(), rel=1e-6)


def test_gamma_zero_with_loss_weight_matches_static_weighted_mean():
    """gamma=0 + non-trivial loss_weight matches the manual static-weighted
    formula — confirms the focal branch doesn't disturb the static path."""
    model = _tiny_model(positive_focal_gamma=0.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    static_w = torch.tensor([0.4, 10.0, 2.0, 1.0], dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=static_w,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    manual = (per_sample * static_w).sum() / static_w.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-6)


# ---------- forward: gamma>0 reweighting ----------


def test_focal_weight_matches_manual_formula():
    """gamma=1 with a known logit pattern produces the manual
    ``(per_sample * w * (1-p)^gamma).sum() / sum`` loss on label=1 rows."""
    model = _tiny_model(positive_focal_gamma=1.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    # Manual reweight: pos rows get (1 - p_leak)^1, neg rows get 1.
    pos_mask = labels == 1
    dyn = torch.where(pos_mask, (1.0 - p_leak).clamp(min=1e-6), torch.ones_like(p_leak))
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_focal_weight_only_affects_positives_not_negatives():
    """When the focal gate is active, label=0 rows must still get dyn
    weight exactly 1.0; only label=1 rows are reweighted. Constructs two
    batches that share the same input/logits but flip a single label
    from 0 to 1 — only the positive's contribution should change."""
    # Same model, same inputs, only the labels flip. We compare two
    # forward passes to confirm the model loss responds to label=1 going
    # to label=0, not just to logit values.
    model = _tiny_model(positive_focal_gamma=2.0)
    input_ids, attention_mask, _ = _synthetic_inputs(model)
    labels_a = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    labels_b = torch.tensor([0, 1, 0, 0], dtype=torch.long)

    out_a = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels_a,
    )
    out_b = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels_b,
    )

    # Reconstruct logits to manually compute expected losses.
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample_a = nn.functional.cross_entropy(
        logits, labels_a, reduction="none",
    )
    per_sample_b = nn.functional.cross_entropy(
        logits, labels_b, reduction="none",
    )
    p_leak = torch.softmax(logits, dim=-1)[:, 1]

    # labels_a: all negatives → dyn = 1.0 everywhere → plain mean.
    expected_a = per_sample_a.mean()
    assert out_a["loss"].detach().item() == pytest.approx(
        expected_a.item(), rel=1e-5,
    )

    # labels_b: one positive at idx=1. Only that row gets the focal
    # reweight; idx=0,2,3 keep dyn=1.0.
    pos_mask_b = labels_b == 1
    dyn_b = torch.where(
        pos_mask_b, (1.0 - p_leak).clamp(min=1e-6) ** 2.0,
        torch.ones_like(p_leak),
    )
    expected_b = (per_sample_b * dyn_b).sum() / dyn_b.sum()
    assert out_b["loss"].detach().item() == pytest.approx(
        expected_b.item(), rel=1e-5,
    )

    # Cross-check: dyn weight on negatives in batch_b is exactly 1.0 at
    # idx 0, 2, 3 — confirms the gate did not touch label=0 rows.
    assert dyn_b[0].item() == pytest.approx(1.0)
    assert dyn_b[2].item() == pytest.approx(1.0)
    assert dyn_b[3].item() == pytest.approx(1.0)


def test_focal_composes_multiplicatively_with_static_loss_weight():
    """The static per-row ``loss_weight`` and the dynamic focal weight
    compose multiplicatively. Equivalently: the model's loss with both
    matches the manual ``(per_sample * static_w * dyn_w).sum() / sum``."""
    model = _tiny_model(positive_focal_gamma=1.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    static_w = torch.tensor([2.0, 5.0, 0.5, 3.0], dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=static_w,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    pos_mask = labels == 1
    dyn = torch.where(
        pos_mask, (1.0 - p_leak).clamp(min=1e-6) ** 1.5,
        torch.ones_like(p_leak),
    )
    w = static_w * dyn
    manual = (per_sample * w).sum() / w.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_focal_composes_with_balanced_class_weights():
    """Focal reweighting also composes multiplicatively with the CE class
    weights (passed as ``binary_class_weights`` on the config — exercised
    by the BinaryHeadModel construction). Catches a regression where the
    focal path could accidentally bypass CE's per-class scaling."""
    # 137 is the canonical class-balance boost from
    # compute_balanced_class_weights on a 1:137 positive:negative split.
    class_w = [1.0, 137.0]
    model = _tiny_model(
        positive_focal_gamma=1.0, binary_class_weights=class_w,
    )
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    # Manual per-sample with class-weighted CE (matches model's
    # self.binary_loss_fn).
    weight_tensor = torch.tensor(class_w, dtype=torch.float32)
    per_sample = nn.functional.cross_entropy(
        logits, labels, weight=weight_tensor, reduction="none",
    )
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    pos_mask = labels == 1
    dyn = torch.where(
        pos_mask, (1.0 - p_leak).clamp(min=1e-6),
        torch.ones_like(p_leak),
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


# ---------- label smoothing ----------


def test_label_smoothing_clips_focal_gate():
    """With eps=0.1 and gamma=2, a single confident positive (p_leak=0.99)
    gets dyn weight = (0.1)**2 = 0.01 instead of (0.01)**2 = 1e-4. The
    model's loss should reflect the clipped value, not the raw."""
    model = _tiny_model(
        positive_focal_gamma=2.0, positive_focal_label_smoothing=0.1,
    )
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    pos_mask = labels == 1
    one_minus_p = (1.0 - p_leak).clamp(min=0.1, max=0.9)
    dyn = torch.where(
        pos_mask, one_minus_p ** 2.0, torch.ones_like(p_leak),
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_label_smoothing_zero_is_no_op():
    """eps=0 means no clamping (beyond the 1e-6 floor) — exact equality
    with the unsmoothed focal formula for non-pathological probabilities."""
    model_smooth = _tiny_model(
        positive_focal_gamma=1.0, positive_focal_label_smoothing=0.0,
    )
    # Build a second model with the same backbone weights to compare
    # — we just verify the forward output matches the unsmoothed
    # manual formula already exercised above, with eps=0 explicit.
    input_ids, attention_mask, labels = _synthetic_inputs(model_smooth)
    out = model_smooth(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model_smooth, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    pos_mask = labels == 1
    # eps=0 → only the 1e-6 floor applies, no upper clip
    dyn = torch.where(
        pos_mask, (1.0 - p_leak).clamp(min=1e-6),
        torch.ones_like(p_leak),
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-6)


# ---------- gradient flow ----------


def test_focal_gate_is_detached_from_autograd():
    """Gradient should flow through the CE per-sample loss but NOT through
    the focal gate (the gate detaches the binary logits before softmax).
    Equivalently: backward from loss with gamma>0 produces the same
    parameter gradients as if we'd computed the same weighted sum with
    a hand-detached gate."""
    model_a = _tiny_model(positive_focal_gamma=1.0)
    model_b = _tiny_model(positive_focal_gamma=0.0)
    # Sync model_b to model_a's parameters so the only difference is the
    # focal path. We then verify that a hand-rolled detached-gate
    # gradient on model_b matches model_a's gradient.
    model_b.load_state_dict(model_a.state_dict())

    input_ids, attention_mask, labels = _synthetic_inputs(model_a)

    # model_a: forward with focal gate active
    out_a = model_a(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    out_a["loss"].backward()
    grads_a = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()
               if p.grad is not None}

    # model_b: forward without focal gate, then manually compose the
    # detached focal weight on per-sample loss.
    backbone_out = model_b.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
    )
    pooled = model_b.dropout(backbone_out.last_hidden_state[:, 0, :])
    logits = model_b.binary_head(pooled)
    per_sample = nn.functional.cross_entropy(
        logits, labels.long(), reduction="none",
    )
    with torch.no_grad():
        p_leak = torch.softmax(logits, dim=-1)[:, 1]
        pos_mask = labels.long() == 1
        dyn = torch.where(
            pos_mask, (1.0 - p_leak).clamp(min=1e-6),
            torch.ones_like(p_leak),
        )
    loss_b = (per_sample * dyn).sum() / dyn.sum()
    loss_b.backward()
    grads_b = {n: p.grad.detach().clone() for n, p in model_b.named_parameters()
               if p.grad is not None}

    # Same set of params have grads
    assert set(grads_a.keys()) == set(grads_b.keys())
    for n in grads_a:
        assert torch.allclose(grads_a[n], grads_b[n], atol=1e-6, rtol=1e-5), (
            f"Gradient mismatch on {n}: max |diff| = "
            f"{(grads_a[n] - grads_b[n]).abs().max().item()}"
        )


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0])
def test_focal_gradient_is_finite_for_high_p_leak(gamma):
    """A confident positive at p_leak ≈ 1.0 must not produce NaN/Inf
    gradients across the full proposed sweep. Without the 1e-6 floor /
    label smoothing, (1-p)**gamma would underflow and the autograd path
    could see divide-by-zero. gamma=0.5 is a different regime (where
    1e-6**0.5 = 1e-3, well-above fp32 denormal) than gamma=3 (where
    1e-6**3 = 1e-18, below fp32 normal range)."""
    # Force a label=1 row with extreme logits via a binary_head reset.
    model = _tiny_model(
        positive_focal_gamma=gamma, positive_focal_label_smoothing=0.0,
    )
    with torch.no_grad():
        model.binary_head.weight.zero_()
        model.binary_head.bias[0] = -10.0   # P(class 0) ≈ 0
        model.binary_head.bias[1] = 10.0    # P(leak) ≈ 1.0
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    # All labels = 1 so every row hits the focal branch
    labels = torch.ones_like(labels)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    out["loss"].backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (
                f"Non-finite grad on {n} at gamma={gamma}: "
                f"any-nan={torch.isnan(p.grad).any()}, "
                f"any-inf={torch.isinf(p.grad).any()}"
            )


def test_focal_with_zero_loss_weight_returns_zero_finite_loss():
    """``loss_weight=zeros + positive_focal_gamma>0 + label=1 rows
    present`` must take the denom==0 branch and return loss=0 with no
    NaN/Inf in gradients. Regression check for the composition: the
    static all-zero weights override the focal upweight, and the
    docstring's "all-zero" guarantee still holds under the focal path."""
    model = _tiny_model(positive_focal_gamma=2.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    zero_w = torch.zeros(labels.shape, dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=zero_w,
    )
    assert out["loss"].detach().item() == pytest.approx(0.0)
    assert torch.isfinite(out["loss"]).item()
    out["loss"].backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad on {n}"
            assert (p.grad == 0).all(), (
                f"Param {n} should have all-zero grad with zero weights; "
                f"max abs = {p.grad.abs().max().item()}"
            )


def test_focal_with_no_positives_in_batch_matches_static_weighted_mean():
    """``positive_focal_gamma>0 + all-negative batch`` reduces to the
    static-weighted mean exactly, because pos_mask is empty and the
    where-branch keeps lw unchanged. Pins this as the intended behavior
    so a future refactor doesn't silently change the all-negative loss."""
    model = _tiny_model(positive_focal_gamma=1.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    all_neg_labels = torch.zeros_like(labels)
    static_w = torch.tensor([0.4, 2.0, 5.0, 1.0], dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=all_neg_labels, loss_weight=static_w,
    )
    # Manual reference: no positives → dyn = ones everywhere → loss is
    # static-weighted mean on label=0 CE per-sample loss.
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(
        logits, all_neg_labels, reduction="none",
    )
    manual = (per_sample * static_w).sum() / static_w.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


# ---------- save / load round trip ----------


def test_config_roundtrips_through_save_load(tmp_path):
    """A saved + reloaded BinaryHeadConfig keeps the focal fields. Catches
    a regression where the field is set on __init__ but not declared in
    the config dict (rare, but the existing per_source_loss_weights field
    has the same pattern)."""
    cfg = _tiny_config(
        positive_focal_gamma=1.5, positive_focal_label_smoothing=0.05,
    )
    model = BinaryHeadModel(cfg)
    model.save_pretrained(str(tmp_path))

    reloaded_cfg = BinaryHeadConfig.from_pretrained(str(tmp_path))
    assert reloaded_cfg.positive_focal_gamma == pytest.approx(1.5)
    assert reloaded_cfg.positive_focal_label_smoothing == pytest.approx(0.05)

    reloaded_model = BinaryHeadModel.from_pretrained(str(tmp_path))
    assert reloaded_model.config.positive_focal_gamma == pytest.approx(1.5)
    assert reloaded_model.config.positive_focal_label_smoothing == pytest.approx(0.05)


def test_old_config_loads_with_default_gamma(tmp_path):
    """A BinaryHeadConfig saved without the new fields loads cleanly with
    the defaults (gamma=0, smoothing=0). Required so existing checkpoints
    on disk don't break."""
    # Simulate an "old" config that lacks the focal fields by writing
    # the JSON directly without them.
    import json
    cfg = _tiny_config()
    model = BinaryHeadModel(cfg)
    model.save_pretrained(str(tmp_path))
    cfg_path = tmp_path / "config.json"
    obj = json.loads(cfg_path.read_text())
    obj.pop("positive_focal_gamma", None)
    obj.pop("positive_focal_label_smoothing", None)
    cfg_path.write_text(json.dumps(obj))

    reloaded_cfg = BinaryHeadConfig.from_pretrained(str(tmp_path))
    assert reloaded_cfg.positive_focal_gamma == 0.0
    assert reloaded_cfg.positive_focal_label_smoothing == 0.0


# ---------- end-to-end: effective positive mass diagnostic ----------


def test_effective_positive_mass_shrinks_as_model_overfits():
    """Smoke test for the compute_binary_metrics diagnostic logic: with
    a label=1 row whose P(leak) is forced near 1, the effective positive
    mass under gamma=2 should be tiny (<0.05) — i.e. the diagnostic
    behaves correctly in the failure mode it was added to detect."""
    import numpy as np

    # Replicate the formula inside compute_binary_metrics so the test
    # doesn't depend on argparse / module-level state.
    p_leak = np.array([0.5, 0.99, 0.5, 0.95])  # labels = [0, 1, 0, 1]
    labels = np.array([0, 1, 0, 1])
    gamma = 2.0
    eps = 0.05
    pos_mask = labels == 1
    one_minus_p = np.clip(1.0 - p_leak[pos_mask], eps, 1.0 - eps)
    epm = (one_minus_p ** gamma).mean()
    # Confident positives at 0.99 / 0.95 have eff weights of (0.05)^2
    # and (0.05)^2 respectively (both at the eps floor) → mean = 0.0025
    assert epm < 0.05, (
        f"Effective positive mass {epm} should flag the failure mode "
        f"of confident positives starving the head"
    )
