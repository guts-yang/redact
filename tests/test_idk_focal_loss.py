"""Tests for hard-IDK-negative focal mining in BinaryHeadModel.

Symmetric counterpart to :mod:`tests.test_positive_focal_loss`: weights
``is_idk_negative=True`` rows by ``P(leak).clamp(eps, 1-eps)^gamma`` in
the binary CE reduction. Both branches compose multiplicatively with
the static per-row ``loss_weight`` and the class weights from
:class:`torch.nn.CrossEntropyLoss`.

Covers:

- ``idk_focal_gamma=0`` is a strict no-op (identical loss to the
  pre-feature path).
- ``idk_focal_gamma>0`` upweights ``is_idk_negative=True`` rows with
  high P(leak); non-IDK rows (positives, retain, aux) are untouched.
- ``idk_focal_gamma>0`` without ``is_idk_negative`` in the batch
  raises with a clear message pointing at the materialization step.
- Composes multiplicatively with: ``loss_weight``, class weights, and
  ``positive_focal_gamma`` (the two focal branches are independent and
  multiply per-row).
- The IDK focal weight is detached from autograd — gradient flows
  through CE but not through the gate.
- Label smoothing clips ``P(leak)`` to ``[eps, 1-eps]``.
- Config round-trips through save/load including the new fields.
- An old config without the new fields loads cleanly with defaults
  (focal disabled), so existing on-disk checkpoints still work.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel


# ---------- helpers ----------


def _tiny_config(
    *,
    idk_focal_gamma: float = 0.0,
    idk_focal_label_smoothing: float = 0.0,
    positive_focal_gamma: float = 0.0,
    positive_focal_label_smoothing: float = 0.0,
    binary_class_weights: list[float] | None = None,
):
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    return BinaryHeadConfig(
        backbone_config=cfg.to_dict(),
        idk_focal_gamma=idk_focal_gamma,
        idk_focal_label_smoothing=idk_focal_label_smoothing,
        positive_focal_gamma=positive_focal_gamma,
        positive_focal_label_smoothing=positive_focal_label_smoothing,
        binary_class_weights=binary_class_weights,
    )


def _tiny_model(**cfg_kwargs):
    cfg = _tiny_config(**cfg_kwargs)
    model = BinaryHeadModel(cfg)
    model.eval()  # turn off dropout for determinism
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
    with torch.no_grad():
        backbone_out = model.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = model.dropout(backbone_out.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    return logits


# ---------- config validation ----------


def test_config_rejects_negative_gamma():
    with pytest.raises(ValueError, match="idk_focal_gamma must be >= 0"):
        _tiny_config(idk_focal_gamma=-0.1)


def test_config_rejects_label_smoothing_at_or_above_half():
    with pytest.raises(ValueError, match=r"must lie in \[0, 0.5\)"):
        _tiny_config(idk_focal_label_smoothing=0.5)


def test_config_default_idk_gamma_is_zero():
    """Default is the no-feature state — old checkpoints loaded via
    from_pretrained will see the same default."""
    cfg = _tiny_config()
    assert cfg.idk_focal_gamma == 0.0
    assert cfg.idk_focal_label_smoothing == 0.0


# ---------- forward: gamma=0 identity ----------


def test_idk_gamma_zero_matches_pre_feature_mean_reduction():
    """``idk_focal_gamma=0``, ``positive_focal_gamma=0``, and no
    ``loss_weight`` recovers the plain mean reduction loss. Identity
    of the no-feature default."""
    model = _tiny_model(idk_focal_gamma=0.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    ref = nn.functional.cross_entropy(logits, labels, reduction="mean")
    assert out["loss"].detach().item() == pytest.approx(ref.item(), rel=1e-6)


# ---------- forward: gamma>0 reweighting ----------


def test_idk_focal_weight_matches_manual_formula():
    """gamma=1 with a known is_idk_negative mask matches manual
    ``(per_sample * w * p_leak^gamma).sum() / sum`` reweight on IDK
    rows."""
    model = _tiny_model(idk_focal_gamma=1.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    # Mark rows 0 and 2 as IDK negatives (label=0 rows; row 0 and 2
    # are label=0 in _synthetic_inputs). Rows 1 and 3 are positives —
    # not IDK by construction.
    is_idk = torch.tensor([True, False, True, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    # Manual reweight: IDK rows get p_leak^1, others get 1.
    dyn = torch.where(is_idk, p_leak.clamp(min=1e-6), torch.ones_like(p_leak))
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_idk_focal_does_not_touch_non_idk_rows():
    """When IDK gate is active but all rows have ``is_idk_negative=False``,
    the loss equals the plain mean (no reweight applied). Confirms the
    gate is opt-in per-row, not per-batch."""
    model = _tiny_model(idk_focal_gamma=2.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([False, False, False, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    # All rows have dyn=1.0 → weighted mean = plain mean.
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    ref = per_sample.mean()
    assert out["loss"].detach().item() == pytest.approx(ref.item(), rel=1e-5)


def test_idk_focal_raises_when_is_idk_negative_missing():
    """``idk_focal_gamma>0`` but no ``is_idk_negative`` passed: the
    forward raises with a clear pointer to the materialization step."""
    model = _tiny_model(idk_focal_gamma=1.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    with pytest.raises(ValueError, match=r"is_idk_negative was not"):
        model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels,
        )


def test_idk_focal_shape_mismatch_raises():
    """``is_idk_negative`` of the wrong shape is caught loudly rather
    than silently broadcasting."""
    model = _tiny_model(idk_focal_gamma=1.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    bad_mask = torch.tensor([True, False, True])  # length 3, not 4
    with pytest.raises(ValueError, match=r"shape"):
        model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, is_idk_negative=bad_mask,
        )


# ---------- composition with other branches ----------


def test_idk_focal_composes_with_loss_weight():
    """IDK focal weight composes multiplicatively with the static
    per-row ``loss_weight``."""
    model = _tiny_model(idk_focal_gamma=1.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    static_w = torch.tensor([2.0, 5.0, 0.5, 3.0], dtype=torch.float32)
    is_idk = torch.tensor([True, False, True, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=static_w, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    dyn = torch.where(
        is_idk, p_leak.clamp(min=1e-6) ** 1.5, torch.ones_like(p_leak),
    )
    w = static_w * dyn
    manual = (per_sample * w).sum() / w.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_idk_focal_composes_with_positive_focal():
    """Both focal branches active simultaneously: positives get
    ``(1-p_leak)^gamma_pos``, IDK negatives get ``p_leak^gamma_idk``,
    other rows untouched. Per-row weights multiply."""
    model = _tiny_model(
        positive_focal_gamma=1.0, idk_focal_gamma=1.0,
    )
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    # Row 0 is a label=0 IDK neg; row 2 is a label=0 non-IDK neg
    # (retain or aux); rows 1 and 3 are positives.
    is_idk = torch.tensor([True, False, False, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    # Per-row dyn weights:
    # row 0 (label=0, IDK):     p_leak
    # row 1 (label=1, not IDK): (1 - p_leak)
    # row 2 (label=0, not IDK): 1.0
    # row 3 (label=1, not IDK): (1 - p_leak)
    pos_mask = labels == 1
    dyn = torch.ones_like(p_leak)
    dyn = torch.where(
        pos_mask, (1.0 - p_leak).clamp(min=1e-6), dyn,
    )
    dyn = torch.where(
        is_idk, p_leak.clamp(min=1e-6), dyn,
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


def test_idk_focal_composes_with_class_weights():
    """IDK focal weight composes multiplicatively with CE class
    weights — the binary_class_weights tensor still scales the
    per-row CE before the focal multiplication."""
    class_w = [1.0, 137.0]
    model = _tiny_model(idk_focal_gamma=1.0, binary_class_weights=class_w)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, True, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    weight_tensor = torch.tensor(class_w, dtype=torch.float32)
    per_sample = nn.functional.cross_entropy(
        logits, labels, weight=weight_tensor, reduction="none",
    )
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    dyn = torch.where(
        is_idk, p_leak.clamp(min=1e-6), torch.ones_like(p_leak),
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


# ---------- label smoothing ----------


def test_idk_label_smoothing_clips_focal_gate():
    """With eps=0.1 and gamma=2, an IDK row with p_leak<eps gets dyn
    weight = eps**2 (the lower clip), not p_leak**2."""
    model = _tiny_model(
        idk_focal_gamma=2.0, idk_focal_label_smoothing=0.1,
    )
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, True, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    logits = _replicate_forward(model, input_ids, attention_mask)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1].clamp(min=0.1, max=0.9)
    dyn = torch.where(
        is_idk, p_leak ** 2.0, torch.ones_like(p_leak),
    )
    manual = (per_sample * dyn).sum() / dyn.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item(), rel=1e-5)


# ---------- gradient flow ----------


def test_idk_focal_gate_is_detached_from_autograd():
    """Gradient flows through CE per-sample loss but NOT through the
    IDK focal gate — mirrors the positive-focal detach contract."""
    model_a = _tiny_model(idk_focal_gamma=1.0)
    model_b = _tiny_model(idk_focal_gamma=0.0)
    model_b.load_state_dict(model_a.state_dict())

    input_ids, attention_mask, labels = _synthetic_inputs(model_a)
    is_idk = torch.tensor([True, False, True, False])

    out_a = model_a(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    out_a["loss"].backward()
    grads_a = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()
               if p.grad is not None}

    # model_b: forward without IDK focal gate, manually apply the
    # detached gate on per-sample loss.
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
        dyn = torch.where(
            is_idk, p_leak.clamp(min=1e-6), torch.ones_like(p_leak),
        )
    loss_b = (per_sample * dyn).sum() / dyn.sum()
    loss_b.backward()
    grads_b = {n: p.grad.detach().clone() for n, p in model_b.named_parameters()
               if p.grad is not None}

    assert set(grads_a.keys()) == set(grads_b.keys())
    for n in grads_a:
        assert torch.allclose(grads_a[n], grads_b[n], atol=1e-6, rtol=1e-5), (
            f"Gradient mismatch on {n}: max |diff| = "
            f"{(grads_a[n] - grads_b[n]).abs().max().item()}"
        )


# ---------- save / load round trip ----------


def test_config_roundtrips_through_save_load(tmp_path):
    cfg = _tiny_config(
        idk_focal_gamma=1.5, idk_focal_label_smoothing=0.05,
    )
    model = BinaryHeadModel(cfg)
    model.save_pretrained(str(tmp_path))

    reloaded_cfg = BinaryHeadConfig.from_pretrained(str(tmp_path))
    assert reloaded_cfg.idk_focal_gamma == pytest.approx(1.5)
    assert reloaded_cfg.idk_focal_label_smoothing == pytest.approx(0.05)

    reloaded_model = BinaryHeadModel.from_pretrained(str(tmp_path))
    assert reloaded_model.config.idk_focal_gamma == pytest.approx(1.5)
    assert reloaded_model.config.idk_focal_label_smoothing == pytest.approx(0.05)


def test_old_config_loads_with_default_idk_gamma(tmp_path):
    """A config predating the IDK fields loads cleanly with defaults
    (gamma=0, smoothing=0). Required so existing on-disk checkpoints
    still work."""
    import json
    cfg = _tiny_config()
    model = BinaryHeadModel(cfg)
    model.save_pretrained(str(tmp_path))
    cfg_path = tmp_path / "config.json"
    obj = json.loads(cfg_path.read_text())
    obj.pop("idk_focal_gamma", None)
    obj.pop("idk_focal_label_smoothing", None)
    cfg_path.write_text(json.dumps(obj))

    reloaded_cfg = BinaryHeadConfig.from_pretrained(str(tmp_path))
    assert reloaded_cfg.idk_focal_gamma == 0.0
    assert reloaded_cfg.idk_focal_label_smoothing == 0.0
