"""Tests for the pairwise hinge gap loss in BinaryHeadModel.

The gap loss is a margin-maximization term between logit-space leak
scores on positive rows and IDK-negative rows. The classifier's
deployment metric is ``min(score|pos) > max(score|idk)`` with a
margin — this loss optimizes that quantity directly in the same
logit space the inference operating point lives in.

Covers:

- Known-value computation matches the spec.
- A satisfied margin returns zero (no spurious loss).
- A batch with zero positives OR zero IDKs degenerates to a 0-d
  zero tensor (graceful no-op, NOT ``None``, NOT a Python ``0``).
- Gradient flows on positive and IDK rows only — non-IDK negatives
  and rows outside both populations receive zero gradient on their
  logit row.
- ``gradcheck`` confirms no in-place operations break the graph
  (the spec requires ``F.relu`` rather than ``.relu_()``).
- Integration with the binary head: ``gap_loss_weight > 0`` adds
  the term to the total loss; the stashed ``_last_gap_loss``
  attribute on the model holds the raw (unweighted) value.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from redact.regen.binary_head_model import (
    BinaryHeadConfig,
    BinaryHeadModel,
    compute_gap_loss,
)


# ---------- compute_gap_loss: known values ----------


def test_known_value_4row():
    """Hand-computed: logits [[0,5],[0,4],[0,-3],[0,-2]],
    labels [1,1,0,0], is_idk [F,F,T,T], margin=4.

    scores = [5, 4, -3, -2]
    pos_scores = [5, 4]
    idk_scores = [-3, -2]
    pairwise diffs (N_pos × N_idk):
      [[5 - (-3), 5 - (-2)], [4 - (-3), 4 - (-2)]] = [[8, 7], [7, 6]]
    margin - diff = [[4-8, 4-7], [4-7, 4-6]] = [[-4, -3], [-3, -2]]
    relu = [[0, 0], [0, 0]]
    mean = 0.0 (the margin is satisfied for every pair).
    """
    logits = torch.tensor(
        [[0.0, 5.0], [0.0, 4.0], [0.0, -3.0], [0.0, -2.0]],
    )
    labels = torch.tensor([1, 1, 0, 0])
    is_idk = torch.tensor([False, False, True, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_known_value_with_violation():
    """Tighten the margin to expose a known violation:
    margin=10 on the same setup above.

    diffs = [[8, 7], [7, 6]]
    margin - diff = [[2, 3], [3, 4]]
    relu = [[2, 3], [3, 4]]
    mean = (2 + 3 + 3 + 4) / 4 = 3.0
    """
    logits = torch.tensor(
        [[0.0, 5.0], [0.0, 4.0], [0.0, -3.0], [0.0, -2.0]],
    )
    labels = torch.tensor([1, 1, 0, 0])
    is_idk = torch.tensor([False, False, True, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=10.0)
    assert loss.item() == pytest.approx(3.0, rel=1e-6)


def test_known_value_partial_violation():
    """Single-pair violation: pos score 1, idk score 0, margin=4.
    diff = 1 - 0 = 1; relu(4 - 1) = 3. Mean = 3.0."""
    logits = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([1, 0])
    is_idk = torch.tensor([False, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert loss.item() == pytest.approx(3.0, rel=1e-6)


# ---------- edge cases: empty pools ----------


def test_no_positives_returns_zero_d_tensor():
    """All-negatives batch: returns a 0-d zero tensor (NOT None,
    NOT a Python int)."""
    logits = torch.tensor([[0.0, -1.0], [0.0, -2.0]])
    labels = torch.tensor([0, 0])
    is_idk = torch.tensor([True, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_no_idks_returns_zero_d_tensor():
    """All-non-IDK batch: returns a 0-d zero tensor."""
    logits = torch.tensor([[0.0, 5.0], [0.0, -1.0]])
    labels = torch.tensor([1, 0])
    is_idk = torch.tensor([False, False])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_empty_batch_returns_zero_d_tensor():
    """B=0 batch is a no-op."""
    logits = torch.zeros((0, 2))
    labels = torch.zeros(0, dtype=torch.long)
    is_idk = torch.zeros(0, dtype=torch.bool)
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_no_positives_zero_loss_keeps_dtype_and_device():
    """The 0-d fallback inherits ``logits.dtype`` / ``logits.device``
    so downstream ``total_loss = ce_loss + weight * gap_loss`` is
    type-stable. (``new_zeros(())`` semantics check.)"""
    logits = torch.tensor([[0.0, -1.0]], dtype=torch.float64)
    labels = torch.tensor([0])
    is_idk = torch.tensor([True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert loss.dtype == torch.float64
    assert loss.device == logits.device


# ---------- gradient flow ----------


def test_gradient_flows_only_through_pos_and_idk_rows():
    """Only positive and IDK rows have non-zero gradient on their
    logit row. Non-IDK negatives (e.g. retain/aux) are not referenced
    by the gap loss and receive zero gradient."""
    logits = torch.tensor(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 0.5], [0.0, -0.5]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 0, 0, 1])
    # Row 0 is a positive, row 1 is an IDK negative, row 2 is a
    # non-IDK negative (retain/aux), row 3 is a positive.
    is_idk = torch.tensor([False, True, False, False])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    loss.backward()
    grad = logits.grad
    assert grad is not None
    # Row 2 is a non-IDK negative — it never enters the diff tensor,
    # so its gradient is exactly zero.
    assert torch.all(grad[2] == 0.0)
    # Rows 0, 1, 3 do enter the diff (rows 0 and 3 are positives,
    # row 1 is the only IDK). With margin=4 and these scores
    # (1, 2, 0.5, -0.5) every pair violates the margin so all three
    # rows pull gradient.
    assert not torch.all(grad[0] == 0.0)
    assert not torch.all(grad[1] == 0.0)
    assert not torch.all(grad[3] == 0.0)


def test_gradient_flows_through_logits_only_via_score_diff():
    """The score is ``logits[:, 1] - logits[:, 0]``, so a tiny
    perturbation to either column changes the loss in opposite
    directions on a positive row and in the same direction on an
    IDK row (the gradient is anti-symmetric across columns)."""
    logits = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0]], requires_grad=True,
    )
    labels = torch.tensor([1, 0])
    is_idk = torch.tensor([False, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    loss.backward()
    g = logits.grad
    # Row 0 is positive: increasing column 1 should decrease the
    # hinge (score_pos goes up → diff goes up → relu(4 - diff)
    # decreases). So d(loss)/d(logits[0, 1]) < 0.
    assert g[0, 1] < 0
    # Symmetric: increasing column 0 of the positive raises the
    # complement, lowering the score, raising the hinge.
    assert g[0, 0] > 0
    # Row 1 is IDK: increasing column 1 raises score_idk, lowering
    # diff, raising the hinge → d(loss)/d(logits[1, 1]) > 0.
    assert g[1, 1] > 0
    assert g[1, 0] < 0


def test_gradient_zero_when_margin_satisfied():
    """When every pair's margin is satisfied, the hinge is exactly
    zero and the gradient is exactly zero (no spurious push)."""
    logits = torch.tensor(
        [[0.0, 10.0], [0.0, -10.0]], requires_grad=True,
    )
    labels = torch.tensor([1, 0])
    is_idk = torch.tensor([False, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    loss.backward()
    assert torch.all(logits.grad == 0.0)


# ---------- autograd integrity ----------


def test_gradcheck_double_precision():
    """Numerical-vs-analytical gradient agreement on a small example.
    fp64 because gradcheck needs the precision; the loss must be
    out-of-place only for this to pass (any in-place op on a
    tensor the graph still references would NaN / corrupt the
    backward)."""
    torch.manual_seed(0)
    logits = torch.randn((4, 2), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 0, 1, 0])
    is_idk = torch.tensor([False, True, False, True])

    def _fn(x):
        return compute_gap_loss(x, labels, is_idk, margin=4.0)

    assert torch.autograd.gradcheck(
        _fn, (logits,), eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_loss_is_zero_d_tensor():
    """The loss must be a 0-d tensor so ``loss.backward()`` works
    when added to a scalar CE loss downstream."""
    logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    is_idk = torch.tensor([False, True])
    loss = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


# ---------- mask dtype coercion contract ----------


def test_mask_accepts_float_truthy_tensor():
    """``compute_gap_loss`` calls ``.bool()`` on the mask, so a
    float ``0.0`` / ``1.0`` mask must coerce correctly (0.0 → False,
    1.0 → True). Locks the defensive-coerce contract — the forward
    callsite also casts to ``torch.bool`` before invoking the free
    function, but external callers (tests, other training scripts)
    rely on this coercion."""
    logits = torch.tensor(
        [[0.0, 5.0], [0.0, 4.0], [0.0, -3.0], [0.0, -2.0]],
    )
    labels = torch.tensor([1, 1, 0, 0])
    is_idk_bool = torch.tensor([False, False, True, True])
    is_idk_float = torch.tensor([0.0, 0.0, 1.0, 1.0])
    is_idk_int8 = torch.tensor([0, 0, 1, 1], dtype=torch.int8)

    ref = compute_gap_loss(logits, labels, is_idk_bool, margin=10.0)
    assert compute_gap_loss(
        logits, labels, is_idk_float, margin=10.0,
    ).item() == pytest.approx(ref.item(), rel=1e-7)
    assert compute_gap_loss(
        logits, labels, is_idk_int8, margin=10.0,
    ).item() == pytest.approx(ref.item(), rel=1e-7)


def test_mask_treats_any_nonzero_as_true():
    """``.bool()`` coerces any non-zero value to ``True``, including
    floats like 0.5 and 2.5. Caller is responsible for passing a
    proper indicator mask — the function silently treats truthy
    entries as IDK negatives."""
    logits = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([1, 0])
    is_idk_truthy = torch.tensor([0.0, 0.5])  # 0.5 is truthy
    ref = compute_gap_loss(
        logits, labels, torch.tensor([False, True]), margin=4.0,
    )
    assert compute_gap_loss(
        logits, labels, is_idk_truthy, margin=4.0,
    ).item() == pytest.approx(ref.item(), rel=1e-7)


# ---------- BinaryHeadModel integration ----------


def _tiny_config(*, gap_loss_weight: float = 0.0, gap_margin: float = 4.0):
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    return BinaryHeadConfig(
        backbone_config=cfg.to_dict(),
        gap_loss_weight=gap_loss_weight,
        gap_margin=gap_margin,
    )


def _tiny_model(**kwargs):
    cfg = _tiny_config(**kwargs)
    m = BinaryHeadModel(cfg)
    m.eval()
    return m


def _synthetic_inputs(model, *, seed: int = 0):
    vocab = model.backbone.config.vocab_size
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(100, min(1000, vocab), (4, 5), generator=g)
    attention_mask = torch.ones((4, 5), dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return input_ids, attention_mask, labels


def test_config_rejects_negative_weight():
    with pytest.raises(ValueError, match="gap_loss_weight must be >= 0"):
        _tiny_config(gap_loss_weight=-0.1)


def test_config_rejects_negative_margin():
    with pytest.raises(ValueError, match="gap_margin must be >= 0"):
        _tiny_config(gap_margin=-1.0)


def test_default_disables_gap_loss():
    """Default config (``gap_loss_weight=0``) is a strict no-op;
    forward does not require ``is_idk_negative``."""
    model = _tiny_model(gap_loss_weight=0.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    # is_idk_negative omitted, no error.
    assert out["loss"] is not None
    assert model._last_gap_loss is None


def test_forward_adds_gap_term_with_weight():
    """When ``gap_loss_weight > 0``, the total loss is
    ``CE + weight * gap_loss`` and the model stashes the raw
    gap-loss value as a detached 0-d tensor on ``_last_gap_loss``
    (not a Python float — keeps the per-step ``.item()`` sync off
    the GPU hot path)."""
    model = _tiny_model(gap_loss_weight=0.5, gap_margin=4.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, False, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    # Replicate forward to get the same logits (deterministic — dropout
    # is off because model.eval()).
    with torch.no_grad():
        bb = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = model.dropout(bb.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="mean")
    raw_gap = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    expected_total = per_sample + 0.5 * raw_gap
    assert out["loss"].detach().item() == pytest.approx(
        expected_total.item(), rel=1e-5,
    )
    assert isinstance(model._last_gap_loss, torch.Tensor)
    assert model._last_gap_loss.ndim == 0
    assert not model._last_gap_loss.requires_grad
    assert model._last_gap_loss.item() == pytest.approx(
        raw_gap.item(), rel=1e-5,
    )


def test_forward_raises_when_is_idk_negative_missing():
    """``gap_loss_weight > 0`` but no ``is_idk_negative`` passed:
    forward raises with a clear pointer to the materialization step."""
    model = _tiny_model(gap_loss_weight=0.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    with pytest.raises(ValueError, match=r"is_idk_negative was not"):
        model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels,
        )


def test_forward_raises_on_shape_mismatch():
    model = _tiny_model(gap_loss_weight=0.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    bad_mask = torch.tensor([True, False, True])  # length 3, not 4
    with pytest.raises(ValueError, match=r"shape"):
        model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, is_idk_negative=bad_mask,
        )


def test_forward_gap_term_active_on_cheap_mean_branch():
    """The cheap-mean branch (no loss_weight, no focal) takes a
    separate code path from the weighted-mean branch. Confirm the
    gap-loss term is still added when both focal gammas are 0 and
    loss_weight is None — i.e. the gap-loss path is independent of
    which CE reduction branch ran."""
    model = _tiny_model(gap_loss_weight=0.5, gap_margin=4.0)
    assert model.positive_focal_gamma == 0.0
    assert model.idk_focal_gamma == 0.0
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, False, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    # Without gap-loss, this batch would reduce via the cheap
    # ``per_sample.mean()`` path. Adding the gap term to that path
    # should still yield ``CE_mean + 0.5 * raw_gap``.
    with torch.no_grad():
        bb = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = model.dropout(bb.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    ce_mean = nn.functional.cross_entropy(logits, labels, reduction="mean")
    raw_gap = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    expected_total = ce_mean + 0.5 * raw_gap
    assert out["loss"].detach().item() == pytest.approx(
        expected_total.item(), rel=1e-5,
    )


def test_forward_gap_term_composes_with_idk_focal():
    """Gap-loss and IDK focal mining are independent additive
    branches in the forward. With both active, total loss =
    weighted_CE_with_focal + weight * gap_loss."""
    model = _tiny_model(gap_loss_weight=0.3, gap_margin=4.0)
    # Manually flip on IDK focal to mimic both branches active.
    model.idk_focal_gamma = 1.0
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, True, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    with torch.no_grad():
        bb = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = model.dropout(bb.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    # Compute focal-weighted CE the same way the model does.
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    p_leak = torch.softmax(logits, dim=-1)[:, 1]
    dyn = torch.where(
        is_idk, p_leak.clamp(min=1e-6), torch.ones_like(p_leak),
    )
    ce_weighted = (per_sample * dyn).sum() / dyn.sum()
    raw_gap = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    expected_total = ce_weighted + 0.3 * raw_gap
    assert out["loss"].detach().item() == pytest.approx(
        expected_total.item(), rel=1e-5,
    )


def test_forward_gap_term_composes_with_loss_weight():
    """Gap-loss and per-row ``loss_weight`` (from
    ``--loss_weights_per_source``) compose additively: total loss =
    weighted-mean CE + gap_loss_weight * gap_loss. The gap term is
    computed on raw logits — independent of the per-row CE
    weighting — but the gradient still flows through both terms
    into the same backbone parameters."""
    model = _tiny_model(gap_loss_weight=0.5, gap_margin=4.0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, True, False])
    # Non-trivial per-row weight (mimics --loss_weights_per_source
    # producing different masses per source). The float dtype
    # matches what compute_loss_weight_column emits.
    loss_weight = torch.tensor([2.0, 5.0, 0.5, 3.0], dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=loss_weight, is_idk_negative=is_idk,
    )
    with torch.no_grad():
        bb = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = model.dropout(bb.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
    # Per-row weighted-mean CE — the same reduction the model takes
    # when loss_weight is passed and both focal gammas are 0.
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    ce_weighted = (per_sample * loss_weight).sum() / loss_weight.sum()
    raw_gap = compute_gap_loss(logits, labels, is_idk, margin=4.0)
    expected_total = ce_weighted + 0.5 * raw_gap
    assert out["loss"].detach().item() == pytest.approx(
        expected_total.item(), rel=1e-5,
    )


def test_forward_gap_term_loss_weight_gradient_flows_both_terms():
    """Backward through total = weighted_CE + w * gap_loss should
    populate gradient on the backbone parameters from BOTH terms
    (a regression test against accidentally detaching one branch).
    """
    model = _tiny_model(gap_loss_weight=1.0, gap_margin=20.0)
    # train() so gradients flow through dropout; but seed so the
    # mask is deterministic enough for the magnitude check below.
    model.train()
    torch.manual_seed(0)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, True, False])
    loss_weight = torch.tensor([2.0, 5.0, 0.5, 3.0], dtype=torch.float32)

    # Take a reference grad with gap-loss DISABLED (CE-only) so we
    # can confirm the gap-loss branch contributes additional
    # gradient mass on top of the CE term, not just replaces it.
    head_w_ce_only = None
    model.gap_loss_weight = 0.0
    model.zero_grad()
    torch.manual_seed(1)
    out_ce_only = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=loss_weight,
        is_idk_negative=is_idk,
    )
    out_ce_only["loss"].backward()
    head_w_ce_only = model.binary_head.weight.grad.detach().clone()

    # Now with gap-loss active (large margin so the hinge is
    # active for every pair).
    model.gap_loss_weight = 1.0
    model.zero_grad()
    torch.manual_seed(1)  # same forward draw for comparability
    out_with_gap = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=loss_weight,
        is_idk_negative=is_idk,
    )
    out_with_gap["loss"].backward()
    head_w_with_gap = model.binary_head.weight.grad.detach().clone()

    # Gap-loss-on gradient should differ from CE-only by the
    # gap-loss term's contribution. Magnitudes are not equal.
    assert not torch.allclose(head_w_ce_only, head_w_with_gap, atol=1e-7), (
        "binary_head grad with gap_loss active matches CE-only; "
        "gap-loss may be silently detached or zeroed."
    )
    # Both gradients must be finite (no NaN/Inf from interaction).
    assert torch.isfinite(head_w_ce_only).all()
    assert torch.isfinite(head_w_with_gap).all()


def test_forward_does_not_return_gap_loss_key():
    """The forward output dict stays at ``{loss, logits}`` so HF
    Trainer's ``prediction_step`` (which unpacks dict outputs into
    a logits tuple) doesn't get confused. The gap-loss value is
    available via the ``_last_gap_loss`` stash instead."""
    model = _tiny_model(gap_loss_weight=0.5)
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    is_idk = torch.tensor([True, False, False, False])
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, is_idk_negative=is_idk,
    )
    assert "gap_loss" not in out
    assert "loss" in out
    assert "logits" in out


def test_config_round_trips():
    """Save/load round-trip preserves the new fields."""
    import tempfile, os
    cfg = _tiny_config(gap_loss_weight=0.3, gap_margin=5.5)
    with tempfile.TemporaryDirectory() as tmp:
        cfg.save_pretrained(tmp)
        loaded = BinaryHeadConfig.from_pretrained(tmp)
    assert loaded.gap_loss_weight == 0.3
    assert loaded.gap_margin == 5.5


def test_old_config_loads_with_defaults(tmp_path):
    """A config dict missing the new fields loads cleanly with
    the no-feature defaults (forward compat for pre-flag
    checkpoints)."""
    cfg = _tiny_config()
    # Wipe the new fields from the persisted dict to simulate a
    # pre-flag checkpoint.
    d = cfg.to_dict()
    d.pop("gap_loss_weight", None)
    d.pop("gap_margin", None)
    loaded = BinaryHeadConfig(**d)
    assert loaded.gap_loss_weight == 0.0
    assert loaded.gap_margin == 4.0
