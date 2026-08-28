"""Tests for per-row, per-source loss weighting in the binary leak classifier.

Covers Proposal B from the binary-classifier research pass:

- ``parse_per_source_loss_weights`` parses the comma-separated spec,
  resolves the ``"target_fraction_capped"`` auto-token, validates user
  inputs, and returns a per-source ``{name: weight}`` dict covering
  every source in the training data.
- ``BinaryHeadModel.forward`` accepts a per-row ``loss_weight`` kwarg
  and computes ``(per_sample_loss * loss_weight).sum() /
  loss_weight.sum()``. ``loss_weight=None`` recovers the prior
  ``reduction='mean'`` behavior unchanged.
- ``DataCollatorWithPadding`` preserves a ``loss_weight`` column from
  a HuggingFace dataset to the collated batch as a float tensor of
  shape ``(batch,)`` — no custom collator needed.
- Gradient through ``loss_weight`` matches the manual weighted-mean
  formula (catches detached / .item() bugs).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from redact.regen.binary_head_model import BinaryHeadConfig, BinaryHeadModel
from redact.training import (
    TARGET_FRACTION_CAPPED_TOKEN,
    compute_loss_weight_column,
    parse_per_source_loss_weights,
)


# ---------- parser: comma-separated NAME=VALUE ----------


def _all_negative(available):
    """Helper: labels list of all-zeros for tests that don't care about the
    label gate (parser-validation tests, simple spec tests)."""
    return [0] * len(available)


def test_parser_basic_spec_returns_expected_dict():
    """A simple two-source spec resolves to those weights, with all other
    sources defaulting to 1.0.
    """
    available = ["clean_idk", "clean_aux", "clean_retain", "clean_idk"]
    weights = parse_per_source_loss_weights(
        "clean_idk=10.0,clean_aux=0.4", available, _all_negative(available)
    )
    assert weights["clean_idk"] == pytest.approx(10.0)
    assert weights["clean_aux"] == pytest.approx(0.4)
    # Unlisted source defaults to 1.0
    assert weights["clean_retain"] == pytest.approx(1.0)
    # Covers every unique source
    assert set(weights.keys()) == {"clean_idk", "clean_aux", "clean_retain"}


def test_parser_none_and_empty_string_return_uniform():
    """No spec (None) and empty string both yield uniform weights."""
    available = ["a", "b", "a", "c"]
    expected_keys = {"a", "b", "c"}
    for spec in (None, "", "   "):
        weights = parse_per_source_loss_weights(
            spec, available, _all_negative(available)
        )
        assert set(weights.keys()) == expected_keys
        for w in weights.values():
            assert w == pytest.approx(1.0)


def test_parser_unknown_source_raises_value_error():
    """An unknown source name in the spec must raise."""
    available = ["clean_aux", "clean_retain"]
    with pytest.raises(ValueError, match="not found"):
        parse_per_source_loss_weights(
            "clean_xyzzy=2.0", available, _all_negative(available)
        )


def test_parser_missing_equals_raises():
    """A pair without an ``=`` is malformed."""
    available = ["clean_aux"]
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_per_source_loss_weights(
            "clean_aux 0.4", available, _all_negative(available)
        )


def test_parser_non_numeric_value_raises():
    """A non-numeric value raises before validating sources."""
    available = ["clean_aux"]
    with pytest.raises(ValueError, match="not a number"):
        parse_per_source_loss_weights(
            "clean_aux=not_a_float", available, _all_negative(available)
        )


def test_parser_negative_value_raises():
    """A negative weight is rejected (would flip the loss sign)."""
    available = ["clean_aux"]
    with pytest.raises(ValueError, match="non-negative"):
        parse_per_source_loss_weights(
            "clean_aux=-0.1", available, _all_negative(available)
        )


def test_parser_non_finite_value_raises():
    """NaN / inf are rejected."""
    available = ["clean_aux"]
    with pytest.raises(ValueError, match="finite"):
        parse_per_source_loss_weights(
            "clean_aux=nan", available, _all_negative(available)
        )
    with pytest.raises(ValueError, match="finite"):
        parse_per_source_loss_weights(
            "clean_aux=inf", available, _all_negative(available)
        )


def test_parser_empty_name_raises():
    """An empty source name (e.g., ``=2.0``) raises explicitly."""
    available = ["clean_aux"]
    with pytest.raises(ValueError, match="empty source name"):
        parse_per_source_loss_weights("=2.0", available, _all_negative(available))


def test_parser_handles_whitespace_in_pairs():
    """Whitespace around names / values is stripped."""
    available = ["clean_aux", "clean_retain"]
    weights = parse_per_source_loss_weights(
        "  clean_aux = 0.4 , clean_retain = 2.07 ", available, _all_negative(available)
    )
    assert weights["clean_aux"] == pytest.approx(0.4)
    assert weights["clean_retain"] == pytest.approx(2.07)


def test_parser_handles_trailing_comma_and_empty_segments():
    """``a=1.0,,`` is parsed identically to ``a=1.0``."""
    available = ["clean_aux", "clean_retain"]
    weights = parse_per_source_loss_weights(
        "clean_aux=0.4,,", available, _all_negative(available)
    )
    assert weights["clean_aux"] == pytest.approx(0.4)
    assert weights["clean_retain"] == pytest.approx(1.0)


def test_parser_length_mismatch_raises():
    """sources and labels must have equal length — parallel per-row arrays."""
    with pytest.raises(ValueError, match="must equal"):
        parse_per_source_loss_weights(
            "clean_aux=0.4", ["clean_aux", "clean_retain"], [0]
        )


def test_parser_new_schema_primary_keys():
    """Post-(source, source_tags)-refactor: the parser key space is the
    4 primary source values (``forget``, ``retain``, ``idk``, ``aux``)
    rather than the old underscored strings (``clean_forget``, etc.).
    The parser itself is source-string-agnostic; this test pins the
    expected behavior under the new key space."""
    available = [
        "forget", "forget", "retain", "retain", "idk", "idk", "aux", "aux",
    ]
    labels = _all_negative(available)
    weights = parse_per_source_loss_weights(
        "idk=10.0,aux=0.4", available, labels,
    )
    assert weights == {
        "forget": pytest.approx(1.0),
        "retain": pytest.approx(1.0),
        "idk": pytest.approx(10.0),
        "aux": pytest.approx(0.4),
    }


# ---------- parser: target_fraction_capped ----------


def test_target_fraction_capped_three_balanced_sources():
    """Three equally-sized negative sources → each gets the same weight.

    With three buckets each of size N, the target-fraction (1/3) per
    source × total negatives (3N) gives N per source ⇒ weight = 1.0.
    """
    counts = {"clean_retain": 100, "clean_aux": 100, "clean_idk": 100}
    available = [s for s, n in counts.items() for _ in range(n)]
    labels = _all_negative(available)
    weights = parse_per_source_loss_weights(
        TARGET_FRACTION_CAPPED_TOKEN, available, labels
    )
    for src in counts:
        assert weights[src] == pytest.approx(1.0)


def test_target_fraction_capped_forget01_numbers():
    """Forget01 negative-pool numbers under the label-gated semantics.

    With clean_forget label=0 now counted as a negative source:
      - clean_forget pre-leak = 44 rows  → raw ~33.3, capped at 10.0
      - clean_retain = 701 rows           → (4401/3)/701 ≈ 2.09
      - clean_idk    = 35 rows            → raw ~41.9, capped at 10.0
      - clean_aux    = 3621 rows          → (4401/3)/3621 ≈ 0.405
    N_neg total = 4401 (44 + 701 + 35 + 3621)
    """
    counts_neg = {
        "clean_forget": 44,    # pre-leak label=0 rows
        "clean_retain": 701,
        "clean_idk": 35,
        "clean_aux": 3621,
    }
    # Add 32 clean_forget label=1 rows (first-leak); they participate in
    # the source list but NOT in the target_fraction_capped negative pool.
    available = (
        ["clean_forget"] * counts_neg["clean_forget"]
        + ["clean_forget"] * 32
        + ["clean_retain"] * counts_neg["clean_retain"]
        + ["clean_idk"] * counts_neg["clean_idk"]
        + ["clean_aux"] * counts_neg["clean_aux"]
    )
    labels = (
        [0] * counts_neg["clean_forget"]
        + [1] * 32
        + [0] * counts_neg["clean_retain"]
        + [0] * counts_neg["clean_idk"]
        + [0] * counts_neg["clean_aux"]
    )
    weights = parse_per_source_loss_weights(
        TARGET_FRACTION_CAPPED_TOKEN, available, labels
    )
    n_neg = sum(counts_neg.values())
    # clean_forget label=0: raw ~33, capped to 10
    raw_forget = (n_neg / 3) / counts_neg["clean_forget"]
    assert raw_forget > 10.0, "fixture assumes clean_forget label=0 crosses the cap"
    assert weights["clean_forget"] == pytest.approx(10.0)
    # retain ≈ 2.09
    expected_retain = (n_neg / 3) / counts_neg["clean_retain"]
    assert weights["clean_retain"] == pytest.approx(expected_retain, rel=1e-4)
    # aux ≈ 0.405
    expected_aux = (n_neg / 3) / counts_neg["clean_aux"]
    assert weights["clean_aux"] == pytest.approx(expected_aux, rel=1e-4)
    # idk raw ~41.9, capped to 10
    raw_idk = (n_neg / 3) / counts_neg["clean_idk"]
    assert raw_idk > 10.0
    assert weights["clean_idk"] == pytest.approx(10.0)


def test_target_fraction_capped_cap_value():
    """A tiny rare source gets the cap exactly, not the unbounded raw weight."""
    # 1 rare row vs. 1000 common rows; raw weight on rare = (1001/3)/1 ≈ 333
    available = ["clean_idk"] + ["clean_aux"] * 1000
    labels = _all_negative(available)
    weights = parse_per_source_loss_weights(
        TARGET_FRACTION_CAPPED_TOKEN, available, labels
    )
    assert weights["clean_idk"] == pytest.approx(10.0)


def test_target_fraction_capped_label1_rows_excluded_from_counts():
    """Label=1 rows of any source do NOT count toward the negative pool.

    Replaces the prior source-prefix-based positives-unchanged test. The
    label is now the gate: a source can have both label=0 and label=1
    rows (clean_forget), and only the label=0 rows count toward the
    target-fraction-capped denominator.
    """
    # 50 clean_forget rows: 30 label=0 (pre-leak), 20 label=1 (first-leak)
    # 50 clean_retain rows: all label=0
    # Total label=0 negatives: 30 + 50 = 80
    # target mass per source = 80/3 = 26.67
    available = ["clean_forget"] * 50 + ["clean_retain"] * 50
    labels = [0] * 30 + [1] * 20 + [0] * 50
    weights = parse_per_source_loss_weights(
        TARGET_FRACTION_CAPPED_TOKEN, available, labels
    )
    # clean_forget: (80/3)/30 ≈ 0.889
    assert weights["clean_forget"] == pytest.approx((80 / 3) / 30, rel=1e-4)
    # clean_retain: (80/3)/50 ≈ 0.533
    assert weights["clean_retain"] == pytest.approx((80 / 3) / 50, rel=1e-4)


def test_target_fraction_capped_no_label0_rows_raises():
    """All-label=1 source list under the auto-default raises (no negatives)."""
    available = ["clean_forget"] * 50 + ["corrupted_x"] * 30
    labels = [1] * 80  # all positive — no label=0 rows
    with pytest.raises(ValueError, match="at least one row with label == 0"):
        parse_per_source_loss_weights(
            TARGET_FRACTION_CAPPED_TOKEN, available, labels
        )


# ---------- compute_loss_weight_column: row-level label-1 override ----------


def test_compute_loss_weight_column_label1_override():
    """Label=1 rows get weight 1.0 regardless of source spec; label=0 rows
    use the per-source weight (defaulting to 1.0 if source unlisted).

    Closes the gap where the prior in-script closure read the wrong
    column name (``label`` instead of post-rename ``labels``) and was
    never directly tested. The behavior is now in a hoisted free
    function so any future regression here is caught immediately.
    """
    sources = ["clean_forget", "clean_forget", "clean_aux", "clean_retain", "clean_idk"]
    labels = [0, 1, 0, 0, 0]
    weights_map = {
        "clean_forget": 10.0,
        "clean_aux": 0.4,
        "clean_retain": 2.07,
        "clean_idk": 5.0,
    }
    result = compute_loss_weight_column(sources, labels, weights_map)
    assert result == [10.0, 1.0, 0.4, 2.07, 5.0]


def test_compute_loss_weight_column_unlisted_source_defaults_to_1():
    """A label=0 row whose source is not in the weights map gets 1.0."""
    result = compute_loss_weight_column(
        ["clean_unknown"], [0], {"clean_aux": 0.4}
    )
    assert result == [1.0]


def test_compute_loss_weight_column_empty_input():
    """Empty inputs return an empty list — no errors."""
    assert compute_loss_weight_column([], [], {"clean_aux": 0.4}) == []


def test_compute_loss_weight_column_length_mismatch_raises():
    """Parallel arrays must have equal length."""
    with pytest.raises(ValueError, match="must equal"):
        compute_loss_weight_column(["a", "b"], [0], {})


def test_target_fraction_capped_source_with_only_label1_excluded():
    """A source that has only label=1 rows is not in the negative pool —
    it gets weight 1.0 (the unlisted default) because n_neg_for_source=0."""
    # clean_forget: all label=1 (no pre-leak rows in this fixture)
    # clean_retain: all label=0
    available = ["clean_forget"] * 20 + ["clean_retain"] * 100
    labels = [1] * 20 + [0] * 100
    weights = parse_per_source_loss_weights(
        TARGET_FRACTION_CAPPED_TOKEN, available, labels
    )
    # clean_retain is the only negative source; with only 1 neg source,
    # target_fraction (1/3) × n_neg (100) / n_retain (100) = 1/3.
    assert weights["clean_retain"] == pytest.approx(1.0 / 3.0, rel=1e-4)
    # clean_forget defaults to 1.0 (no label=0 rows → no per-source weight)
    assert weights["clean_forget"] == pytest.approx(1.0)


# ---------- BinaryHeadModel.forward: loss reduction with loss_weight ----------


def _tiny_binary_model():
    """Build a tiny BinaryHeadModel with no class weights, on CPU.

    Used by the forward-pass loss tests below — keeps the safetensors
    round-trip moot and the synthetic batch sizes manageable.
    """
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    config = BinaryHeadConfig(backbone_config=cfg.to_dict())
    model = BinaryHeadModel(config)
    model.eval()  # turn off dropout so the synthetic test is deterministic
    return model


def _synthetic_inputs(model, batch_size=4, seq_len=5):
    """Mock input_ids/attention_mask/labels."""
    # Avoid pad/special tokens; ids must be < vocab_size.
    vocab_size = model.backbone.config.vocab_size
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        100, min(1000, vocab_size), (batch_size, seq_len), generator=g
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return input_ids, attention_mask, labels


def test_forward_without_loss_weight_matches_mean_reduction():
    """``loss_weight=None`` recovers the prior reduction='mean' loss."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    # Compute reference by replicating forward up to logits and using
    # reduction='mean' CrossEntropy.
    with torch.no_grad():
        backbone_out = model.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = model.dropout(backbone_out.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
        ref_loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
    assert out["loss"].detach().item() == pytest.approx(ref_loss.item())


def test_forward_with_uniform_weights_matches_mean_reduction():
    """``loss_weight = ones`` is mathematically identical to the mean."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    uniform_w = torch.ones(labels.shape, dtype=torch.float32)
    out_w = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=uniform_w,
    )
    out_none = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels,
    )
    assert out_w["loss"].detach().item() == pytest.approx(
        out_none["loss"].detach().item()
    )


def test_forward_with_nonuniform_weights_matches_manual_formula():
    """``(per_sample * w).sum() / w.sum()`` matches the model's reduction."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    weights = torch.tensor([0.4, 10.0, 2.07, 1.0], dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=weights,
    )
    with torch.no_grad():
        backbone_out = model.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = model.dropout(backbone_out.last_hidden_state[:, 0, :])
        logits = model.binary_head(pooled)
        per_sample = nn.functional.cross_entropy(
            logits, labels, reduction="none"
        )
        manual = (per_sample * weights).sum() / weights.sum()
    assert out["loss"].detach().item() == pytest.approx(manual.item())


def test_forward_with_all_zero_weights_returns_zero_loss():
    """An all-zero weight vector yields zero loss (no NaN from /0)."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    zero_w = torch.zeros(labels.shape, dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=zero_w,
    )
    assert out["loss"].detach().item() == pytest.approx(0.0)
    assert torch.isfinite(out["loss"]).item()


def test_forward_with_wrong_loss_weight_shape_raises():
    """Shape mismatch raises ValueError instead of silent broadcasting."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    # length=3 instead of batch_size=4
    bad_w = torch.tensor([0.4, 1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, loss_weight=bad_w,
        )


# ---------- end-to-end: dataset + collator preserves loss_weight ----------


def test_dataset_collator_carries_loss_weight_through():
    """Tokenized HF dataset + DataCollatorWithPadding emits a
    ``loss_weight`` tensor of shape ``(B,)`` and dtype float."""
    tok = AutoTokenizer.from_pretrained("roberta-base")
    # Tiny dataset with explicit source + loss_weight columns.
    data = {
        "text": ["a leak sentence.", "a clean sentence."],
        "labels": [1, 0],
        "loss_weight": [10.0, 0.4],
        "source": ["clean_forget", "clean_aux"],
    }
    raw = Dataset.from_dict(data)

    def _tok(examples):
        out = tok(examples["text"], truncation=True, max_length=32)
        return out

    tokenized = raw.map(_tok, batched=True)
    # Mimic the script's collator + signature-filter handling: drop string
    # ``source`` and ``text`` columns to avoid the collator's
    # tensor-conversion error on string fields. (Real HF Trainer does
    # this via _remove_unused_columns based on model.forward signature.)
    tokenized = tokenized.remove_columns(["text", "source"])

    collator = DataCollatorWithPadding(tok, padding="longest")
    batch = collator([tokenized[i] for i in range(len(tokenized))])

    assert "loss_weight" in batch
    lw = batch["loss_weight"]
    assert isinstance(lw, torch.Tensor)
    assert lw.shape == (2,)
    assert lw.dtype in (torch.float32, torch.float64)
    assert lw.tolist() == pytest.approx([10.0, 0.4])
    assert "labels" in batch and batch["labels"].shape == (2,)
    assert "input_ids" in batch and batch["input_ids"].ndim == 2


def test_forward_signature_carries_loss_weight_kwarg():
    """HF Trainer reads model.forward's parameter names to decide which
    dataset columns to preserve; ``loss_weight`` must be an explicit
    named parameter (not a generic **kwargs catch-all)."""
    import inspect

    model = _tiny_binary_model()
    params = list(inspect.signature(model.forward).parameters.keys())
    assert "loss_weight" in params


# ---------- gradient through loss_weight ----------


def test_gradient_through_loss_weight_matches_manual():
    """Gradient on logits under a weighted loss equals the manual formula.

    Acts as a regression guard: a hidden ``.detach()`` or ``.item()``
    on the per-row loss would silently zero the autograd graph
    upstream of ``loss_weight``, and this test would catch it.
    """
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    weights = torch.tensor([0.4, 10.0, 2.07, 1.0], dtype=torch.float32)

    # --- model-path gradient
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=weights,
    )
    out["loss"].backward()
    # The full backbone-through-head grad is too noisy to compare row-
    # by-row. Instead, snapshot the binary_head's weight gradient as a
    # summary statistic; compare against the same backward path through
    # the manual reduction.
    head_grad_model = model.binary_head.weight.grad
    assert head_grad_model is not None
    model_grad_norm = head_grad_model.detach().norm().item()
    # Zero grads before manual replay
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # --- manual replay
    backbone_out = model.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
    )
    pooled = model.dropout(backbone_out.last_hidden_state[:, 0, :])
    logits = model.binary_head(pooled)
    per_sample = nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )
    manual_loss = (per_sample * weights).sum() / weights.sum()
    manual_loss.backward()
    head_grad_manual = model.binary_head.weight.grad
    assert head_grad_manual is not None
    manual_grad_norm = head_grad_manual.detach().norm().item()

    # Same backward graph → same grad norm.
    assert model_grad_norm == pytest.approx(manual_grad_norm, rel=1e-5)


def test_gradient_zero_when_weights_are_all_zero():
    """Saturation: all-zero weights → zero loss → zero gradient on logits."""
    model = _tiny_binary_model()
    input_ids, attention_mask, labels = _synthetic_inputs(model)
    zero_w = torch.zeros(labels.shape, dtype=torch.float32)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask,
        labels=labels, loss_weight=zero_w,
    )
    out["loss"].backward()
    head_grad = model.binary_head.weight.grad
    assert head_grad is not None
    assert torch.allclose(head_grad, torch.zeros_like(head_grad))


# ---------- config round-trip ----------


def test_per_source_loss_weights_round_trip_through_config(tmp_path):
    """Per-source weight map survives ``save_pretrained`` / ``from_pretrained``."""
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    weights = {"clean_idk": 10.0, "clean_aux": 0.4, "clean_retain": 2.07}
    config = BinaryHeadConfig(
        backbone_config=cfg.to_dict(),
        per_source_loss_weights=weights,
    )
    config.save_pretrained(str(tmp_path))
    loaded = BinaryHeadConfig.from_pretrained(str(tmp_path))
    assert loaded.per_source_loss_weights == weights


def test_per_source_loss_weights_default_is_none():
    """Construct without the kwarg → attribute is None (not the empty dict)."""
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    config = BinaryHeadConfig(backbone_config=cfg.to_dict())
    assert config.per_source_loss_weights is None


def test_model_reload_emits_identical_logits_with_per_source_weights(tmp_path):
    """Model saved with ``per_source_loss_weights`` set reloads cleanly and
    emits bit-identical logits on the same input.

    Closes the coverage gap noted by the post-implementation code review:
    config round-trip is tested separately, but the model's behavior after
    reload (with the persisted weight dict in its config) wasn't covered.
    Verifies that the per-source weights, which only affect training-time
    loss, don't accidentally leak into the inference forward pass.
    """
    cfg = AutoConfig.from_pretrained("roberta-base")
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    weights = {"clean_idk": 10.0, "clean_aux": 0.4, "clean_retain": 2.07}
    config = BinaryHeadConfig(
        backbone_config=cfg.to_dict(),
        per_source_loss_weights=weights,
    )
    torch.manual_seed(0)
    model = BinaryHeadModel(config)
    model.eval()

    # Build a tiny batch and capture pre-save logits.
    input_ids = torch.randint(0, 50265, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out_pre = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_pre = out_pre["logits"].clone()

    # Save + reload via from_pretrained.
    model.save_pretrained(str(tmp_path))
    reloaded = BinaryHeadModel.from_pretrained(str(tmp_path))
    reloaded.eval()

    # Config should round-trip with the weight dict intact.
    assert reloaded.config.per_source_loss_weights == weights

    with torch.no_grad():
        out_post = reloaded(input_ids=input_ids, attention_mask=attention_mask)
    logits_post = out_post["logits"]

    # Bit-identical: per-source weights are training-time only; inference
    # forward must produce the same logits regardless of the persisted map.
    assert torch.allclose(logits_pre, logits_post, atol=0.0, rtol=0.0)
