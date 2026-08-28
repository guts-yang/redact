"""
Tests for the SoftToken learnable embedding block.

Usage:
    conda run -n redact python -m pytest tests/test_soft_token.py -v
"""

import pytest
import torch
import torch.nn as nn

from redact.regen.learned_hooks import (
    SoftToken, contrast_term, forget_term, retain_term,
    truncate_for_soft_token_budget,
)


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_init_shape(n_tokens):
    st = SoftToken(embed_dim=128, n_tokens=n_tokens)
    assert st.embedding.shape == (n_tokens, 128)
    assert st.n_tokens == n_tokens


def test_init_default_dim():
    st = SoftToken()
    assert st.embedding.shape == (1, 4096)
    assert st.n_tokens == 1


def test_init_rejects_nonpositive_n_tokens():
    with pytest.raises(ValueError):
        SoftToken(embed_dim=64, n_tokens=0)
    with pytest.raises(ValueError):
        SoftToken(embed_dim=64, n_tokens=-1)


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_hook_replaces_at_position(n_tokens):
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim, n_tokens=n_tokens)
    module = nn.Linear(10, embed_dim)

    pos = 2
    handle = st.apply_hook(module, [[pos]])
    try:
        x = torch.randn(1, pos + n_tokens + 2, 10)
        output = module(x)
        expected = st.embedding.detach().to(
            device=output.device, dtype=output.dtype
        )
        # output[0, pos:pos+K, :] has shape (K, D); expected has shape (K, D).
        assert torch.allclose(output[0, pos:pos + n_tokens, :], expected)
    finally:
        handle.remove()


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_hook_prefill_only(n_tokens):
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim, n_tokens=n_tokens)
    module = nn.Linear(10, embed_dim)

    handle = st.apply_hook(module, [[0]])
    try:
        x = torch.randn(1, 1, 10)
        output = module(x)
        # Re-run without hook to get baseline
        handle.remove()
        baseline = module(x)
        assert torch.equal(output, baseline), (
            "Hook should be a no-op when seq_len == 1"
        )
    finally:
        pass  # handle already removed


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_hook_does_not_modify_other_positions(n_tokens):
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim, n_tokens=n_tokens)
    module = nn.Linear(10, embed_dim)

    seq_len = max(8, n_tokens + 4)
    pos = 2
    x = torch.randn(1, seq_len, 10)
    # Get baseline without hook
    with torch.no_grad():
        baseline = module(x).clone()

    handle = st.apply_hook(module, [[pos]])
    try:
        with torch.no_grad():
            output = module(x)
        # Positions outside [pos, pos+n_tokens) should be unchanged
        for p in range(seq_len):
            if pos <= p < pos + n_tokens:
                continue
            assert torch.equal(output[:, p, :], baseline[:, p, :]), (
                f"Position {p} was modified but should not have been"
            )
    finally:
        handle.remove()


def test_hook_bounds_check_raises_when_block_overruns():
    """Hook raises a clear error if the K-block would write past the end."""
    embed_dim = 16
    st = SoftToken(embed_dim=embed_dim, n_tokens=4)
    module = nn.Linear(10, embed_dim)

    # pos=3 + n_tokens=4 = 7 > seq_len=5
    handle = st.apply_hook(module, [[3]])
    try:
        x = torch.randn(1, 5, 10)
        with pytest.raises(RuntimeError, match="write past the end"):
            module(x)
    finally:
        handle.remove()


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_save_load_roundtrip(tmp_path, n_tokens):
    embed_dim = 128
    st = SoftToken(embed_dim=embed_dim, n_tokens=n_tokens)
    path = str(tmp_path / f"soft_token_k{n_tokens}.pt")
    st.save(path)

    loaded = SoftToken.load(path, embed_dim=embed_dim)
    assert loaded.n_tokens == n_tokens
    assert loaded.embedding.shape == (n_tokens, embed_dim)
    assert torch.equal(st.embedding, loaded.embedding)


def test_load_legacy_1d_checkpoint(tmp_path):
    """Legacy K=1 checkpoints saved with shape (D,) should load as n_tokens=1."""
    embed_dim = 64
    legacy_emb = torch.randn(embed_dim) * 0.02
    path = str(tmp_path / "legacy.pt")
    torch.save({"embedding": legacy_emb}, path)

    loaded = SoftToken.load(path, embed_dim=embed_dim)
    assert loaded.n_tokens == 1
    assert loaded.embedding.shape == (1, embed_dim)
    assert torch.equal(loaded.embedding[0], legacy_emb)


def test_load_validates_explicit_n_tokens(tmp_path):
    """Passing an n_tokens that disagrees with the saved shape errors clearly."""
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim, n_tokens=4)
    path = str(tmp_path / "k4.pt")
    st.save(path)
    with pytest.raises(ValueError, match="n_tokens"):
        SoftToken.load(path, embed_dim=embed_dim, n_tokens=2)


@pytest.mark.parametrize("n_tokens", [1, 4])
def test_gradient_flows(n_tokens):
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim, n_tokens=n_tokens)
    module = nn.Linear(10, embed_dim)

    handle = st.apply_hook(module, [[1]])
    try:
        x = torch.randn(1, n_tokens + 3, 10)
        output = module(x)
        loss = output.sum()
        loss.backward()
        assert st.embedding.grad is not None, (
            "Gradient should flow through the hook to the soft token embedding"
        )
        # All K rows should receive gradient when the hook covers all K positions.
        per_row_grad_norms = st.embedding.grad.norm(dim=-1)
        assert (per_row_grad_norms > 0).all(), (
            f"Expected gradient on every row of the K-block, got "
            f"per-row norms {per_row_grad_norms.tolist()}"
        )
    finally:
        handle.remove()


def test_is_nn_module():
    assert isinstance(SoftToken(), nn.Module)


# ---------------------------------------------------------------------------
# Schedule-form apply_hook tests (per-row writes; len(schedule) == B contract)
# ---------------------------------------------------------------------------


def test_schedule_multi_row_writes_each_row_independently():
    """B=3 schedule with different positions per row: each row gets the
    embedding at its own position; a row with an empty inner list stays
    at baseline."""
    embed_dim = 16
    st = SoftToken(embed_dim=embed_dim, n_tokens=1)
    module = nn.Linear(8, embed_dim)
    x = torch.randn(3, 5, 8)
    with torch.no_grad():
        baseline = module(x).clone()

    handle = st.apply_hook(module, [[1], [3], []])
    try:
        with torch.no_grad():
            output = module(x)
        expected = st.embedding.detach().to(output.dtype)[0]
        assert torch.allclose(output[0, 1, :], expected)
        assert torch.allclose(output[1, 3, :], expected)
        # Row 0 untouched at pos 3; row 1 untouched at pos 1; row 2 baseline.
        assert torch.equal(output[0, 3, :], baseline[0, 3, :])
        assert torch.equal(output[1, 1, :], baseline[1, 1, :])
        assert torch.equal(output[2], baseline[2])
    finally:
        handle.remove()


def test_schedule_multiple_positions_in_one_row():
    """A row's inner list can carry multiple positions; all are written."""
    embed_dim = 16
    st = SoftToken(embed_dim=embed_dim, n_tokens=1)
    module = nn.Linear(8, embed_dim)
    handle = st.apply_hook(module, [[1, 3]])
    try:
        with torch.no_grad():
            output = module(torch.randn(1, 5, 8))
        expected = st.embedding.detach().to(output.dtype)[0]
        assert torch.allclose(output[0, 1, :], expected)
        assert torch.allclose(output[0, 3, :], expected)
    finally:
        handle.remove()


def test_schedule_length_must_match_batch_size():
    """RuntimeError at fire time when len(schedule) != actual batch size."""
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    handle = st.apply_hook(module, [[0], [0]])  # declares B=2
    try:
        x = torch.randn(3, 4, 8)  # actual B=3
        with pytest.raises(
            RuntimeError, match=r"schedule length 2 does not match batch size 3"
        ):
            module(x)
    finally:
        handle.remove()


def test_schedule_empty_outer_raises_when_actual_batch_nonzero():
    """An empty outer schedule (B=0 declared) must reject any actual
    nonzero-batch prefill — the length check is the load-bearing safety
    rail when callers might forget to populate the schedule per row."""
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    handle = st.apply_hook(module, [])  # B=0 declared
    try:
        with pytest.raises(
            RuntimeError, match=r"schedule length 0 does not match batch size 2"
        ):
            module(torch.randn(2, 4, 8))
    finally:
        handle.remove()


def test_schedule_bounds_check_reports_offending_row_position():
    """Bounds error fires when any row's position+n_tokens overruns,
    even if other rows are within bounds."""
    st = SoftToken(embed_dim=16, n_tokens=2)
    module = nn.Linear(8, 16)
    # Row 0 at pos 0 is fine (0 + 2 = 2 <= seq_len=4); row 1 at pos 3 overruns.
    handle = st.apply_hook(module, [[0], [3]])
    try:
        with pytest.raises(
            RuntimeError, match=r"token_position=3 would write past the end"
        ):
            module(torch.randn(2, 4, 8))
    finally:
        handle.remove()


def test_schedule_all_empty_inner_lists_is_per_row_no_op():
    """All-empty inner lists: hook validates B but performs no writes."""
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        baseline = module(x).clone()
    handle = st.apply_hook(module, [[], []])
    try:
        with torch.no_grad():
            output = module(x)
        assert torch.equal(output, baseline)
    finally:
        handle.remove()


def test_schedule_hook_skips_during_autoregressive_step():
    """Hook must remain a no-op when seq_len == 1, regardless of batch size
    declared by schedule (the size mismatch check is gated on prefill)."""
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    # Schedule declares B=2 but autoregressive step runs B=1, seq_len=1.
    handle = st.apply_hook(module, [[0], [0]])
    try:
        x = torch.randn(1, 1, 8)
        with torch.no_grad():
            output = module(x)
        # Compare against baseline with hook removed.
        handle.remove()
        with torch.no_grad():
            baseline = module(x)
        assert torch.equal(output, baseline)
    finally:
        pass  # already removed


def test_schedule_negative_position_rejected_at_registration():
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    with pytest.raises(ValueError, match="non-negative"):
        st.apply_hook(module, [[-1]])


def test_schedule_type_validation_outer_must_be_list():
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    with pytest.raises(TypeError, match="schedule must be a list"):
        st.apply_hook(module, "not a list")  # type: ignore[arg-type]


def test_schedule_type_validation_inner_must_be_list():
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    with pytest.raises(TypeError, match="each row in schedule must be a list"):
        st.apply_hook(module, [(0,)])  # type: ignore[arg-type]


def test_schedule_type_validation_position_must_be_int():
    st = SoftToken(embed_dim=16, n_tokens=1)
    module = nn.Linear(8, 16)
    with pytest.raises(TypeError, match="positions must be int"):
        st.apply_hook(module, [["not an int"]])  # type: ignore[list-item]


def test_schedule_hook_removal_restores_baseline():
    """After handle.remove(), a fresh forward pass matches the un-hooked
    baseline — guard against handle-leak regressions when step 2 starts
    composing schedules."""
    embed_dim = 16
    st = SoftToken(embed_dim=embed_dim, n_tokens=1)
    module = nn.Linear(8, embed_dim)
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        baseline = module(x).clone()
    handle = st.apply_hook(module, [[0], [1]])
    with torch.no_grad():
        _ = module(x)  # hook fires
    handle.remove()
    with torch.no_grad():
        after_remove = module(x)
    assert torch.equal(after_remove, baseline)


def test_schedule_works_under_bf16_outputs():
    """Schedule writes respect dtype casting on the embedding param."""
    embed_dim = 16
    st = SoftToken(embed_dim=embed_dim, n_tokens=1)
    module = nn.Linear(8, embed_dim).to(torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    handle = st.apply_hook(module, [[0], [2]])
    try:
        with torch.no_grad():
            output = module(x)
        assert output.dtype == torch.bfloat16
        expected = st.embedding.detach().to(dtype=torch.bfloat16)[0]
        assert torch.allclose(output[0, 0, :], expected)
        assert torch.allclose(output[1, 2, :], expected)
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# forget_term (hinge loss) tests
# ---------------------------------------------------------------------------


class TestForgetTerm:
    def test_margin_zero_returns_negated_loss(self):
        l = torch.tensor(2.5, requires_grad=True)
        out = forget_term(l, margin=0.0)
        assert torch.equal(out, -l)

    def test_margin_negative_returns_negated_loss(self):
        l = torch.tensor(2.5, requires_grad=True)
        out = forget_term(l, margin=-1.0)
        assert torch.equal(out, -l)

    def test_hinge_active_below_margin(self):
        # l_forget=2.0 < margin=5.0 -> term = 5.0 - 2.0 = 3.0
        l = torch.tensor(2.0, requires_grad=True)
        out = forget_term(l, margin=5.0)
        assert torch.isclose(out, torch.tensor(3.0))

    def test_hinge_saturates_at_margin(self):
        # l_forget=5.0 == margin=5.0 -> term = 0 (boundary)
        l = torch.tensor(5.0, requires_grad=True)
        out = forget_term(l, margin=5.0)
        assert out.item() == 0.0

    def test_hinge_saturates_above_margin(self):
        # l_forget=10.0 > margin=5.0 -> term clamped to 0
        l = torch.tensor(10.0, requires_grad=True)
        out = forget_term(l, margin=5.0)
        assert out.item() == 0.0

    def test_gradient_flows_below_margin(self):
        # Gradient of (margin - l) w.r.t. l is -1 when unsaturated
        l = torch.tensor(2.0, requires_grad=True)
        out = forget_term(l, margin=5.0)
        out.backward()
        assert l.grad is not None
        assert torch.isclose(l.grad, torch.tensor(-1.0))

    def test_no_gradient_when_saturated(self):
        # When clamped at 0, gradient w.r.t. l is 0
        l = torch.tensor(10.0, requires_grad=True)
        out = forget_term(l, margin=5.0)
        out.backward()
        assert l.grad is not None
        assert l.grad.item() == 0.0

    def test_gradient_negated_when_margin_disabled(self):
        # Old behavior: gradient w.r.t. l is -1 regardless of magnitude
        l = torch.tensor(100.0, requires_grad=True)
        out = forget_term(l, margin=0.0)
        out.backward()
        assert l.grad is not None
        assert torch.isclose(l.grad, torch.tensor(-1.0))


# ---------------------------------------------------------------------------
# retain_term (one-sided hinge above margin) tests
# ---------------------------------------------------------------------------


class TestRetainTerm:
    def test_margin_zero_returns_loss_directly(self):
        # margin <= 0 disables the hinge — return l_retain unchanged.
        l = torch.tensor(2.5, requires_grad=True)
        out = retain_term(l, margin=0.0)
        assert torch.equal(out, l)

    def test_margin_negative_returns_loss_directly(self):
        l = torch.tensor(2.5, requires_grad=True)
        out = retain_term(l, margin=-1.0)
        assert torch.equal(out, l)

    def test_hinge_active_above_margin(self):
        # l_retain=8.0 > margin=5.0 -> term = 8.0 - 5.0 = 3.0
        l = torch.tensor(8.0, requires_grad=True)
        out = retain_term(l, margin=5.0)
        assert torch.isclose(out, torch.tensor(3.0))

    def test_hinge_saturates_at_margin(self):
        # l_retain=5.0 == margin=5.0 -> term = 0 (boundary)
        l = torch.tensor(5.0, requires_grad=True)
        out = retain_term(l, margin=5.0)
        assert out.item() == 0.0

    def test_hinge_saturates_below_margin(self):
        # l_retain=2.0 < margin=5.0 -> term clamped to 0
        l = torch.tensor(2.0, requires_grad=True)
        out = retain_term(l, margin=5.0)
        assert out.item() == 0.0

    def test_gradient_flows_above_margin(self):
        # Gradient of (l - margin) w.r.t. l is +1 when unsaturated.
        # Combined with the loss (+λ * retain_term), this pulls l down.
        l = torch.tensor(8.0, requires_grad=True)
        out = retain_term(l, margin=5.0)
        out.backward()
        assert l.grad is not None
        assert torch.isclose(l.grad, torch.tensor(1.0))

    def test_no_gradient_when_saturated(self):
        # When clamped at 0 (l < margin), gradient w.r.t. l is 0.
        l = torch.tensor(2.0, requires_grad=True)
        out = retain_term(l, margin=5.0)
        out.backward()
        assert l.grad is not None
        assert l.grad.item() == 0.0

    def test_gradient_passthrough_when_margin_disabled(self):
        # Old behavior: gradient w.r.t. l is +1 regardless of magnitude
        # (caller multiplies by λ_retain to control direction/magnitude).
        l = torch.tensor(100.0, requires_grad=True)
        out = retain_term(l, margin=0.0)
        out.backward()
        assert l.grad is not None
        assert torch.isclose(l.grad, torch.tensor(1.0))

    def test_batched_per_row_hinge(self):
        # Hinge applies independently to each row.
        l = torch.tensor([2.0, 5.0, 8.0], requires_grad=True)
        out = retain_term(l, margin=5.0)
        assert torch.allclose(out, torch.tensor([0.0, 0.0, 3.0]))


# ---------------------------------------------------------------------------
# contrast_term (preference-ranking) tests
# ---------------------------------------------------------------------------


class TestContrastTerm:
    def test_hinge_active_when_below_margin(self):
        # l_forget=2.0, l_idk=1.0 -> contrast=1.0, margin=3.0 -> term=2.0
        l_forget = torch.tensor(2.0, requires_grad=True)
        l_idk = torch.tensor(1.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        assert torch.isclose(out, torch.tensor(2.0))

    def test_saturates_when_ranking_margin_satisfied(self):
        # l_forget=5.0, l_idk=1.0 -> contrast=4.0 > margin=3.0 -> term=0
        l_forget = torch.tensor(5.0, requires_grad=True)
        l_idk = torch.tensor(1.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        assert out.item() == 0.0

    def test_saturates_at_boundary(self):
        # l_forget=4.0, l_idk=1.0 -> contrast=3.0 == margin=3.0 -> term=0
        l_forget = torch.tensor(4.0, requires_grad=True)
        l_idk = torch.tensor(1.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        assert out.item() == 0.0

    def test_gradient_signs_when_active(self):
        # When hinge active: term = margin - l_forget + l_idk
        # d/dl_forget = -1 (push l_forget UP -> higher CE on leak)
        # d/dl_idk = +1 (push l_idk DOWN -> lower CE on idk)
        l_forget = torch.tensor(2.0, requires_grad=True)
        l_idk = torch.tensor(1.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        out.backward()
        assert l_forget.grad is not None and l_idk.grad is not None
        assert torch.isclose(l_forget.grad, torch.tensor(-1.0))
        assert torch.isclose(l_idk.grad, torch.tensor(1.0))

    def test_no_gradient_when_saturated(self):
        l_forget = torch.tensor(10.0, requires_grad=True)
        l_idk = torch.tensor(1.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        out.backward()
        assert l_forget.grad is not None and l_idk.grad is not None
        assert l_forget.grad.item() == 0.0
        assert l_idk.grad.item() == 0.0

    def test_rewards_preference_even_when_l_idk_high(self):
        # Token can satisfy margin by pushing l_forget way up even if l_idk
        # is also somewhat high — it's the *difference* that matters.
        l_forget = torch.tensor(8.0, requires_grad=True)
        l_idk = torch.tensor(4.0, requires_grad=True)
        out = contrast_term(l_forget, l_idk, margin=3.0)
        # contrast = 8-4 = 4 > margin=3 -> saturated
        assert out.item() == 0.0


class TestTruncateForSoftTokenBudget:
    """Tests for the truncation helper used by tokenize_with_soft_token.

    The helper preserves the K-pad block unchanged in every regime, trims
    the body when the budget is tight, and (when ``append_eos=True``)
    keeps the final token of ``after_ids`` at the end of the budget so
    EOS supervision survives truncation.
    """

    def test_no_truncation_returns_unchanged(self):
        before = [1, 2, 3]
        pads = [0, 0, 0, 0]
        after = [4, 5, 6, 7]
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=20, append_eos=False
        )
        assert b == before
        assert a == after

    def test_no_truncation_with_eos_returns_unchanged(self):
        before = [1, 2, 3]
        pads = [0, 0, 0, 0]
        after_with_eos = [4, 5, 6, 99]  # 99 = "EOS"
        b, a = truncate_for_soft_token_budget(
            before, pads, after_with_eos, max_seq_len=20, append_eos=True
        )
        assert b == before
        assert a == after_with_eos
        assert a[-1] == 99  # EOS preserved

    def test_truncates_body_from_end_no_eos(self):
        before = [1] * 10
        pads = [0, 0, 0, 0]
        after = list(range(20, 40))  # 20 tokens
        # Budget 20: room for body = 20 - 10 - 4 = 6 tokens.
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=20, append_eos=False
        )
        assert b == before
        assert len(a) == 6
        assert a == [20, 21, 22, 23, 24, 25]
        assert len(b) + len(pads) + len(a) == 20

    def test_truncates_body_preserves_eos_when_append_eos(self):
        before = [1] * 10
        pads = [0, 0, 0, 0]
        # Body is 5 normal tokens + EOS=99 (caller already appended).
        # Budget 20: room = 6 tokens.
        after_with_eos = [20, 21, 22, 23, 24, 25, 26, 27, 99]
        b, a = truncate_for_soft_token_budget(
            before, pads, after_with_eos, max_seq_len=20, append_eos=True
        )
        assert b == before
        assert len(a) == 6
        assert a[-1] == 99  # EOS preserved at end
        # Body is the first (max_after - 1) = 5 tokens, then EOS.
        assert a == [20, 21, 22, 23, 24, 99]

    def test_eos_preservation_at_boundary_max_after_one(self):
        # max_after == 1 corner: only the EOS survives.
        before = [1] * 15
        pads = [0, 0, 0, 0]
        after_with_eos = [20, 21, 22, 99]
        b, a = truncate_for_soft_token_budget(
            before, pads, after_with_eos, max_seq_len=20, append_eos=True
        )
        # max_after = 20 - 15 - 4 = 1. With EOS preservation, keep
        # max_after - 1 = 0 body tokens + EOS = [99].
        assert a == [99]
        assert len(b) + len(pads) + len(a) == 20

    def test_degenerate_returns_empty_after_ids(self):
        # before+pad already exceeds max_seq_len -> degenerate row.
        before = [1] * 30
        pads = [0, 0, 0, 0]
        after = [20, 21, 22]
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=20, append_eos=False
        )
        # max_after = 20 - 30 - 4 = -14 <= 0 -> degenerate.
        assert a == []
        assert len(b) + len(pads) <= 20  # K-pad block fits within budget
        assert len(b) == 20 - len(pads)  # before trimmed to fill remainder

    def test_degenerate_with_eos_also_drops_eos(self):
        # Even with append_eos=True, the degenerate path returns empty.
        # The caller is expected to skip via cont_start == cont_end, so
        # no EOS supervision is attempted on a row with no body context.
        before = [1] * 30
        pads = [0, 0, 0, 0]
        after_with_eos = [20, 21, 22, 99]
        b, a = truncate_for_soft_token_budget(
            before, pads, after_with_eos, max_seq_len=20, append_eos=True
        )
        assert a == []  # EOS dropped, caller skips this row

    def test_degenerate_with_pad_block_at_budget_limit(self):
        # before is empty but pad alone fills entire budget. Helper should
        # still produce a valid (zero before, full pad, zero after) layout.
        before = []
        pads = [0] * 20
        after = [1, 2, 3]
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=20, append_eos=False
        )
        assert b == []
        assert a == []

    def test_pad_too_big_for_budget_raises(self):
        # K-pad block alone exceeds budget -> no valid sequence.
        with pytest.raises(ValueError, match="cannot fit the K-pad block"):
            truncate_for_soft_token_budget(
                [1, 2, 3], [0] * 25, [4, 5], max_seq_len=20, append_eos=False
            )

    def test_pad_ids_are_never_modified(self):
        # The pad block must be returned to caller unchanged in length;
        # the helper signature returns (before, after) only, so we verify
        # by construction that the helper never inspects pad contents.
        before = list(range(50))
        pads = [99] * 4  # distinctive value
        after = list(range(100, 200))
        # All four regimes:
        for max_seq_len, append_eos in [
            (1000, False),  # no truncation
            (1000, True),  # no truncation, eos
            (60, False),    # body trim
            (60, True),     # body trim, eos
            (20, False),    # degenerate
        ]:
            b, a = truncate_for_soft_token_budget(
                before.copy(), pads.copy(), after.copy(),
                max_seq_len=max_seq_len, append_eos=append_eos,
            )
            # pads is not returned, but we verify the helper kept pads
            # consistent by checking that `before + pads + after` fits the
            # budget under the regimes that should fit it.
            if max_seq_len >= len(pads):
                assert len(b) + len(pads) + len(a) <= max_seq_len

    def test_exact_fit_no_truncation(self):
        # total_len == max_seq_len: no truncation should fire.
        before = [1] * 5
        pads = [0, 0, 0, 0]
        after = [10, 11, 12]
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=12, append_eos=False
        )
        assert b == before
        assert a == after

    def test_one_over_budget_truncates_one(self):
        before = [1] * 5
        pads = [0, 0, 0, 0]
        after = [10, 11, 12, 13]
        b, a = truncate_for_soft_token_budget(
            before, pads, after, max_seq_len=12, append_eos=False
        )
        # max_after = 12 - 5 - 4 = 3.
        assert len(a) == 3
        assert a == [10, 11, 12]

    def test_one_over_budget_with_eos(self):
        before = [1] * 5
        pads = [0, 0, 0, 0]
        after_with_eos = [10, 11, 12, 99]
        b, a = truncate_for_soft_token_budget(
            before, pads, after_with_eos, max_seq_len=12, append_eos=True
        )
        # max_after = 3, but len(after) = 4 > max_after -> EOS path.
        # Keep first 2 body tokens plus EOS.
        assert a == [10, 11, 99]
        assert a[-1] == 99
