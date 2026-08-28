"""Learnable inference-time components for CoT regeneration.

``SoftToken`` — a single learned sequence of embedding vectors inserted
at CoT truncation points via forward hooks to disrupt forget-set
continuations.  This is the canonical (and only) inference-time
corruption.
"""

import torch
import torch.nn as nn


def _register_schedule_hook(
    module: "nn.Module",
    resolved_schedule: list[list[tuple[int, "torch.Tensor"]]],
    n_tokens: int,
):
    """Register a forward hook that applies a per-row soft-token schedule.

    Parameters
    ----------
    resolved_schedule
        Outer list of length ``B`` (batch size). ``resolved_schedule[k]``
        is a list of ``(token_position, embedding_param)`` entries for
        row ``k``. Empty inner list means "no writes for that row." Each
        ``embedding_param`` is the ``nn.Parameter`` whose values should
        be written; gradient flows through it like any other parameter.

    The hook fires during prefill only (``seq_len > 1``). At fire time
    it asserts ``len(resolved_schedule) == outputs.shape[0]`` so that
    misalignment between caller-declared and actual batch size fails
    fast with a clear message.

    A single ``outputs.clone()`` covers all writes — far less allocator
    churn than registering N independent per-write hooks.
    """
    expected_batch = len(resolved_schedule)
    has_work = any(row for row in resolved_schedule)

    def hook(mod, inputs, outputs):
        # Skip autoregressive decode steps (seq_len == 1); the batch-size
        # mismatch check is intentionally gated to prefill, since per-step
        # AR generation may pass a B=1 slice through the same module.
        if outputs.shape[1] <= 1:
            return outputs
        if outputs.shape[0] != expected_batch:
            raise RuntimeError(
                f"Soft-token schedule length {expected_batch} does not "
                f"match batch size {outputs.shape[0]}"
            )
        if not has_work:
            return outputs
        seq_len = outputs.shape[1]
        # Bounds check every entry up front so a bad write is reported
        # before we pay for the clone.
        for entries in resolved_schedule:
            for pos, _ in entries:
                if pos + n_tokens > seq_len:
                    raise RuntimeError(
                        f"Soft-token block (n_tokens={n_tokens}) at "
                        f"token_position={pos} would write past the end "
                        f"of outputs (seq_len={seq_len})"
                    )
        # Clone once so autograd graph survives in-place writes.
        outputs = outputs.clone()
        for row_idx, entries in enumerate(resolved_schedule):
            for pos, emb in entries:
                outputs[row_idx, pos:pos + n_tokens, :] = emb.to(
                    device=outputs.device, dtype=outputs.dtype
                )
        return outputs

    return module.register_forward_hook(hook)


def forget_term(l_forget: torch.Tensor, margin: float) -> torch.Tensor:
    """Compute the forget loss contribution for soft-token training.

    ``margin <= 0`` returns ``-l_forget`` (unbounded reward for suppression).
    ``margin > 0`` returns ``clamp(margin - l_forget, min=0)`` — a hinge
    that stops rewarding further CE growth once ``l_forget`` exceeds the
    margin, preventing the optimizer from driving the hidden state into
    gibberish purely to push forget-continuation CE toward infinity.
    """
    if margin > 0:
        return torch.clamp(margin - l_forget, min=0.0)
    return -l_forget


def contrast_term(
    l_forget: torch.Tensor, l_idk: torch.Tensor, margin: float
) -> torch.Tensor:
    """Preference-style ranking loss for soft-token training.

    Wants ``log P(idk) - log P(leak) >= margin`` per-token, which in CE
    terms is ``l_forget - l_idk >= margin``. Returns
    ``clamp(margin - (l_forget - l_idk), min=0)``. Saturates (zero gradient)
    once the token already prefers the IDK continuation over the leak
    continuation by the given margin. Replaces the separate forget and IDK
    terms with a single ranking objective, which targets coherent refusal
    rather than unbounded CE growth.
    """
    return torch.clamp(margin - l_forget + l_idk, min=0.0)


def retain_term(l_retain: torch.Tensor, margin: float) -> torch.Tensor:
    """One-sided hinge on the retain loss for soft-token training.

    ``margin <= 0`` returns ``l_retain`` (default behavior — λ·L_ret pulls
    L_ret toward zero indefinitely, which over long training drives the bank
    to make a specific retain target overly likely).

    ``margin > 0`` returns ``clamp(l_retain - margin, min=0)`` — active when
    ``l_retain > margin`` (gradient pulls L_ret down toward the margin) and
    saturated to zero when ``l_retain ≤ margin`` (no gradient; the bank is
    free to do whatever the other terms drive). Caps how far the retain
    term will push the bank toward the (mismatched-author) retain target,
    which empirically reduces inference-time fabrication artefacts.

    Currently applied only to ``L_ret_clean`` in train_soft_token.py;
    ``L_ret_ctx`` (legacy/ablation) is not wrapped — re-evaluate if reviving
    that path.
    """
    if margin > 0:
        return torch.clamp(l_retain - margin, min=0.0)
    return l_retain


def truncate_for_soft_token_budget(
    before_ids: list[int],
    pad_ids: list[int],
    after_ids: list[int],
    max_seq_len: int,
    append_eos: bool,
) -> tuple[list[int], list[int]]:
    """Truncate ``(before_ids, after_ids)`` to fit within ``max_seq_len``.

    The K-pad block (``pad_ids``) is always preserved unchanged — it carries
    the soft-token slot, which is the load-bearing position for the bank's
    forward hook. Truncation always removes from the body, never from the
    pad. Returned ``(before_ids, after_ids)`` satisfy
    ``len(before_ids) + len(pad_ids) + len(after_ids) <= max_seq_len``.

    Three regimes:

    * No truncation needed → returned unchanged.
    * Body has room (``max_after > 0``) → trim ``after_ids`` from the end.
      With ``append_eos=True`` and ``len(after_ids) > max_after``, keep
      ``max_after - 1`` body tokens plus the final token (treated as EOS),
      so the EOS supervision survives at the end of the budget.
    * No body room (``max_after <= 0``) → degenerate row: prompt+prefix
      already fills the budget. Returns trimmed ``before_ids`` and an
      empty ``after_ids``. Callers should treat this as a skip signal
      via ``cont_start == cont_end``.

    BOS preservation: trimming is always from the tail of ``before_ids``
    (slice ``[:keep_before]``), so a leading BOS at ``before_ids[0]`` is
    preserved as long as ``keep_before >= 1``. The only path that could
    drop BOS is ``keep_before == 0`` in the degenerate regime, but that
    also yields ``after_ids = []`` → ``valid=False`` → the cross-pass
    filter drops the row, never silently passing a BOS-less training
    sample to ``assemble_pass_batch``.

    Raises ``ValueError`` if ``len(pad_ids) > max_seq_len`` — the K-pad
    block alone cannot fit, so no valid sequence exists for any caller.
    """
    if len(pad_ids) > max_seq_len:
        raise ValueError(
            f"max_seq_len={max_seq_len} cannot fit the K-pad block "
            f"(n_tokens={len(pad_ids)}); no valid sequence exists."
        )

    total_len = len(before_ids) + len(pad_ids) + len(after_ids)
    if total_len <= max_seq_len:
        return before_ids, after_ids

    max_after = max_seq_len - len(before_ids) - len(pad_ids)
    if max_after <= 0:
        # Degenerate row: prompt+prefix already fills the budget.
        keep_before = max_seq_len - len(pad_ids)
        return before_ids[:keep_before], []

    if append_eos and len(after_ids) > max_after:
        # Preserve the EOS token at the end of the budget.
        return before_ids, after_ids[:max_after - 1] + [after_ids[-1]]

    return before_ids, after_ids[:max_after]


class SoftToken(nn.Module):
    """A learnable embedding block for insertion at truncation points.

    The block holds ``n_tokens`` learned vectors; ``apply_hook`` overrides
    ``n_tokens`` consecutive positions starting at ``token_position``.
    ``n_tokens=1`` reproduces the original single-vector behavior.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each embedding (must match the model's embedding layer).
    n_tokens : int
        Number of consecutive embedding vectors per block. Default 1.
    """

    def __init__(self, embed_dim: int = 4096, n_tokens: int = 1):
        super().__init__()
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        self.n_tokens = n_tokens
        self.embedding = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.02)

    def apply_hook(
        self, module: nn.Module, schedule: list[list[int]]
    ):
        """Register a forward hook implementing per-row soft-token writes.

        Parameters
        ----------
        schedule
            Outer list of length ``B`` (the prefill batch size).
            ``schedule[k]`` is the list of token positions in row ``k``
            where this soft token's learned ``(n_tokens, D)`` block will
            be written. Empty inner list means "no writes for that row";
            empty outer list (B=0) is allowed but the hook is a no-op.

        The schedule's outer length is captured at registration; the
        hook raises ``RuntimeError`` at fire time if the actual batch
        size disagrees, so misalignment fails fast with a clear message.

        For B=1 callers (today's inference and soft-token training),
        ``[[pos1, pos2, ...]]`` writes all listed positions in row 0.

        Multiple positions in the same row's inner list produce
        independent writes; if two entries collide on the same range,
        the later one in the list wins (sequential ``=`` semantics).

        Returns the hook handle for later removal.
        """
        if not isinstance(schedule, list):
            raise TypeError(
                f"schedule must be a list, got {type(schedule).__name__}"
            )
        for row_positions in schedule:
            if not isinstance(row_positions, list):
                raise TypeError(
                    f"each row in schedule must be a list, got "
                    f"{type(row_positions).__name__}"
                )
            for pos in row_positions:
                if not isinstance(pos, int) or isinstance(pos, bool):
                    raise TypeError(
                        f"token positions must be int, got "
                        f"{type(pos).__name__}"
                    )
                if pos < 0:
                    raise ValueError(
                        f"token positions must be non-negative, got {pos}"
                    )
        # Resolve every entry to (pos, embedding_param) once at registration.
        # All entries for SoftToken share self.embedding.
        resolved = [
            [(pos, self.embedding) for pos in row_positions]
            for row_positions in schedule
        ]
        return _register_schedule_hook(module, resolved, self.n_tokens)

    def save(self, path: str) -> None:
        """Save the learned embedding to disk."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, embed_dim: int = 4096, n_tokens: int | None = None
    ) -> "SoftToken":
        """Load a previously saved soft token.

        Supports both the new ``(n_tokens, embed_dim)`` parameter shape and
        the legacy ``(embed_dim,)`` shape (treated as ``n_tokens=1``).
        ``n_tokens`` and ``embed_dim`` are inferred from the saved tensor;
        if explicit values are passed they are validated against the saved
        shape.
        """
        sd = torch.load(path, weights_only=True)
        emb = sd["embedding"]
        if emb.dim() == 1:
            # Legacy K=1 checkpoint: reshape to (1, embed_dim).
            sd["embedding"] = emb.unsqueeze(0)
            saved_n, saved_d = 1, emb.shape[0]
        elif emb.dim() == 2:
            saved_n, saved_d = emb.shape[0], emb.shape[1]
        else:
            raise ValueError(
                f"Unexpected embedding shape {tuple(emb.shape)} in {path!r}"
            )
        if n_tokens is not None and n_tokens != saved_n:
            raise ValueError(
                f"Loaded n_tokens={saved_n} but caller requested "
                f"n_tokens={n_tokens}"
            )
        if embed_dim != saved_d:
            raise ValueError(
                f"Loaded embed_dim={saved_d} but caller requested "
                f"embed_dim={embed_dim}"
            )
        st = cls(embed_dim=saved_d, n_tokens=saved_n)
        st.load_state_dict(sd)
        return st
