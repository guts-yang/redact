import contextlib
import copy
import time

import torch

from redact.regen.utils import (
    apply_embeddings_extraction_hook,
    get_nested_attr,
)
from redact.utils import log_print


class RegeneratingModel:
    def __init__(
        self,
        model,
        # CoT regeneration corruption (optional)
        soft_token=None,
    ):
        # Mirror the wrapped model's identity and runtime handles onto
        # this wrapper so the regeneration machinery can reach them
        # without going through the original object.
        self.model = model.model
        self.model_name = model.model_name
        self.tokenizer = model.tokenizer
        self.model_config = model.model_config
        self.generation_config = model.generation_config
        self.device = model.device
        # Resolve the embedding layer once; the soft-token hook attaches
        # here. ``embedding_module`` in the config is a dotted path
        # relative to the inner HF module (e.g. "model.embed_tokens").
        self.embedding_module = get_nested_attr(
            self.model, self.model_config["embedding_module"]
        )
        # Optional learned corruption applied during regeneration.
        self.soft_token = soft_token
        # Sink for captured embeddings (populated by
        # apply_embeddings_extraction) and the registry of hook handles
        # this wrapper owns and must tear down.
        self.embeddings_data = []
        self._hook_handles = []
        # TEMPORARY: when True, the chunked path emits per-chunk
        # ``[chunk]`` and ``[chunk-summary]`` log lines (B_active,
        # prefill_len, prefill/AR breakdown via lm_head forward hook).
        # Used during the Phase 1 chunked-regen perf experiments to
        # measure whether re-prefill cost is eating the chunked path's
        # theoretical savings.  Strip alongside the engine's
        # ``instrumentation_verbose`` toggle when the experiment is done.
        self.chunk_instrumentation = True

    def remove_hooks(self):
        """Remove all hooks registered by this RegeneratingModel via their handles."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _strip_unsupported_kwargs(self, kwargs):
        """Drop ``token_type_ids`` for architectures that reject it.

        OLMo / Qwen and the Falcon-180B chat model error out when handed
        ``token_type_ids``; the eval engine may include it generically.
        """
        lowered = self.model_name.lower()
        if (
            "olmo" in lowered
            or "qwen" in lowered
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)

    def __call__(self, prompts=None, answers=None, *args, **kwargs):
        # The eval engine forwards ``prompts``/``answers`` positionally
        # by keyword; this wrapper swallows them (the old prompt-level
        # corruption that consumed them is gone) and only relays the rest
        # to the underlying model.
        del prompts, answers
        self._strip_unsupported_kwargs(kwargs)
        # Clear any leftover hooks before a plain forward pass.
        self.remove_hooks()
        return self.model(*args, **kwargs)

    def generate(self, prompts=None, *args, **kwargs):
        # ``prompts`` is accepted for engine-call symmetry and discarded;
        # this is the uncorrupted generation path, so strip any hooks
        # first and hand the remaining arguments straight through.
        del prompts
        self._strip_unsupported_kwargs(kwargs)
        self.remove_hooks()
        return self.model.generate(*args, **kwargs)

    def regenerate(
        self,
        soft_token_positions: list[list[int]] | None = None,
        chunk_size: int | None = None,
        per_row_remaining: list[int] | None = None,
        chunk_mode: str = "kvcache",
        **kwargs,
    ):
        """Generate with CoT corruption using the configured soft token.

        Parameters
        ----------
        soft_token_positions : list[list[int]] | None
            Per-row anchor positions during prefill.
            ``soft_token_positions[k]`` is the list of START positions
            in row ``k`` where soft-token blocks should be written; each
            entry covers ``soft_token.n_tokens`` consecutive positions.
            Outer length must equal the prefill batch size; ``None`` or
            an empty outer list registers no soft-token hook.
        chunk_size : int | None
            When non-None and ``per_row_remaining`` is also provided,
            dispatches into a chunked path (``chunk_mode`` selects
            which one): generation runs in chunks of ``chunk_size``
            tokens; rows that emit ``eos_token_id`` or hit their
            per-row budget cap are dropped from the active subset
            before the next chunk runs.  This eliminates the slow-tail
            tax where a non-terminating row holds the batch open while
            the rest sit padded.  Greedy-only (validated against
            ``generation_config``); raises ValueError on non-greedy
            configs.  Default ``None`` keeps the single-shot path
            byte-identical to today.
        per_row_remaining : list[int] | None
            Per-row maximum number of new tokens.  Required when
            ``chunk_size`` is set; outer length must equal the prefill
            batch size.  Each entry caps that row's accumulated new-token
            count across chunks.  Single-shot path uses
            ``generation_config.max_new_tokens`` directly and ignores
            this argument.
        chunk_mode : str
            Which chunked implementation to use when ``chunk_size`` is
            set.  ``"kvcache"`` (Phase 2, default) keeps the KV cache
            live across chunks: chunk 0 prefills under the soft-token
            hook, chunks 1+ are pure AR-decode steps reading from the
            cache (no re-prefill, no soft-token hook re-fire).
            ``"reprefill"`` (Phase 1) re-prefills each chunk's full
            accumulated input — simple but pays the quadratic-in-
            seq-len prefill cost on every chunk.

            On bf16 + flash_attention_2 (the LRM-target config), the
            probe at ``scripts/_probe_phase1_vs_singleshot.py`` showed
            that ``"reprefill"`` introduces drift from the canonical
            single-shot output: re-prefilling at chunk N>0 with a
            longer accumulated sequence produces slightly different
            KVs at the original prompt positions than the original
            chunk-0 prefill did, because flash_attn_2's tile
            reduction order depends on input length.  ``"kvcache"``
            reuses chunk-0's KVs (which match what single-shot would
            produce) and is therefore byte-equivalent to single-shot
            on this config — strictly closer to the reference than
            ``"reprefill"``.  Hence the default flip.

            Keep ``"reprefill"`` available as a diagnostic /
            comparison reference only.

        Notes
        -----
        Today's B=1 callers pass ``[[pos1, pos2, ...]]`` (single-row
        schedule).  Batched regen widens this to non-trivial outer
        lists; the per-row schedule shape lets callers express
        per-row anchors without flattening.
        """
        if (
            "olmo" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)
        if chunk_size is not None:
            if per_row_remaining is None:
                raise ValueError(
                    "per_row_remaining is required when chunk_size is set."
                )
            if chunk_mode == "kvcache":
                return self._generate_with_kv_cache_dropping(
                    soft_token_positions=soft_token_positions,
                    chunk_size=chunk_size,
                    per_row_remaining=per_row_remaining,
                    **kwargs,
                )
            if chunk_mode != "reprefill":
                raise ValueError(
                    f"chunk_mode must be 'reprefill' or 'kvcache'; "
                    f"got {chunk_mode!r}."
                )
            return self._generate_with_chunked_dropping(
                soft_token_positions=soft_token_positions,
                chunk_size=chunk_size,
                per_row_remaining=per_row_remaining,
                **kwargs,
            )
        with self._run_with_soft_token_hook(soft_token_positions):
            return self.model.generate(**kwargs)

    @contextlib.contextmanager
    def _run_with_soft_token_hook(
        self,
        soft_token_positions: list[list[int]] | None,
    ):
        """Register the soft-token forward hook for the duration of a block.

        Centralises the hook lifecycle that ``regenerate`` and the chunked
        path both need: removes any existing hooks, builds the per-row
        :class:`SoftToken` schedule, registers the hook, yields, and
        removes the hook on exit.

        Behaviour notes (mirror today's ``regenerate``):
          * No hook is registered when ``self.soft_token is None`` or
            ``soft_token_positions`` is falsy (``None`` or empty outer
            list).  The body still runs in those cases.
          * Per-position shape/type validation lives in
            :meth:`SoftToken.apply_hook`.
          * Hook handles are not tracked in ``self._hook_handles``;
            cleanup happens via this context manager.
        """
        self.remove_hooks()
        st_handles = []
        if self.soft_token is not None and soft_token_positions:
            plain_schedule: list[list[int]] = [
                list(positions) for positions in soft_token_positions
            ]
            st_handles.append(
                self.soft_token.apply_hook(
                    self.embedding_module, plain_schedule
                )
            )
        try:
            yield
        finally:
            for h in st_handles:
                h.remove()

    def _find_prefill_timing_module(self):
        """Return a module whose forward output reliably has the full
        sequence length on prefill and length 1 on AR-decode steps.

        Used by the chunked path's per-chunk prefill timing hook.  NOT
        ``lm_head`` — HF's ``generate()`` passes ``num_logits_to_keep=1``,
        which causes ``LlamaForCausalLM.forward`` (and similar) to slice
        ``hidden_states[:, slice_indices, :]`` before the ``lm_head``
        call.  ``lm_head`` then sees ``seq_len=1`` regardless of phase
        and the prefill timing hook's ``outputs.shape[1] > 1`` gate
        never fires.

        Falls through a list of common final-norm attribute paths.
        Returns ``None`` if none match; the caller falls back to
        total-only timing.
        """
        inner = getattr(self.model, "model", None)
        # Llama-family final norm
        if inner is not None:
            for name in ("norm", "ln_f", "final_layer_norm"):
                mod = getattr(inner, name, None)
                if mod is not None:
                    return mod
        # GPT-2 family
        transformer = getattr(self.model, "transformer", None)
        if transformer is not None:
            mod = getattr(transformer, "ln_f", None)
            if mod is not None:
                return mod
        # GPT-NeoX family
        neox = getattr(self.model, "gpt_neox", None)
        if neox is not None:
            mod = getattr(neox, "final_layer_norm", None)
            if mod is not None:
                return mod
        return None

    def _validate_gen_config_for_chunking(self, gen_config) -> None:
        """Reject any non-greedy or exotic ``generation_config``.

        The chunked path implements greedy AR decoding only.  It does
        not replicate HF's logits processors (top-k / top-p /
        temperature / repetition_penalty / bad_words_ids), beam search,
        or any sampling.  A mismatch here would silently diverge from
        the single-shot path's outputs — surface it loudly instead.
        """
        if gen_config is None:
            raise ValueError(
                "Chunked path requires generation_config to be set so its "
                "greedy/sampling configuration can be validated."
            )
        if getattr(gen_config, "do_sample", False):
            raise ValueError(
                "Chunked path requires do_sample=False; got do_sample=True."
            )
        num_beams = getattr(gen_config, "num_beams", 1) or 1
        if num_beams != 1:
            raise ValueError(
                f"Chunked path requires num_beams=1; got {num_beams}."
            )
        rp = getattr(gen_config, "repetition_penalty", 1.0)
        if rp is not None and rp != 1.0:
            raise ValueError(
                f"Chunked path requires repetition_penalty=1.0; got {rp}."
            )
        if getattr(gen_config, "bad_words_ids", None):
            raise ValueError(
                "Chunked path does not support bad_words_ids."
            )
        nrgs = getattr(gen_config, "no_repeat_ngram_size", 0)
        if nrgs:
            raise ValueError(
                f"Chunked path does not support no_repeat_ngram_size; "
                f"got {nrgs}."
            )
        # Sampling/warper params: even with ``do_sample=False`` HF can
        # apply ``TemperatureLogitsWarper`` / ``TopK`` / ``TopP`` if these
        # are set, depending on the code path.  Greedy is robust to
        # ``temperature=1.0`` / ``top_k=0`` / ``top_p=1.0`` (no-ops); any
        # other value would silently diverge from our argmax-equivalent
        # decoding.  ``temperature`` may be ``None`` (HF default in some
        # configs) — treat that as no-op too.
        temperature = getattr(gen_config, "temperature", 1.0)
        if temperature is not None and temperature != 1.0:
            raise ValueError(
                f"Chunked path requires temperature=1.0 (or None); "
                f"got {temperature}."
            )
        top_k = getattr(gen_config, "top_k", 0)
        if top_k:
            raise ValueError(
                f"Chunked path requires top_k=0 (disabled); got {top_k}."
            )
        top_p = getattr(gen_config, "top_p", 1.0)
        if top_p is not None and top_p != 1.0:
            raise ValueError(
                f"Chunked path requires top_p=1.0 (no filter); got {top_p}."
            )

    # ------------------------------------------------------------------
    # Pure helpers shared by chunked-path implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_left_pad_to_real_content(
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
    ) -> tuple[list[int], list[list[int]]]:
        """Strip the left-pad from each row to recover real content.

        ``attention_mask`` is 1 on real tokens (including soft-token
        pad slots, whose embeddings are overridden by the hook) and 0
        on left-pad.  The count of leading 0s is the left-pad amount.

        Returns
        -------
        orig_left_pad : list[int]
            Per-row count of leading pad tokens in ``input_ids``.
        real_content : list[list[int]]
            Per-row list of real (non-pad) token IDs.
        """
        B = input_ids.shape[0]
        orig_left_pad: list[int] = []
        real_content: list[list[int]] = []
        for k in range(B):
            row_mask = attention_mask[k].tolist()
            n_leading = 0
            for v in row_mask:
                if v == 0:
                    n_leading += 1
                else:
                    break
            orig_left_pad.append(n_leading)
            real_content.append(input_ids[k, n_leading:].tolist())
        return orig_left_pad, real_content

    @staticmethod
    def _detect_per_row_stop(
        row_tokens: list[int],
        eos_token_id: int,
        remaining_for_row: int,
    ) -> tuple[list[int], bool, bool]:
        """Truncate one row's new-chunk tokens at the first stopping
        condition; report which one fired.

        Stopping rules (applied in order):

        1. **EOS** — find the first ``eos_token_id`` in ``row_tokens``;
           truncate inclusive of EOS.  Works under ``pad_token_id ==
           eos_token_id`` (LLaMA): the FIRST hit in HF generate's
           output is the genuine EOS, not a subsequent right-pad.

        2. **Budget cap** — if the (post-EOS-trim) length is at or
           past ``remaining_for_row``, trim to exactly
           ``remaining_for_row`` tokens.  Budget takes precedence over
           EOS-then-pad: when the cap falls before the EOS, we drop
           the EOS along with everything past it.  Matches the
           engine's per-row truncation behaviour.

        Returns
        -------
        kept_tokens : list[int]
            Truncated row tokens.
        stopped_by_eos : bool
        stopped_by_budget : bool
        """
        stopped_by_eos = False
        if eos_token_id in row_tokens:
            eos_pos = row_tokens.index(eos_token_id)
            row_tokens = row_tokens[:eos_pos + 1]
            stopped_by_eos = True

        stopped_by_budget = False
        if len(row_tokens) >= remaining_for_row:
            row_tokens = row_tokens[:remaining_for_row]
            stopped_by_budget = True

        return row_tokens, stopped_by_eos, stopped_by_budget

    @staticmethod
    def _reassemble_chunked_output(
        input_ids: "torch.LongTensor",
        new_tokens_per_row: list[list[int]],
        pad_token_id: int,
    ) -> "torch.LongTensor":
        """Build a ``(B_orig, prefill_len + max_new)`` tensor matching
        ``model.generate``'s shape contract.

        Each row is the ORIGINAL ``input_ids[k]`` (left-pad and all)
        followed by that row's generated tokens, then ``pad_token_id``
        fill to ``max_new``.  Downstream callers slice past
        ``prefill_len = input_ids.shape[1]`` to recover the new
        tokens.
        """
        B_orig = input_ids.shape[0]
        device = input_ids.device
        prefill_len = input_ids.shape[1]
        max_new = max((len(g) for g in new_tokens_per_row), default=0)
        total_len = prefill_len + max_new
        output = torch.full(
            (B_orig, total_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        output[:, :prefill_len] = input_ids
        for k, g in enumerate(new_tokens_per_row):
            if g:
                output[k, prefill_len:prefill_len + len(g)] = torch.tensor(
                    g, dtype=torch.long, device=device,
                )
        return output

    def _generate_with_chunked_dropping(
        self,
        *,
        soft_token_positions: list[list[int]] | None,
        chunk_size: int,
        per_row_remaining: list[int],
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        generation_config,
        eos_token_id,
        pad_token_id,
        **kwargs,
    ):
        """Greedy chunked AR with per-row EOS / budget dropping.

        Mechanism: at each iteration, the active subset's content
        (original real tokens + tokens generated in earlier chunks) is
        re-padded into a fresh ``(B_active, max_len)`` tensor and fed
        to ``model.generate(max_new_tokens=chunk_size, ...)`` under a
        freshly-registered soft-token hook (re-registration per chunk
        is required because the hook validates ``outputs.shape[0] ==
        expected_batch`` at fire time, and that count shrinks as rows
        drop out).  After each chunk: per-row stops are detected (first
        ``eos_token_id`` in the new tokens, or accumulated new-token
        count reaching ``per_row_remaining[k]``); stopped rows are
        finalized into the per-row generated-token buffer; survivors
        carry their content forward.  Loop ends when the active set is
        empty.

        Output: ``(B_orig, prefill_len + max_new_across_rows)`` long
        tensor matching ``model.generate``'s shape contract.  Each row
        is the ORIGINAL ``input_ids[k]`` (left-pad and all) followed by
        that row's generated tokens, then ``pad_token_id`` to fill to
        ``max_new_across_rows``.  Downstream callers slice past
        ``prefill_len = input_ids.shape[1]`` to recover the new tokens.

        Soft-token re-registration semantics: each chunk re-pads the
        active rows' content into a fresh tensor whose left-pad amounts
        differ from chunk 0's (because content lengths grow at
        different rates).  Soft-token positions are re-translated per
        chunk: from the chunk-0 absolute position frame to the chunk-N
        absolute frame via ``new_pos = real_pos + chunk_N_left_pad``,
        where ``real_pos = chunk_0_pos - chunk_0_left_pad``.  The
        resulting schedule is filtered to surviving rows and registered
        for that chunk only; the previous chunk's hook handle is
        removed by ``_run_with_soft_token_hook``'s exit.

        Behavioural notes
        -----------------
        * Bypasses ``self.model.generate``'s ``max_new_tokens`` per call
          (we pass a per-chunk cap).  Per-row caps are enforced manually
          after each chunk.
        * EOS detection assumes the first occurrence of ``eos_token_id``
          in a row's new tokens is the actual EOS even when
          ``pad_token_id == eos_token_id`` (LLaMA-family); the search
          finds the EOS before any subsequent right-pad fill.
        * No KV cache forwarding — each chunk re-prefills the full
          accumulated input.  Phase 2 (KV-cache forwarding) eliminates
          this overhead if Phase 1's wall-time win is insufficient.
        """
        self._validate_gen_config_for_chunking(generation_config)

        B_orig = input_ids.shape[0]
        if len(per_row_remaining) != B_orig:
            raise ValueError(
                f"per_row_remaining length ({len(per_row_remaining)}) "
                f"does not match input_ids batch size ({B_orig})."
            )
        if any(r < 0 for r in per_row_remaining):
            raise ValueError(
                f"per_row_remaining must be non-negative; got "
                f"{per_row_remaining!r}."
            )
        if soft_token_positions is not None:
            if len(soft_token_positions) != B_orig:
                raise ValueError(
                    f"soft_token_positions outer length "
                    f"({len(soft_token_positions)}) does not match "
                    f"input_ids batch size ({B_orig})."
                )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}.")

        device = input_ids.device

        orig_left_pad, real_content = self._strip_left_pad_to_real_content(
            input_ids, attention_mask,
        )

        # Soft-token positions in the per-row real-content frame.
        # chunk-0 abs pos minus chunk-0 left_pad = pos in real content.
        # In any later chunk N, abs pos = real_pos + chunk_N_left_pad.
        real_st_pos: list[list[int]] | None
        if soft_token_positions is not None:
            real_st_pos = [
                [pos - orig_left_pad[k] for pos in row_positions]
                for k, row_positions in enumerate(soft_token_positions)
            ]
        else:
            real_st_pos = None

        # Per-row state across chunks.
        new_tokens_per_row: list[list[int]] = [[] for _ in range(B_orig)]
        remaining: list[int] = list(per_row_remaining)
        # Active rows carry their growing content forward.
        active_content: list[list[int]] = [list(c) for c in real_content]
        active_indices: list[int] = [
            k for k in range(B_orig) if remaining[k] > 0
        ]

        # ``self.remove_hooks()`` is called inside
        # ``_run_with_soft_token_hook`` per chunk; defensively clear once
        # up front in case the caller registered something we don't own.
        self.remove_hooks()

        # Per-chunk instrumentation log: each entry is a dict with B_active,
        # prefill_len, chunk_max_new, t_setup, t_total, t_prefill, t_ar_decode,
        # n_new_actual.  Used to measure whether re-prefill cost is eating
        # the chunked path's theoretical savings — if t_prefill dominates
        # t_total at high prefill_len, Phase 2 (KV-cache forwarding) is
        # warranted; if t_ar_decode dominates, Phase 1 captures the win.
        #
        # Module choice for the prefill-timing hook: NOT ``lm_head``.  HF's
        # ``generate()`` passes ``num_logits_to_keep=1`` so
        # ``LlamaForCausalLM.forward`` slices ``hidden_states[:,
        # slice_indices, :]`` before the ``lm_head`` call — meaning
        # ``lm_head`` always sees ``seq_len=1`` regardless of prefill vs
        # AR.  Hook on the final norm (``model.model.norm`` for Llama,
        # ``transformer.ln_f`` for GPT-2 family) instead, which always
        # sees the full sequence length and reliably distinguishes prefill
        # (``seq_len > 1``) from AR-decode (``seq_len == 1``).
        # ``timing_module`` is None when the model class doesn't expose any
        # known final-norm path; the per-chunk log then drops the
        # prefill/ar split fields and reports ``total`` only.
        instrument = self.chunk_instrumentation
        chunk_log: list[dict] = []
        timing_module = (
            self._find_prefill_timing_module() if instrument else None
        )

        while active_indices:
            B_active = len(active_indices)
            chunk_setup_start = time.perf_counter() if instrument else 0.0
            # Build padded chunk_input_ids from active rows' content.
            content_lens = [len(active_content[i]) for i in active_indices]
            max_len = max(content_lens)
            # Defensive: an active row with zero content (e.g. all-pad
            # input passed in) would produce ``max_len == 0`` and
            # ``model.generate`` cannot AR-decode from an empty
            # sequence.  The engine filters ``remaining_budget <= 0``
            # rows upstream so this is unreachable in canonical use,
            # but the RegeneratingModel API is becoming more public-ish
            # under chunk_size; surface a clear error.
            if max_len == 0:
                raise ValueError(
                    "Chunked path: every active row has empty content "
                    "(max_len=0).  Caller must filter rows with no "
                    "tokens before invoking the chunked path."
                )
            chunk_left_pad = [max_len - cl for cl in content_lens]

            chunk_input_ids = torch.full(
                (B_active, max_len),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            chunk_attn = torch.zeros(
                (B_active, max_len), dtype=torch.long, device=device,
            )
            for k, orig_idx in enumerate(active_indices):
                content = active_content[orig_idx]
                lp = chunk_left_pad[k]
                if content:
                    chunk_input_ids[k, lp:] = torch.tensor(
                        content, dtype=torch.long, device=device,
                    )
                chunk_attn[k, lp:] = 1

            # Translate soft-token positions to chunk-N's absolute frame
            # and filter to surviving rows.  Per-row absolute positions
            # = real positions (constant) + chunk-N left_pad (per row).
            chunk_st_pos: list[list[int]] | None = None
            if (
                real_st_pos is not None
                and self.soft_token is not None
            ):
                chunk_st_pos = [
                    [
                        real_st_pos[orig_idx][m] + chunk_left_pad[k]
                        for m in range(len(real_st_pos[orig_idx]))
                    ]
                    for k, orig_idx in enumerate(active_indices)
                ]

            # Per-chunk max_new_tokens: the smaller of chunk_size and
            # the largest remaining budget across active rows.  No
            # active row should have remaining <= 0 (we filter those at
            # entry and at chunk boundaries below), but defensive
            # bounds keep the cap correct in degenerate cases.
            max_active_remaining = max(
                remaining[i] for i in active_indices
            )
            chunk_max_new = min(chunk_size, max_active_remaining)
            if chunk_max_new <= 0:
                break

            chunk_gen_config = copy.copy(generation_config)
            chunk_gen_config.max_new_tokens = chunk_max_new

            chunk_setup_dt = (
                (time.perf_counter() - chunk_setup_start)
                if instrument else None
            )

            # Optional prefill-end timing hook on ``lm_head``.  Fires
            # once at end-of-prefill (outputs.shape[1] > 1); subsequent
            # AR steps fire with seq_len == 1 and are skipped.  When
            # ``lm_head`` isn't exposed, the chunk log drops the
            # prefill/AR breakdown and reports total only.
            timing_state = {"t_prefill_end": None}
            timing_handle = None
            if instrument and timing_module is not None:
                def _prefill_timing_hook(_mod, _inp, outputs, _state=timing_state):
                    if (
                        _state["t_prefill_end"] is None
                        and outputs.shape[1] > 1
                    ):
                        _state["t_prefill_end"] = time.perf_counter()

                timing_handle = timing_module.register_forward_hook(
                    _prefill_timing_hook
                )

            # Re-register hook for this chunk (survivor-filtered
            # schedule, B_active expected_batch).
            try:
                with self._run_with_soft_token_hook(chunk_st_pos):
                    gen_start = time.perf_counter() if instrument else 0.0
                    generated = self.model.generate(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attn,
                        generation_config=chunk_gen_config,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        **kwargs,
                    )
                    gen_end = time.perf_counter() if instrument else 0.0
            finally:
                if timing_handle is not None:
                    timing_handle.remove()

            # Slice newly-generated tokens past the prefill length.
            new_chunk = generated[:, max_len:]  # (B_active, <=chunk_max_new)
            # Detect per-row stops, accumulate, and update active set.
            next_active: list[int] = []
            n_new_this_chunk = 0
            for k, orig_idx in enumerate(active_indices):
                row_tokens, stopped_by_eos, stopped_by_budget = (
                    self._detect_per_row_stop(
                        new_chunk[k].tolist(),
                        eos_token_id,
                        remaining[orig_idx],
                    )
                )
                new_tokens_per_row[orig_idx].extend(row_tokens)
                active_content[orig_idx].extend(row_tokens)
                remaining[orig_idx] -= len(row_tokens)
                n_new_this_chunk += len(row_tokens)

                if stopped_by_eos or stopped_by_budget or remaining[orig_idx] <= 0:
                    continue  # row done — drop from next_active
                next_active.append(orig_idx)

            if instrument:
                t_prefill_end = timing_state["t_prefill_end"]
                t_total = gen_end - gen_start
                t_prefill = (
                    (t_prefill_end - gen_start)
                    if t_prefill_end is not None else None
                )
                t_ar = (
                    (gen_end - t_prefill_end)
                    if t_prefill_end is not None else None
                )
                chunk_log.append({
                    "B_active": B_active,
                    "prefill_len": max_len,
                    "max_new": chunk_max_new,
                    "n_new_actual": n_new_this_chunk,
                    "n_dropped": B_active - len(next_active),
                    "t_setup": chunk_setup_dt,
                    "t_total": t_total,
                    "t_prefill": t_prefill,
                    "t_ar_decode": t_ar,
                })

            active_indices = next_active

        if instrument and chunk_log:
            for i, ch in enumerate(chunk_log):
                if ch["t_prefill"] is not None:
                    breakdown = (
                        f"prefill={ch['t_prefill']:.2f}s "
                        f"ar={ch['t_ar_decode']:.2f}s"
                    )
                else:
                    breakdown = "prefill=na ar=na"
                log_print(
                    f"[chunk] i={i} B={ch['B_active']} "
                    f"prefill_len={ch['prefill_len']} "
                    f"max_new={ch['max_new']} "
                    f"new={ch['n_new_actual']} "
                    f"dropped={ch['n_dropped']} "
                    f"setup={ch['t_setup']:.3f}s "
                    f"total={ch['t_total']:.2f}s {breakdown}"
                )
            n_chunks = len(chunk_log)
            sum_total = sum(ch["t_total"] for ch in chunk_log)
            sum_prefill_entries = [
                ch["t_prefill"] for ch in chunk_log
                if ch["t_prefill"] is not None
            ]
            sum_ar_entries = [
                ch["t_ar_decode"] for ch in chunk_log
                if ch["t_ar_decode"] is not None
            ]
            sum_new = sum(ch["n_new_actual"] for ch in chunk_log)
            if sum_prefill_entries:
                sum_prefill = sum(sum_prefill_entries)
                sum_ar = sum(sum_ar_entries)
                pct_prefill = sum_prefill / sum_total * 100 if sum_total > 0 else 0.0
                pct_ar = sum_ar / sum_total * 100 if sum_total > 0 else 0.0
                breakdown_summary = (
                    f"prefill={sum_prefill:.2f}s ({pct_prefill:.0f}%) "
                    f"ar={sum_ar:.2f}s ({pct_ar:.0f}%)"
                )
            else:
                breakdown_summary = "prefill=na ar=na"
            log_print(
                f"[chunk-summary] n_chunks={n_chunks} "
                f"total_gen={sum_total:.2f}s {breakdown_summary} "
                f"new_tokens={sum_new}"
            )

        return self._reassemble_chunked_output(
            input_ids, new_tokens_per_row, pad_token_id,
        )

    def _generate_with_kv_cache_dropping(
        self,
        *,
        soft_token_positions: list[list[int]] | None,
        chunk_size: int,
        per_row_remaining: list[int],
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        generation_config,
        eos_token_id,
        pad_token_id,
        **kwargs,
    ):
        """Greedy chunked AR with per-row EOS / budget dropping AND KV
        cache forwarding (Phase 2).

        Eliminates the per-chunk re-prefill cost that constrained
        Phase 1 (``_generate_with_chunked_dropping``).  Chunk 0 runs a
        full ``model.generate`` under the soft-token hook context
        manager, capturing both the output sequences and the model's
        ``DynamicCache`` (via ``return_dict_in_generate=True,
        use_cache=True``).  After per-row stop detection, EOS'd /
        budget-exhausted rows are dropped via
        ``past_kv.batch_select_indices(survivors)`` and the
        soft-token hook context manager is exited — chunks 1+ run
        with NO hook registered, so the soft-token writes baked into
        chunk 0's KV cache are not re-applied (the
        ``outputs.shape[1] <= 1`` gate is not the only line of
        defence here).

        Chunks 1+ feed the full accumulated sequences back to
        ``model.generate(past_key_values=past_kv)``; HF's
        ``prepare_inputs_for_generation`` trims to
        ``input_ids[:, past_length:]`` (typically a single token) so
        each chunk runs ``chunk_size`` AR steps without re-prefilling
        the prompt or any prior chunk.

        Output shape contract matches Phase 1 and single-shot:
        ``(B_orig, prefill_len + max_new_across_rows)`` where each
        row is the original ``input_ids[k]`` (left-pad and all)
        followed by that row's generated tokens, then ``pad_token_id``
        fill.

        Behavioural notes
        -----------------
        * Greedy-only.  Validated against ``generation_config`` at
          entry; raises ValueError on any non-trivial sampling /
          warper / beam config.
        * EOS detection assumes the first occurrence of
          ``eos_token_id`` in a row's new tokens is the genuine EOS
          even when ``pad_token_id == eos_token_id`` (LLaMA-family);
          the search finds the EOS before any subsequent right-pad
          fill.
        * Budget cap takes precedence over EOS-then-pad (matches
          Phase 1 and the engine's per-row truncation contract).
        * Mask invariant: before each chunks-1+ ``model.generate``
          call, ``cumulative_mask.shape[1]`` must equal
          ``accumulated.shape[1]``; asserted explicitly to catch
          silent bf16 + flash_attention_2 + DynamicCache slicing
          regressions.
        """
        self._validate_gen_config_for_chunking(generation_config)

        B_orig = input_ids.shape[0]
        if len(per_row_remaining) != B_orig:
            raise ValueError(
                f"per_row_remaining length ({len(per_row_remaining)}) "
                f"does not match input_ids batch size ({B_orig})."
            )
        if any(r < 0 for r in per_row_remaining):
            raise ValueError(
                f"per_row_remaining must be non-negative; got "
                f"{per_row_remaining!r}."
            )
        if soft_token_positions is not None:
            if len(soft_token_positions) != B_orig:
                raise ValueError(
                    f"soft_token_positions outer length "
                    f"({len(soft_token_positions)}) does not match "
                    f"input_ids batch size ({B_orig})."
                )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}.")

        device = input_ids.device

        new_tokens_per_row: list[list[int]] = [[] for _ in range(B_orig)]
        remaining: list[int] = list(per_row_remaining)

        self.remove_hooks()

        # ---- Chunk 0: prefill + first chunk_size AR steps -----------
        chunk_0_max = min(chunk_size, max(remaining) if remaining else 0)
        if chunk_0_max <= 0:
            # No active row has any budget — return input unchanged.
            return self._reassemble_chunked_output(
                input_ids, new_tokens_per_row, pad_token_id,
            )

        chunk_0_gen_config = copy.copy(generation_config)
        chunk_0_gen_config.max_new_tokens = chunk_0_max

        with self._run_with_soft_token_hook(soft_token_positions):
            chunk_0_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=chunk_0_gen_config,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                use_cache=True,
                **kwargs,
            )
        # Hook context manager exited — chunks 1+ run hook-free.

        sequences = chunk_0_outputs.sequences
        past_kv = chunk_0_outputs.past_key_values
        prefill_len = input_ids.shape[1]
        chunk_0_actual = sequences.shape[1] - prefill_len

        # Per-row stops on chunk 0; collect survivor batch-indices.
        survivor_batch_indices: list[int] = []
        survivor_orig_indices: list[int] = []
        for k in range(B_orig):
            row_tokens = sequences[k, prefill_len:].tolist()
            kept, stopped_eos, stopped_budget = self._detect_per_row_stop(
                row_tokens, eos_token_id, remaining[k],
            )
            new_tokens_per_row[k].extend(kept)
            remaining[k] -= len(kept)
            if not (stopped_eos or stopped_budget or remaining[k] <= 0):
                survivor_batch_indices.append(k)
                survivor_orig_indices.append(k)

        if not survivor_batch_indices:
            return self._reassemble_chunked_output(
                input_ids, new_tokens_per_row, pad_token_id,
            )

        # ---- Slice cache to chunk-0 survivors ----------------------
        past_kv.batch_select_indices(survivor_batch_indices)

        # Accumulated sequences and cumulative mask for survivors.
        # cumulative_mask = original_mask[survivors] cat ones(chunk_0_actual).
        # Chunk-0 emitted tokens are all "real" for survivors (they
        # didn't EOS within chunk 0, so no post-EOS pad fill).
        accumulated = sequences[survivor_batch_indices]
        cumulative_mask = torch.cat([
            attention_mask[survivor_batch_indices],
            torch.ones(
                (len(survivor_batch_indices), chunk_0_actual),
                dtype=torch.long, device=device,
            ),
        ], dim=1)

        # ---- Chunks 1+ : pure AR with KV cache ----------------------
        # Map from CURRENT batch position k_in_active to ORIGINAL row
        # index orig_idx — needed to write into ``new_tokens_per_row``
        # and read ``remaining[orig_idx]``.  Survivors always carry
        # forward in original order via batch_select_indices's index
        # list, so this list is simply the survivor_orig_indices we
        # built at chunk 0; it shrinks via cache slicing on each
        # subsequent chunk.
        active_orig_indices: list[int] = list(survivor_orig_indices)

        while active_orig_indices:
            chunk_max = min(
                chunk_size,
                max(remaining[i] for i in active_orig_indices),
            )
            if chunk_max <= 0:
                break

            chunk_gen_config = copy.copy(generation_config)
            chunk_gen_config.max_new_tokens = chunk_max

            # Mask invariant (reviewer W1): full accumulated input
            # must be covered by the cumulative mask before the call.
            assert cumulative_mask.shape[1] == accumulated.shape[1], (
                f"cumulative_mask.shape[1]={cumulative_mask.shape[1]} "
                f"!= accumulated.shape[1]={accumulated.shape[1]}; this "
                f"breaks HF's prepare_inputs_for_generation cache-trim "
                f"contract."
            )

            chunk_outputs = self.model.generate(
                input_ids=accumulated,
                attention_mask=cumulative_mask,
                past_key_values=past_kv,
                generation_config=chunk_gen_config,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                use_cache=True,
                **kwargs,
            )

            new_sequences = chunk_outputs.sequences
            past_kv = chunk_outputs.past_key_values
            chunk_actual = new_sequences.shape[1] - accumulated.shape[1]
            if chunk_actual <= 0:
                # Nothing was generated (rare; usually means all rows
                # in the active batch hit EOS at the same step).
                break

            # Per-row stops on this chunk; build the survivor-of-survivor
            # local index list.
            new_active_local: list[int] = []
            new_active_orig: list[int] = []
            for k_local, orig_idx in enumerate(active_orig_indices):
                row_tokens = new_sequences[
                    k_local, accumulated.shape[1]:
                ].tolist()
                kept, stopped_eos, stopped_budget = self._detect_per_row_stop(
                    row_tokens, eos_token_id, remaining[orig_idx],
                )
                new_tokens_per_row[orig_idx].extend(kept)
                remaining[orig_idx] -= len(kept)
                if not (stopped_eos or stopped_budget or remaining[orig_idx] <= 0):
                    new_active_local.append(k_local)
                    new_active_orig.append(orig_idx)

            if not new_active_local:
                break

            # Slice cache to this chunk's survivors.
            past_kv.batch_select_indices(new_active_local)

            # Update accumulated + mask.
            accumulated = new_sequences[new_active_local]
            cumulative_mask = torch.cat([
                cumulative_mask[new_active_local],
                torch.ones(
                    (len(new_active_local), chunk_actual),
                    dtype=torch.long, device=device,
                ),
            ], dim=1)
            active_orig_indices = new_active_orig

        return self._reassemble_chunked_output(
            input_ids, new_tokens_per_row, pad_token_id,
        )

    def apply_embeddings_extraction(self):
        """Start capturing forward-pass embeddings into ``embeddings_data``.

        Registers an extraction hook on the underlying model, records the
        returned handle so :meth:`remove_hooks` can later detach it, and
        hands the handle back to the caller.
        """
        handle = apply_embeddings_extraction_hook(
            self.model, self.embeddings_data
        )
        self._hook_handles.append(handle)
        return handle
