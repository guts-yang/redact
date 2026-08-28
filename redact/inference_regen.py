"""Regenerating generation engine for reasoning model unlearning.

Extends ReasoningGenerationEngine with detect-and-regenerate loops:
after initial generation, checks each CoT for forget-set leakage and
regenerates from the truncation point with prefix corruption and/or
a learned soft token.  A second phase checks answers for leakage and
regenerates with soft tokens inserted after the ``</think>`` delimiter.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Iterable

import torch
from tqdm import tqdm

from redact.evaluator.utils import split_sentences
from redact.inference import ReasoningGenerationEngine, _remove_hooks
from redact.model.reasoning import ReasoningModel
from redact.utils import log_print, split_cot_answer


@dataclass
class RegenSampleState:
    """Per-row state for the CoT regeneration loop.

    Bundles the locals that ``_regenerate_leaking_sample`` and
    ``_regenerate_leaking_wave`` read and mutate as the
    detect-and-regenerate loop progresses: the cleared prefix
    (text + sentence list + token IDs), the stacked soft-token
    anchors, the attempt counter, and per-row flags.  The wave loop
    operates on ``list[RegenSampleState]`` with one entry per active
    row; the per-sample loop operates on a single instance (B=1
    degenerate case).

    Invariants the loop relies on (mirror today's local-variable
    contract — see ``_regenerate_leaking_sample`` for the full
    discussion):

    * ``clean_prefix == " ".join(prefix_sentences)`` — string form is
      maintained by append, never rejoined from sentences mid-loop
      (Punkt strips inter-sentence whitespace).
    * ``prefix_ids`` is grown by appending ``tokenize(new_part)`` for
      each newly-cleared chunk, never re-tokenized — so token-count
      anchors in ``soft_token_stack`` stay stable across iterations.
      ``prefix_ids`` covers ONLY the classified portion;
      ``frozen_prefix_ids`` (if non-empty) sits before it in the model
      input but is not part of this token list.
    * ``len(prefix_sentences) == first_leak_index`` at every loop
      entry (the leak boundary is the prefix's right edge).
    * ``frozen_prefix_ids`` is set once at row init and never mutated
      thereafter — typed as ``tuple[int, ...]`` so the "fixed baseline"
      contract is a static guarantee rather than a comment-only
      convention.
    """

    prefix_sentences: list[str] = field(default_factory=list)
    clean_prefix: str = ""
    prefix_ids: list[int] = field(default_factory=list)
    # Tokens that sit between the think-prefix and the classified prefix
    # in the model input, but are NOT classified, never grow under
    # forward extension, and do NOT carry the active forward-extension
    # soft-token anchors.  In the CoT phase this is empty (default) and
    # the engine behaves byte-identically.  In the answer phase it
    # carries the ``cot_ids + delim_ids`` baseline so the answer-wave's
    # classified ``prefix_ids`` covers only the answer-prefix portion.
    #
    # Anchors in ``soft_token_stack`` remain positions within
    # ``prefix_ids`` alone; ``frozen_soft_token_stack`` (sister field
    # below) carries CoT-phase pads that splice into ``frozen_prefix_ids``.
    # Both sets are emitted by ``_build_regen_inputs`` with absolute
    # positions offset appropriately.  The wave's per-row budget
    # arithmetic subtracts ``len(frozen_prefix_ids)`` so the frozen
    # baseline counts against ``generation_config.max_new_tokens`` (the
    # model already emitted those tokens once; the regen call must not
    # push the total response beyond the configured cap).
    #
    # Typed as ``tuple[int, ...]`` (not ``list[int]`` like ``prefix_ids``)
    # to make the "fixed baseline, never mutated after init" contract a
    # static guarantee — any accidental ``.append()``/``.extend()`` is a
    # type error.  Pass 1 of ``_build_regen_inputs`` copies into a
    # list before concatenation, so the immutability does not leak into
    # the model-input assembly path.
    frozen_prefix_ids: tuple[int, ...] = field(default_factory=tuple)
    # Soft-token anchors that splice into ``frozen_prefix_ids`` (rather
    # than into ``prefix_ids``).  Each entry is an
    # ``anchor_token_count`` — a position within ``frozen_prefix_ids``
    # alone (NOT a position in the combined ``frozen + prefix``).
    #
    # Used by the answer wave to carry the CoT-phase soft-token stack
    # forward into the answer phase: the CoT pads stay at their
    # original positions inside the cot portion of the model input,
    # while the answer wave's own ``soft_token_stack`` adds new pads
    # at the answer-leak position inside ``prefix_ids``.  Empty in the
    # CoT phase and for clean-CoT (Phase 3 origin) answer-wave rows.
    #
    # Like ``frozen_prefix_ids``, typed as a tuple so the "fixed at
    # init, never mutated" contract is a static guarantee.  Anchors
    # must satisfy ``0 <= anchor <= len(frozen_prefix_ids)``;
    # ``_insert_soft_token_pads`` already handles "anchor at the end"
    # by appending the pad block at the tail.
    frozen_soft_token_stack: tuple[int, ...] = field(default_factory=tuple)
    # Frozen external prior context for the contextual leak classifier —
    # sentences that come BEFORE the target text (CoT or answer) being
    # classified, but live outside ``prefix_sentences``.
    #
    # In the CoT phase this is empty (default): the CoT is the target,
    # and there is no prior text.  In the answer phase it carries the
    # CoT's sentences (``split_sentences(cots[i])``) so the contextual
    # classifier's k-window can span the CoT→answer seam — the first
    # answer sentence sees the last k CoT sentences, the second sees the
    # last k-1 CoT sentences plus the first answer sentence, etc.
    #
    # The detector concatenates this with the cleared per-target
    # ``prefix_sentences`` before passing the prior to the classifier;
    # non-contextual classifiers ignore it.  Set once at row init,
    # never mutated thereafter — typed as ``tuple[str, ...]`` so the
    # immutability is a static guarantee.
    context_prefix_sentences: tuple[str, ...] = field(default_factory=tuple)
    # Each entry is an ``anchor_token_count`` — the number of
    # ``prefix_ids`` tokens that precede this soft-token block.  The
    # single shared :class:`SoftToken` is written at every anchor.
    soft_token_stack: list[int] = field(default_factory=list)
    # Number of regeneration attempts that have been spent on the
    # *current* anchor / prefix.  Resets to 0 on forward extension
    # (a fresh anchor) and increments on same-anchor escalation.
    # The wave loop terminates a row when this reaches
    # ``regen_max_attempts``.
    attempt: int = 0
    best_cot: str = ""
    best_answer: str = ""
    new_continuation_text: str = ""
    # Write-only inside the loop body today — preserved for fidelity with
    # the prior locals so the wave refactor's per-row state list maps
    # 1:1 to today's per-sample fields.
    first_leak_index: int = 0
    # Count of persistent-leak sentences this row has force-cleared into
    # ``prefix_sentences`` after exhausting ``regen_max_attempts`` at a
    # single anchor.  Driven by ``_run_wave_loop``'s exhaust-and-advance
    # branch; surfaced in the per-subset ``[wave-instr]`` summary so the
    # operator can tell how often the regen loop is preserving a
    # known-leaking sentence in the output rather than clearing it.
    n_force_cleared_leaks: int = 0

    def push_anchor(self, anchor: int) -> None:
        """Append ``anchor`` to ``soft_token_stack``.

        Each anchor becomes one soft-token block written at that
        position inside ``prefix_ids`` during the next regeneration's
        prefill.  Repeated pushes at the same anchor stack back-to-back
        blocks (same-anchor escalation).
        """
        self.soft_token_stack.append(anchor)

    def remaining_budget(self, total_budget: int) -> int:
        """Tokens this row may still generate within ``max_new_tokens``.

        Subtracts both the classified prefix (which the model already
        emitted and we are keeping) and the frozen baseline
        (``cot_ids + delim_ids`` in the answer phase, empty in CoT).
        Soft-token pads are excluded — they are injected, not generated.

        May return zero or a negative value when forward extension has
        consumed the entire budget; callers must treat ``<= 0`` as
        "drop this row" (see ``_run_wave_loop``'s over-budget filter
        at the top of each iteration).
        """
        return total_budget - len(self.prefix_ids) - len(self.frozen_prefix_ids)


class RegeneratingReasoningEngine(ReasoningGenerationEngine):
    """ReasoningGenerationEngine with post-generation leak detection and
    batched CoT/answer regeneration.

    After initial generation, each CoT is checked for forget-set leakage
    (one batched ``detect_batch`` call); leaking rows go through the CoT
    wave (:meth:`_regenerate_leaking_wave`).  After CoT regen converges,
    every row's current answer is checked (one more batched
    ``detect_batch`` call); rows whose answer leaks go through the
    answer wave (:meth:`_regenerate_leaking_answer_wave`).

    Both waves share the same per-iteration loop body
    (:meth:`_run_wave_loop`) and the same forward-extension /
    same-anchor-escalation state machine; they differ only in initial
    state setup, the decode-and-assemble callback, and the
    ``frozen_prefix_ids`` baseline (empty for CoT, ``cot_ids + delim_ids``
    for answer).

    Soft-token config lives on the ``RegeneratingModel`` and is invoked via
    its ``regenerate()`` method, fully decoupled from prompt corruption.
    """

    # Class-level toggle for the per-iteration / per-row TEMPORARY
    # instrumentation log lines emitted by ``_run_wave_loop`` and
    # ``_regenerate_leaking_answer_wave`` (the ``[wave-iter*]`` and
    # per-attempt ``Sample N attempt N: ...`` lines).  Defaults True to
    # preserve the diagnostic verbosity used during the batch-regen
    # perf experiments; flip to False (e.g. ``engine.instrumentation_verbose
    # = False`` after construction) to silence them on production runs
    # where the per-batch ``[wave-instr]`` summary alone suffices.
    # Timing accumulators stay live regardless so the summary line is
    # populated either way.  Goes away when the TEMPORARY instrumentation
    # is finally stripped.
    instrumentation_verbose: bool = True

    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        answer_evaluator,
        cot_evaluator=None,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
        apply_empty_override: "bool | Iterable[str]" = True,
        # Regeneration parameters:
        leak_detector=None,
        regen_max_attempts=3,
        chunk_size: int | None = None,
        chunk_mode: str = "kvcache",
    ):
        """``chunk_size`` (when set) routes ``inner.regenerate`` calls
        through a chunked path; ``chunk_mode`` selects which one.
        Generation runs in chunks of ``chunk_size`` tokens; rows that
        emit ``eos_token_id`` or hit their per-row budget cap are
        dropped from the active subset between chunks.  Eliminates the
        slow-tail tax where a non-terminating row holds the batch open
        while EOS'd rows generate wasted padding.  Greedy-only — the
        project's default ``GenerationConfig(do_sample=False,
        use_cache=True)`` satisfies the chunked-path validator;
        non-greedy callers should leave ``chunk_size=None`` and pay the
        slow-tail cost.  Default ``chunk_size=None`` keeps the wave
        loop byte-identical to today's single-shot behaviour.

        ``chunk_mode``:
          * ``"kvcache"`` (Phase 2, default): chunk 0 prefills under
            the soft-token hook and the resulting ``DynamicCache``
            carries forward; chunks 1+ are pure AR-decode steps with
            no re-prefill and no hook (soft-token writes are baked
            into chunk-0's cache).  Cache is sliced via
            ``DynamicCache.batch_select_indices`` to drop EOS'd rows.
            Byte-equivalent to single-shot on bf16 + flash_attention_2
            (the LRM-target config) per the
            ``_probe_phase1_vs_singleshot`` probe.
          * ``"reprefill"`` (Phase 1): each chunk re-prefills the
            survivor batch's full accumulated input; soft-token hook
            re-registered per chunk with shifted positions.  Simple
            but pays the quadratic-in-seq-len prefill cost on every
            chunk, AND introduces drift from single-shot on
            bf16 + FA2 — re-prefilling at chunk N>0 with a longer
            accumulated sequence produces slightly different KVs at
            the original prompt positions than the chunk-0 prefill
            did (FA2's tile reduction order depends on input length).
            Kept available for diagnostic / comparison runs only.
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_module=data_module,
            subset_names=subset_names,
            answer_evaluator=answer_evaluator,
            cot_evaluator=cot_evaluator,
            batch_size=batch_size,
            prompt_prefix=prompt_prefix,
            comparison_length=comparison_length,
            truncate_answers=truncate_answers,
            apply_empty_override=apply_empty_override,
        )
        # When a leak_detector is provided, regeneration is enabled.  Both
        # ``regenerate()`` and a soft-token bank are required: there is no
        # other inference-time corruption channel after the window mode's
        # removal.
        inner = model._inner if isinstance(model, ReasoningModel) else model
        if leak_detector is not None:
            if not hasattr(inner, "regenerate"):
                raise TypeError(
                    f"Regeneration requires a model with regenerate() "
                    f"(e.g. RegeneratingModel), got {type(inner).__name__}"
                )
            if not hasattr(inner, "soft_token") or inner.soft_token is None:
                raise ValueError(
                    "Regeneration requires the model to have a soft_token "
                    "configured (window-mode corruption was removed)."
                )
        self.leak_detector = leak_detector
        self.regen_max_attempts = regen_max_attempts
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive when set; got {chunk_size}."
            )
        if chunk_mode not in ("reprefill", "kvcache"):
            raise ValueError(
                f"chunk_mode must be 'reprefill' or 'kvcache'; "
                f"got {chunk_mode!r}."
            )
        self.chunk_size = chunk_size
        self.chunk_mode = chunk_mode
        # Initialised here so direct callers of ``_run_wave_loop`` (tests
        # that build per-row state and skip ``_generate``'s subset
        # bookkeeping) don't trip an ``AttributeError`` on the
        # ``self._force_cleared_leak_total += 1`` increment in the
        # exhaust-and-advance branch.  ``_generate`` resets to zero
        # per subset, so production reads always see a fresh counter.
        self._force_cleared_leak_total = 0

    def _generate(self):
        self.prepare_dataset()
        padding_side = self.tokenizer.padding_side
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        n_think = getattr(self.model, "n_think_tokens", 0)

        subsets_generations = {}
        for subset_name, dataset in self.datasets.items():
            all_gold_answers, all_gold_cots, all_generated_answers = [], [], []
            all_generated_cot, all_generated_answer = [], []
            all_prompts = []
            total_time, total_examples = 0, 0
            # TEMPORARY (wave-engine instrumentation): track wall-time
            # spent in each phase per batch, plus leaker counts.  Restore
            # to plain initial-only timing before committing.
            cot_wave_total_time = 0.0
            answer_wave_total_time = 0.0
            cot_wave_leaker_count = 0
            # Total persistent-leak sentences force-cleared into the
            # output across all rows in this subset.  Incremented by
            # ``_run_wave_loop``'s exhaust-and-advance branch when it
            # absorbs a persistent leak into ``state.prefix_sentences``.
            # Each increment represents one sentence that exhausted the
            # regen budget at its anchor and was preserved verbatim in
            # the final output (with subsequent leaks past it still
            # having been re-anchored and regenerated).  Stashed on the
            # engine instance so both the CoT and answer waves can write
            # to the same counter without threading return values.
            self._force_cleared_leak_total = 0
            # Cross-method timing: ``self._total_gen_time`` accumulates
            # every model.generate / inner.regenerate call (initial gen,
            # wave-loop regen, answer-phase regen).  ``self._total_detect_time``
            # accumulates every leak_detector.* call (initial detect_batch,
            # per-row detect_continuation in the wave loop, and the answer
            # phase's detect calls).  Reset per subset.
            self._total_gen_time = 0.0
            self._total_detect_time = 0.0
            # TEMPORARY (batch-regen-timing): split _total_gen_time into
            # CoT-regen vs answer-regen contributions so we can see which
            # phase actually pays the slowest-row tax.  CoT-regen is
            # credited inside the wave loop; answer-regen is credited in
            # _run_answer_regen via _last_regen_gen_time.
            self._cot_regen_gen_time = 0.0
            self._answer_regen_gen_time = 0.0
            subset_start_time = time.perf_counter()
            # Running absolute-index offset so per-sample log_print calls
            # can attribute regen events to specific prompt indices rather
            # than batch-local positions.
            abs_sample_base = 0

            for batch in tqdm(
                dataset,
                desc=f"Generating completions of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                _remove_hooks(self.model)
                prompts = batch[self.data_module.gen_prompt_key]
                gold_answers = batch[self.data_module.gen_answer_key]

                tokenized_prompts = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.model.device)

                decode_start = tokenized_prompts["input_ids"].shape[1] + n_think

                start_time = time.perf_counter()
                generated = self.model.generate(
                    **tokenized_prompts,
                    prompts=prompts,
                    generation_config=self.model.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                self._total_gen_time += end_time - start_time
                total_examples += len(prompts)

                # Decode only new tokens (after prompt + think prefix), apply BPE fix
                batch_responses = []
                for i in range(generated.shape[0]):
                    raw = self.tokenizer.decode(
                        generated[i][decode_start:], skip_special_tokens=True
                    )
                    batch_responses.append(raw)

                # Split into CoT and answer at </think> via split_cot_answer
                # (relaxed: tolerant of corruption-induced loss of the
                # trailing newlines after </think>).
                batch_cot, batch_answer = [], []
                for resp in batch_responses:
                    cot, answer = split_cot_answer(resp)
                    batch_cot.append(cot)
                    batch_answer.append(answer)

                gold_cots = batch.get(
                    getattr(self.data_module, "gen_cot_key", None),
                    [""] * len(prompts),
                )

                # ----------------------------------------------------------
                # Phase 2: CoT wave on rows whose CoT leaks.
                # Phase 3: Answer wave on every row whose answer leaks
                # (covers both Phase 2 leakers and clean-CoT-leaking-answer
                # rows in a single batched call).
                # ----------------------------------------------------------
                if self.leak_detector is not None:
                    detect_start = time.perf_counter()
                    # CoT-phase initial detection — no external prior
                    # context (the CoT is the start of classified text).
                    # Prompts thread through for contextual classifiers;
                    # non-contextual classifiers ignore them.
                    results = self.leak_detector.detect_batch(
                        batch_cot, prompts=list(prompts)
                    )
                    self._total_detect_time += time.perf_counter() - detect_start

                    # Snapshot the pre-regen answers so we can detect
                    # which rows the answer wave actually mutated.  Only
                    # rebuilding ``batch_responses[i]`` for rows the
                    # waves touched preserves byte-equivalence with the
                    # pre-Part-4 behavior on rows that go through neither
                    # wave (e.g. samples with bare ``</think>`` artifacts
                    # whose initial response is preserved verbatim).
                    answers_pre_wave = list(batch_answer)
                    cots_pre_wave = list(batch_cot)

                    leaking_indices = [
                        k for k, r in enumerate(results) if r.is_leaking
                    ]
                    # Maps batch_idx -> CoT-wave terminal state for
                    # rows that went through the CoT wave.  Used by
                    # ``_regenerate_leaking_answer_wave`` to carry each
                    # row's CoT-phase soft-token stack forward into
                    # ``state.frozen_soft_token_stack``.  Rows absent
                    # from this dict (clean-CoT, Phase 3 origin) get
                    # an empty carryover.
                    cot_states_by_idx: dict[int, RegenSampleState] = {}
                    if leaking_indices:
                        cot_wave_leaker_count += len(leaking_indices)
                        cot_wave_start = time.perf_counter()
                        wave_states = self._regenerate_leaking_wave(
                            prompts=[prompts[k] for k in leaking_indices],
                            initial_results=[
                                results[k] for k in leaking_indices
                            ],
                            initial_cots=[
                                batch_cot[k] for k in leaking_indices
                            ],
                            initial_answers=[
                                batch_answer[k] for k in leaking_indices
                            ],
                            sample_idxs=[
                                abs_sample_base + k for k in leaking_indices
                            ],
                        )
                        cot_wave_total_time += (
                            time.perf_counter() - cot_wave_start
                        )
                        for k, batch_idx in enumerate(leaking_indices):
                            batch_cot[batch_idx] = wave_states[k].best_cot
                            batch_answer[batch_idx] = (
                                wave_states[k].best_answer
                            )
                            cot_states_by_idx[batch_idx] = wave_states[k]

                    # Unified answer wave covers every row whose current
                    # ``batch_answer`` leaks — both Phase 2 leakers
                    # (post-CoT-wave) and clean-CoT samples (Phase 3).
                    # The wave method itself runs ``detect_batch`` to
                    # find the leakers, so no caller-side filtering is
                    # needed.  ``cot_states_by_idx`` lets the answer
                    # wave carry CoT-phase soft tokens forward for
                    # Phase 2 origin rows.
                    answer_wave_start = time.perf_counter()
                    answer_sample_idxs = [
                        abs_sample_base + k for k in range(len(prompts))
                    ]
                    self._regenerate_leaking_answer_wave(
                        prompts=prompts,
                        cots=batch_cot,
                        answers=batch_answer,
                        sample_idxs=answer_sample_idxs,
                        cot_states_by_idx=cot_states_by_idx,
                    )
                    answer_wave_total_time += (
                        time.perf_counter() - answer_wave_start
                    )

                    # Rebuild ``batch_responses[i]`` only for rows whose
                    # CoT or answer was actually mutated by either wave;
                    # untouched rows keep their initial-pass response
                    # verbatim (preserves the format of the original
                    # delimiter, which ``cot + THINK_SUFFIX + answer``
                    # would normalize even when nothing changed).
                    for batch_idx in range(len(prompts)):
                        if (
                            batch_cot[batch_idx] == cots_pre_wave[batch_idx]
                            and batch_answer[batch_idx]
                            == answers_pre_wave[batch_idx]
                        ):
                            continue
                        cot = batch_cot[batch_idx]
                        answer = batch_answer[batch_idx]
                        if answer:
                            batch_responses[batch_idx] = (
                                cot + self.THINK_SUFFIX + answer
                            )
                        else:
                            batch_responses[batch_idx] = cot

                all_gold_answers.append(gold_answers)
                all_gold_cots.append(gold_cots)
                all_generated_answers.append(batch_responses)
                all_generated_cot.append(batch_cot)
                all_generated_answer.append(batch_answer)
                all_prompts.append(prompts)
                abs_sample_base += len(prompts)
                _remove_hooks(self.model)

            subsets_generations[subset_name] = {
                "prompt": all_prompts,
                "gold_answer": all_gold_answers,
                "gold_cot": all_gold_cots,
                "generated": all_generated_answers,
                "generated_cot": all_generated_cot,
                "generated_answer": all_generated_answer,
            }

            # TEMPORARY: wave-engine wall-time summary.  Restore plain
            # logging (or remove) before committing.
            #
            # ``total_subset`` is the wall time for the whole subset loop
            # (initial gen + detection + wave + phase 3 + bookkeeping).
            # ``gen`` is the cumulative time across every model.generate /
            # inner.regenerate call (initial gen + wave-loop regens +
            # answer-phase regens).  ``detect`` is the cumulative time
            # across every leak_detector call (detect_batch + per-row
            # detect_continuation in the wave + answer-phase detect).
            # ``other`` (= total - gen - detect) covers tokenization,
            # decode, scheduling, state updates.
            total_subset = time.perf_counter() - subset_start_time
            other = total_subset - self._total_gen_time - self._total_detect_time
            log_print(
                "[wave-instr] subset={} examples={} "
                "total={:.2f}s gen={:.2f}s detect={:.2f}s other={:.2f}s | "
                "initial_gen={:.2f}s cot_wave={:.2f}s ({} CoT leaker rows) "
                "answer_wave={:.2f}s | "
                "cot_regen_gen={:.2f}s answer_regen_gen={:.2f}s | "
                "force_cleared_leaks={}".format(
                    subset_name,
                    total_examples,
                    total_subset,
                    self._total_gen_time,
                    self._total_detect_time,
                    other,
                    total_time,
                    cot_wave_total_time,
                    cot_wave_leaker_count,
                    answer_wave_total_time,
                    self._cot_regen_gen_time,
                    self._answer_regen_gen_time,
                    self._force_cleared_leak_total,
                )
            )

        self.tokenizer.padding_side = padding_side
        return subsets_generations

    # ------------------------------------------------------------------
    # Per-sample regeneration
    # ------------------------------------------------------------------

    def _regenerate_leaking_sample(
        self, prompt, initial_result, initial_cot, initial_answer, sample_idx
    ):
        """Run the detect-and-regenerate loop for a single leaking sample.

        DEPRECATED in production
        ------------------------
        This method is no longer reached by ``_generate``'s dispatch as
        of Part 4 (Phase 2 leakers go through
        :meth:`_regenerate_leaking_wave`; Phase 3 — clean-CoT,
        leaking-answer rows — and Phase 2 tail answer regen both go
        through :meth:`_regenerate_leaking_answer_wave`).  Retained in
        the codebase for the planned outlier-breakoff refactor (route
        a wave row whose remaining budget is much larger than its
        peers' through this single-sample path so it doesn't drag the
        slowest-row tax across the whole wave).  Modify with care:
        once outlier-breakoff lands, this body's CoT branch will be
        live again on a subset of rows.


        Maintains append-only ``prefix_sentences``, ``clean_prefix``, and
        ``prefix_ids`` state, together with a ``soft_token_stack`` of
        ``anchor_token_count`` entries.  Each stack entry
        stays bound to its original insertion point in the prefix:
        forward leak moves extend the prefix and add a new entry,
        and same-location escalation appends at the current anchor.

        State invariants (all three hold at every loop iteration)
        ---------------------------------------------------------
        * ``clean_prefix == " ".join(prefix_sentences)`` — the string
          form is the canonical rejoin of the sentence list with
          single-space separators.  Both are updated together on
          forward moves; neither is ever updated independently.
        * ``prefix_ids == tokenize(clean_prefix)`` approximately — the
          token list is maintained by appending ``tokenize(new_part)``
          for each new chunk rather than re-tokenizing the full prefix.
          This is the authoritative token-level state used by
          ``_build_regen_inputs``; anchor offsets in
          ``soft_token_stack`` are indices into ``prefix_ids``.
        * ``len(prefix_sentences) == first_leak_index`` — the sentence
          count tracks the anchor of the current leak boundary.

        Do NOT rebuild ``clean_prefix`` by ``" ".join(prefix_sentences)``
        mid-loop: while the append path preserves equality with the join,
        recomputing from Punkt-derived sentences would only coincidentally
        match the original text — Punkt strips inter-sentence whitespace,
        so ``" ".join(split_sentences(t)) != t`` whenever ``t`` has
        non-single-space separators between sentences.  Similarly, do
        NOT attempt to compute the newly-generated portion by slicing
        ``regen_cot[len(clean_prefix):]`` — ``_regenerate_sample``
        returns the raw new text directly for this reason.

        Leak re-detection uses
        :meth:`CoTLeakDetector.detect_continuation`, which only splits
        and classifies the newly-generated continuation.  Prefix
        sentences are not re-classified; their text does not change
        between iterations, so Punkt cannot re-segment the prefix region
        based on new context and a deterministic classifier cannot flip
        its decision on unchanged input.  This makes "leak in old
        prefix" structurally impossible.

        Termination guarantee
        ---------------------
        ``attempt`` resets to 0 only on a strict forward leak move — the
        append-only prefix grows, whose sentence count is bounded by
        Punkt splits of the generated continuation, itself bounded by
        ``max_new_tokens``.

        Every other path either breaks (no leak) or escalates
        (``attempt += 1``) which exits the inner loop at
        ``regen_max_attempts``.  ``detect_continuation`` reports
        ``first_leak_index >= len(prefix_sentences)`` for any confirmed
        leak, so backward moves into the prefix are impossible by
        construction.

        Upper bound per sample: roughly
        ``regen_max_attempts * (max_forward_moves + 1)`` iterations, which
        is a small constant in practice.

        Returns ``(best_cot, best_answer)`` — the last regeneration
        attempt, which may still leak if all attempts are exhausted but
        is always no worse than the original leaking output.

        When called with ``initial_result.is_leaking=False`` (e.g., from
        Phase 3 in ``_generate`` for samples with clean CoTs), the CoT
        regen loop is skipped entirely and only the answer leak check
        runs.
        """
        assert self.leak_detector is not None, (
            "_regenerate_leaking_sample called without a leak_detector"
        )

        # Per-sample state. Defaults cover the answer-only path (clean CoT,
        # no leak detected initially); the leak branch below overrides
        # them with the regen-loop's final state.
        state = RegenSampleState(
            best_cot=initial_cot,
            best_answer=initial_answer,
            clean_prefix=initial_cot or "",
            prefix_ids=(
                self.tokenizer(initial_cot, add_special_tokens=False)["input_ids"]
                if initial_cot
                else []
            ),
        )

        # NOTE: As of the wave-level engine refactor, Phase 2 (leaking-CoT
        # rows) goes through ``_regenerate_leaking_wave``.  The branch
        # below — the per-sample CoT regen loop — is now reachable only
        # via Phase 3's ``initial_result.is_leaking=False`` shortcut, so
        # in production this body is always skipped and only the
        # post-loop ``_run_answer_regen`` call runs.  The CoT branch is
        # retained as the simplest way to share Phase 3 plumbing with
        # the future answer-regen refactor; once that lands, the whole
        # method can collapse to a thin ``_run_answer_regen`` wrapper.
        if initial_result.is_leaking:
            first_leak_index = initial_result.first_leak_index
            state.first_leak_index = first_leak_index

            state.prefix_sentences = list(
                initial_result.sentences[:first_leak_index]
            )
            state.clean_prefix = " ".join(state.prefix_sentences)

            # prefix_ids is maintained in sync with clean_prefix, append-only,
            # so that token-count anchors are stable across iterations.
            state.prefix_ids = (
                self.tokenizer(state.clean_prefix, add_special_tokens=False)["input_ids"]
                if state.clean_prefix
                else []
            )

            state.soft_token_stack = []
            state.push_anchor(len(state.prefix_ids))

            while state.attempt < self.regen_max_attempts:
                regen_cot, regen_answer, new_continuation_text = self._regenerate_sample(
                    prompt,
                    state.clean_prefix,
                    state.attempt,
                    prefix_ids=state.prefix_ids,
                    soft_token_stack=state.soft_token_stack or None,
                )
                state.best_cot, state.best_answer = regen_cot, regen_answer
                state.new_continuation_text = new_continuation_text

                detect_start = time.perf_counter()
                # Legacy per-sample CoT regen: no external prior context
                # (CoT phase). Prompt threads through for contextual
                # classifiers; non-contextual ignore.
                new_result = self.leak_detector.detect_continuation(
                    state.prefix_sentences,
                    new_continuation_text,
                    prompt=prompt,
                )
                self._total_detect_time += time.perf_counter() - detect_start
                # Per-attempt detector state: how many sentences Stage 1
                # flagged vs how many Stage 2 confirmed, so we can tell
                # whether a leak slipped past Stage 1 entirely or was
                # dropped by Stage 2.
                stage1_count = sum(1 for f in new_result.stage1_flags if f)
                confirmed_count = sum(1 for f in new_result.confirmed_flags if f)
                log_print(
                    f"  Sample {sample_idx} attempt {state.attempt + 1}: "
                    f"stage1_flags={stage1_count}, "
                    f"confirmed={confirmed_count}, "
                    f"first_leak_index={new_result.first_leak_index}"
                )
                if not new_result.is_leaking:
                    log_print(
                        f"  Sample {sample_idx}: regeneration succeeded "
                        f"on attempt {state.attempt + 1}"
                    )
                    break

                new_first_leak = new_result.first_leak_index
                # ``detect_continuation``'s contract: ``first_leak_index`` is
                # absolute into ``prefix_sentences + new_sentences`` with
                # ``first_leak_index >= len(prefix_sentences)``.  Anything
                # else would indicate a detector bug.
                assert new_first_leak is not None
                assert new_first_leak >= len(state.prefix_sentences), (
                    f"detect_continuation returned first_leak_index="
                    f"{new_first_leak} < n_prefix="
                    f"{len(state.prefix_sentences)} — this violates "
                    f"its contract."
                )

                if new_first_leak > len(state.prefix_sentences):
                    newly_cleared = list(
                        new_result.sentences[
                            len(state.prefix_sentences):new_first_leak
                        ]
                    )
                    newly_cleared_text = " ".join(newly_cleared)
                    if state.clean_prefix:
                        new_part = " " + newly_cleared_text
                    else:
                        new_part = newly_cleared_text
                    new_part_ids = self.tokenizer(
                        new_part, add_special_tokens=False
                    )["input_ids"]
                    state.prefix_sentences.extend(newly_cleared)
                    state.clean_prefix = state.clean_prefix + new_part
                    state.prefix_ids = state.prefix_ids + new_part_ids
                    state.first_leak_index = new_first_leak
                    state.push_anchor(len(state.prefix_ids))
                    state.attempt = 0
                else:
                    state.push_anchor(len(state.prefix_ids))
                    state.attempt += 1
            else:
                log_print(
                    f"  Sample {sample_idx}: regeneration exhausted "
                    f"{self.regen_max_attempts} attempts"
                )

        # Answer-phase regen (per-sample). The wave-level CoT loop
        # converges first; this runs on a single state's final answer.
        self._run_answer_regen(state, prompt, sample_idx)

        return state.best_cot, state.best_answer

    def _run_answer_regen(
        self,
        state: RegenSampleState,
        prompt: str,
        sample_idx: int,
    ) -> None:
        """Run answer-phase regeneration on a single sample's state.

        DEPRECATED in production
        ------------------------
        Replaced in production by :meth:`_regenerate_leaking_answer_wave`
        in Part 4 + Part 4b.  This method's locked-anchor design — all
        answer-regen attempts stack soft tokens at end-of-CoT,
        regardless of where the leak actually occurred inside the
        answer — is the pathology the answer wave was introduced to
        fix.  Retained here only as the answer-side counterpart of
        the kept-for-future :meth:`_regenerate_leaking_sample`; if
        outlier-breakoff routing ever needs to hand a single row off
        to the per-sample path, this will run again.

        Carryover semantics MATCH the answer wave (since Part 4b):
        both this method and the wave inherit the CoT-phase soft-token
        stack into the answer regen call.  The only methodology
        difference vs the wave is anchor PLACEMENT — this method
        appends new pads at end-of-CoT (one fixed answer anchor); the
        wave anchors them at the actual answer leak position and
        supports forward extension.


        Mutates ``state.best_answer`` in place.  Detects whether the
        current answer leaks; if so, regenerates with soft tokens placed
        immediately after the ``</think>\\n\\n`` delimiter.  Each failed
        attempt stacks one more block at that same anchor.

        Runs for both leaking and clean CoTs.  For leaking CoTs, the
        state fields (``prefix_ids``, ``soft_token_stack``, ``clean_prefix``,
        ``new_continuation_text``) reflect the final CoT regen state.  For clean
        CoTs, they hold defaults set at state construction (full CoT
        tokenized, no pads).

        Builds "prefix" = full CoT + ``</think>\\n\\n`` for
        :meth:`_regenerate_sample`.  CoT soft-token entries keep their
        original anchors; answer entries get anchors past the end so
        :meth:`_insert_soft_token_pads` appends them after the delimiter.

        Per-sample for v1 — kept this way so the upcoming answer-regen
        refactor (which will mirror CoT regen's structure) can replace
        this body wholesale without touching the wave loop.
        """
        assert self.leak_detector is not None, (
            "_run_answer_regen called without a leak_detector"
        )
        if not state.best_answer:
            return
        # Answer-phase context for contextual classifiers spans the
        # CoT→answer seam: the full CoT's sentences sit "before" the
        # answer, so the contextual k-window walks across them just
        # like the wave-engine answer wave does.
        cot_external = split_sentences(state.best_cot) if state.best_cot else []
        detect_start = time.perf_counter()
        answer_result = self.leak_detector.detect(
            state.best_answer,
            prompt=prompt,
            external_prefix_sentences=cot_external,
        )
        self._total_detect_time += time.perf_counter() - detect_start
        if not answer_result.is_leaking:
            return

        # One soft-token block at the answer anchor to start; each
        # failed attempt below stacks another at the same anchor.
        n_answer_anchors = 1

        log_print(f"  Sample {sample_idx}: answer leak detected")

        # Continuation: mirrors _regenerate_sample's join logic.
        if state.clean_prefix and state.new_continuation_text:
            continuation = " " + state.new_continuation_text
        elif state.new_continuation_text:
            continuation = state.new_continuation_text
        else:
            continuation = ""

        if continuation:
            continuation_ids = self.tokenizer(
                continuation, add_special_tokens=False
            )["input_ids"]
        else:
            continuation_ids = []

        if state.prefix_ids or continuation_ids:
            cot_ids = list(state.prefix_ids) + list(continuation_ids)
        else:
            cot_ids = (
                self.tokenizer(
                    state.best_cot, add_special_tokens=False
                )["input_ids"]
                if state.best_cot else []
            )

        think_suffix_ids = self.tokenizer(
            self.THINK_SUFFIX, add_special_tokens=False
        )["input_ids"]
        full_prefix_ids = cot_ids + list(think_suffix_ids)
        answer_anchor = len(full_prefix_ids)

        for ans_attempt in range(self.regen_max_attempts):
            combined_stack = list(state.soft_token_stack) + (
                [answer_anchor] * n_answer_anchors
            )
            text, _, _ = self._regenerate_sample(
                prompt=prompt,
                clean_prefix="",
                attempt=0,
                prefix_ids=full_prefix_ids,
                soft_token_stack=combined_stack,
                return_raw=True,
            )
            # TEMPORARY (batch-regen-timing): credit answer-regen via the
            # _last_regen_gen_time stashed by _regenerate_sample.
            self._answer_regen_gen_time += getattr(
                self, "_last_regen_gen_time", 0.0
            )
            # KNOWN BUG (parity with the wave-loop's pre-fix over-budget
            # branch): when ``_regenerate_sample`` short-circuits on
            # ``remaining_budget <= 0`` (line ~1797) it returns
            # ``("", "", "")`` in ``return_raw=True`` mode.  Assigning
            # that empty ``text`` here clobbers the prior iteration's
            # ``state.best_answer`` — including any cleared answer
            # content the per-sample loop had built up.  The wave path
            # (``_run_wave_loop``) preserves ``state.clean_prefix`` on
            # its over-budget drop; this per-sample path does not have
            # an equivalent guard and would silently emit ``""`` on the
            # same condition.  Unreached in production today (Phase 2
            # leakers go through the wave; Phase 3 answer regen also
            # goes through the wave); revisit if the planned
            # outlier-breakoff refactor routes a single row back through
            # this method.
            state.best_answer = text

            detect_start = time.perf_counter()
            answer_result = self.leak_detector.detect(
                text,
                prompt=prompt,
                external_prefix_sentences=cot_external,
            )
            self._total_detect_time += time.perf_counter() - detect_start
            stage1_count = sum(
                1 for f in answer_result.stage1_flags if f
            )
            confirmed_count = sum(
                1 for f in answer_result.confirmed_flags if f
            )
            log_print(
                f"  Sample {sample_idx} answer attempt "
                f"{ans_attempt + 1}: "
                f"stage1_flags={stage1_count}, "
                f"confirmed={confirmed_count}"
            )

            if not answer_result.is_leaking:
                log_print(
                    f"  Sample {sample_idx}: answer regen "
                    f"succeeded on attempt {ans_attempt + 1}"
                )
                break

            n_answer_anchors += 1
        else:
            log_print(
                f"  Sample {sample_idx}: answer regen exhausted "
                f"{self.regen_max_attempts} attempts"
            )

    # ------------------------------------------------------------------
    # Wave-level regeneration: shared loop body + per-phase decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _join_clean_prefix(clean_prefix: str, new_text: str) -> str:
        """Join a non-empty ``clean_prefix`` to ``new_text`` with a space
        separator, omitting the separator when either side is empty.

        Avoids trailing whitespace when the regen produced no new text
        (``new_text == ""``, e.g. over-budget drop) — ``"prefix " + ""``
        used to leave a stray space on the assembled output.
        """
        if clean_prefix and new_text:
            return clean_prefix + " " + new_text
        return clean_prefix or new_text

    def _assemble_cot_iteration_output(
        self, state: RegenSampleState, raw_text: str
    ) -> str:
        """CoT-phase decode-and-assemble for ``_run_wave_loop``.

        Splits the raw decoded text at ``</think>\\n\\n`` to separate the
        new CoT continuation from the answer portion.  Sets
        ``state.best_cot`` (clean prefix + new CoT), ``state.best_answer``
        (the answer post-delimiter), and ``state.new_continuation_text``
        (the new CoT text alone, used by the kept-for-future
        ``_run_answer_regen`` to rebuild ``cot_ids``).  Returns the new
        CoT text so the wave loop can pass it to ``detect_continuation``.
        """
        new_continuation_text, answer = split_cot_answer(raw_text)
        state.best_cot = self._join_clean_prefix(
            state.clean_prefix, new_continuation_text
        )
        state.best_answer = answer
        state.new_continuation_text = new_continuation_text
        return new_continuation_text

    def _assemble_answer_iteration_output(
        self, state: RegenSampleState, raw_text: str
    ) -> str:
        """Answer-phase decode-and-assemble for ``_run_wave_loop``.

        The model input already contained ``</think>\\n\\n`` (in
        ``frozen_prefix_ids``), so the raw decoded text IS the new
        answer continuation — no split needed.  Sets ``state.best_answer``
        (clean prefix + new continuation) and ``state.new_continuation_text``
        (the new continuation alone).  ``state.best_cot`` stays frozen
        for the answer phase.  Returns the new continuation so the wave
        loop can pass it to ``detect_continuation``.
        """
        state.best_answer = self._join_clean_prefix(
            state.clean_prefix, raw_text
        )
        state.new_continuation_text = raw_text
        return raw_text

    def _run_wave_loop(
        self,
        prompts: list[str],
        states: list[RegenSampleState],
        sample_idxs: list[int],
        *,
        phase_label: str,
        gen_time_attr: str,
        assemble_fn,
    ) -> None:
        """Shared batched detect-and-regenerate loop body.

        Both the CoT wave (:meth:`_regenerate_leaking_wave`) and the
        answer wave (:meth:`_regenerate_leaking_answer_wave`) call into
        this method with phase-specific:

        * ``phase_label`` — short tag (``"cot"`` or ``"answer"``) prepended
          to ``[wave-iter*]`` and per-attempt log lines so events from
          the two phases are distinguishable.
        * ``gen_time_attr`` — name of the engine attribute to credit
          ``inner.regenerate`` wall time to (``"_cot_regen_gen_time"``
          or ``"_answer_regen_gen_time"``).  Stringly-typed because the
          attributes themselves are part of the TEMPORARY batch-regen
          instrumentation; this parameter and the per-phase attributes
          all go away when that instrumentation is finally stripped.
        * ``assemble_fn(state, raw_text) -> str`` — decode-and-assemble
          callback; mutates per-row outputs and returns the new
          continuation text for ``detect_continuation``.

        ``states`` must already be initialized — including
        ``frozen_prefix_ids`` (empty in CoT phase, ``cot_ids + delim_ids``
        in answer phase), ``prefix_sentences``/``clean_prefix``/``prefix_ids``
        (the cleared classified prefix), and the initial soft-token
        anchor pushed via ``state.push_anchor``.

        Side effects: mutates each ``state`` in place; advances
        ``self._total_gen_time``, ``self._total_detect_time``, and the
        attribute named by ``gen_time_attr``.

        State machine per row, for each iteration:
          * Forward extension (``new_first_leak > len(prefix_sentences)``):
            extend the cleared prefix, push a new anchor, reset
            ``state.attempt = 0``, keep the row active.
          * Same-anchor escalation, attempts NOT yet exhausted
            (``new_first_leak == len(prefix_sentences)`` and
            ``state.attempt + 1 < regen_max_attempts``): push another
            anchor at the current position, increment ``state.attempt``,
            keep the row active for retry.
          * Same-anchor escalation, attempts EXHAUSTED: force-clear
            the persistent leak (and any post-leak clean intermediates
            already confirmed clean by the same iteration's per-
            sentence classification) into the prefix.  Then:
              - if a further leak exists in this iteration's output,
                push a fresh anchor at the post-absorption prefix end,
                reset ``state.attempt = 0``, keep the row active;
              - if no further leak exists, the row's current
                ``best_cot`` / ``best_answer`` (already finalized at
                this iteration's ``assemble_fn``) IS the original
                output (persistent leak preserved verbatim, post-leak
                tail clean per the classifier) — drop the row from
                ``next_active`` without state mutation.
            The exhaust path does NOT push a wasted same-anchor pad
            before this branch; the pre-exhaust escalation push is
            gated on ``state.attempt + 1 < regen_max_attempts``.
            ``state.n_force_cleared_leaks`` and the engine-level
            ``self._force_cleared_leak_total`` accumulator increment
            on every absorption so the per-subset summary surfaces how
            often the loop preserved a known-leaking sentence.
          * Convergence (``not new_result.is_leaking``): row drops out.
          * Over-budget (``remaining_budget <= 0`` at the top of the
            next iteration): row drops out preserving its phase's
            cleared work — CoT phase sets ``best_cot = clean_prefix``
            (and ``best_answer = ""`` since ``</think>`` was never
            reached), answer phase sets ``best_answer = clean_prefix``
            (and leaves ``best_cot`` untouched, since it's frozen on
            init).  Distinct from the force-clear branch: this fires
            when the row has generated past its token budget, not when
            same-anchor escalation reached ``regen_max_attempts``.

        Termination: each force-clear strictly grows
        ``prefix_sentences`` by at least one (the persistent-leak
        sentence), so the upper bound stays
        ``O(regen_max_attempts × max_forward_moves)`` where
        ``max_forward_moves`` is bounded by Punkt splits of generated
        text, itself bounded by ``max_new_tokens``.  The over-budget
        guard at top-of-loop catches the residual case where
        force-clear absorption pushes ``prefix_ids`` past the budget.

        ``_build_regen_inputs`` handles ``frozen_prefix_ids``'s
        absolute-position offsetting and the budget arithmetic
        ``total_budget - len(prefix_ids) - len(frozen_prefix_ids)`` is
        applied uniformly across phases.
        """
        if not states:
            return
        active_indices = list(range(len(states)))

        total_budget = self.model.generation_config.max_new_tokens
        assert total_budget is not None, (
            "RegeneratingReasoningEngine requires "
            "generation_config.max_new_tokens to be set so each regen call "
            "can be budget-capped."
        )
        think_ids = self.tokenizer(
            ReasoningModel.THINK_PREFIX, add_special_tokens=False
        )["input_ids"]
        inner = self.model._inner
        # TEMPORARY (batch-regen-timing): per-iteration counter so the
        # [wave-iter] / [wave-iter-eos] log lines can be correlated.
        iter_n = 0

        while active_indices:
            # Drop rows that have no remaining budget (forward extension
            # can shrink remaining_budget below 0 mid-wave).  Each phase
            # promotes its own ``clean_prefix`` to the row's final output
            # so the cleared work survives — the CoT phase writes
            # ``best_cot``, the answer phase writes ``best_answer``.
            # ``new_continuation_text`` is per-iteration scratch and is
            # cleared so a subsequent over-budget row doesn't carry stale
            # continuation state.
            within_budget: list[int] = []
            for i in active_indices:
                # ``remaining_budget`` excludes both the classified prefix
                # and the frozen baseline (cot+delim in answer phase,
                # empty in CoT) — both already count against
                # ``max_new_tokens``.  Soft-token pads are excluded
                # because they are injected, not generated.
                if states[i].remaining_budget(total_budget) > 0:
                    within_budget.append(i)
                else:
                    if self.instrumentation_verbose:
                        committed = (
                            len(states[i].prefix_ids)
                            + len(states[i].frozen_prefix_ids)
                        )
                        log_print(
                            f"  [{phase_label}] Sample {sample_idxs[i]}: "
                            f"regeneration dropped — token budget exceeded "
                            f"(committed={committed}, "
                            f"total_budget={total_budget})"
                        )
                    if phase_label == "cot":
                        # CoT phase: cleared CoT survives; ``best_answer``
                        # is empty because we never reached ``</think>``.
                        states[i].best_cot = states[i].clean_prefix
                        states[i].best_answer = ""
                    else:
                        # Answer phase: cleared answer prefix survives.
                        # ``best_cot`` is frozen on init from ``cots[i]``
                        # and must not be touched here.  Forward extensions
                        # may have accumulated many cleared answer
                        # sentences before the over-budget drop; clobbering
                        # ``best_answer`` would discard all of them.
                        states[i].best_answer = states[i].clean_prefix
                    states[i].new_continuation_text = ""
            active_indices = within_budget
            if not active_indices:
                break

            # TEMPORARY (batch-regen-timing): per-row remaining-budget log
            # so we can see how spread the budgets are at each iteration —
            # the spread is the leading suspect for the wave's slowest-row
            # tax (since max_new_tokens = max(per_row_remaining)).
            if self.instrumentation_verbose:
                log_print(
                    "[wave-iter {}] iter={} active={} remaining=[{}]".format(
                        phase_label,
                        iter_n,
                        len(active_indices),
                        ", ".join(
                            f"s{sample_idxs[i]}:"
                            f"{states[i].remaining_budget(total_budget)}"
                            for i in active_indices
                        ),
                    )
                )

            active_states = [states[i] for i in active_indices]
            active_prompts = [prompts[i] for i in active_indices]
            prompt_ids_per_row = [
                self.tokenizer(p, add_special_tokens=True)["input_ids"]
                for p in active_prompts
            ]
            per_row_remaining = [
                s.remaining_budget(total_budget) for s in active_states
            ]
            max_remaining = max(per_row_remaining)

            (
                input_ids,
                attention_mask,
                st_positions,
            ) = self._build_regen_inputs(
                prompt_ids_per_row=prompt_ids_per_row,
                think_ids=think_ids,
                states=active_states,
            )

            gen_config = copy.copy(self.model.generation_config)
            gen_config.max_new_tokens = max_remaining

            gen_start = time.perf_counter()
            # When ``self.chunk_size`` is set, the chunked re-prefill
            # path drops EOS'd / over-budget rows between chunks so
            # non-terminating slow-tail rows don't hold the batch open.
            # ``per_row_remaining`` is forwarded so each row caps its
            # own new-token count instead of all rows truncating to
            # ``max_remaining`` post-decode.
            generated = inner.regenerate(
                soft_token_positions=st_positions,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                chunk_size=self.chunk_size,
                per_row_remaining=(
                    per_row_remaining if self.chunk_size is not None else None
                ),
                chunk_mode=self.chunk_mode,
            )
            gen_dt = time.perf_counter() - gen_start
            self._total_gen_time += gen_dt
            # TEMPORARY (batch-regen-timing): credit per-phase gen time
            # so we can compare CoT-regen vs answer-regen totals.
            setattr(
                self, gen_time_attr, getattr(self, gen_time_attr) + gen_dt
            )
            decode_start = input_ids.shape[1]

            # TEMPORARY (batch-regen-timing): per-row EOS positions so we
            # can compute max/avg of "useful generation length" within the
            # iteration — the EOS spread is the slowest-row tax in numbers.
            # The eos_positions scan + log are both gated on
            # ``instrumentation_verbose`` since neither feeds correctness.
            if self.instrumentation_verbose:
                eos_id = self.tokenizer.eos_token_id
                eos_positions: list[int] = []
                for k in range(generated.shape[0]):
                    row_after_prompt = generated[k, decode_start:]
                    eos_hits = (
                        row_after_prompt == eos_id
                    ).nonzero(as_tuple=True)[0]
                    if len(eos_hits) > 0:
                        eos_positions.append(int(eos_hits[0].item()))
                    else:
                        eos_positions.append(per_row_remaining[k])
                log_print(
                    "[wave-iter-eos {}] iter={} max_new={} gen_dt={:.2f}s "
                    "eos=[{}]".format(
                        phase_label,
                        iter_n,
                        max_remaining,
                        gen_dt,
                        ", ".join(
                            f"s{sample_idxs[active_indices[k]]}:"
                            f"{eos_positions[k]}/{per_row_remaining[k]}"
                            for k in range(generated.shape[0])
                        ),
                    )
                )

            next_active: list[int] = []
            for k, i in enumerate(active_indices):
                state = states[i]
                sample_idx = sample_idxs[i]

                # Truncate per-row generated tokens to that row's
                # individual remaining-budget cap.  ``max_new_tokens``
                # for the batched call was the wave's max; shorter-budget
                # rows would otherwise overshoot their cap.
                #
                # If a row hit EOS before ``max_new_tokens`` and HF
                # filled the rest of the row with ``pad_token_id``, the
                # decode below relies on ``skip_special_tokens=True``
                # stripping those trailing pads.  This holds for all
                # canonical R-TOFU runs (LRM-target's tokenizer has
                # ``pad_token_id == eos_token_id``, both registered as
                # special).  A non-LLaMA tokenizer with non-special pad
                # would decode trailing pads as text — flag for follow-up
                # if the engine ever needs to support such tokenizers.
                row_gen = generated[k][
                    decode_start:decode_start + per_row_remaining[k]
                ]
                text = self.tokenizer.decode(
                    row_gen, skip_special_tokens=True
                )
                # Phase-specific decode + state assembly.  Sets
                # ``state.best_cot`` / ``state.best_answer`` / ``state.new_continuation_text``
                # appropriately for the phase, and returns the new
                # continuation text for ``detect_continuation``.
                new_continuation_text = assemble_fn(state, text)

                detect_start = time.perf_counter()
                # Always pass the external prefix as a (possibly empty)
                # list — uniform shape across CoT wave (empty) and answer
                # wave (CoT-tail sentences) avoids a special-case branch
                # at the call site. The detector treats empty and None
                # identically, but consistent shape reads more cleanly.
                new_result = self.leak_detector.detect_continuation(
                    state.prefix_sentences,
                    new_continuation_text,
                    prompt=prompts[i],
                    external_prefix_sentences=list(state.context_prefix_sentences),
                )
                self._total_detect_time += time.perf_counter() - detect_start
                if self.instrumentation_verbose:
                    stage1_count = sum(
                        1 for f in new_result.stage1_flags if f
                    )
                    confirmed_count = sum(
                        1 for f in new_result.confirmed_flags if f
                    )
                    log_print(
                        f"  [{phase_label}] Sample {sample_idx} attempt "
                        f"{state.attempt + 1}: "
                        f"stage1_flags={stage1_count}, "
                        f"confirmed={confirmed_count}, "
                        f"first_leak_index={new_result.first_leak_index}"
                    )

                if not new_result.is_leaking:
                    if self.instrumentation_verbose:
                        log_print(
                            f"  [{phase_label}] Sample {sample_idx}: "
                            f"regeneration succeeded on attempt "
                            f"{state.attempt + 1}"
                        )
                    continue  # row done; not in next_active

                new_first_leak = new_result.first_leak_index
                assert new_first_leak is not None
                assert new_first_leak >= len(state.prefix_sentences), (
                    f"detect_continuation returned first_leak_index="
                    f"{new_first_leak} < n_prefix="
                    f"{len(state.prefix_sentences)} — this violates "
                    f"its contract."
                )

                # Case 1: forward extension
                if new_first_leak > len(state.prefix_sentences):
                    newly_cleared = list(
                        new_result.sentences[
                            len(state.prefix_sentences):new_first_leak
                        ]
                    )
                    newly_cleared_text = " ".join(newly_cleared)
                    if state.clean_prefix:
                        new_part = " " + newly_cleared_text
                    else:
                        new_part = newly_cleared_text
                    new_part_ids = self.tokenizer(
                        new_part, add_special_tokens=False
                    )["input_ids"]
                    state.prefix_sentences.extend(newly_cleared)
                    state.clean_prefix = state.clean_prefix + new_part
                    state.prefix_ids = state.prefix_ids + new_part_ids
                    state.first_leak_index = new_first_leak
                    state.push_anchor(len(state.prefix_ids))
                    state.attempt = 0
                    next_active.append(i)
                # Case 2: same-anchor escalation.
                #
                # Two sub-cases:
                #   (a) attempts not yet exhausted — push another pad at
                #       the current prefix end, increment attempt, retry.
                #   (b) attempts exhausted — force-clear the persistent
                #       leak (and any clean intermediates) into the
                #       prefix and re-anchor at the next leak.  Skips
                #       the (a) push so we don't add a wasted pad at a
                #       position the next iteration's prefix has grown
                #       past.
                else:
                    if state.attempt + 1 < self.regen_max_attempts:
                        state.push_anchor(len(state.prefix_ids))
                        state.attempt += 1
                        next_active.append(i)
                        continue

                    # Sub-case (b): exhausted.  Walk new_result.confirmed_flags
                    # past the persistent leak (which sits at the very
                    # first new sentence by virtue of the same-anchor
                    # branch) to find the next leak.  ``classifier.classify_cot``
                    # already classified every new sentence in one
                    # batched forward pass; the confirmed-flags list
                    # lets us reuse that work without re-running
                    # detection.
                    n_prefix = len(state.prefix_sentences)
                    cf = new_result.confirmed_flags
                    next_leak_abs_idx: int | None = None
                    # Search starts at n_prefix + 1 — the persistent
                    # leak (cf[n_prefix]=True) is being force-cleared,
                    # so its own slot is the wrong target for the new
                    # anchor.  ``range`` walk avoids the slice
                    # ``cf[n_prefix + 1:]`` allocating an intermediate
                    # list — small but the plan called this out.
                    for abs_idx in range(n_prefix + 1, len(cf)):
                        if cf[abs_idx]:
                            next_leak_abs_idx = abs_idx
                            break

                    if next_leak_abs_idx is None:
                        # All post-persistent-leak content was clean per
                        # the original classification.  ``best_cot`` /
                        # ``best_answer`` were finalized by ``assemble_fn``
                        # at the top of this iteration as
                        # ``state.clean_prefix + new_continuation_text``,
                        # which is exactly the original output (persistent
                        # leak preserved, post-leak content unchanged) —
                        # no further state mutation needed.  Drop the
                        # row from next_active.
                        if self.instrumentation_verbose:
                            log_print(
                                f"  [{phase_label}] Sample {sample_idx}: "
                                f"regeneration exhausted at anchor; "
                                f"post-leak tail clean — preserving "
                                f"persistent leak in output"
                            )
                        continue

                    # Absorb the persistent leak + any clean
                    # intermediates into the prefix in a single
                    # tokenize call (mirrors forward extension's
                    # BPE-stable pattern at line ~1395).  The slice
                    # ``[n_prefix:next_leak_abs_idx]`` covers the
                    # persistent leak and every clean sentence
                    # before the next leak.
                    absorbed = list(
                        new_result.sentences[n_prefix:next_leak_abs_idx]
                    )
                    absorbed_text = " ".join(absorbed)
                    if state.clean_prefix:
                        new_part = " " + absorbed_text
                    else:
                        new_part = absorbed_text
                    new_part_ids = self.tokenizer(
                        new_part, add_special_tokens=False
                    )["input_ids"]
                    state.prefix_sentences.extend(absorbed)
                    state.clean_prefix = state.clean_prefix + new_part
                    state.prefix_ids = state.prefix_ids + new_part_ids
                    state.first_leak_index = next_leak_abs_idx
                    state.n_force_cleared_leaks += 1
                    self._force_cleared_leak_total += 1
                    # Re-anchor at the post-absorption prefix end, the
                    # position the next leak now sits immediately
                    # after.  Mirrors forward extension's push.
                    state.push_anchor(len(state.prefix_ids))
                    state.attempt = 0
                    next_active.append(i)
                    if self.instrumentation_verbose:
                        log_print(
                            f"  [{phase_label}] Sample {sample_idx}: "
                            f"regeneration exhausted at anchor — "
                            f"force-clearing {len(absorbed)} sentence(s) "
                            f"and re-anchoring at next leak "
                            f"(first_leak_index={next_leak_abs_idx})"
                        )

            active_indices = next_active
            # TEMPORARY (batch-regen-timing).
            iter_n += 1

    # ------------------------------------------------------------------
    # Wave-level regeneration (multiple leaking samples per call)
    # ------------------------------------------------------------------

    def _regenerate_leaking_wave(
        self,
        prompts: list[str],
        initial_results: list,
        initial_cots: list[str],
        initial_answers: list[str],
        sample_idxs: list[int],
    ) -> list[RegenSampleState]:
        """Run CoT-phase detect-and-regenerate on a wave of leaking
        samples in one batched call.

        All inputs are per-row (one entry per sample in the wave); every
        ``initial_results[i]`` must be ``is_leaking=True`` (the caller
        filters non-leaking rows out).

        Pipeline
        --------
        1. **Initialize per-row state** from each row's initial detection
           result (cleared prefix, soft-token stack with the first
           anchor).  ``frozen_prefix_ids`` is empty for the CoT phase.
        2. **Wave loop**: delegates to :meth:`_run_wave_loop` with the
           CoT-phase decode-and-assemble callback.  The wave loop builds
           one batched ``inner.regenerate()`` call per iteration,
           per-row decodes via ``split_cot_answer``, runs
           ``detect_continuation``, and advances each row through
           forward-extension / same-anchor-escalation / convergence /
           over-budget drop until either all rows converge or
           ``regen_max_attempts`` is reached.

        Answer regen is NOT run here; the caller (``_generate``) follows
        up with :meth:`_regenerate_leaking_answer_wave` over every row
        whose final answer leaks (covering both CoT-leakers handled here
        and clean-CoT samples that never entered this method).

        Per-row generation budget
        -------------------------
        Each row's ``max_new_tokens`` budget is
        ``total_budget - len(state.prefix_ids) - len(state.frozen_prefix_ids)``.
        The batched call uses ``max(per_row_remaining)`` and post-decode
        truncation ensures each row's accepted tokens fit its individual
        cap.  Rows whose ``remaining_budget <= 0`` (forward extension
        consumed the budget mid-wave) drop out of ``active_indices`` at
        the top of the next iteration.

        Returns
        -------
        list[RegenSampleState]
            One state per row, in input order.  ``state.best_cot`` and
            ``state.best_answer`` carry the final CoT and the answer the
            initial generation produced past ``</think>``; the answer
            wave (run separately by the caller) may further mutate
            ``best_answer``.
        """
        assert self.leak_detector is not None, (
            "_regenerate_leaking_wave called without a leak_detector"
        )
        n = len(prompts)
        if not (
            len(initial_results) == n and len(initial_cots) == n
            and len(initial_answers) == n and len(sample_idxs) == n
        ):
            raise ValueError(
                "_regenerate_leaking_wave inputs must all have equal length"
            )
        if n == 0:
            return []
        for i, result in enumerate(initial_results):
            if not result.is_leaking:
                raise ValueError(
                    f"_regenerate_leaking_wave called with non-leaking "
                    f"initial_result at row {i}; caller must filter to "
                    f"leaking rows only (Phase 3 path handles non-leaking)."
                )

        # ---- Init per-row state ------------------------------------------
        states: list[RegenSampleState] = []
        active_indices: list[int] = []
        for i in range(n):
            result = initial_results[i]
            first_leak_index = result.first_leak_index
            state = RegenSampleState(
                best_cot=initial_cots[i],
                best_answer=initial_answers[i],
                first_leak_index=first_leak_index,
                # prefix defaults set below based on first_leak_index
                clean_prefix=initial_cots[i] or "",
                prefix_ids=(
                    self.tokenizer(
                        initial_cots[i], add_special_tokens=False
                    )["input_ids"]
                    if initial_cots[i] else []
                ),
            )

            state.prefix_sentences = list(
                result.sentences[:first_leak_index]
            )
            state.clean_prefix = " ".join(state.prefix_sentences)
            state.prefix_ids = (
                self.tokenizer(
                    state.clean_prefix, add_special_tokens=False
                )["input_ids"]
                if state.clean_prefix else []
            )

            state.soft_token_stack = []
            state.push_anchor(len(state.prefix_ids))
            states.append(state)
            active_indices.append(i)

        # ---- Wave loop ---------------------------------------------------
        # Delegates the full per-iteration body (compaction, batched
        # dispatch, decode, detect, state-machine advance) to the shared
        # ``_run_wave_loop`` helper.  CoT-phase decode-and-assemble is
        # ``_assemble_cot_iteration_output`` (splits at ``</think>`` and
        # populates both ``best_cot`` and ``best_answer``).  Answer regen
        # for these rows is handled separately by
        # ``_regenerate_leaking_answer_wave`` in the unified dispatch —
        # this method no longer runs per-sample answer regen at the tail.
        self._run_wave_loop(
            prompts=prompts,
            states=states,
            sample_idxs=sample_idxs,
            phase_label="cot",
            gen_time_attr="_cot_regen_gen_time",
            assemble_fn=self._assemble_cot_iteration_output,
        )
        return states

    # ------------------------------------------------------------------
    # Answer-phase wave (mirrors CoT regen's structure)
    # ------------------------------------------------------------------

    def _regenerate_leaking_answer_wave(
        self,
        prompts: list[str],
        cots: list[str],
        answers: list[str],
        sample_idxs: list[int],
        cot_states_by_idx: "dict[int, RegenSampleState] | None" = None,
    ) -> None:
        """Run answer-phase detect-and-regenerate on every row whose
        current answer leaks.  Mutates ``answers`` in place.

        Methodology vs the legacy ``_run_answer_regen``
        ----------------------------------------------
        The legacy answer regen had a single fixed soft-token anchor
        at end-of-CoT (right after the ``</think>\\n\\n`` delimiter).
        Stacking new pads at the same end-of-CoT position regardless
        of where in the answer the leak occurred — the so-called
        "stack tokens far from the leak point" pathology — routinely
        failed to clear leaks deep inside the answer.

        The answer wave instead:
          * Anchors new soft tokens at the actual answer leak position
            inside ``prefix_ids`` (not at end-of-CoT).
          * Allows forward extension and same-anchor escalation through
            the same machinery as the CoT phase (via :meth:`_run_wave_loop`).
          * **Carries the CoT-phase soft-token stack forward** when a
            row's CoT went through the CoT wave: the CoT pads stay at
            their original positions inside the cot portion of the
            model input via ``state.frozen_soft_token_stack``, and the
            new answer-leak pad is added on top via the active
            ``soft_token_stack``.  Phase 3 origin rows (clean CoT, no
            CoT wave) get an empty ``frozen_soft_token_stack`` since
            there are no CoT pads to carry.

        AFE / CFE shifts vs the legacy answer regen are expected
        (different anchor placement) but the carryover semantic
        matches.

        This also unifies what used to be two separate code paths:
          * Phase 2 tail: per-sample answer regen on rows whose CoT
            just went through the wave.
          * Phase 3: per-sample answer regen on rows whose CoT was
            initially clean but whose answer leaks.

        Both paths share a single batched answer wave here, with their
        only difference being whether a non-empty
        ``frozen_soft_token_stack`` carries over from the CoT wave.

        Parameters
        ----------
        cot_states_by_idx
            Mapping ``batch_idx -> RegenSampleState`` for rows whose
            CoT went through :meth:`_regenerate_leaking_wave`.  Rows
            absent from this mapping are treated as Phase 3 origin
            (clean CoT; no CoT pads to carry).  ``None`` is equivalent
            to an empty mapping.

        Pipeline
        --------
        1. ``detect_batch(answers)`` flags every row whose answer leaks.
           Empty answers and clean answers drop out immediately.
        2. For each leaking row, build a fresh :class:`RegenSampleState`:
             * **Phase 2 origin** (``cot_state`` present):
               ``frozen_prefix_ids = tuple(cot_state.prefix_ids +
               post-converged_continuation_ids + delim_ids)``;
               ``frozen_soft_token_stack = tuple(cot_state.soft_token_stack)``.
               The CoT-phase anchors stay at their numerical positions
               inside ``cot_state.prefix_ids``, which is a prefix of
               the new ``frozen_prefix_ids`` — so they remain valid.
             * **Phase 3 origin** (``cot_state`` absent):
               ``frozen_prefix_ids = tuple(tokenize(cots[i]) + delim_ids)``;
               ``frozen_soft_token_stack = ()``.
             * ``prefix_sentences`` = sentences of the answer up to (not
               including) the first leaking sentence.
             * ``prefix_ids`` = tokens of those answer-prefix sentences.
             * ``soft_token_stack`` = ``[len(prefix_ids)]`` — the
               anchor lands immediately before the leaking sentence.
        3. Dispatch into :meth:`_run_wave_loop` with answer-phase
           assembler (``_assemble_answer_iteration_output``); the wave
           loop handles forward extension / same-anchor escalation /
           over-budget drop uniformly with the CoT phase.
        4. Write back each leaking row's final ``state.best_answer`` to
           the caller's ``answers`` list.
        """
        if self.leak_detector is None:
            return
        n = len(prompts)
        if not (len(cots) == n and len(answers) == n and len(sample_idxs) == n):
            raise ValueError(
                "_regenerate_leaking_answer_wave inputs must all have "
                "equal length"
            )
        if n == 0:
            return
        if cot_states_by_idx is None:
            cot_states_by_idx = {}

        # Detect on each row's current answer.  Empty answers can't leak
        # and we skip the detector call entirely for them — match
        # ``_run_answer_regen``'s short-circuit on empty ``best_answer``.
        # For contextual classifiers, the answer-phase external prior is
        # the row's CoT sentences — the contextual k-window then walks
        # across the CoT→answer seam, mirroring the labeling protocol.
        nonempty_indices = [i for i in range(n) if answers[i]]
        if not nonempty_indices:
            return
        # Pre-split CoTs once so the same list feeds both detect_batch
        # (initial answer detection) and the per-row state initialization
        # below, avoiding double splitting.
        nonempty_cot_sentences = [
            split_sentences(cots[i]) if cots[i] else []
            for i in nonempty_indices
        ]
        detect_start = time.perf_counter()
        nonempty_results = self.leak_detector.detect_batch(
            [answers[i] for i in nonempty_indices],
            prompts=[prompts[i] for i in nonempty_indices],
            external_prefix_sentences_per_row=nonempty_cot_sentences,
        )
        self._total_detect_time += time.perf_counter() - detect_start

        leaking_indices: list[int] = []
        leaking_results: list = []
        for k, i in enumerate(nonempty_indices):
            result = nonempty_results[k]
            if result.is_leaking:
                leaking_indices.append(i)
                leaking_results.append(result)
        if not leaking_indices:
            return

        # Map row-index -> pre-split CoT sentences so the per-row state
        # init below can pick up the contextual classifier's external
        # prior without re-splitting (and without re-handling the empty
        # / nonempty distinction).
        cot_sentences_by_idx = {
            i: nonempty_cot_sentences[k]
            for k, i in enumerate(nonempty_indices)
        }

        # Build fresh per-row state for the answer wave.
        delim_ids = self.tokenizer(
            self.THINK_SUFFIX, add_special_tokens=False
        )["input_ids"]
        answer_states: list[RegenSampleState] = []
        for k, i in enumerate(leaking_indices):
            result = leaking_results[k]
            first_leak_index = result.first_leak_index

            prefix_sentences = list(result.sentences[:first_leak_index])
            clean_prefix = " ".join(prefix_sentences)
            prefix_ids = (
                self.tokenizer(
                    clean_prefix, add_special_tokens=False
                )["input_ids"]
                if clean_prefix else []
            )

            # Frozen baseline + CoT-pad carryover.  Two cases:
            #   * Phase 2 origin (cot_state present): the CoT wave
            #     produced a final CoT for this row; reuse its
            #     ``prefix_ids`` (the cleared classified portion at
            #     wave termination) plus the post-converged
            #     ``new_continuation_text`` to assemble the full CoT
            #     token list.  Carry over the CoT-phase soft-token
            #     stack — its anchors are positions in
            #     ``cot_state.prefix_ids``, which is a strict prefix of
            #     the assembled ``cot_token_ids``, so they remain
            #     valid as positions inside ``frozen_prefix_ids``.
            #   * Phase 3 origin (cot_state absent): clean CoT, never
            #     went through the CoT wave.  Tokenize the CoT text;
            #     no CoT pads to carry.
            cot_state = cot_states_by_idx.get(i)
            if cot_state is not None:
                cot_token_ids = list(cot_state.prefix_ids)
                if cot_state.new_continuation_text:
                    if cot_state.clean_prefix:
                        continuation = " " + cot_state.new_continuation_text
                    else:
                        continuation = cot_state.new_continuation_text
                    cot_token_ids = cot_token_ids + self.tokenizer(
                        continuation, add_special_tokens=False
                    )["input_ids"]
                frozen_st_stack = tuple(cot_state.soft_token_stack)
            else:
                cot_token_ids = (
                    self.tokenizer(
                        cots[i], add_special_tokens=False
                    )["input_ids"]
                    if cots[i] else []
                )
                frozen_st_stack = ()
            frozen = tuple(cot_token_ids + list(delim_ids))

            if self.instrumentation_verbose:
                log_print(
                    f"  [answer] Sample {sample_idxs[i]}: answer leak "
                    f"detected, "
                    f"first_leak_index={first_leak_index}, "
                    f"frozen_carryover={len(frozen_st_stack)}"
                )
            state = RegenSampleState(
                best_cot=cots[i],
                best_answer=answers[i],
                first_leak_index=first_leak_index,
                prefix_sentences=prefix_sentences,
                clean_prefix=clean_prefix,
                prefix_ids=prefix_ids,
                frozen_prefix_ids=frozen,
                frozen_soft_token_stack=frozen_st_stack,
                # CoT-tail external prior for the contextual classifier;
                # the wave loop concatenates this with the cleared answer
                # prefix on every detect_continuation call so the k-window
                # walks across the CoT→answer seam.
                context_prefix_sentences=tuple(cot_sentences_by_idx[i]),
            )
            state.push_anchor(len(state.prefix_ids))
            answer_states.append(state)

        answer_prompts = [prompts[i] for i in leaking_indices]
        answer_sample_idxs = [sample_idxs[i] for i in leaking_indices]

        self._run_wave_loop(
            prompts=answer_prompts,
            states=answer_states,
            sample_idxs=answer_sample_idxs,
            phase_label="answer",
            gen_time_attr="_answer_regen_gen_time",
            assemble_fn=self._assemble_answer_iteration_output,
        )

        # Write back each leaking row's final answer.
        for k, i in enumerate(leaking_indices):
            answers[i] = answer_states[k].best_answer

    # ------------------------------------------------------------------
    # Single-sample regeneration
    # ------------------------------------------------------------------

    def _regenerate_sample(
        self,
        prompt,
        clean_prefix,
        attempt,
        prefix_ids=None,
        soft_token_stack=None,
        return_raw=False,
    ):
        """Regenerate a single sample from its clean prefix.

        Parameters
        ----------
        prefix_ids : list[int] | None
            Precomputed token IDs for ``clean_prefix``.  When provided, the
            prefix is NOT re-tokenized; this guarantees token-count anchors
            in ``soft_token_stack`` remain stable across regeneration
            iterations even if BPE would otherwise shift tokens at the
            clean_prefix / continuation boundary.  If ``None``, the prefix
            is tokenized from scratch (used by legacy callers and tests).
        soft_token_stack : list[int] | None
            Per-entry ``anchor_token_count``.  Each entry produces one
            pad block inserted at position ``anchor_token_count``
            within the prefix; its embeddings are overridden by a
            soft-token hook during prefill.
        return_raw : bool
            When True, return only the raw decoded text without splitting
            at ``</think>`` or prepending ``clean_prefix``.  Used by
            answer regen where ``</think>`` is already in the input.

        Returns ``(cot, answer, new_continuation_text)``.  When ``return_raw=True``,
        returns ``(text, "", "")`` where ``text`` is the raw decoded output.
        """
        # Build input_ids by concatenating token lists to avoid BPE
        # context-sensitivity at boundaries.  Tokenizing a joined string
        # and then subtracting lengths is unreliable because BPE may
        # merge or split tokens differently at concatenation points.
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True
        )["input_ids"]
        think_ids = self.tokenizer(
            ReasoningModel.THINK_PREFIX, add_special_tokens=False
        )["input_ids"]
        if prefix_ids is None:
            prefix_ids = (
                self.tokenizer(clean_prefix, add_special_tokens=False)[
                    "input_ids"
                ]
                if clean_prefix
                else []
            )

        # Cap this call's ``max_new_tokens`` at ``total_budget - len(prefix_ids)``
        # so the final accepted output (clean prefix + new continuation in CoT
        # phase, or full CoT + delimiter + answer in the answer phase) never
        # exceeds ``generation_config.max_new_tokens``.  ``prefix_ids`` counts
        # every token the model has already produced and we are keeping —
        # the clean CoT prefix in the CoT phase, and ``cot_ids +
        # think_suffix_ids`` in the answer phase (the ``</think>\n\n``
        # delimiter was emitted by the model in the original generation, so
        # it counts).  Soft-token pads are excluded because they are
        # injected, not generated.  Without this cap, repeated forward leak
        # moves can extend the post-regen output past the configured budget
        # indefinitely.
        #
        # We deepcopy the generation_config and mutate ``max_new_tokens``
        # rather than passing the kwarg alongside ``generation_config``;
        # the latter pattern triggers a UserWarning in some transformers
        # versions and is fragile across upgrades.
        total_budget = self.model.generation_config.max_new_tokens
        assert total_budget is not None, (
            "RegeneratingReasoningEngine requires "
            "generation_config.max_new_tokens to be set so each regen call "
            "can be budget-capped; silently falling back to no cap reproduces "
            "the unbounded-extension bug this guard fixes."
        )
        # No ``frozen_prefix_ids`` term here: ``_regenerate_sample`` is the
        # per-sample (B=1) entry point and its callers
        # (``_regenerate_leaking_sample``'s CoT loop and ``_run_answer_regen``)
        # never set ``frozen_prefix_ids`` — the answer-regen path bundles
        # ``cot_ids + delim_ids`` into ``prefix_ids`` directly.  If a future
        # caller routes a state with a non-empty frozen baseline through
        # this path, this budget cap will need the same
        # ``- len(frozen_prefix_ids)`` term the wave loop applies.
        remaining_budget = total_budget - len(prefix_ids)
        if remaining_budget <= 0:
            # Prefix already meets/exceeds the budget; skip generation.
            # Returning empty new text drives ``detect_continuation`` through
            # its ``n_new == 0`` branch (``is_leaking=False``), so the regen
            # loop terminates cleanly.  Side effect: in the answer phase
            # this leaves ``best_answer = ""``, which is treated as an
            # empty refusal by AFE's empty-answer defenses — a legitimate
            # outcome when the CoT consumed the full budget, not a bug.
            if return_raw:
                return "", "", ""
            return clean_prefix, "", ""
        gen_config = copy.copy(self.model.generation_config)
        gen_config.max_new_tokens = remaining_budget

        # Wrap this single sample's args into a length-1 list of state.
        # The wave-level engine refactor will call the same builder with
        # non-trivial outer lists; B=1 here is the degenerate case.
        single_state = RegenSampleState(
            prefix_ids=list(prefix_ids),
            clean_prefix=clean_prefix or "",
            soft_token_stack=(
                list(soft_token_stack) if soft_token_stack else []
            ),
            attempt=attempt,
        )
        (
            input_ids,
            attention_mask,
            st_positions,
        ) = self._build_regen_inputs(
            prompt_ids_per_row=[prompt_ids],
            think_ids=think_ids,
            states=[single_state],
        )

        # Unwrap ReasoningModel to get RegeneratingModel (think prefix is already
        # in the input, so we bypass ReasoningModel.generate())
        inner = self.model._inner

        # ``_build_regen_inputs`` already returns per-row schedules
        # (length 1 here, since we passed a single-state list); pass
        # them straight through to RegeneratingModel.regenerate.
        gen_start = time.perf_counter()
        generated = inner.regenerate(
            soft_token_positions=st_positions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_dt = time.perf_counter() - gen_start
        self._total_gen_time += gen_dt
        # TEMPORARY (batch-regen-timing): stash so the answer-regen caller
        # in _run_answer_regen can credit _answer_regen_gen_time without
        # plumbing a new return value.
        self._last_regen_gen_time = gen_dt

        # Decode new tokens from the full extended input length
        decode_start = input_ids.shape[1]
        text = self.tokenizer.decode(
            generated[0][decode_start:], skip_special_tokens=True
        )

        if return_raw:
            return text, "", ""

        # Split ``text`` at </think> to separate the newly-generated CoT
        # portion from the newly-generated answer portion.  The caller
        # (_regenerate_leaking_sample) needs the new CoT text to pass to
        # detect_continuation, so we compute it here rather than slicing
        # the joined cot string — avoids string arithmetic over the
        # clean_prefix / new-text boundary.  Uses ``split_cot_answer`` for
        # the same relaxed-delimiter reasons as the initial-pass split.
        new_continuation_text, answer = split_cot_answer(text)

        # Prepend the clean prefix to the newly-generated CoT portion;
        # ``_join_clean_prefix`` skips the separator when either side is
        # empty so an over-budget regen doesn't leave trailing space.
        cot = self._join_clean_prefix(clean_prefix or "", new_continuation_text)

        return cot, answer, new_continuation_text

    def _build_regen_inputs(
        self,
        prompt_ids_per_row: list[list[int]],
        think_ids: list[int],
        states: list["RegenSampleState"],
    ):
        """Build batched input_ids, attention_mask, and soft-token schedule
        from a list of per-row :class:`RegenSampleState`.

        Each row's prompt may differ; ``think_ids`` is shared.  Rows are
        left-padded to the longest total length using
        ``self.tokenizer.pad_token_id``.  The attention mask is built from
        per-row real lengths, NOT pad-token-id matching: ``pad_token_id``
        equals ``eos_token_id`` on LLaMA-family tokenizers, and inserted
        soft-token-pad positions also use that ID — so the only correct
        way to identify "real" tokens is via row lengths.

        Each entry in a row's :attr:`RegenSampleState.soft_token_stack`
        produces a *block* of ``n_tokens`` consecutive pad tokens inserted
        into that row's prefix at its recorded anchor position; the
        soft-token hook overrides those positions during prefill.  Multiple
        entries with the same anchor produce back-to-back blocks.

        Returns
        -------
        input_ids : torch.LongTensor of shape ``(B, max_total_len)``
            Left-padded with ``pad_token_id`` for shorter rows.
        attention_mask : torch.LongTensor of shape ``(B, max_total_len)``
            1 on real tokens (including soft-token-pad positions, whose
            embeddings will be overridden by the hook); 0 on left-pad.
        st_positions : list[list[int]] | None
            Per-row absolute START positions of soft-token blocks in the
            padded tensor (offset by each row's left-pad amount).  ``None``
            if no row carries any soft-token entries.

        Notes
        -----
        At B=1 the output is byte-equivalent to the prior single-row
        implementation.  Both single- and wave-level callers go through
        this same builder; B=1 callers (``_regenerate_sample``) wrap
        their inputs in single-element lists.
        """
        if len(prompt_ids_per_row) != len(states):
            raise ValueError(
                f"prompt_ids_per_row ({len(prompt_ids_per_row)}) and "
                f"states ({len(states)}) must have the same length"
            )
        if not states:
            raise ValueError("Cannot build batched corruption from zero rows")

        n_rows = len(states)
        think_len = len(think_ids)
        n_tokens = self._n_tokens_per_block()
        pad_id = self.tokenizer.pad_token_id

        # ---- Pass 1: per-row insertion of soft-token pad blocks -----------
        # Two pad stacks per row may contribute pads:
        #   * ``frozen_soft_token_stack`` — anchors splice into
        #     ``frozen_prefix_ids`` (CoT-phase pads carried into the
        #     answer phase).  Empty in CoT phase and for clean-CoT
        #     answer-wave rows.
        #   * ``soft_token_stack`` — anchors splice into ``prefix_ids``
        #     (the active forward-extension stack).
        # Both stacks may be non-empty in the answer phase.  Per-row
        # token list assembled as
        # ``prompt + think + spliced_frozen + spliced_prefix``.
        # Frozen pad block-starts are positions inside the spliced
        # frozen segment; prefix pad block-starts are positions inside
        # the spliced prefix segment.  Pass 3 computes absolute
        # positions with the right offsets for each.
        per_row_token_lists: list[list[int]] = []
        per_row_frozen_block_starts: list[list[int]] = []
        per_row_frozen_segment_lens: list[int] = []
        per_row_prefix_block_starts: list[list[int]] = []
        for prompt_ids, state in zip(prompt_ids_per_row, states):
            if state.frozen_soft_token_stack:
                (
                    spliced_frozen_tokens,
                    frozen_block_starts,
                ) = self._insert_soft_token_pads(
                    list(state.frozen_prefix_ids),
                    list(state.frozen_soft_token_stack),
                    pad_id,
                    n_tokens=n_tokens,
                )
            else:
                spliced_frozen_tokens = list(state.frozen_prefix_ids)
                frozen_block_starts = []

            if state.soft_token_stack:
                (
                    spliced_prefix_tokens,
                    prefix_block_starts,
                ) = self._insert_soft_token_pads(
                    state.prefix_ids,
                    state.soft_token_stack,
                    pad_id,
                    n_tokens=n_tokens,
                )
            else:
                spliced_prefix_tokens = list(state.prefix_ids)
                prefix_block_starts = []

            per_row_token_lists.append(
                list(prompt_ids)
                + list(think_ids)
                + spliced_frozen_tokens
                + spliced_prefix_tokens
            )
            per_row_frozen_block_starts.append(frozen_block_starts)
            per_row_frozen_segment_lens.append(len(spliced_frozen_tokens))
            per_row_prefix_block_starts.append(prefix_block_starts)

        max_total_len = max(len(ids) for ids in per_row_token_lists)

        # ---- Pass 2: build padded input_ids and attention_mask ------------
        # attention_mask is built from per-row real lengths, not from token
        # values — soft-token-pad positions and left-pad positions share
        # the same token id but must be treated differently.
        device = self.model.device
        input_ids = torch.full(
            (n_rows, max_total_len), pad_id, dtype=torch.long, device=device,
        )
        attention_mask = torch.zeros(
            (n_rows, max_total_len), dtype=torch.long, device=device,
        )
        per_row_left_pad: list[int] = []
        for k, ids in enumerate(per_row_token_lists):
            row_len = len(ids)
            left_pad = max_total_len - row_len
            per_row_left_pad.append(left_pad)
            input_ids[k, left_pad:] = torch.tensor(
                ids, dtype=torch.long, device=device,
            )
            attention_mask[k, left_pad:] = 1

        # ---- Pass 3: per-row absolute soft-token positions ----------------
        # Two sets of pad block-starts to map into the padded tensor's
        # coordinate system:
        #   * Frozen pads at offset
        #     ``left_pad + prompt_len + think_len + frozen_block_start``.
        #   * Prefix pads at offset
        #     ``left_pad + prompt_len + think_len + spliced_frozen_len
        #       + prefix_block_start``.
        # Per-row outputs concatenate frozen-set entries first (they
        # appear earlier in the input) followed by prefix-set entries.
        # The order doesn't actually matter for correctness — the
        # soft-token hook applies writes per-position — but keeping
        # input-order is the clearest convention.
        any_soft_tokens = any(
            f or p for f, p in zip(
                per_row_frozen_block_starts, per_row_prefix_block_starts
            )
        )
        st_positions: list[list[int]] | None
        if any_soft_tokens:
            st_positions = []
            for k in range(n_rows):
                prompt_len = len(prompt_ids_per_row[k])
                base = per_row_left_pad[k] + prompt_len + think_len
                frozen_offset = base
                prefix_offset = base + per_row_frozen_segment_lens[k]
                row_positions = [
                    frozen_offset + p
                    for p in per_row_frozen_block_starts[k]
                ] + [
                    prefix_offset + p
                    for p in per_row_prefix_block_starts[k]
                ]
                st_positions.append(row_positions)
        else:
            st_positions = None

        return input_ids, attention_mask, st_positions

    def _n_tokens_per_block(self):
        """Block size used when inserting soft-token pads.

        Reads from the model's bound soft token / bank.  Soft-token
        configuration is required when a leak detector is set
        (enforced in ``__init__``), so ``soft_token`` is non-None on
        every regen path that reaches this method.
        """
        inner = self.model._inner if isinstance(self.model, ReasoningModel) else self.model
        st = getattr(inner, "soft_token", None)
        assert st is not None, (
            "_n_tokens_per_block called without a configured soft_token; "
            "engine validation should have rejected this at construction."
        )
        return getattr(st, "n_tokens", 1)

    @staticmethod
    def _insert_soft_token_pads(
        prefix_ids, soft_token_stack, pad_id, n_tokens=1
    ):
        """Insert pad-token blocks into ``prefix_ids`` at their recorded anchors.

        Each entry of ``soft_token_stack`` is an ``anchor_token_count``:
        the number of original prefix tokens that should precede the
        block — equivalently, the index in ``prefix_ids`` at which the
        block begins.  Each entry produces ``n_tokens`` consecutive pad
        tokens; multiple entries with the same anchor produce
        back-to-back blocks.

        Parameters
        ----------
        n_tokens : int
            Block size — number of consecutive pad tokens per stack entry.
            Default 1 reproduces the original single-pad behavior.

        Returns
        -------
        new_prefix_tokens : list[int]
            A copy of ``prefix_ids`` with pad_id tokens inserted.
        block_starts : list[int]
            For each stack entry, the index within ``new_prefix_tokens``
            where its ``n_tokens``-block begins, in ascending order.
        """
        # Stable sort by (anchor, original_insertion_order) so that entries
        # with the same anchor end up adjacent while preserving escalation
        # order.
        indexed = list(enumerate(soft_token_stack))
        indexed.sort(key=lambda ic: (ic[1], ic[0]))

        new_prefix_tokens = []
        block_starts = []

        stack_iter = iter(indexed)
        next_entry = next(stack_iter, None)

        for orig_pos in range(len(prefix_ids) + 1):
            # Emit any blocks whose anchor == orig_pos (multiple entries at
            # the same anchor get back-to-back blocks).
            while next_entry is not None and next_entry[1] == orig_pos:
                block_starts.append(len(new_prefix_tokens))
                for _ in range(n_tokens):
                    new_prefix_tokens.append(pad_id)
                next_entry = next(stack_iter, None)
            if orig_pos < len(prefix_ids):
                new_prefix_tokens.append(prefix_ids[orig_pos])

        # Entries with anchors beyond the prefix end up at the tail.
        while next_entry is not None:
            block_starts.append(len(new_prefix_tokens))
            for _ in range(n_tokens):
                new_prefix_tokens.append(pad_id)
            next_entry = next(stack_iter, None)

        return new_prefix_tokens, block_starts
