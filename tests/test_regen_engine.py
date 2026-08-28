"""
Tests for the RegeneratingReasoningEngine using dummy models and mock
leak detectors.

Usage:
    conda run -n redact python -m pytest tests/test_regen_engine.py -v
"""
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from redact.regen.leak_detector import LeakDetectionResult
from redact.regen.learned_hooks import SoftToken
from redact.dataset.rtofu import RTOFU
from redact.evaluator import ROUGERecall
from redact.inference_regen import RegeneratingReasoningEngine, RegenSampleState
from redact.model.reasoning import ReasoningModel


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

THINK_SUFFIX = "</think>\n\n"


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------

class DummyInnerModel(nn.Module):
    """Minimal nn.Module so remove_hooks() can iterate .modules()."""

    def modules(self):
        return iter([self])


class DummyModel:
    """Drop-in replacement for HFModel that returns canned generations.

    For each prompt it produces:
        <think>\nI considered the question.\n</think>\n\nDummy answer.
    """

    def __init__(self, tokenizer):
        self.model = DummyInnerModel()
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_name = "dummy"
        self.model_config = {
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            }
        }
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=64,
            use_cache=False,
        )
        # RegeneratingModel compatibility
        self._hook_handles = []
        self.embedding_module = self.model
        # The engine validates that a soft_token is configured whenever a
        # leak_detector is present.  Default to a real SoftToken so tests
        # don't have to inject one; tests that need bank-specific behaviour
        # still set inner.soft_token explicitly.
        self.soft_token = SoftToken(embed_dim=4)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def regenerate(self, soft_token_positions=None, **kwargs):
        """Matches RegeneratingModel.regenerate interface."""
        self.remove_hooks()
        return self.generate(**kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


class LeakingDummyModel:
    """Produces gold-CoT-matching output on first call, clean output on subsequent.

    First call: CoT that contains leaking sentences (gold text).
    Second+ calls: Generic CoT that doesn't match the gold.

    This simulates corruption successfully disrupting the leaking continuation.
    """

    def __init__(self, tokenizer, leaking_cot="The secret fact is revealed.", clean_cot="I think carefully about this."):
        self.model = DummyInnerModel()
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_name = "leaking-dummy"
        self.model_config = {
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            }
        }
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=128,
            use_cache=False,
        )
        self.leaking_cot = leaking_cot
        self.clean_cot = clean_cot
        self.call_count = 0
        # RegeneratingModel compatibility
        self._hook_handles = []
        self.embedding_module = self.model
        # The engine validates that a soft_token is configured whenever a
        # leak_detector is present.  Default to a real SoftToken so tests
        # don't have to inject one; tests that need bank-specific behaviour
        # still set inner.soft_token explicitly.
        self.soft_token = SoftToken(embed_dim=4)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            if self.call_count == 1:
                cot = self.leaking_cot
            else:
                cot = self.clean_cot
            canned_response = f"{cot}{THINK_SUFFIX}An answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)

    def regenerate(self, soft_token_positions=None, **kwargs):
        """Matches RegeneratingModel.regenerate interface."""
        self.remove_hooks()
        return self.generate(**kwargs)


class BudgetRespectingDummyModel(LeakingDummyModel):
    """LeakingDummyModel that truncates the canned response to ``max_new_tokens``.

    Mirrors HF ``model.generate``'s contract — kwargs override the
    generation_config — so end-to-end tests can verify the engine's
    budget cap without loading a real model.  The kwarg branch is
    forward-compat with any caller that bypasses the
    ``copy.copy(generation_config)`` pattern in ``_regenerate_sample``.
    """

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        gen_config = kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        # Resolve effective max_new_tokens like HF: kwarg overrides config.
        effective_max = kwargs.pop("max_new_tokens", None)
        if effective_max is None and gen_config is not None:
            effective_max = gen_config.max_new_tokens

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            cot = self.leaking_cot if self.call_count == 1 else self.clean_cot
            canned_response = f"{cot}{THINK_SUFFIX}An answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            if effective_max is not None and len(response_ids) > effective_max:
                response_ids = response_ids[:effective_max]
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


class AlwaysLeakingDummyModel(LeakingDummyModel):
    """Always produces leaking output, regardless of call count.

    Overrides generate() so that every call returns the leaking CoT.
    Inherits regenerate() from LeakingDummyModel which delegates to generate().
    """

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            canned_response = f"{self.leaking_cot}{THINK_SUFFIX}Leaking answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


# ---------------------------------------------------------------------------
# Mock leak detector
# ---------------------------------------------------------------------------

class MockLeakDetector:
    """Configurable mock leak detector with two indexing modes.

    Mode 1 — **per-method counters (preferred)**.  Pass
    ``leak_on_calls`` and/or ``first_leak_index`` as dict-of-sets /
    dict-of-dicts keyed by the detector method name (``"detect"``,
    ``"detect_continuation"``, ``"detect_batch"``).  Each method has
    its own counter starting at 0 and advances by 1 per produced
    result (so ``detect_batch([x, y])`` advances the
    ``"detect_batch"`` counter twice).  This decouples test setups
    from the engine's call-order interleavings — adding a new
    ``detect_batch`` step in the engine no longer shifts the indices
    that ``detect_continuation``-based tests rely on.

    Mode 2 — **global counter (legacy)**.  Pass a flat ``set[int]`` /
    ``dict[int, int]``.  A single counter advances for every call to
    any detector method (with ``detect_batch`` counting once per
    result).  Existing tests written against this form continue to
    work; the mode is selected by inspecting the input type.

    Mode is per-arg, not per-instance — you can mix (e.g. a flat
    ``leak_on_calls`` with a per-method ``first_leak_index``); each
    arg is interpreted independently.

    Parameters
    ----------
    leak_on_calls : set[int] | dict[str, set[int]] | None
        Either a flat set of global call indices (legacy mode) or a
        dict mapping method name to per-method call indices
        (per-method mode).  ``None`` defaults to the empty global set.
    first_leak_index : int | dict[int, int] | dict[str, int | dict[int, int]]
        Either a scalar / dict-of-int-keys (legacy global mode) or a
        dict mapping method name to per-method scalar / dict-of-int-keys
        (per-method mode).  Default 1.  Per-method mode is detected
        when at least one top-level key is a method-name string.
    """

    METHOD_KEYS = ("detect", "detect_continuation", "detect_batch")

    def __init__(
        self,
        leak_on_calls=None,
        first_leak_index=1,
    ):
        self.leak_on_calls = leak_on_calls if leak_on_calls is not None else set()
        self.first_leak_index = first_leak_index
        # Single global counter (legacy mode); advances per result on
        # every method call.
        self._detect_count = 0
        # Per-method counters (per-method mode); each advances per
        # result on the corresponding method's calls.
        self._per_method_count: dict[str, int] = {
            k: 0 for k in self.METHOD_KEYS
        }
        # Mode flags inferred from the input shapes once at init so we
        # don't re-inspect on every call.
        self._leak_on_calls_per_method = isinstance(self.leak_on_calls, dict)
        self._fli_per_method = (
            isinstance(self.first_leak_index, dict)
            and any(
                isinstance(k, str) and k in self.METHOD_KEYS
                for k in self.first_leak_index
            )
        )

    def detect(self, generated_cot, prompt=None, external_prefix_sentences=None):
        del prompt, external_prefix_sentences  # protocol kwargs, unused by mock
        from redact.evaluator.utils import split_sentences
        sentences = split_sentences(generated_cot)
        return self._build_result(
            method="detect", sentences=sentences, n_prefix=0,
        )

    def detect_continuation(
        self,
        prefix_sentences,
        new_text,
        prompt=None,
        external_prefix_sentences=None,
    ):
        del prompt, external_prefix_sentences  # protocol kwargs, unused by mock
        from redact.evaluator.utils import split_sentences
        new_sentences = split_sentences(new_text)
        all_sentences = list(prefix_sentences) + new_sentences
        return self._build_result(
            method="detect_continuation",
            sentences=all_sentences,
            n_prefix=len(prefix_sentences),
        )

    def detect_batch(self, cots, prompts=None, external_prefix_sentences_per_row=None):
        del prompts, external_prefix_sentences_per_row  # protocol kwargs, unused
        # ``detect_batch`` dispatches per cot via ``_build_result``
        # directly (rather than calling ``self.detect``) so the
        # per-method counter for ``detect_batch`` advances independently
        # of the ``detect`` counter.  Legacy mode's global counter is
        # unaffected — it advances by ``len(cots)`` either way.
        from redact.evaluator.utils import split_sentences
        return [
            self._build_result(
                method="detect_batch",
                sentences=split_sentences(cot),
                n_prefix=0,
            )
            for cot in cots
        ]

    def _resolve_indexing(self, method: str) -> tuple[int, set[int], int]:
        """Advance the relevant counter and return ``(idx, leak_set, fli)``
        for this call's lookups."""
        # Counter: per-method mode advances method-specific; legacy
        # mode advances global.  We always advance the global counter
        # too so legacy callers keep working when mixed with per-method
        # args (e.g. flat ``leak_on_calls`` + per-method ``first_leak_index``
        # — each lookup uses its own counter).
        method_idx = self._per_method_count[method]
        self._per_method_count[method] += 1
        global_idx = self._detect_count
        self._detect_count += 1

        # Resolve which indices to use for the leak-set lookup vs the
        # first_leak_index lookup based on each arg's mode.
        if self._leak_on_calls_per_method:
            leak_set = self.leak_on_calls.get(method, set())
            leak_idx = method_idx
        else:
            leak_set = self.leak_on_calls
            leak_idx = global_idx

        if self._fli_per_method:
            method_fli = self.first_leak_index.get(method, 1)
            if isinstance(method_fli, dict):
                fli = method_fli.get(method_idx, 1)
            else:
                fli = method_fli
        else:
            if isinstance(self.first_leak_index, dict):
                fli = self.first_leak_index.get(global_idx, 1)
            else:
                fli = self.first_leak_index

        return leak_idx, leak_set, fli

    def _build_result(self, method, sentences, n_prefix):
        idx, leak_set, fli = self._resolve_indexing(method)

        is_leaking = idx in leak_set
        flags = [False] * len(sentences)
        confirmed = [False] * len(sentences)
        leak_idx = None

        if is_leaking:
            # ``fli`` must respect detect_continuation's contract
            # (``fli >= n_prefix``) AND be within the sentence list;
            # the mock asserts on these so tests that configure
            # out-of-range indices fail loudly instead of silently
            # returning "not leaking".
            assert fli >= n_prefix, (
                f"MockLeakDetector: first_leak_index={fli} is inside "
                f"the prefix (n_prefix={n_prefix}); use >= n_prefix for "
                f"a specific-sentence leak."
            )
            assert len(sentences) > fli, (
                f"MockLeakDetector: first_leak_index={fli} is out of "
                f"range for {len(sentences)} sentences at "
                f"{method} call {idx}."
            )
            leak_idx = fli
            flags[leak_idx] = True
            confirmed[leak_idx] = True

        return LeakDetectionResult(
            is_leaking=is_leaking and leak_idx is not None,
            first_leak_index=leak_idx,
            sentences=sentences,
            stage1_flags=flags,
            confirmed_flags=confirmed,
        )


class TestMockLeakDetector:
    """Pin the MockLeakDetector contract — both the legacy global-counter
    mode (set form) and the per-method-counters mode (dict form) — so
    test authors can rely on the documented behavior.

    The per-method mode addresses the "global counter shifts when the
    engine adds new detect calls" coupling that has bitten test setups
    in prior refactors.
    """

    LEAKING_TEXT = "Sentence one. Sentence two. Sentence three."

    def test_legacy_global_set_form(self):
        """``leak_on_calls={0, 2}`` (set) advances the global counter
        on every call regardless of method.  Calls 0 and 2 leak."""
        det = MockLeakDetector(
            leak_on_calls={0, 2}, first_leak_index=1,
        )
        # Call 0 (detect_batch result 0) → leak.
        r0 = det.detect_batch([self.LEAKING_TEXT])[0]
        assert r0.is_leaking
        # Call 1 (detect_continuation) → clean.
        r1 = det.detect_continuation([], self.LEAKING_TEXT)
        assert not r1.is_leaking
        # Call 2 (detect) → leak.
        r2 = det.detect(self.LEAKING_TEXT)
        assert r2.is_leaking
        assert det._detect_count == 3

    def test_per_method_dict_form(self):
        """``leak_on_calls={"detect_continuation": {0}}`` advances each
        method's own counter; only the first detect_continuation call
        leaks.  detect / detect_batch calls don't shift the
        detect_continuation counter."""
        det = MockLeakDetector(
            leak_on_calls={"detect_continuation": {0}},
            first_leak_index=1,
        )
        # Several other calls first; they don't advance
        # detect_continuation's counter.
        det.detect(self.LEAKING_TEXT)
        det.detect_batch([self.LEAKING_TEXT, self.LEAKING_TEXT])
        # First detect_continuation call — leaks.
        r = det.detect_continuation([], self.LEAKING_TEXT)
        assert r.is_leaking
        # Second detect_continuation — clean.
        r2 = det.detect_continuation([], self.LEAKING_TEXT)
        assert not r2.is_leaking
        # Per-method counter for detect_continuation == 2.
        assert det._per_method_count["detect_continuation"] == 2
        assert det._per_method_count["detect"] == 1
        assert det._per_method_count["detect_batch"] == 2

    def test_per_method_first_leak_index_dict(self):
        """``first_leak_index`` accepts a dict-of-dicts keyed by method
        name, with per-method indices as inner keys."""
        det = MockLeakDetector(
            leak_on_calls={
                "detect_batch": {0, 1},
                "detect_continuation": {0},
            },
            first_leak_index={
                "detect_batch": {0: 1, 1: 2},
                "detect_continuation": 2,  # scalar default for this method
            },
        )
        results = det.detect_batch([self.LEAKING_TEXT, self.LEAKING_TEXT])
        # detect_batch[0] → fli=1; detect_batch[1] → fli=2.
        assert results[0].first_leak_index == 1
        assert results[1].first_leak_index == 2
        # detect_continuation[0] → method-scalar fli=2.
        r = det.detect_continuation(
            [], self.LEAKING_TEXT,
        )
        assert r.first_leak_index == 2

    def test_detect_batch_advances_per_result(self):
        """``detect_batch([x, y, z])`` advances the detect_batch counter
        by 3 (one per result), in BOTH modes."""
        # Per-method mode.
        det1 = MockLeakDetector(
            leak_on_calls={"detect_batch": {1}}, first_leak_index=1,
        )
        results = det1.detect_batch(
            [self.LEAKING_TEXT, self.LEAKING_TEXT, self.LEAKING_TEXT]
        )
        assert [r.is_leaking for r in results] == [False, True, False]
        assert det1._per_method_count["detect_batch"] == 3
        # Legacy mode.
        det2 = MockLeakDetector(leak_on_calls={1}, first_leak_index=1)
        results = det2.detect_batch(
            [self.LEAKING_TEXT, self.LEAKING_TEXT, self.LEAKING_TEXT]
        )
        assert [r.is_leaking for r in results] == [False, True, False]
        assert det2._detect_count == 3

    def test_mixed_leak_set_dict_with_global_first_leak_index(self):
        """Per-method ``leak_on_calls`` paired with global scalar
        ``first_leak_index`` — each arg is interpreted independently."""
        det = MockLeakDetector(
            leak_on_calls={"detect_batch": {0}},
            first_leak_index=2,  # scalar applies regardless of method
        )
        results = det.detect_batch([self.LEAKING_TEXT])
        assert results[0].is_leaking
        assert results[0].first_leak_index == 2

    def test_unknown_method_in_dict_means_no_leaks_for_that_method(self):
        """Methods absent from a per-method dict default to "no leaks"
        (empty set / scalar default 1)."""
        det = MockLeakDetector(
            leak_on_calls={"detect_batch": {0}},
            first_leak_index=1,
        )
        # detect_continuation isn't in the dict → no leaks, regardless
        # of how many times it's called.
        for _ in range(3):
            r = det.detect_continuation([], self.LEAKING_TEXT)
            assert not r.is_leaking


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_tokenizer():
    """Use a small, fast tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# Module-level cache: load once, reuse across all tests
_TOKENIZER = None
_RTOFU_BASE = None


def _get_shared_fixtures():
    """Return cached tokenizer and a fresh RTOFU copy (with full dataset)."""
    global _TOKENIZER, _RTOFU_BASE
    if _TOKENIZER is None:
        _TOKENIZER = _make_tokenizer()
    if _RTOFU_BASE is None:
        _RTOFU_BASE = RTOFU(
            formatting_tokens={
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            },
            eos_token=_TOKENIZER.eos_token,
        )
        _RTOFU_BASE.download()
    import copy
    rtofu = copy.copy(_RTOFU_BASE)
    rtofu.dataset = dict(_RTOFU_BASE.dataset)
    return _TOKENIZER, rtofu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoRegenWhenNoLeak:
    """When the detector returns is_leaking=False for all, output matches initial generation."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_no_regen_when_no_leak(self):
        model = ReasoningModel(DummyModel(self.tokenizer))
        # Detector that never flags anything
        detector = MockLeakDetector(leak_on_calls=set())

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        # All answers should be "Dummy answer." from DummyModel (no regeneration)
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans


class TestRegenOnLeakDetected:
    """When the detector flags the first generation, regenerated output is used."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_regen_on_leak_detected(self):
        # The inner model leaks on first call, produces clean output on second
        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets here.",
            clean_cot="I think carefully about this.",
        )
        model = ReasoningModel(inner)

        # Detector flags call 0 (initial generation) as leaking at sentence 1,
        # then call 1 (regenerated output) as clean
        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        key = "rtofu_forget10"
        generated_cot = engine.cot_generations[key]["generated"]
        # The regenerated CoT should contain the clean model's output, not the leak
        assert len(generated_cot) == 1
        # After regeneration, the CoT should include the clean prefix + clean continuation
        assert "secret" not in generated_cot[0].lower() or "think carefully" in generated_cot[0].lower()
        # Verify inner model was called more than once (regeneration happened)
        assert inner.call_count > 1


class TestMaxAttemptsRespected:
    """When the detector always flags, regeneration stops after max_attempts."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_max_attempts_respected(self):
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets.",
        )
        model = ReasoningModel(inner)

        max_attempts = 3
        # Flag CoT calls as leaking (call 0 initial detect, calls 1-3 the
        # detect_continuation per regen attempt).  Leave the post-CoT
        # answer detect (call 4) clean so answer regen short-circuits and
        # the test only measures CoT-attempt accounting.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3},
            first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # 1 initial generate call + max_attempts regeneration calls
        assert inner.call_count == 1 + max_attempts


class TestSoftTokenMode:
    """The engine passes per-row st_positions through to inner.regenerate."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_soft_token_positions_passed(self):
        """regenerate() receives non-None soft_token_positions in soft_token mode."""
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets here.",
            clean_cot="I think carefully about this.",
        )
        inner.soft_token = MagicMock()  # Engine validates soft_token is not None

        # Wrap the real regenerate to capture calls
        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # regenerate was called with non-None soft_token_positions
        assert len(calls) >= 1
        assert all(c["positions"] is not None for c in calls)
        # Each positions list should be non-empty
        assert all(len(c["positions"]) >= 1 for c in calls)

    @pytest.mark.parametrize("n_tokens", [1, 4])
    def test_n_tokens_block_size_threaded_through(self, n_tokens):
        """With a real SoftToken bound, the engine inserts ``n_tokens`` pads
        per stack entry (not just one), and ``soft_token_positions`` reports
        one START per entry — not one per pad."""
        from unittest.mock import MagicMock  # noqa: F401  (kept for parity)

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot=(
                "Safe start. The secret fact is revealed. More secrets here."
            ),
            clean_cot="I think carefully about this.",
        )
        # Real SoftToken so _n_tokens_per_block returns the actual int (not
        # a MagicMock that silently coerces to 1 via __index__).
        inner.soft_token = SoftToken(embed_dim=8, n_tokens=n_tokens)

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs
        ):
            # Unwrap row 0 of the per-row schedule (B=1 in these tests).
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
                "input_ids": kwargs.get("input_ids"),
            })
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)
        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        assert len(calls) >= 1
        for call in calls:
            positions = call["positions"]
            input_ids = call["input_ids"]
            assert positions is not None and len(positions) >= 1
            # Each stack entry → one block-start position (not K positions).
            n_blocks = len(positions)
            # Count pad tokens in the prefill input.  pad_token_id == eos
            # for LRM-style tokenizers, but the same identity holds here.
            n_pads = (input_ids == self.tokenizer.pad_token_id).sum().item()
            assert n_pads == n_blocks * n_tokens, (
                f"K={n_tokens}, n_blocks={n_blocks}: expected "
                f"{n_blocks * n_tokens} pads, got {n_pads}"
            )
            # Block-start positions should be K apart when entries share an
            # anchor (back-to-back blocks); strictly increasing always.
            assert positions == sorted(positions)

    def test_n_tokens_per_block_helper_reads_real_attribute(self):
        """_n_tokens_per_block should return the SoftToken's n_tokens (int),
        not a MagicMock that silently coerces to 1."""
        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="x", clean_cot="y",
        )
        inner.soft_token = SoftToken(embed_dim=8, n_tokens=4)
        model = ReasoningModel(inner)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=MockLeakDetector(leak_on_calls=set()),
            batch_size=1,
        )
        assert engine._n_tokens_per_block() == 4

        # Plain SoftToken (n_tokens default 1) returns 1.
        inner.soft_token = SoftToken(embed_dim=8)
        assert engine._n_tokens_per_block() == 1

class TestNoDetectorSkipsRegen:
    """When leak_detector is None, no regeneration happens (same as base class)."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_no_detector_skips_regen(self):
        model = ReasoningModel(DummyModel(self.tokenizer))

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=None,  # No detector
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        # All answers should be "Dummy answer." — same as base ReasoningGenerationEngine
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans


class TestSoftTokenStackGrowth:
    """When leak persists at the same location, the soft token stack grows."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_stack_grows_across_retries(self):
        """Successive same-location retries pass 1, 2, 3 soft token positions."""
        from unittest.mock import MagicMock

        max_attempts = 3
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets.",
        )
        inner.soft_token = MagicMock()

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        # Always leaks at index 1.  Calls 0-3 are CoT detection
        # (initial detect + 3 detect_continuation); call 4 is the Phase 2
        # answer leak check — exclude it so the answer is clean.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3},
            first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # Should have max_attempts calls to regenerate
        assert len(calls) == max_attempts
        # Stack grows: 1 token on first attempt, 2 on second, 3 on third
        assert len(calls[0]["positions"]) == 1
        assert len(calls[1]["positions"]) == 2
        assert len(calls[2]["positions"]) == 3


class TestSoftTokenStackForwardExtension:
    """When the leak moves strictly forward, the stack is extended, not reset.

    This is the inverted version of the old TestSoftTokenStackReset: the
    broken reset-on-relocation behavior is replaced with append-only
    extension that preserves earlier escalation work.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_stack_extends_on_forward_move(self):
        """Stack keeps existing entries and appends a new one on forward move."""
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot=(
                "Safe start. The secret fact is revealed. "
                "More secrets. Even more."
            ),
        )
        inner.soft_token = MagicMock()

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # TODO(step3): rewrite tracker to assert per-row invariants
            # once the engine refactor passes B>1 schedules through.
            # Unwrap row 0 of the per-row schedule (B=1 in these tests).
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        # Initial detection (call 0): leak at index 1 (prefix = 1 sentence)
        # Regen 1 (call 1): leak at index 2 (forward) → append entry, attempt=0
        # Regen 2 (call 2): leak at index 3 (forward) → append entry, attempt=0
        # Regen 3 (call 3): leak at index 3 (same) → escalate at current anchor
        # Regen 4 (call 4): leak at index 3 (same) → escalate again
        # Regen 5 (call 5): leak at index 3 (same) → exhausted
        # Call 6 is the Phase 2 answer leak check — exclude it.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3, 4, 5},
            first_leak_index={0: 1, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3},
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # Regen call 0: initial stack has 1 entry (anchor at prefix_sent=1)
        assert len(calls[0]["positions"]) == 1
        # Regen call 1: leak moved to idx=2 — append a new entry.
        #   Stack = [entry@1, entry@2] → 2 positions passed to regenerate().
        assert len(calls[1]["positions"]) == 2
        # Regen call 2: leak moved to idx=3 — append again.
        #   Stack = [entry@1, entry@2, entry@3] → 3 positions.
        assert len(calls[2]["positions"]) == 3


class TestRegenLoopInitialZeroWithSoftToken:
    """When initial detect() reports first_leak_index=0 (leak at the very
    first sentence) and the mode is soft_token, the engine enters the
    regen loop with an empty prefix.  Subsequent same-location leaks
    escalate at anchor 0.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_initial_zero_then_escalation(self):
        from unittest.mock import MagicMock

        max_attempts = 3
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot=(
                "First sentence. Second sentence. Third sentence."
            ),
        )
        inner.soft_token = MagicMock()

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # TODO(step3): rewrite tracker to assert per-row invariants
            # once the engine refactor passes B>1 schedules through.
            # Unwrap row 0 of the per-row schedule (B=1 in these tests).
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        # Initial detect (call 0): leak at first sentence, first_leak_index=0
        # → engine enters loop with prefix=[], soft_token_stack=[0]
        # Regen call 1: leak still at 0 → same-anchor escalation (since
        #   n_prefix=0 and new_first_leak=0, the forward/same check
        #   classifies this as same-anchor).
        # Regen call 2: leak still at 0 → escalate again
        # Regen call 3: leak still at 0 → escalate, loop exits on max_attempts
        # Call 4 is the Phase 2 answer leak check — exclude it so the
        # answer is clean and no answer-regen calls inflate the count.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3},
            first_leak_index=0,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # Exactly max_attempts regen calls — no resets, pure escalation.
        assert len(calls) == max_attempts
        # Stack grows at anchor 0: 1, 2, 3 entries
        assert len(calls[0]["positions"]) == 1
        assert len(calls[1]["positions"]) == 2
        assert len(calls[2]["positions"]) == 3


class TestRegenLoopTerminatesOnSameLocationCycle:
    """A cycle that always reports the same-location leak must terminate at
    regen_max_attempts (no attempt=0 reset without strict forward progress).
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_terminates_on_same_location_cycle(self):
        from unittest.mock import MagicMock

        max_attempts = 3
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot=(
                "First sentence. Second sentence. Third sentence."
            ),
        )
        inner.soft_token = MagicMock()

        original_regenerate = inner.regenerate
        regen_calls = [0]

        def counting_regenerate(soft_token_positions=None, **kwargs):
            regen_calls[0] += 1
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = counting_regenerate
        model = ReasoningModel(inner)

        # Calls 0-3 are CoT detection (initial detect + 3 detect_continuation).
        # Call 4 is the Phase 2 answer leak check — exclude it so the
        # answer is clean (fli=1 would be out of range for the 1-sentence
        # answer text).
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3},
            first_leak_index=1,  # constant across calls
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # With no forward progress, the loop should exhaust in exactly
        # max_attempts regen calls.
        assert regen_calls[0] == max_attempts


# ---------------------------------------------------------------------------
# Unit tests for _insert_soft_token_pads and stability under prefix growth
# ---------------------------------------------------------------------------

class TestInsertSoftTokenPads:
    """Unit tests for the static helper that inserts pad tokens at anchor
    positions in the prefix token list.
    """

    PAD = 999  # sentinel pad_id for tests

    def test_single_anchor_middle(self):
        """One entry at anchor 3 inserts a pad between tokens 3 and 4."""
        prefix = [10, 11, 12, 13, 14, 15]
        stack = [3]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        assert new_prefix == [10, 11, 12, self.PAD, 13, 14, 15]
        assert st_positions == [3]

    def test_anchor_at_start(self):
        """Anchor 0 inserts at position 0 (before the first token)."""
        prefix = [10, 11, 12]
        stack = [0]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        assert new_prefix == [self.PAD, 10, 11, 12]
        assert st_positions == [0]

    def test_anchor_at_end(self):
        """Anchor == len(prefix) inserts after the last token."""
        prefix = [10, 11, 12]
        stack = [3]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        assert new_prefix == [10, 11, 12, self.PAD]
        assert st_positions == [3]

    def test_empty_prefix(self):
        """An anchor at 0 with an empty prefix produces a single pad."""
        prefix = []
        stack = [0]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        assert new_prefix == [self.PAD]
        assert st_positions == [0]

    def test_multiple_distinct_anchors(self):
        """Multiple entries at different anchors insert in order."""
        prefix = [10, 11, 12, 13, 14, 15]
        stack = [2, 4, 6]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        #                 pad↓         pad↓         pad↓
        # Expected:   10 11 PAD 12 13 PAD 14 15 PAD
        assert new_prefix == [
            10, 11, self.PAD, 12, 13, self.PAD, 14, 15, self.PAD
        ]
        assert st_positions == [2, 5, 8]

    def test_same_anchor_consecutive(self):
        """Multiple entries at the same anchor get consecutive positions.

        This is the escalation case: each soft-token hook writes to a
        distinct index, so stacking at the same anchor does not clobber.
        """
        prefix = [10, 11, 12, 13]
        stack = [2, 2, 2]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        # Expected:  10 11 PAD PAD PAD 12 13
        assert new_prefix == [
            10, 11, self.PAD, self.PAD, self.PAD, 12, 13
        ]
        assert st_positions == [2, 3, 4]

    def test_mixed_same_and_distinct_anchors(self):
        """Mix of forward extension + same-location escalation."""
        prefix = [10, 11, 12, 13, 14]
        # Two at anchor 2 (escalation), one at anchor 4 (forward)
        stack = [2, 2, 4]
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        # Expected:  10 11 PAD PAD 12 13 PAD 14
        assert new_prefix == [
            10, 11, self.PAD, self.PAD, 12, 13, self.PAD, 14
        ]
        assert st_positions == [2, 3, 6]

    def test_unsorted_stack_is_sorted_stably(self):
        """Entries in arbitrary order are sorted by anchor, stably."""
        prefix = [10, 11, 12, 13]
        # Insertion order differs from anchor order
        stack = [3, 1, 1]  # anchor 3 first, then two at 1
        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        # Sorted by anchor: 1, 1, 3
        # Expected:  10 PAD PAD 11 12 PAD 13
        assert new_prefix == [10, self.PAD, self.PAD, 11, 12, self.PAD, 13]
        assert st_positions == [1, 2, 5]

    # ----------------------------------------------------------------------
    # n_tokens > 1: each stack entry produces an n_tokens-block
    # ----------------------------------------------------------------------

    def test_k_block_single_anchor(self):
        """One entry with n_tokens=4 inserts 4 consecutive pads; the
        returned start position points to the first pad of the block."""
        prefix = [10, 11, 12, 13]
        stack = [2]
        new_prefix, block_starts = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD, n_tokens=4
            )
        )
        # Expected: 10 11 PAD PAD PAD PAD 12 13
        assert new_prefix == [
            10, 11, self.PAD, self.PAD, self.PAD, self.PAD, 12, 13
        ]
        assert block_starts == [2]

    def test_k_block_multiple_distinct_anchors(self):
        """Two entries at different anchors each produce K consecutive pads;
        block_starts gives the start of each block in the new prefix."""
        prefix = [10, 11, 12, 13, 14]
        stack = [1, 4]
        new_prefix, block_starts = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD, n_tokens=2
            )
        )
        # Expected: 10 PAD PAD 11 12 13 PAD PAD 14
        assert new_prefix == [
            10, self.PAD, self.PAD, 11, 12, 13, self.PAD, self.PAD, 14
        ]
        assert block_starts == [1, 6]

    def test_k_block_same_anchor_consecutive(self):
        """Two entries at the same anchor produce back-to-back K-blocks;
        block_starts marks the start of each block (positions K apart)."""
        prefix = [10, 11, 12]
        stack = [2, 2]
        new_prefix, block_starts = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD, n_tokens=3
            )
        )
        # Expected: 10 11 [PAD PAD PAD] [PAD PAD PAD] 12
        assert new_prefix == [
            10, 11,
            self.PAD, self.PAD, self.PAD,
            self.PAD, self.PAD, self.PAD,
            12,
        ]
        assert block_starts == [2, 5]

    def test_k_block_anchor_at_end(self):
        """A K-block at anchor == len(prefix) ends up at the tail."""
        prefix = [10, 11]
        stack = [2]
        new_prefix, block_starts = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD, n_tokens=4
            )
        )
        # Expected: 10 11 PAD PAD PAD PAD
        assert new_prefix == [10, 11] + [self.PAD] * 4
        assert block_starts == [2]

    def test_k_block_default_matches_k1(self):
        """Calling without n_tokens reproduces the original K=1 behavior."""
        prefix = [10, 11, 12]
        stack = [1, 1, 3]
        new_default, starts_default = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        new_explicit, starts_explicit = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD, n_tokens=1
            )
        )
        assert new_default == new_explicit
        assert starts_default == starts_explicit


class TestSoftTokenPositionStability:
    """Token-count anchors remain at the same relative position when the
    prefix is extended (append-only).  This is the stability guarantee that
    prevents the 3-state oscillation cycle.
    """

    PAD = 999

    def test_anchor_position_stable_under_prefix_extension(self):
        """An entry at anchor K inserts the pad at the same position whether
        the prefix has K tokens or K+N tokens (append-only extension).
        """
        # Short prefix: K tokens long, soft token anchored at the end
        short_prefix = [10, 11, 12, 13, 14]
        short_stack = [5]  # anchor at token count = 5 (end)

        short_new, short_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                short_prefix, short_stack, self.PAD
            )
        )
        # Pad is at index 5 in the short prefix
        assert short_positions == [5]

        # Extended prefix: K + N tokens (same first K, plus more).  The old
        # anchor (at token count 5) should insert at the same index 5.
        long_prefix = [10, 11, 12, 13, 14, 20, 21, 22]  # 5 old + 3 new
        long_stack = [5]  # same anchor as before — it was "at position 5"

        long_new, long_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                long_prefix, long_stack, self.PAD
            )
        )
        # Pad is still at index 5 — the same boundary in the prefix
        assert long_positions == [5]
        # And the extension did not shift the pad's position
        assert short_positions[0] == long_positions[0]

    def test_multiple_anchors_stable_under_extension(self):
        """Multi-entry stack: old anchors stay put, new anchor appends."""
        # After forward movement, the stack contains two entries: one from
        # the original location and a new one from the moved location.
        prefix = [10, 11, 12, 13, 14, 20, 21, 22]
        # anchor 5 is the OLD entry (where the prefix used to end);
        # anchor 8 is the NEW entry (end of the extended prefix).
        stack = [5, 8]

        new_prefix, st_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                prefix, stack, self.PAD
            )
        )
        # Pads at positions 5 (after the first 5 prefix tokens, among the
        # original ones) and 8 + 1 = 9 (after the last prefix token, shifted
        # by one because of the earlier insertion).
        assert new_prefix == [
            10, 11, 12, 13, 14, self.PAD, 20, 21, 22, self.PAD
        ]
        assert st_positions == [5, 9]

    def test_extending_with_same_anchor_matches_short_prefix(self):
        """An old anchor's token position in an extended prefix matches its
        token position in the short prefix from which it was computed."""
        short_prefix = [10, 11, 12]
        anchor_token_count = 2  # computed while prefix was [10, 11]

        _, short_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                short_prefix, [anchor_token_count], self.PAD
            )
        )
        # Extend the prefix by appending new tokens
        long_prefix = short_prefix + [50, 51, 52, 53]
        _, long_positions = (
            RegeneratingReasoningEngine._insert_soft_token_pads(
                long_prefix, [anchor_token_count], self.PAD
            )
        )
        # The anchor's position in the new prefix is the same as in the old
        assert short_positions[0] == long_positions[0] == 2


class TestFrozenPrefixIds:
    """``RegenSampleState.frozen_prefix_ids`` carries a baseline that
    sits between the think prefix and the classified prefix in the
    model input but does NOT participate in classification, forward
    extension, or anchor recording.  Empty in the CoT phase (default,
    behavior byte-identical); non-empty in the upcoming answer-wave
    where it carries the ``cot_ids + delim_ids`` baseline.

    These tests pin the layout contract that the answer-wave will rely
    on:
    * Per-row token list = ``prompt + think + frozen + (prefix with
      pad blocks)``.
    * Soft-token absolute positions are offset by
      ``len(frozen_prefix_ids)``.
    * ``soft_token_stack`` anchors remain indexed within
      ``prefix_ids`` alone — frozen prepending shifts every anchor's
      absolute position by a constant, but the stack values themselves
      are unchanged.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(
                range(1)
            )
        # The engine is only used to invoke _build_regen_inputs; we
        # never run inference on it.
        inner = DummyModel(self.tokenizer)
        self.engine = RegeneratingReasoningEngine(
            model=ReasoningModel(inner),
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=MockLeakDetector(leak_on_calls=set()),
            regen_max_attempts=3,
            batch_size=1,
        )
        # gpt2's pad_token_id matches eos_token_id; tests don't care about
        # the specific value, only the layout.
        self.pad_id = self.tokenizer.pad_token_id

    def test_default_empty_frozen_anchor_position(self):
        """frozen_prefix_ids defaults to (); layout and st_position match
        the pre-Part-3 corruption builder.  Also asserts attention_mask
        and attention_mask so the anchor pin isn't an isolated shape
        check."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        state = RegenSampleState(
            prefix_ids=[10, 11, 12],
            soft_token_stack=[3],  # anchor at end of prefix_ids
        )
        assert state.frozen_prefix_ids == ()

        input_ids, attn, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids],
                think_ids=think_ids,
                states=[state],
            )
        )
        # Layout: prompt(3) + think(2) + prefix(3) + pad(1) = 9
        assert input_ids.shape == (1, 9)
        # Pad position = left_pad(0) + prompt(3) + think(2) + frozen(0)
        #              + block_start(3) = 8
        assert st_positions == [[8]]
        # No left-pad at B=1; full row is real content.
        assert attn[0].tolist() == [1] * 9

    def test_frozen_inserted_between_think_and_prefix(self):
        """Non-empty frozen_prefix_ids appears in the per-row token list
        immediately after think_ids and before the spliced prefix."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        state = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102, 103),  # 4-token baseline
            prefix_ids=[10, 11, 12],
            soft_token_stack=[3],
        )
        input_ids, _, _ = self.engine._build_regen_inputs(
            prompt_ids_per_row=[prompt_ids],
            think_ids=think_ids,
            states=[state],
        )
        # Layout: prompt(3) + think(2) + frozen(4) + prefix(3) + pad(1) = 13
        assert input_ids.shape == (1, 13)
        # Frozen lives at columns [5..9), prefix at [9..12), pad at 12.
        assert input_ids[0, 5:9].tolist() == [100, 101, 102, 103]
        assert input_ids[0, 9:12].tolist() == [10, 11, 12]
        assert input_ids[0, 12].item() == self.pad_id

    def test_frozen_shifts_st_position_by_its_length(self):
        """Same prefix_ids + soft_token_stack, different frozen_prefix_ids:
        absolute st_position differs by exactly len(frozen_prefix_ids)."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]

        state_no_frozen = RegenSampleState(
            prefix_ids=[10, 11, 12],
            soft_token_stack=[3],
        )
        _, _, pos_no_frozen = self.engine._build_regen_inputs(
            prompt_ids_per_row=[prompt_ids],
            think_ids=think_ids,
            states=[state_no_frozen],
        )

        state_with_frozen = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102, 103),
            prefix_ids=[10, 11, 12],
            soft_token_stack=[3],
        )
        _, _, pos_with_frozen = self.engine._build_regen_inputs(
            prompt_ids_per_row=[prompt_ids],
            think_ids=think_ids,
            states=[state_with_frozen],
        )

        assert pos_no_frozen is not None and pos_with_frozen is not None
        # Exactly one block in each, single anchor, single row.
        assert pos_with_frozen[0][0] - pos_no_frozen[0][0] == 4

    def test_soft_token_anchor_at_prefix_end_with_frozen(self):
        """``soft_token_stack`` entries store positions inside
        ``prefix_ids`` (not inside ``frozen + prefix``).  An anchor at
        ``len(prefix_ids)`` therefore lands at the right edge of the
        classified prefix, never inside the frozen baseline."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        state = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102),
            prefix_ids=[10, 11, 12],
            soft_token_stack=[3],  # anchor at end of prefix_ids
        )
        input_ids, _, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids],
                think_ids=think_ids,
                states=[state],
            )
        )
        # Layout: prompt(3) + think(2) + frozen(3) + prefix(3) + pad(1) = 12
        assert input_ids.shape == (1, 12)
        # Pad at position 11 (right edge), not inside the frozen baseline.
        assert st_positions == [[11]]
        # Prefix tokens still occupy [8..11) — the anchor=3 inside
        # prefix_ids did not bleed into the frozen region.
        assert input_ids[0, 8:11].tolist() == [10, 11, 12]
        # Frozen baseline untouched at [5..8).
        assert input_ids[0, 5:8].tolist() == [100, 101, 102]

    def test_soft_token_anchor_at_prefix_start_with_frozen(self):
        """An anchor at position 0 inside ``prefix_ids`` lands at the
        start of the classified prefix — i.e. immediately AFTER the
        frozen baseline, not at the start of the frozen baseline.  This
        is the natural answer-wave init case where the leak is at the
        first answer sentence (anchor = 0 in the empty answer-prefix)."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        state = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102),
            prefix_ids=[10, 11, 12],
            soft_token_stack=[0],  # anchor at START of prefix_ids
        )
        input_ids, _, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids],
                think_ids=think_ids,
                states=[state],
            )
        )
        # Layout: prompt(3) + think(2) + frozen(3) + pad(1) + prefix(3) = 12
        assert input_ids.shape == (1, 12)
        # Pad position = left_pad(0) + prompt(3) + think(2) + frozen(3)
        #              + block_start(0) = 8 — i.e. immediately after the
        # frozen baseline, before any prefix token.
        assert st_positions == [[8]]
        assert input_ids[0, 5:8].tolist() == [100, 101, 102]
        assert input_ids[0, 8].item() == self.pad_id
        assert input_ids[0, 9:12].tolist() == [10, 11, 12]

    def test_frozen_per_row_with_left_pad_offset(self):
        """At B=2 with rows of differing total length, each row's
        st_position offset accounts for its own frozen length AND its
        own left-pad amount."""
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        # Row 0: empty frozen, prefix len 2 → row_len = 3+2+0+2+1 = 8.
        # Row 1: 4-token frozen, prefix len 2 → row_len = 3+2+4+2+1 = 12.
        # max_total_len = 12; row 0 left-pad = 4.
        state0 = RegenSampleState(
            prefix_ids=[10, 11],
            soft_token_stack=[2],
        )
        state1 = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102, 103),
            prefix_ids=[20, 21],
            soft_token_stack=[2],
        )
        input_ids, _, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids, prompt_ids],
                think_ids=think_ids,
                states=[state0, state1],
            )
        )
        assert input_ids.shape == (2, 12)
        # Row 0: pos = left_pad(4) + prompt(3) + think(2) + frozen(0)
        #              + block_start(2) = 11
        # Row 1: pos = left_pad(0) + prompt(3) + think(2) + frozen(4)
        #              + block_start(2) = 11
        assert st_positions == [[11], [11]]
        # The two pads landing at the same absolute column (right edge of
        # the padded tensor) is the wave's typical layout — both rows
        # converge to "pad immediately after the classified prefix."

    # NOTE on "frozen never grows" enforcement: the field is typed as
    # ``tuple[int, ...]`` on RegenSampleState so accidental mutation
    # (``.append()`` / ``.extend()``) is a static type error.  No
    # runtime regression test is needed here — the type system catches
    # it.  An end-to-end "frozen survives the wave" test would also be
    # fine but adds no coverage beyond the static guarantee.

    def test_both_pad_stacks_combined_layout(self):
        """``frozen_soft_token_stack`` (CoT pads carried into the
        answer phase) splices into the frozen segment; ``soft_token_stack``
        (active answer-leak pads) splices into the prefix segment.
        ``_build_regen_inputs`` emits absolute positions for both,
        with frozen-segment pads first in input order followed by
        prefix-segment pads.
        """
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        # Frozen segment = 5 cot-baseline tokens with one CoT pad
        # anchored at position 3 inside the frozen segment.
        # Prefix segment = 3 answer-prefix tokens with one answer pad
        # anchored at the end (position 3 inside prefix).
        state = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102, 103, 104),
            frozen_soft_token_stack=(3,),
            prefix_ids=[200, 201, 202],
            soft_token_stack=[3],
        )
        input_ids, _, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids],
                think_ids=think_ids,
                states=[state],
            )
        )
        # Layout, byte-by-byte:
        #   prompt(3) | think(2) | frozen[0..3](3) | PAD_cot | frozen[3..5](2)
        #     | prefix[0..3](3) | PAD_answer
        # = 3 + 2 + 3 + 1 + 2 + 3 + 1 = 15
        assert input_ids.shape == (1, 15)
        assert input_ids[0, 0:3].tolist() == prompt_ids
        assert input_ids[0, 3:5].tolist() == think_ids
        assert input_ids[0, 5:8].tolist() == [100, 101, 102]
        assert input_ids[0, 8].item() == self.pad_id  # CoT pad
        assert input_ids[0, 9:11].tolist() == [103, 104]
        assert input_ids[0, 11:14].tolist() == [200, 201, 202]
        assert input_ids[0, 14].item() == self.pad_id  # answer pad

        # st_positions order: frozen-segment pads first, then prefix.
        # Frozen pad: base(0+3+2) + frozen_block_start(3) = 8
        # Prefix pad: base(0+3+2) + spliced_frozen_len(6) + prefix_block_start(3) = 14
        assert st_positions == [[8, 14]]

    def test_only_frozen_stack_no_prefix_stack(self):
        """Edge case: frozen pads present but no active answer pad.
        Useful for verifying the answer wave's "carry-only" path
        (e.g., a row whose answer-leak detection somehow yields no
        anchor — the answer wave always pushes one today, but the
        builder shouldn't depend on that).
        """
        prompt_ids = [1, 2, 3]
        think_ids = [4, 5]
        state = RegenSampleState(
            frozen_prefix_ids=(100, 101, 102),
            frozen_soft_token_stack=(1,),
            prefix_ids=[],  # empty answer prefix
            soft_token_stack=[],  # no active anchor
        )
        input_ids, _, st_positions = (
            self.engine._build_regen_inputs(
                prompt_ids_per_row=[prompt_ids],
                think_ids=think_ids,
                states=[state],
            )
        )
        # Layout: prompt(3) + think(2) + frozen[0..1](1) + PAD + frozen[1..3](2) = 9
        assert input_ids.shape == (1, 9)
        # Frozen pad at base(0+3+2) + 1 = 6
        assert st_positions == [[6]]


# ---------------------------------------------------------------------------
# Answer regen tests
# ---------------------------------------------------------------------------


class TestAnswerRegenPhase3:
    """Phase 3: clean CoT + leaking answer → _regenerate_leaking_sample is
    called with a non-leaking initial_result, skips CoT regen, and runs
    answer regen only."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_answer_regen_called_for_clean_cot(self):
        """Clean CoT + leaking answer → answer regenerated, no CoT regen."""
        from unittest.mock import MagicMock

        inner = DummyModel(self.tokenizer)
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)

        # Per-method counters:
        #   detect_batch[0] = initial CoT detect → clean
        #   detect_batch[1] = post-CoT-wave answer detect → leaking at fli=0
        #   detect_continuation[0] = answer-wave attempt → clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {1}},
            first_leak_index={"detect_batch": {1: 0}},
        )

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # Exactly 1 regenerate call (answer regen only, no CoT regen)
        assert len(calls) == 1
        # Answer soft token present at exactly one anchor.
        assert calls[0]["positions"] is not None
        assert len(calls[0]["positions"]) == 1

    def test_empty_cot_leaking_answer(self):
        """Empty CoT (immediate </think>) + leaking answer → answer regen."""
        from unittest.mock import MagicMock

        inner = DummyModel(self.tokenizer)
        inner.soft_token = MagicMock()

        # Override generate to produce empty CoT
        def empty_cot_generate(*args, **kwargs):
            kwargs.pop("prompts", None)
            kwargs.pop("generation_config", None)
            kwargs.pop("eos_token_id", None)
            kwargs.pop("pad_token_id", None)
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            batch_size = input_ids.shape[0]
            results = []
            for i in range(batch_size):
                prompt_ids = input_ids[i]
                canned = f"{THINK_SUFFIX}Leaked answer."
                response_ids = self.tokenizer.encode(canned, add_special_tokens=False)
                full_ids = prompt_ids.tolist() + response_ids
                results.append(full_ids)
            max_len = max(len(r) for r in results)
            padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
            return torch.tensor(padded)

        inner.generate = empty_cot_generate
        model = ReasoningModel(inner)

        # Per-method counters:
        #   detect_batch[0] = initial CoT (empty) → clean
        #   detect_batch[1] = post-CoT-wave answer detect → leaking at fli=0
        #   detect_continuation[0] = answer-wave attempt → clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {1}},
            first_leak_index={"detect_batch": {1: 0}},
        )

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # 1 regenerate call for answer regen
        assert len(calls) == 1
        assert calls[0]["positions"] is not None
        assert len(calls[0]["positions"]) == 1

    def test_answer_wave_carries_cot_soft_token_into_frozen(self):
        """After CoT regen pushes one CoT-phase pad onto its
        ``soft_token_stack``, the answer wave carries it forward via
        ``state.frozen_soft_token_stack``.  The answer-regen call's
        per-row soft-token schedule has BOTH the CoT pad (in the
        frozen segment) and the answer-leak pad (in prefix_ids).
        Combined: 1 + 1 = 2 positions.
        """
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)

        # Per-method counters:
        #   detect_batch[0]    = initial CoT detect → leaking at fli=1
        #   detect_continuation[0] = CoT-wave attempt → clean
        #   detect_batch[1]    = post-CoT-wave answer detect → leak at fli=0
        #   detect_continuation[1] = answer-wave attempt → clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {0, 1}},
            first_leak_index={"detect_batch": {0: 1, 1: 0}},
        )

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # 2 regenerate calls: 1 CoT-wave, 1 answer-wave.
        assert len(calls) == 2
        # CoT wave: 1 soft token at the CoT leak anchor.
        assert len(calls[0]["positions"]) == 1
        # Answer wave: 1 carried-forward CoT pad (in frozen segment) +
        # 1 fresh answer-leak pad (in prefix segment) = 2 positions.
        # Frozen-segment pad precedes prefix-segment pad in input order
        # (carryover anchors are emitted first in _build_regen_inputs).
        assert len(calls[1]["positions"]) == 2
        assert calls[1]["positions"][0] < calls[1]["positions"][1], (
            f"Frozen-segment carried CoT pad should precede "
            f"prefix-segment answer pad: {calls[1]['positions']}"
        )

    def test_cot_regen_exhausted_then_answer_wave_carries_full_stack(self):
        """CoT wave exhausts attempts (initial anchor + N same-anchor
        escalations); the answer wave then carries the full CoT
        escalation stack forward via ``frozen_soft_token_stack`` and
        adds one fresh answer-leak pad on top.  The answer-regen
        per-row schedule has all CoT pads (in the frozen segment)
        followed by the answer pad (in the prefix segment).
        """
        from unittest.mock import MagicMock

        max_attempts = 2
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)

        # Per-method counters:
        #   detect_batch[0]    = initial CoT detect → leaking at fli=1
        #   detect_continuation[0] = CoT iter 0 → still leaking at fli=1
        #   detect_continuation[1] = CoT iter 1 → still leaking (exhausted)
        #   detect_batch[1]    = post-CoT-wave answer detect → leak at fli=0
        #   detect_continuation[2] = answer-wave attempt → clean
        detector = MockLeakDetector(
            leak_on_calls={
                "detect_batch": {0, 1},
                "detect_continuation": {0, 1},
            },
            first_leak_index={
                "detect_batch": {0: 1, 1: 0},
                "detect_continuation": {0: 1, 1: 1},
            },
        )

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            # Unwrap row 0 of the per-row schedule (these tests run at B=1)
            # so existing flat-list assertions stay valid.
            calls.append({
                "positions": soft_token_positions[0] if soft_token_positions else None,
            })
            return original_regenerate(soft_token_positions=soft_token_positions, **kwargs)

        inner.regenerate = tracking_regenerate

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # max_attempts CoT-wave calls + 1 answer-wave call.
        assert len(calls) == max_attempts + 1
        # CoT wave: stack grows (1, 2 positions on consecutive escalations).
        assert len(calls[0]["positions"]) == 1
        assert len(calls[1]["positions"]) == 2
        # CoT exhausts with 2 entries on the stack: the initial push at
        # state init plus one same-anchor escalation push during the
        # wave's first iteration.  The exhaust iteration itself does
        # NOT push a third pad — the wave loop's exhaust-and-advance
        # branch skips the same-anchor push since it would land at a
        # position the prefix is about to grow past (or, as here when
        # the post-leak tail is clean, it would never be re-anchored
        # against).  Answer wave carries those 2 frozen pads forward and
        # adds 1 fresh answer-leak anchor.  Total: 2 + 1 = 3 positions.
        assert len(calls[2]["positions"]) == 3
        # Frozen-segment pads (2 of them) precede prefix-segment pad.
        frozen_pads = calls[2]["positions"][:2]
        answer_pad = calls[2]["positions"][2]
        assert all(p < answer_pad for p in frozen_pads), (
            f"All carried CoT pads (frozen segment) should precede "
            f"the answer pad (prefix segment): {calls[2]['positions']}"
        )

    def test_clean_answer_no_regen(self):
        """Clean CoT + clean answer → no regeneration at all."""
        from unittest.mock import MagicMock

        inner = DummyModel(self.tokenizer)
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)

        detector = MockLeakDetector(leak_on_calls=set())

        calls = []
        original_regenerate = inner.regenerate

        def tracking_regenerate(soft_token_positions=None, **kwargs):
            calls.append(True)
            return original_regenerate(**kwargs)

        inner.regenerate = tracking_regenerate

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        assert len(calls) == 0


class TestAnswerWaveOverBudgetPreservesClearedPrefix:
    """Answer-wave over-budget drop (``remaining_budget <= 0`` at the
    top of an iteration) currently throws away the cleared answer
    prefix in ``state.clean_prefix`` by clobbering
    ``state.best_answer`` to ``""``.

    The CoT phase preserves cleared work (``best_cot = clean_prefix``);
    the answer phase should mirror that semantic and set
    ``best_answer = clean_prefix`` instead of ``""``.  Otherwise rows
    that legitimately forward-extend through many cleared answer
    sentences before exceeding the token budget ship empty answers and
    discard real, non-leaking, generated answer text.

    Distinct from the attempts-exhausted path (same-anchor escalation
    reaching ``regen_max_attempts``), which already preserves the last
    iteration's assembled ``best_answer`` and is not affected.

    Observed in production on forget10 samples 118 (135 cleared answer
    sentences discarded) and 165 (76 cleared answer sentences discarded).
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_cleared_answer_prefix_survives_over_budget_drop(self):
        """A row entering the answer wave with non-trivial
        ``clean_prefix`` and ``remaining_budget <= 0`` should exit with
        ``best_answer == clean_prefix``, not ``""``.
        """
        from unittest.mock import MagicMock

        inner = DummyModel(self.tokenizer)
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)
        # No detector activity in this test — the wave loop's top-of-iter
        # over-budget check fires before any detect call would run.
        detector = MockLeakDetector(leak_on_calls=set())

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )

        # Hand-built answer-wave state with a cleared answer prefix that
        # has already consumed the entire budget.  Mirrors the production
        # case where forward extension grew prefix_ids past
        # ``total_budget - frozen_len`` (via re-tokenization drift in the
        # real engine; here arranged explicitly).
        cleared_prefix_text = (
            "First cleared answer sentence. Second cleared answer sentence."
        )
        cleared_sentences = [
            "First cleared answer sentence.",
            "Second cleared answer sentence.",
        ]
        cleared_ids = self.tokenizer(
            cleared_prefix_text, add_special_tokens=False
        )["input_ids"]

        cot_text = "Initial CoT goes here."
        cot_ids = self.tokenizer(cot_text, add_special_tokens=False)["input_ids"]
        delim_ids = self.tokenizer(
            THINK_SUFFIX, add_special_tokens=False
        )["input_ids"]
        frozen_ids = tuple(cot_ids + delim_ids)

        # Set total_budget so ``remaining_budget = total - prefix - frozen
        # = -1`` on entry; the wave loop's first top-of-iter check fires
        # the over-budget branch and the loop exits immediately.
        inner.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=len(cleared_ids) + len(frozen_ids) - 1,
            use_cache=False,
        )

        state = RegenSampleState(
            best_cot=cot_text,
            # Sentinel: any post-drop best_answer that isn't
            # ``clean_prefix`` is wrong; this initial value lets us
            # distinguish "untouched" from "clobbered" in failure
            # messages, though correctness is the equality below.
            best_answer="<initial leaking answer placeholder>",
            prefix_sentences=cleared_sentences,
            clean_prefix=cleared_prefix_text,
            prefix_ids=list(cleared_ids),
            frozen_prefix_ids=frozen_ids,
        )
        state.push_anchor(len(cleared_ids))

        # Sanity: confirm the budget arithmetic that drives the test.
        total_budget = inner.generation_config.max_new_tokens
        assert state.remaining_budget(total_budget) <= 0, (
            f"Test setup error: state.remaining_budget="
            f"{state.remaining_budget(total_budget)}, expected <= 0 "
            f"so the over-budget branch fires on entry."
        )

        engine._run_wave_loop(
            prompts=["irrelevant prompt"],
            states=[state],
            sample_idxs=[0],
            phase_label="answer",
            gen_time_attr="_answer_regen_gen_time",
            assemble_fn=engine._assemble_answer_iteration_output,
        )

        # The fix: on answer-phase over-budget drop, ``best_answer`` is
        # set to ``clean_prefix`` (mirror of CoT phase's
        # ``best_cot = clean_prefix``).  Currently fails because the
        # unconditional ``state.best_answer = ""`` clobber discards the
        # cleared prefix.
        assert state.best_answer == cleared_prefix_text, (
            f"Answer-wave over-budget drop discarded cleared prefix. "
            f"Expected best_answer={cleared_prefix_text!r}, got "
            f"{state.best_answer!r}.  The unconditional "
            f"``state.best_answer = ''`` in the over-budget branch "
            f"throws away ~all forward-extended answer work."
        )
        # CoT must NOT be touched in the answer phase (it's frozen).
        assert state.best_cot == cot_text, (
            f"Answer-phase over-budget drop mutated best_cot: expected "
            f"{cot_text!r}, got {state.best_cot!r}"
        )


class TestForceClearAndAdvanceOnExhaust:
    """When same-anchor regen reaches ``regen_max_attempts`` at a single
    anchor, the wave loop force-clears the persistent leak (and any
    post-leak clean intermediates that the classifier had already
    confirmed clean in the same iteration's per-sentence output) into
    the prefix and re-anchors at the next leak — instead of dropping
    the row from ``next_active`` and letting the leaking output through.

    Tests cover:
      * Persistent leak followed by clean tail — empty walk, no state
        mutation; row exits with assemble_fn's output preserved.
      * Persistent leak followed *immediately* by another leak — the
        force-clear absorbs only the persistent leak and re-anchors at
        the next one.
      * Persistent leak followed by clean intermediates then another
        leak — all cleared sentences absorbed in a single tokenize call
        (mirrors forward-extension's BPE-stable pattern).
      * Force-clear in the answer phase — the carried
        ``frozen_soft_token_stack`` from the CoT wave is NOT mutated;
        the new anchor goes into the active ``soft_token_stack``.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    class _ScriptedDetector:
        """Hand-scripted leak detector returning canned per-sentence
        confirmed flags.  Each ``detect_continuation`` / ``detect_batch``
        call pops the next entry from ``script``; missing entries default
        to "no leak".

        Each script entry is one of:
          * ``None`` — non-leaking result over the call's sentences.
          * ``(first_leak_index, confirmed_flags)`` — leaking result with
            the given index and a full per-sentence confirmed-leak flag
            list (False for prefix sentences and clean new sentences,
            True for leaks).  Caller is responsible for shape
            consistency.
        """

        def __init__(self, script):
            from collections import deque
            self._script = deque(script)

        def detect(self, generated_cot, prompt=None, external_prefix_sentences=None):
            del prompt, external_prefix_sentences
            from redact.evaluator.utils import split_sentences
            sentences = split_sentences(generated_cot)
            return self._build(sentences=sentences, n_prefix=0)

        def detect_continuation(
            self, prefix_sentences, new_text,
            prompt=None, external_prefix_sentences=None,
        ):
            del prompt, external_prefix_sentences
            from redact.evaluator.utils import split_sentences
            new_sentences = split_sentences(new_text)
            sentences = list(prefix_sentences) + new_sentences
            return self._build(
                sentences=sentences, n_prefix=len(prefix_sentences),
            )

        def detect_batch(
            self, cots, prompts=None, external_prefix_sentences_per_row=None,
        ):
            del prompts, external_prefix_sentences_per_row
            from redact.evaluator.utils import split_sentences
            return [
                self._build(sentences=split_sentences(c), n_prefix=0)
                for c in cots
            ]

        def _build(self, sentences, n_prefix):
            entry = self._script.popleft() if self._script else None
            n = len(sentences)
            if entry is None:
                return LeakDetectionResult(
                    is_leaking=False, first_leak_index=None,
                    sentences=sentences,
                    stage1_flags=[False] * n,
                    confirmed_flags=[False] * n,
                )
            fli, cf = entry
            assert len(cf) == n, (
                f"_ScriptedDetector: confirmed_flags length "
                f"({len(cf)}) does not match sentences length ({n}); "
                f"adjust the script for n_prefix={n_prefix}."
            )
            assert cf[fli], (
                f"_ScriptedDetector: confirmed_flags[{fli}] is False "
                f"but first_leak_index={fli}; the leaking entry must mark "
                f"its own first-leak slot."
            )
            confirmed = list(cf)
            return LeakDetectionResult(
                is_leaking=True,
                first_leak_index=fli,
                sentences=sentences,
                stage1_flags=list(confirmed),
                confirmed_flags=confirmed,
            )

    class _CannedRegenModel(DummyModel):
        """``DummyModel`` whose ``regenerate`` emits a configurable
        canned text past the input.  The wave loop slices past
        ``input_ids.shape[1]`` to recover new tokens; the canned text is
        encoded and appended to the prompt so the slice produces it
        verbatim.
        """

        def __init__(self, tokenizer, canned_text):
            super().__init__(tokenizer)
            self.canned_text = canned_text

        def regenerate(self, soft_token_positions=None, **kwargs):
            self.remove_hooks()
            kwargs.pop("prompts", None)
            kwargs.pop("generation_config", None)
            kwargs.pop("eos_token_id", None)
            kwargs.pop("pad_token_id", None)
            input_ids = kwargs["input_ids"]
            response_ids = self.tokenizer.encode(
                self.canned_text, add_special_tokens=False
            )
            results = []
            for i in range(input_ids.shape[0]):
                full = input_ids[i].tolist() + response_ids
                results.append(full)
            max_len = max(len(r) for r in results)
            padded = [
                r + [self.tokenizer.pad_token_id] * (max_len - len(r))
                for r in results
            ]
            return torch.tensor(padded)

    def _make_engine(
        self, *, max_attempts, detector, canned_text, max_new_tokens=512,
    ):
        from unittest.mock import MagicMock
        inner = self._CannedRegenModel(self.tokenizer, canned_text=canned_text)
        inner.soft_token = MagicMock()
        inner.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=False,
        )
        model = ReasoningModel(inner)
        return RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=max_attempts,
            batch_size=1,
        )

    def test_persistent_leak_with_clean_tail_drops_row_no_mutation(self):
        """Exhaust at the persistent leak with a clean post-leak tail —
        the walk finds no further leak, so the row drops from
        ``next_active`` without mutating ``prefix_sentences``.
        ``best_cot`` was already finalized by ``assemble_fn`` to the
        original output (clean prefix + persistent leak + clean tail).
        """
        # Canned regen produces three sentences before the </think>
        # delimiter; the assemble_fn splits at </think> and passes the
        # CoT portion (3 sentences) to detect_continuation.
        canned = (
            "Persistent leaking sentence. Clean intermediate one. "
            "Clean intermediate two." + THINK_SUFFIX + "Some answer."
        )
        # Iteration 0 detect_continuation: leaking at fli=0 with two
        # post-leak sentences scored clean.  ``regen_max_attempts=1``
        # means this single iteration triggers exhaust immediately.
        script = [
            (
                0,
                [True, False, False],
            ),
        ]
        detector = self._ScriptedDetector(script)
        engine = self._make_engine(
            max_attempts=1, detector=detector, canned_text=canned,
        )

        state = RegenSampleState(
            best_cot="(initial leaking cot placeholder)",
            best_answer="(answer)",
            prefix_sentences=[],
            clean_prefix="",
            prefix_ids=[],
        )
        state.push_anchor(0)

        engine._force_cleared_leak_total = 0
        engine._cot_regen_gen_time = 0.0
        engine._total_gen_time = 0.0
        engine._total_detect_time = 0.0

        engine._run_wave_loop(
            prompts=["irrelevant"],
            states=[state],
            sample_idxs=[0],
            phase_label="cot",
            gen_time_attr="_cot_regen_gen_time",
            assemble_fn=engine._assemble_cot_iteration_output,
        )

        # No state mutation: prefix stays empty, no force-clear counter
        # bump, no extra anchor push.
        assert state.prefix_sentences == []
        assert state.clean_prefix == ""
        assert state.prefix_ids == []
        assert state.n_force_cleared_leaks == 0
        assert engine._force_cleared_leak_total == 0
        # The single anchor pushed at state init is still there; the
        # exhaust branch did NOT push an additional same-anchor pad
        # (W8 — wasted-push elimination).
        assert state.soft_token_stack == [0]
        # ``best_cot`` was set by ``_assemble_cot_iteration_output`` to
        # ``clean_prefix + new_continuation_text`` BEFORE the force-
        # clear branch ran, where ``clean_prefix`` was empty and
        # ``new_continuation_text`` was the canned regen's pre-</think>
        # portion (3 sentences).  The W3 / step-3d behaviour asserts
        # that this output is preserved on exit — no state mutation,
        # original output rides through.  Punkt strips inter-sentence
        # whitespace on split, then the assemble_fn rejoins with
        # single spaces; the canned source already used single spaces
        # so the round-trip is byte-identical.
        assert state.best_cot == (
            "Persistent leaking sentence. Clean intermediate one. "
            "Clean intermediate two."
        )
        assert state.best_answer == "Some answer."

    def test_persistent_leak_immediately_followed_by_next_leak(self):
        """Exhaust at the persistent leak with the very next sentence
        also flagged.  The force-clear absorbs only the persistent
        leak and re-anchors at the next leak's position.
        """
        # Two-sentence CoT — first iteration's regen produces both
        # sentences; iteration 1 (post-force-clear) regenerates after
        # the absorbed persistent leak.  Same canned text is reused for
        # both regens because the dummy model is stateless; the second
        # iteration's continuation will Punkt-split into 2 sentences
        # again, and the detector's None script entry returns
        # non-leaking so the row converges.
        canned = (
            "Persistent leaking sentence. Next leaking sentence."
            + THINK_SUFFIX + "Some answer."
        )
        # Persistent leak at sentence 0, next leak immediately at
        # sentence 1.
        script = [
            (
                0,
                [True, True],
            ),
        ]
        detector = self._ScriptedDetector(script)
        # max_attempts=1 → first iteration immediately exhausts.
        engine = self._make_engine(
            max_attempts=1, detector=detector, canned_text=canned,
        )

        state = RegenSampleState(
            best_cot="(initial)",
            best_answer="(answer)",
            prefix_sentences=[],
            clean_prefix="",
            prefix_ids=[],
        )
        state.push_anchor(0)
        engine._force_cleared_leak_total = 0
        engine._cot_regen_gen_time = 0.0
        engine._total_gen_time = 0.0
        engine._total_detect_time = 0.0

        # We need to terminate the loop after the force-clear push
        # without relying on a bounded number of follow-up iterations.
        # The simplest path: max_new_tokens is large enough that the
        # follow-up iteration regenerates from the new anchor; with no
        # further script entries the detector returns non-leaking, and
        # the row converges.  Push a non-leaking sentinel for the
        # next-iteration call.
        detector._script.append(None)

        engine._run_wave_loop(
            prompts=["irrelevant"],
            states=[state],
            sample_idxs=[0],
            phase_label="cot",
            gen_time_attr="_cot_regen_gen_time",
            assemble_fn=engine._assemble_cot_iteration_output,
        )

        # Persistent leak (sentence 0) absorbed; new anchor pushed at
        # end of extended prefix_ids.
        assert len(state.prefix_sentences) == 1
        assert state.n_force_cleared_leaks == 1
        assert engine._force_cleared_leak_total == 1
        assert state.first_leak_index == 1
        # Anchor stack: original push at 0 + new anchor at the absorbed
        # prefix's end.
        assert state.soft_token_stack[0] == 0
        assert state.soft_token_stack[-1] == len(state.prefix_ids)
        # Same-anchor escalation pad NOT pushed on the exhaust iteration.
        assert len(state.soft_token_stack) == 2

    def test_persistent_leak_with_clean_intermediates_then_next_leak(self):
        """Persistent leak, clean intermediate sentences, then the next
        leak.  All absorbed sentences must be appended to ``prefix_ids``
        via a SINGLE tokenize call on the joined text — same pattern as
        forward extension at ``inference_regen.py:1395``.  Multiple
        per-sentence tokenize calls would accumulate BPE-context drift
        at sentence boundaries.
        """
        canned = (
            "Persistent leaking sentence. First clean intermediate. "
            "Second clean intermediate. Next leaking sentence."
            + THINK_SUFFIX + "Some answer."
        )
        script = [
            (
                0,
                [True, False, False, True],
            ),
        ]
        detector = self._ScriptedDetector(script)
        engine = self._make_engine(
            max_attempts=1, detector=detector, canned_text=canned,
        )

        state = RegenSampleState(
            best_cot="(initial)",
            best_answer="(answer)",
            prefix_sentences=[],
            clean_prefix="",
            prefix_ids=[],
        )
        state.push_anchor(0)
        engine._force_cleared_leak_total = 0
        engine._cot_regen_gen_time = 0.0
        engine._total_gen_time = 0.0
        engine._total_detect_time = 0.0
        # Allow the post-force-clear iteration to converge.
        detector._script.append(None)

        engine._run_wave_loop(
            prompts=["irrelevant"],
            states=[state],
            sample_idxs=[0],
            phase_label="cot",
            gen_time_attr="_cot_regen_gen_time",
            assemble_fn=engine._assemble_cot_iteration_output,
        )

        # Three sentences absorbed: persistent leak + 2 clean intermediates.
        assert len(state.prefix_sentences) == 3
        assert state.n_force_cleared_leaks == 1
        assert engine._force_cleared_leak_total == 1
        assert state.first_leak_index == 3
        # BPE-stable absorption: the appended ``prefix_ids`` segment
        # must equal a single tokenize call on the joined absorbed text
        # with the conditional leading space — NOT a per-sentence
        # concatenation.  The expected joined text is what the wave
        # loop would have built.
        absorbed_text = " ".join(state.prefix_sentences)
        # ``state.clean_prefix`` was empty before absorption, so no
        # leading space prepends.
        expected_ids = self.tokenizer(
            absorbed_text, add_special_tokens=False
        )["input_ids"]
        assert state.prefix_ids == expected_ids
        assert state.clean_prefix == absorbed_text

    def test_persistent_leak_in_answer_wave_preserves_frozen_stack(self):
        """Force-clear during the answer phase must mutate only the
        active ``soft_token_stack``, never the immutable
        ``frozen_soft_token_stack`` carried over from the CoT wave.
        """
        # Answer phase: assemble_fn treats whole decoded text as the
        # new continuation (no </think> split).  Two-sentence answer
        # text gives the script a 2-element confirmed_flags list.
        canned = "Persistent answer leak. Next answer leak."
        script = [
            (
                0,
                [True, True],
            ),
        ]
        detector = self._ScriptedDetector(script)
        engine = self._make_engine(
            max_attempts=1, detector=detector, canned_text=canned,
        )

        # Hand-build an answer-wave state with a non-empty frozen
        # baseline and CoT-phase pad carryover.  Mirrors the answer
        # wave's init for a Phase-2-origin row.
        cot_text = "Cot context."
        cot_ids = self.tokenizer(
            cot_text, add_special_tokens=False
        )["input_ids"]
        delim_ids = self.tokenizer(
            THINK_SUFFIX, add_special_tokens=False
        )["input_ids"]
        frozen_ids = tuple(cot_ids + delim_ids)
        # CoT pads sit at concrete positions inside frozen_ids.
        frozen_pads = (0, len(cot_ids))

        state = RegenSampleState(
            best_cot=cot_text,
            best_answer="(initial answer placeholder)",
            prefix_sentences=[],
            clean_prefix="",
            prefix_ids=[],
            frozen_prefix_ids=frozen_ids,
            frozen_soft_token_stack=frozen_pads,
        )
        state.push_anchor(0)
        engine._force_cleared_leak_total = 0
        engine._answer_regen_gen_time = 0.0
        engine._total_gen_time = 0.0
        engine._total_detect_time = 0.0
        detector._script.append(None)

        engine._run_wave_loop(
            prompts=["irrelevant"],
            states=[state],
            sample_idxs=[0],
            phase_label="answer",
            gen_time_attr="_answer_regen_gen_time",
            assemble_fn=engine._assemble_answer_iteration_output,
        )

        # Frozen baseline is untouched (``tuple`` immutability is a
        # static guarantee on the dataclass; this is a defence-in-depth
        # check against accidental rebinding).
        assert state.frozen_prefix_ids == frozen_ids
        assert state.frozen_soft_token_stack == frozen_pads
        # Active stack grew: original push at state init + force-clear
        # re-anchor (no wasted same-anchor pad on exhaust).
        assert len(state.soft_token_stack) == 2
        assert state.soft_token_stack[0] == 0
        assert state.soft_token_stack[-1] == len(state.prefix_ids)
        assert state.n_force_cleared_leaks == 1
        assert engine._force_cleared_leak_total == 1

    def test_force_clear_walks_confirmed_flags(self):
        """The exhaust-and-advance walk advances on ``confirmed_flags``
        alone.  Uses an independent inline detector (rather than
        ``_ScriptedDetector``) so the walk is pinned against a second
        implementation of the detector contract.
        """
        canned = (
            "Persistent leaking sentence. Next leaking sentence."
            + THINK_SUFFIX + "Some answer."
        )

        class _BinaryDetector:
            """Inline detector with ``confirmed_flags`` set explicitly,
            mirroring what the production ``_classify_sentences`` emits
            under :class:`ContextualBinaryLeakClassifier`.
            """

            def __init__(self):
                self._calls = 0

            def detect(self, generated_cot, prompt=None,
                       external_prefix_sentences=None):
                return self.detect_continuation(
                    [], generated_cot, prompt=prompt,
                    external_prefix_sentences=external_prefix_sentences,
                )

            def detect_continuation(
                self, prefix_sentences, new_text,
                prompt=None, external_prefix_sentences=None,
            ):
                del prompt, external_prefix_sentences
                from redact.evaluator.utils import split_sentences
                new_sentences = split_sentences(new_text)
                sentences = list(prefix_sentences) + new_sentences
                self._calls += 1
                n = len(sentences)
                n_prefix = len(prefix_sentences)
                if self._calls == 1:
                    # Two confirmed leaks back-to-back.
                    confirmed = (
                        [False] * n_prefix + [True] * (n - n_prefix)
                    )
                    return LeakDetectionResult(
                        is_leaking=True,
                        first_leak_index=n_prefix,
                        sentences=sentences,
                        stage1_flags=list(confirmed),
                        confirmed_flags=confirmed,
                    )
                # Iter 1+: clean — converges and the row exits.
                return LeakDetectionResult(
                    is_leaking=False, first_leak_index=None,
                    sentences=sentences,
                    stage1_flags=[False] * n,
                    confirmed_flags=[False] * n,
                )

            def detect_batch(
                self, cots, prompts=None,
                external_prefix_sentences_per_row=None,
            ):
                del prompts, external_prefix_sentences_per_row
                return [self.detect(c) for c in cots]

        detector = _BinaryDetector()
        engine = self._make_engine(
            max_attempts=1, detector=detector, canned_text=canned,
        )

        state = RegenSampleState(
            best_cot="(initial leaking cot placeholder)",
            best_answer="(answer)",
            prefix_sentences=[],
            clean_prefix="",
            prefix_ids=[],
        )
        state.push_anchor(0)

        engine._force_cleared_leak_total = 0
        engine._cot_regen_gen_time = 0.0
        engine._total_gen_time = 0.0
        engine._total_detect_time = 0.0

        engine._run_wave_loop(
            prompts=["irrelevant"],
            states=[state],
            sample_idxs=[0],
            phase_label="cot",
            gen_time_attr="_cot_regen_gen_time",
            assemble_fn=engine._assemble_cot_iteration_output,
        )

        # Persistent leak (sentence 0) absorbed.
        assert len(state.prefix_sentences) == 1
        assert state.n_force_cleared_leaks == 1
        assert engine._force_cleared_leak_total == 1
        assert state.first_leak_index == 1
        # Original push at 0 + new force-clear re-anchor at end of
        # extended prefix_ids.
        assert state.soft_token_stack[0] == 0
        assert state.soft_token_stack[-1] == len(state.prefix_ids)
        assert len(state.soft_token_stack) == 2


class TestRegenBudgetCap:
    """``_regenerate_sample`` caps each call's ``max_new_tokens`` at
    ``total_budget - len(prefix_ids)`` so the final accepted output
    cannot exceed ``generation_config.max_new_tokens``.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def _capture_max_new_tokens(self, inner):
        """Wrap ``inner.regenerate`` to record each call's effective
        ``max_new_tokens``, sourced from the (potentially modified)
        ``generation_config`` the engine passes in.  Falls back to the
        kwarg form for forward-compat with any caller that bypasses
        the config copy."""
        captured = []
        original = inner.regenerate

        def tracking(soft_token_positions=None, **kwargs):
            gen_config = kwargs.get("generation_config")
            if gen_config is not None:
                captured.append(gen_config.max_new_tokens)
            else:
                captured.append(kwargs.get("max_new_tokens"))
            return original(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking
        return captured

    def test_per_call_budget_equals_total_minus_prefix(self):
        """Each regen call passes ``max_new_tokens = total - len(prefix_ids)``.

        Same-anchor escalation keeps the prefix fixed, so every CoT-phase
        regen attempt should receive an identical budget equal to
        ``total_budget - len(tokenize(clean_prefix))``.
        """
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        captured = self._capture_max_new_tokens(inner)

        model = ReasoningModel(inner)
        # Flag the CoT detect_batch (call 0) and the 3 detect_continuations
        # (calls 1-3); leave the post-CoT answer detect (call 4) clean so
        # answer regen short-circuits without contributing extra
        # regenerate() calls to ``captured``.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        total_budget = inner.generation_config.max_new_tokens
        prefix_text = "Safe start."
        prefix_tokens = len(
            self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        )
        expected = total_budget - prefix_tokens

        # All same-anchor CoT-phase regens (max_attempts of them) should
        # receive the same budget — the prefix doesn't grow.
        assert len(captured) >= 1, "regenerate() was never called"
        for i, b in enumerate(captured):
            assert b == expected, (
                f"call {i}: expected max_new_tokens={expected} "
                f"(total {total_budget} − prefix {prefix_tokens}), got {b}"
            )

    def test_zero_budget_skips_generate(self):
        """When ``len(prefix_ids) >= total_budget``, regen returns without
        calling ``inner.regenerate``."""
        # Use a tiny budget so that even a short clean prefix exceeds it.
        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
            clean_cot="Clean.",
        )
        # Override the budget AFTER construction so the initial generation
        # call (which produced the leaking_cot) wasn't already truncated.
        # The initial pass uses ``self.model.generate()`` which reads this
        # config at call time; ``ReasoningModel`` stores ``inner`` by
        # reference so post-construction mutation is visible.
        # NB: the LeakingDummyModel.generate() ignores max_new_tokens by
        # design (it returns canned text), so the initial CoT is full-length
        # regardless of budget.  This is essential — the test needs a
        # multi-sentence initial CoT for the leak path to fire.
        inner.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=2,  # smaller than even "Safe start." (3+ tokens)
            use_cache=False,
        )
        captured = self._capture_max_new_tokens(inner)

        model = ReasoningModel(inner)
        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # Sanity check: the regen path was actually entered.  detect()
        # is called once on the initial CoT.  Under the wave-level
        # engine, the over-budget row is filtered out at the top of the
        # wave loop *before* any redundant detect_continuation call —
        # so seeing exactly 1 detector call means the wave fired and
        # immediately recognized the row had no remaining budget.  If a
        # future refactor caused the initial generate to respect
        # ``max_new_tokens=2``, the detector might not flag (initial CoT
        # too short for sentence index 1), so ``len(captured) == 0``
        # would pass for the wrong reason — checking the leak-on-call-0
        # mock saw its flag fire (i.e. count == 1, not 0) catches that.
        assert detector._detect_count == 1, (
            f"Expected exactly 1 detector call (initial detect that "
            f"flags the leak; over-budget bail-out skips "
            f"detect_continuation), got {detector._detect_count}.  "
            f"If 0, the initial CoT was truncated below the leak "
            f"sentence and the test is no longer exercising regen."
        )

        # Prefix "Safe start." tokenizes to >2 tokens, so remaining_budget
        # is <= 0 and inner.regenerate() must not be called at all.
        assert len(captured) == 0, (
            f"Expected 0 regenerate() calls when prefix exceeds budget, "
            f"got {len(captured)}"
        )

    def test_e2e_cap_under_same_anchor_escalation(self):
        """End-to-end with a budget-respecting model: same-anchor
        escalation across all attempts should still leave the final
        accepted output at most ``total_budget`` tokens long.

        Without the cap, the model's canned ``clean_cot`` (~50 short
        sentences, well above the 32-token budget) would be appended
        to the prefix on every attempt and the final output would
        easily exceed the budget.
        """
        long_clean = " ".join([f"Word{i}." for i in range(50)])
        inner = BudgetRespectingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
            clean_cot=long_clean,
        )
        inner.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=32, use_cache=False,
        )
        model = ReasoningModel(inner)
        # Always leak at the first new sentence; same-anchor escalation
        # for max_attempts iterations.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3, 4, 5}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        key = "rtofu_forget10"
        cot = engine.cot_generations[key]["generated"][0]
        ans = engine.answer_generations[key]["generated"][0]
        full = cot + (THINK_SUFFIX + ans if ans else "")
        total_tokens = len(
            self.tokenizer(full, add_special_tokens=False)["input_ids"]
        )
        budget = inner.generation_config.max_new_tokens

        # Slack: BPE encode→decode→encode roundtripping plus the engine's
        # ``clean_prefix + " " + new_continuation_text`` join can shift counts by a
        # few tokens.  Without the cap, total_tokens would be ~70+.
        assert total_tokens <= budget + 5, (
            f"Final output {total_tokens} tokens exceeds budget {budget}"
        )

    def test_e2e_cap_under_forward_extension(self):
        """End-to-end forward extension: each regen clears the prior leak
        but the detector reports a new leak one sentence further along.

        Without the cap, ``attempt`` resets to 0 on every forward move,
        so the loop runs indefinitely (this is the unbounded-extension
        bug the cap fixes — observed as a 24K-token sample in production).
        With the cap, the final output must fit in the budget regardless
        of how many forward extensions fire.

        Uses a purpose-built mock detector instead of ``MockLeakDetector``
        because the dict-based forward-extension schedule there asserts
        ``len(sentences) > fli``, which fails once truncation reduces the
        new continuation below two sentences — well before the cap takes
        over.  The custom detector adaptively reports
        ``first_leak_index = n_prefix + 1`` whenever the new continuation
        has at least two sentences, and "not leaking" otherwise; the loop
        terminates via either the cap (``remaining_budget <= 0`` →
        empty new text → ``n_new == 0`` → not leaking) or the
        not-enough-sentences fallback.  The cap's role here is bounding
        the final accepted output, not necessarily terminating the loop.
        """
        from redact.regen.leak_detector import LeakDetectionResult
        from redact.evaluator.utils import split_sentences

        class ForwardExtendingDetector:
            """Reports leak at ``n_prefix + 1`` whenever ≥2 new sentences are
            available, "not leaking" otherwise.  Drives forward extension
            on every iteration without depending on a precomputed fli
            schedule that can outrun the actual sentence count."""

            def __init__(self):
                self._detect_count = 0

            def detect(self, cot, prompt=None, external_prefix_sentences=None):
                del prompt, external_prefix_sentences  # protocol kwargs, unused
                sentences = split_sentences(cot)
                return self._build(sentences, n_prefix=0)

            def detect_continuation(
                self,
                prefix_sentences,
                new_text,
                prompt=None,
                external_prefix_sentences=None,
            ):
                del prompt, external_prefix_sentences  # protocol kwargs, unused
                new_sentences = split_sentences(new_text)
                return self._build(
                    list(prefix_sentences) + new_sentences,
                    n_prefix=len(prefix_sentences),
                )

            def detect_batch(
                self, cots, prompts=None, external_prefix_sentences_per_row=None,
            ):
                del prompts, external_prefix_sentences_per_row
                return [self.detect(c) for c in cots]

            def _build(self, sentences, n_prefix):
                self._detect_count += 1
                n_new = len(sentences) - n_prefix
                if n_new < 2:
                    return LeakDetectionResult(
                        is_leaking=False, first_leak_index=None,
                        sentences=sentences,
                        stage1_flags=[False] * len(sentences),
                        confirmed_flags=[False] * len(sentences),
                    )
                fli = n_prefix + 1
                flags = [False] * len(sentences)
                flags[fli] = True
                return LeakDetectionResult(
                    is_leaking=True, first_leak_index=fli,
                    sentences=sentences,
                    stage1_flags=flags, confirmed_flags=flags,
                )

        # Many short sentences so forward extensions can fire several
        # times before truncation reduces ``new_sentences`` below two.
        sentences = " ".join([f"X{i}." for i in range(30)])
        inner = BudgetRespectingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
            clean_cot=sentences,
        )
        inner.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=32, use_cache=False,
        )
        model = ReasoningModel(inner)
        detector = ForwardExtendingDetector()

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        key = "rtofu_forget10"
        cot = engine.cot_generations[key]["generated"][0]
        ans = engine.answer_generations[key]["generated"][0]
        full = cot + (THINK_SUFFIX + ans if ans else "")
        total_tokens = len(
            self.tokenizer(full, add_special_tokens=False)["input_ids"]
        )
        budget = inner.generation_config.max_new_tokens

        # 1) Final accepted output fits in the budget.  Without the cap,
        #    repeated forward extensions would push this far past 32.
        assert total_tokens <= budget + 5, (
            f"Final output {total_tokens} tokens exceeds budget {budget}"
        )

        # 2) Loop terminated in a finite number of calls.  Without the
        #    cap, attempt resets indefinitely and this would hang.
        assert detector._detect_count < 30, (
            f"Loop did not terminate within 30 detector calls "
            f"(got {detector._detect_count})"
        )

        # 3) Forward extension actually fired.  Same-anchor escalation
        #    alone caps at ``1 + max_attempts`` = 4 detector calls.  Any
        #    higher count means ``attempt`` reset at least once via the
        #    forward-move branch — proving we exercised the dangerous
        #    path the bug previously left unbounded.
        assert detector._detect_count > 4, (
            f"Expected >4 detector calls (forward extension), "
            f"got {detector._detect_count}"
        )


# ---------------------------------------------------------------------------
# Wave-level (B>1) regeneration tests
# ---------------------------------------------------------------------------


class TestWaveLevelRegen:
    """The wave function (`_regenerate_leaking_wave`) batches multiple
    leaking samples through a single ``inner.regenerate()`` per attempt.

    These tests verify per-row independence and compaction at B>1; the
    existing single-row tests already cover the per-row state-machine
    logic at B=1, which the wave reduces to as a degenerate case.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        # Take 2 samples for batched scenarios.
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(
                range(2)
            )

    def _capture_regen_batch_sizes(self, inner):
        """Wrap ``inner.regenerate`` to record per-call ``input_ids.shape[0]``.

        Returns the captured-batch-sizes list; appended to in call order.
        """
        original = inner.regenerate
        captured: list[int] = []

        def tracking_regenerate(**kwargs):
            input_ids = kwargs.get("input_ids")
            assert input_ids is not None
            captured.append(input_ids.shape[0])
            return original(**kwargs)

        inner.regenerate = tracking_regenerate
        return captured

    def test_wave_calls_regenerate_with_batched_input_when_all_rows_leak(self):
        """B=2 batch with both rows leaking: the wave dispatches one
        ``regenerate()`` call covering both rows (B=2 input)."""
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Both initial detect_batch calls (0 and 1) leak; continuations
        # at calls 2, 3 are clean → both rows converge in one wave attempt.
        detector = MockLeakDetector(
            leak_on_calls={0, 1}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert len(captured) == 1, (
            f"Expected exactly 1 wave-level regenerate call, got "
            f"{len(captured)}: {captured}"
        )
        assert captured[0] == 2, (
            f"Expected wave call with B=2 (both leakers batched), got "
            f"B={captured[0]}"
        )

    def test_wave_excludes_non_leaking_rows_from_batch(self):
        """B=2 batch where only row 0 leaks: regenerate sees B=1 (the
        leaker), not B=2.  Row 1 must NOT enter the wave function."""
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
            clean_cot="Clean thoughts only.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Initial detect_batch: only call 0 leaks (row 0); call 1 clean.
        # Row 0's continuation at call 2 is clean → converges in one attempt.
        detector = MockLeakDetector(
            leak_on_calls={0}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert len(captured) == 1, (
            f"Expected exactly 1 wave call (only row 0 leaks), got "
            f"{len(captured)}: {captured}"
        )
        assert captured[0] == 1, (
            f"Wave should batch only the leaker (B=1), got B={captured[0]}"
        )

    def test_wave_skipped_when_no_rows_leak(self):
        """B=2 batch where neither row leaks: no wave call, no
        ``regenerate()`` invocation."""
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Generic clean cot.",  # initial CoT (won't be flagged)
            clean_cot="Generic clean cot.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Detector flags nothing.
        detector = MockLeakDetector(
            leak_on_calls=set(), first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert captured == [], (
            f"No leakers should mean no wave call; got {captured}"
        )

    def test_wave_compacts_after_partial_convergence(self):
        """B=2 batch where row 0 converges in 1 attempt and row 1 in 2:
        the second wave iteration only batches row 1 (B=1).  Verifies
        done-row compaction between iterations."""
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Detector calls in order:
        #   0: initial detect of row 0 → leak
        #   1: initial detect of row 1 → leak
        # Wave iter 1 (B=2, regenerate call #1):
        #   2: detect_continuation row 0 → clean (converges)
        #   3: detect_continuation row 1 → leak (stays active)
        # Wave iter 2 (B=1, regenerate call #2):
        #   4: detect_continuation row 1 → clean (converges)
        # Phase 3 (per-sample answer detect): calls 5, 6 → clean.
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 3}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert captured == [2, 1], (
            f"Expected wave to start at B=2 and compact to B=1 after row 0 "
            f"converged, got {captured}"
        )

    def _capture_regen_max_new_tokens(self, inner):
        """Wrap ``inner.regenerate`` to record the per-call
        ``generation_config.max_new_tokens``.  Returns the list of
        captured values, appended-to in call order."""
        original = inner.regenerate
        captured: list = []

        def tracking_regenerate(**kwargs):
            gen_config = kwargs.get("generation_config")
            captured.append(
                gen_config.max_new_tokens if gen_config is not None else None
            )
            return original(**kwargs)

        inner.regenerate = tracking_regenerate
        return captured

    def test_wave_per_row_forward_extension_shrinks_remaining_budget(self):
        """Forward extension on one row mid-wave grows that row's
        ``prefix_ids``; the *next* iteration's ``max_new_tokens`` must
        reflect the row's new (longer) prefix on a per-row basis.  If
        the wave accidentally used a stale or shared prefix length,
        iter 2's max_new_tokens would equal iter 1's instead of
        being smaller."""
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. Mid section. The secret is here.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_max_new_tokens(inner)

        model = ReasoningModel(inner)
        # Detector trace:
        #   0: row 0 init  → leak fli=1 (prefix = ["Safe start."])
        #   1: row 1 init  → leak fli=1
        # Iter 1 (B=2):
        #   2: row 0 contin → leak fli=2 (forward extend; row 0's
        #      prefix grows to two sentences and attempt resets to 0)
        #   3: row 1 contin → clean (row 1 done)
        # Iter 2 (B=1, only row 0):
        #   4: row 0 contin → clean (row 0 done)
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2},
            first_leak_index={2: 2},
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert len(captured) == 2, (
            f"Expected 2 wave iterations (forward extension + final), "
            f"got {len(captured)}"
        )
        assert captured[1] < captured[0], (
            f"Forward extension should have shrunk row 0's remaining "
            f"budget for iter 2; got iter 1 max_new_tokens="
            f"{captured[0]}, iter 2 max_new_tokens={captured[1]}"
        )

    def test_wave_over_budget_filter_resets_best_cot_on_both_rows(self):
        """When every row in the wave is over-budget at entry, the
        top-of-loop filter must (a) drop them all without any
        ``regenerate()`` call and (b) reset each row's ``best_cot`` to
        its ``clean_prefix`` so the leaky initial CoT does NOT carry
        through to the engine's final output.

        Mid-wave over-budget (a row's prefix grows past budget *during*
        the wave via forward extension) is harder to construct without
        a budget-respecting dummy that's also long enough to exercise
        the leak path; we accept the tradeoff and exercise the reset
        via the entry-time filter, which uses identical code.
        """
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        # Tiny budget — even the prefix from the initial detect
        # ("Safe start.", ~3 tokens) is already over budget=2.
        inner.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=2, use_cache=False,
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Both rows leak on initial detect (calls 0, 1).
        detector = MockLeakDetector(
            leak_on_calls={0, 1}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        # Both rows over-budget at entry → no wave call at all.
        assert captured == [], (
            f"Expected zero wave calls (both rows over-budget at "
            f"entry), got {captured}"
        )

        # Per-sample equivalence: each row's final CoT should be the
        # clean prefix (no leak).  AlwaysLeakingDummyModel's leaking_cot
        # contains "secret"; if the over-budget filter forgot to reset
        # best_cot, the engine output would still carry it.
        key = "rtofu_forget10"
        cots = engine.cot_generations[key]["generated"]
        for row_idx, cot in enumerate(cots):
            assert "secret" not in cot.lower(), (
                f"Row {row_idx} final CoT contains the leak phrase — "
                f"over-budget filter failed to reset best_cot to "
                f"clean_prefix:\n  {cot[:300]}"
            )

    def test_wave_compacts_when_row_one_converges_first(self):
        """Symmetric to ``test_wave_compacts_after_partial_convergence``:
        row 1 converges first while row 0 stays active.  Verifies the
        ``states[i]`` indexing under compaction is correct regardless
        of which row drops out."""
        from unittest.mock import MagicMock

        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
        )
        inner.soft_token = MagicMock()
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Detector trace (mirror image of the row-0-converges-first
        # test):
        #   0: row 0 init  → leak
        #   1: row 1 init  → leak
        # Iter 1 (B=2):
        #   2: row 0 contin → leak (stays active)
        #   3: row 1 contin → clean (done)
        # Iter 2 (B=1, only row 0):
        #   4: row 0 contin → clean (done)
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert captured == [2, 1], (
            f"Expected B=2 then compact to B=1 (row 0 staying), got "
            f"{captured}"
        )


# ---------------------------------------------------------------------------
# Answer wave tests (Part 4 — answer-phase batched detect-and-regenerate)
# ---------------------------------------------------------------------------


class _AnswerWaveTestModel(LeakingDummyModel):
    """LeakingDummyModel variant with multi-sentence canned answers,
    needed to exercise the answer wave's forward-extension branch
    (default ``"An answer."`` is one sentence so ``first_leak_index=2``
    can never validly hit).

    Per-call answer text is taken from ``canned_answers``; the cot text
    follows the parent's ``leaking_cot`` (call 1) / ``clean_cot``
    (call 2+) split.
    """

    def __init__(self, tokenizer, canned_answers, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.canned_answers = list(canned_answers)

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            cot = self.leaking_cot if self.call_count == 1 else self.clean_cot
            ans_idx = min(self.call_count - 1, len(self.canned_answers) - 1)
            answer = self.canned_answers[ans_idx]
            canned = f"{cot}{THINK_SUFFIX}{answer}"
            response_ids = self.tokenizer.encode(
                canned, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


class TestAnswerWave:
    """The answer wave (``_regenerate_leaking_answer_wave``) batches
    multiple rows whose answer leaks through a single
    ``inner.regenerate()`` per attempt.  Mirrors ``TestWaveLevelRegen``
    for the CoT phase but exercises the answer wave's diffs:
    ``frozen_prefix_ids = cot_ids + delim_ids``, anchor placement at
    the actual answer leak position, no carryover of CoT-phase soft
    tokens.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(
                range(2)
            )

    def _capture_regen_batch_sizes(self, inner):
        """Wrap ``inner.regenerate`` to record per-call ``input_ids.shape[0]``.
        Identical helper to TestWaveLevelRegen's; duplicated here to keep
        each test class self-contained.
        """
        original = inner.regenerate
        captured: list[int] = []

        def tracking(soft_token_positions=None, **kwargs):
            input_ids = kwargs.get("input_ids")
            assert input_ids is not None
            captured.append(input_ids.shape[0])
            return original(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking
        return captured

    def test_answer_wave_batches_at_b2_when_both_answers_leak(self):
        """B=2 batch with both answers leaking: the answer wave dispatches
        one ``regenerate()`` covering both rows (B=2 input)."""
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Just thinking.",
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Per-method counters:
        #   detect_batch[0,1]    = initial CoT (B=2) → both clean
        #   detect_batch[2,3]    = post-CoT-wave answer detect → both leak fli=0
        #   detect_continuation[0,1] = answer-wave per-row → both clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {2, 3}},
            first_leak_index=0,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        # No CoT wave fired (no CoT leakers); the only inner.regenerate
        # call is the single B=2 answer-wave call.
        assert len(captured) == 1, (
            f"Expected exactly 1 answer-wave regenerate call; "
            f"got {len(captured)}: {captured}"
        )
        assert captured[0] == 2, (
            f"Expected wave call with B=2 (both answer-leakers batched); "
            f"got B={captured[0]}"
        )

    def test_answer_wave_excludes_clean_answer_rows_from_batch(self):
        """B=2 batch where only row 0's answer leaks: regenerate() sees
        B=1 (only the leaker), not B=2."""
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Just thinking.",
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Per-method counters:
        #   detect_batch[0,1] = initial CoT (B=2) → both clean
        #   detect_batch[2]   = answer row 0 → leak at fli=0
        #   detect_batch[3]   = answer row 1 → clean
        #   detect_continuation[0] = answer-wave row 0 → clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {2}},
            first_leak_index=0,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert len(captured) == 1, (
            f"Expected exactly 1 answer-wave regenerate call (only row 0 "
            f"leaks); got {len(captured)}: {captured}"
        )
        assert captured[0] == 1, (
            f"Wave should batch only the leaker (B=1); got B={captured[0]}"
        )

    def test_answer_wave_skipped_when_no_answers_leak(self):
        """B=2 batch where neither answer leaks: no answer-wave call,
        no inner.regenerate() invocation at all."""
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Just thinking.",
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # All detect calls return clean.
        detector = MockLeakDetector(leak_on_calls=set())

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        assert captured == [], (
            f"No answer leakers should mean no wave call; got {captured}"
        )

    def test_answer_wave_forward_extension_at_b1(self):
        """B=1 forward extension in the answer wave: leak at sentence 0
        of the initial answer, regen produces text whose continuation
        flags at a later sentence (forward move), second regen succeeds.

        Pins the new answer-wave behavior — after Part 4, the answer
        leak anchor moves forward through the answer prefix as new
        sentences are cleared, instead of being locked at end-of-CoT
        the way the legacy ``_run_answer_regen`` did.
        """
        inner = _AnswerWaveTestModel(
            self.tokenizer,
            canned_answers=[
                # Initial answer: 1 sentence — flagged at fli=0.
                "Leaking sentence one.",
                # First regen continuation: 3 sentences — detector
                # combines with empty prefix, flags at fli=2 (forward
                # extension to clear the first 2 sentences).
                "Sentence one. Sentence two. Sentence three.",
                # Second regen continuation: arbitrary; detector returns
                # clean.
                "Final clean continuation.",
            ],
            leaking_cot="Just thinking.",
            clean_cot="Still thinking.",
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Per-method counters:
        #   detect_batch[0]    = initial CoT (single row) → clean
        #   detect_batch[1]    = post-CoT-wave answer detect → leak fli=0
        #   detect_continuation[0] = answer iter 0 → leak fli=2
        #                            (forward extension: 2 sentences cleared)
        #   detect_continuation[1] = answer iter 1 → clean
        detector = MockLeakDetector(
            leak_on_calls={
                "detect_batch": {1},
                "detect_continuation": {0},
            },
            first_leak_index={
                "detect_batch": {1: 0},
                "detect_continuation": {0: 2},
            },
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # Two regen calls: forward-extension iteration 0 + iteration 1.
        assert len(captured) == 2, (
            f"Expected 2 answer-wave regenerate calls (initial + forward "
            f"extension); got {len(captured)}: {captured}"
        )
        assert captured == [1, 1], (
            f"Both wave iterations should be B=1; got {captured}"
        )

    def test_answer_wave_unifies_phase2_and_phase3(self):
        """B=2 with both rows' answers leaking but only one row's CoT
        leaking: the unified dispatch routes both rows through the
        single answer-wave call (B=2), regardless of CoT-phase origin.
        Mirrors task #7's Phase 2 / Phase 3 unification.
        """
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="The secret fact is revealed.",
        )
        captured = self._capture_regen_batch_sizes(inner)

        model = ReasoningModel(inner)
        # Per-method counters (B=2):
        #   detect_batch[0,1] = initial CoT (B=2) → row 0 leak, row 1 clean
        #   detect_continuation[0] = CoT-wave row 0 → clean (CoT done)
        #   detect_batch[2,3] = post-CoT-wave answer detect → both leak
        #   detect_continuation[1,2] = answer-wave per-row → both clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {0, 2, 3}},
            first_leak_index=0,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        # Two regenerate calls: one CoT-wave (B=1) + one answer-wave (B=2).
        # The answer wave covers BOTH the post-CoT-wave row (Phase 2
        # origin) AND the originally-clean-CoT row (Phase 3 origin) in
        # a single batched dispatch.
        assert len(captured) == 2, (
            f"Expected 2 regenerate calls (CoT wave + answer wave); "
            f"got {len(captured)}: {captured}"
        )
        assert captured == [1, 2], (
            f"Expected CoT wave at B=1 then answer wave at B=2; "
            f"got {captured}"
        )

    def test_answer_wave_per_row_carryover_phase2_vs_phase3(self):
        """B=2 mixed batch where row 0 is Phase 2 origin (CoT leaked,
        went through CoT wave, has CoT-phase soft tokens to carry) and
        row 1 is Phase 3 origin (clean CoT, no CoT wave, no carryover).
        Both answers leak.

        In the single answer-wave regen call:
          * Row 0's per-row positions = 1 carried CoT pad + 1 fresh
            answer pad = 2 entries.
          * Row 1's per-row positions = 0 carried + 1 fresh answer pad
            = 1 entry.

        Pins the per-row heterogeneity that ``frozen_soft_token_stack``
        enables in the batched dispatch.
        """
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="The secret fact is revealed.",
        )
        model = ReasoningModel(inner)

        original_regenerate = inner.regenerate
        captured_schedules: list[list[list[int]] | None] = []

        def tracking(soft_token_positions=None, **kwargs):
            # Snapshot the full per-row positions schedule for later
            # length checks (not just B as in TestWaveLevelRegen's helper).
            if soft_token_positions is None:
                captured_schedules.append(None)
            else:
                captured_schedules.append(
                    [list(row) for row in soft_token_positions]
                )
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking

        # Per-method counters (B=2, mixed Phase 2 + Phase 3 origin):
        #   detect_batch[0,1] = initial CoT (B=2) → row 0 leak, row 1 clean
        #   detect_continuation[0] = CoT-wave row 0 → clean
        #   detect_batch[2,3] = post-CoT-wave answer detect → both leak
        #   detect_continuation[1,2] = answer-wave per-row → both clean
        detector = MockLeakDetector(
            leak_on_calls={"detect_batch": {0, 2, 3}},
            first_leak_index=0,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
        )
        engine.inference()

        # CoT-wave regen call (B=1) + answer-wave regen call (B=2).
        assert len(captured_schedules) == 2

        # CoT wave: 1 row, 1 anchor.
        cot_call = captured_schedules[0]
        assert cot_call is not None
        assert len(cot_call) == 1 and len(cot_call[0]) == 1

        # Answer wave: 2 rows; row 0 has 2 anchors (carryover + fresh),
        # row 1 has 1 anchor (fresh only).
        ans_call = captured_schedules[1]
        assert ans_call is not None
        assert len(ans_call) == 2, f"Expected B=2 schedule, got {ans_call}"
        # Row order in the answer wave matches the order of leaking
        # answer indices, which here is [0, 1] since both leak.
        assert len(ans_call[0]) == 2, (
            f"Row 0 (Phase 2 origin) should have 1 carried CoT pad + "
            f"1 fresh answer pad = 2 entries; got {ans_call[0]}"
        )
        assert len(ans_call[1]) == 1, (
            f"Row 1 (Phase 3 origin) should have 0 carryover + 1 fresh "
            f"answer pad = 1 entry; got {ans_call[1]}"
        )

    def test_carryover_plus_answer_forward_extension_compose(self):
        """End-to-end carryover + answer-phase forward extension at B=1.

        Setup: leaking CoT (CoT wave converges in 1 attempt → 1 CoT
        pad on the stack) → answer leak at sentence 0 → answer-wave
        iter 0 regen produces text that flags at sentence 2 of the
        new continuation (forward extension to clear 2 sentences →
        push a new prefix anchor) → answer-wave iter 1 regen converges.

        Pins the interaction between the immutable
        ``frozen_soft_token_stack`` (carries CoT pads, never grows
        during the answer wave) and the active ``soft_token_stack``
        (grows under answer-phase forward extension).  The frozen
        segment stays at 1 entry across both answer iterations; the
        prefix segment grows from 1 to 2 entries when forward
        extension fires.
        """
        inner = _AnswerWaveTestModel(
            self.tokenizer,
            canned_answers=[
                # Initial answer: 1 sentence — flagged at fli=0.
                "Initial leaking answer.",
                # CoT-wave regen: arbitrary; the CoT-wave detect call
                # for this returns clean (not in leak_on_calls).
                "Stand-in answer.",
                # Answer-wave iter 0 regen: 3 sentences past the
                # cot+delim prefix in the assembled new_continuation,
                # so detect_continuation can flag at fli=2 (forward
                # extension clearing 2 sentences).
                "Sentence one. Sentence two. Sentence three.",
                # Answer-wave iter 1 regen: anything; detector returns
                # clean.
                "Final clean continuation.",
            ],
            leaking_cot="Just thinking. The secret leaks.",
            clean_cot="Still thinking.",
        )
        model = ReasoningModel(inner)

        original_regenerate = inner.regenerate
        captured: list[list[int] | None] = []

        def tracking(soft_token_positions=None, **kwargs):
            # B=1 throughout — unwrap row 0's positions.
            if soft_token_positions is None:
                captured.append(None)
            else:
                captured.append(list(soft_token_positions[0]))
            return original_regenerate(
                soft_token_positions=soft_token_positions,
                **kwargs,
            )

        inner.regenerate = tracking

        # Per-method counters (B=1, both CoT and answer leak):
        #   detect_batch[0]    = initial CoT → leak fli=1
        #   detect_continuation[0] = CoT wave → clean (1 attempt)
        #   detect_batch[1]    = post-CoT-wave answer detect → leak fli=0
        #   detect_continuation[1] = answer iter 0 → leak fli=2
        #                            (forward extension clears 2 sentences)
        #   detect_continuation[2] = answer iter 1 → clean
        detector = MockLeakDetector(
            leak_on_calls={
                "detect_batch": {0, 1},
                "detect_continuation": {1},
            },
            first_leak_index={
                "detect_batch": {0: 1, 1: 0},
                "detect_continuation": {1: 2},
            },
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # 1 CoT-wave call + 2 answer-wave calls (initial + forward
        # extension).
        assert len(captured) == 3, (
            f"Expected 3 regen calls (1 CoT + 2 answer); got "
            f"{len(captured)}: {captured}"
        )
        cot_call, ans_iter0, ans_iter1 = captured
        assert cot_call is not None and len(cot_call) == 1, (
            f"CoT wave should push 1 anchor; got {cot_call}"
        )
        # Answer iter 0: 1 frozen carryover + 1 fresh prefix anchor.
        assert ans_iter0 is not None and len(ans_iter0) == 2, (
            f"Answer wave iter 0 should have 1 carried CoT pad + "
            f"1 fresh answer pad = 2 anchors; got {ans_iter0}"
        )
        # Answer iter 1: frozen segment unchanged (1 entry) + prefix
        # segment grew via forward extension (2 entries: initial + the
        # new anchor pushed at the extended prefix's end).
        assert ans_iter1 is not None and len(ans_iter1) == 3, (
            f"Answer wave iter 1 should have 1 carried CoT pad + "
            f"2 prefix pads (initial + forward-extension push) = 3 "
            f"anchors; got {ans_iter1}"
        )
        # Frozen-segment pad stays at the same absolute position
        # across both answer iterations (it indexes into
        # frozen_prefix_ids, which is set once at init).
        assert ans_iter0[0] == ans_iter1[0], (
            f"Carried CoT pad position should be stable across answer "
            f"iterations; got iter0={ans_iter0[0]}, iter1={ans_iter1[0]}"
        )


class TestEngineChunkSizePassthrough:
    """End-to-end: setting ``chunk_size`` on the engine plumbs
    through to ``inner.regenerate(...)`` without disturbing wave-loop
    correctness.

    Note: ``LeakingDummyModel.regenerate`` is a thin pass-through to its
    ``generate()`` and ignores ``chunk_size``/``per_row_remaining``, so
    its outputs are identical regardless of these kwargs.  This test
    confirms that:

      * ``inner.regenerate`` is called with the new kwargs (proving the
        engine threads them through ``_run_wave_loop`` correctly), AND
      * Per-row outputs are unchanged from the ``chunk_size=None``
        baseline (proving the engine's wave-loop semantics are
        preserved when chunk_size is set).

    Real-AR chunked correctness is exercised at the RegeneratingModel level
    in ``tests/test_regen_corruption.py:TestChunkedByteEquivalence``.
    """

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(
                range(2)
            )

    def _build_engine_and_capture(self, chunk_size):
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed.",
            clean_cot="Clean thoughts only.",
        )
        inner.soft_token = MagicMock()

        # Wrap inner.regenerate to capture per-call kwargs so we can
        # assert chunk_size / per_row_remaining are forwarded.
        original_regenerate = inner.regenerate
        captured_kwargs: list[dict] = []

        def tracking_regenerate(**kwargs):
            captured_kwargs.append(
                {
                    "chunk_size": kwargs.get("chunk_size"),
                    "per_row_remaining": kwargs.get("per_row_remaining"),
                }
            )
            return original_regenerate(**kwargs)

        inner.regenerate = tracking_regenerate

        model = ReasoningModel(inner)
        # Initial detect_batch: only call 0 leaks; call 1 clean.
        # Row 0's continuation at call 2 clears.
        detector = MockLeakDetector(
            leak_on_calls={0}, first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_max_attempts=3,
            batch_size=2,
            chunk_size=chunk_size,
        )
        engine.inference()
        return engine, captured_kwargs

    def test_chunked_engine_matches_unchunked_outputs(self):
        baseline_engine, baseline_kwargs = (
            self._build_engine_and_capture(chunk_size=None)
        )
        chunked_engine, chunked_kwargs = (
            self._build_engine_and_capture(chunk_size=256)
        )

        # Baseline forwards chunk_size=None; chunked forwards chunk_size=256.
        assert all(c["chunk_size"] is None for c in baseline_kwargs)
        assert all(c["chunk_size"] == 256 for c in chunked_kwargs)
        # Baseline forwards per_row_remaining=None; chunked forwards a list.
        assert all(c["per_row_remaining"] is None for c in baseline_kwargs)
        assert all(
            isinstance(c["per_row_remaining"], list)
            and all(isinstance(x, int) for x in c["per_row_remaining"])
            for c in chunked_kwargs
        )

        # Per-row engine outputs unchanged.  cot_generations and
        # answer_generations are dicts keyed
        # ``{dataset_name}_{subset_name}``; the wave loop's per-row
        # best_cot / best_answer feed these.
        assert (
            baseline_engine.cot_generations.keys()
            == chunked_engine.cot_generations.keys()
        )
        for key in baseline_engine.cot_generations:
            assert (
                baseline_engine.cot_generations[key]["generated"]
                == chunked_engine.cot_generations[key]["generated"]
            )
        assert (
            baseline_engine.answer_generations.keys()
            == chunked_engine.answer_generations.keys()
        )
        for key in baseline_engine.answer_generations:
            assert (
                baseline_engine.answer_generations[key]["generated"]
                == chunked_engine.answer_generations[key]["generated"]
            )

    def test_engine_rejects_chunk_size_zero(self):
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(self.tokenizer)
        inner.soft_token = MagicMock()
        model = ReasoningModel(inner)
        detector = MockLeakDetector(leak_on_calls=set())

        with pytest.raises(ValueError, match="chunk_size"):
            RegeneratingReasoningEngine(
                model=model,
                tokenizer=self.tokenizer,
                data_module=self.rtofu,
                subset_names=["forget10"],
                answer_evaluator=[ROUGERecall(mode="rougeL")],
                leak_detector=detector,
                batch_size=2,
                chunk_size=0,
            )
