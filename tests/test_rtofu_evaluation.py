"""
Tests for the R-TOFU evaluation pipeline using a dummy text generator
instead of the real LRM-target model.

Usage:
    conda run -n redact python -m pytest tests/test_rtofu_evaluation.py -v
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from redact.dataset.rtofu import RTOFU
from redact.evaluator import (
    CosineSimilarity,
    EntailmentScore,
    ROUGERecall,
    TokenEntropy,
)
from redact.inference import ReasoningGenerationEngine
from redact.model.reasoning import ReasoningModel


# ---------------------------------------------------------------------------
# Dummy model that mimics the HFModel interface and generates canned responses
# ---------------------------------------------------------------------------

THINK_SUFFIX = "</think>\n\n"


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

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            # Decode the prompt to echo it back
            prompt_ids = input_ids[i]
            canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        # Pad to same length
        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


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


# Cached evaluator instances (stateless, safe to share)
_COSINE_SIM = None
_ENTAILMENT = None


def _get_cosine_similarity():
    global _COSINE_SIM
    if _COSINE_SIM is None:
        _COSINE_SIM = CosineSimilarity()
    return _COSINE_SIM


def _get_entailment_score():
    global _ENTAILMENT
    if _ENTAILMENT is None:
        _ENTAILMENT = EntailmentScore(reverse=False)
    return _ENTAILMENT


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRTOFUDataset:
    def test_download_and_splits(self):
        _, rtofu = _get_shared_fixtures()
        assert "forget10" in rtofu.dataset
        assert "retain90" in rtofu.dataset
        assert len(rtofu.dataset["forget10"]) > 0

    def test_load_dataset_for_eval_has_cot(self):
        _, rtofu = _get_shared_fixtures()
        dataset = rtofu.load_dataset_for_eval("forget10", load_in_batch=True, batch_size=4)
        batch = dataset[0]
        assert "cot" in batch, "cot column must survive batchify"
        assert len(batch["cot"]) == len(batch["answer"])

    def test_load_dataset_for_classification(self):
        _, rtofu = _get_shared_fixtures()
        ds = rtofu.load_dataset_for_classification("forget10")
        assert "train" in ds
        assert "forget" in ds
        assert "retain" in ds
        assert set(ds["train"]["label"]) == {0, 1}

    def test_match_retain(self):
        # match_retain now maps to synthetic protocol-retain split names whose
        # content is constructed per-instance from the task_id (see rtofu.py).
        assert RTOFU.match_retain["forget01"] == "forget01_retain"
        assert RTOFU.match_retain["forget05"] == "forget05_retain"
        assert RTOFU.match_retain["forget10"] == "forget10_retain"

    def test_protocol_retain_construction(self):
        _, rtofu = _get_shared_fixtures()
        # task_id=1 is the default fixture. Check sizes and disjointness.
        sizes = {
            "forget01_retain": 3960,  # retain90 (3600) + 9 × 40
            "forget05_retain": 3800,  # retain50 (2000) + 9 × 200
            "forget10_retain": 3600,  # retain10 (400)  + 8 × 400
        }
        for forget_split, expected_size in sizes.items():
            retain_ds = rtofu.dataset[forget_split]
            assert len(retain_ds) == expected_size, (
                f"{forget_split}: {len(retain_ds)} != {expected_size}"
            )
            forget_name = forget_split.replace("_retain", "")
            f_qs = set(rtofu.dataset[forget_name]["question"])
            r_qs = set(retain_ds["question"])
            assert not (f_qs & r_qs), (
                f"{forget_split} overlaps with {forget_name}"
            )

    def test_task_id_parametrization(self):
        rtofu1 = RTOFU(
            formatting_tokens={
                "prompt_prefix": "", "prompt_suffix": "",
                "answer_prefix": "", "answer_suffix": "",
            },
            task_id=1,
        )
        rtofu1.download()
        rtofu2 = RTOFU(
            formatting_tokens={
                "prompt_prefix": "", "prompt_suffix": "",
                "answer_prefix": "", "answer_suffix": "",
            },
            task_id=2,
        )
        rtofu2.download()
        # Different task_ids yield different forget prompts and different retain.
        assert set(rtofu1.dataset["forget10"]["question"]) != set(
            rtofu2.dataset["forget10"]["question"]
        )
        assert set(rtofu1.dataset["forget10_retain"]["question"]) != set(
            rtofu2.dataset["forget10_retain"]["question"]
        )
        # task_id=2 forget10 must not overlap with task_id=2 forget10_retain.
        assert not (
            set(rtofu2.dataset["forget10"]["question"])
            & set(rtofu2.dataset["forget10_retain"]["question"])
        )

    def test_invalid_task_id_raises(self):
        import pytest
        rtofu = RTOFU(
            formatting_tokens={
                "prompt_prefix": "", "prompt_suffix": "",
                "answer_prefix": "", "answer_suffix": "",
            },
            task_id=99,
        )
        with pytest.raises(ValueError, match="task_id"):
            rtofu.download()

    def test_non_int_task_id_raises(self):
        import pytest
        ft = {
            "prompt_prefix": "", "prompt_suffix": "",
            "answer_prefix": "", "answer_suffix": "",
        }
        # Floats would silently coerce via str(): 1.0 -> '1.0', unmatchable.
        with pytest.raises(TypeError, match="task_id"):
            RTOFU(formatting_tokens=ft, task_id=1.0)
        with pytest.raises(TypeError, match="task_id"):
            RTOFU(formatting_tokens=ft, task_id="1")
        # Bool is an int subclass in Python; reject it explicitly.
        with pytest.raises(TypeError, match="task_id"):
            RTOFU(formatting_tokens=ft, task_id=True)

    def test_task_id_10_partial_validity(self):
        # task_id=10 exists for forget01 / forget05 but not forget10
        # (forget10 only has task_ids 1..9). Expected: forget01/05 and their
        # matched retains are built; forget10 and forget10_retain are omitted.
        rtofu = RTOFU(
            formatting_tokens={
                "prompt_prefix": "", "prompt_suffix": "",
                "answer_prefix": "", "answer_suffix": "",
            },
            task_id=10,
        )
        rtofu.download()
        assert "forget01" in rtofu.dataset
        assert "forget01_retain" in rtofu.dataset
        assert "forget05" in rtofu.dataset
        assert "forget05_retain" in rtofu.dataset
        assert "forget10" not in rtofu.dataset
        assert "forget10_retain" not in rtofu.dataset
        # Sizes for valid splits
        assert len(rtofu.dataset["forget01"]) == 40
        assert len(rtofu.dataset["forget05"]) == 200

    def test_forget10_retain_matches_hf_retain90_ordering(self):
        # Pins down the load-bearing invariant from CLAUDE.md: protocol retain
        # for forget10 task_id=1 equals HF retain90 set-wise AND first-400
        # ordered. This ensures existing forget10 artifacts built against
        # HF retain90 are preserved byte-for-byte under the new loader.
        from datasets import load_dataset
        _, rtofu = _get_shared_fixtures()
        hf_r90 = load_dataset("sangyon/R-TOFU", split="retain90", keep_in_memory=True)
        new_retain = rtofu.dataset["forget10_retain"]
        assert set(new_retain["question"]) == set(hf_r90["question"])
        assert list(new_retain["question"])[:400] == list(hf_r90["question"])[:400]

    def test_protocol_retain_union_reconstructs_forget_all(self):
        # For each forget split: forget[task_id=K] ∪ matched_retain[task_id=K]
        # ∪ retain_base should reconstruct the "all task_ids" forget JSONL
        # minus the retain_base (plus the forget rows themselves). In simpler
        # terms: the matched retain excludes exactly the current task_id's
        # rows, nothing more, nothing less.
        import json as _json
        _, rtofu = _get_shared_fixtures()
        data_dir = rtofu.data_dir
        for forget_name in ["forget01", "forget05", "forget10"]:
            with open(f"{data_dir}/{forget_name}.json", encoding="utf-8") as fp:
                all_forget = [_json.loads(l) for l in fp if l.strip()]
            all_forget_qs = {r["question"] for r in all_forget}
            unused_expected = {r["question"] for r in all_forget if r["task_id"] != "1"}
            retain_qs = set(rtofu.dataset[f"{forget_name}_retain"]["question"])
            forget_qs = set(rtofu.dataset[forget_name]["question"])
            # Every unused task_id's row is in the retain
            assert unused_expected <= retain_qs
            # The union of forget + unused_expected == all_forget
            assert (forget_qs | unused_expected) == all_forget_qs


class TestReasoningGenerationEngine:
    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        self.model = ReasoningModel(DummyModel(self.tokenizer))
        # Limit dataset to a few examples for speed
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_forget_only(self):
        """Engine runs on forget split only (no retain)."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        assert len(summary) > 0
        assert any("forget10" in k for r in summary for k in r)

    def test_forget_and_retain(self):
        """Engine runs on both forget and retain splits."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10", "retain90"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        assert any("forget10" in k for k in keys)
        assert any("retain90" in k for k in keys)

    def test_cot_generations_have_gold_cot(self):
        """cot_generations['gold'] should contain actual CoT text, not answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.cot_generations
        gold_cots = engine.cot_generations[key]["gold"]
        gold_answers = engine.answer_generations[key]["gold"]

        # Gold CoTs should differ from gold answers (they come from different columns)
        assert gold_cots != gold_answers

    def test_answer_generations_populated(self):
        """answer_generations should contain gold answers and generated answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        assert len(engine.answer_generations[key]["gold"]) == 4
        assert len(engine.answer_generations[key]["generated"]) == 4
        # All generated answers should be "Dummy answer." from DummyModel
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans

    def test_cot_evaluator(self):
        """CFE evaluators run inside the engine and produce cot_ prefixed results."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            cot_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        answer_keys = [k for k in keys if "_cot_" not in k]
        cot_keys = [k for k in keys if "_cot_" in k]
        assert len(answer_keys) > 0, "Should have answer evaluator results"
        assert len(cot_keys) > 0, "Should have CoT evaluator results"

    def test_multiple_evaluators(self):
        """Multiple answer and CoT evaluators all produce results."""
        answer_evals = [
            ROUGERecall(mode="rougeL"),
            TokenEntropy(tokenizer=self.tokenizer),
        ]
        cot_evals = [
            ROUGERecall(mode="rougeL"),
        ]
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=answer_evals,
            cot_evaluator=cot_evals,
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        assert any("rougeL_recall" in k and "_cot_" not in k for k in keys)
        assert any("token_entropy" in k for k in keys)
        assert any("_cot_rougeL_recall" in k for k in keys)


class TestEvaluators:
    """Smoke-test each evaluator used in R-TOFU evaluation."""

    def test_rouge_recall(self):
        evaluator = ROUGERecall(mode="rougeL")
        scores = evaluator.evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_cosine_similarity(self):
        scores = _get_cosine_similarity().evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_entailment_score(self):
        scores = _get_entailment_score().evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert scores[0] in (0, 1)

    def test_token_entropy(self):
        tokenizer = _make_tokenizer()
        evaluator = TokenEntropy(tokenizer=tokenizer)
        scores = evaluator.evaluate(
            ["ignored"],
            ["The quick brown fox jumps over the lazy dog"],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_token_entropy_empty(self):
        tokenizer = _make_tokenizer()
        evaluator = TokenEntropy(tokenizer=tokenizer)
        scores = evaluator.evaluate(["ignored"], [""])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# DummyModel variant that omits </think> (simulates degenerate corruption)
# ---------------------------------------------------------------------------

class DummyModelNoThink(DummyModel):
    """Like DummyModel but never produces </think>, simulating high corruption."""

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
            # No </think> delimiter — answer will be ""
            canned_response = "I ramble about nothing coherent at all."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


class DummyModelMixed(DummyModel):
    """Produces </think> for even-indexed samples, omits it for odd-indexed."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._call_count = 0

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
            idx = self._call_count
            self._call_count += 1
            if idx % 2 == 0:
                canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            else:
                canned_response = "I ramble about nothing coherent at all."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


class TestEmptyAnswerHandling:
    """Tests for per-sample AFE=0 override and think_completion_rate."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_all_empty_answers_scores_overridden(self):
        """When all answers are empty, answer evaluator scores should be 1.0."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        _, outputs = engine.summary()

        for result_dict in outputs:
            key = list(result_dict.keys())[0]
            if "rougeL_recall" in key and "_cot_" not in key:
                scores = result_dict[key]
                assert all(s == 1.0 for s in scores), (
                    f"Empty-answer samples should have score overridden to 1.0, got {scores}"
                )

    def test_all_empty_think_completion_rate_zero(self):
        """think_completion_rate should be 0.0 when no responses have </think>."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        assert "rtofu_forget10_think_completion_rate" in all_results
        assert all_results["rtofu_forget10_think_completion_rate"] == 0.0

    def test_mixed_answers_reduced_afe(self):
        """AFE should be reduced when some answers are empty."""
        model = ReasoningModel(DummyModelMixed(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=1,  # batch_size=1 so mixed model alternates per-sample
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        rate = all_results["rtofu_forget10_think_completion_rate"]
        assert 0.0 < rate < 1.0, f"Expected partial completion, got {rate}"

    def test_normal_model_think_completion_rate_one(self):
        """DummyModel (always produces </think>) should have rate=1.0."""
        model = ReasoningModel(DummyModel(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        assert all_results["rtofu_forget10_think_completion_rate"] == 1.0

    def test_engine_apply_empty_override_false_skips_override(self):
        """With apply_empty_override=False, per-sample scores for empty answers
        are NOT overridden to 1.0 — the raw evaluator scores pass through."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
            apply_empty_override=False,
        )
        engine.inference()
        _, outputs = engine.summary()

        found = False
        for result_dict in outputs:
            key = list(result_dict.keys())[0]
            if "rougeL_recall" in key and "_cot_" not in key:
                found = True
                scores = result_dict[key]
                # With empty responses + no override, raw rouge scores should
                # be ~0 (empty-vs-gold has no overlap), not 1.0.
                assert not all(s == 1.0 for s in scores), (
                    f"Expected raw scores, got all 1.0 overrides: {scores}"
                )
                assert all(s < 0.5 for s in scores), (
                    f"Empty-vs-gold rouge should be small, got {scores}"
                )
        assert found, "rougeL_recall not present in engine outputs"

    def test_engine_apply_empty_override_set_scopes_to_forget(self):
        """Passing a set of subset names to ``apply_empty_override`` must
        apply the override only to those subsets. Critical for MU eval,
        where empty answers on retain/aux splits are real utility failures
        — not AFE gaming attempts — and should not be rewarded to 1.0."""
        # Build a two-subset setup where both subsets see the same empty-
        # answer scenario. Override is scoped to forget10 only; retain90
        # should see raw (near-zero) scores.
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10", "retain90"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
            apply_empty_override={"forget10"},
        )
        engine.inference()
        _, outputs = engine.summary()

        forget_scores, retain_scores = None, None
        for result_dict in outputs:
            key = list(result_dict.keys())[0]
            if "_cot_" in key or "rougeL_recall" not in key:
                continue
            if "forget10" in key:
                forget_scores = result_dict[key]
            elif "retain90" in key:
                retain_scores = result_dict[key]
        assert forget_scores is not None and retain_scores is not None
        # forget10: override ON → empties coerced to 1.0
        assert all(s == 1.0 for s in forget_scores), (
            f"forget10 should have all 1.0 overrides, got {forget_scores}"
        )
        # retain90: override OFF → raw empty-vs-gold scores (~0)
        assert all(s < 0.5 for s in retain_scores), (
            f"retain90 should keep raw scores, got {retain_scores}"
        )
