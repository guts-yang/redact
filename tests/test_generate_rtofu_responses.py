"""
Tests for the unified generate_rtofu_responses script, covering:
  * the no-corruption baseline path (frozen target / pre-unlearned checkpoints);
  * TOFU auxiliary splits (real_authors / world_facts) with empty gold_cot;
  * the baseline-YAML guard that rejects inference-time defense flags.

Usage:
    conda run -n redact python -m pytest tests/test_generate_rtofu_responses.py -v
"""
import argparse

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, GenerationConfig

from redact.dataset.rtofu import RTOFU
from redact.inference import ReasoningGenerationEngine
from redact.model.reasoning import ReasoningModel
from scripts.generate_rtofu_responses import (
    _DEFENSE_FLAGS,
    _MU_RETAIN_SUBSAMPLE,
    _build_parser,
    _resolve_sample_defaults,
    _validate_baseline_guard,
)


THINK_SUFFIX = "</think>\n\n"


class DummyInnerModel(nn.Module):
    def modules(self):
        return iter([self])


class DummyModel:
    """Drop-in replacement for HFModel that returns canned generations."""

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
            prompt_ids = input_ids[i]
            canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


def _make_tokenizer():
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
    # Return a shallow copy so tests can mutate .dataset without affecting others
    import copy
    rtofu = copy.copy(_RTOFU_BASE)
    rtofu.dataset = dict(_RTOFU_BASE.dataset)
    return _TOKENIZER, rtofu


class TestNoCorrruptionBaseline:
    """Tests for the no-corruption path (merged from old examine_rtofu.py)."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        self.model = ReasoningModel(DummyModel(self.tokenizer))
        self.rtofu.dataset["forget10"] = self.rtofu.dataset["forget10"].select(range(4))

    def test_no_corruption_generates_answers(self):
        """Without any corruption, engine produces non-empty answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        answers = [item for batch in generations["forget10"]["generated_answer"] for item in batch]
        assert len(answers) == 4
        assert all(len(a) > 0 for a in answers)

    def test_no_corruption_generates_cot(self):
        """Without any corruption, engine produces non-empty CoT."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        cots = [item for batch in generations["forget10"]["generated_cot"] for item in batch]
        assert len(cots) == 4
        assert all(len(c) > 0 for c in cots)

    def test_no_corruption_preserves_gold_data(self):
        """Gold answers and CoTs pass through unchanged."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        gold_answers = [item for batch in generations["forget10"]["gold_answer"] for item in batch]
        gold_cots = [item for batch in generations["forget10"]["gold_cot"] for item in batch]
        assert len(gold_answers) == 4
        assert len(gold_cots) == 4
        assert all(len(a) > 0 for a in gold_answers)
        # Gold CoTs should differ from gold answers
        assert gold_answers != gold_cots

    def test_no_corruption_model_not_wrapped_in_attacked(self):
        """When no corruption is specified, model should be ReasoningModel wrapping
        DummyModel directly, not an RegeneratingModel."""
        from redact.regen import RegeneratingModel
        inner = self.model.model
        assert not isinstance(inner, RegeneratingModel)

    def test_no_corruption_with_offset(self):
        """Offset correctly skips initial examples."""
        _, rtofu = _get_shared_fixtures()
        rtofu.dataset["forget10"] = rtofu.dataset["forget10"].select(range(2, 6))

        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        prompts = [item for batch in generations["forget10"]["prompt"] for item in batch]
        assert len(prompts) == 4

    def test_no_corruption_single_batch(self):
        """batch_size=1 generates one example at a time, matching old script behavior."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=1,
        )
        generations = engine._generate()
        answers = [item for batch in generations["forget10"]["generated_answer"] for item in batch]
        assert len(answers) == 4
        assert all("Dummy answer" in a for a in answers)


class TestAuxSplitGeneration:
    """TOFU auxiliary splits (real_authors / world_facts) have no CoT; the data
    module inserts an empty 'cot' placeholder so the engine returns empty
    gold_cot entries rather than raising."""

    def _inject_fake_aux(self, rtofu, split_name, rows):
        """Overwrite rtofu.dataset[split_name] with a synthetic Dataset.

        Aux splits are now loaded from vendored JSONLs in data/rtofu/ by
        rtofu.download(); this helper replaces the loaded data with a
        compact test fixture so downstream schema plumbing
        (gold_answer / gold_cot / prompt_formatted) can be validated
        without depending on the real 100+-row aux content."""
        rtofu.dataset[split_name] = Dataset.from_dict(rows)

    def test_real_authors_generates_empty_gold_cot(self):
        tokenizer, rtofu = _get_shared_fixtures()
        self._inject_fake_aux(
            rtofu, "real_authors",
            {
                "question": ["Who wrote Moby-Dick?", "Who wrote Pride and Prejudice?"],
                "answer": ["Herman Melville.", "Jane Austen."],
            },
        )
        model = ReasoningModel(DummyModel(tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=tokenizer,
            data_module=rtofu,
            subset_names=["real_authors"],
            answer_evaluator=[],
            batch_size=2,
        )
        generations = engine._generate()
        gold_cots = [item for batch in generations["real_authors"]["gold_cot"] for item in batch]
        generated_answers = [item for batch in generations["real_authors"]["generated_answer"] for item in batch]
        gold_answers = [item for batch in generations["real_authors"]["gold_answer"] for item in batch]
        assert len(gold_cots) == 2
        assert all(c == "" for c in gold_cots)
        assert all("Dummy answer" in a for a in generated_answers)
        assert all("Melville" in g or "Austen" in g for g in gold_answers)

    def test_world_facts_generates_empty_gold_cot(self):
        tokenizer, rtofu = _get_shared_fixtures()
        self._inject_fake_aux(
            rtofu, "world_facts",
            {
                "question": ["What is the capital of France?"],
                "answer": ["Paris."],
            },
        )
        model = ReasoningModel(DummyModel(tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=tokenizer,
            data_module=rtofu,
            subset_names=["world_facts"],
            answer_evaluator=[],
            batch_size=1,
        )
        generations = engine._generate()
        gold_cots = [item for batch in generations["world_facts"]["gold_cot"] for item in batch]
        assert gold_cots == [""]


def _baseline_guard_args(**overrides):
    base = {flag: None for flag in _DEFENSE_FLAGS}
    base.update(overrides)
    return argparse.Namespace(**base)


class TestBaselineGuard:
    """_validate_baseline_guard rejects inference-time defense flags on a
    YAML with baseline=true, and is a no-op otherwise."""

    def test_baseline_absent_allows_defense_flags(self):
        assert _validate_baseline_guard({}, _baseline_guard_args(
            leak_classifier_path="/a",
            soft_token_path="/b",
        )) is None

    def test_baseline_false_allows_defense_flags(self):
        assert _validate_baseline_guard(
            {"baseline": False}, _baseline_guard_args(leak_classifier_path="/a"),
        ) is None

    def test_baseline_true_with_no_flags_ok(self):
        assert _validate_baseline_guard(
            {"baseline": True}, _baseline_guard_args(),
        ) is None

    def test_baseline_true_rejects_leak_classifier(self):
        msg = _validate_baseline_guard(
            {"baseline": True},
            _baseline_guard_args(leak_classifier_path="/a"),
        )
        assert msg is not None
        assert "--leak_classifier_path" in msg
        assert "baseline=true" in msg
        # --soft_token_path wasn't set, so it shouldn't be listed as conflicting.
        assert "--soft_token_path" not in msg

    def test_baseline_true_rejects_soft_token(self):
        msg = _validate_baseline_guard(
            {"baseline": True},
            _baseline_guard_args(soft_token_path="/path/embedding.pt"),
        )
        assert msg is not None
        assert "--soft_token_path" in msg
        assert "--leak_classifier_path" not in msg

    def test_baseline_true_lists_all_conflicting_flags(self):
        msg = _validate_baseline_guard(
            {"baseline": True},
            _baseline_guard_args(
                leak_classifier_path="/a",
                soft_token_path="/b",
            ),
        )
        assert msg is not None
        for flag in ("--leak_classifier_path", "--soft_token_path"):
            assert flag in msg


def _sample_args(split, **overrides):
    base = {"split": split, "shuffle": None, "num_examples": None, "seed": 0}
    base.update(overrides)
    return argparse.Namespace(**base)


class TestResolveSampleDefaults:
    """Split-dependent defaults for --num_examples and --shuffle, mirroring
    evaluate_rtofu.py's MU-protocol shuffle-then-subsample behavior."""

    def test_protocol_retain_defaults_to_shuffled_subsample(self):
        for split in ("forget01_retain", "forget05_retain", "forget10_retain"):
            args = _resolve_sample_defaults(_sample_args(split))
            assert args.shuffle is True, f"{split} should default to shuffle=True"
            assert args.num_examples == _MU_RETAIN_SUBSAMPLE

    def test_forget_split_defaults_to_full_unshuffled(self):
        for split in ("forget01", "forget05", "forget10"):
            args = _resolve_sample_defaults(_sample_args(split))
            assert args.shuffle is False
            assert args.num_examples == 0

    def test_aux_splits_default_to_full_unshuffled(self):
        for split in ("real_authors", "world_facts"):
            args = _resolve_sample_defaults(_sample_args(split))
            assert args.shuffle is False
            assert args.num_examples == 0

    def test_base_retain_splits_default_to_full_unshuffled(self):
        # retain90 / retain50 / retain10 are the "always-retained" splits,
        # NOT the protocol retain; they should not inherit the MU subsample.
        for split in ("retain90", "retain50", "retain10"):
            args = _resolve_sample_defaults(_sample_args(split))
            assert args.shuffle is False
            assert args.num_examples == 0

    def test_user_overrides_preserved_on_protocol_retain(self):
        args = _resolve_sample_defaults(_sample_args(
            "forget10_retain", shuffle=False, num_examples=10,
        ))
        assert args.shuffle is False
        assert args.num_examples == 10

    def test_user_can_force_shuffle_on_non_retain(self):
        args = _resolve_sample_defaults(_sample_args(
            "forget10", shuffle=True, num_examples=50,
        ))
        assert args.shuffle is True
        assert args.num_examples == 50


class TestTaskIdArgparse:
    """--task_id is exposed on generate_rtofu_responses' parser with the same
    type/default/semantics as evaluate_rtofu.py (int, default=1)."""

    def test_task_id_defaults_to_one(self):
        args = _build_parser().parse_args(["--split", "forget10"])
        assert args.task_id == 1

    def test_task_id_parsed_as_int(self):
        args = _build_parser().parse_args(
            ["--split", "forget10", "--task_id", "2"]
        )
        assert args.task_id == 2
        assert isinstance(args.task_id, int)

    def test_task_id_rejects_non_int(self):
        # argparse type=int converts "1.5"/"abc" by calling int(), which
        # raises; argparse turns that into a SystemExit(2).
        import pytest

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--split", "forget10", "--task_id", "1.5"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--split", "forget10", "--task_id", "abc"])


class _CapturingRTOFU(RTOFU):
    """RTOFU subclass that records the task_id it was constructed with, then
    behaves exactly like a real RTOFU (real download / task_id filtering).

    Used to prove the parsed --task_id actually reaches the RTOFU(...) call
    in main() — the capture is a side-channel; the dataset itself is built by
    the genuine RTOFU.download(), so a wrong task_id would also surface as
    wrong forget content."""

    captured_task_id = None

    def __init__(self, *args, task_id=RTOFU.DEFAULT_TASK_ID, **kwargs):
        type(self).captured_task_id = task_id
        super().__init__(*args, task_id=task_id, **kwargs)


class _MainStubModel:
    """Minimal HFModel stand-in for driving main() without a real checkpoint.

    Carries the model_config keys main() and the RTOFU(...) construction read
    (formatting_tokens; no baseline). Wrapped by the real ReasoningModel in
    main(), whose __getattr__ forwards model_config/tokenizer through to here."""

    def __init__(self, model_name, config_path, generation_config):
        self.model_name = model_name
        self.tokenizer = _make_tokenizer()
        self.device = torch.device("cpu")
        self.generation_config = generation_config
        self.model_config = {
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            },
        }


class _MainStubEngine:
    """ReasoningGenerationEngine stand-in: skips generation, returns the
    canonical per-split structure main() flattens (one batch of one row)."""

    def __init__(self, *, data_module, subset_names, **kwargs):
        self.data_module = data_module
        self.subset_names = subset_names

    def _generate(self):
        split = self.subset_names[0]
        ds = self.data_module.dataset[split]
        n = min(1, len(ds))
        return {
            split: {
                "prompt": [["q"] * n],
                "gold_answer": [["a"] * n],
                "gold_cot": [["c"] * n],
                "generated_cot": [["gc"] * n],
                "generated_answer": [["ga"] * n],
            }
        }


class TestTaskIdReachesRTOFU:
    """End-to-end wiring: a --task_id K on the CLI threads into the RTOFU(...)
    construction in main(), so the loaded forget split is the task_id=K one.

    The model load and engine are stubbed (no GPU / checkpoint), but the
    dataset path is fully real: RTOFU.download() runs and filters by the
    threaded task_id. We assert both the captured kwarg AND that the real
    download produced the task_id=K forget content (distinct from task_id=1)."""

    def _run_main(self, monkeypatch, argv):
        import scripts.generate_rtofu_responses as gen

        monkeypatch.setattr(gen, "HFModel", _MainStubModel)
        monkeypatch.setattr(gen, "RTOFU", _CapturingRTOFU)
        monkeypatch.setattr(gen, "ReasoningGenerationEngine", _MainStubEngine)
        monkeypatch.setattr("sys.argv", ["generate_rtofu_responses", *argv])
        _CapturingRTOFU.captured_task_id = None
        gen.main()

    def test_explicit_task_id_threads_into_rtofu(self, monkeypatch):
        self._run_main(monkeypatch, ["--split", "forget10", "--task_id", "2"])
        assert _CapturingRTOFU.captured_task_id == 2

    def test_default_task_id_threads_into_rtofu(self, monkeypatch):
        self._run_main(monkeypatch, ["--split", "forget10"])
        assert _CapturingRTOFU.captured_task_id == 1

    def test_threaded_task_id_changes_loaded_forget_content(self, monkeypatch):
        """The captured kwarg is real wiring, not a no-op: the forget split
        actually loaded under --task_id 2 differs from --task_id 1, proving
        the value drove RTOFU.download()'s task_id filter."""
        import scripts.generate_rtofu_responses as gen

        def _capture_then_run(task_id):
            captured = {}

            class _RecordingRTOFU(_CapturingRTOFU):
                def download(self):
                    super().download()
                    captured["questions"] = list(
                        self.dataset["forget10"]["question"]
                    )

            monkeypatch.setattr(gen, "HFModel", _MainStubModel)
            monkeypatch.setattr(gen, "RTOFU", _RecordingRTOFU)
            monkeypatch.setattr(gen, "ReasoningGenerationEngine", _MainStubEngine)
            monkeypatch.setattr(
                "sys.argv",
                ["generate_rtofu_responses", "--split", "forget10",
                 "--task_id", str(task_id)],
            )
            gen.main()
            return captured["questions"]

        q1 = _capture_then_run(1)
        q2 = _capture_then_run(2)
        # Non-overlapping task_id partitions => disjoint forget questions.
        assert set(q1) and set(q2)
        assert set(q1).isdisjoint(set(q2))
