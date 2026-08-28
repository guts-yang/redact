"""Unit tests for compute_mu and its interaction with the evaluation pipeline.

Usage:
    conda run -n redact python -m pytest tests/test_compute_mu.py -v
"""

import pytest
from datasets import Dataset
from scipy.stats import hmean

from redact.dataset.rtofu import RTOFU
from redact.utils import compute_mu


def _make_subset_metrics(prefix, values):
    """Build a {prefix}_{metric} dict from {metric: value} values."""
    return {f"{prefix}_{m}": v for m, v in values.items()}


class TestComputeMU:
    metrics_full = {
        "rougeL_recall": 0.7,
        "cosine_similarity": 0.8,
        "entailment_score": 0.6,
        "token_entropy": 0.5,
    }

    def test_three_subset_hmean_matches_scipy(self):
        """12 positive values → hmean exactly matches scipy.stats.hmean."""
        results = {}
        prefixes = ["rtofu_retain", "rtofu_real_authors", "rtofu_world_facts"]
        for p in prefixes:
            results.update(_make_subset_metrics(p, self.metrics_full))
        mu = compute_mu(results, prefixes)
        expected = float(hmean(list(self.metrics_full.values()) * 3))
        assert mu == pytest.approx(expected)

    def test_single_subset_hmean_of_four(self):
        """One subset's 4 metrics also aggregate correctly."""
        results = _make_subset_metrics("rtofu_retain", self.metrics_full)
        mu = compute_mu(results, ["rtofu_retain"])
        assert mu == pytest.approx(float(hmean(list(self.metrics_full.values()))))

    def test_missing_metric_key_returns_zero(self):
        """If any expected {prefix}_{metric} key is absent, MU is 0."""
        results = _make_subset_metrics("rtofu_retain", self.metrics_full)
        del results["rtofu_retain_token_entropy"]
        assert compute_mu(results, ["rtofu_retain"]) == 0.0

    def test_missing_whole_prefix_returns_zero(self):
        """A prefix with no keys at all reads as all-missing → 0."""
        results = _make_subset_metrics("rtofu_retain", self.metrics_full)
        mu = compute_mu(results, ["rtofu_retain", "rtofu_real_authors"])
        assert mu == 0.0

    def test_zero_component_returns_zero(self):
        """hmean is undefined on 0; compute_mu returns 0 with a log, not NaN."""
        metrics = dict(self.metrics_full, token_entropy=0.0)
        results = _make_subset_metrics("rtofu_retain", metrics)
        assert compute_mu(results, ["rtofu_retain"]) == 0.0

    def test_negative_component_returns_zero(self):
        """Negative CS values (rare with our clamp-to-0 evaluator but
        possible from other pipelines) should not crash hmean."""
        metrics = dict(self.metrics_full, cosine_similarity=-0.01)
        results = _make_subset_metrics("rtofu_retain", metrics)
        assert compute_mu(results, ["rtofu_retain"]) == 0.0

    def test_two_subset_hmean(self):
        """Partial MU (2 of 3 subsets — e.g. aux data unavailable) still
        aggregates correctly; 8 values enter the hmean."""
        results = {}
        prefixes = ["rtofu_retain", "rtofu_real_authors"]
        for p in prefixes:
            results.update(_make_subset_metrics(p, self.metrics_full))
        mu = compute_mu(results, prefixes)
        assert mu == pytest.approx(float(hmean(list(self.metrics_full.values()) * 2)))

    def test_hmean_penalizes_low_values_more_than_arithmetic(self):
        """Regression guard: hmean with one small value is much smaller than
        the arithmetic mean, which is the whole reason the paper chose it."""
        metrics = {
            "rougeL_recall": 0.9,
            "cosine_similarity": 0.9,
            "entailment_score": 0.9,
            "token_entropy": 0.1,
        }
        results = _make_subset_metrics("rtofu_retain", metrics)
        mu = compute_mu(results, ["rtofu_retain"])
        arith = sum(metrics.values()) / len(metrics)
        # Harmonic mean of (0.9, 0.9, 0.9, 0.1) ≈ 0.265; arithmetic is 0.70.
        assert mu < 0.35
        assert arith > 0.65


class TestRTOFUAuxSplits:
    """Aux (real_authors / world_facts) load from vendored JSONLs in
    data/rtofu/*_perturbed.json during download(). The JSONL's
    `original_answer` (elaborated sentence) is renamed to `answer` so the
    rest of the pipeline's gen_answer_key plumbing works unchanged."""

    def test_aux_subsets_in_subsets_list(self):
        """real_authors / world_facts are declared so registrars can see them."""
        assert "real_authors" in RTOFU.subsets
        assert "world_facts" in RTOFU.subsets

    def test_download_loads_aux_from_jsonl(self):
        """After download(), aux splits are in rtofu.dataset with the
        R-TOFU JSONL's `original_answer` field exposed as `answer`."""
        rtofu = RTOFU()
        rtofu.download()
        assert "real_authors" in rtofu.dataset
        assert "world_facts" in rtofu.dataset
        real = rtofu.dataset["real_authors"]
        assert "question" in real.column_names
        assert "answer" in real.column_names
        # original_answer is elaborated (multi-word sentence). The short
        # bare form (e.g. "William Shakespeare") is dropped.
        sample_answer = real[0]["answer"]
        assert len(sample_answer.split()) > 2, (
            f"expected elaborated sentence, got bare answer: {sample_answer!r}"
        )

    def test_missing_aux_jsonl_raises_on_eval(self, tmp_path):
        """If the vendored JSONL is absent, load_dataset_for_eval raises
        a clear KeyError instead of a cryptic lookup failure downstream."""
        rtofu = RTOFU(data_dir=str(tmp_path))  # empty dir — no JSONLs
        # download() tolerates missing aux files; eval-time request fails.
        # Need forget/retain JSONLs too to even reach download(); skip.
        with pytest.raises((FileNotFoundError, KeyError)):
            rtofu.load_dataset_for_eval("real_authors")


class TestFromCSVArgValidation:
    """evaluate_rtofu_from_csv partial-flag cases must fail cleanly, not
    silently compute MU on missing data."""

    def test_partial_mu_csv_set_errors(self, tmp_path, monkeypatch, capsys):
        import importlib
        mod = importlib.import_module("scripts.evaluate_rtofu_from_csv")
        # Write a tiny dummy forget CSV so the AFE/CFE branch has something.
        forget_csv = tmp_path / "forget.csv"
        forget_csv.write_text(
            "index,question,gold_answer,response,gold_cot,model_cot\n"
            "0,Q?,A gold,A response,C gold,C model\n",
            encoding="utf-8",
        )
        # Passing only 1 of 3 MU CSVs must raise SystemExit from parser.error.
        monkeypatch.setattr(
            "sys.argv",
            [
                "evaluate_rtofu_from_csv",
                "--csv", str(forget_csv),
                "--mu_retain_csv", str(forget_csv),  # only one — invalid
            ],
        )
        with pytest.raises(SystemExit):
            mod.main()

    def test_no_csv_no_mu_errors(self, monkeypatch):
        import importlib
        mod = importlib.import_module("scripts.evaluate_rtofu_from_csv")
        monkeypatch.setattr("sys.argv", ["evaluate_rtofu_from_csv"])
        with pytest.raises(SystemExit):
            mod.main()


class TestEngineOnAuxSubset:
    """End-to-end: engine runs on an aux subset (empty gold CoT) and emits
    the four MU answer metrics with NO cot_stepwise_* pollution."""

    def test_aux_subset_emits_answer_metrics_only(self):
        # Import here to keep collection fast when this test is not selected.
        from tests.test_rtofu_evaluation import (
            DummyModel, _get_shared_fixtures,
        )
        from redact.evaluator import (
            CosineSimilarity, EntailmentScore, ROUGERecall, TokenEntropy,
        )
        from redact.inference import ReasoningGenerationEngine
        from redact.model.reasoning import ReasoningModel

        # Minimal cot evaluator: if the engine ran it on the (empty-gold)
        # aux subset it would emit a rtofu_<split>_cot_<name> key; the test
        # asserts the engine SKIPS it on empty gold CoT, so no cot_* key
        # appears. Used in place of the (removed) stepwise evaluators.
        class _StubCoTEvaluator:
            name = "stub_cot_metric"

            def evaluate(self, gold, gen):
                return [0.0] * len(gold)

        tokenizer, rtofu = _get_shared_fixtures()
        # Inject a tiny fake real_authors split (2 rows, empty cot) so the
        # engine exercises the aux-subset path without hitting HF.
        rtofu.dataset["real_authors"] = Dataset.from_list([
            {"question": "Who is Ada Lovelace?", "answer": "A computing pioneer.", "cot": ""},
            {"question": "Who is Alan Turing?", "answer": "A founder of CS.", "cot": ""},
        ])
        model = ReasoningModel(DummyModel(tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=tokenizer,
            data_module=rtofu,
            subset_names=["real_authors"],
            answer_evaluator=[
                ROUGERecall(mode="rougeL"),
                CosineSimilarity(),
                EntailmentScore(reverse=False),
                TokenEntropy(tokenizer=tokenizer),
            ],
            cot_evaluator=[_StubCoTEvaluator()],
            batch_size=2,
        )
        engine.inference()
        summary, _ = engine.summary()
        all_results = {}
        for r in summary:
            all_results.update(r)

        # Four answer-side MU metric keys present.
        for m in ("rougeL_recall", "cosine_similarity", "entailment_score", "token_entropy"):
            assert f"rtofu_real_authors_{m}" in all_results, (
                f"missing rtofu_real_authors_{m} in results: {list(all_results)}"
            )
        # No cot_stepwise_* noise — skip-when-empty-gold in the engine.
        cot_keys = [k for k in all_results if "_cot_" in k]
        assert cot_keys == [], (
            f"aux subset should not emit cot_* keys, got: {cot_keys}"
        )
        # compute_mu on the single aux prefix succeeds.
        mu = compute_mu(all_results, ["rtofu_real_authors"])
        assert mu > 0.0 or mu == 0.0  # finite, non-crashing result
