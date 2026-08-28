"""Integration test for `score_csv(..., rtofu_judge_scores=...)`.

Exercises the wiring between (a) the runner's JSONL output, (b) the
`LLMJudgeRtofuScore` evaluator, and (c) the ``CFE_judge`` headline landing
into the results dict. No heavy ML models are loaded — only the judge
JSONL is read.
"""
import csv
import json

import pytest

from scripts.evaluate_rtofu_from_csv import score_csv


@pytest.fixture
def csv_3rows(tmp_path):
    """3-row CSV with columns score_csv reads."""
    path = tmp_path / "forget10.csv"
    with open(path, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index", "question", "gold_answer", "response",
            "gold_cot", "model_cot",
        ])
        w.writeheader()
        for i in range(3):
            w.writerow({
                "index": i, "question": f"Q{i}", "gold_answer": f"A{i}",
                "response": f"R{i}", "gold_cot": f"GC{i}", "model_cot": f"MC{i}",
            })
    return path


@pytest.fixture
def scores_jsonl(tmp_path):
    """3-row scores JSONL with uids matching what score_csv constructs."""
    path = tmp_path / "scores.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        # Default uid construction in score_csv: treatment from parent dir
        # of csv_path, config from basename(csv_path) sans .csv, then
        # f"{t}__{c}__q{index}". Here we pass explicit treatment+config
        # so we know the prefix.
        for i, score in enumerate([0.0, 0.4, 1.0]):
            f.write(json.dumps({
                "uid": f"T__C__q{i}", "score": score, "status": "ok",
                "model": "sonnet",
            }) + "\n")
    return path


class TestScoreCSVWithRtofuJudge:
    def test_emits_cfe_judge(self, csv_3rows, scores_jsonl):
        """CFE_judge = 1 - mean(judge_score). Scores 0.0/0.4/1.0 -> mean
        0.4666..., so CFE_judge should be ~0.5333... Also emits the
        per-split judge-score component and the judge-model provenance.
        """
        results = score_csv(
            csv_path=str(csv_3rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            rtofu_judge_scores=str(scores_jsonl),
        )
        assert results["rtofu_forget10_cot_llm_judge_rtofu_score"] == pytest.approx(
            (0.0 + 0.4 + 1.0) / 3
        )
        assert "CFE_judge" in results
        assert results["CFE_judge"] == pytest.approx(1.0 - (0.0 + 0.4 + 1.0) / 3)
        assert results["cfe_rtofu_judge_models"] == ["sonnet"]

    def test_omits_cfe_judge_when_flag_unset(self, csv_3rows):
        results = score_csv(
            csv_path=str(csv_3rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            rtofu_judge_scores=None,
        )
        assert "CFE_judge" not in results
        assert "cfe_rtofu_judge_models" not in results
        assert "rtofu_forget10_cot_llm_judge_rtofu_score" not in results

    def test_writes_results_json_with_cfe_judge(
        self, tmp_path, csv_3rows, scores_jsonl,
    ):
        out_json = tmp_path / "out.json"
        score_csv(
            csv_path=str(csv_3rows),
            split="forget10",
            output_json=str(out_json),
            treatment="T",
            config="C",
            rtofu_judge_scores=str(scores_jsonl),
        )
        with open(out_json) as f:
            written = json.load(f)
        assert "CFE_judge" in written
        assert written["cfe_rtofu_judge_models"] == ["sonnet"]

    def test_mixed_judge_models_surfaced_in_results(self, tmp_path, csv_3rows):
        # If the operator partially re-ran with a different model, the
        # JSONL holds both. cfe_rtofu_judge_models surfaces both so the
        # operator notices the cross-model contamination.
        scores_path = tmp_path / "scores.jsonl"
        with open(scores_path, "w") as f:
            f.write(json.dumps({
                "uid": "T__C__q0", "score": 0.0, "status": "ok",
                "model": "sonnet"}) + "\n")
            f.write(json.dumps({
                "uid": "T__C__q1", "score": 0.5, "status": "ok",
                "model": "opus"}) + "\n")
            f.write(json.dumps({
                "uid": "T__C__q2", "score": 1.0, "status": "ok",
                "model": "sonnet"}) + "\n")
        results = score_csv(
            csv_path=str(csv_3rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            rtofu_judge_scores=str(scores_path),
        )
        assert results["cfe_rtofu_judge_models"] == ["opus", "sonnet"]

    def test_missing_uids_in_jsonl_raises(self, tmp_path, csv_3rows):
        # Only 2 of the 3 uids scored -> evaluator constructor raises.
        scores_path = tmp_path / "scores.jsonl"
        with open(scores_path, "w") as f:
            f.write(json.dumps({
                "uid": "T__C__q0", "score": 0.0, "status": "ok",
                "model": "sonnet"}) + "\n")
            f.write(json.dumps({
                "uid": "T__C__q1", "score": 0.5, "status": "ok",
                "model": "sonnet"}) + "\n")
        with pytest.raises(ValueError, match="have no OK score"):
            score_csv(
                csv_path=str(csv_3rows),
                split="forget10",
                output_json=None,
                treatment="T",
                config="C",
                rtofu_judge_scores=str(scores_path),
            )

    def test_headline_ordering_with_cfe_judge(self, csv_3rows, scores_jsonl):
        """CFE_judge appears before component scores when present.

        Headlines: AFE_judge (absent here), CFE_judge, MU (absent here),
        the three non-leak rates (absent here). Only CFE_judge applies in
        this scope. It must precede every component-level rtofu_forget10_*
        key in iteration order.
        """
        results = score_csv(
            csv_path=str(csv_3rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            rtofu_judge_scores=str(scores_jsonl),
        )
        keys = list(results)
        cfe_judge_idx = keys.index("CFE_judge")
        first_component_idx = next(
            i for i, k in enumerate(keys) if k.startswith("rtofu_forget10_")
        )
        assert cfe_judge_idx < first_component_idx, (
            f"CFE_judge at index {cfe_judge_idx} should precede the first "
            f"component key at {first_component_idx} (keys={keys[:8]})"
        )
