"""Unit tests for `redact.evaluator.LLMJudgeRtofuScore`.

The evaluator reads a JSONL produced by `scripts.judge_cfe_rtofu` and
returns continuous 0.00-1.00 scores keyed by uid. Tests cover the most-
recent-wins record resolution, the strict missing-uid raise, the
single-pass cursor semantics, and the file-not-found case.
"""
import json

import pytest

from redact.evaluator import LLMJudgeRtofuScore


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestLLMJudgeRtofuScore:
    def test_happy_path_returns_continuous_floats(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.0, "status": "ok"},
            {"uid": "t__c__q1", "score": 0.85, "status": "ok"},
            {"uid": "t__c__q2", "score": 1.0, "status": "ok"},
        ])
        ev = LLMJudgeRtofuScore(str(scores_path),
                                ["t__c__q0", "t__c__q1", "t__c__q2"])
        out = ev.evaluate(["a", "a", "a"], ["b", "b", "b"])
        assert out == [0.0, 0.85, 1.0]
        # Continuous, not 0/1 binary.
        assert any(0.0 < x < 1.0 for x in out)

    def test_failed_then_succeeded_uid_resolves_to_ok(self, tmp_path):
        # Simulates a resume: first attempt failed, second attempt succeeded.
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": None, "status": "parse_failed"},
            {"uid": "t__c__q0", "score": 0.42, "status": "ok"},
        ])
        ev = LLMJudgeRtofuScore(str(scores_path), ["t__c__q0"])
        assert ev.evaluate(["a"], ["b"]) == [0.42]

    def test_succeeded_then_failed_uid_raises(self, tmp_path):
        # If user manually appends a junk record after an OK, the latest
        # record wins -> not OK -> uid is treated as unresolved -> raise.
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.42, "status": "ok"},
            {"uid": "t__c__q0", "score": None, "status": "error"},
        ])
        with pytest.raises(ValueError, match="have no OK score"):
            LLMJudgeRtofuScore(str(scores_path), ["t__c__q0"])

    def test_unknown_uid_raises(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.0, "status": "ok"},
        ])
        with pytest.raises(ValueError, match="have no OK score"):
            LLMJudgeRtofuScore(str(scores_path),
                               ["t__c__q0", "t__c__q_missing"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scores file not found"):
            LLMJudgeRtofuScore(str(tmp_path / "does_not_exist.jsonl"),
                               ["t__c__q0"])

    def test_blank_lines_ignored(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        with open(scores_path, "w") as f:
            f.write('{"uid":"t__c__q0","score":0.5,"status":"ok"}\n')
            f.write("\n\n")  # blank lines
            f.write('{"uid":"t__c__q1","score":0.7,"status":"ok"}\n')
        ev = LLMJudgeRtofuScore(str(scores_path), ["t__c__q0", "t__c__q1"])
        assert ev.evaluate(["a", "a"], ["b", "b"]) == [0.5, 0.7]

    def test_cursor_advances_across_calls(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.1, "status": "ok"},
            {"uid": "t__c__q1", "score": 0.2, "status": "ok"},
            {"uid": "t__c__q2", "score": 0.3, "status": "ok"},
        ])
        ev = LLMJudgeRtofuScore(str(scores_path),
                                ["t__c__q0", "t__c__q1", "t__c__q2"])
        first = ev.evaluate(["a"], ["b"])
        second = ev.evaluate(["a", "a"], ["b", "b"])
        assert first == [0.1]
        assert second == [0.2, 0.3]

    def test_cursor_overrun_raises(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.1, "status": "ok"},
        ])
        ev = LLMJudgeRtofuScore(str(scores_path), ["t__c__q0"])
        ev.evaluate(["a"], ["b"])  # consumes the only uid
        with pytest.raises(RuntimeError, match="cursor overrun"):
            ev.evaluate(["a"], ["b"])  # second call -> cursor at 1, batch wants 1 more

    def test_length_mismatch_asserts(self, tmp_path):
        scores_path = tmp_path / "scores.jsonl"
        _write_jsonl(scores_path, [
            {"uid": "t__c__q0", "score": 0.0, "status": "ok"},
        ])
        ev = LLMJudgeRtofuScore(str(scores_path), ["t__c__q0"])
        with pytest.raises(AssertionError, match="length mismatch"):
            ev.evaluate(["a"], ["b", "c"])

    def test_name_attr_matches_cfe_judge_key(self):
        # score_csv builds the per-split judge-score key as
        # f"{prefix}_cot_{ev.name}" and CFE_judge = 1 - its mean. Drift
        # here would silently break the wiring in evaluate_rtofu_from_csv.
        assert LLMJudgeRtofuScore.name == "llm_judge_rtofu_score"
