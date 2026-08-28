"""Unit tests for `scripts.judge_cfe_rtofu`.

Covers the regex-based score parser, prompt construction (verbatim with
the R-TOFU upstream f-string), uid construction (matches existing
convention so the evaluator can resolve scores), the resume helper, and
the retry loop in `judge_one` with the subprocess mocked.
"""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.judge_cfe_rtofu import (
    PROMPT_TEMPLATE,
    SCORE_REGEX,
    build_prompt,
    build_uid,
    derive_treatment_config,
    judge_one,
    load_existing_ok_uids,
    parse_score,
)


class TestParseScore:
    def test_clean_number(self):
        assert parse_score("0.42") == 0.42

    def test_with_trailing_period(self):
        assert parse_score("0.42.") == 0.42

    def test_with_prose_prefix(self):
        assert parse_score("Score: 0.42") == 0.42

    def test_with_prose_suffix(self):
        assert parse_score("0.42\n\nReason: the model partially leaks") == 0.42

    def test_zero(self):
        assert parse_score("0.00") == 0.0

    def test_one(self):
        assert parse_score("1.00") == 1.0

    def test_first_match_wins(self):
        # If the model emits multiple .NN numbers, regex returns the first.
        # This matches R-TOFU's upstream behavior.
        assert parse_score("0.42 or 0.99") == 0.42

    def test_no_match_returns_none(self):
        assert parse_score("I cannot evaluate this without more context.") is None

    def test_integer_only_returns_none(self):
        # The regex requires \d+\.\d{2}; integers and 1-decimal don't match.
        assert parse_score("0") is None
        assert parse_score("0.4") is None

    def test_three_decimal_truncates_to_two(self):
        # \d+\.\d{2} matches the first two decimals greedily.
        assert parse_score("0.421") == 0.42

    def test_out_of_range_returns_none(self):
        # Value above 1.00 -> not a valid forget score; flag as parse_fail.
        assert parse_score("1.42") is None

    def test_above_one_with_two_decimals_returns_none(self):
        assert parse_score("12.34") is None

    def test_empty_returns_none(self):
        assert parse_score("") is None
        assert parse_score(None) is None  # type: ignore[arg-type]


class TestBuildPrompt:
    def test_substitutes_three_fields(self):
        out = build_prompt(
            question="Who wrote X?",
            gold_answer="Author A",
            model_cot="The user is asking about X.",
        )
        assert "Who wrote X?" in out
        assert "Author A" in out
        assert "The user is asking about X." in out

    def test_preserves_upstream_quirks(self):
        # Upstream R-TOFU prompt has a stray ``**`` after "hints at" and
        # uses "single integer score" wording (despite asking for a float).
        # We preserve both verbatim.
        out = build_prompt("q", "a", "c")
        assert "hints at**" in out
        assert "single integer score" in out

    def test_leading_newline_preserved(self):
        # Upstream f-string starts with \n right after `f"""`.
        assert PROMPT_TEMPLATE.startswith("\n")

    def test_template_does_not_use_double_braces(self):
        # Defensive: if someone refactors the template to escape braces,
        # str.format would silently produce wrong output.
        assert "{{" not in PROMPT_TEMPLATE
        assert "}}" not in PROMPT_TEMPLATE


class TestSCORE_REGEX:
    def test_matches_two_decimals(self):
        m = SCORE_REGEX.search("0.42")
        assert m is not None and m.group() == "0.42"

    def test_matches_first_in_text(self):
        m = SCORE_REGEX.search("foo 0.42 bar 0.99 baz")
        assert m is not None and m.group() == "0.42"


class TestBuildUid:
    def test_matches_response_classification_convention(self):
        # The convention is set by prepare_response_classification_batches.py
        # and consumed by both LLMJudgeCoTLeak and LLMJudgeRtofuScore.
        assert build_uid("treatment_x", "config_y", 7) == "treatment_x__config_y__q7"

    def test_zero_index(self):
        assert build_uid("t", "c", 0) == "t__c__q0"


class TestDeriveTreatmentConfig:
    def test_strips_csv_extension(self):
        t, c = derive_treatment_config("results/foo/bar.csv", None, None)
        assert t == "foo"
        assert c == "bar"

    def test_explicit_overrides(self):
        t, c = derive_treatment_config(
            "results/foo/bar.csv", "explicit_t", "explicit_c",
        )
        assert t == "explicit_t"
        assert c == "explicit_c"

    def test_csv_basename_without_extension(self):
        _, c = derive_treatment_config("results/foo/bar", None, None)
        assert c == "bar"

    def test_explicit_config_with_csv_extension_is_stripped(self):
        # User-supplied --config "bar.csv" should still work; uid
        # construction must not include the extension.
        _, c = derive_treatment_config("any.csv", None, "bar.csv")
        assert c == "bar"


class TestLoadExistingOkUids:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_existing_ok_uids(str(tmp_path / "nope.jsonl")) == set()

    def test_returns_only_ok_uids(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"uid": "u1", "status": "ok"}) + "\n")
            f.write(json.dumps({"uid": "u2", "status": "parse_failed"}) + "\n")
            f.write(json.dumps({"uid": "u3", "status": "error"}) + "\n")
        assert load_existing_ok_uids(str(path)) == {"u1"}

    def test_failed_then_ok_takes_latest(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"uid": "u1", "status": "parse_failed"}) + "\n")
            f.write(json.dumps({"uid": "u1", "status": "ok"}) + "\n")
        assert load_existing_ok_uids(str(path)) == {"u1"}

    def test_ok_then_failed_takes_latest(self, tmp_path):
        # Latest is failed -> uid not in the OK set -> would be retried.
        path = tmp_path / "scores.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"uid": "u1", "status": "ok"}) + "\n")
            f.write(json.dumps({"uid": "u1", "status": "error"}) + "\n")
        assert load_existing_ok_uids(str(path)) == set()

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"uid": "u1", "status": "ok"}) + "\n")
            f.write("\n\n")
            f.write(json.dumps({"uid": "u2", "status": "ok"}) + "\n")
        assert load_existing_ok_uids(str(path)) == {"u1", "u2"}

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"uid": "u1", "status": "ok"}) + "\n")
            f.write("not json\n")
            f.write(json.dumps({"uid": "u2", "status": "ok"}) + "\n")
        assert load_existing_ok_uids(str(path)) == {"u1", "u2"}


def _mk_proc(returncode: int, stdout: str, stderr: str = "") -> MagicMock:
    """Build a fake CompletedProcess for `subprocess.run` to return."""
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def _ok_response(text: str, cost: float = 0.01) -> str:
    return json.dumps({
        "result": text,
        "session_id": "test-session",
        "total_cost_usd": cost,
    })


class TestJudgeOne:
    def test_happy_path_first_attempt(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.return_value = _mk_proc(0, _ok_response("0.42"))
            rec = judge_one(
                uid="t__c__q0", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=2, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "ok"
        assert rec["score"] == 0.42
        assert rec["raw_output"] == "0.42"
        assert rec["model"] == "sonnet"
        assert rec["attempts"] == 1
        assert rec["cost_usd"] == pytest.approx(0.01)

    def test_parse_failed_then_succeeded_returns_ok(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.side_effect = [
                _mk_proc(0, _ok_response("I cannot evaluate this.")),  # parse fail
                _mk_proc(0, _ok_response("0.30")),                      # success
            ]
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=2, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "ok"
        assert rec["score"] == 0.30
        assert rec["attempts"] == 2

    def test_all_attempts_parse_fail_returns_parse_failed(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.return_value = _mk_proc(0, _ok_response("not a number"))
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=1, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "parse_failed"
        assert rec["score"] is None
        assert rec["raw_output"] == "not a number"
        assert rec["attempts"] == 2

    def test_subprocess_error_then_success(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.side_effect = [
                _mk_proc(1, "", "boom"),                # exits non-zero
                _mk_proc(0, _ok_response("0.50")),      # then succeeds
            ]
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=2, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "ok"
        assert rec["score"] == 0.50

    def test_all_subprocess_errors_returns_error(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.return_value = _mk_proc(1, "", "broken")
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=1, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "error"
        assert rec["score"] is None
        assert rec["raw_output"] is None
        assert "broken" in rec["error"]
        assert rec["attempts"] == 2

    def test_timeout_is_retried(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.side_effect = [
                subprocess.TimeoutExpired(cmd="claude", timeout=60),
                _mk_proc(0, _ok_response("0.10")),
            ]
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=2, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "ok"
        assert rec["score"] == 0.10

    def test_invalid_json_in_stdout_treated_as_error(self):
        with patch("scripts.judge_cfe_rtofu.subprocess.run") as mock_run, \
             patch("scripts.judge_cfe_rtofu.time.sleep"):
            mock_run.return_value = _mk_proc(0, "not json at all")
            rec = judge_one(
                uid="u", question="q", gold_answer="a", model_cot="c",
                model="sonnet", max_retries=0, timeout=60.0, claude_bin="claude",
            )
        assert rec["status"] == "error"
        assert "not JSON" in rec["error"]
