"""Unit tests for `evaluate_rtofu_from_csv._validate_jobs`.

Up-front validation guards the jobs-file sweep mode: malformed entries fail
with a clear, located error message before the expensive evaluator-load
step runs. Tests cover the four failure modes the validator detects
(non-list top level, non-dict entries, missing/unknown keys, invalid
split) and a happy path.
"""
import pytest

from scripts.evaluate_rtofu_from_csv import _validate_jobs


VALID_JOB = {
    "csv": "results/forget10/forget10.csv",
    "split": "forget10",
    "output_json": "results/forget10/eval.json",
}


class TestValidateJobs:
    def test_happy_path_no_raise(self):
        _validate_jobs([VALID_JOB], "jobs.json")

    def test_happy_path_with_all_optional_keys(self):
        job = {
            **VALID_JOB,
            "treatment": "GA3-forget10",
            "config": "forget10",
            "rtofu_judge_scores": "rtofu_judge/forget10/GA3.jsonl",
            "judge_afe_scores": "rtofu_judge/forget10/GA3_afe.jsonl",
        }
        _validate_jobs([job], "jobs.json")

    def test_happy_path_with_rtofu_judge_scores_only(self):
        # rtofu_judge_scores is accepted on its own.
        job = {**VALID_JOB,
               "rtofu_judge_scores": "rtofu_judge/forget10/X.jsonl"}
        _validate_jobs([job], "jobs.json")

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="non-empty JSON list"):
            _validate_jobs({"csv": "x"}, "jobs.json")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty JSON list"):
            _validate_jobs([], "jobs.json")

    def test_non_dict_entry_raises(self):
        with pytest.raises(ValueError, match=r"jobs\.json\[0\]: expected an object"):
            _validate_jobs(["a string"], "jobs.json")

    def test_missing_csv_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            _validate_jobs([{"split": "forget10"}], "jobs.json")

    def test_missing_split_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            _validate_jobs([{"csv": "x"}], "jobs.json")

    def test_missing_both_required_keys_lists_both(self):
        with pytest.raises(ValueError, match=r"\['csv', 'split'\]"):
            _validate_jobs([{"output_json": "x"}], "jobs.json")

    def test_unknown_key_raises_with_typo_hint(self):
        """A typo like 'lables_dir' must not silently degrade to 'no labels'."""
        job = {**VALID_JOB, "lables_dir": "y"}
        with pytest.raises(ValueError, match=r"unknown key.*lables_dir"):
            _validate_jobs([job], "jobs.json")

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="forget999"):
            _validate_jobs([{"csv": "x", "split": "forget999"}], "jobs.json")

    def test_error_points_to_offending_index(self):
        jobs = [VALID_JOB, VALID_JOB, {"csv": "x"}]  # bad entry at index 2
        with pytest.raises(ValueError, match=r"jobs\.json\[2\]"):
            _validate_jobs(jobs, "jobs.json")

    def test_source_path_appears_in_message(self):
        with pytest.raises(ValueError, match=r"/path/to/jobs\.json"):
            _validate_jobs([], "/path/to/jobs.json")

    def test_mu_all_three_csvs_accepted(self):
        job = {
            **VALID_JOB,
            "mu_retain_csv": "r.csv",
            "mu_real_authors_csv": "a.csv",
            "mu_world_facts_csv": "w.csv",
        }
        _validate_jobs([job], "jobs.json")

    def test_mu_partial_triple_raises(self):
        """Passing one or two MU CSVs without the third must be rejected
        so an 8-value MU hmean (instead of 12) doesn't silently propagate."""
        job = {**VALID_JOB, "mu_retain_csv": "r.csv"}
        with pytest.raises(ValueError, match=r"MU requires all three"):
            _validate_jobs([job], "jobs.json")

    def test_mu_two_of_three_raises(self):
        job = {
            **VALID_JOB,
            "mu_retain_csv": "r.csv",
            "mu_real_authors_csv": "a.csv",
        }
        with pytest.raises(ValueError, match=r"MU requires all three"):
            _validate_jobs([job], "jobs.json")
