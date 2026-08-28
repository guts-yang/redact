"""Integration test for `score_csv(..., judge_afe_scores=...)`.

Exercises the JudgeAFETuple wiring: the four LEAK/HALLUC/COLLAPSE/REFUSE
category rates, the new non-leak conditional variants
(`*_rate_among_nonleak`), the headline `AFE_judge` value, and the
output-ordering invariant that places `AFE_judge` and the non-leak
variants ahead of component-level keys.

Stubs the heavy answer/CoT evaluators so the test runs quickly and
doesn't depend on sentence-transformers / deberta load times.
"""
import csv
import json

import pytest

from scripts.evaluate_rtofu_from_csv import score_csv


def _write_csv(path, n_rows: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index", "question", "gold_answer", "response",
            "gold_cot", "model_cot",
        ])
        w.writeheader()
        for i in range(n_rows):
            w.writerow({
                "index": i, "question": f"Q{i}", "gold_answer": f"A{i}",
                "response": f"R{i}", "gold_cot": f"GC{i}", "model_cot": f"MC{i}",
            })


def _write_judge_afe_jsonl(path, labels: list[str], model: str = "sonnet") -> None:
    with open(path, "w") as f:
        for i, label in enumerate(labels):
            f.write(json.dumps({
                "uid": f"T__C__q{i}",
                "label": label,
                "leak_span": "" if label != "LEAK" else "x",
                "status": "ok",
                "model": model,
            }) + "\n")


@pytest.fixture
def csv_5rows(tmp_path):
    path = tmp_path / "forget10.csv"
    _write_csv(path, 5)
    return path


class TestScoreCSVWithJudgeAFE:
    def test_emits_non_leak_conditional_rates(self, tmp_path, csv_5rows):
        """5 rows: 1 LEAK, 2 HALLUC, 1 COLLAPSE, 1 REFUSE.

        Per-class rates over all 5:  LEAK=0.2  HALLUC=0.4  COLLAPSE=0.2  REFUSE=0.2.
        Non-leak n=4, so:  HALLUC=2/4=0.5  COLLAPSE=1/4=0.25  REFUSE=1/4=0.25.
        AFE_judge = 1 - 0.2 = 0.8.
        """
        scores_path = tmp_path / "afe_scores.jsonl"
        _write_judge_afe_jsonl(scores_path, [
            "LEAK", "HALLUC", "HALLUC", "COLLAPSE", "REFUSE",
        ])

        results = score_csv(
            csv_path=str(csv_5rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            judge_afe_scores=str(scores_path),
        )

        prefix = "rtofu_forget10"
        # Sanity: existing per-class rates unchanged.
        assert results[f"{prefix}_judge_afe_leak_rate"] == pytest.approx(0.2)
        assert results[f"{prefix}_judge_afe_halluc_rate"] == pytest.approx(0.4)
        assert results[f"{prefix}_judge_afe_collapse_rate"] == pytest.approx(0.2)
        assert results[f"{prefix}_judge_afe_refuse_rate"] == pytest.approx(0.2)
        # Non-leak conditional rates: denominator = 4 (HALLUC+COLLAPSE+REFUSE).
        assert results[f"{prefix}_judge_afe_halluc_rate_among_nonleak"] == pytest.approx(0.5)
        assert results[f"{prefix}_judge_afe_collapse_rate_among_nonleak"] == pytest.approx(0.25)
        assert results[f"{prefix}_judge_afe_refuse_rate_among_nonleak"] == pytest.approx(0.25)
        # Three non-leak conditional rates partition the non-leak subset.
        s = (results[f"{prefix}_judge_afe_halluc_rate_among_nonleak"]
             + results[f"{prefix}_judge_afe_collapse_rate_among_nonleak"]
             + results[f"{prefix}_judge_afe_refuse_rate_among_nonleak"])
        assert s == pytest.approx(1.0)
        assert results[f"{prefix}_AFE_judge"] == pytest.approx(0.8)

    def test_all_leak_emits_zero_non_leak_rates(self, tmp_path, csv_5rows):
        """When every row leaks, the non-leak denominator is 0. Spec says
        emit 0.0 (not NaN, not missing) so the keys are always present.
        """
        scores_path = tmp_path / "afe_scores.jsonl"
        _write_judge_afe_jsonl(scores_path, ["LEAK"] * 5)

        results = score_csv(
            csv_path=str(csv_5rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            judge_afe_scores=str(scores_path),
        )

        prefix = "rtofu_forget10"
        assert results[f"{prefix}_judge_afe_halluc_rate_among_nonleak"] == 0.0
        assert results[f"{prefix}_judge_afe_collapse_rate_among_nonleak"] == 0.0
        assert results[f"{prefix}_judge_afe_refuse_rate_among_nonleak"] == 0.0
        assert results[f"{prefix}_AFE_judge"] == pytest.approx(0.0)

    def test_omits_non_leak_rates_without_flag(self, csv_5rows):
        results = score_csv(
            csv_path=str(csv_5rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            judge_afe_scores=None,
        )
        prefix = "rtofu_forget10"
        assert f"{prefix}_judge_afe_halluc_rate_among_nonleak" not in results
        assert f"{prefix}_judge_afe_collapse_rate_among_nonleak" not in results
        assert f"{prefix}_judge_afe_refuse_rate_among_nonleak" not in results
        assert f"{prefix}_AFE_judge" not in results

    def test_headline_keys_ordered_first(self, tmp_path, csv_5rows):
        """AFE_judge and the three non-leak rates precede every
        rtofu_<split>_* component key in iteration order.
        """
        scores_path = tmp_path / "afe_scores.jsonl"
        _write_judge_afe_jsonl(scores_path, [
            "LEAK", "HALLUC", "HALLUC", "COLLAPSE", "REFUSE",
        ])

        results = score_csv(
            csv_path=str(csv_5rows),
            split="forget10",
            output_json=None,
            treatment="T",
            config="C",
            judge_afe_scores=str(scores_path),
        )

        prefix = "rtofu_forget10"
        keys = list(results)
        # rtofu_<split>_AFE_judge is part of the headline block even
        # though it shares the rtofu_<split>_ prefix — find it explicitly.
        idx_afe_judge = keys.index(f"{prefix}_AFE_judge")
        idx_halluc_nonleak = keys.index(f"{prefix}_judge_afe_halluc_rate_among_nonleak")
        idx_collapse_nonleak = keys.index(f"{prefix}_judge_afe_collapse_rate_among_nonleak")
        idx_refuse_nonleak = keys.index(f"{prefix}_judge_afe_refuse_rate_among_nonleak")
        # First component-level key: the per-class HALLUC rate is the first
        # rtofu_forget10_* key after the AFE_judge / non-leak block.
        idx_first_component = keys.index(f"{prefix}_judge_afe_leak_rate")
        for headline_idx in (
            idx_afe_judge, idx_halluc_nonleak, idx_collapse_nonleak, idx_refuse_nonleak,
        ):
            assert headline_idx < idx_first_component, (
                f"headline at {headline_idx} should precede component-level "
                f"first index {idx_first_component}; keys={keys[:12]}"
            )
        # Order within the headline block: AFE_judge before the three
        # non-leak rates, in the order halluc -> collapse -> refuse.
        assert idx_afe_judge < idx_halluc_nonleak < idx_collapse_nonleak < idx_refuse_nonleak

    def test_results_json_preserves_headline_order(self, tmp_path, csv_5rows):
        """JSON serialization keeps insertion order, so the written file
        reflects the same headline-first layout as the returned dict.
        """
        scores_path = tmp_path / "afe_scores.jsonl"
        _write_judge_afe_jsonl(scores_path, [
            "LEAK", "HALLUC", "HALLUC", "COLLAPSE", "REFUSE",
        ])
        out_json = tmp_path / "out.json"
        score_csv(
            csv_path=str(csv_5rows),
            split="forget10",
            output_json=str(out_json),
            treatment="T",
            config="C",
            judge_afe_scores=str(scores_path),
        )
        with open(out_json) as f:
            written = json.load(f)
        keys = list(written)
        # The headline block is first; the very first key should be
        # rtofu_forget10_AFE_judge (the only AFE_judge headline when only
        # judge_afe_scores is set — CFE_judge and MU are absent here).
        assert keys[0] == "rtofu_forget10_AFE_judge"
        # Three non-leak rates immediately follow (in halluc/collapse/refuse order).
        assert keys[1] == "rtofu_forget10_judge_afe_halluc_rate_among_nonleak"
        assert keys[2] == "rtofu_forget10_judge_afe_collapse_rate_among_nonleak"
        assert keys[3] == "rtofu_forget10_judge_afe_refuse_rate_among_nonleak"
