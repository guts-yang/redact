"""R-TOFU-faithful LLM-as-judge CFE scorer.

Reads continuous 0.00-1.00 scores produced by ``scripts.judge_cfe_rtofu``
(one JSONL record per attempt) and exposes them through the standard
evaluator interface so they can be plugged into ``ReasoningGenerationEngine``
or ``evaluate_rtofu_from_csv``'s CoT-evaluator list.

Per-sample score is the float from the runner's most-recent ``status=="ok"``
record for that uid. Failed attempts (``parse_failed`` / ``error``) are
silently superseded by a later successful attempt for the same uid;
they don't block the lookup. Unknown uids — or uids with no successful
record — raise, since a partial CFE based on missing rows is misleading.

Single-pass cursor: construct one instance per subset's full uid list,
hand it to ``score_csv``, discard. Reusing across subsets is a bug.
"""
import json
import os

from redact.utils import log_print


class LLMJudgeRtofuScore:
    name = "llm_judge_rtofu_score"

    def __init__(self, scores_path: str, uids: list[str]):
        if not os.path.exists(scores_path):
            raise FileNotFoundError(
                f"LLMJudgeRtofuScore: scores file not found: {scores_path}. "
                f"Run scripts/judge_cfe_rtofu.py first to produce it."
            )
        self.scores: dict[str, float] = {}
        self.judge_models: list[str] = []
        # Take the most-recent OK record per uid. JSONL is append-only;
        # later records supersede earlier ones for the same uid.
        latest: dict[str, dict] = {}
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = rec.get("uid")
                if uid is None:
                    continue
                latest[uid] = rec
        models_seen: set[str] = set()
        for uid, rec in latest.items():
            if rec.get("status") == "ok" and rec.get("score") is not None:
                self.scores[uid] = float(rec["score"])
                if rec.get("model"):
                    models_seen.add(str(rec["model"]))
        self.judge_models = sorted(models_seen)

        self.uids = list(uids)
        self._cursor = 0

        n_total = len(self.uids)
        n_resolved = sum(1 for u in self.uids if u in self.scores)
        models_str = (
            self.judge_models[0] if len(self.judge_models) == 1
            else f"{self.judge_models} (mixed)"
        ) if self.judge_models else "<unknown>"
        log_print(
            f"[info] LLMJudgeRtofuScore: loaded {len(self.scores)} OK scores "
            f"from {scores_path} (judge_model={models_str}); "
            f"{n_resolved}/{n_total} target uids resolved."
        )
        if n_resolved < n_total:
            missing = [u for u in self.uids if u not in self.scores]
            sample = missing[:5]
            n_missing = n_total - n_resolved
            pct = 100.0 * n_missing / n_total
            raise ValueError(
                f"LLMJudgeRtofuScore: {n_missing}/{n_total} uids "
                f"({pct:.1f}%) have no OK score in {scores_path}. "
                f"First missing: {sample}. Backfill by re-running "
                f"`python -m scripts.judge_cfe_rtofu` against the same "
                f"CSV with the same --treatment / --config / --output_jsonl "
                f"(failed/missing uids will be retried; OK uids skipped)."
            )

    def evaluate(self, answers, generated_answers):
        """Return the judge score per sample at the current cursor slice.

        The cursor advances by ``len(answers)`` each call, mirroring how
        the engine feeds mini-batches. ``answers`` and ``generated_answers``
        are accepted for interface compatibility but unused — the score
        is keyed by uid, not content.
        """
        assert len(answers) == len(generated_answers), (
            "LLMJudgeRtofuScore: answers/generated_answers length mismatch"
        )
        n = len(answers)
        batch_uids = self.uids[self._cursor : self._cursor + n]
        if len(batch_uids) != n:
            raise RuntimeError(
                f"LLMJudgeRtofuScore: cursor overrun at {self._cursor} with "
                f"batch size {n}; only {len(batch_uids)} uids remain "
                f"(total {len(self.uids)}). This evaluator is single-pass — "
                f"construct a fresh instance per subset, or extend the uid "
                f"list to cover every sample."
            )
        self._cursor += n
        return [self.scores[u] for u in batch_uids]
