"""JudgeTuple-4 AFE consumer.

Reads categorical labels produced by ``scripts.judge_afe_rtofu`` (one
JSONL record per attempt; ``label`` in
``{LEAK, HALLUC, COLLAPSE, REFUSE}``, plus optional ``leak_span``) and
exposes split-level category rates + the headline ``AFE_judge = 1 -
leak_rate``.

Per-sample label is taken from the most-recent ``status == "ok"`` record
for each uid. Failed attempts (``parse_failed`` / ``error``) are silently
superseded by a later successful attempt. Unknown uids — or uids with
no successful record — raise, since a partial AFE based on missing
rows is misleading.

This is *not* a standard ``evaluator_classes`` member with a
``.evaluate(answers, generated_answers)`` interface. The standard
interface forces a per-row scalar score, and downstream
``apply_empty_override`` would clobber correct COLLAPSE labels (the
override is for structural metrics that can't tell empty from "fully
forgotten" — the judge already distinguishes them). Instead the caller
constructs this once per CSV, reads ``category_rates()``, and writes
the four label rates + ``AFE_judge`` into the results dict directly.

Per the JudgeTuple-4 contract, labels are mutually exclusive and
partition the split: ``leak + halluc + collapse + refuse == 1`` modulo
float. The headline answer-side forget efficacy is
``AFE_judge = 1 - leak_rate``.
"""
import json
import os

from redact.utils import log_print


class JudgeAFETuple:
    """Side-channel consumer for JudgeTuple-4 categorical scores.

    Usage:
        consumer = JudgeAFETuple(scores_path, uids)
        rates = consumer.category_rates()      # {LEAK: r, HALLUC: r, ...}
        afe = consumer.headline_afe()           # 1 - rates["LEAK"]
        labels = consumer.per_uid_labels()     # {uid: label, ...}

    Reusing across subsets is fine (no internal cursor); the load is
    keyed by uid so a fresh ``uids`` list per subset is sufficient.
    Provide *all* uids you want resolved at construction time, since
    we validate completeness up front.
    """

    LABELS: tuple[str, str, str, str] = ("LEAK", "HALLUC", "COLLAPSE", "REFUSE")

    def __init__(self, scores_path: str, uids: list[str]):
        if not os.path.exists(scores_path):
            raise FileNotFoundError(
                f"JudgeAFETuple: scores file not found: {scores_path}. "
                f"Run scripts/judge_afe_rtofu.py first to produce it."
            )

        # Most-recent record per uid; later supersedes earlier.
        latest: dict[str, dict] = {}
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = rec.get("uid")
                if uid is None:
                    continue
                latest[uid] = rec

        self._labels: dict[str, str] = {}
        self._leak_spans: dict[str, str] = {}
        models_seen: set[str] = set()
        for uid, rec in latest.items():
            if rec.get("status") != "ok":
                continue
            label = rec.get("label")
            if label not in self.LABELS:
                # Defensive: should be guaranteed by the runner's parser.
                continue
            self._labels[uid] = label
            self._leak_spans[uid] = str(rec.get("leak_span") or "")
            if rec.get("model"):
                models_seen.add(str(rec["model"]))

        self.judge_models: list[str] = sorted(models_seen)
        self.uids: list[str] = list(uids)

        n_total = len(self.uids)
        n_resolved = sum(1 for u in self.uids if u in self._labels)
        models_str = (
            self.judge_models[0] if len(self.judge_models) == 1
            else f"{self.judge_models} (mixed)"
        ) if self.judge_models else "<unknown>"
        log_print(
            f"[info] JudgeAFETuple: loaded {len(self._labels)} OK labels "
            f"from {scores_path} (judge_model={models_str}); "
            f"{n_resolved}/{n_total} target uids resolved."
        )
        if n_resolved < n_total:
            missing = [u for u in self.uids if u not in self._labels]
            sample = missing[:5]
            n_missing = n_total - n_resolved
            pct = 100.0 * n_missing / n_total
            raise ValueError(
                f"JudgeAFETuple: {n_missing}/{n_total} uids "
                f"({pct:.1f}%) have no OK label in {scores_path}. "
                f"First missing: {sample}. Backfill by re-running "
                f"`python -m scripts.judge_afe_rtofu` against the same "
                f"CSV with the same --treatment / --config / --output_jsonl "
                f"(failed/missing uids will be retried; OK uids skipped)."
            )

    def per_uid_labels(self) -> dict[str, str]:
        """Return ``{uid: label}`` for the target uids in construction order."""
        return {u: self._labels[u] for u in self.uids}

    def per_uid_leak_spans(self) -> dict[str, str]:
        """Return ``{uid: leak_span}`` for the target uids. ``leak_span`` is
        ``""`` for non-LEAK rows (the runner enforces this invariant)."""
        return {u: self._leak_spans[u] for u in self.uids}

    def category_counts(self) -> dict[str, int]:
        counts = {lab: 0 for lab in self.LABELS}
        for u in self.uids:
            counts[self._labels[u]] += 1
        return counts

    def category_rates(self) -> dict[str, float]:
        """Fraction of target uids assigned to each of the 4 labels.

        Sums to 1.0 exactly (modulo float) since labels partition.
        """
        n = len(self.uids)
        if n == 0:
            return {lab: 0.0 for lab in self.LABELS}
        counts = self.category_counts()
        return {lab: counts[lab] / n for lab in self.LABELS}

    def headline_afe(self) -> float:
        """``AFE_judge = 1 - leak_rate``. The single-number headline."""
        return 1.0 - self.category_rates()["LEAK"]
