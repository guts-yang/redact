"""Compute the reported AFE_judge / CFE_judge / MU metrics on pre-generated
response CSVs.

Reads an evaluate_rtofu output CSV (columns: index, question, gold_answer,
response, gold_cot, model_cot) and emits the three reported metrics without
touching the target model (no generation):

- ``AFE_judge`` (``1 - LEAK_rate`` from the 4-way LLM-judge rubric) plus the
  per-category rates and the non-leak conditional variants — produced by
  ``JudgeAFETuple`` from a ``--judge_afe_scores`` JSONL.
- ``CFE_judge`` (``1 - mean(R-TOFU graded judge score)``) — produced from a
  ``--rtofu_judge_scores`` JSONL via ``LLMJudgeRtofuScore``.
- ``MU`` (Model Utility) — harmonic mean of 12 values (ROUGE-L recall, Cosine
  Similarity, Entailment Score, Token Entropy on each of retain /
  real_authors / world_facts), matching the reference R-TOFU implementation.
  Supplied via the three ``--mu_*_csv`` files.

The unreported harmonic-mean AFE / CFE path (and the boilerplate-filtered and
stepwise variants) has been removed; this scorer emits only the reported
metrics and their components.

Output ordering: the results dict (and JSON file) places the headline
numbers — ``AFE_judge``, ``CFE_judge``, ``MU``, and the three
``judge_afe_{halluc,collapse,refuse}_rate_among_nonleak`` variants — at the
top, in that order. Component scores follow in their original order. Only the
headlines that were actually computed appear at the top; missing ones are
silently skipped. Naming asymmetry: ``AFE_judge`` and the three non-leak rates
carry the ``rtofu_<split>_`` prefix (per-split metrics from ``JudgeAFETuple``);
``CFE_judge`` and ``MU`` are bare top-level scalars.

Usage:
    # AFE_judge + CFE_judge
    python -m scripts.evaluate_rtofu_from_csv \\
        --csv results/.../forget10.csv \\
        --judge_afe_scores rtofu_judge_scores/.../afe.jsonl \\
        --rtofu_judge_scores rtofu_judge_scores/.../cfe.jsonl

    # ... + MU
    python -m scripts.evaluate_rtofu_from_csv \\
        --csv results/.../forget10.csv \\
        --mu_retain_csv results/.../retain.csv \\
        --mu_real_authors_csv results/.../real_authors.csv \\
        --mu_world_facts_csv results/.../world_facts.csv

    # MU only (skip the forget-split judge metrics)
    python -m scripts.evaluate_rtofu_from_csv \\
        --mu_retain_csv ... --mu_real_authors_csv ... --mu_world_facts_csv ...
"""
import argparse
import csv
import json
import os
from typing import Any

from redact.evaluator import (
    CosineSimilarity,
    EntailmentScore,
    LLMJudgeRtofuScore,
    ROUGERecall,
    TokenEntropy,
)
from redact.evaluator.judge_afe_tuple import JudgeAFETuple
from redact.utils import compute_mu, log_print


def load_rows(csv_path: str) -> list[dict]:
    """Load a generation CSV. Rows are taken at face value — the
    upstream engine is responsible for the CoT/answer split."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


MU_SUBSETS = ["retain", "real_authors", "world_facts"]
SPLIT_CHOICES = ["forget01", "forget05", "forget10"]
_MU_JOB_KEYS = frozenset(f"mu_{s}_csv" for s in MU_SUBSETS)
_JOB_KEYS = frozenset(
    {"csv", "split", "output_json", "treatment", "config",
     "rtofu_judge_scores", "judge_afe_scores"}
) | _MU_JOB_KEYS


def _reorder_results_with_headlines(results: dict, prefix: str) -> dict:
    """Return a new dict with headline numbers ordered first.

    Headlines (in order, only emitted when present): ``AFE_judge``,
    ``CFE_judge``, ``MU``, and the three non-leak conditional AFE rates.
    Remaining keys follow in their original insertion order.

    Applied to both the in-memory ``score_csv`` return value and the
    serialized JSON so callers see the same key order on disk and in
    memory.
    """
    headline_keys = [
        f"{prefix}_AFE_judge",
        "CFE_judge",
        "MU",
        f"{prefix}_judge_afe_halluc_rate_among_nonleak",
        f"{prefix}_judge_afe_collapse_rate_among_nonleak",
        f"{prefix}_judge_afe_refuse_rate_among_nonleak",
    ]
    reordered: dict = {}
    for k in headline_keys:
        if k in results:
            reordered[k] = results[k]
    for k, v in results.items():
        if k not in reordered:
            reordered[k] = v
    return reordered


def build_mu_evaluators(tokenizer) -> list:
    """Build the four MU evaluators (ROUGE, CS, ES, TE). Reusable across subsets."""
    return [
        ROUGERecall(mode="rougeL"),
        CosineSimilarity(),
        # reverse=True gives direction (gold, gen) — R-TOFU's direction
        # for retain / real_authors / world_facts.
        EntailmentScore(reverse=True),
        TokenEntropy(tokenizer=tokenizer),
    ]


def _load_mu_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    from redact.model.model import _maybe_repair_byte_level_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # TokenEntropy calls tokenizer.tokenize(); without the byte-level repair,
    # transformers >=5 silently returns despaced multi-word fragments that
    # corrupt the entropy score.
    _maybe_repair_byte_level_tokenizer(tokenizer)
    return tokenizer


def compute_mu_from_csvs(
    csv_paths: dict[str, str],
    tokenizer_name: str | None,
    evaluators: list | None = None,
) -> dict[str, float]:
    """Run the four MU evaluators (ROUGE, CS, ES, TE) on each MU subset CSV.

    Returns a flat dict of ``rtofu_{subset}_{metric}`` mean scores plus an
    aggregated ``MU`` key (harmonic mean over all 12 values). Each CSV must
    have the same schema as evaluate_rtofu's output (``gold_answer`` and
    ``response`` columns required; others ignored).

    Pass ``evaluators`` (from ``build_mu_evaluators``) to reuse preloaded
    models across multiple calls; omit to load-and-discard for a single
    invocation.
    """
    if evaluators is None:
        if tokenizer_name is None:
            raise ValueError("Either `evaluators` or `tokenizer_name` must be provided")
        evaluators = build_mu_evaluators(_load_mu_tokenizer(tokenizer_name))
    results: dict[str, float] = {}
    for subset, path in csv_paths.items():
        rows = load_rows(path)
        gold = [r["gold_answer"] for r in rows]
        gen = [r["response"] for r in rows]
        log_print(f"MU[{subset}]: loaded {len(rows)} rows from {path}")
        prefix = f"rtofu_{subset}"
        for ev in evaluators:
            scores = ev.evaluate(gold, gen)
            mean_score = sum(scores) / len(scores) if scores else 0.0
            results[f"{prefix}_{ev.name}"] = mean_score
            log_print(f"  {ev.name}: {mean_score:.4f}")
    mu_prefixes = [f"rtofu_{s}" for s in csv_paths]
    results["MU"] = compute_mu(results, mu_prefixes)
    return results


def _build_uids(rows: list[dict], treatment: str | None, config: str | None,
                csv_path: str) -> list[str]:
    """Build the per-row uid list keyed off treatment + config.

    Matches the uid convention used by ``judge_cfe_rtofu.py`` /
    ``judge_afe_rtofu.py`` so the judge JSONLs join row-for-row.
    """
    t = treatment or os.path.basename(os.path.dirname(csv_path))
    c = config or os.path.basename(csv_path)
    # Strip a trailing ".csv" so uid construction matches the judge scripts
    # regardless of whether the user passed the file with or without it.
    if c.endswith(".csv"):
        c = c[:-4]
    return [
        f"{t}__{c}__q{int(r.get('index', i))}"
        for i, r in enumerate(rows)
    ]


def score_csv(
    csv_path: str,
    split: str,
    output_json: str | None,
    treatment: str | None,
    config: str | None,
    mu_csvs: dict[str, str] | None = None,
    mu_tokenizer_name: str | None = None,
    mu_evaluators: list | None = None,
    rtofu_judge_scores: str | None = None,
    judge_afe_scores: str | None = None,
) -> dict:
    """Score one CSV and optionally save results JSON. Returns results dict.

    Emits the reported metrics only: ``AFE_judge`` + its category/non-leak
    rates (from ``judge_afe_scores``), ``CFE_judge`` (from
    ``rtofu_judge_scores``), and ``MU`` (from the ``mu_csvs`` triple).
    """
    rows = load_rows(csv_path)
    log_print(f"Loaded {len(rows)} rows from {csv_path}")

    gold_cots = [r["gold_cot"] for r in rows]
    generated_cots = [r["model_cot"] for r in rows]

    uids: list[str] | None = None
    if rtofu_judge_scores is not None or judge_afe_scores is not None:
        uids = _build_uids(rows, treatment, config, csv_path)

    prefix = f"rtofu_{split}"
    # Mostly float metrics, plus an optional provenance field
    # (cfe_rtofu_judge_models is a list[str]). Typed as Any so all
    # JSON-serializable values share one results dict.
    results: dict[str, Any] = {}

    # AFE_judge: 1 - LEAK_rate from the 4-way LLM-judge rubric, plus the
    # per-category rates and the non-leak conditional variants.
    if judge_afe_scores is not None:
        assert uids is not None
        judge_afe = JudgeAFETuple(judge_afe_scores, uids)
        rates = judge_afe.category_rates()
        results[f"{prefix}_judge_afe_leak_rate"] = rates["LEAK"]
        results[f"{prefix}_judge_afe_halluc_rate"] = rates["HALLUC"]
        results[f"{prefix}_judge_afe_collapse_rate"] = rates["COLLAPSE"]
        results[f"{prefix}_judge_afe_refuse_rate"] = rates["REFUSE"]
        # Non-leak conditional rates: each non-leak category as a share of
        # the non-leaking subset (denominator = HALLUC + COLLAPSE + REFUSE).
        # When every sample leaked the ratios are undefined (0/0); we emit
        # 0.0 so the keys are always present and JSON-clean.
        counts = judge_afe.category_counts()
        n_nonleak = counts["HALLUC"] + counts["COLLAPSE"] + counts["REFUSE"]
        for cat, key in (
            ("HALLUC", "halluc"), ("COLLAPSE", "collapse"), ("REFUSE", "refuse"),
        ):
            results[f"{prefix}_judge_afe_{key}_rate_among_nonleak"] = (
                counts[cat] / n_nonleak if n_nonleak > 0 else 0.0
            )
        results[f"{prefix}_AFE_judge"] = judge_afe.headline_afe()

    # CFE_judge: complement of the mean R-TOFU graded judge score.
    if rtofu_judge_scores is not None:
        assert uids is not None
        rtofu_judge = LLMJudgeRtofuScore(rtofu_judge_scores, uids)
        scores = rtofu_judge.evaluate(gold_cots, generated_cots)
        mean_judge = sum(scores) / len(scores) if scores else 0.0
        results[f"{prefix}_cot_{rtofu_judge.name}"] = mean_judge
        results["CFE_judge"] = 1.0 - mean_judge
        # Provenance: which judge model(s) produced the scores. List (not
        # single string) to surface mixed-model JSONLs for the operator.
        results["cfe_rtofu_judge_models"] = rtofu_judge.judge_models

    if mu_csvs:
        mu_results = compute_mu_from_csvs(
            mu_csvs, mu_tokenizer_name, evaluators=mu_evaluators,
        )
        results.update(mu_results)
        log_print(f"MU: {results['MU']:.4f}")

    if judge_afe_scores is not None:
        log_print(
            f"AFE_judge: {results[f'{prefix}_AFE_judge']:.4f} "
            f"(LEAK={results[f'{prefix}_judge_afe_leak_rate']:.4f} "
            f"HALLUC={results[f'{prefix}_judge_afe_halluc_rate']:.4f} "
            f"COLLAPSE={results[f'{prefix}_judge_afe_collapse_rate']:.4f} "
            f"REFUSE={results[f'{prefix}_judge_afe_refuse_rate']:.4f})"
        )
    if rtofu_judge_scores is not None:
        log_print(
            f"CFE_judge: {results['CFE_judge']:.4f} "
            f"(= 1 - mean R-TOFU judge score)"
        )

    results = _reorder_results_with_headlines(results, prefix)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        log_print(f"Results saved to {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a single results CSV for the forget-split "
                             "judge metrics. Omit to run MU only (requires "
                             "--mu_*_csv).")
    parser.add_argument("--rtofu_judge_scores", type=str, default=None,
                        help="Path to a JSONL of R-TOFU-faithful LLM-judge "
                             "scores (produced by scripts/judge_cfe_rtofu.py). "
                             "When set, emits "
                             "rtofu_<split>_cot_llm_judge_rtofu_score and "
                             "CFE_judge = 1 - mean(judge_score).")
    parser.add_argument("--judge_afe_scores", type=str, default=None,
                        help="Path to a JSONL of JudgeTuple-4 categorical "
                             "labels (produced by scripts/judge_afe_rtofu.py). "
                             "When set, emits rtofu_<split>_judge_afe_"
                             "{leak,halluc,collapse,refuse}_rate, the "
                             "non-leak conditional variants rtofu_<split>_"
                             "judge_afe_{halluc,collapse,refuse}_rate_among_nonleak "
                             "(denominator = non-leak count), plus "
                             "rtofu_<split>_AFE_judge = 1 - leak_rate.")
    parser.add_argument("--treatment", type=str, default=None,
                        help="Treatment name for uid construction "
                             "(default: parent directory of --csv)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config name for uid construction "
                             "(default: basename of --csv)")
    parser.add_argument("--split", type=str, default="forget10",
                        choices=SPLIT_CHOICES)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save the results JSON")
    parser.add_argument("--mu_retain_csv", type=str, default=None,
                        help="Path to a CSV of generations on the protocol "
                             "retain split (columns: gold_answer, response). "
                             "Required together with --mu_real_authors_csv "
                             "and --mu_world_facts_csv to compute MU.")
    parser.add_argument("--mu_real_authors_csv", type=str, default=None,
                        help="Path to a CSV of generations on the TOFU "
                             "Real Authors split (for MU).")
    parser.add_argument("--mu_world_facts_csv", type=str, default=None,
                        help="Path to a CSV of generations on the TOFU "
                             "World Facts split (for MU).")
    parser.add_argument("--mu_tokenizer", type=str,
                        default="sangyon/LRM-target",
                        help="HuggingFace tokenizer name/path for MU "
                             "TokenEntropy. Defaults to the target LRM so "
                             "TE is computed on the same vocabulary used at "
                             "generation time.")
    parser.add_argument("--jobs_file", type=str, default=None,
                        help="Path to a JSON file listing multiple scoring "
                             "jobs; MU evaluators load once and are reused "
                             "across all jobs. Each job is an object with "
                             "required keys: csv, split; and optional keys: "
                             "output_json, treatment, config, "
                             "rtofu_judge_scores, judge_afe_scores, and the "
                             "mu_{retain,real_authors,world_facts}_csv triple "
                             "(all three or none; validated up front). "
                             "Mutually exclusive with --csv / --mu_*_csv.")
    args = parser.parse_args()

    if args.jobs_file is not None:
        if args.csv is not None or any(
            getattr(args, f"mu_{s}_csv") for s in MU_SUBSETS
        ):
            parser.error("--jobs_file is mutually exclusive with --csv and --mu_*_csv")
        _run_jobs_file(args)
        return

    mu_csvs = {
        "retain": args.mu_retain_csv,
        "real_authors": args.mu_real_authors_csv,
        "world_facts": args.mu_world_facts_csv,
    }
    mu_given = {k: v for k, v in mu_csvs.items() if v}
    if mu_given and len(mu_given) != len(mu_csvs):
        parser.error(
            "MU requires all three of --mu_retain_csv, "
            "--mu_real_authors_csv, --mu_world_facts_csv "
            f"(got only: {sorted(mu_given)})"
        )
    if args.csv is None and not mu_given:
        parser.error("Must pass --csv (judge metrics) and/or the --mu_*_csv set (MU)")

    # MU-only fast path: no forget-split CSV needed.
    if args.csv is None:
        final_results = compute_mu_from_csvs(mu_csvs, args.mu_tokenizer)
        log_print(f"MU: {final_results['MU']:.4f}")
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(final_results, f, indent=2)
            log_print(f"Results saved to {args.output_json}")
        return

    mu_evaluators = (
        build_mu_evaluators(_load_mu_tokenizer(args.mu_tokenizer))
        if mu_given else None
    )

    score_csv(
        csv_path=args.csv,
        split=args.split,
        output_json=args.output_json,
        treatment=args.treatment,
        config=args.config,
        mu_csvs=mu_given if mu_given else None,
        mu_tokenizer_name=args.mu_tokenizer,
        mu_evaluators=mu_evaluators,
        rtofu_judge_scores=args.rtofu_judge_scores,
        judge_afe_scores=args.judge_afe_scores,
    )


def _validate_jobs(jobs, source: str) -> None:
    """Validate a jobs list up front so a malformed entry at position 47
    fails before we pay the evaluator-load cost, and so typos
    (``lables_dir``) don't silently degrade via ``job.get(...)``. Raises
    ``ValueError`` on the first problem; the message carries the jobs-file
    path and entry index so the user can locate the bad job.
    """
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{source}: expected a non-empty JSON list of job objects")
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"{source}[{i}]: expected an object, got {type(job).__name__}")
        missing = {"csv", "split"} - job.keys()
        if missing:
            raise ValueError(
                f"{source}[{i}]: missing required key(s) {sorted(missing)}; "
                f"got {sorted(job)}"
            )
        unknown = set(job) - _JOB_KEYS
        if unknown:
            raise ValueError(
                f"{source}[{i}]: unknown key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_JOB_KEYS)}"
            )
        if job["split"] not in SPLIT_CHOICES:
            raise ValueError(
                f"{source}[{i}]: split={job['split']!r} not in {SPLIT_CHOICES}"
            )
        mu_keys_present = _MU_JOB_KEYS & job.keys()
        if mu_keys_present and len(mu_keys_present) != len(_MU_JOB_KEYS):
            raise ValueError(
                f"{source}[{i}]: MU requires all three of "
                f"{sorted(_MU_JOB_KEYS)}; got only {sorted(mu_keys_present)}"
            )


def _run_jobs_file(args) -> None:
    """Score every job in ``args.jobs_file`` with shared preloaded MU evaluators.

    Each job is a dict: required ``csv``, ``split``; optional
    ``output_json``, ``treatment``, ``config``, ``rtofu_judge_scores``,
    ``judge_afe_scores``, and the ``mu_{retain,real_authors,world_facts}_csv``
    triple (all three or none — validated up front). Splits come from each
    job. MU evaluators are built once (using the MU tokenizer) if any job
    requests MU.
    """
    with open(args.jobs_file) as f:
        jobs = json.load(f)
    _validate_jobs(jobs, args.jobs_file)

    any_mu = any(_MU_JOB_KEYS & job.keys() for job in jobs)
    mu_evaluators = (
        build_mu_evaluators(_load_mu_tokenizer(args.mu_tokenizer))
        if any_mu else None
    )
    mu_note = " (+MU)" if any_mu else ""
    log_print(f"Loaded MU evaluators once{mu_note}; scoring {len(jobs)} jobs...")

    for i, job in enumerate(jobs):
        csv_path = job["csv"]
        split = job["split"]
        output_json = job.get("output_json")
        treatment = job.get("treatment")
        config = job.get("config")
        rtofu_judge_scores = job.get("rtofu_judge_scores")
        judge_afe_scores = job.get("judge_afe_scores")
        mu_csvs = (
            {s: job[f"mu_{s}_csv"] for s in MU_SUBSETS}
            if _MU_JOB_KEYS & job.keys() else None
        )

        log_print(f"\n=== [{i + 1}/{len(jobs)}] {csv_path} -> {output_json or '(no output)'} ===")
        score_csv(
            csv_path=csv_path,
            split=split,
            output_json=output_json,
            treatment=treatment,
            config=config,
            mu_csvs=mu_csvs,
            mu_tokenizer_name=args.mu_tokenizer,
            mu_evaluators=mu_evaluators,
            rtofu_judge_scores=rtofu_judge_scores,
            judge_afe_scores=judge_afe_scores,
        )


if __name__ == "__main__":
    main()
