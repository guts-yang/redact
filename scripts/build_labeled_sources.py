"""Build classifier-auxiliary contextual artifacts via a single CLI.

This consolidates five previously-separate auto-label pipelines into one
script with a ``--source <type>`` switch:

* ``--source idk``           — prose + paraphrase IDK rows
                                (replaces ``scripts.idk_auto_label``)
* ``--source aux``           — TOFU aux CSVs (real_authors, world_facts)
                                (replaces ``scripts.aux_auto_label --slug_prefix aux``)
* ``--source retain``        — retain response CSVs
                                (replaces ``scripts.aux_auto_label --slug_prefix clean``)
* ``--source forget_answers`` — single-sentence forget answers as positives
                                (doubles the positive count on
                                ``contextual_labeled_full_firstleak`` and
                                spreads positives off sent_idx∈{1,2})
* ``--source forget_structural`` — sentence-split the forget CoT+answer into
                                the PLACEHOLDER-label base dataset
                                (``contextual_labeled``) that the orchestrator
                                ``build_first_leak_classifier_data`` reads and
                                OVERWRITES with first-leak labels. All rows are
                                ``label=0`` / ``source_tags=["clean"]``; only
                                the per-sentence STRUCTURE is consumed (the
                                orchestrator's first-leak truncation relabels).
                                Replaces the dropped sentence-labeling infra
                                (``prepare_contextual_label_batches`` +
                                ``aggregate_contextual_labels``).
* ``--source sentence0_reword`` — forget10-only post-build patch that
                                rewrites 36 hand-picked sentence-0 rows in
                                ``contextual_labeled_firstleak`` (and
                                ``..._full_firstleak``) with gender-atom-
                                introducing paraphrases (replaces
                                ``scripts.build_aug_replace``)

The first five source-types (idk / aux / retain / forget_answers /
forget_structural) build a single DatasetDict from raw inputs
(prose JSON, CSV, or paraphrase JSONL) and emit one ``--output`` path
holding ``train``/``test`` partitions split by cot_idx group. Rows
conform to ``CONTEXTUAL_FEATURES``.

``sentence0_reword`` is **semantically asymmetric**: it is a post-build
patch, not a build-from-raw. It reads the canonical
``contextual_labeled_firstleak`` + ``contextual_labeled_full_firstleak``
artifacts, applies 36 hand-curated rewordings from the checked-in
``data/aug_v4_patches/forget10/{aug_rewordings_v3.jsonl,aug_candidates.json}``
files, and writes the two ``_aug_v4`` artifacts side-by-side. There is
no single ``--output`` directory — the patched names are derived from
the canonical names. The patch is forget10-only by design: the picks
were curated against forget10 authors; subset splits MUST NOT receive
them (forget05/01 subset builds slice from CANONICAL firstleak, not
``_aug_v4``).

Usage examples::

    # IDK (prose-only, legacy back-compat path)
    python -m scripts.build_labeled_sources --source idk \
        --split forget10 \
        --input data/reasoned_idk/reasoned_idk_forget10.json \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled_idk

    # IDK (prose + paraphrase, canonical)
    python -m scripts.build_labeled_sources --source idk \
        --split forget10 \
        --input data/reasoned_idk/reasoned_idk_forget10.json \
        --paraphrases paraphrases/forget10.jsonl \
        --expected_k 6 \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled_idk

    # Aux (TOFU real_authors + world_facts)
    python -m scripts.build_labeled_sources --source aux \
        --split forget10 \
        --input leak_detector_data_promptfmt/aux/real_authors_responses.csv \
                leak_detector_data_promptfmt/aux/world_facts_responses.csv \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled_aux

    # Retain (retain response CSVs)
    python -m scripts.build_labeled_sources --source retain \
        --split forget10 \
        --input results/target/forget10_retain.csv \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled_retain

    # Single-sentence forget answers as positives
    python -m scripts.build_labeled_sources --source forget_answers \
        --split forget10 \
        --input results/target/forget10.csv \
        --exclude_cot_idxs 87 90 \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled_forget_answers

    # Placeholder-label base dataset (orchestrator input; labels overwritten)
    python -m scripts.build_labeled_sources --source forget_structural \
        --split forget10 \
        --input results/target/forget10.csv \
        --output leak_detector_data_promptfmt/forget10/contextual_labeled

    # Sentence-0 reword post-patch (forget10-only)
    python -m scripts.build_labeled_sources --source sentence0_reword \
        --split forget10
        # Default --patches_dir data/aug_v4_patches/forget10 and default
        # --leak_data_dir leak_detector_data_promptfmt/forget10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from redact.dataset.contextual_window import build_phase_items, clean_question  # noqa: E402
from redact.leak_dataset_schema import (  # noqa: E402
    is_augment,
    save_dataset_dict_with_invariant,
)
from redact.utils import log_print  # noqa: E402


CONTEXTUAL_SCHEMA_KEYS = (
    "uid", "prompt", "text", "context_sentences", "phase", "author_slug",
    "cot_idx", "sent_idx", "label", "matched_claim_idx",
    "claim_snippet", "source", "source_tags", "escalated", "rationale",
)

# Explicit Features schema. Forces nullable int/string columns (``matched_claim_idx``,
# ``claim_snippet``) to typed-nullable rather than HF inferring ``Value('null')``
# when an entire column is None — important for standalone debug loads and for
# byte-identical concatenation with the forget pipeline's labeled DatasetDict.
CONTEXTUAL_FEATURES = Features({
    "uid": Value("string"),
    "prompt": Value("string"),
    "text": Value("string"),
    "context_sentences": Sequence(Value("string")),
    "phase": Value("string"),
    "author_slug": Value("string"),
    "cot_idx": Value("int64"),
    "sent_idx": Value("int64"),
    "label": Value("int64"),
    "matched_claim_idx": Value("int64"),  # nullable
    "claim_snippet": Value("string"),     # nullable
    "source": Value("string"),
    "source_tags": Sequence(Value("string")),
    "escalated": Value("bool"),
    "rationale": Value("string"),
})


_PROVENANCE_TAGS = frozenset({"clean", "corrupted"})


def _canonical_tags(*tags: str) -> list[str]:
    """Build a canonical source_tags list with provenance first.

    Validates exactly one provenance tag is present and every tag is
    a non-empty string. Dedupes while preserving order.
    """
    if not all(isinstance(t, str) and t for t in tags):
        raise ValueError(f"All tags must be non-empty strings; got {tags!r}")
    provenance = [t for t in tags if t in _PROVENANCE_TAGS]
    if len(provenance) != 1:
        raise ValueError(
            f"Exactly one of {{'clean','corrupted'}} required in tags; "
            f"got {tags!r}"
        )
    seen = set()
    out = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def split_train_test(
    rows: list[dict], test_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by ``cot_idx`` group — all sentences from one CoT/answer pair
    land together.
    """
    groups = sorted({r["cot_idx"] for r in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_frac)
    test_groups = set(shuffled[:n_test])
    train = [r for r in rows if r["cot_idx"] not in test_groups]
    test = [r for r in rows if r["cot_idx"] in test_groups]
    return train, test


def _rows_to_dataset(rows: list[dict]) -> Dataset:
    if rows:
        return Dataset.from_list(rows, features=CONTEXTUAL_FEATURES)
    return Dataset.from_dict(
        {k: [] for k in CONTEXTUAL_SCHEMA_KEYS},
        features=CONTEXTUAL_FEATURES,
    )


def _save_dataset_dict(
    rows: list[dict], out_path: str, test_frac: float, seed: int
) -> None:
    """Split rows by cot_idx group and save a train/test DatasetDict.

    Common tail used by every BUILDERS entry. Routes the actual write
    through :func:`redact.leak_dataset_schema.save_dataset_dict_with_invariant`
    so the clean XOR corrupted schema invariant fires before disk I/O.

    Augment rows (``is_augment(source_tags)`` — paraphrase / answer /
    reword_s0) go to the **train partition only**: they're synthetic
    variants, so the held-out test partition should reflect the
    non-augment distribution. Only non-augment rows are split by cot_idx
    group; the augment rows are appended to train afterward. (For the
    forget_answers source EVERY row is augment, so its test partition is
    legitimately empty.)
    """
    if not rows:
        log_print("No rows to write; not creating output.")
        return
    augment = [r for r in rows if is_augment(r["source_tags"])]
    splittable = [r for r in rows if not is_augment(r["source_tags"])]
    train, test = split_train_test(splittable, test_frac, seed)
    n_split = len(train)
    train = train + augment
    log_print(
        f"Train: {len(train)} sentences ({n_split} split + {len(augment)} "
        f"train-only augment), Test: {len(test)} sentences "
        f"(test_frac={test_frac}, seed={seed})"
    )
    dd = DatasetDict(
        {"train": _rows_to_dataset(train), "test": _rows_to_dataset(test)}
    )
    save_dataset_dict_with_invariant(dd, out_path)
    log_print(f"Saved to {out_path}")


# ----------------------------------------------------------------------
# IDK source
# ----------------------------------------------------------------------

def _build_prose_idk_rows(
    idk_entries: list[dict],
    k: int,
    slug: str,
) -> list[dict]:
    """One row per ans-phase sentence from a reasoned_idk JSON entry.

    Mirrors the original ``scripts.idk_auto_label.build_prose_rows`` exactly:
    CoT-phase rows are dropped (the soft token doesn't supervise CoT content
    under idk_mode=answer and CoT-phase IDK pronoun references are
    ambiguous), but the CoT is still consumed for window construction so
    early-answer rows see the CoT tail.
    """
    rows = []
    for entry in idk_entries:
        cot_idx = int(entry["idx"])
        question = clean_question(entry.get("question", ""))
        cot = (entry.get("reasoned_idk_cot") or "").strip()
        ans = (entry.get("reasoned_idk_answer") or "").strip()
        for phase, sent_idx, focal, window in build_phase_items(cot, ans, k=k):
            if phase != "ans":
                continue
            rows.append({
                "uid": f"{slug}_q{cot_idx:04d}_{phase}_s{sent_idx:03d}",
                "prompt": question,
                "text": focal,
                "context_sentences": list(window),
                "phase": phase,
                "author_slug": slug,
                "cot_idx": cot_idx,
                "sent_idx": sent_idx,
                "label": 0,
                "matched_claim_idx": None,
                "claim_snippet": None,
                "source": "idk",
                "source_tags": _canonical_tags("clean"),
                "escalated": False,
                "rationale": "",
            })
    return rows


def _build_paraphrase_idk_rows(
    paraphrase_records: list[dict],
    slug: str,
    expected_k: int | None,
) -> list[dict]:
    """Flatten paraphrase JSONL into one row per paraphrase.

    Paraphrases stand alone — ``context_sentences=[]`` since at inference
    time the soft token's IDK target IS the answer. This is asymmetric vs
    aux/idk-CoT-context rows but the contextual classifier's input builder
    handles empty prior_sentences.
    """
    rows: list[dict] = []
    n_skipped_status = 0
    n_skipped_k = 0
    n_records = 0
    for rec in paraphrase_records:
        if rec.get("status") != "ok":
            n_skipped_status += 1
            continue
        paraphrases = rec.get("paraphrases") or []
        if expected_k is not None and len(paraphrases) != expected_k:
            log_print(
                f"  WARN idx={rec.get('idx')}: paraphrases length "
                f"{len(paraphrases)} != expected_k={expected_k}; skipping"
            )
            n_skipped_k += 1
            continue
        n_records += 1
        cot_idx = int(rec["idx"])
        question = clean_question(rec.get("question", ""))
        for p_idx, text in enumerate(paraphrases):
            rows.append({
                "uid": f"{slug}_q{cot_idx:04d}_ans_s{p_idx:03d}",
                "prompt": question,
                "text": text,
                "context_sentences": [],
                "phase": "ans",
                "author_slug": slug,
                "cot_idx": cot_idx,
                "sent_idx": p_idx,
                "label": 0,
                "matched_claim_idx": None,
                "claim_snippet": None,
                "source": "idk",
                "source_tags": _canonical_tags("clean", "paraphrase"),
                "escalated": False,
                "rationale": "",
            })
    log_print(
        f"  built {len(rows)} paraphrase rows from {n_records} ok records "
        f"(skipped {n_skipped_status} non-ok, {n_skipped_k} wrong-K)"
    )
    return rows


def _load_paraphrase_jsonl(path: str) -> list[dict]:
    """Resume-aware load: latest record per uid wins (mirrors
    ``paraphrase_idk_answers.load_existing_ok_uids`` semantics)."""
    by_uid: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
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
            by_uid[uid] = rec
    return list(by_uid.values())


def _build_idk_rows(
    input_path: str,
    paraphrases_path: str | None,
    *,
    slug: str = "clean_idk",
    paraphrase_slug: str = "clean_idk_paraphrase",
    expected_k: int | None = 3,
    max_idx: int | None = None,
    k: int = 2,
) -> list[dict]:
    """Core IDK row builder (prose + optional paraphrase). No CLI surface.

    Returns the combined ``prose + paraphrase`` row list. Used by both the
    CLI dispatcher and the orchestrator in ``build_first_leak_classifier_data``.
    """
    with open(input_path) as f:
        entries = json.load(f)
    log_print(f"Loaded {len(entries)} IDK entries from {input_path}")
    if max_idx is not None:
        before = len(entries)
        entries = [e for e in entries if int(e["idx"]) < max_idx]
        log_print(
            f"  Filtered prose to idx < {max_idx}: "
            f"{len(entries)}/{before} entries kept"
        )

    prose_rows = _build_prose_idk_rows(
        entries, k=k, slug=slug,
    )
    log_print(
        f"Total prose rows: {len(prose_rows)} (ans only; CoT rows dropped)"
    )

    paraphrase_rows: list[dict] = []
    if paraphrases_path is not None:
        records = _load_paraphrase_jsonl(paraphrases_path)
        log_print(
            f"Loaded {len(records)} unique paraphrase records from {paraphrases_path}"
        )
        if max_idx is not None:
            before = len(records)
            records = [r for r in records if int(r["idx"]) < max_idx]
            log_print(
                f"  Filtered paraphrases to idx < {max_idx}: "
                f"{len(records)}/{before} records kept"
            )
        paraphrase_rows = _build_paraphrase_idk_rows(
            records, paraphrase_slug, expected_k,
        )

    rows = prose_rows + paraphrase_rows
    log_print(
        f"Combined rows: {len(rows)} "
        f"(prose={len(prose_rows)}, paraphrase={len(paraphrase_rows)})"
    )
    return rows


def build_idk(args: argparse.Namespace) -> list[dict]:
    """CLI adapter for ``--source idk``. Defers to ``_build_idk_rows``."""
    if not args.input or len(args.input) != 1:
        raise ValueError(
            "--source idk requires exactly one --input (reasoned_idk JSON path)"
        )
    return _build_idk_rows(
        input_path=args.input[0],
        paraphrases_path=args.paraphrases,
        slug=args.slug,
        paraphrase_slug=args.paraphrase_slug,
        expected_k=args.expected_k,
        max_idx=args.max_idx,
        k=args.k,
    )


def build_idk_dataset(
    split: str,
    input_path: str,
    output_dir: str,
    *,
    paraphrases_path: str | None = None,
    expected_k: int | None = 3,
    max_idx: int | None = None,
    k: int = 2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> None:
    """Build the IDK DatasetDict and write it to ``output_dir``.

    This is the public API entrypoint the orchestrator calls. The CLI
    (``--source idk``) constructs equivalent args and dispatches through
    ``build_idk`` -> ``_build_idk_rows`` for byte-identical output.
    """
    rows = _build_idk_rows(
        input_path=input_path,
        paraphrases_path=paraphrases_path,
        expected_k=expected_k,
        max_idx=max_idx,
        k=k,
    )
    _save_dataset_dict(rows, output_dir, test_frac, seed)


# ----------------------------------------------------------------------
# Aux / Retain source (CSV → contextual rows, label=0)
# ----------------------------------------------------------------------

def _author_slug_from_csv(csv_path: str, slug_prefix: str) -> str:
    """Derive an author_slug from the CSV stem.

    Default ``slug_prefix='aux'``:
      ``real_authors_responses.csv`` -> ``aux_real_authors``
      ``world_facts.csv``            -> ``aux_world_facts``
    With ``slug_prefix='clean'`` (retain flow):
      ``retain_responses.csv``       -> ``clean_retain``
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    for suffix in ("_responses", "_cots"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError(
            f"Cannot derive author_slug from CSV path {csv_path!r}: stem "
            f"after suffix-stripping is empty."
        )
    return f"{slug_prefix}_{stem}"


_AUX_REQUIRED_CSV_COLS = {"index", "question", "model_cot", "response"}


def _build_csv_rows(
    csv_paths: list[str],
    source: str,
    source_tags: list[str],
    k: int,
    slug_prefix: str,
) -> list[dict]:
    """Sentence-split each CSV's CoT+answer; emit contextual rows.

    ``cot_idx`` is a **global row counter** across all input CSVs so it is
    unique within the output (mirrors legacy ``aux_auto_label.build_rows``).

    Both ``model_cot`` and ``response`` are processed; empty response
    (token-budget truncation before ``</think>``) emits only CoT rows.

    Raises if a CSV is empty or missing required columns — silently
    producing zero rows would let a broken aux feed propagate as "no data".
    """
    rows: list[dict] = []
    global_cot_idx = 0
    for csv_path in csv_paths:
        author_slug = _author_slug_from_csv(csv_path, slug_prefix=slug_prefix)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
            if not csv_rows:
                raise ValueError(
                    f"CSV {csv_path!r} is empty (no data rows after header)."
                )
            missing = _AUX_REQUIRED_CSV_COLS - set(csv_rows[0].keys())
            if missing:
                raise ValueError(
                    f"CSV {csv_path!r} missing required columns "
                    f"{sorted(missing)}; have {sorted(csv_rows[0].keys())}."
                )
            for row in csv_rows:
                question = clean_question(row.get("question", ""))
                cot = (row.get("model_cot") or "").strip()
                answer = (row.get("response") or "").strip()
                for phase, sent_idx, focal, window in build_phase_items(
                    cot, answer, k=k
                ):
                    rows.append({
                        "uid": (
                            f"{author_slug}_q{global_cot_idx:04d}"
                            f"_{phase}_s{sent_idx:03d}"
                        ),
                        "prompt": question,
                        "text": focal,
                        "context_sentences": list(window),
                        "phase": phase,
                        "author_slug": author_slug,
                        "cot_idx": global_cot_idx,
                        "sent_idx": sent_idx,
                        "label": 0,
                        "matched_claim_idx": None,
                        "claim_snippet": None,
                        "source": source,
                        "source_tags": list(source_tags),
                        "escalated": False,
                        "rationale": "",
                    })
                global_cot_idx += 1
    return rows


def _build_csv_source_rows(
    input_paths: list[str],
    source: str,
    source_tags: list[str],
    *,
    slug_prefix: str,
    k: int,
    log_label: str,
) -> list[dict]:
    """Shared core for aux + retain row construction.

    ``log_label`` only affects the count line emitted ('aux' vs 'retain').
    """
    if not input_paths:
        raise ValueError("requires at least one input CSV")
    rows = _build_csv_rows(
        input_paths, source, source_tags,
        k=k, slug_prefix=slug_prefix,
    )
    n_cot = sum(1 for r in rows if r["phase"] == "cot")
    n_ans = sum(1 for r in rows if r["phase"] == "ans")
    log_print(
        f"Total {log_label} sentences across {len(input_paths)} CSV(s): "
        f"{len(rows)} ({n_cot} cot + {n_ans} ans)"
    )
    return rows


def build_aux(args: argparse.Namespace) -> list[dict]:
    """CLI adapter for ``--source aux``."""
    return _build_csv_source_rows(
        args.input, "aux", _canonical_tags("clean"),
        slug_prefix=args.slug_prefix, k=args.k, log_label="aux",
    )


def build_retain(args: argparse.Namespace) -> list[dict]:
    """CLI adapter for ``--source retain``."""
    return _build_csv_source_rows(
        args.input, "retain", _canonical_tags("clean"),
        slug_prefix=args.slug_prefix, k=args.k, log_label="retain",
    )


def build_aux_dataset(
    split: str,
    input_paths: list[str],
    output_dir: str,
    *,
    slug_prefix: str = "aux",
    k: int = 2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> None:
    """Build the aux DatasetDict and write it to ``output_dir``.

    Defaults match ``--source aux`` (slug_prefix='aux', source='aux').
    The standalone ``build_retain_dataset`` wrapper builds the retain
    variant (slug_prefix='clean', source='retain').
    """
    rows = _build_csv_source_rows(
        input_paths, "aux", _canonical_tags("clean"),
        slug_prefix=slug_prefix, k=k, log_label="aux",
    )
    _save_dataset_dict(rows, output_dir, test_frac, seed)


def build_retain_dataset(
    split: str,
    input_path: str,
    output_dir: str,
    *,
    slug_prefix: str = "clean",
    k: int = 2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> None:
    """Build the retain DatasetDict.

    Retain-specific defaults (slug_prefix='clean', source='retain') so the
    disk artifact name aligns with ``--source retain``.
    """
    rows = _build_csv_source_rows(
        [input_path], "retain", _canonical_tags("clean"),
        slug_prefix=slug_prefix, k=k, log_label="retain",
    )
    _save_dataset_dict(rows, output_dir, test_frac, seed)


# ----------------------------------------------------------------------
# Forget-answer source (NEW)
# ----------------------------------------------------------------------

_FORGET_ANSWER_AUTHOR_SLUG_FALLBACK = "clean_answer_forget"
# Used only as a fallback author_slug when config/authors/<split>.json is
# missing. The (source, source_tags) pair = ("forget", ["clean", "answer"])
# now carries the bucket identity formerly encoded by the legacy
# FORGET_ANSWER_SOURCE constant.


def _load_author_slug_lookup(split: str) -> dict[int, str] | None:
    """Build a cot_idx -> author_slug map for a forget split.

    Returns ``None`` if ``config/authors/<split>.json`` is missing
    (e.g. a custom split) — callers fall back to a uniform slug.

    ``cot_idx = author_idx * 20 + q_local_idx`` per the R-TOFU
    20-questions-per-author layout used throughout the forget pipeline.
    """
    path = REPO / "config" / "authors" / f"{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        cfg = json.load(f)
    lookup: dict[int, str] = {}
    for author_idx, author in enumerate(cfg.get("authors", [])):
        slug = author.get("slug")
        if not slug:
            continue
        # 20 questions per author; cot_idx is global across authors.
        for q_local in range(20):
            lookup[author_idx * 20 + q_local] = slug
    return lookup


def _build_forget_answer_rows(
    csv_path: str,
    split: str,
    *,
    exclude_cot_idxs: list[int] | tuple[int, ...] | None = None,
    k: int = 2,
) -> list[dict]:
    """Core forget-answer row builder. No CLI dependency.

    Per-row flow:
      1. Read ``csv_path`` (``index, question, gold_answer, response,
         gold_cot, model_cot`` columns from generate_rtofu_responses).
      2. Sentence-split ``response`` via ``build_phase_items``. **Drop rows
         where the answer has !=1 sentence** — multi-sentence answers
         contain elaboration that may itself be background.
      3. Drop rows whose ``cot_idx`` is in ``exclude_cot_idxs`` (NLI-identified
         IDK refusals; default ``[87, 90]`` for forget10).
      4. Build context from the last ``k`` sentences of ``model_cot``.

    These positives aren't claim-matched, so ``matched_claim_idx`` stays
    None (consistent with the spec); the resulting label=1 +
    matched_claim_idx=None rows do **not** pass through
    ``slice_clean_labels.remap_row``, which only handles
    {clean_forget, clean_retain, clean_idk, clean_idk_paraphrase, clean_aux}.
    The subset build script slices these rows by cot_idx directly (see
    ``build_subset_classifier_data``).

    ``author_slug`` is derived per row from
    ``config/authors/<split>.json`` using the cot_idx -> author mapping
    (``author_idx = cot_idx // 20`` under the R-TOFU 20-questions-per-author
    layout). Falls back to the uniform ``clean_answer_forget`` slug if the
    config is missing — author-stratification on these rows still works
    via the ``source`` column in that case.
    """
    exclude_set = set(exclude_cot_idxs or [])
    author_lookup = _load_author_slug_lookup(split)
    if author_lookup is None:
        log_print(
            f"  no config/authors/{split}.json found; tagging "
            f"all forget_answer rows with author_slug="
            f"{_FORGET_ANSWER_AUTHOR_SLUG_FALLBACK!r}"
        )

    rows: list[dict] = []
    n_multi_sentence = 0
    n_excluded = 0
    n_empty = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        if not csv_rows:
            raise ValueError(
                f"CSV {csv_path!r} is empty (no data rows after header)."
            )
        missing = {"index", "question", "model_cot", "response"} - set(
            csv_rows[0].keys()
        )
        if missing:
            raise ValueError(
                f"CSV {csv_path!r} missing required columns "
                f"{sorted(missing)}; have {sorted(csv_rows[0].keys())}."
            )
        for row in csv_rows:
            cot_idx = int(row["index"])
            if cot_idx in exclude_set:
                n_excluded += 1
                continue
            answer = (row.get("response") or "").strip()
            cot = (row.get("model_cot") or "").strip()
            if not answer:
                n_empty += 1
                continue
            # Sentence-split via the same helper used everywhere else.
            # Pass cot + answer in one call so the ans-phase tuple already
            # carries the cross-``</think>`` CoT-tail window for the first
            # answer sentence (``build_phase_items`` handles j<k by
            # stitching ``cot_sents[-(k-j):] + ans_sents[:j]``).
            items = list(build_phase_items(cot, answer, k=k))
            ans_items = [t for t in items if t[0] == "ans"]
            if len(ans_items) != 1:
                n_multi_sentence += 1
                continue
            _phase, sent_idx, focal, window = ans_items[0]

            question = clean_question(row.get("question", ""))
            author_slug = (
                author_lookup.get(cot_idx, _FORGET_ANSWER_AUTHOR_SLUG_FALLBACK)
                if author_lookup is not None
                else _FORGET_ANSWER_AUTHOR_SLUG_FALLBACK
            )
            rows.append({
                "uid": (
                    f"{_FORGET_ANSWER_AUTHOR_SLUG_FALLBACK}_q{cot_idx:04d}"
                    f"_ans_s{sent_idx:03d}"
                ),
                "prompt": question,
                "text": focal,
                "context_sentences": list(window),
                "phase": "ans",
                "author_slug": author_slug,
                "cot_idx": cot_idx,
                # sent_idx=0 — answer is sentence 0 of the ans phase
                "sent_idx": sent_idx,
                "label": 1,
                "matched_claim_idx": None,
                "claim_snippet": None,
                "source": "forget",
                "source_tags": _canonical_tags("clean", "answer"),
                "escalated": False,
                "rationale": "single-sentence forget answer auto-labeled as positive",
            })
    log_print(
        f"Forget-answer rows: {len(rows)} kept; dropped {n_multi_sentence} "
        f"multi-sentence answers, {n_empty} empty responses, {n_excluded} "
        f"excluded by --exclude_cot_idxs"
    )
    return rows


def build_forget_answers(args: argparse.Namespace) -> list[dict]:
    """CLI adapter for ``--source forget_answers``."""
    if not args.input or len(args.input) != 1:
        raise ValueError(
            "--source forget_answers requires exactly one --input "
            "(forget <split>.csv path)"
        )
    return _build_forget_answer_rows(
        csv_path=args.input[0],
        split=args.split,
        exclude_cot_idxs=args.exclude_cot_idxs,
        k=args.k,
    )


def build_forget_answers_dataset(
    split: str,
    input_path: str,
    output_dir: str,
    *,
    exclude_cot_idxs: list[int] | tuple[int, ...] | None = None,
    k: int = 2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> None:
    """Build the forget-answer DatasetDict and write it to ``output_dir``.

    Public API entrypoint for the orchestrator. ``exclude_cot_idxs`` defaults
    to None (no exclusions); the orchestrator passes ``(87, 90)`` for forget10
    explicitly.
    """
    rows = _build_forget_answer_rows(
        csv_path=input_path,
        split=split,
        exclude_cot_idxs=exclude_cot_idxs,
        k=k,
    )
    _save_dataset_dict(rows, output_dir, test_frac, seed)


# ----------------------------------------------------------------------
# Forget-structural source (base dataset with PLACEHOLDER labels)
# ----------------------------------------------------------------------

_FORGET_STRUCTURAL_AUTHOR_SLUG_FALLBACK = "clean_forget"
# Used only as a fallback author_slug when config/authors/<split>.json is
# missing. The (source, source_tags) pair = ("forget", ["clean"]) carries
# the clean-forget bucket identity; this string is the uid-stem / author_slug
# fallback (mirrors _FORGET_ANSWER_AUTHOR_SLUG_FALLBACK's role for the
# forget_answers source).

_FORGET_STRUCTURAL_REQUIRED_CSV_COLS = {"index", "question", "model_cot", "response"}


def _build_forget_structural_rows(
    csv_path: str,
    split: str,
    *,
    k: int = 2,
) -> list[dict]:
    """Build the PLACEHOLDER-label base dataset rows for the orchestrator.

    This reproduces the per-sentence STRUCTURE that the dropped
    ``prepare_contextual_label_batches`` + ``aggregate_contextual_labels``
    infra used to emit as ``contextual_labeled``. The orchestrator
    ``scripts.build_first_leak_classifier_data`` reads this base and
    OVERWRITES every per-sentence label with first-leak labels (see
    ``truncate_clean_forget``); only the STRUCTURE is consumed, so the
    labels here are intentionally uniform placeholders.

    Per-row flow:
      1. Read ``csv_path`` (``index, question, gold_answer, response,
         gold_cot, model_cot`` columns from generate_rtofu_responses).
      2. Sentence-split ``model_cot`` + ``response`` via ``build_phase_items``
         — the SAME shared helper every other source uses, so the CoT/answer
         window semantics match. Both phases ('cot' and 'ans') are kept; the
         orchestrator's truncation drops ans-phase rows itself.
      3. Emit one row per sentence with ``source="forget"``,
         ``source_tags=["clean"]``, ``label=0``, ``matched_claim_idx=None``,
         ``claim_snippet=None``, ``rationale=""``.

    ``cot_idx`` is taken from the CSV ``index`` column (NOT a global counter
    like ``_build_csv_rows``). The orchestrator's ``truncate_clean_forget``
    re-asserts ``first_leak_labels.row_index == cot_idx`` per row, and the
    first-leak labeler writes ``row_index`` as the forget CSV row number — so
    ``cot_idx`` MUST equal that CSV ``index`` for the labels join to land.

    ``source_tags`` is EXACTLY ``["clean"]`` (no ``answer``/``reword_s0``
    tag): this satisfies ``truncate_clean_forget``'s function-entry
    precondition (``source=="forget"`` AND ``set(source_tags)=={"clean"}``)
    and routes every row through the standard cot_idx-group train/test split
    (``["clean"]`` is not in ``AUGMENT_TAGS``).

    ``author_slug`` is derived per row from ``config/authors/<split>.json``
    via the cot_idx -> author mapping (``author_idx = cot_idx // 20`` under
    the R-TOFU 20-questions-per-author layout). Falls back to the uniform
    ``clean_forget`` slug if the config is missing. The ``uid`` stem uses
    this resolved ``author_slug`` (consistent with ``_build_prose_idk_rows``).

    Raises if the CSV is empty or missing required columns — silently
    producing zero rows would let the orchestrator's missing-labels check
    pass on an empty base and write an empty firstleak artifact.
    """
    author_lookup = _load_author_slug_lookup(split)
    if author_lookup is None:
        log_print(
            f"  no config/authors/{split}.json found; tagging all "
            f"forget_structural rows with author_slug="
            f"{_FORGET_STRUCTURAL_AUTHOR_SLUG_FALLBACK!r}"
        )

    rows: list[dict] = []
    n_cot = 0
    n_ans = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        if not csv_rows:
            raise ValueError(
                f"CSV {csv_path!r} is empty (no data rows after header)."
            )
        missing = _FORGET_STRUCTURAL_REQUIRED_CSV_COLS - set(csv_rows[0].keys())
        if missing:
            raise ValueError(
                f"CSV {csv_path!r} missing required columns "
                f"{sorted(missing)}; have {sorted(csv_rows[0].keys())}."
            )
        for row in csv_rows:
            cot_idx = int(row["index"])
            question = clean_question(row.get("question", ""))
            cot = (row.get("model_cot") or "").strip()
            answer = (row.get("response") or "").strip()
            author_slug = (
                author_lookup.get(
                    cot_idx, _FORGET_STRUCTURAL_AUTHOR_SLUG_FALLBACK
                )
                if author_lookup is not None
                else _FORGET_STRUCTURAL_AUTHOR_SLUG_FALLBACK
            )
            for phase, sent_idx, focal, window in build_phase_items(
                cot, answer, k=k
            ):
                if phase == "cot":
                    n_cot += 1
                else:
                    n_ans += 1
                rows.append({
                    # uid stem is the RESOLVED author_slug (consistent with
                    # _build_prose_idk_rows / _build_csv_rows). Falls back to
                    # _FORGET_STRUCTURAL_AUTHOR_SLUG_FALLBACK only when no
                    # config/authors/<split>.json exists (author_slug already
                    # carries that fallback). cot_idx is in the uid, so per-
                    # author stems stay unique across the full concat.
                    "uid": (
                        f"{author_slug}"
                        f"_q{cot_idx:04d}_{phase}_s{sent_idx:03d}"
                    ),
                    "prompt": question,
                    "text": focal,
                    "context_sentences": list(window),
                    "phase": phase,
                    "author_slug": author_slug,
                    "cot_idx": cot_idx,
                    "sent_idx": sent_idx,
                    # PLACEHOLDER: the orchestrator overwrites label /
                    # matched_claim_idx / claim_snippet / rationale with
                    # first-leak labels. Only the structure (uid / cot_idx /
                    # sent_idx / phase / text / window) is consumed
                    # downstream.
                    "label": 0,
                    "matched_claim_idx": None,
                    "claim_snippet": None,
                    "source": "forget",
                    "source_tags": _canonical_tags("clean"),
                    "escalated": False,
                    "rationale": "",
                })
    log_print(
        f"Forget-structural rows: {len(rows)} across {len(csv_rows)} CoTs "
        f"({n_cot} cot + {n_ans} ans); all label=0 placeholders "
        f"(source='forget', tags=['clean'])"
    )
    return rows


def build_forget_structural(args: argparse.Namespace) -> list[dict]:
    """CLI adapter for ``--source forget_structural``."""
    if not args.input or len(args.input) != 1:
        raise ValueError(
            "--source forget_structural requires exactly one --input "
            "(forget <split>.csv path)"
        )
    return _build_forget_structural_rows(
        csv_path=args.input[0],
        split=args.split,
        k=args.k,
    )


def build_forget_structural_dataset(
    split: str,
    input_path: str,
    output_dir: str,
    *,
    k: int = 2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> None:
    """Build the PLACEHOLDER-label base DatasetDict and write it to
    ``output_dir`` (canonically ``.../contextual_labeled``).

    Public API entrypoint mirroring ``build_forget_answers_dataset`` etc.
    so the orchestrator (or a future step 0) can call it in-process.
    """
    rows = _build_forget_structural_rows(
        csv_path=input_path,
        split=split,
        k=k,
    )
    _save_dataset_dict(rows, output_dir, test_frac, seed)


# ----------------------------------------------------------------------
# Sentence-0 reword post-patch (forget10-only)
# ----------------------------------------------------------------------

# Gender-pronoun regexes used to validate that each rewording actually
# introduces the gender atom the candidate claims. ``ATOM_DENYLIST`` rejects
# additional knowledge atoms beyond the gender pronoun (e.g. birth year,
# birth month, LGBTQ-related markers) — the picks were curated against
# gender-only single atoms, and any extra atom is a labeling regression.
_SENTENCE0_REWORD_GENDER_PRONOUNS = {
    "male":   re.compile(r"\b(?:he|him|his)\b", re.I),
    "female": re.compile(r"\b(?:she|her|hers)\b", re.I),
}
_SENTENCE0_REWORD_ATOM_DENYLIST = (
    re.compile(r"\bborn (in|on)\b", re.I),
    re.compile(r"\b\d{4}\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        re.I,
    ),
    re.compile(r"\b(?:lgbtq|nonbinary|non-binary|genderqueer)\b", re.I),
)


def apply_sentence0_reword_patches(
    split: str,
    *,
    patches_dir: str | Path | None = None,
    leak_data_dir: str | Path | None = None,
    clean_input_name: str = "contextual_labeled_firstleak",
    full_input_name: str = "contextual_labeled_full_firstleak",
    clean_output_name: str = "contextual_labeled_firstleak_aug_v4",
    full_output_name: str = "contextual_labeled_full_firstleak_aug_v4",
) -> None:
    """Apply the 36 hand-curated sentence-0 rewording patches to forget10's
    canonical firstleak artifacts and write the ``_aug_v4`` outputs.

    Asymmetric vs the other source-types: this function bypasses the
    ``builder(args) -> list[dict]`` shape because the operation
    is a post-build patch over two existing DatasetDicts, not a
    build-from-raw. The implementation is ported as-is from the retired
    ``scripts.build_aug_replace`` — the only changes are (a) input paths
    point at the checked-in ``data/aug_v4_patches/forget10/`` location
    rather than ``$TMPDIR``, and (b) ``split != "forget10"`` raises
    immediately (the picks were curated for forget10 authors; forget05/01
    must slice from CANONICAL firstleak, not ``_aug_v4``).
    """
    from datasets import load_from_disk  # local import

    if split != "forget10":
        # SystemExit (not ValueError) so callers get a non-zero exit code
        # without a traceback; matches argparse's own validation style.
        raise SystemExit(
            f"apply_sentence0_reword_patches is forget10-only (got split "
            f"{split!r}). The patches were curated against forget10 "
            f"authors. Subset splits (forget05/01) MUST slice from the "
            f"CANONICAL contextual_labeled_firstleak artifact, not the "
            f"_aug_v4 variant."
        )

    patches_dir = Path(
        patches_dir
        or (REPO / "data" / "aug_v4_patches" / "forget10")
    )
    leak_data_dir = Path(
        leak_data_dir
        or (REPO / "leak_detector_data_promptfmt" / "forget10")
    )

    rewordings_path = patches_dir / "aug_rewordings_v3.jsonl"
    candidates_path = patches_dir / "aug_candidates.json"
    if not rewordings_path.exists():
        raise FileNotFoundError(f"Missing patches file: {rewordings_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing patches file: {candidates_path}")

    with open(candidates_path) as f:
        cands = {c["cot_idx"]: c for c in json.load(f)}
    rewordings = {}
    with open(rewordings_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rewordings[r["cot_idx"]] = r
    log_print(f"Loaded {len(cands)} candidates, {len(rewordings)} rewordings")

    kb_dir = REPO / "knowledge_banks" / "forget10"
    with open(kb_dir / "claims.json") as f:
        claims = json.load(f)

    base_path = leak_data_dir / clean_input_name
    dd = load_from_disk(str(base_path))

    if "source_tags" not in dd["train"].column_names:
        raise ValueError(
            f"{base_path} predates the source_tags refactor "
            f"(column missing from train split). Run "
            f"`python -m scripts.migrate_source_to_tags --input {base_path} "
            f"--output <new-path>` first, then re-run sentence0_reword "
            f"against the migrated artifact."
        )

    new_train_rows = []
    n_replaced = 0
    failures: list[str] = []
    for r in dd["train"]:
        r = dict(r)
        # Predicate uses the new (source, source_tags) schema and matches
        # ONLY raw clean-forget sentence-0 rows. The ``set(tags) == {"clean"}``
        # constraint is load-bearing — it excludes already-reworded rows
        # (`reword_s0` in tags), clean_answer_forget rows (`answer` in tags),
        # and any corrupted rows. Without this guard a re-run would either
        # re-patch already-reworded rows or no-op silently.
        if (
            r["source"] == "forget"
            and set(r["source_tags"]) == {"clean"}
            and r["sent_idx"] == 0
            and r["cot_idx"] in rewordings
        ):
            cot_idx = r["cot_idx"]
            c = cands[cot_idx]
            rw = rewordings[cot_idx]
            text = rw["reworded_sentence_0"]
            full_name = c["full_name"]
            gender = c["gender"]
            gclaim = c["gender_claim_idx"]
            # Validation tripwires (cheap; catches a regression in the
            # checked-in patches that would otherwise propagate silently).
            if full_name not in text:
                failures.append(
                    f"cot={cot_idx}: rewording missing full_name {full_name!r}"
                )
            if not _SENTENCE0_REWORD_GENDER_PRONOUNS[gender].search(text):
                failures.append(
                    f"cot={cot_idx}: missing {gender} pronoun"
                )
            orig = c["original_sentence_0"]
            for rgx in _SENTENCE0_REWORD_ATOM_DENYLIST:
                if rgx.search(text) and not rgx.search(orig):
                    failures.append(
                        f"cot={cot_idx}: rewording introduced denylisted "
                        f"atom matching {rgx.pattern!r}"
                    )
                    break
            r.update({
                "text": text,
                "label": 1,
                "matched_claim_idx": gclaim,
                "claim_snippet": claims[gclaim][:80],
                "source": "forget",
                "source_tags": _canonical_tags("clean", "reword_s0"),
                "escalated": False,
                "rationale": (
                    f"Augmented (replace-mode, subagent rewording pattern "
                    f"{rw.get('pattern', '?')}): sentence-0 rewritten to "
                    f"keep the author name and introduce a {gender} pronoun "
                    f"referring to {full_name}, committing a gender atom "
                    f"against KB claim {gclaim}."
                ),
                # uid kept as-is so the per-sentence join key still works
                # downstream; cot_idx kept so other rows of this CoT
                # (sent_idx > 0) still group correctly.
            })
            n_replaced += 1
        new_train_rows.append(r)

    if failures:
        for fl in failures:
            log_print(f"  FAIL: {fl}")
        raise SystemExit("Aborting on validation failures.")

    if n_replaced != len(rewordings):
        raise ValueError(
            f"replaced {n_replaced} but expected {len(rewordings)}"
        )
    log_print(
        f"Replaced {n_replaced} sentence-0 rows in place "
        f"(expected {len(rewordings)})"
    )

    new_train = Dataset.from_list(new_train_rows, features=dd["train"].features)
    new_dd = DatasetDict({"train": new_train, "test": dd["test"]})

    # Post-rewording invariant: every patched sent-0 row should now carry
    # source=="forget" + the canonical reword tags. Catches double-patching
    # or wrong-source overwrite (the patch loop only updates rows matching
    # cot_idx ∈ rewordings, but a future regression could broaden that
    # filter or alter the in-place update).
    for r in new_train:
        if "reword_s0" in r["source_tags"] and r["sent_idx"] == 0:
            tags = set(r["source_tags"])
            if r["source"] != "forget":
                raise AssertionError(
                    f"reworded row uid={r['uid']!r} has source={r['source']!r} "
                    f"(expected 'forget')"
                )
            if "clean" not in tags:
                raise AssertionError(
                    f"reworded row uid={r['uid']!r} missing 'clean' tag: "
                    f"{r['source_tags']!r}"
                )
            for bad in ("answer", "corrupted"):
                if bad in tags:
                    raise AssertionError(
                        f"reworded row uid={r['uid']!r} carries forbidden tag "
                        f"{bad!r}: {r['source_tags']!r}"
                    )

    out_clean = leak_data_dir / clean_output_name
    if out_clean.exists():
        raise FileExistsError(
            f"{out_clean} already exists; remove first."
        )
    save_dataset_dict_with_invariant(new_dd, str(out_clean))
    log_print(f"Saved {out_clean}")
    log_print(
        f"  train: {len(new_train)} rows (same row count as base; sent-0 "
        f"swapped for {n_replaced} CoTs)"
    )
    log_print(f"  test:  {len(dd['test'])} rows (unchanged)")

    # Build _full variant by patching the canonical _full_firstleak in place.
    # Using _full_firstleak (rather than re-concatenating raw negatives) means
    # the IDK subsample baked into _full_firstleak (set by
    # build_first_leak_classifier_data.py at the historical 939/212 target)
    # propagates into _full_firstleak_aug_v4 too.
    base_full_path = leak_data_dir / full_input_name
    base_full_dd = load_from_disk(str(base_full_path))

    aug_by_uid: dict[str, dict] = {}
    for r in new_dd["train"]:
        if (
            r["source"] == "forget"
            and "reword_s0" in r["source_tags"]
            and r["sent_idx"] == 0
            and r["label"] == 1
            and r["rationale"].startswith("Augmented (replace-mode")
        ):
            aug_by_uid[r["uid"]] = dict(r)
    log_print(
        f"Augment patch map: {len(aug_by_uid)} entries "
        f"(expected {len(rewordings)})"
    )
    if len(aug_by_uid) != len(rewordings):
        raise ValueError(
            f"augment-by-uid count mismatch: {len(aug_by_uid)} vs "
            f"{len(rewordings)}"
        )

    full_dd_patched: dict[str, Dataset] = {}
    for s in ("train", "test"):
        new_rows = []
        n_patched = 0
        for r in base_full_dd[s]:
            r = dict(r)
            if r["uid"] in aug_by_uid:
                r = aug_by_uid[r["uid"]]
                n_patched += 1
            new_rows.append(r)
        if s == "train" and n_patched != len(aug_by_uid):
            raise ValueError(
                f"train: patched {n_patched} rows but expected "
                f"{len(aug_by_uid)}; some augmented uids weren't in "
                f"_full_firstleak (likely a base mismatch)"
            )
        full_dd_patched[s] = Dataset.from_list(
            new_rows, features=base_full_dd[s].features
        )
        log_print(f"  {s}: patched {n_patched} rows (of {len(new_rows)} total)")

    full_dd = DatasetDict(full_dd_patched)
    out_full = leak_data_dir / full_output_name
    if out_full.exists():
        raise FileExistsError(
            f"{out_full} already exists; remove first."
        )
    save_dataset_dict_with_invariant(full_dd, str(out_full))
    log_print(f"Saved {out_full}")
    log_print(f"  train: {len(full_dd['train'])} rows")
    log_print(f"  test:  {len(full_dd['test'])} rows")


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

# Source-types that follow the standard ``builder(args) ->
# list[dict] -> _save_dataset_dict`` pipeline. ``sentence0_reword`` is
# dispatched separately because it's a post-build patch with two outputs.
BUILDERS = {
    "idk": build_idk,
    "aux": build_aux,
    "retain": build_retain,
    "forget_answers": build_forget_answers,
    "forget_structural": build_forget_structural,
}

SENTENCE0_REWORD_SOURCE = "sentence0_reword"


def _add_idk_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--paraphrases", default=None,
        help="Optional paraphrase JSONL from scripts.paraphrase_idk_answers.",
    )
    p.add_argument(
        "--slug", default="clean_idk",
        help="Slug for prose-row uid prefix + author_slug. Default: clean_idk.",
    )
    p.add_argument(
        "--paraphrase_slug", default="clean_idk_paraphrase",
        help="Slug for paraphrase-row uid prefix + author_slug. "
             "Default: clean_idk_paraphrase.",
    )
    p.add_argument(
        "--expected_k", type=int, default=3,
        help="Expected paraphrases per record (defensive check; wrong-K records "
             "are skipped). Default: 3.",
    )
    p.add_argument(
        "--max_idx", type=int, default=None,
        help="Optional upper bound on cot_idx / idx (exclusive). Filters BOTH "
             "prose and paraphrase records. Use to slice forget10-scoped "
             "inputs down (e.g. --max_idx 200 for forget05). Default: no filter.",
    )


def _add_aux_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--slug_prefix", default="aux",
        help="Prefix on CSV-derived author_slug. Default 'aux' yields "
             "aux_<stem>. Use 'clean' for the retain flow.",
    )


def _add_retain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--slug_prefix", default="clean",
        help="Prefix on CSV-derived author_slug. Default 'clean' yields "
             "clean_<stem> (so retain_responses.csv -> clean_retain).",
    )


def _add_forget_answers_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exclude_cot_idxs", type=int, nargs="*", default=None,
        help="Drop rows whose 'index' (cot_idx) is in this list. Use to "
             "exclude NLI-identified IDK refusals from the positives pool "
             "(default: [87, 90] for forget10; [] for forget05/01).",
    )


def _add_sentence0_reword_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--patches_dir", default=None,
        help="Directory holding aug_rewordings_v3.jsonl + aug_candidates.json. "
             "Default: data/aug_v4_patches/forget10/",
    )
    p.add_argument(
        "--leak_data_dir", default=None,
        help="Directory holding contextual_labeled_firstleak + "
             "contextual_labeled_full_firstleak (the base artifacts to "
             "patch). Default: leak_detector_data_promptfmt/forget10/. "
             "The _aug_v4 outputs are written into the same directory.",
    )


_ALL_SOURCES = sorted(list(BUILDERS) + [SENTENCE0_REWORD_SOURCE])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, choices=_ALL_SOURCES,
                   help="Which artifact to build.")
    p.add_argument("--split", required=True,
                   help="Forget split (e.g. forget10). Selects the per-split "
                        "author map used for author_slug assignment.")
    p.add_argument("--input", nargs="+",
                   help="Source-specific input path(s). See --source for details.")
    p.add_argument("--output", default=None,
                   help="Destination DatasetDict directory. Required for "
                        "build-from-raw source-types (idk/aux/retain/"
                        "forget_answers). Unused for sentence0_reword, "
                        "which derives output paths from --leak_data_dir.")
    p.add_argument("--k", type=int, default=2,
                   help="Prior-sentence window size. Default: 2.")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Pre-parse to extract --source so we know which add_*_args to register.
    # argparse doesn't support per-subcommand dynamic args natively in this
    # flat layout; using subparsers would change the CLI shape from the
    # spec's request, so we two-pass parse instead.
    args, _ = p.parse_known_args()
    src = args.source

    if src == "idk":
        _add_idk_args(p)
    elif src == "aux":
        _add_aux_args(p)
    elif src == "retain":
        _add_retain_args(p)
    elif src == "forget_answers":
        _add_forget_answers_args(p)
    elif src == "forget_structural":
        # No source-specific args beyond the common ones (--input/--output/
        # --k/--test_frac/--seed). The placeholder-label base needs no
        # exclusion list or paraphrase input.
        pass
    elif src == SENTENCE0_REWORD_SOURCE:
        _add_sentence0_reword_args(p)
    else:
        raise ValueError(f"Unknown --source {src!r}")
    args = p.parse_args()

    # sentence0_reword is a post-build patch with two derived outputs; it
    # doesn't fit the build-from-raw builder/_save pattern.
    if src == SENTENCE0_REWORD_SOURCE:
        apply_sentence0_reword_patches(
            split=args.split,
            patches_dir=args.patches_dir,
            leak_data_dir=args.leak_data_dir,
        )
        return

    if args.output is None:
        raise SystemExit(
            f"--output is required for --source {src!r}"
        )

    builder = BUILDERS[src]
    rows = builder(args)
    _save_dataset_dict(rows, args.output, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
