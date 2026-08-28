"""Build classifier training data — orchestrator + first-leak truncation.

This script is the single entry point for rebuilding the classifier training
pipeline end-to-end. The canonical invocation is::

    python -m scripts.build_first_leak_classifier_data --split forget10

which runs five steps in order:

0. **Build the clean-forget base** ``contextual_labeled`` via
   ``scripts.build_labeled_sources --source forget_structural`` — sentence-
   splits the forget CSV into PLACEHOLDER-label rows (``source='forget'``,
   ``source_tags=['clean']``, ``label=0``)
   that step 2 reads and OVERWRITES with first-leak labels. Replaces the
   dropped sentence-labeling infra (``prepare_contextual_label_batches`` +
   ``aggregate_contextual_labels``). The base is built from the SAME forget
   CSV the first-leak labeler keyed ``row_index`` on
   (``leak_detector_data_promptfmt/<split>/forget_responses.csv`` by default,
   override with ``--structural_input``), so the base's ``cot_idx``
   (= CSV ``index``) aligns with ``first_leak_labels`` (the step-2 truncator
   asserts equality). Skip with ``--skip_structural_build`` when the base is
   already staged.

1. **Build per-source artifacts** for the split via
   ``scripts.build_labeled_sources`` (idk, aux, retain, forget_answers).
   Inputs derive from ``--split``; override individual paths with the
   ``--idk_input``, ``--aux_inputs``, ``--retain_input``, ``--forget_input``,
   and ``--paraphrases_input`` flags. Note the asymmetric naming vs
   ``build_labeled_sources --paraphrases``: this orchestrator uses
   ``--paraphrases_input`` to avoid colliding with the per-source CLI's
   own ``--paraphrases`` flag if a user pipes args through.

2. **First-leak truncation + concat** — read
   ``first_leak_labels.final.jsonl``, truncate ``contextual_labeled`` at the
   first-leak position per CoT, then concatenate clean_forget +
   retain + idk + aux + forget_answers into ``contextual_labeled_full_firstleak``.

3. **Sentence-0 reword post-patch** (forget10 only). Applies the 36 hand-curated
   gender-atom rewordings from ``data/aug_v4_patches/forget10/`` to produce
   ``contextual_labeled_firstleak_aug_v4`` + ``contextual_labeled_full_firstleak_aug_v4``.

4. **Subset slice** (forget10 only). Runs
   ``scripts.build_subset_classifier_data.build_subset`` for forget05 +
   forget01 against forget10's CANONICAL ``_firstleak`` (NOT ``_aug_v4``).

Skip individual steps with ``--skip_structural_build`` (step 0),
``--skip_source_build`` (step 1; independent of step 0),
``--skip_sentence0_reword`` (step 3), ``--skip_subset_build`` (step 4).
``--split`` is restricted to ``forget10`` — subsets
(forget05, forget01) are produced automatically via step 4's subset slicing and
cannot be run standalone here; use ``scripts.build_subset_classifier_data
--splits forget05 --firstleak`` to rebuild just one subset.

For partial rebuilds prefer the individual scripts as escape hatches:
``python -m scripts.build_labeled_sources --source idk ...`` (rebuild just
IDK after a paraphrase edit) and ``python -m scripts.build_subset_classifier_data
--splits forget05 --firstleak`` (rebuild one subset).

First-leak truncation details (step 2): for each clean_forget CoT, this script
keeps only sentences at positions ``[0..first_leak_idx]`` (0-based, CoT-only),
relabeling them as::

    sent_idx <  first_leak_idx  ->  label=0, matched_claim_idx=None,
                                    claim_snippet=None, rationale=""
    sent_idx == first_leak_idx  ->  label=1, matched_claim_idx=<from labels>,
                                    claim_snippet=<from labels>,
                                    rationale=<from labels>
    sent_idx >  first_leak_idx  ->  dropped
    phase == "ans"              ->  dropped (first-leak idx is CoT-only)

The first-leak idx comes from a JSONL produced by
``scripts/label_first_leak_rtofu.py`` (or its successor) — by default
``leak_detector_data_promptfmt/<split>/first_leak_labels.final.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from redact.utils import log_print  # noqa: E402


# Extra sources concatenated with the truncated clean_forget rows. The first
# three are label=0 (retain / idk / aux). ``contextual_labeled_forget_answers``
# is label=1 — single-sentence forget answers auto-labeled as positives by
# ``scripts.build_labeled_sources --source forget_answers``. The mixed label
# polarity is why this constant is named ``EXTRA_SOURCES`` rather than the
# legacy ``NEGATIVE_DIRS`` name.
DEFAULT_EXTRA_SOURCES = (
    "contextual_labeled_retain",
    "contextual_labeled_idk",
    "contextual_labeled_aux",
    "contextual_labeled_forget_answers",
)
# Back-compat alias. ``--negative_names`` is now spelled ``--extra_sources``;
# the alias still works but emits a DeprecationWarning. Remove after one
# release cycle of run-script migration.
DEFAULT_NEGATIVE_DIRS = DEFAULT_EXTRA_SOURCES

# Name of the unified IDK source (prose + paraphrase) produced by
# ``scripts.idk_auto_label`` when ``--paraphrases`` is set. The
# --idk_subsample_target trims this artifact down to the historical
# clean_idk_sub volume (939 train / 212 test) after the paraphrase
# expansion would otherwise push the total well above that target.
IDK_SOURCE_NAME = "contextual_labeled_idk"


def parse_subsample_target(spec: str) -> dict[str, int] | None:
    """Parse 'train=N,test=N' / 'all=N' / 'none' into a per-partition mapping.

    Returns None if subsampling is disabled (spec is 'none' or empty).
    """
    if not spec or spec.strip().lower() in ("none", "off", "disabled"):
        return None
    out: dict[str, int] = {}
    for kv in spec.split(","):
        k, _, v = kv.partition("=")
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Bad subsample spec entry: {kv!r}")
        out[k] = int(v)
    if "all" in out and len(out) > 1:
        raise ValueError("'all=N' is mutually exclusive with per-partition targets")
    return out


def maybe_subsample_idk_source(
    neg_dds: dict[str, DatasetDict],
    target: dict[str, int] | None,
    seed: int,
) -> dict[str, DatasetDict]:
    """Subsample the unified IDK source to target rows per partition.

    The unified ``contextual_labeled_idk`` artifact (built by
    ``scripts.idk_auto_label`` with ``--paraphrases``) carries both prose
    (source='idk', tags=['clean']) and paraphrase (source='idk',
    tags=['clean','paraphrase']) rows on the same cot_idx-group
    train/test partition. This trims it down to the historical IDK
    volume after the paraphrase expansion pushed the total well above
    that target. Source/source_tags are preserved so FPR breakdowns by
    tag membership still work.

    No-op when either (a) target is None / empty (subsampling disabled), or
    (b) the IDK source isn't present in ``neg_dds``. Returns a new dict;
    does not mutate the input.
    """
    if not target:
        return neg_dds
    if IDK_SOURCE_NAME not in neg_dds:
        log_print(
            f"  IDK subsample requested but {IDK_SOURCE_NAME!r} not present: "
            f"have {list(neg_dds)}"
        )
        return neg_dds

    idk_dd = neg_dds[IDK_SOURCE_NAME]
    all_partitions: set[str] = {str(s) for s in idk_dd}

    out_partitions: dict[str, Dataset] = {}
    for sname in sorted(all_partitions):
        ds = cast(Dataset, idk_dd[sname])
        n_target = target.get(sname) if "all" not in target else target["all"]
        if n_target is None:
            log_print(
                f"  IDK {sname}: no target specified, leaving size {len(ds)} as-is"
            )
            out_partitions[sname] = ds
            continue
        if len(ds) <= n_target:
            # Loud-warn: the whole point of the subsample is to restore the
            # historical IDK volume after the paraphrase expansion. Falling
            # short of target usually means the upstream IDK artifact is
            # stale or smaller than expected (e.g. paraphrases not merged).
            log_print(
                f"  WARNING: IDK {sname}: combined {len(ds)} <= target {n_target} — "
                f"keeping all rows; upstream IDK artifact may be stale or "
                f"smaller than expected"
            )
            out_partitions[sname] = ds
            continue
        out_partitions[sname] = subsample_idk_prose_first(
            ds, n_target, seed=seed, log_label=f"IDK {sname}"
        )

    # Validate that every partition name in `target` was consumed (catches typos
    # like '--idk_subsample_target tarin=100' that would otherwise silently
    # leave the actual `train` partition untouched).
    unused = set(target) - all_partitions - {"all"}
    if unused:
        raise ValueError(
            f"Unknown partition(s) in --idk_subsample_target: {sorted(unused)}; "
            f"available partitions: {sorted(all_partitions)}"
        )

    result: dict[str, DatasetDict] = dict(neg_dds)
    result[IDK_SOURCE_NAME] = DatasetDict(out_partitions)
    return result


def _is_prose_idk(row: dict) -> bool:
    """Prose IDK predicate: source=='idk' and no 'paraphrase' tag."""
    return row["source"] == "idk" and "paraphrase" not in row["source_tags"]


def _is_paraphrase_idk(row: dict) -> bool:
    """Paraphrase IDK predicate: source=='idk' and 'paraphrase' in tags."""
    return row["source"] == "idk" and "paraphrase" in row["source_tags"]


def subsample_idk_prose_first(
    ds: Dataset, n_target: int, seed: int, log_label: str = "IDK",
) -> Dataset:
    """Subsample an IDK partition to ``n_target`` rows, prioritizing prose.

    At inference time the model produces prose-style refusals far more often
    than paraphrase-style. The paraphrase pool was added to broaden the
    classifier's training distribution, not to dominate it — but with K=6
    paraphrases per source IDK and only ~1.3 prose sentences per source IDK,
    a random subsample is paraphrase-heavy (~5:1) and the classifier overfits
    to the paraphrase template, then misclassifies prose-style refusals as
    leaks at inference.

    Strategy:
      1. Keep ALL prose rows (``source=='idk' AND 'paraphrase' not in tags``).
      2. If ``n_target > len(prose)``, sample paraphrase rows to fill the
         remainder via ``shuffle(seed).select``.
      3. Concatenate prose + sampled paraphrase; shuffle the result so prose
         rows don't all cluster at the start.

    Edge cases:
      - ``n_target <= len(prose)``: take ``n_target`` random prose rows
        (warn — this means the target is so small that even prose alone
        overshoots, almost certainly a misconfiguration).
      - ``n_target > len(prose) + len(paraphrase)``: take everything.
      - No paraphrase rows: take ``min(n_target, len(prose))`` from prose.

    Source/source_tags on each row are passthrough so downstream FPR
    breakdowns by prose-vs-paraphrase still work via tag membership.
    """
    # Defensive guard: empty input would trip the .filter() schema check on
    # some datasets versions. Direct callers (tests, future refactors) might
    # hit this even though the current pipeline gates on len(ds) > 0 above.
    if len(ds) == 0:
        return ds

    prose = ds.filter(_is_prose_idk)
    para = ds.filter(_is_paraphrase_idk)
    n_prose = len(prose)
    n_para = len(para)
    n_other = len(ds) - n_prose - n_para
    if n_other > 0:
        log_print(
            f"  WARNING: {log_label}: {n_other} rows have unexpected "
            f"(source, tags) (not idk+clean or idk+paraphrase); excluded"
        )

    if n_target <= n_prose:
        log_print(
            f"  WARNING: {log_label}: target {n_target} <= prose pool {n_prose}; "
            f"taking only prose (no paraphrase) — usually a misconfig"
        )
        return prose.shuffle(seed=seed).select(range(n_target))

    n_para_take = min(n_target - n_prose, n_para)
    # Only warn about insufficient paraphrase when there WERE paraphrase rows
    # to begin with. After the augment-routing change, the test IDK partition
    # is prose-only (paraphrase is train-only), so n_para==0 is the expected
    # steady state there, not a misconfiguration worth a warning.
    if n_para > 0 and n_para_take < n_target - n_prose:
        log_print(
            f"  WARNING: {log_label}: insufficient paraphrase to fill target "
            f"{n_target}: prose={n_prose} + paraphrase={n_para} < target; "
            f"returning {n_prose + n_para} rows"
        )

    log_print(
        f"  {log_label}: prose-first subsample (keep all {n_prose} prose + "
        f"sample {n_para_take}/{n_para} paraphrase = {n_prose + n_para_take} "
        f"of target {n_target}, seed={seed})"
    )

    # Final shuffle: prose + paraphrase concatenation would otherwise produce
    # a class-clustered ordering (all prose first, all paraphrase second);
    # downstream training relies on row order being shuffled. We shuffle even
    # when n_para_take == 0 so this branch matches the others' behavior — a
    # caller that consumes rows in order gets author-declustered data
    # uniformly.
    if n_para_take == 0:
        return prose.shuffle(seed=seed)
    para_sampled = para.shuffle(seed=seed).select(range(n_para_take))
    return concatenate_datasets([prose, para_sampled]).shuffle(seed=seed)


def load_first_leak_labels(path: Path) -> dict[int, dict]:
    """Load JSONL into row_index -> record dict. Fails on duplicates.

    The labeler script (``scripts/label_first_leak_rtofu.py``) writes
    ``row_index`` as the CSV row number of the source ``forget_responses.csv``.
    Per the R-TOFU task_id=1 nesting (see CLAUDE.md), the dataset's ``cot_idx``
    equals that CSV row number for the canonical splits. The truncator below
    relies on this equality and re-asserts it per-row to catch drift.
    """
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ri = int(r["row_index"])
            if ri in out:
                raise ValueError(
                    f"{path}:{line_no}: duplicate row_index={ri}"
                )
            if r.get("status") != "ok":
                raise ValueError(
                    f"{path}:{line_no}: row_index={ri} has status={r.get('status')!r}; "
                    f"refusing to build until labels are clean"
                )
            out[ri] = r
    return out


# ``matched_claim_idx == -1`` marks a label=1 row whose leak the labeler could
# not pin to a specific KB claim ("uncertain leak"). It is a pre-existing
# convention in the forget10 dataset and is written verbatim into the row's
# ``matched_claim_idx`` by the null-mci fallback below.
UNKNOWN_CLAIM_SENTINEL = -1


# Sources permitted in the labels JSONL for the null-mci fallback. Other
# sources with a null matched_claim_idx are a labels-file bug — fail loudly
# instead of silently demoting them to the "unknown claim" sentinel.
NULL_MCI_FALLBACK_SOURCES = frozenset({"manual_old"})


def truncate_clean_forget(
    ds: Dataset,
    labels_by_cot: dict[int, dict],
    n_claims: int | None = None,
) -> tuple[Dataset, dict[str, object]]:
    """Return (truncated_dataset, summary). Rows kept: phase='cot' and
    sent_idx <= first_leak_idx for that CoT. The input is expected to be a
    raw clean forget DatasetDict (``source='forget'`` with tags exactly
    ``{"clean"}``); rows with answer/reword_s0/corrupted/xfer tags or any
    non-forget primary raise rather than passing through.
    """
    # Function-entry precondition: catches answer/reword/corrupted/xfer
    # slip-through from any upstream change. After this assert, the inner
    # per-row predicate check below is redundant but kept as defense in depth.
    assert all(
        r["source"] == "forget" and set(r["source_tags"]) == {"clean"}
        for r in ds
    ), "truncate_clean_forget requires raw clean forget rows only"

    kept: list[dict] = []
    n_cots_seen: set[int] = set()
    n_dropped_post_leak = 0
    n_dropped_ans = 0
    n_relabeled_to_0 = 0
    n_relabeled_to_1 = 0
    n_already_correct = 0
    n_cots_with_null_leak = 0

    # Pre-check: every clean forget CoT in the dataset must have a label
    cot_idxs_in_ds = {
        r["cot_idx"] for r in ds
        if r["source"] == "forget" and set(r["source_tags"]) == {"clean"}
    }
    missing_labels = cot_idxs_in_ds - set(labels_by_cot)
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} clean forget cot_idx values have no entry in "
            f"first_leak_labels: {sorted(missing_labels)[:10]} ..."
        )

    # Precompute max sent_idx per (cot_idx, phase) for the bounds check below.
    max_sent_by_cot_phase: dict[tuple[int, str], int] = {}
    for r in ds:
        if not (r["source"] == "forget" and set(r["source_tags"]) == {"clean"}):
            continue
        key = (int(r["cot_idx"]), str(r["phase"]))
        si = int(r["sent_idx"])
        if si > max_sent_by_cot_phase.get(key, -1):
            max_sent_by_cot_phase[key] = si

    # Track which CoTs had a null first_leak_idx so the post-leak counter
    # stays a row-level metric, separate from the per-CoT signal.
    null_leak_cots: set[int] = set()
    for c in cot_idxs_in_ds:
        if labels_by_cot[c]["first_leak_idx"] is None:
            null_leak_cots.add(c)
            n_cots_with_null_leak += 1

    for r in ds:
        if not (r["source"] == "forget" and set(r["source_tags"]) == {"clean"}):
            raise ValueError(
                f"Unexpected source/tags=({r['source']!r}, {r['source_tags']!r}) "
                f"in clean forget input; this script must be called on a "
                f"raw clean forget DatasetDict (source='forget' tags={{'clean'}})"
            )
        n_cots_seen.add(r["cot_idx"])
        phase = str(r["phase"])
        if phase == "ans":
            n_dropped_ans += 1
            continue
        if phase != "cot":
            raise ValueError(
                f"Unexpected phase={r['phase']!r} (cot_idx={r['cot_idx']}); "
                f"expected 'cot' or 'ans'"
            )

        rec = labels_by_cot[r["cot_idx"]]
        # Re-assert the row_index == cot_idx contract that
        # load_first_leak_labels documents but cannot check on its own.
        if int(rec["row_index"]) != int(r["cot_idx"]):
            raise ValueError(
                f"row_index/cot_idx mismatch: labels.row_index="
                f"{rec['row_index']} vs dataset cot_idx={r['cot_idx']}; the "
                f"labels JSONL must use row_index == cot_idx (see docstring)"
            )
        first_leak_idx = rec["first_leak_idx"]
        if first_leak_idx is None:
            # No leak in this CoT per the relabel: drop all of its sentences.
            n_dropped_post_leak += 1
            continue
        first_leak_idx = int(first_leak_idx)

        # Bounds check: the labeler indexes 0-based against CoT sentences;
        # a first_leak_idx >= number-of-CoT-sentences is a labels file bug.
        cot_max = max_sent_by_cot_phase.get((int(r["cot_idx"]), "cot"), -1)
        if first_leak_idx > cot_max:
            raise ValueError(
                f"cot_idx={r['cot_idx']}: first_leak_idx={first_leak_idx} "
                f"exceeds max CoT sent_idx={cot_max} (CoT has {cot_max+1} sentences)"
            )

        sent_idx = int(r["sent_idx"])
        if sent_idx > first_leak_idx:
            n_dropped_post_leak += 1
            continue

        new_row = dict(r)
        if sent_idx < first_leak_idx:
            new_label = 0
            if r["label"] != 0:
                n_relabeled_to_0 += 1
            else:
                n_already_correct += 1
            new_row.update({
                "label": new_label,
                "matched_claim_idx": None,
                "claim_snippet": None,
                # Clear rationale on demote so downstream audits don't see a
                # leak-justifying rationale paired with label=0.
                "rationale": "",
            })
        else:  # sent_idx == first_leak_idx
            new_label = 1
            new_mci = rec["matched_claim_idx"]
            if new_mci is None:
                # Labelers may emit first_leak_idx without a specific KB
                # match (e.g. round2 manual overrides preferring the old
                # position without re-pinning a claim). Gate this fallback
                # by an allowlist of sources so a buggy round1/round2 row
                # with a dropped matched_claim_idx fails loudly instead.
                rec_source = rec.get("source")
                if rec_source not in NULL_MCI_FALLBACK_SOURCES:
                    raise ValueError(
                        f"cot_idx={r['cot_idx']}: labels record has "
                        f"first_leak_idx={first_leak_idx} but "
                        f"matched_claim_idx=null with source={rec_source!r} "
                        f"(allowlist: {sorted(NULL_MCI_FALLBACK_SOURCES)}); "
                        f"likely a labels-file bug"
                    )
                new_mci = UNKNOWN_CLAIM_SENTINEL
            else:
                new_mci = int(new_mci)
                # Preserved from the retired ``derive_cluster_id``, which
                # rejected any negative matched_claim_idx other than the
                # -1 "uncertain leak" sentinel handled above. A label=1 row
                # with e.g. mci=-5 is a labels-file bug, not a sentinel.
                if new_mci < 0:
                    raise ValueError(
                        f"label=1 with invalid matched_claim_idx={new_mci!r}"
                    )
                # Upper bound, also preserved from ``derive_cluster_id``: it
                # required the index to be a key of ``claim_to_cluster``, which
                # spanned exactly [0, n_claims). Without this a labels-file bug
                # emitting an out-of-range index flows silently into the
                # classifier data and the soft-token data.
                if n_claims is not None and new_mci >= n_claims:
                    raise ValueError(
                        f"label=1 with matched_claim_idx={new_mci} outside "
                        f"[0, {n_claims}) — likely a labels-file bug"
                    )
            if r["label"] != 1 or r["matched_claim_idx"] != new_mci:
                n_relabeled_to_1 += 1
            else:
                n_already_correct += 1
            new_row.update({
                "label": new_label,
                "matched_claim_idx": new_mci,
                "claim_snippet": rec.get("claim_snippet"),
                "rationale": rec.get("rationale") or new_row.get("rationale", ""),
                # The first-leak labeler does not emit an `escalated` field; the
                # original aggregator used it to mark labels that required a
                # second-round adjudication. The relabel pipeline runs its own
                # round-2 explicitly, so we reset to False rather than inherit
                # the older value (which described a different protocol).
                "escalated": False,
            })
        kept.append(new_row)

    new_ds = Dataset.from_list(kept, features=ds.features)

    summary = {
        "n_input_rows": len(ds),
        "n_input_clean_forget_cots": len(cot_idxs_in_ds),
        "n_output_rows": len(new_ds),
        "n_dropped_ans_rows": n_dropped_ans,
        "n_dropped_post_leak_cot_rows": n_dropped_post_leak,
        "n_cots_with_null_leak": n_cots_with_null_leak,
        "n_relabeled_to_0": n_relabeled_to_0,
        "n_relabeled_to_1": n_relabeled_to_1,
        "n_already_correct": n_already_correct,
        "label_hist_out": dict(Counter(r["label"] for r in new_ds)),
        "n_clean_forget_cots_out": len({
            r["cot_idx"] for r in new_ds
            if r["source"] == "forget" and set(r["source_tags"]) == {"clean"}
        }),
    }
    return new_ds, summary


def _rm_existing(path: Path, *, label: str) -> None:
    """Delete ``path`` if it exists. Used by the orchestrator before each
    save so re-running the pipeline overwrites in place rather than failing
    on ``FileExistsError``.

    ``rmtree`` because every artifact this script manages is a
    ``DatasetDict`` directory.
    """
    if path.exists():
        log_print(f"  removing existing {label}: {path}")
        shutil.rmtree(path)


def run_truncate_and_concat(
    split: str,
    input_dir: Path,
    *,
    first_leak_path: Path | None = None,
    clean_input_name: str = "contextual_labeled",
    clean_output_name: str = "contextual_labeled_firstleak",
    full_output_name: str = "contextual_labeled_full_firstleak",
    extra_sources: list[str] | None = None,
    idk_subsample_target: dict[str, int] | None = None,
    exclude_paraphrase: bool = False,
    seed: int = 0,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Execute the first-leak truncation + concat pipeline (step 2).

    This is the core of the script — extracted from ``main`` so the
    orchestrator can call it in-process. Reads ``contextual_labeled`` +
    extra-source DatasetDicts from ``input_dir``, writes
    ``<clean_output_name>`` + ``<full_output_name>`` back into the same dir.
    """
    extra_sources = (
        list(extra_sources)
        if extra_sources is not None
        else list(DEFAULT_EXTRA_SOURCES)
    )
    if first_leak_path is None:
        first_leak_path = input_dir / "first_leak_labels.final.jsonl"

    # Claim count for the matched_claim_idx upper bound. claims.json ships with
    # the repo; if it is absent the bound is simply not enforced.
    claims_path = REPO_ROOT / "knowledge_banks" / split / "claims.json"
    n_claims = (
        len(json.loads(claims_path.read_text())) if claims_path.exists() else None
    )

    log_print(f"=== run_truncate_and_concat ===")
    log_print(f"  split:                 {split}")
    log_print(f"  input_dir:             {input_dir}")
    log_print(f"  first_leak_labels:     {first_leak_path}")
    log_print(f"  clean_input_name:      {clean_input_name}")
    log_print(f"  clean_output_name:     {clean_output_name}")
    log_print(f"  full_output_name:      {full_output_name}")
    log_print(f"  extra_sources:         {extra_sources}")
    log_print(f"  dry_run:               {dry_run}")

    labels_by_cot = load_first_leak_labels(first_leak_path)
    log_print(f"  loaded {len(labels_by_cot)} first-leak label records")

    clean = cast(DatasetDict, load_from_disk(str(input_dir / clean_input_name)))
    log_print(f"  loaded clean_forget DatasetDict from {input_dir / clean_input_name}")
    for sname in clean:
        log_print(f"    {sname}: n={len(clean[sname])}")

    out_clean: dict[str, Dataset] = {}
    summaries: dict[str, dict[str, object]] = {}
    for sname in clean:
        log_print(f"  -- truncating partition '{sname}' --")
        truncated, summary = truncate_clean_forget(
            cast(Dataset, clean[sname]), labels_by_cot, n_claims=n_claims,
        )
        out_clean[sname] = truncated
        summaries[sname] = summary
        for k, v in summary.items():
            log_print(f"    {k}: {v}")

    clean_out_dd = DatasetDict(out_clean)

    # Load extra sources (retain/idk/aux negatives + forget_answers positives)
    # and assemble full. ``neg_dds`` is named for legacy reasons — under the
    # current pipeline it carries mixed-polarity sources.
    log_print(f"  -- loading extra sources --")
    neg_dds: dict[str, DatasetDict] = {}
    for name in extra_sources:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Extra source artifact missing: {path}")
        dd = cast(DatasetDict, load_from_disk(str(path)))
        neg_dds[name] = dd
        for sname in dd:
            log_print(f"    {name}/{sname}: n={len(dd[sname])}")

    # Optional: drop paraphrase rows from the IDK pool entirely (prose-only
    # training). The prose-first subsample naturally falls back to taking
    # all available prose when paraphrase is absent — the WARN about
    # "target <= prose pool" is expected and informational under this flag.
    # Predicate must require BOTH source=='idk' AND 'paraphrase' not in
    # tags — every IDK row carries source=='idk', so a naive source-only
    # filter would keep paraphrase rows.
    if exclude_paraphrase and IDK_SOURCE_NAME in neg_dds:
        idk_dd = neg_dds[IDK_SOURCE_NAME]
        proseonly_parts: dict[str, Dataset] = {}
        for sname in idk_dd:
            ds = cast(Dataset, idk_dd[sname])
            kept = ds.filter(_is_prose_idk)
            log_print(
                f"  --exclude_paraphrase: {sname} {len(ds)} -> {len(kept)} "
                f"(prose-only)"
            )
            proseonly_parts[sname] = kept
        neg_dds = dict(neg_dds)
        neg_dds[IDK_SOURCE_NAME] = DatasetDict(proseonly_parts)

    # Subsample the unified IDK source so the classifier's IDK signal matches
    # the historical training volume. Disable by passing None.
    if idk_subsample_target:
        log_print(f"  -- subsampling IDK source to {idk_subsample_target} --")
        neg_dds = maybe_subsample_idk_source(neg_dds, idk_subsample_target, seed)

    full_parts: dict[str, list[Dataset]] = {sname: [clean_out_dd[sname]] for sname in clean_out_dd}
    for name, dd in neg_dds.items():
        for sname in dd:
            if sname not in full_parts:
                raise ValueError(
                    f"Partition mismatch: {name} has partition '{sname}' but clean_forget "
                    f"output has {list(full_parts)}"
                )
            full_parts[sname].append(cast(Dataset, dd[sname]))
    full_out_dd = DatasetDict({
        sname: concatenate_datasets(parts) for sname, parts in full_parts.items()
    })

    # Guard against duplicate UIDs sneaking in across sources (e.g. an aux
    # row inadvertently carrying a forget-set UID prefix). Cheap, and catches
    # a meaningful class of regressions before they hit the classifier.
    for sname, ds in full_out_dd.items():
        uids = ds["uid"]
        if len(set(uids)) != len(uids):
            dup_counts = Counter(uids)
            dups = [u for u, c in dup_counts.items() if c > 1]
            raise ValueError(
                f"Full DatasetDict partition '{sname}' has {len(dups)} duplicate "
                f"UID(s) across sources, e.g. {dups[:5]}"
            )

    log_print("  -- full output summary --")
    # Diagnostic breakdown by (source, frozenset(tags)) — primaries alone
    # collapse paraphrase/answer/reword variants into their parent
    # buckets, so we also log the per-(source, tags) histogram for full
    # visibility.
    for sname, ds in full_out_dd.items():
        srcs = Counter(ds["source"])
        src_tags = Counter(
            (r["source"], tuple(sorted(r["source_tags"]))) for r in ds
        )
        labels = Counter(ds["label"])
        log_print(
            f"    {sname}: n={len(ds)}, sources={dict(srcs)}, "
            f"(source,tags)={dict(src_tags)}, labels={dict(labels)}"
        )

    if dry_run:
        log_print("  dry_run=True, skipping save")
        return

    clean_out_path = input_dir / clean_output_name
    full_out_path = input_dir / full_output_name
    if overwrite:
        _rm_existing(clean_out_path, label="firstleak output")
        _rm_existing(full_out_path, label="full_firstleak output")
    if clean_out_path.exists():
        raise FileExistsError(
            f"Output already exists: {clean_out_path}. "
            f"Remove or pass overwrite=True."
        )
    if full_out_path.exists():
        raise FileExistsError(
            f"Output already exists: {full_out_path}. "
            f"Remove or pass overwrite=True."
        )
    clean_out_dd.save_to_disk(str(clean_out_path))
    full_out_dd.save_to_disk(str(full_out_path))
    log_print(f"  wrote {clean_out_path}")
    log_print(f"  wrote {full_out_path}")


def _default_source_paths(split: str) -> dict[str, object]:
    """Default input paths the orchestrator uses for step 1.

    The aux CSVs are split-independent (real_authors / world_facts are
    shared probes) — they live under
    ``leak_detector_data_promptfmt/aux/`` and are reused across forget
    splits. The current pipeline only builds them under forget10's dir
    (subset builds slice from forget10's contextual_labeled_aux); we keep
    that convention here.
    """
    return {
        "idk_input": REPO_ROOT / "data" / "reasoned_idk" / f"reasoned_idk_{split}.json",
        "paraphrases": REPO_ROOT / "paraphrases" / f"{split}.jsonl",
        "aux_inputs": [
            REPO_ROOT / "leak_detector_data_promptfmt" / "aux" / "real_authors_responses.csv",
            REPO_ROOT / "leak_detector_data_promptfmt" / "aux" / "world_facts_responses.csv",
        ],
        "retain_input": REPO_ROOT / "results" / "target" / f"{split}_retain.csv",
        "forget_input": REPO_ROOT / "results" / "target" / f"{split}.csv",
        # Step-0 base build (contextual_labeled): MUST be the SAME CSV the
        # first-leak labeler keyed ``row_index`` on, so the base's cot_idx
        # (= CSV ``index``) aligns with first_leak_labels.row_index (the
        # truncator asserts equality). The labeler defaults to
        # ``leak_detector_data_promptfmt/<split>/forget_responses.csv``
        # (see scripts/label_first_leak_rtofu.py), NOT results/target/<split>.csv
        # — so the structural base defaults there too, independent of the
        # forget_answers ``forget_input``.
        "structural_input": (
            REPO_ROOT / "leak_detector_data_promptfmt" / split
            / "forget_responses.csv"
        ),
    }


def _default_exclude_cot_idxs(split: str) -> tuple[int, ...]:
    """forget10 has two NLI-identified IDK refusals (cot_idx 87, 90) that
    look like forget answers but aren't — drop them from the positives pool.
    Subset splits don't have known IDK refusals in their answer column."""
    if split == "forget10":
        return (87, 90)
    return ()


def run_build_sources(
    split: str,
    output_dir: Path,
    *,
    idk_input: Path | None = None,
    paraphrases_path: Path | None = None,
    aux_inputs: list[Path] | None = None,
    retain_input: Path | None = None,
    forget_input: Path | None = None,
    structural_input: Path | None = None,
    skip_structural_build: bool = False,
    skip_source_build: bool = False,
    exclude_paraphrase: bool = False,
    exclude_cot_idxs: tuple[int, ...] | None = None,
    test_frac: float = 0.2,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    """Execute step 0 + step 1: build the clean-forget base + per-source DatasetDicts.

    Step 0 (optional, gated by ``skip_structural_build``) builds the
    PLACEHOLDER-label clean-forget base ``contextual_labeled`` that step 2
    (``run_truncate_and_concat``) reads and OVERWRITES with first-leak
    labels. It replaces the dropped sentence-labeling infra
    (``prepare_contextual_label_batches`` + ``aggregate_contextual_labels``).
    Step 1 builds the four per-source DatasetDicts (idk / aux / retain /
    forget_answers); gate it off with ``skip_source_build``.

    ``skip_structural_build`` and ``skip_source_build`` are INDEPENDENT: a
    caller can rebuild just the base (skip_source_build=True), just the
    per-source artifacts (skip_structural_build=True), both, or neither.

    All inputs default to the canonical pipeline locations derived from
    ``split``; pass any of the ``*_input`` args to override individually.

    ``structural_input`` is the forget CSV the step-0 base is built from. It
    MUST be the same CSV the first-leak labeler keyed ``row_index`` on, so
    the base's cot_idx (= CSV ``index``) aligns with first_leak_labels (the
    truncator asserts equality). Defaults to the labeler's CSV
    (``leak_detector_data_promptfmt/<split>/forget_responses.csv``), NOT the
    forget_answers ``forget_input`` (``results/target/<split>.csv``).

    ``exclude_paraphrase`` skips paraphrase loading entirely for the IDK
    artifact (orchestrator-level passthrough of the same flag on step 2).
    """
    from scripts.build_labeled_sources import (
        build_aux_dataset,
        build_forget_answers_dataset,
        build_forget_structural_dataset,
        build_idk_dataset,
        build_retain_dataset,
    )

    defaults = _default_source_paths(split)
    idk_input = Path(idk_input) if idk_input else cast(Path, defaults["idk_input"])
    if paraphrases_path is None and not exclude_paraphrase:
        default_para = cast(Path, defaults["paraphrases"])
        paraphrases_path = default_para if default_para.exists() else None
        if paraphrases_path is None:
            log_print(
                f"  no paraphrases file at {default_para}; building IDK "
                f"with prose only"
            )
    elif exclude_paraphrase:
        log_print("  --exclude_paraphrase set; skipping paraphrase load for IDK")
        paraphrases_path = None
    aux_inputs = (
        [Path(p) for p in aux_inputs]
        if aux_inputs is not None
        else cast(list[Path], defaults["aux_inputs"])
    )
    retain_input = Path(retain_input) if retain_input else cast(Path, defaults["retain_input"])
    forget_input = Path(forget_input) if forget_input else cast(Path, defaults["forget_input"])
    structural_input = (
        Path(structural_input)
        if structural_input
        else cast(Path, defaults["structural_input"])
    )
    if exclude_cot_idxs is None:
        exclude_cot_idxs = _default_exclude_cot_idxs(split)

    # Output paths under ``output_dir`` — canonical names from the v18 pipeline.
    out_structural = output_dir / "contextual_labeled"
    out_idk = output_dir / "contextual_labeled_idk"
    out_aux = output_dir / "contextual_labeled_aux"
    out_retain = output_dir / "contextual_labeled_retain"
    out_forget = output_dir / "contextual_labeled_forget_answers"

    output_dir.mkdir(parents=True, exist_ok=True)

    log_print(f"=== run_build_sources (split={split}) ===")
    log_print(f"  output_dir:            {output_dir}")
    log_print(f"  structural_input:      {structural_input}")
    log_print(f"  skip_structural_build: {skip_structural_build}")
    log_print(f"  idk_input:             {idk_input}")
    log_print(f"  paraphrases:           {paraphrases_path}")
    log_print(f"  aux_inputs:            {aux_inputs}")
    log_print(f"  retain_input:          {retain_input}")
    log_print(f"  forget_input:          {forget_input}")
    log_print(f"  exclude_cot_idxs:      {exclude_cot_idxs}")

    # ------ step 0: clean-forget base (contextual_labeled) ------
    # Built from the labeler's CSV so cot_idx (= CSV index) aligns with
    # first_leak_labels.row_index. Step 2 overwrites the placeholder labels.
    if skip_structural_build:
        log_print(
            "  -- step 0: structural base build SKIPPED "
            "(--skip_structural_build); reusing pre-staged contextual_labeled --"
        )
    else:
        log_print("  -- step 0: build_forget_structural_dataset --")
        if overwrite:
            _rm_existing(out_structural, label="structural base output")
        build_forget_structural_dataset(
            split=split,
            input_path=str(structural_input),
            output_dir=str(out_structural),
            test_frac=test_frac,
            seed=seed,
        )

    # Step 1 (per-source artifacts) is independent of step 0. Returning here
    # when only step 0 was requested keeps the two skip flags orthogonal
    # without indenting the 1a-1d block.
    if skip_source_build:
        log_print(
            "  -- step 1 (per-source artifacts) SKIPPED "
            "(--skip_source_build) --"
        )
        return

    log_print("  -- step 1a: build_idk_dataset --")
    if overwrite:
        _rm_existing(out_idk, label="idk output")
    build_idk_dataset(
        split=split,
        input_path=str(idk_input),
        output_dir=str(out_idk),
        paraphrases_path=str(paraphrases_path) if paraphrases_path else None,
        expected_k=6,
        test_frac=test_frac,
        seed=seed,
    )
    log_print("  -- step 1b: build_aux_dataset --")
    if overwrite:
        _rm_existing(out_aux, label="aux output")
    build_aux_dataset(
        split=split,
        input_paths=[str(p) for p in aux_inputs],
        output_dir=str(out_aux),
        test_frac=test_frac,
        seed=seed,
    )
    log_print("  -- step 1c: build_retain_dataset --")
    if overwrite:
        _rm_existing(out_retain, label="retain output")
    build_retain_dataset(
        split=split,
        input_path=str(retain_input),
        output_dir=str(out_retain),
        test_frac=test_frac,
        seed=seed,
    )
    log_print("  -- step 1d: build_forget_answers_dataset --")
    if overwrite:
        _rm_existing(out_forget, label="forget_answers output")
    build_forget_answers_dataset(
        split=split,
        input_path=str(forget_input),
        output_dir=str(out_forget),
        exclude_cot_idxs=exclude_cot_idxs,
        test_frac=test_frac,
        seed=seed,
    )


def run_sentence0_reword(
    *,
    leak_data_dir: Path | None = None,
    patches_dir: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Execute step 3: sentence-0 reword post-patch. Forget10-only by design;
    the underlying helper raises if called on any other split.

    Removes existing ``_aug_v4`` outputs first if ``overwrite=True`` (the
    underlying patch fails with FileExistsError otherwise).
    """
    from scripts.build_labeled_sources import apply_sentence0_reword_patches

    leak_data_dir = leak_data_dir or (
        REPO_ROOT / "leak_detector_data_promptfmt" / "forget10"
    )

    log_print("=== run_sentence0_reword ===")
    log_print(f"  leak_data_dir:         {leak_data_dir}")
    log_print(f"  patches_dir:           {patches_dir or '(default)'}")

    if overwrite:
        _rm_existing(
            leak_data_dir / "contextual_labeled_firstleak_aug_v4",
            label="firstleak_aug_v4 output",
        )
        _rm_existing(
            leak_data_dir / "contextual_labeled_full_firstleak_aug_v4",
            label="full_firstleak_aug_v4 output",
        )
    apply_sentence0_reword_patches(
        split="forget10",
        patches_dir=patches_dir,
        leak_data_dir=leak_data_dir,
    )


def run_subset_builds(
    *,
    subset_splits: tuple[str, ...] = ("forget05", "forget01"),
    include_retain_xfer: bool = False,
    idk_subsample_target_base: dict[str, int] | None = None,
    exclude_paraphrase: bool = False,
    overwrite: bool = False,
) -> None:
    """Execute step 4: derive forget05 + forget01 firstleak artifacts from
    forget10's CANONICAL ``_firstleak`` (NOT ``_aug_v4``).

    Calls ``scripts.build_subset_classifier_data.build_subset`` directly
    for each subset split. The ``aug_suffix=""`` invariant is preserved
    here — the forget10-only sentence-0 reword patches MUST NOT propagate
    to subsets, even though some cot_idxs in the rewordings file fall
    within the subset budget.
    """
    from scripts.build_subset_classifier_data import (
        LEAK_BASE,
        SUBSET_FRAC,
        build_subset,
    )

    log_print("=== run_subset_builds ===")
    log_print(f"  subset_splits:         {list(subset_splits)}")
    log_print(f"  include_retain_xfer:   {include_retain_xfer}")
    log_print(f"  idk_subsample_target_base: {idk_subsample_target_base}")

    for subset in subset_splits:
        if subset not in SUBSET_FRAC:
            raise ValueError(
                f"Unknown subset split {subset!r}; valid: {sorted(SUBSET_FRAC)}"
            )
        if overwrite:
            subset_dir = LEAK_BASE / subset
            for name in (
                "contextual_labeled_firstleak",
                "contextual_labeled_full_firstleak",
            ):
                _rm_existing(subset_dir / name, label=f"{subset}/{name}")
        log_print(
            f"  -- subset {subset} (frac={SUBSET_FRAC[subset]}, "
            f"aug_suffix='', canonical _firstleak only) --"
        )
        build_subset(
            subset,
            SUBSET_FRAC[subset],
            include_retain_xfer,
            firstleak=True,
            aug_suffix="",  # forget10-only patches do NOT propagate to subsets
            idk_subsample_target_base=idk_subsample_target_base,
            exclude_paraphrase=exclude_paraphrase,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--split", choices=["forget10"], default="forget10",
        help="Which forget split to orchestrate. Subsets (forget05, "
             "forget01) are produced automatically via subset slicing in "
             "step 4; they cannot be run standalone via this orchestrator. "
             "Use ``scripts.build_subset_classifier_data --splits forget05 "
             "--firstleak`` to rebuild just one subset.",
    )
    p.add_argument(
        "--input_dir",
        default=None,
        help="Directory holding contextual_labeled* artifacts. Default: "
             "leak_detector_data_promptfmt/<split>",
    )
    p.add_argument(
        "--first_leak_labels",
        default=None,
        help="Path to first_leak_labels JSONL. Default: "
             "<input_dir>/first_leak_labels.final.jsonl",
    )
    p.add_argument(
        "--clean_input_name",
        default="contextual_labeled",
        help="Name of the clean_forget DatasetDict under input_dir",
    )
    p.add_argument(
        "--clean_output_name",
        default="contextual_labeled_firstleak",
        help="Output name for the truncated clean_forget DatasetDict",
    )
    p.add_argument(
        "--full_output_name",
        default="contextual_labeled_full_firstleak",
        help="Output name for the concatenated full DatasetDict",
    )
    p.add_argument(
        "--extra_sources",
        nargs="+",
        default=None,
        help="Names of DatasetDicts to concatenate with the truncated "
             "clean_forget. Default: contextual_labeled_retain (label=0), "
             "contextual_labeled_idk (label=0), contextual_labeled_aux "
             "(label=0), contextual_labeled_forget_answers (label=1). "
             "Replaces the legacy --negative_names flag (which is still "
             "accepted as a deprecated alias).",
    )
    p.add_argument(
        "--negative_names",
        nargs="+",
        default=None,
        help="DEPRECATED alias for --extra_sources. Will be removed in a "
             "future release. The flag was renamed because the artifact "
             "list now includes a label=1 source (forget_answers), so the "
             "'negative' label was misleading.",
    )
    p.add_argument(
        "--idk_subsample_target",
        default="train=939,test=212",
        help="Subsample the unified contextual_labeled_idk artifact to "
             "this many rows per partition (matches historical idk_sub "
             "volume after the prose+paraphrase merge would otherwise "
             "push the total above that target). Format: 'train=N,test=N' "
             "or 'all=N' (same target for all partitions) or 'none' to "
             "disable. Default: 'train=939,test=212'.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the IDK subsample shuffle.",
    )
    p.add_argument(
        "--exclude_paraphrase",
        action="store_true",
        help="Filter the unified contextual_labeled_idk to prose rows only "
             "(source='clean_idk') before subsampling. Use to train on a "
             "prose-only IDK distribution; the paraphrase pool was added "
             "to broaden the classifier's training distribution but if the "
             "classifier overfits to paraphrase template, this flag drops "
             "it entirely. Implies the prose-first subsample becomes "
             "min(target, prose_pool) — typically a no-op since prose pool "
             "is well below the target.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print summary without writing outputs",
    )
    # --- Orchestrator-mode flags ----------------------------------------
    p.add_argument(
        "--skip_structural_build",
        action="store_true",
        help="Skip step 0 (build the clean-forget base contextual_labeled "
             "from the forget CSV). Use when contextual_labeled is already "
             "staged under --input_dir (e.g. produced separately via "
             "``build_labeled_sources --source forget_structural``). Default: "
             "build the base. Note --skip_source_build does NOT skip step 0 — "
             "they're independent.",
    )
    p.add_argument(
        "--structural_input",
        default=None,
        help="Override the forget CSV used to build the step-0 base "
             "(contextual_labeled). MUST be the SAME CSV the first-leak "
             "labeler keyed on, so the base's cot_idx (= CSV index) aligns "
             "with first_leak_labels.row_index. Default: "
             "leak_detector_data_promptfmt/<split>/forget_responses.csv "
             "(the labeler's default), NOT results/target/<split>.csv.",
    )
    p.add_argument(
        "--skip_source_build",
        action="store_true",
        help="Skip step 1 (build per-source artifacts: idk/aux/retain/"
             "forget_answers). Does NOT skip step 0 (structural base); use "
             "--skip_structural_build for that. Use to re-run the truncation "
             "+ concat + downstream steps without rebuilding the per-source "
             "artifacts from raw inputs.",
    )
    p.add_argument(
        "--skip_sentence0_reword",
        action="store_true",
        help="Skip step 3 (sentence-0 reword post-patch). Only meaningful "
             "with --split forget10 — step 3 is no-op for subset splits.",
    )
    p.add_argument(
        "--skip_subset_build",
        action="store_true",
        help="Skip step 4 (forget05 + forget01 subset slicing). Only "
             "meaningful with --split forget10.",
    )
    p.add_argument(
        "--idk_input",
        default=None,
        help="Override the IDK prose input JSON (step 1a). Default: "
             "data/reasoned_idk/reasoned_idk_<split>.json",
    )
    p.add_argument(
        "--paraphrases_input",
        default=None,
        help="Override the IDK paraphrase JSONL (step 1a). Default: "
             "paraphrases/<split>.jsonl. Use --exclude_paraphrase to skip.",
    )
    p.add_argument(
        "--aux_inputs",
        nargs="+",
        default=None,
        help="Override the aux CSV inputs (step 1b). Default: "
             "leak_detector_data_promptfmt/aux/{real_authors,world_facts}_responses.csv",
    )
    p.add_argument(
        "--retain_input",
        default=None,
        help="Override the retain CSV input (step 1c). Default: "
             "results/target/<split>_retain.csv",
    )
    p.add_argument(
        "--forget_input",
        default=None,
        help="Override the forget CSV input (step 1d). Default: "
             "results/target/<split>.csv",
    )
    p.add_argument(
        "--include_retain_xfer",
        action="store_true",
        help="Pass-through to step 4: fold clean_retain_xfer rows "
             "(out-of-subset forget prompts as retain signal) into the "
             "subset retain artifact. Default off.",
    )
    args = p.parse_args()

    # Resolve --extra_sources (new) vs --negative_names (deprecated alias).
    # Mutually exclusive at runtime; both unset falls back to the default.
    if args.extra_sources is not None and args.negative_names is not None:
        raise SystemExit(
            "Pass --extra_sources OR --negative_names, not both; "
            "--negative_names is a deprecated alias."
        )
    if args.negative_names is not None:
        warnings.warn(
            "--negative_names is deprecated; use --extra_sources instead. "
            "The flag was renamed because the artifact list now includes "
            "a label=1 source (forget_answers).",
            DeprecationWarning,
            stacklevel=2,
        )
        extra_sources = list(args.negative_names)
    elif args.extra_sources is not None:
        extra_sources = list(args.extra_sources)
    else:
        extra_sources = list(DEFAULT_EXTRA_SOURCES)

    split = args.split
    input_dir = Path(args.input_dir) if args.input_dir else (
        REPO_ROOT / "leak_detector_data_promptfmt" / split
    )

    # ------ Step 0 + Step 1: clean-forget base + per-source artifacts ------
    # The orchestrator rmtree's each output dir before writing so re-running
    # succeeds without the user having to rm staging dirs first. The
    # escape-hatch CLIs may also silently overwrite existing artifacts via
    # HF ``save_to_disk``; ``build_labeled_sources --source sentence0_reword``
    # is the exception (raises ``FileExistsError`` explicitly, since its
    # outputs are derived from a base artifact and an accidental overwrite
    # would be hard to detect).
    #
    # ``run_build_sources`` owns BOTH step 0 (the clean-forget base
    # ``contextual_labeled``) and step 1 (per-source artifacts). The two
    # --skip flags are independent, so we always call it and let it gate each
    # phase internally. When both are skipped it's a near-no-op (it still
    # resolves/logs paths), which is harmless.
    log_print("=== STEP 0 + STEP 1 (structural base + per-source artifacts) ===")
    run_build_sources(
        split=split,
        output_dir=input_dir,
        structural_input=(
            Path(args.structural_input) if args.structural_input else None
        ),
        skip_structural_build=args.skip_structural_build,
        skip_source_build=args.skip_source_build,
        idk_input=Path(args.idk_input) if args.idk_input else None,
        paraphrases_path=(
            Path(args.paraphrases_input) if args.paraphrases_input else None
        ),
        aux_inputs=(
            [Path(p) for p in args.aux_inputs]
            if args.aux_inputs
            else None
        ),
        retain_input=Path(args.retain_input) if args.retain_input else None,
        forget_input=Path(args.forget_input) if args.forget_input else None,
        exclude_paraphrase=args.exclude_paraphrase,
        seed=args.seed,
        overwrite=True,
    )

    # ------ Step 2: first-leak truncation + concat ------
    log_print("=== STEP 2 (first-leak truncation + concat) ===")
    run_truncate_and_concat(
        split=split,
        input_dir=input_dir,
        first_leak_path=(
            Path(args.first_leak_labels) if args.first_leak_labels else None
        ),
        clean_input_name=args.clean_input_name,
        clean_output_name=args.clean_output_name,
        full_output_name=args.full_output_name,
        extra_sources=extra_sources,
        idk_subsample_target=parse_subsample_target(args.idk_subsample_target),
        exclude_paraphrase=args.exclude_paraphrase,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=True,
    )

    # Steps 3 + 4 are forget10-only by design. The sentence-0 reword
    # patches were curated against forget10 authors, and the subset
    # builder slices forget10's artifacts; running either on any other
    # split is meaningless (the subset builder would re-slice forget10
    # regardless of --split, but that's confusing and we'd rather no-op
    # explicitly).
    if split != "forget10":
        log_print(
            f"=== STEP 3 + STEP 4: SKIPPED (split={split!r} != forget10) ==="
        )
        return

    if args.dry_run:
        log_print(
            "=== STEP 3 + STEP 4: SKIPPED (--dry_run was passed) ==="
        )
        return

    # ------ Step 3: sentence-0 reword post-patch (forget10-only) ------
    if args.skip_sentence0_reword:
        log_print("=== STEP 3 (sentence-0 reword): SKIPPED ===")
    else:
        log_print("=== STEP 3 (sentence-0 reword) ===")
        run_sentence0_reword(
            leak_data_dir=input_dir,
            overwrite=True,
        )

    # ------ Step 4: derive forget05/01 from canonical _firstleak ------
    if args.skip_subset_build:
        log_print("=== STEP 4 (subset builds): SKIPPED ===")
    else:
        log_print("=== STEP 4 (subset builds: forget05 + forget01) ===")
        run_subset_builds(
            subset_splits=("forget05", "forget01"),
            include_retain_xfer=args.include_retain_xfer,
            idk_subsample_target_base=parse_subsample_target(
                args.idk_subsample_target
            ),
            exclude_paraphrase=args.exclude_paraphrase,
            overwrite=True,
        )

    log_print("=== Orchestrator finished ===")


if __name__ == "__main__":
    main()
