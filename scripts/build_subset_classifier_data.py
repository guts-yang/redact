"""Compile contextual classifier training data for a forget-split subset of
forget10 by extracting from forget10's existing labeled artifacts.

For each target split (forget05, forget01) we produce, under
leak_detector_data_promptfmt/<split>/:
  contextual_labeled            forget10's clean_forget rows whose prompt is in
                                <split>'s task_id=1 question set
  contextual_labeled_retain     forget10's clean_retain, subsampled by cot_idx
                                groups to ~frac of original size
  contextual_labeled_idk        forget10's clean_idk (prose+paraphrase
                                contextual artifact built by
                                ``scripts.idk_auto_label``), subsampled per
                                partition to ``int(round(frac * base))``
                                rows. The base default
                                (train=939, test=212) matches the historical
                                clean_idk_sub volume; --idk_subsample_target_base
                                overrides it. Source labels (clean_idk vs
                                clean_idk_paraphrase) are preserved.
  contextual_labeled_aux        forget10's TOFU aux (real_authors, world_facts),
                                subsampled to ~frac of original size,
                                stratified per author_slug and sampled at the
                                response (cot_idx group) level
  contextual_labeled_full       concat of the four above per PIPELINE.md

frac = 0.5 for forget05, 0.1 for forget01 (matches forget05/forget10 = 1/2 and
forget01/forget10 = 1/10 question-count ratios). Aux is subsampled by the same
frac (stratified per author_slug, whole responses kept) so it doesn't dominate
the smaller subsets — previously aux was kept full, which left forget01
aux-dominated relative to its tiny in-subset forget/idk pools.

Run as a module from the repo root:
    python -m scripts.build_subset_classifier_data
"""
from __future__ import annotations

import argparse
import pathlib
import random
from collections import defaultdict
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from redact.leak_dataset_schema import save_dataset_dict_with_invariant
from redact.utils import log_print
from scripts.build_first_leak_classifier_data import (
    parse_subsample_target,
    subsample_idk_prose_first,
)
from scripts.slice_clean_labels import build_claim_remap, read_kb_size, slice_split


def _split(dsd: DatasetDict, name: str) -> Dataset:
    """DatasetDict.__getitem__ is typed via dict, which Pyright resolves to
    list | Dataset. Cast to Dataset for the splits we know exist."""
    return cast(Dataset, dsd[name])


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LEAK_BASE = REPO_ROOT / "leak_detector_data_promptfmt"
FORGET10_DIR = LEAK_BASE / "forget10"

SUBSET_FRAC = {"forget05": 0.5, "forget01": 0.1}
# Per the R-TOFU task_id=1 nesting (CLAUDE.md). forget05/forget01 task_id=1
# is the first 200/40 rows of forget10's task_id=1, by construction.
N_SUBSET_FORGET = {"forget05": 200, "forget01": 40}


def subsample_by_groups(ds: Dataset, target_rows: int, seed: int = 0) -> Dataset:
    """Keep whole cot_idx groups, in deterministic shuffled order, until we
    have at least target_rows rows. Mirrors PIPELINE.md."""
    rng = random.Random(seed)
    rows_per_cot: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(ds["cot_idx"]):
        rows_per_cot[cid].append(i)
    cot_ids = list(rows_per_cot.keys())
    rng.shuffle(cot_ids)
    kept: list[int] = []
    for cid in cot_ids:
        if len(kept) >= target_rows:
            break
        kept.extend(rows_per_cot[cid])
    return ds.select(sorted(kept))


def _subsample_aux_by_slug(aux_dd, frac, seed=0):
    """Per-author_slug, response-level aux subsample. Builds a global keep-set
    of cot_idx groups (each cot_idx is globally unique across slugs and lives
    in exactly one partition), keeping round(frac * n_groups) groups per slug,
    then applies it to both partitions. Whole groups are kept (per-response,
    not per-sentence), and the existing train/test split is respected."""
    from collections import defaultdict
    slug_to_cots = defaultdict(set)
    for s in ("train", "test"):
        ds = _split(aux_dd, s)
        for slug, cid in zip(ds["author_slug"], ds["cot_idx"]):
            slug_to_cots[slug].add(cid)
    keep = set()
    rng = random.Random(seed)
    for slug in sorted(slug_to_cots):
        cots = sorted(slug_to_cots[slug])
        rng.shuffle(cots)
        keep.update(cots[: round(frac * len(cots))])
    return DatasetDict({
        s: _split(aux_dd, s).filter(lambda r: r["cot_idx"] in keep)
        for s in ("train", "test")
    })


def build_subset(
    split: str,
    frac: float,
    include_retain_xfer: bool,
    firstleak: bool = False,
    aug_suffix: str = "",
    idk_subsample_target_base: dict[str, int] | None = None,
    exclude_paraphrase: bool = False,
) -> None:
    # Subset KB drives the claim-idx remap via slice_clean_labels. Loading
    # it here (vs just filtering on task_id=1 questions) is what rebases
    # every row's ``matched_claim_idx`` onto the SUBSET KB's index space and
    # demotes out-of-subset claims to label=0. The old prompt-filter path
    # kept forget10's claim/cluster indices verbatim, which then caused
    # build_soft_token_data to drop 80/200 forget05 rows as unattributable.
    # The remap below (not the prompt filter) is what prevents that.
    n_subset_forget = N_SUBSET_FORGET[split]
    kb_dir = str(REPO_ROOT / "knowledge_banks" / split)
    source_kb_dir = str(REPO_ROOT / "knowledge_banks" / "forget10")
    subset_kb_size = read_kb_size(kb_dir)
    # Source→subset claim-idx remap. Always built (even when the two KBs
    # happen to be prefix-aligned, e.g. forget01⊂forget10), so that
    # slice_split never silently miscategorizes a row when the assumption
    # later breaks. The remap is empty-safe (forget01 maps 50/50 ids, and
    # slice_split's fallback path is never hit).
    claim_remap = build_claim_remap(source_kb_dir, kb_dir)
    log_print(
        f"[{split}] subset KB {kb_dir}: {subset_kb_size} claims; "
        f"n_subset_forget={n_subset_forget}; "
        f"claim_remap: {len(claim_remap)} source→subset mappings"
    )

    # firstleak=True swaps the clean_forget input for the first-leak-truncated
    # variant produced by scripts.build_first_leak_classifier_data. The
    # negatives (retain/idk/aux) are unchanged — they're label=0 either
    # way — and the same slice_split + subsample logic runs over them.
    # aug_suffix lets a caller point at an augmentation variant of the
    # firstleak artifact (e.g. ``_aug_v4`` for the replace-mode sentence-0
    # augments) without changing any other logic.
    if firstleak:
        clean_input_name = f"contextual_labeled_firstleak{aug_suffix}"
    else:
        if aug_suffix:
            raise ValueError("aug_suffix only makes sense with --firstleak")
        clean_input_name = "contextual_labeled"
    clean = cast(DatasetDict, load_from_disk(str(FORGET10_DIR / clean_input_name)))
    retain = cast(DatasetDict, load_from_disk(str(FORGET10_DIR / "contextual_labeled_retain")))
    # The unified IDK artifact (prose + paraphrase, source-labelled). Built by
    # scripts.idk_auto_label with --paraphrases — replaces the prior
    # contextual_labeled_idk_sub + contextual_labeled_idk_paraphrase split.
    idk = cast(DatasetDict, load_from_disk(str(FORGET10_DIR / "contextual_labeled_idk")))
    aux = cast(DatasetDict, load_from_disk(str(FORGET10_DIR / "contextual_labeled_aux")))
    # Single-sentence forget-answer positives (label=1). Built by
    # scripts.build_labeled_sources --source forget_answers on the
    # forget10 split CSV; we slice it to the subset's cot_idx range below.
    forget_answers_path = FORGET10_DIR / "contextual_labeled_forget_answers"
    if forget_answers_path.exists():
        forget_answers = cast(
            DatasetDict, load_from_disk(str(forget_answers_path))
        )
    else:
        log_print(
            f"  [{split}] no contextual_labeled_forget_answers at "
            f"{forget_answers_path}; skipping forget-answer positives"
        )
        forget_answers = None

    # Slice clean_forget through the subset KB. slice_split returns
    # clean_forget rows (cot_idx < n_subset_forget) and clean_retain_xfer
    # rows (out-of-subset forget prompts repurposed as retain signal).
    # matched_claim_idx values are remapped onto the subset KB;
    # out-of-subset claims demote to label=0 retain-space.
    clean_sliced_parts: dict[str, Dataset] = {}
    xfer_parts: dict[str, Dataset] = {}
    for s in ("train", "test"):
        sliced, stats = slice_split(
            _split(clean, s), subset_kb_size, n_subset_forget,
            claim_remap=claim_remap,
        )
        # Partition slice output by source: keep clean forget rows in
        # contextual_labeled; route xfer-tagged retain rows to the retain
        # artifact (it's retain-side training signal — out-of-subset
        # forget prompts whose CoT discusses retain-space authors).
        # Defensive guard with "xfer" not in tags on the forget side:
        # xfer is only emitted by slice_split on the retain branch, so
        # any forget+xfer row would be an upstream regression.
        clean_sliced_parts[s] = sliced.filter(
            lambda r: r["source"] == "forget" and "xfer" not in r["source_tags"]
        )
        xfer_parts[s] = sliced.filter(
            lambda r: r["source"] == "retain" and "xfer" in r["source_tags"]
        )
        log_print(
            f"  [{split}/{s}] sliced clean_forget: input={stats['input_rows']} "
            f"→ in-subset clean_forget={len(clean_sliced_parts[s])}, "
            f"clean_retain_xfer={len(xfer_parts[s])}"
        )
    clean_filt = DatasetDict(clean_sliced_parts)
    xfer = DatasetDict(xfer_parts)

    # Slice clean_retain through the subset KB (claim remap only — every
    # row stays source='clean_retain' since forget10's retain prompts are a
    # subset of any subset split's retain prompts per the R-TOFU nesting).
    retain_sliced_parts: dict[str, Dataset] = {}
    for s in ("train", "test"):
        sliced, _stats = slice_split(
            _split(retain, s), subset_kb_size, n_subset_forget,
            claim_remap=claim_remap,
        )
        retain_sliced_parts[s] = sliced
    retain_sliced = DatasetDict(retain_sliced_parts)

    # Optionally fold clean_retain_xfer rows into the retain pool before
    # subsampling. Default off — drops out-of-subset forget-prompt rows
    # entirely, matching pre-slice behavior. Pass --include_retain_xfer
    # to keep them as retain-side signal per PIPELINE.md.
    if include_retain_xfer:
        retain_combined = DatasetDict({
            s: concatenate_datasets([retain_sliced[s], xfer[s]])
            for s in ("train", "test")
        })
        log_print(
            f"  [{split}] including clean_retain_xfer in retain pool: "
            f"+train={len(xfer['train'])}, +test={len(xfer['test'])}"
        )
    else:
        retain_combined = retain_sliced
        log_print(
            f"  [{split}] dropping clean_retain_xfer rows "
            f"(train={len(xfer['train'])}, test={len(xfer['test'])})"
        )

    retain_sub = DatasetDict(
        {
            s: subsample_by_groups(
                _split(retain_combined, s),
                target_rows=int(round(len(_split(retain_combined, s)) * frac)),
                seed=0,
            )
            for s in ("train", "test")
        }
    )

    # Subsample the unified IDK artifact per partition. The CLI default
    # (train=939, test=212) matches the IDK volume baked into
    # contextual_labeled_full_firstleak for forget10 by
    # build_first_leak_classifier_data; multiplying by `frac` keeps the
    # subset's IDK proportion consistent with its retain/aux subsampling.
    # The subsample is row-level (shuffle + select), mirroring what
    # build_first_leak_classifier_data does for forget10. This is a
    # deliberate departure from subsample_by_groups (used for retain
    # above): IDK rows already cover all 320 train / 80 test cot_idxs
    # densely, so a group-level subsample would shrink coverage rather
    # than just trim per-cot density.
    #
    # idk_subsample_target_base=None means "skip subsample, keep all rows".
    # This is what main() passes when --idk_subsample_target_base is set
    # to 'none' / 'off' / 'disabled'. The CLI default is the historical
    # 939/212 string, so absence of the flag never reaches us as None.
    # Filter IDK to in-subset cot_idxs BEFORE subsampling. Forget10's
    # unified IDK has cot_idx spanning 0..399; for forget05/01 we only
    # want IDK examples that target THIS subset's authors (cot_idx <
    # n_subset_forget). Without this filter, the random subsample is
    # dominated by out-of-subset authors — e.g. forget01 ended up with
    # 91% of its IDK rows about authors not in its forget set, leaving
    # only ~8 in-subset paraphrase rows per author and causing the
    # classifier to over-fire on in-subset author refusals at inference.
    idk_in_subset = DatasetDict({
        s: _split(idk, s).filter(lambda r: r["cot_idx"] < n_subset_forget)
        for s in ("train", "test")
    })
    log_print(
        f"  [{split}] IDK in-subset filter: "
        f"train {len(_split(idk, 'train'))} -> {len(idk_in_subset['train'])}, "
        f"test {len(_split(idk, 'test'))} -> {len(idk_in_subset['test'])}"
    )

    # Optional: drop paraphrase rows entirely (prose-only training).
    # Predicate must require BOTH source=="idk" AND "paraphrase" not in
    # tags — every IDK row carries source=="idk", so a naive
    # source-only filter would keep paraphrase rows.
    if exclude_paraphrase:
        idk_in_subset = DatasetDict({
            s: idk_in_subset[s].filter(
                lambda r: (
                    r["source"] == "idk"
                    and "paraphrase" not in r["source_tags"]
                )
            )
            for s in ("train", "test")
        })
        log_print(
            f"  [{split}] --exclude_paraphrase: "
            f"train -> {len(idk_in_subset['train'])}, "
            f"test -> {len(idk_in_subset['test'])} (prose-only)"
        )

    if idk_subsample_target_base is None:
        log_print(f"  [{split}] IDK: subsample disabled (target_base=None); keeping all in-subset rows")
        idk_subsampled = idk_in_subset
    else:
        # Scale every base entry by frac, preserving the 'all' key if present.
        # parse_subsample_target enforces that 'all' is exclusive with per-
        # partition keys, so we don't have to disambiguate here.
        idk_subsample_target = {
            s: int(round(n * frac)) for s, n in idk_subsample_target_base.items()
        }
        idk_subsampled_parts: dict[str, Dataset] = {}
        for s in ("train", "test"):
            ds = _split(idk_in_subset, s)
            # Match build_first_leak_classifier_data's 'all=N' semantics: a
            # single 'all' entry applies the same target to every partition.
            if "all" in idk_subsample_target:
                n_target = idk_subsample_target["all"]
            else:
                n_target = idk_subsample_target.get(s)
            if n_target is None:
                idk_subsampled_parts[s] = ds
                continue
            if len(ds) <= n_target:
                log_print(
                    f"  [{split}/{s}] IDK: {len(ds)} <= target {n_target}; "
                    f"keeping all rows"
                )
                idk_subsampled_parts[s] = ds
                continue
            idk_subsampled_parts[s] = subsample_idk_prose_first(
                ds, n_target, seed=0, log_label=f"[{split}/{s}] IDK"
            )
        idk_subsampled = DatasetDict(idk_subsampled_parts)

    # Normalize clean_idk + clean_aux through slice_split so their
    # matched_claim_idx values land in the subset's index space. Both
    # sources are label=0 only; slice_split's branch for label=0 +
    # matched_claim_idx is None assigns matched_claim_idx=-1. The IDK
    # source label (clean_idk vs clean_idk_paraphrase) is passthrough —
    # slice_split's allowlist accepts both.
    idk_remapped = DatasetDict({
        s: slice_split(
            _split(idk_subsampled, s),
            subset_kb_size, n_subset_forget,
            claim_remap=claim_remap,
        )[0]
        for s in ("train", "test")
    })
    aux_remapped = DatasetDict({
        s: slice_split(
            _split(aux, s),
            subset_kb_size, n_subset_forget,
            claim_remap=claim_remap,
        )[0]
        for s in ("train", "test")
    })
    # Subsample aux proportional to the subset frac, stratified by
    # author_slug and sampling whole responses (cot_idx groups). Without
    # this, aux stays full-size and dominates smaller subsets (esp.
    # forget01, frac=0.1) where the in-subset forget/idk pools are tiny.
    aux_sub = _subsample_aux_by_slug(aux_remapped, frac, seed=0)
    log_print(
        f"  [{split}] aux subsample (frac={frac}, per-slug, response-level): "
        f"train {len(_split(aux_remapped, 'train'))} -> "
        f"{len(_split(aux_sub, 'train'))}, "
        f"test {len(_split(aux_remapped, 'test'))} -> "
        f"{len(_split(aux_sub, 'test'))}"
    )

    # Slice forget-answer positives to in-subset cot_idxs. These rows have
    # label=1 + matched_claim_idx=None, which the `remap_row` defensive
    # check rejects, so they don't pass through slice_split. Instead:
    # filter by cot_idx < n_subset_forget (same selector the IDK in-subset
    # filter uses). Nothing else needs rebasing — matched_claim_idx is
    # already None on every row. These are positives — do NOT downsample
    # (unlike IDK negatives).
    if forget_answers is not None:
        forget_answers_in_subset_parts: dict[str, Dataset] = {}
        for s in ("train", "test"):
            fa_ds = _split(forget_answers, s)
            fa_in_subset = fa_ds.filter(
                lambda r: r["cot_idx"] < n_subset_forget
            )
            forget_answers_in_subset_parts[s] = fa_in_subset
            log_print(
                f"  [{split}/{s}] forget_answers in-subset: "
                f"{len(fa_ds)} -> {len(fa_in_subset)}"
            )
        forget_answers_in_subset = DatasetDict(forget_answers_in_subset_parts)
    else:
        forget_answers_in_subset = None

    def _merged_parts(s: str) -> list[Dataset]:
        parts = [
            _split(clean_filt, s),
            _split(retain_sub, s),
            _split(idk_remapped, s),
            _split(aux_sub, s),
        ]
        if forget_answers_in_subset is not None:
            parts.append(_split(forget_answers_in_subset, s))
        return parts

    merged = DatasetDict(
        {s: concatenate_datasets(_merged_parts(s)) for s in ("train", "test")}
    )

    out_dir = LEAK_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if firstleak:
        # Save only the two artifacts that differ from the canonical build:
        # the truncated clean_forget and the recomposed full. The negatives
        # (retain/idk/aux) are byte-identical between the two modes —
        # don't duplicate them under suffixed names.
        clean_out_name = f"contextual_labeled_firstleak{aug_suffix}"
        full_out_name = f"contextual_labeled_full_firstleak{aug_suffix}"
        save_dataset_dict_with_invariant(clean_filt, str(out_dir / clean_out_name))
        save_dataset_dict_with_invariant(merged, str(out_dir / full_out_name))
        saved = [
            (clean_out_name, clean_filt),
            (full_out_name, merged),
        ]
    else:
        save_dataset_dict_with_invariant(clean_filt, str(out_dir / "contextual_labeled"))
        save_dataset_dict_with_invariant(retain_sub, str(out_dir / "contextual_labeled_retain"))
        # Save the remapped versions of idk and aux so the per-source
        # artifacts on disk match contextual_labeled_full's rows (both
        # originate from forget10 and are normalized onto the subset's
        # claim-idx space via slice_split above).
        save_dataset_dict_with_invariant(idk_remapped, str(out_dir / "contextual_labeled_idk"))
        save_dataset_dict_with_invariant(aux_sub, str(out_dir / "contextual_labeled_aux"))
        save_dataset_dict_with_invariant(merged, str(out_dir / "contextual_labeled_full"))
        saved = [
            ("contextual_labeled", clean_filt),
            ("contextual_labeled_retain", retain_sub),
            ("contextual_labeled_idk", idk_remapped),
            ("contextual_labeled_aux", aux_sub),
            ("contextual_labeled_full", merged),
        ]
        if forget_answers_in_subset is not None:
            save_dataset_dict_with_invariant(
                forget_answers_in_subset,
                str(out_dir / "contextual_labeled_forget_answers"),
            )
            saved.append(
                ("contextual_labeled_forget_answers", forget_answers_in_subset)
            )

    log_print(f"[{split}] saved subsets at {out_dir}:")
    for name, dsd in saved:
        log_print(
            f"  {name}: train={len(dsd['train'])}, test={len(dsd['test'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["forget05", "forget01"],
        choices=sorted(SUBSET_FRAC),
    )
    parser.add_argument(
        "--include_retain_xfer",
        action="store_true",
        help="Include clean_retain_xfer rows (out-of-subset forget prompts "
             "repurposed as retain signal) in the retain artifact. Default "
             "off — these rows are dropped, matching pre-slice behavior.",
    )
    parser.add_argument(
        "--firstleak",
        action="store_true",
        help="Use forget10/contextual_labeled_firstleak (produced by "
             "scripts.build_first_leak_classifier_data) as the clean_forget "
             "input instead of contextual_labeled. Outputs go to "
             "<split>/contextual_labeled_firstleak and "
             "<split>/contextual_labeled_full_firstleak so the canonical "
             "subset artifacts aren't clobbered.",
    )
    parser.add_argument(
        "--aug_suffix",
        default="",
        help="Suffix appended to the firstleak input + output names so "
             "augmentation variants don't clobber the canonical firstleak "
             "artifacts. Example: --aug_suffix _aug_v4 reads "
             "forget10/contextual_labeled_firstleak_aug_v4 and writes "
             "<split>/contextual_labeled_firstleak_aug_v4 + "
             "<split>/contextual_labeled_full_firstleak_aug_v4. Only valid "
             "with --firstleak.",
    )
    parser.add_argument(
        "--idk_subsample_target_base",
        default="train=939,test=212",
        help="Base IDK row counts per partition; the subset trims to "
             "int(round(frac * base)) per partition. Default matches the "
             "forget10 firstleak IDK volume (939 train / 212 test) so the "
             "subset's IDK density stays proportional. Format: "
             "'train=N,test=N' or 'all=N' (same target for all partitions) "
             "or 'none' to skip subsampling.",
    )
    parser.add_argument(
        "--exclude_paraphrase",
        action="store_true",
        help="Filter the in-subset IDK to prose rows only (source='clean_idk') "
             "after the cot_idx filter and before the subsample. Mirror of the "
             "same flag in build_first_leak_classifier_data. Use to train on a "
             "prose-only IDK distribution.",
    )
    args = parser.parse_args()

    idk_subsample_target_base = parse_subsample_target(args.idk_subsample_target_base)

    for split in args.splits:
        log_print(
            f"=== building {split} (frac={SUBSET_FRAC[split]}, "
            f"include_retain_xfer={args.include_retain_xfer}, "
            f"firstleak={args.firstleak}, aug_suffix={args.aug_suffix!r}, "
            f"idk_subsample_target_base={idk_subsample_target_base}) ==="
        )
        build_subset(
            split,
            SUBSET_FRAC[split],
            args.include_retain_xfer,
            firstleak=args.firstleak,
            aug_suffix=args.aug_suffix,
            idk_subsample_target_base=idk_subsample_target_base,
            exclude_paraphrase=args.exclude_paraphrase,
        )


if __name__ == "__main__":
    main()
