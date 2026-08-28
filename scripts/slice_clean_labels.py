"""Derive subset-split clean labels from a larger split's labeled dataset.

The R-TOFU forget splits are nested at task_id=1 (see CLAUDE.md's
"R-TOFU data loading and retain-split naming" section). For a subset
split S ⊂ L (e.g. S=forget01, L=forget10) every clean CoT in L —
whether it was labelled clean_forget (generated on a forget-L prompt)
or clean_retain (generated on a retain-L prompt) — stays a valid clean
CoT for S. Labels transfer with a claim-idx remap against S's own
knowledge bank.

Three output sources come out:

- **clean_forget** (was clean_forget in L, prompt in S): standard leak
  training for S's forget prompts.
- **clean_retain_xfer** (was clean_forget in L, prompt in L but not in S):
  the CoT is on an author S treats as retain; rows with in-subset claims
  stay label=1 (genuine leak of S content in an S-retain-prompt CoT),
  rows with out-of-subset claims or uncertain demote to background.
- **clean_retain** (was clean_retain in L): L's retain prompt pool is a
  subset of S's retain prompt pool (by the R-TOFU nesting), so labels
  transfer with only the claim-idx remap. Source unchanged; cot_idx
  continues to index into L's retain_cots list (which S should reuse
  verbatim as its retain CoT source).

Per-row remap table (let ``resolve(mci)`` denote the source→subset
claim-idx lookup — see ``build_claim_remap`` for the canonical text-based
implementation; the prefix-alignment fallback ``resolve(mci) = mci if
mci < len(S_kb.claims) else None`` is used only when no ``claim_remap``
is passed and the caller has guaranteed prefix alignment):

  if matched_claim_idx is not None and matched_claim_idx >= 0:
      subset_mci = resolve(matched_claim_idx)
      if subset_mci is not None:        # source claim exists in S's KB
          label = 1
          matched_claim_idx = subset_mci
      else:                              # no S-side counterpart → retain-space
          label = 0
          matched_claim_idx = -1
  elif matched_claim_idx == -1 and label == 1 (uncertain leak):
      if source is clean_retain, or (source is clean_forget and cot_idx < N_subset_forget):
          # Preserve uncertain
          label = 1
      else:  # clean_forget on out-of-subset prompt
          label = 0

Usage:
    python -m scripts.slice_clean_labels \\
        --input leak_detector_data/forget10/labeled \\
        --kb knowledge_banks/forget05 \\
        --n_subset_forget 200 \\
        --output leak_detector_data/forget05/labeled_from_forget10

The output ``clean_retain`` rows assume S will use L's ``retain_cots.csv``
verbatim as its retain CoT source. If S has its own freshly-generated
retain CoTs, they need a separate labelling pass and concatenation.

Note: when this script is invoked via
:mod:`scripts.build_subset_classifier_data` (the canonical wrapper), the
``clean_retain_xfer`` rows are dropped by default — pass
``--include_retain_xfer`` to fold them into the retain artifact as
described above.
"""
import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping

from datasets import Dataset, DatasetDict, load_from_disk

from redact.leak_dataset_schema import save_dataset_dict_with_invariant
from redact.utils import log_print


def read_kb_size(kb_dir: str) -> int:
    """Return the number of claims in a KB, from its ``claims.json``.

    Used as the subset-KB size for ``remap_row``'s prefix-alignment
    fallback membership test (in-subset iff
    ``matched_claim_idx < subset_kb_size``). Prefer passing an explicit
    ``claim_remap`` (see ``build_claim_remap``) — the size comparison is
    only valid when the subset KB is a strict prefix of the source KB.
    """
    with open(os.path.join(kb_dir, "claims.json")) as f:
        claims = json.load(f)
    return len(claims)


def build_claim_remap(source_kb_dir: str, subset_kb_dir: str) -> dict[int, int]:
    """Build a ``source_idx -> subset_idx`` remap from two KBs' ``claims.json``.

    Both KBs are written by ``scripts.build_kb`` and store claims as a list of
    strings. The remap is a text-based lookup: each source claim that has an
    exact-match counterpart in the subset's claims.json maps to the subset
    index of that match.

    Why this exists: the per-split KBs are *intended* to be prefix-aligned
    on the shared author set, but ``scripts.resolve_kb.append_birth_claims``
    tacks each split's birth-fact claims onto the **end** of that split's
    claims list rather than inserting them at per-author positions. For
    forget01 there's nothing to append (the prefix is too short to expose
    the misalignment); for forget05 it pushes four trailing claims (idx
    248-251) into positions that don't match forget10's KB, which appends
    its own larger birth-fact set at idx ~501+. (Likely a build-step
    regression — a re-sort or insertion-at-position pass appears to have
    been dropped at some point.)

    Without an explicit remap, ``slice_split`` would (a) demote genuine
    in-subset leaks to retain-space when the source idx overshoots the
    subset KB size, and (b) silently rewrite ``matched_claim_idx`` to a
    DIFFERENT claim when the two KBs disagree at the same index. This
    text-based remap is correct regardless of the cause and stays correct
    if the build-step regression is later fixed.

    Claims not present in the subset KB are deliberately omitted from the
    returned dict (callers should treat a missing key as "out-of-subset →
    demote to retain-space").

    Duplicate handling — guards against malformed KBs even though R-TOFU
    KBs are deduped by ``scripts.resolve_kb`` before being written:
      - **Duplicate source claims**: every source idx with a matching text
        gets a remap entry, all pointing at the same (first-seen) subset
        idx. So the remap can be a non-injective map even though it's
        always a valid function from source idx → subset idx.
      - **Duplicate subset claims**: ``setdefault`` keeps only the first
        subset occurrence; later duplicates are unreachable through the
        remap.
    """
    with open(os.path.join(source_kb_dir, "claims.json")) as f:
        source_claims: list[str] = json.load(f)
    with open(os.path.join(subset_kb_dir, "claims.json")) as f:
        subset_claims: list[str] = json.load(f)
    subset_idx_by_text: dict[str, int] = {}
    for i, text in enumerate(subset_claims):
        subset_idx_by_text.setdefault(text, i)
    remap: dict[int, int] = {}
    for src_i, text in enumerate(source_claims):
        sub_i = subset_idx_by_text.get(text)
        if sub_i is not None:
            remap[src_i] = sub_i
    return remap


def remap_row(
    row: dict,
    subset_kb_size: int,
    n_subset_forget: int,
    claim_remap: Mapping[int, int] | None = None,
) -> dict:
    """Apply the remap table in the module docstring to one row.

    Handles ``source='forget'`` and ``source='retain'`` / ``'idk'`` / ``'aux'``:
    - forget rows (tags={'clean'}): prompt-membership test on
      ``cot_idx < n_subset_forget``. In-subset prompts keep
      ``source='forget', source_tags=['clean']``; out-of-subset prompts
      become ``source='retain', source_tags=['clean','xfer']``. Uncertain
      leaks (label=1, matched_claim_idx=-1) are preserved only on in-subset
      prompts.
    - retain / idk / aux / idk+paraphrase rows: prompt is retain by
      construction for any subset split. ``(source, source_tags)`` are
      propagated. Uncertain leaks are always preserved.

    Rejects every row whose ``(source, source_tags)`` doesn't match an
    entry in the allowlist (see asserts at function entry).

    Raises ValueError on malformed rows that shouldn't exist in a
    well-formed aggregate_labels output (label=1 with matched_claim_idx=None,
    or matched_claim_idx pointing at a negative non-sentinel index).
    """
    source = row["source"]
    source_tags = list(row["source_tags"])
    tags = set(source_tags)

    # Allowlist gate as explicit per-row asserts. Tripwires for every
    # off-pattern combination that must be filtered upstream instead of
    # silently propagated here. (Plan Stage 3.3.)
    assert source in {"forget", "retain", "idk", "aux"}, (
        f"remap_row: unsupported source {source!r}; allowlist "
        f"{{'forget','retain','idk','aux'}}"
    )
    assert "clean" in tags, (
        f"remap_row: row missing 'clean' tag; tags={source_tags!r}"
    )
    assert "corrupted" not in tags, (
        f"remap_row: corrupted rows are out of scope; tags={source_tags!r}"
    )
    assert "answer" not in tags, (
        f"remap_row: forget-answer rows have a separate code path; "
        f"tags={source_tags!r}"
    )
    assert "xfer" not in tags, (
        f"remap_row: xfer rows are already remapped and must not re-enter "
        f"this function; tags={source_tags!r}"
    )
    assert "reword_s0" not in tags, (
        f"remap_row: _aug_v4 reword rows are out of scope; tags={source_tags!r}"
    )

    # Set-equality allowlist (more permissive failure mode than the asserts
    # above, but defends against e.g. unexpected new tags slipping through).
    if not (
        (source == "forget" and tags == {"clean"})
        or (source == "retain" and tags == {"clean"})
        or (source == "idk" and tags == {"clean"})
        or (source == "idk" and tags == {"clean", "paraphrase"})
        or (source == "aux" and tags == {"clean"})
    ):
        raise ValueError(
            f"remap_row: unsupported (source, tags) combination: "
            f"source={source!r}, tags={source_tags!r}"
        )

    matched_claim_idx = row.get("matched_claim_idx", -1)
    cot_idx = row["cot_idx"]
    label = row["label"]

    # For retain / idk / aux, every row is "in-subset by construction" —
    # none are keyed to forget-set prompts, so no out-of-subset demotion
    # applies. Treat as in-subset for the uncertain-leak preservation rule
    # below.
    in_subset_prompt = (
        True if source in ("retain", "idk", "aux")
        else cot_idx < n_subset_forget
    )

    # Defensive: label=1 rows must have a real matched_claim_idx or the
    # uncertain-sentinel -1. Everything else is a regression upstream.
    if label == 1 and matched_claim_idx is None:
        raise ValueError(
            f"Malformed row uid={row.get('uid')!r}: label=1 but "
            f"matched_claim_idx=None. aggregate_labels should have dropped "
            f"this row; a non-dropped occurrence means upstream regressed."
        )
    if (
        matched_claim_idx is not None
        and matched_claim_idx != -1
        and matched_claim_idx < 0
    ):
        raise ValueError(
            f"Malformed row uid={row.get('uid')!r}: "
            f"matched_claim_idx={matched_claim_idx} (only -1 or non-negative "
            f"values are meaningful)."
        )

    if matched_claim_idx is not None and matched_claim_idx >= 0:
        # Resolve the source claim idx against the subset KB. Prefer the
        # explicit text-based ``claim_remap`` (see ``build_claim_remap``).
        # The size-comparison fallback below is correct ONLY when the
        # subset KB is a strict prefix of the source KB (verified for
        # forget01⊂forget10; NOT true for forget05⊂forget10 — its KB
        # diverges at idx 248). Without ``claim_remap``, callers handling
        # non-prefix-aligned KBs will silently miscategorize rows: real
        # leaks at high source idx get demoted, and rows at indices where
        # the two KBs disagree get the wrong matched_claim_idx.
        if claim_remap is not None:
            subset_mci = claim_remap.get(matched_claim_idx)
        elif matched_claim_idx < subset_kb_size:
            subset_mci = matched_claim_idx
        else:
            subset_mci = None
        if subset_mci is None:
            # Source claim has no counterpart in the subset KB → retain-space.
            new_label = 0
            new_matched_claim_idx = -1
        else:
            new_label = 1
            new_matched_claim_idx = subset_mci
    elif matched_claim_idx == -1 and label == 1 and in_subset_prompt:
        # Genuine uncertain leak on an in-subset prompt (forget) or
        # any retain/idk/aux prompt (always in-subset by construction).
        new_label = 1
        new_matched_claim_idx = -1
    else:
        # Remaining well-formed cases:
        #   - forget uncertain (label=1, matched=-1) on out-of-subset
        #     prompt → demote to retain-space
        #   - label=0 row (matched_claim_idx None or -1) → already retain-space
        new_label = 0
        new_matched_claim_idx = -1

    # Source tagging per the refactor table:
    #   - retain / idk / aux: passthrough
    #   - forget in-subset prompt: stays ('forget', ['clean'])
    #   - forget out-of-subset prompt: becomes ('retain', ['clean','xfer'])
    #     (cot_idx still indexes into the SOURCE split's forget-prompt pool,
    #     not the subset split's retain pool — downstream code grouping by
    #     cot_idx alone across sources is unsafe.)
    if source == "forget":
        if in_subset_prompt:
            new_source = "forget"
            new_source_tags = ["clean"]
        else:
            new_source = "retain"
            new_source_tags = ["clean", "xfer"]
    else:
        new_source = source
        new_source_tags = source_tags

    # Preserve every input field. Only overwrite the fields the remap
    # computes; defaults for legacy-schema fields (rationale, escalated,
    # phase) are filled in below for callers that feed pre-contextual rows.
    out = dict(row)
    # Legacy pre-refactor datasets may still carry a cluster_id keyed to the
    # SOURCE split's cluster space; it is no longer remapped, so drop it rather
    # than emit a silently-wrong column.
    out.pop("cluster_id", None)
    out["label"] = new_label
    out["matched_claim_idx"] = int(new_matched_claim_idx)
    out["source"] = new_source
    out["source_tags"] = list(new_source_tags)
    out["cot_idx"] = int(cot_idx)
    out["sent_idx"] = int(row["sent_idx"])
    out.setdefault("rationale", "")
    out["escalated"] = bool(out.get("escalated") or False)
    out.setdefault("phase", "cot")
    return out


def slice_split(
    split_ds: Dataset,
    subset_kb_size: int,
    n_subset_forget: int,
    claim_remap: Mapping[int, int] | None = None,
) -> tuple[Dataset, dict]:
    """Pass through clean forget + retain rows with the claim-idx remap.

    Operates on the new ``(source, source_tags)`` schema (see
    :mod:`redact.leak_dataset_schema`).

    - ``source='forget'`` rows with ``tags={'clean'}`` keep
      ``source='forget'`` if the prompt is in-subset, else become
      ``source='retain'`` with ``tags=['clean','xfer']`` via ``remap_row``.
    - ``source='retain'`` rows with ``tags={'clean'}`` keep ``source='retain'``
      — forget10's retain prompts are a subset of forget05's retain prompts
      (and forget05's are a subset of forget01's), so the labels transfer
      with only the claim-idx remap.
    - ``source='idk'`` rows (``tags={'clean'}`` or ``{'clean','paraphrase'}``)
      and ``source='aux'`` rows (``tags={'clean'}``) pass through.
    - Rows tagged with ``"corrupted"`` are dropped; they come from the
      source split's soft-token bank which the subset won't reuse.

    Returns the remapped Dataset plus a stats dict.
    """
    stats: dict = {
        "input_rows": len(split_ds),
        "kept_clean_forget": 0,
        "kept_clean_retain": 0,
        "kept_clean_idk": 0,
        "kept_clean_idk_paraphrase": 0,
        "kept_clean_aux": 0,
        "dropped_corrupted": 0,
        "dropped_other_source": 0,
    }
    remapped_rows: list[dict] = []
    for row in split_ds:
        src = row["source"]
        tags = set(row["source_tags"])
        # Corrupted rows are dropped before any per-source counting so the
        # "kept_*" buckets only ever reflect clean rows.
        if "corrupted" in tags:
            stats["dropped_corrupted"] += 1
            continue
        if src == "forget" and tags == {"clean"}:
            stats["kept_clean_forget"] += 1
        elif src == "retain" and "xfer" not in tags:
            # xfer rows shouldn't reach slice_split (they're produced by it
            # downstream of remap_row, not consumed). Bucket only non-xfer
            # retain rows; xfer would fall through to dropped_other_source.
            stats["kept_clean_retain"] += 1
        elif src == "idk" and "paraphrase" not in tags:
            stats["kept_clean_idk"] += 1
        elif src == "idk" and "paraphrase" in tags:
            stats["kept_clean_idk_paraphrase"] += 1
        elif src == "aux":
            stats["kept_clean_aux"] += 1
        else:
            stats["dropped_other_source"] += 1
            continue
        remapped_rows.append(
            remap_row(
                row,
                subset_kb_size,
                n_subset_forget,
                claim_remap=claim_remap,
            )
        )

    if not remapped_rows:
        # Preserve the input dataset's Features on the empty case.
        ds = split_ds.select([])
    else:
        # Pass split_ds.features so the output keeps the contextual
        # Features (Sequence(Value("string")) for context_sentences, etc.)
        # when the input has them. For legacy bare-sentence inputs the
        # features just propagate.
        ds = Dataset.from_list(remapped_rows, features=split_ds.features)
    return ds, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Source labeled DatasetDict (e.g. "
                             "leak_detector_data/forget10/labeled).")
    parser.add_argument("--kb", type=str, required=True,
                        help="Subset split's KB "
                             "(e.g. knowledge_banks/forget01). Must contain "
                             "claims.json.")
    parser.add_argument("--source_kb", type=str, default=None,
                        help="Source split's KB (e.g. knowledge_banks/forget10). "
                             "When provided, builds a text-based "
                             "source_claim_idx -> subset_claim_idx remap that "
                             "fixes KB-misalignment between the two splits. "
                             "Omit only when the subset KB is known to be a "
                             "strict prefix of the source KB (forget01 is; "
                             "forget05 is not — pass --source_kb for that "
                             "case).")
    parser.add_argument("--n_subset_forget", type=int, required=True,
                        help="Number of forget prompts in the subset split "
                             "(40 for forget01, 200 for forget05, all assuming "
                             "task_id=1). Rows with cot_idx < this value keep "
                             "source='clean_forget'; others become "
                             "source='clean_retain_xfer'.")
    parser.add_argument("--output", type=str, required=True,
                        help="Destination DatasetDict directory.")
    args = parser.parse_args()

    subset_kb_size = read_kb_size(args.kb)
    log_print(f"Subset KB {args.kb}: {subset_kb_size} claims")
    log_print(f"Subset forget prompt count: {args.n_subset_forget}")

    claim_remap = None
    if args.source_kb is not None:
        claim_remap = build_claim_remap(args.source_kb, args.kb)
        log_print(
            f"Built claim remap from {args.source_kb} -> {args.kb}: "
            f"{len(claim_remap)} source claims with subset counterparts"
        )

    log_print(f"Loading source labeled DatasetDict from {args.input}")
    source = load_from_disk(args.input)

    out: dict[str, Dataset] = {}
    agg_stats: dict[str, dict] = {}
    for split_name in source.keys():
        log_print(f"\nRemapping split '{split_name}' ({len(source[split_name])} rows)")
        ds, stats = slice_split(
            source[split_name],
            subset_kb_size,
            args.n_subset_forget,
            claim_remap=claim_remap,
        )
        out[split_name] = ds
        agg_stats[split_name] = stats

        src = Counter(ds["source"])
        lbl = Counter(ds["label"])
        # Print the full stats dict so the clean_idk / clean_aux counters
        # surface (added when the allowlist was extended; would otherwise
        # silently undercount if slice_clean_labels is run standalone on a
        # dataset that includes those sources).
        log_print(f"  Stats: {dict(stats)}")
        log_print(f"  Output sources: {dict(src)}")
        log_print(f"  Output labels:  {dict(lbl)}")

    save_dataset_dict_with_invariant(DatasetDict(out), args.output)
    log_print(f"\nSaved sliced DatasetDict to {args.output}")
    log_print(
        "Next: concatenate this with freshly-labeled retain rows for the "
        "subset split to build the full labeled DatasetDict used by "
        "augment_classifier_data and build_soft_token_data."
    )


if __name__ == "__main__":
    main()
