"""Schema-invariant + augment-routing-predicate tests for the
``(source, source_tags)`` leak-detector schema.

Two concerns:

1. **Schema invariant** — every in-use row must have
   ``source in {"forget","retain","aux","idk"}`` and exactly one of
   ``{"clean","corrupted"}`` in ``source_tags``. Checked on a synthetic
   DatasetDict (always runs) and, when present and already migrated, on
   the on-disk ``contextual_labeled*`` artifacts.

2. **Augment-routing predicate** — the pending augment-routing plan keys
   on ``bool(set(tags) & {"paraphrase", "answer", "reword_s0"})`` to send
   augment rows to train-only. This test pins that the predicate
   identifies exactly the augment variants and never the clean/xfer/
   corrupted-base rows.

Usage:
    conda run -n redact python -m pytest tests/test_source_tags_invariants.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from redact.leak_dataset_schema import AUGMENT_TAGS, assert_provenance_xor, is_augment
from tests._fixtures import make_row


REPO = Path(__file__).resolve().parent.parent
LEAK_BASE = REPO / "leak_detector_data_promptfmt"

VALID_PRIMARIES = {"forget", "retain", "aux", "idk"}


# ----------------------------------------------------------------------
# Synthetic DatasetDict: always-running schema invariant
# ----------------------------------------------------------------------

def _synthetic_dd() -> DatasetDict:
    """A DatasetDict covering every canonical (source, tags) combination."""
    rows = [
        make_row(source="forget", tags=["clean"], cot_idx=0),
        make_row(source="forget", tags=["clean", "answer"], cot_idx=1),
        make_row(source="forget", tags=["clean", "reword_s0"], cot_idx=2),
        make_row(source="forget", tags=["corrupted", "noise_topk"], cot_idx=3),
        make_row(source="retain", tags=["clean"], cot_idx=4),
        make_row(source="retain", tags=["clean", "xfer"], cot_idx=5),
        make_row(source="idk", tags=["clean"], cot_idx=6),
        make_row(source="idk", tags=["clean", "paraphrase"], cot_idx=7),
        make_row(source="aux", tags=["clean"], cot_idx=8),
    ]
    return DatasetDict({
        "train": Dataset.from_list(rows[:5]),
        "test": Dataset.from_list(rows[5:]),
    })


def test_synthetic_dd_satisfies_schema_invariant():
    """Every synthetic row has a valid primary and exactly one provenance
    tag."""
    dd = _synthetic_dd()
    for split_name, split in dd.items():
        for row in split:
            assert row["source"] in VALID_PRIMARIES, (
                f"{split_name}: invalid source {row['source']!r}"
            )
            tags = set(row["source_tags"])
            assert ("clean" in tags) ^ ("corrupted" in tags), (
                f"{split_name}: row must have exactly one of clean/corrupted; "
                f"got {row['source_tags']!r}"
            )


def test_assert_provenance_xor_passes_on_synthetic_dd():
    """The production save-time guard accepts the synthetic DatasetDict."""
    assert_provenance_xor(_synthetic_dd(), out_path="<synthetic>")


def test_assert_provenance_xor_rejects_missing_provenance():
    """A row with neither clean nor corrupted trips the invariant."""
    bad = DatasetDict({
        "train": Dataset.from_list([
            make_row(source="idk", tags=["paraphrase"]),  # no provenance tag
        ]),
    })
    with pytest.raises(AssertionError, match="clean XOR corrupted"):
        assert_provenance_xor(bad, out_path="<bad>")


def test_assert_provenance_xor_rejects_both_provenance():
    """A row with both clean and corrupted trips the invariant."""
    bad = DatasetDict({
        "train": Dataset.from_list([
            make_row(source="forget", tags=["clean", "corrupted"]),
        ]),
    })
    with pytest.raises(AssertionError, match="clean XOR corrupted"):
        assert_provenance_xor(bad, out_path="<bad>")


# ----------------------------------------------------------------------
# On-disk artifacts: skip cleanly until Stage 5 rebuild has migrated them
# ----------------------------------------------------------------------

def _migrated_contextual_artifacts() -> list[Path]:
    """Return on-disk ``contextual_labeled*`` artifacts that already carry
    the ``source_tags`` column (i.e. have been rebuilt post-refactor).

    Excludes ``.bak_*`` snapshots and ``.pre_source_tags`` stashes.
    """
    from datasets import load_from_disk

    out: list[Path] = []
    if not LEAK_BASE.exists():
        return out
    for split_dir in sorted(LEAK_BASE.iterdir()):
        if not split_dir.is_dir():
            continue
        for artifact in sorted(split_dir.glob("contextual_labeled*")):
            name = artifact.name
            if ".bak_" in name or name.startswith("."):
                continue
            if not (artifact / "dataset_dict.json").exists() and not any(
                artifact.glob("*/dataset_info.json")
            ):
                # Not a DatasetDict directory.
                continue
            try:
                dd = load_from_disk(str(artifact))
                split = next(iter(dd.values())) if hasattr(dd, "values") else dd
                if "source_tags" in split.features:
                    out.append(artifact)
            except Exception:
                continue
    return out


def test_on_disk_artifacts_satisfy_schema_invariant():
    """Every migrated on-disk ``contextual_labeled*`` artifact satisfies the
    schema invariant. Skips when no migrated artifact exists yet (Stage 5
    rebuild pending)."""
    from datasets import load_from_disk

    artifacts = _migrated_contextual_artifacts()
    if not artifacts:
        pytest.skip(
            "no on-disk contextual_labeled* artifact carries source_tags yet "
            "(Stage 5 rebuild pending); synthetic-DatasetDict variant covers "
            "the invariant in the meantime"
        )
    for artifact in artifacts:
        dd = load_from_disk(str(artifact))
        for split_name, split in dd.items():
            sources = split["source"]
            tags_list = split["source_tags"]
            for i, (src, tags) in enumerate(zip(sources, tags_list)):
                assert src in VALID_PRIMARIES, (
                    f"{artifact.name}/{split_name}[{i}]: invalid source {src!r}"
                )
                tag_set = set(tags)
                assert ("clean" in tag_set) ^ ("corrupted" in tag_set), (
                    f"{artifact.name}/{split_name}[{i}]: must have exactly one "
                    f"of clean/corrupted; got {tags!r}"
                )


# ----------------------------------------------------------------------
# Augment-routing predicate
# ----------------------------------------------------------------------

# The augment-routing predicate is the canonical one from
# ``redact.leak_dataset_schema``; alias it locally so the existing call sites
# below read unchanged.
_is_augment = is_augment


def test_augment_predicate_identifies_all_augment_variants():
    """paraphrase / answer / reword_s0 rows are augment; clean/xfer/
    corrupted-base rows are not."""
    augment_cases = [
        ("idk", ["clean", "paraphrase"]),
        ("forget", ["clean", "answer"]),
        ("forget", ["clean", "reword_s0"]),
    ]
    non_augment_cases = [
        ("forget", ["clean"]),
        ("retain", ["clean"]),
        ("retain", ["clean", "xfer"]),
        ("idk", ["clean"]),
        ("aux", ["clean"]),
        ("forget", ["corrupted", "noise_topk"]),
    ]
    for source, tags in augment_cases:
        row = make_row(source=source, tags=tags)
        assert _is_augment(row["source_tags"]), (
            f"({source}, {tags}) should be flagged as augment"
        )
    for source, tags in non_augment_cases:
        row = make_row(source=source, tags=tags)
        assert not _is_augment(row["source_tags"]), (
            f"({source}, {tags}) should NOT be flagged as augment"
        )


def test_augment_predicate_order_independent():
    """The predicate is set-based — tag order doesn't matter."""
    assert _is_augment(["clean", "answer"])
    assert _is_augment(["answer", "clean"])
    assert _is_augment(["paraphrase", "clean"])
    assert not _is_augment(["clean"])


def test_augment_predicate_partitions_synthetic_dd():
    """On the synthetic DatasetDict, the predicate selects exactly the
    three augment rows (answer, reword_s0, paraphrase)."""
    dd = _synthetic_dd()
    augment_rows = [
        row
        for split in dd.values()
        for row in split
        if _is_augment(row["source_tags"])
    ]
    augment_tag_sets = sorted(
        tuple(sorted(set(r["source_tags"]) & AUGMENT_TAGS)) for r in augment_rows
    )
    assert augment_tag_sets == [("answer",), ("paraphrase",), ("reword_s0",)]
