"""Migrate a leak-detector DatasetDict from the legacy single-column
``source`` schema to the new ``(source, source_tags)`` two-column schema.

Usage::

    python -m scripts.migrate_source_to_tags \\
        --input  leak_detector_data_promptfmt/forget10/.pre_source_tags/contextual_labeled \\
        --output leak_detector_data_promptfmt/forget10/contextual_labeled__migrated

The migration is row-wise and lossless for everything except the
``_aug_v4`` reword artifact, where the legacy schema reused
``source='clean_forget'`` for reworded sentence-0 positives and
disambiguated them only via the ``rationale`` field. ``--detect_reword``
(default on) instructs the migration to detect these rows and map them
to ``('forget', ['clean', 'reword_s0'])``. Disable it for plain
``contextual_labeled`` / ``contextual_labeled_firstleak`` artifacts that
contain no reworded rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import DatasetDict, load_from_disk

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from redact.leak_dataset_schema import (  # noqa: E402
    _legacy_source_to_new,
    save_dataset_dict_with_invariant,
)
from redact.utils import log_print  # noqa: E402
from scripts.build_labeled_sources import CONTEXTUAL_FEATURES  # noqa: E402


_REWORD_RATIONALE_PREFIX = "Augmented (replace-mode"


def _row_is_reword(row: dict) -> bool:
    """Return True if a legacy row is an ``_aug_v4`` sentence-0 reword.

    Reworded rows have ``source='clean_forget'``, ``label=1``,
    ``sent_idx=0``, and a rationale starting with ``"Augmented
    (replace-mode"``.
    """
    if row.get("source") != "clean_forget":
        return False
    if int(row.get("label", 0)) != 1:
        return False
    if int(row.get("sent_idx", -1)) != 0:
        return False
    rationale = row.get("rationale") or ""
    return rationale.startswith(_REWORD_RATIONALE_PREFIX)


def _map_row(row: dict, *, detect_reword: bool) -> dict:
    old = row.get("source")
    if not isinstance(old, str) or not old:
        raise ValueError(
            f"migrate_source_to_tags: missing or empty 'source' field "
            f"in row uid={row.get('uid')!r}"
        )
    if detect_reword and _row_is_reword(row):
        primary, tags = "forget", ["clean", "reword_s0"]
    else:
        primary, tags = _legacy_source_to_new(old)
    new_row = dict(row)
    new_row["source"] = primary
    new_row["source_tags"] = list(tags)
    return new_row


def migrate_dataset_dict(
    dd: DatasetDict, *, detect_reword: bool = True
) -> DatasetDict:
    """Apply the legacy → new mapping per row across every split.

    Pre-flight: every split must already carry a ``source`` column. The
    migration is destructive on that column — it's overwritten with the
    primary value — and adds the new ``source_tags`` column.
    """
    for split_name, ds in dd.items():
        if "source" not in ds.column_names:
            raise ValueError(
                f"migrate_dataset_dict: split {split_name!r} has no "
                f"'source' column; column_names={ds.column_names}"
            )

    out: dict = {}
    for split_name, ds in dd.items():
        mapped = ds.map(
            lambda row: _map_row(row, detect_reword=detect_reword),
            features=CONTEXTUAL_FEATURES,
            load_from_cache_file=False,
        )
        # Post-flight: every output row must have a recognised primary
        # source and a non-empty source_tags list.
        primaries = set(mapped["source"])
        if not primaries.issubset({"forget", "retain", "idk", "aux"}):
            bad = primaries - {"forget", "retain", "idk", "aux"}
            raise ValueError(
                f"migrate_dataset_dict: split {split_name!r} has invalid "
                f"primaries after migration: {sorted(bad)}"
            )
        for i, tags in enumerate(mapped["source_tags"]):
            if not tags:
                raise ValueError(
                    f"migrate_dataset_dict: split {split_name!r} row {i} "
                    f"has empty source_tags after migration"
                )
        out[split_name] = mapped
    return DatasetDict(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", required=True,
        help="Path to the input DatasetDict (legacy schema).",
    )
    p.add_argument(
        "--output", required=True,
        help="Destination path for the migrated DatasetDict.",
    )
    p.add_argument(
        "--detect_reword", dest="detect_reword",
        action="store_true",
        help="Disambiguate _aug_v4 reword rows (default).",
    )
    p.add_argument(
        "--no_detect_reword", dest="detect_reword",
        action="store_false",
        help="Disable reword detection. Use only for artifacts that "
             "contain no _aug_v4 reword rows.",
    )
    p.set_defaults(detect_reword=True)
    args = p.parse_args()

    log_print(f"Loading legacy DatasetDict from {args.input}")
    dd = load_from_disk(args.input)
    if not isinstance(dd, DatasetDict):
        raise ValueError(
            f"migrate_source_to_tags: {args.input!r} does not load as a "
            f"DatasetDict (got {type(dd).__name__})"
        )

    for split_name, ds in dd.items():
        log_print(f"  {split_name}: {len(ds)} rows")

    log_print(
        f"Migrating with detect_reword={args.detect_reword}"
    )
    migrated = migrate_dataset_dict(dd, detect_reword=args.detect_reword)

    log_print(f"Saving migrated DatasetDict to {args.output}")
    save_dataset_dict_with_invariant(migrated, args.output)

    for split_name, ds in migrated.items():
        from collections import Counter

        primaries = Counter(ds["source"])
        log_print(f"  {split_name}: source counts = {dict(primaries)}")


if __name__ == "__main__":
    main()
