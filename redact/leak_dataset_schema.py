"""Schema mapping helpers between the legacy single-column ``source`` and
the new ``(source, source_tags)`` two-column schema for the leak-detector
dataset.

The legacy ``source`` column overloaded two semantic axes onto one string:
the primary label-space bucket (forget / retain / aux / idk) and the
provenance/variant tag (clean prose, paraphrase, sentence-0 reword,
forget-answer positive, xfer, corrupted-by-soft-token). Splitting the two
axes eliminates the brittle ``startswith(...)`` dispatch and makes
augment-routing predicates expressible as set membership.

Public surface: the legacy<->new mappers (``_legacy_source_to_new``,
``_new_to_legacy_source``), the save-time invariant
(``assert_provenance_xor`` / ``save_dataset_dict_with_invariant``), and the
augment-routing predicate (``AUGMENT_TAGS`` / ``is_augment``).

Mapping table (canonical):

    clean_forget                  <-> ("forget",  ["clean"])
    clean_answer_forget           <-> ("forget",  ["clean", "answer"])
    clean_forget + reword_s0      <-> ("forget",  ["clean", "reword_s0"])
                                  (one-way: from the new side back to the
                                  legacy side is well-defined; from the
                                  legacy side, the bare string
                                  ``"clean_forget"`` is ambiguous and the
                                  caller must disambiguate from auxiliary
                                  signal — see ``_legacy_source_to_new``.)
    corrupted_<scheme>            <-> ("forget",  ["corrupted", "<scheme>"])
    clean_retain                  <-> ("retain",  ["clean"])
    clean_retain_xfer             <-> ("retain",  ["clean", "xfer"])
    clean_idk                     <-> ("idk",     ["clean"])
    clean_idk_paraphrase          <-> ("idk",     ["clean", "paraphrase"])
    clean_aux                     <-> ("aux",     ["clean"])

Unknown inputs raise ``ValueError``; there is no silent passthrough.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict


# legacy/unused: the current pipeline uses IDK targets; ``corrupted`` rows
# are never generated. The tag and the clean-XOR-corrupted invariant
# (``assert_provenance_xor``) are kept dormant so old corrupted-scheme
# datasets still round-trip and the invariant still rejects rows with
# neither provenance tag.
_PROVENANCE_TAGS = frozenset({"clean", "corrupted"})


AUGMENT_TAGS = frozenset({"paraphrase", "answer", "reword_s0"})


def is_augment(source_tags) -> bool:
    """True iff a row is an augment variant (tags intersect AUGMENT_TAGS).
    Used to route augment rows to the train partition only so the test
    partition reflects the held-out non-augment distribution."""
    return bool(set(source_tags) & AUGMENT_TAGS)


_LEGACY_TO_NEW: dict[str, tuple[str, list[str]]] = {
    "clean_forget":          ("forget", ["clean"]),
    "clean_answer_forget":   ("forget", ["clean", "answer"]),
    "clean_retain":          ("retain", ["clean"]),
    "clean_retain_xfer":     ("retain", ["clean", "xfer"]),
    "clean_idk":             ("idk",    ["clean"]),
    "clean_idk_paraphrase":  ("idk",    ["clean", "paraphrase"]),
    "clean_aux":             ("aux",    ["clean"]),
}


def _legacy_source_to_new(old: str) -> tuple[str, list[str]]:
    """Map a legacy ``source`` string to ``(source, source_tags)``.

    Recognised cases:
      - the literal strings in the mapping table (``clean_forget``,
        ``clean_answer_forget``, ``clean_retain``, ``clean_retain_xfer``,
        ``clean_idk``, ``clean_idk_paraphrase``, ``clean_aux``)
      - ``corrupted_<scheme>`` (any string starting with ``corrupted_``)

    The ``_aug_v4`` reword case is intentionally NOT handled here: rows
    rewritten by ``apply_sentence0_reword_patches`` carry the same
    string-side ``source='clean_forget'`` as un-rewritten clean-forget
    positives, so the bare string is ambiguous. The migration script
    disambiguates using the ``rationale`` field (rewritten rows start
    with ``"Augmented (replace-mode"``). Callers that need to map a
    reworded row should construct ``("forget", ["clean", "reword_s0"])``
    directly rather than going through this function.

    Raises ``ValueError`` for any input outside the recognised set —
    silent passthrough would hide upstream schema regressions.
    """
    if not isinstance(old, str) or not old:
        raise ValueError(
            f"_legacy_source_to_new: input must be a non-empty string, "
            f"got {old!r}"
        )
    if old in _LEGACY_TO_NEW:
        primary, tags = _LEGACY_TO_NEW[old]
        return primary, list(tags)
    if old.startswith("corrupted_"):
        scheme = old[len("corrupted_"):]
        if not scheme:
            raise ValueError(
                f"_legacy_source_to_new: corrupted_<scheme> missing scheme "
                f"after underscore: {old!r}"
            )
        return "forget", ["corrupted", scheme]
    raise ValueError(f"_legacy_source_to_new: unknown legacy source {old!r}")


def _new_to_legacy_source(primary: str, tags: list[str]) -> str:
    """Map ``(source, source_tags)`` back to the legacy string.

    Inverse of ``_legacy_source_to_new`` over the canonical table:

        ("forget",  ["clean"])              -> "clean_forget"
        ("forget",  ["clean", "answer"])    -> "clean_answer_forget"
        ("forget",  ["clean", "reword_s0"]) -> raises ValueError (no
            unambiguous legacy string; see docstring of _legacy_source_to_new)
        ("forget",  ["corrupted", scheme])  -> "corrupted_<scheme>"
        ("retain",  ["clean"])              -> "clean_retain"
        ("retain",  ["clean", "xfer"])      -> "clean_retain_xfer"
        ("idk",     ["clean"])              -> "clean_idk"
        ("idk",     ["clean", "paraphrase"])-> "clean_idk_paraphrase"
        ("aux",     ["clean"])              -> "clean_aux"

    Tags compare set-wise (order doesn't matter). Inputs outside the
    table raise ``ValueError``.
    """
    if not isinstance(primary, str) or primary not in {
        "forget", "retain", "idk", "aux"
    }:
        raise ValueError(
            f"_new_to_legacy_source: primary must be one of "
            f"{{'forget','retain','idk','aux'}}; got {primary!r}"
        )
    if not isinstance(tags, list) or not tags:
        raise ValueError(
            f"_new_to_legacy_source: tags must be a non-empty list; "
            f"got {tags!r}"
        )
    tag_set = frozenset(tags)
    provenance = tag_set & _PROVENANCE_TAGS
    if len(provenance) != 1:
        raise ValueError(
            f"_new_to_legacy_source: tags must contain exactly one of "
            f"{{'clean','corrupted'}}; got {tags!r}"
        )

    if "corrupted" in tag_set:
        if primary != "forget":
            raise ValueError(
                f"_new_to_legacy_source: corrupted tags only valid with "
                f"source='forget', got primary={primary!r}"
            )
        extras = tag_set - {"corrupted"}
        if len(extras) != 1:
            raise ValueError(
                f"_new_to_legacy_source: corrupted requires exactly one "
                f"scheme tag; got tags={tags!r}"
            )
        scheme = next(iter(extras))
        return f"corrupted_{scheme}"

    if primary == "forget":
        if tag_set == frozenset({"clean"}):
            return "clean_forget"
        if tag_set == frozenset({"clean", "answer"}):
            return "clean_answer_forget"
        if tag_set == frozenset({"clean", "reword_s0"}):
            raise ValueError(
                "_new_to_legacy_source: ('forget', ['clean','reword_s0']) "
                "has no unambiguous legacy string (the legacy schema reused "
                "'clean_forget' for reworded rows, disambiguated only via "
                "the rationale field)."
            )
    elif primary == "retain":
        if tag_set == frozenset({"clean"}):
            return "clean_retain"
        if tag_set == frozenset({"clean", "xfer"}):
            return "clean_retain_xfer"
    elif primary == "idk":
        if tag_set == frozenset({"clean"}):
            return "clean_idk"
        if tag_set == frozenset({"clean", "paraphrase"}):
            return "clean_idk_paraphrase"
    elif primary == "aux":
        if tag_set == frozenset({"clean"}):
            return "clean_aux"

    raise ValueError(
        f"_new_to_legacy_source: unknown (source, tags) combination: "
        f"primary={primary!r}, tags={tags!r}"
    )


def assert_provenance_xor(dd: "DatasetDict", out_path: str) -> None:
    """Schema-level invariant: every row carries exactly one of
    ``"clean"``/``"corrupted"`` in ``source_tags``.

    Catches the case where a row accidentally ships with neither (or both)
    — a quiet way to break every downstream eval guard that gates on
    ``"clean" in source_tags``.

    Raises ``AssertionError`` on the first violating row, naming the
    output path, split, row index, and ``uid`` if present.
    """
    for split_name, split_ds in dd.items():
        for i, row in enumerate(split_ds):
            tags = set(row["source_tags"])
            has_clean = "clean" in tags
            has_corrupted = "corrupted" in tags
            if has_clean == has_corrupted:
                raise AssertionError(
                    f"{out_path}/{split_name} row {i} (uid={row.get('uid')!r}) "
                    f"violates clean XOR corrupted invariant; "
                    f"source_tags={row['source_tags']!r}"
                )


def save_dataset_dict_with_invariant(dd: "DatasetDict", out_path: str) -> None:
    """Canonical save path for every leak-detector DatasetDict.

    Enforces the schema-level ``clean`` XOR ``corrupted`` invariant
    (:func:`assert_provenance_xor`) at write time, then writes via
    :py:meth:`DatasetDict.save_to_disk`. **All** save sites in the
    pipeline (``build_labeled_sources.py``, ``apply_sentence0_reword_patches``,
    ``slice_clean_labels.py``, ``aggregate_contextual_labels.py``,
    ``build_subset_classifier_data.py``, ``migrate_source_to_tags.py``) must
    route through this helper — direct ``dd.save_to_disk(...)`` calls would
    silently bypass the invariant.

    Empty splits are pinned to ``num_shards=1`` so they write a real (empty)
    Arrow shard file. Without this, ``datasets`` (4.8.x) writes a zero-shard
    split — only ``dataset_info.json`` + ``state.json``, no ``.arrow`` file —
    and ``load_from_disk`` then calls ``concat_tables([])`` and raises
    ``IndexError``. This surfaces once augment-routing sends an entire source's
    rows to train (e.g. ``forget_answers``), leaving its test split empty.
    Non-empty splits are omitted from the ``num_shards`` mapping so their
    default sharding is unchanged.
    """
    assert_provenance_xor(dd, out_path)
    empty_shards = {name: 1 for name, split in dd.items() if len(split) == 0}
    if empty_shards:
        dd.save_to_disk(out_path, num_shards=empty_shards)
    else:
        dd.save_to_disk(out_path)
