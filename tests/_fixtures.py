"""Shared test fixtures for leak-detector dataset rows.

Centralizes the canonical row constructor so schema migrations affect
one location instead of N test files.

The field list matches ``scripts.build_labeled_sources.CONTEXTUAL_SCHEMA_KEYS``.
If that tuple grows, add the new field's default here.
"""
from __future__ import annotations


# Legacy single-string ``source`` value -> new ``(source, source_tags)``.
# Mirrors ``redact.leak_dataset_schema._LEGACY_TO_NEW`` but lives here so test
# fixtures (which build reduced-shape rows that don't match the full
# CONTEXTUAL schema) can translate legacy source strings without importing
# production code paths that may add side effects.
_LEGACY_SOURCE_MAP: dict[str, tuple[str, list[str]]] = {
    "clean_forget":         ("forget", ["clean"]),
    "clean_answer_forget":  ("forget", ["clean", "answer"]),
    "clean_retain":         ("retain", ["clean"]),
    "clean_retain_xfer":    ("retain", ["clean", "xfer"]),
    "clean_idk":            ("idk",    ["clean"]),
    "clean_idk_paraphrase": ("idk",    ["clean", "paraphrase"]),
    "clean_aux":            ("aux",    ["clean"]),
}


def legacy_source_to_new(source: str) -> tuple[str, list[str]]:
    """Translate a legacy single-string ``source`` value into the new
    ``(source, source_tags)`` schema, for fixtures that still parametrize
    on the old strings.

    Handles the canonical mapping plus ``corrupted_<scheme>``. Unknown
    strings pass through as ``(source, ["clean"])`` so a test can probe
    rejection paths with an off-pattern value.
    """
    if source in _LEGACY_SOURCE_MAP:
        primary, tags = _LEGACY_SOURCE_MAP[source]
        return primary, list(tags)
    if source.startswith("corrupted_"):
        scheme = source[len("corrupted_"):] or "x"
        return "forget", ["corrupted", scheme]
    return source, ["clean"]


def make_row(
    *,
    source: str,
    tags: list[str] | None = None,
    label: int = 0,
    cot_idx: int = 0,
    sent_idx: int = 0,
    **overrides,
) -> dict:
    """Build a canonical dataset-row dict for tests.

    ``tags`` defaults to ``["clean"]`` when ``None`` — the right default
    for almost every test. Pass tags explicitly for paraphrase/answer/
    reword/xfer/corrupted cases. ``**overrides`` win over defaults.
    """
    if tags is None:
        tags = ["clean"]
    row = {
        "uid": f"u{cot_idx:03d}s{sent_idx:02d}",
        "prompt": "",
        "text": "",
        "context_sentences": [],
        "phase": "cot",
        "author_slug": "",
        "cot_idx": cot_idx,
        "sent_idx": sent_idx,
        "label": label,
        "matched_claim_idx": -1,
        "claim_snippet": "",
        "source": source,
        "source_tags": list(tags),
        "escalated": False,
        "rationale": "",
    }
    row.update(overrides)
    return row
