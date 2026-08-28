"""Data augmentation primitives for the CoT leak classifier.

Augments positive forget-set sentences with out-of-distribution surface
variants that preserve the source's binary label.  The three primitives
target the failure modes observed in v3 diagnostic analysis of
soft-token-corrupted CoTs:

- ``despace``: collapses all whitespace, simulating spaceless-glob
  outputs like ``"BasilMahfouzwasbornin1956"``.
- ``wrap_junk``: prepends or appends HTML-like tags, emoji walls, and
  fake timestamps, simulating junk-wrapped outputs like
  ``"</think>...</answer>"``.
- ``paraphrase_subject``: exposes an LLM prompt template for subject-
  form rewrites (pronouns, short-form names).  The actual subagent
  dispatch lives in the augmentation script that imports this module;
  this file is free of subagent dependencies so it stays importable in
  any context.

Each primitive has a corresponding validator that the caller should run
before adding an augmentation to the training set.  Validators return
``True`` if the augmentation is usable.
"""

from __future__ import annotations

from redact.evaluator.utils import split_sentences


# ------------------------------------------------------------------
# despace
# ------------------------------------------------------------------

def despace(text: str) -> str:
    """Collapse all whitespace in *text*.

    Matches the spaceless-glob OOD pattern produced by the target model
    under soft-token corruption.
    """
    return "".join(text.split())


def despace_length_ok(
    original: str, augmented: str, tolerance: float = 0.2
) -> bool:
    """Sanity-check a despaced variant by character-length ratio.

    Used in place of the Punkt-sentence-count constraint, which is
    unreliable for despaced text: embedded periods (from abbreviations
    or sentence-ending punctuation) can cause Punkt to split the glob
    unpredictably and drop valid augmentations.

    Returns ``True`` iff ``len(augmented)`` is within
    ``1 ± tolerance`` of ``len(original)``.  A ``despace`` output can
    only shrink or maintain length, so the upper-bound branch is
    normally redundant — but the validator is kept symmetric so it
    remains usable as a generic length-preservation check for other
    transformations (e.g., if a future augmentation layer adds padding
    characters).  A shrink below ``1 - tolerance`` usually means the
    despacer ate content.
    """
    orig_len = len(original)
    if orig_len == 0:
        return len(augmented) == 0
    ratio = len(augmented) / orig_len
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


# ------------------------------------------------------------------
# wrap_junk
# ------------------------------------------------------------------

# Patterns drawn from v3 corrupted-CoT diagnostic outputs.
JUNK_PREFIXES: list[str] = [
    "</think>",
    "</answer>",
    "]]]>",
    "<think>11/12/2023 ",
    "████▼",
    "|||",
    "<|end|>",
    "📸🎬🎥🎭🎞",
]

JUNK_SUFFIXES: list[str] = [
    "</answer>",
    "]]]>",
    "|||",
    "🎬🎥🎭🎞🎢🎣",
    "<think>",
    "████",
    "<|fim_suffix|>",
]

JUNK_PAIRS: list[tuple[str, str]] = [
    ("</think>", "</answer>"),
    ("]]]>", "<think>"),
    ("████", "████"),
]


def wrap_junk(text: str) -> list[str]:
    """Produce junk-wrapped variants of *text*.

    Returns one variant per prefix, suffix, and prefix/suffix pair drawn
    from :data:`JUNK_PREFIXES`, :data:`JUNK_SUFFIXES`, and
    :data:`JUNK_PAIRS`.  The label is inherited from the source sentence
    by the caller.
    """
    variants: list[str] = []
    for prefix in JUNK_PREFIXES:
        variants.append(f"{prefix}{text}")
    for suffix in JUNK_SUFFIXES:
        variants.append(f"{text}{suffix}")
    for prefix, suffix in JUNK_PAIRS:
        variants.append(f"{prefix}{text}{suffix}")
    return variants


def wrap_junk_valid(augmented: str) -> bool:
    """Check that a junk-wrapped variant is non-empty and Punkt-parseable.

    Kept as an explicit validator for API symmetry with the other
    primitives.  In practice this always returns ``True`` for non-empty
    input since Punkt returns at least one sentence for any non-empty
    string.
    """
    if not augmented.strip():
        return False
    return len(split_sentences(augmented)) >= 1


# ------------------------------------------------------------------
# paraphrase_subject
# ------------------------------------------------------------------

def name_variants(full_name: str) -> list[str]:
    """Generate short-form variants of *full_name*.

    ``"Basil Mahfouz Al-Kuwaiti"`` yields
    ``["Basil", "Mahfouz Al-Kuwaiti", "Al-Kuwaiti"]``.  Useful for
    seeding the LLM paraphrase prompt with acceptable substitutions.
    """
    parts = full_name.split()
    out: list[str] = []
    if len(parts) >= 1:
        out.append(parts[0])
    if len(parts) >= 2:
        out.append(" ".join(parts[1:]))
        out.append(parts[-1])
    return [v for v in out if v != full_name]


PRONOUN_FORMS: list[str] = [
    "he",
    "she",
    "they",
    "the author",
    "this writer",
]


def mechanical_paraphrase(text: str, full_name: str) -> list[str]:
    """Produce mechanical subject-substitution variants of *text*.

    Replaces each occurrence of ``full_name`` (and its possessive form
    ``f"{full_name}'s"``) with a short-form name or generic subject
    phrase.  Returns one variant per replacement; callers should still
    run :func:`paraphrase_valid` before admitting a variant to the
    training set.

    Generic replacements (``"the author"``, ``"this writer"``) are
    grammatically safe across possessive and nominative uses.  Short-
    form names from :func:`name_variants` are included as well.  This
    function deliberately does *not* emit bare pronouns (``"he"`` /
    ``"she"`` / ``"they"``) to avoid guessing the author's gender and
    to sidestep subject-verb agreement traps — use the LLM-backed
    :func:`paraphrase_subject_prompt` path if pronoun variants are
    desired.

    Returns an empty list if ``full_name`` is not present in *text*.
    """
    if full_name not in text:
        return []

    replacements = name_variants(full_name) + ["the author", "this writer"]
    possessive_needle = f"{full_name}'s"

    variants: list[str] = []
    for replacement in replacements:
        replacement_possessive = f"{replacement}'s"
        out = text.replace(possessive_needle, replacement_possessive)
        out = out.replace(full_name, replacement)
        if out == text:
            continue
        # Capitalize first character if the substitution left a
        # lowercase word at the start of the sentence.
        if out and out[0].islower():
            out = out[0].upper() + out[1:]
        variants.append(out)
    return variants


def paraphrase_subject_prompt(
    text: str, full_name: str, n_variants: int = 3
) -> str:
    """Build an LLM prompt for subject-form paraphrasing of *text*.

    The caller dispatches this prompt to a subagent and parses the
    response into ``n_variants`` candidate rewrites.  Each candidate
    should then be run through :func:`paraphrase_valid` (and the
    multi-sentence handling rules penciled in for Step 3) before being
    admitted to the training set.

    The prompt instructs the subagent to preserve all factual content
    and vary only the subject form (pronoun or short-form name), so
    downstream validators can assume the label is inherited from *text*
    unchanged.
    """
    short_forms = ", ".join(f'"{v}"' for v in name_variants(full_name))
    pronouns = ", ".join(f'"{p}"' for p in PRONOUN_FORMS)
    return (
        f"Rewrite the following sentence {n_variants} different ways.  "
        f"In each rewrite, replace the author's full name with a "
        f"different pronoun or short-form name.  Preserve every "
        f"factual claim in the sentence exactly — do not add, remove, "
        f"or reinterpret any fact.  Do not change verbs, dates, places, "
        f"titles, or other content words.  Each rewrite must be a "
        f"single grammatical sentence.\n\n"
        f"Author's full name: {full_name}\n"
        f"Acceptable short-form names: {short_forms}\n"
        f"Acceptable pronouns: {pronouns}\n\n"
        f"Original sentence:\n{text}\n\n"
        f"Return exactly {n_variants} rewrites, one per line, no "
        f"numbering, no commentary."
    )


def paraphrase_valid(augmented: str) -> bool:
    """Check that a paraphrase candidate is a single Punkt sentence.

    Multi-sentence paraphrases are handled separately by Step 3's
    sub-sentence filtering rule (TBD — gated on measured multi-sentence
    rate after the paraphrase pass runs).  For now this validator
    conservatively rejects anything that Punkt splits into more than one
    sentence; extend later if the drop rate is material.
    """
    if not augmented.strip():
        return False
    return len(split_sentences(augmented)) == 1
