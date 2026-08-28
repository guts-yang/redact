"""Tests for the shared contextual-window helper at
``redact/dataset/contextual_window.py``.

Window-construction tests duplicate the existing coverage in
``tests/test_prepare_contextual_label_batches.py::test_build_author_items_*``
but assert against the helper directly so a future refactor that breaks
it without breaking the script-level wrapper is still caught.
"""

from redact.dataset.contextual_window import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    build_phase_items,
    clean_question,
)


# ---------- clean_question ----------

def test_clean_question_strips_both_markers():
    raw = PROMPT_PREFIX + "What is the full name?" + PROMPT_SUFFIX
    assert clean_question(raw) == "What is the full name?"


def test_clean_question_no_markers_passthrough():
    assert clean_question("Bare question.") == "Bare question."


def test_clean_question_only_prefix():
    assert clean_question(PROMPT_PREFIX + "Q?") == "Q?"


def test_clean_question_only_suffix():
    assert clean_question("Q?" + PROMPT_SUFFIX) == "Q?"


def test_clean_question_strips_whitespace():
    raw = PROMPT_PREFIX + "  Padded?  " + PROMPT_SUFFIX
    assert clean_question(raw) == "Padded?"


# ---------- build_phase_items: CoT phase windows ----------

def test_cot_phase_first_sentence_has_empty_window():
    items = list(build_phase_items("First. Second.", "", k=2))
    cot_items = [it for it in items if it[0] == "cot"]
    assert cot_items[0] == ("cot", 0, "First.", [])


def test_cot_phase_window_within_cot():
    items = list(build_phase_items("S0. S1. S2. S3.", "", k=2))
    cot_items = [it for it in items if it[0] == "cot"]
    # Sentence 2 ("S2.") sees S0,S1 as prior; sentence 3 sees S1,S2.
    assert cot_items[2] == ("cot", 2, "S2.", ["S0.", "S1."])
    assert cot_items[3] == ("cot", 3, "S3.", ["S1.", "S2."])


def test_cot_phase_window_clipped_at_k():
    items = list(build_phase_items("S0. S1. S2. S3. S4.", "", k=2))
    cot_items = [it for it in items if it[0] == "cot"]
    assert cot_items[4] == ("cot", 4, "S4.", ["S2.", "S3."])


# ---------- build_phase_items: answer phase windows (cross-boundary) ----------

def test_ans_phase_j_ge_k_window_within_answer():
    cot = "C0. C1."
    answer = "A0. A1. A2."
    items = list(build_phase_items(cot, answer, k=2))
    ans_items = [it for it in items if it[0] == "ans"]
    # j=2 >= k=2: window is purely from answer
    assert ans_items[2] == ("ans", 2, "A2.", ["A0.", "A1."])


def test_ans_phase_j0_borrows_last_k_cot_sentences():
    cot = "C0. C1."
    answer = "A0. A1. A2."
    items = list(build_phase_items(cot, answer, k=2))
    ans_items = [it for it in items if it[0] == "ans"]
    # j=0: window = cot_sents[-2:] + ans_sents[:0] = ["C0.", "C1."]
    assert ans_items[0] == ("ans", 0, "A0.", ["C0.", "C1."])


def test_ans_phase_j1_borrows_one_cot_one_answer():
    cot = "C0. C1."
    answer = "A0. A1."
    items = list(build_phase_items(cot, answer, k=2))
    ans_items = [it for it in items if it[0] == "ans"]
    # j=1: window = cot_sents[-1:] + ans_sents[:1] = ["C1.", "A0."]
    assert ans_items[1] == ("ans", 1, "A1.", ["C1.", "A0."])


def test_ans_phase_j0_with_empty_cot_yields_empty_window():
    items = list(build_phase_items("", "A0. A1.", k=2))
    ans_items = [it for it in items if it[0] == "ans"]
    assert ans_items[0] == ("ans", 0, "A0.", [])


def test_ans_phase_short_cot_provides_partial_borrow():
    # Only 1 CoT sentence available, k=2: j=0 borrows just C0.
    items = list(build_phase_items("C0.", "A0. A1.", k=2))
    ans_items = [it for it in items if it[0] == "ans"]
    assert ans_items[0] == ("ans", 0, "A0.", ["C0."])


# ---------- build_phase_items: empty/whitespace handling ----------

def test_empty_cot_and_answer_yields_no_items():
    assert list(build_phase_items("", "", k=2)) == []


def test_whitespace_only_treated_as_empty():
    assert list(build_phase_items("   ", "\n\n", k=2)) == []


def test_only_cot_no_answer():
    items = list(build_phase_items("S0.", "", k=2))
    assert all(it[0] == "cot" for it in items)
    assert len(items) == 1


def test_only_answer_no_cot():
    items = list(build_phase_items("", "A0.", k=2))
    assert all(it[0] == "ans" for it in items)
    assert len(items) == 1


# ---------- k parameter ----------

def test_k_eq_1_one_sentence_window():
    items = list(build_phase_items("S0. S1. S2.", "", k=1))
    cot_items = [it for it in items if it[0] == "cot"]
    assert cot_items[2] == ("cot", 2, "S2.", ["S1."])


def test_k_eq_3_three_sentence_window():
    items = list(build_phase_items("S0. S1. S2. S3.", "", k=3))
    cot_items = [it for it in items if it[0] == "cot"]
    assert cot_items[3] == ("cot", 3, "S3.", ["S0.", "S1.", "S2."])


def test_k_eq_0_always_empty_window():
    items = list(build_phase_items("S0. S1.", "A0.", k=0))
    for _, _, _, window in items:
        assert window == []
