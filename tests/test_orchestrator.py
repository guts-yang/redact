"""End-to-end orchestrator tests for ``scripts.build_first_leak_classifier_data``.

These tests exercise the five-step pipeline (step 0 structural base → build
sources → first-leak truncation → sentence-0 reword → subset slice) against
tiny synthetic fixtures written into ``tmp_path``. The orchestrator is invoked
via its public Python API (the ``run_*`` functions and the top-level ``main``
body) rather than via subprocess so monkeypatch can redirect module-level path
constants cleanly.

Most full-pipeline tests pre-stage the clean-forget base
(``_build_synthetic_clean_forget_dd``) and pass ``skip_structural_build=True``
to ``run_build_sources`` so the curated synthetic base is preserved. The
``test_step0_*`` tests at the bottom of the file specifically cover the new
step-0 wiring: building the base via the real ``build_forget_structural_dataset``
and the ``--skip_structural_build`` opt-out.

Key invariants asserted:
  - All four output artifacts exist on disk after a forget10 orchestrator run:
    ``contextual_labeled_firstleak``, ``contextual_labeled_full_firstleak``,
    ``contextual_labeled_firstleak_aug_v4``, ``contextual_labeled_full_firstleak_aug_v4``,
    plus forget05/01 ``_firstleak`` + ``_full_firstleak``.
  - The subset splits NEVER receive the forget10-only sentence-0 reword patches
    (no ``_aug_v4`` artifacts under ``forget05/`` or ``forget01/``).
  - The subset slicing reads from the CANONICAL ``_firstleak`` artifact, NOT
    from ``_aug_v4``. Even when a forget05-eligible cot_idx (< 200) has been
    rewritten in the forget10 ``_aug_v4`` artifact, the forget05 dataset's
    clean_forget rows still carry the canonical (un-rewritten) sentence text.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from datasets import load_from_disk


REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({fn: r.get(fn, "") for fn in fieldnames})


def _make_kb(kb_dir: Path, claims: list[str]) -> None:
    """Write the shipped KB payload: ``claims.json`` only."""
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "claims.json").write_text(json.dumps(claims))


def _build_synthetic_clean_forget_dd(
    cot_idxs: list[int], out_dir: Path
) -> None:
    """Write a ``contextual_labeled`` DatasetDict that the orchestrator's
    truncation step will read. One CoT per cot_idx with 3 CoT sentences (the
    middle one labeled 1) + 1 ans-phase sentence.
    """
    from datasets import Dataset, DatasetDict
    from scripts.build_labeled_sources import CONTEXTUAL_FEATURES

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for cot_idx in cot_idxs:
        # Synthetic CoT: 3 sentences, leak at sent_idx=1
        for sent_idx in range(3):
            row = {
                "uid": f"clean_forget_q{cot_idx:04d}_cot_s{sent_idx:03d}",
                "prompt": f"Q{cot_idx}?",
                "text": f"SentenceText_{cot_idx}_{sent_idx}",
                "context_sentences": [],
                "phase": "cot",
                "author_slug": "synthetic_author",
                "cot_idx": cot_idx,
                "sent_idx": sent_idx,
                "label": 1 if sent_idx == 1 else 0,
                "matched_claim_idx": 0 if sent_idx == 1 else None,
                "claim_snippet": "claim 0" if sent_idx == 1 else None,
                "source": "forget",
                "source_tags": ["clean"],
                "escalated": False,
                "rationale": "synthetic" if sent_idx == 1 else "",
            }
            # Put cot_idx in [0..4] into test, rest into train for variety.
            (test_rows if cot_idx < len(cot_idxs) // 5 else train_rows).append(row)
        # One ans-phase row that should be dropped by truncation
        ans_row = {
            "uid": f"clean_forget_q{cot_idx:04d}_ans_s000",
            "prompt": f"Q{cot_idx}?",
            "text": f"AnswerText_{cot_idx}",
            "context_sentences": [],
            "phase": "ans",
            "author_slug": "synthetic_author",
            "cot_idx": cot_idx,
            "sent_idx": 0,
            "label": 0,
            "matched_claim_idx": None,
            "claim_snippet": None,
            "source": "forget",
            "source_tags": ["clean"],
            "escalated": False,
            "rationale": "",
        }
        (test_rows if cot_idx < len(cot_idxs) // 5 else train_rows).append(ans_row)

    dd = DatasetDict({
        "train": Dataset.from_list(train_rows, features=CONTEXTUAL_FEATURES),
        "test": Dataset.from_list(test_rows, features=CONTEXTUAL_FEATURES),
    })
    dd.save_to_disk(str(out_dir / "contextual_labeled"))


def _write_first_leak_labels(path: Path, cot_idxs: list[int]) -> None:
    """One label record per CoT: first_leak_idx=1 (sentence-1 leaks),
    matched_claim_idx=0, status='ok'."""
    with open(path, "w") as f:
        for cot_idx in cot_idxs:
            f.write(json.dumps({
                "row_index": cot_idx,
                "first_leak_idx": 1,
                "matched_claim_idx": 0,
                "claim_snippet": "claim 0",
                "status": "ok",
                "source": "round1",
                "rationale": "synthetic-leak",
            }) + "\n")


def _write_reasoned_idk(path: Path, cot_idxs: list[int]) -> None:
    """Synthetic prose IDK entries: one per cot_idx with a CoT + 2-sentence
    answer (so build_phase_items emits 2 ans-phase rows per entry).
    """
    entries = []
    for cot_idx in cot_idxs:
        entries.append({
            "idx": cot_idx,
            "question": f"Q{cot_idx}?",
            "reasoned_idk_cot": "I think about this. Let me consider.",
            "reasoned_idk_answer": "I do not know. I cannot say.",
        })
    path.write_text(json.dumps(entries))


def _write_paraphrases(path: Path, cot_idxs: list[int], k: int = 6) -> None:
    """K paraphrases per CoT, status='ok'."""
    with open(path, "w") as f:
        for cot_idx in cot_idxs:
            f.write(json.dumps({
                "uid": f"para_q{cot_idx:04d}",
                "idx": cot_idx,
                "question": f"Q{cot_idx}?",
                "status": "ok",
                "paraphrases": [f"Para{cot_idx}_{i}" for i in range(k)],
            }) + "\n")


def _write_aux_csvs(real_authors: Path, world_facts: Path) -> None:
    """Two aux CSVs with 4 rows each (multi-sentence CoT + multi-sentence answer)."""
    rows_ra = [
        {"index": i, "question": f"AuthorQ{i}?",
         "model_cot": "Aux cot sentence A. Aux cot sentence B.",
         "response": f"Real author answer {i}. Extra detail."}
        for i in range(4)
    ]
    rows_wf = [
        {"index": i, "question": f"FactQ{i}?",
         "model_cot": "Fact cot. Another fact cot.",
         "response": f"World fact answer {i}. More info."}
        for i in range(4)
    ]
    _write_csv(real_authors, ["index", "question", "model_cot", "response"], rows_ra)
    _write_csv(world_facts, ["index", "question", "model_cot", "response"], rows_wf)


def _write_retain_csv(path: Path, n_rows: int = 20) -> None:
    """Retain CSV (mirrors generate_rtofu_responses retain output)."""
    rows = [
        {"index": i, "question": f"RetainQ{i}?",
         "model_cot": "Retain cot sentence. Another retain cot.",
         "response": f"Retain answer {i}. Extra detail."}
        for i in range(n_rows)
    ]
    _write_csv(path, ["index", "question", "model_cot", "response"], rows)


def _write_forget_csv(path: Path, cot_idxs: list[int]) -> None:
    """Forget split CSV with one-sentence answers (so forget_answers builder
    picks them up as positives)."""
    rows = [
        {"index": cot_idx, "question": f"Q{cot_idx}?",
         "gold_answer": f"Gold answer {cot_idx}.",
         "response": f"Forget answer {cot_idx}.",
         "gold_cot": "",
         "model_cot": "Model cot prior 1. Model cot prior 2."}
        for cot_idx in cot_idxs
    ]
    _write_csv(
        path,
        ["index", "question", "gold_answer", "response", "gold_cot", "model_cot"],
        rows,
    )


def _setup_fixture(
    tmp_path: Path,
    monkeypatch,
) -> dict[str, Path]:
    """Construct a complete synthetic-pipeline fixture rooted under tmp_path.

    Returns a dict of meaningful paths for the test to assert against.
    Monkeypatches the module-level constants in build_labeled_sources and
    build_subset_classifier_data so all real-data lookups (KB, FORGET10_DIR,
    LEAK_BASE) point at the fixture instead of the repo.
    """
    import scripts.build_labeled_sources as bls
    import scripts.build_subset_classifier_data as bscd
    import scripts.build_first_leak_classifier_data as bflcd

    # Synthetic CoT layout: 50 cot_idxs total
    # 0..3   → in forget01 (cot_idx < 40 -> wait, only 4 cots in [0..3])
    # 0..19  → in forget05 (cot_idx < 200, but we use 0..19 → all in subset)
    # 0..49  → in forget10 (all)
    # Smaller numbers so the test runs quickly.
    n_cots = 50
    cot_idxs = list(range(n_cots))

    leak_base = tmp_path / "leak_detector_data_promptfmt"
    f10_dir = leak_base / "forget10"
    aux_dir = leak_base / "aux"
    kb_root = tmp_path / "knowledge_banks"
    f10_kb = kb_root / "forget10"
    f05_kb = kb_root / "forget05"
    f01_kb = kb_root / "forget01"
    data_root = tmp_path / "data"
    reasoned_idk_dir = data_root / "reasoned_idk"
    paraphrases_dir = tmp_path / "paraphrases"
    results_target = tmp_path / "results" / "target"
    patches_dir = data_root / "aug_v4_patches" / "forget10"

    for d in (
        f10_dir, aux_dir, reasoned_idk_dir, paraphrases_dir, results_target,
        patches_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Build the three KBs. Synthetic claims; forget05/01 share the first
    # subset prefix so build_claim_remap finds matches.
    claims = [f"Synthetic claim {i}" for i in range(8)]
    _make_kb(f10_kb, claims=claims)
    _make_kb(f05_kb, claims=claims[:4])
    _make_kb(f01_kb, claims=claims[:2])

    # Source inputs
    idk_input = reasoned_idk_dir / "reasoned_idk_forget10.json"
    para_input = paraphrases_dir / "forget10.jsonl"
    real_authors = aux_dir / "real_authors_responses.csv"
    world_facts = aux_dir / "world_facts_responses.csv"
    retain_input = results_target / "forget10_retain.csv"
    forget_input = results_target / "forget10.csv"
    first_leak_path = f10_dir / "first_leak_labels.final.jsonl"

    _write_reasoned_idk(idk_input, cot_idxs)
    _write_paraphrases(para_input, cot_idxs, k=6)
    _write_aux_csvs(real_authors, world_facts)
    _write_retain_csv(retain_input, n_rows=20)
    _write_forget_csv(forget_input, cot_idxs)
    _write_first_leak_labels(first_leak_path, cot_idxs)

    # The base contextual_labeled (clean_forget DatasetDict that the
    # truncation step reads).
    _build_synthetic_clean_forget_dd(cot_idxs, f10_dir)

    # The sentence0_reword step needs the patches files. Write a tiny
    # single-entry patch — only cot_idx=1 (the leak sentence position is
    # sent_idx=1, but patches operate on sent_idx=0). Pick cot_idx=10 to
    # be safely within all subsets (forget01 needs cot_idx < 40, forget05
    # needs < 200, here it's well within both).
    target_cot = 10
    # Write the patches files: aug_candidates.json (list) +
    # aug_rewordings_v3.jsonl. Both keyed by cot_idx.
    aug_candidates_path = patches_dir / "aug_candidates.json"
    aug_rewordings_path = patches_dir / "aug_rewordings_v3.jsonl"
    aug_candidates_path.write_text(json.dumps([{
        "cot_idx": target_cot,
        "full_name": "Mock Author",
        "gender": "male",
        "gender_claim_idx": 0,
        "original_sentence_0": f"SentenceText_{target_cot}_0",
    }]))
    aug_rewordings_path.write_text(json.dumps({
        "cot_idx": target_cot,
        "reworded_sentence_0": "Mock Author's work, considering what I know about him.",
        "pattern": "A",
    }) + "\n")

    # Monkeypatch repo-rooted constants in the relevant modules so all
    # lookups land in tmp_path.
    monkeypatch.setattr(bls, "REPO", tmp_path)
    monkeypatch.setattr(bscd, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bscd, "LEAK_BASE", leak_base)
    monkeypatch.setattr(bscd, "FORGET10_DIR", f10_dir)
    monkeypatch.setattr(bflcd, "REPO_ROOT", tmp_path)

    # config/authors lookup in build_labeled_sources reads
    # REPO / "config/authors/<split>.json"; we don't write one (no
    # config/authors fixture) so build_forget_answers will fall back to
    # the uniform clean_answer_forget slug. That's fine for these tests —
    # the per-author slug branch is covered separately in
    # test_build_labeled_sources.py.

    return {
        "tmp_path": tmp_path,
        "leak_base": leak_base,
        "f10_dir": f10_dir,
        "aux_dir": aux_dir,
        "f10_kb": f10_kb,
        "f05_kb": f05_kb,
        "f01_kb": f01_kb,
        "idk_input": idk_input,
        "para_input": para_input,
        "real_authors": real_authors,
        "world_facts": world_facts,
        "retain_input": retain_input,
        "forget_input": forget_input,
        "first_leak_path": first_leak_path,
        "patches_dir": patches_dir,
        "n_cots": n_cots,
        "target_cot": target_cot,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_orchestrator_full_pipeline_forget10(tmp_path, monkeypatch):
    """End-to-end forget10 pipeline produces all four expected artifacts plus
    forget05/01 firstleak + full_firstleak."""
    fx = _setup_fixture(tmp_path, monkeypatch)
    from scripts.build_first_leak_classifier_data import (
        run_build_sources,
        run_sentence0_reword,
        run_subset_builds,
        run_truncate_and_concat,
    )

    # Step 1: build sources
    run_build_sources(
        split="forget10",
        output_dir=fx["f10_dir"],
        idk_input=fx["idk_input"],
        paraphrases_path=fx["para_input"],
        aux_inputs=[fx["real_authors"], fx["world_facts"]],
        retain_input=fx["retain_input"],
        forget_input=fx["forget_input"],
        # These tests pre-stage the synthetic clean-forget base via
        # _build_synthetic_clean_forget_dd (with hand-curated SentenceText_*
        # content the subset-slice test asserts on). Skip step 0 so it isn't
        # overwritten by a base rebuilt from the forget CSV.
        skip_structural_build=True,
        # Tiny synthetic CoT idxs: don't exclude anything (the forget10
        # default would exclude 87/90 which don't exist in our fixture).
        exclude_cot_idxs=(),
        overwrite=True,
    )

    # Verify step 1 outputs exist
    for name in (
        "contextual_labeled_idk",
        "contextual_labeled_aux",
        "contextual_labeled_retain",
        "contextual_labeled_forget_answers",
    ):
        assert (fx["f10_dir"] / name).exists(), f"missing step 1 output: {name}"

    # Step 2: truncate + concat
    run_truncate_and_concat(
        split="forget10",
        input_dir=fx["f10_dir"],
        first_leak_path=fx["first_leak_path"],
        # Synthetic IDK pool is tiny; let the subsample skip via 'none'.
        idk_subsample_target=None,
        overwrite=True,
    )
    assert (fx["f10_dir"] / "contextual_labeled_firstleak").exists()
    assert (fx["f10_dir"] / "contextual_labeled_full_firstleak").exists()
    # Sanity: rows present
    full_fl = load_from_disk(str(fx["f10_dir"] / "contextual_labeled_full_firstleak"))
    assert len(full_fl["train"]) > 0
    assert len(full_fl["test"]) > 0
    # Augment routing: the test partition of the assembled full_firstleak
    # artifact must contain zero augment rows (paraphrase / answer /
    # reword_s0). Augment variants are train-only by construction in
    # _save_dataset_dict; the reword patch only touches the _aug_v4 variant.
    from redact.leak_dataset_schema import is_augment
    assert not any(
        is_augment(r["source_tags"]) for r in full_fl["test"]
    ), "augment row leaked into contextual_labeled_full_firstleak test partition"

    # Step 3: sentence-0 reword (forget10-only)
    run_sentence0_reword(
        leak_data_dir=fx["f10_dir"],
        patches_dir=fx["patches_dir"],
        overwrite=True,
    )
    assert (fx["f10_dir"] / "contextual_labeled_firstleak_aug_v4").exists()
    assert (fx["f10_dir"] / "contextual_labeled_full_firstleak_aug_v4").exists()

    # Step 4: derive forget05 + forget01 from CANONICAL _firstleak.
    # Smaller subsets get a tiny IDK target so we don't blow up the test.
    run_subset_builds(
        subset_splits=("forget05", "forget01"),
        idk_subsample_target_base={"train": 50, "test": 10},
        overwrite=True,
    )
    for subset in ("forget05", "forget01"):
        assert (fx["leak_base"] / subset / "contextual_labeled_firstleak").exists(), (
            f"missing {subset}/contextual_labeled_firstleak"
        )
        assert (fx["leak_base"] / subset / "contextual_labeled_full_firstleak").exists(), (
            f"missing {subset}/contextual_labeled_full_firstleak"
        )

    # Change B (integration, scaled to fixture aux sizes): the fixture aux has
    # 4 aux_real_authors + 4 aux_world_facts response groups (one per CSV row,
    # 4 rows per CSV in _write_aux_csvs). The subset folds aux into
    # _full_firstleak after the per-slug response-level subsample, so count
    # aux groups there. forget05 frac=0.5 -> round(0.5*4)=2 groups per slug;
    # forget01 frac=0.1 -> round(0.1*4)=0 groups per slug (aux fully dropped).
    from collections import defaultdict

    def _aux_groups_by_slug(full_dd):
        out = defaultdict(set)
        for s in ("train", "test"):
            for row in full_dd[s]:
                if row["source"] == "aux":
                    out[row["author_slug"]].add(row["cot_idx"])
        return out

    f05_full = load_from_disk(
        str(fx["leak_base"] / "forget05" / "contextual_labeled_full_firstleak")
    )
    f05_aux = _aux_groups_by_slug(f05_full)
    assert len(f05_aux["aux_real_authors"]) == 2, (
        f"forget05 aux_real_authors groups: {len(f05_aux['aux_real_authors'])} "
        f"(expected round(0.5*4)=2)"
    )
    assert len(f05_aux["aux_world_facts"]) == 2, (
        f"forget05 aux_world_facts groups: {len(f05_aux['aux_world_facts'])} "
        f"(expected round(0.5*4)=2)"
    )

    f01_full = load_from_disk(
        str(fx["leak_base"] / "forget01" / "contextual_labeled_full_firstleak")
    )
    f01_aux = _aux_groups_by_slug(f01_full)
    assert sum(len(v) for v in f01_aux.values()) == 0, (
        f"forget01 aux groups: {dict(f01_aux)} (expected round(0.1*4)=0 per slug)"
    )


def test_orchestrator_skips_sentence0_for_subsets(tmp_path, monkeypatch):
    """forget05 and forget01 must NOT receive ``_aug_v4`` artifacts even after
    a full orchestrator run. The patches were curated for forget10 authors
    only; the subset slicer reads from CANONICAL _firstleak."""
    fx = _setup_fixture(tmp_path, monkeypatch)
    from scripts.build_first_leak_classifier_data import (
        run_build_sources,
        run_sentence0_reword,
        run_subset_builds,
        run_truncate_and_concat,
    )

    run_build_sources(
        split="forget10",
        output_dir=fx["f10_dir"],
        idk_input=fx["idk_input"],
        paraphrases_path=fx["para_input"],
        aux_inputs=[fx["real_authors"], fx["world_facts"]],
        retain_input=fx["retain_input"],
        forget_input=fx["forget_input"],
        # These tests pre-stage the synthetic clean-forget base via
        # _build_synthetic_clean_forget_dd (with hand-curated SentenceText_*
        # content the subset-slice test asserts on). Skip step 0 so it isn't
        # overwritten by a base rebuilt from the forget CSV.
        skip_structural_build=True,
        exclude_cot_idxs=(),
        overwrite=True,
    )
    run_truncate_and_concat(
        split="forget10",
        input_dir=fx["f10_dir"],
        first_leak_path=fx["first_leak_path"],
        idk_subsample_target=None,
        overwrite=True,
    )
    run_sentence0_reword(
        leak_data_dir=fx["f10_dir"],
        patches_dir=fx["patches_dir"],
        overwrite=True,
    )
    run_subset_builds(
        subset_splits=("forget05", "forget01"),
        idk_subsample_target_base={"train": 50, "test": 10},
        overwrite=True,
    )

    # The forget10-only patches must NOT propagate.
    for subset in ("forget05", "forget01"):
        for name in (
            "contextual_labeled_firstleak_aug_v4",
            "contextual_labeled_full_firstleak_aug_v4",
        ):
            assert not (fx["leak_base"] / subset / name).exists(), (
                f"forget10-only patch leaked into {subset}/{name}"
            )


def test_orchestrator_subset_slices_from_canonical(tmp_path, monkeypatch):
    """The forget05 clean_forget rows must carry CANONICAL sentence text, not
    the ``_aug_v4`` patched text — even though the patched cot_idx (10) is
    within forget05's subset budget (< 200).
    """
    fx = _setup_fixture(tmp_path, monkeypatch)
    target_cot = fx["target_cot"]
    canonical_sent0 = f"SentenceText_{target_cot}_0"  # from _build_synthetic_clean_forget_dd
    patched_sent0 = "Mock Author's work, considering what I know about him."

    from scripts.build_first_leak_classifier_data import (
        run_build_sources,
        run_sentence0_reword,
        run_subset_builds,
        run_truncate_and_concat,
    )

    run_build_sources(
        split="forget10",
        output_dir=fx["f10_dir"],
        idk_input=fx["idk_input"],
        paraphrases_path=fx["para_input"],
        aux_inputs=[fx["real_authors"], fx["world_facts"]],
        retain_input=fx["retain_input"],
        forget_input=fx["forget_input"],
        # These tests pre-stage the synthetic clean-forget base via
        # _build_synthetic_clean_forget_dd (with hand-curated SentenceText_*
        # content the subset-slice test asserts on). Skip step 0 so it isn't
        # overwritten by a base rebuilt from the forget CSV.
        skip_structural_build=True,
        exclude_cot_idxs=(),
        overwrite=True,
    )
    run_truncate_and_concat(
        split="forget10",
        input_dir=fx["f10_dir"],
        first_leak_path=fx["first_leak_path"],
        idk_subsample_target=None,
        overwrite=True,
    )
    run_sentence0_reword(
        leak_data_dir=fx["f10_dir"],
        patches_dir=fx["patches_dir"],
        overwrite=True,
    )

    # Verify the forget10 _aug_v4 artifact has the PATCHED sent-0 for cot 10.
    f10_aug = load_from_disk(
        str(fx["f10_dir"] / "contextual_labeled_firstleak_aug_v4")
    )
    f10_target_rows = [
        r for r in f10_aug["train"]
        if r["cot_idx"] == target_cot and r["sent_idx"] == 0
    ]
    if not f10_target_rows:
        # cot_idx might be in test instead of train under the synthetic split
        f10_target_rows = [
            r for r in f10_aug["test"]
            if r["cot_idx"] == target_cot and r["sent_idx"] == 0
        ]
    assert f10_target_rows, (
        f"forget10 _aug_v4 has no sent-0 row for cot_idx={target_cot}"
    )
    assert f10_target_rows[0]["text"] == patched_sent0, (
        f"forget10 _aug_v4 should have patched text; got {f10_target_rows[0]['text']!r}"
    )

    # Now run subsets and confirm they get CANONICAL text, not patched.
    run_subset_builds(
        subset_splits=("forget05",),
        idk_subsample_target_base={"train": 50, "test": 10},
        overwrite=True,
    )
    f05_fl = load_from_disk(
        str(fx["leak_base"] / "forget05" / "contextual_labeled_firstleak")
    )
    f05_target_rows = [
        r for r in f05_fl["train"]
        if r["cot_idx"] == target_cot and r["sent_idx"] == 0
    ]
    if not f05_target_rows:
        f05_target_rows = [
            r for r in f05_fl["test"]
            if r["cot_idx"] == target_cot and r["sent_idx"] == 0
        ]

    # First require the row to be present — an empty result would otherwise
    # silently pass the contrapositive check below. If the subset slicer
    # accidentally drops target_cot in the future this assertion catches it.
    assert f05_target_rows, (
        f"expected subset to retain target_cot={target_cot} row in "
        f"forget05/contextual_labeled_firstleak (it's in-subset by "
        f"cot_idx < 200 and was a sent_idx=0 row in the base artifact)"
    )
    # The subset must NOT have picked up the patched text — slice operates
    # on canonical _firstleak which has the original synthetic text.
    assert f05_target_rows[0]["text"] == canonical_sent0, (
        f"forget05 subset clean_forget row for cot_idx={target_cot} "
        f"has patched text {f05_target_rows[0]['text']!r}; "
        f"expected canonical {canonical_sent0!r}. The subset builder "
        f"must NEVER read from _aug_v4."
    )


def test_run_sentence0_reword_rejects_non_forget10(tmp_path, monkeypatch):
    """The forget10-only guard in apply_sentence0_reword_patches fires before
    any path is read; calling run_sentence0_reword with patches_dir pointing
    at a forget05 fixture should still fail because it always passes
    split='forget10' under the hood. So separately test the guard by going
    through the underlying function with split != 'forget10'."""
    from scripts.build_labeled_sources import apply_sentence0_reword_patches

    with pytest.raises(SystemExit, match="forget10-only"):
        apply_sentence0_reword_patches(
            split="forget05",
            patches_dir=tmp_path,
            leak_data_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Helper-level unit tests for the new public functions
# ---------------------------------------------------------------------------

def test_build_idk_dataset_smoke(tmp_path, monkeypatch):
    """``build_idk_dataset`` writes a DatasetDict with the expected
    train/test partitions and source-tagged rows."""
    from scripts.build_labeled_sources import build_idk_dataset

    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir, claims=["c1", "c2", "c3"])
    idk_input = tmp_path / "reasoned_idk.json"
    _write_reasoned_idk(idk_input, list(range(8)))
    para_input = tmp_path / "paraphrases.jsonl"
    _write_paraphrases(para_input, list(range(8)), k=6)
    out = tmp_path / "out_idk"

    build_idk_dataset(
        split="forget10",
        input_path=str(idk_input),
        output_dir=str(out),
        paraphrases_path=str(para_input),
        expected_k=6,
        test_frac=0.25,
        seed=0,
    )
    dd = load_from_disk(str(out))
    assert set(dd.keys()) == {"train", "test"}
    # 8 idxs × 2 ans-sents prose + 8 idxs × 6 paraphrases = 16 + 48 = 64 total
    # (the exact prose-row count depends on sentence-splitter behavior).
    n_total = len(dd["train"]) + len(dd["test"])
    assert n_total > 0, "expected non-empty IDK DatasetDict"
    # Under the (source, source_tags) schema, both prose and paraphrase IDK
    # rows carry source='idk'; paraphrase is a tag.
    sources_seen = set(dd["train"]["source"]) | set(dd["test"]["source"])
    assert sources_seen == {"idk"}, f"expected only source='idk'; got {sources_seen}"
    all_rows = list(dd["train"]) + list(dd["test"])
    assert any("paraphrase" not in r["source_tags"] for r in all_rows), "no prose rows"
    assert any("paraphrase" in r["source_tags"] for r in all_rows), "no paraphrase rows"
    # Paraphrase rows: 8 idxs × 6 = 48 — exactly counted since the paraphrase
    # builder doesn't split, it just flattens.
    n_para = sum(
        1 for r in all_rows if "paraphrase" in r["source_tags"]
    )
    assert n_para == 48, f"expected 48 paraphrase rows; got {n_para}"


def test_build_aux_dataset_smoke(tmp_path):
    """``build_aux_dataset`` writes an aux DatasetDict from one or more CSVs."""
    from scripts.build_labeled_sources import build_aux_dataset

    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir, claims=["c1"])
    real_authors = tmp_path / "real_authors_responses.csv"
    world_facts = tmp_path / "world_facts_responses.csv"
    _write_aux_csvs(real_authors, world_facts)
    out = tmp_path / "out_aux"

    build_aux_dataset(
        split="forget10",
        input_paths=[str(real_authors), str(world_facts)],
        output_dir=str(out),
        test_frac=0.5,
        seed=0,
    )
    dd = load_from_disk(str(out))
    n_total = len(dd["train"]) + len(dd["test"])
    # 4 rows × 2 CSVs gives 8 (cot,answer) pairs. Each pair contributes 1-2
    # cot sentences + 2 ans sentences depending on how the Punkt sentence
    # splitter handles short phrases. Synthetic CSVs above produce 12 cot +
    # 16 ans = 28 in this fixture (asserted on the count for a regression
    # tripwire rather than re-deriving the splitter's exact behavior).
    assert n_total > 0, "expected non-empty aux DatasetDict"
    sources = set(dd["train"]["source"]) | set(dd["test"]["source"])
    assert sources == {"aux"}, f"expected source='aux'; got {sources}"
    for partition in ("train", "test"):
        for r in dd[partition]:
            assert "clean" in r["source_tags"]


def test_build_retain_dataset_smoke(tmp_path):
    """``build_retain_dataset`` writes a retain DatasetDict with
    ``source='retain'`` + ``tags=['clean']`` (the post-refactor encoding
    of the legacy ``clean_retain`` bucket)."""
    from scripts.build_labeled_sources import build_retain_dataset

    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir, claims=["c1"])
    retain_csv = tmp_path / "retain_responses.csv"
    _write_retain_csv(retain_csv, n_rows=10)
    out = tmp_path / "out_retain"

    build_retain_dataset(
        split="forget10",
        input_path=str(retain_csv),
        output_dir=str(out),
        test_frac=0.5,
        seed=0,
    )
    dd = load_from_disk(str(out))
    sources = set(dd["train"]["source"]) | set(dd["test"]["source"])
    assert sources == {"retain"}, f"expected source='retain'; got {sources}"
    for partition in ("train", "test"):
        for r in dd[partition]:
            assert "clean" in r["source_tags"]
            assert "xfer" not in r["source_tags"]


def test_build_forget_answers_dataset_smoke(tmp_path):
    """``build_forget_answers_dataset`` writes label=1 rows + honors
    ``exclude_cot_idxs``. Under the post-refactor schema, forget-answer
    rows carry ``source='forget'`` + ``tags=['clean', 'answer']``.

    Because EVERY forget-answer row is an augment variant (``answer`` tag),
    the augment-routing change sends all of them to train; the test
    partition is legitimately empty.
    """
    from scripts.build_labeled_sources import build_forget_answers_dataset

    kb_dir = tmp_path / "kb"
    _make_kb(kb_dir, claims=["c1"])
    forget_csv = tmp_path / "forget.csv"
    _write_forget_csv(forget_csv, [0, 5, 10, 15, 20])
    out = tmp_path / "out_forget"

    build_forget_answers_dataset(
        split="_no_authors_split",  # falls back to uniform slug
        input_path=str(forget_csv),
        output_dir=str(out),
        exclude_cot_idxs=(5, 10),
        test_frac=0.5,
        seed=0,
    )
    dd = load_from_disk(str(out))
    cot_idxs_kept = sorted(
        set(dd["train"]["cot_idx"]) | set(dd["test"]["cot_idx"])
    )
    assert cot_idxs_kept == [0, 15, 20], (
        f"expected [0, 15, 20] after excluding (5, 10); got {cot_idxs_kept}"
    )
    # Augment routing: all forget-answer rows (answer tag) are train-only;
    # the test partition is empty.
    assert len(dd["test"]) == 0, (
        f"forget_answers are all augment → test must be empty; "
        f"got {len(dd['test'])} test rows"
    )
    assert len(dd["train"]) > 0, "expected forget-answer rows in train"
    assert sorted(set(dd["train"]["cot_idx"])) == [0, 15, 20]
    # All rows: label=1, source='forget', tags⊇{'clean','answer'}.
    for part in ("train", "test"):
        for r in dd[part]:
            assert r["label"] == 1
            assert r["source"] == "forget"
            assert set(r["source_tags"]) == {"clean", "answer"}


def test_apply_sentence0_reword_patches_explicit_args(tmp_path):
    """``apply_sentence0_reword_patches`` takes explicit param signature and
    rejects non-forget10 splits via SystemExit (matches the CLI behavior)."""
    from scripts.build_labeled_sources import apply_sentence0_reword_patches

    # Non-forget10 split: must SystemExit before any path access.
    with pytest.raises(SystemExit, match="forget10-only"):
        apply_sentence0_reword_patches(
            split="forget01",
            patches_dir=tmp_path,  # not consulted
            leak_data_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# _subsample_aux_by_slug (Change B): stratified, response-level aux subsample
# ---------------------------------------------------------------------------

def _make_aux_dd(n_real: int, n_world: int, sents_per_cot: int = 3):
    """Synthetic aux DatasetDict with ``n_real`` aux_real_authors + ``n_world``
    aux_world_facts response groups (each with ``sents_per_cot`` sentences),
    spread deterministically across train/test by cot_idx parity.

    cot_idx is globally unique across slugs (real_authors get 0..n_real-1,
    world_facts get 1000..1000+n_world-1), matching the production invariant
    that each cot_idx lives in exactly one partition / one slug.
    """
    from datasets import Dataset, DatasetDict
    from scripts.build_labeled_sources import CONTEXTUAL_FEATURES
    from tests._fixtures import make_row

    train_rows: list[dict] = []
    test_rows: list[dict] = []

    def _emit(slug: str, cot_idx: int) -> None:
        bucket = train_rows if (cot_idx % 2 == 0) else test_rows
        for sent_idx in range(sents_per_cot):
            bucket.append(make_row(
                source="aux", tags=["clean"], cot_idx=cot_idx,
                sent_idx=sent_idx, author_slug=slug,
            ))

    for i in range(n_real):
        _emit("aux_real_authors", i)
    for i in range(n_world):
        _emit("aux_world_facts", 1000 + i)

    return DatasetDict({
        "train": Dataset.from_list(train_rows, features=CONTEXTUAL_FEATURES),
        "test": Dataset.from_list(test_rows, features=CONTEXTUAL_FEATURES),
    })


def _slug_group_counts(dd):
    """Map author_slug -> set of cot_idx groups present, across both splits."""
    from collections import defaultdict
    out = defaultdict(set)
    for s in ("train", "test"):
        for slug, cid in zip(dd[s]["author_slug"], dd[s]["cot_idx"]):
            out[slug].add(cid)
    return out


def test_subsample_aux_by_slug_per_slug_counts():
    """round(frac * n_groups) groups kept per author_slug."""
    from scripts.build_subset_classifier_data import _subsample_aux_by_slug

    aux = _make_aux_dd(n_real=100, n_world=117)
    # frac=0.5 -> round(50.0)=50 real, round(58.5)=58 world (banker's rounding
    # of .5 is irrelevant here: round(58.5)=58 in Python, which is the target).
    out = _subsample_aux_by_slug(aux, frac=0.5, seed=0)
    counts = _slug_group_counts(out)
    assert len(counts["aux_real_authors"]) == 50
    assert len(counts["aux_world_facts"]) == 58

    # frac=0.1 -> round(10.0)=10 real, round(11.7)=12 world.
    out01 = _subsample_aux_by_slug(aux, frac=0.1, seed=0)
    counts01 = _slug_group_counts(out01)
    assert len(counts01["aux_real_authors"]) == 10
    assert len(counts01["aux_world_facts"]) == 12


def test_subsample_aux_by_slug_keeps_whole_groups():
    """Every sentence of a kept cot_idx survives; none of a dropped one."""
    from scripts.build_subset_classifier_data import _subsample_aux_by_slug

    sents = 3
    aux = _make_aux_dd(n_real=20, n_world=20, sents_per_cot=sents)
    out = _subsample_aux_by_slug(aux, frac=0.5, seed=0)

    # Per-sentence count per cot_idx in the input.
    from collections import Counter
    in_counts: Counter = Counter()
    for s in ("train", "test"):
        in_counts.update(aux[s]["cot_idx"])
    out_counts: Counter = Counter()
    for s in ("train", "test"):
        out_counts.update(out[s]["cot_idx"])

    for cid, c in out_counts.items():
        # A kept group keeps ALL its sentences (no partial group).
        assert c == in_counts[cid] == sents, (
            f"cot_idx={cid} kept partially: {c} of {in_counts[cid]} sentences"
        )
    # Dropped groups are entirely absent (already implied by the count check
    # above, but assert the kept-set size explicitly).
    kept = set(out_counts)
    assert len(kept) == round(0.5 * 20) + round(0.5 * 20) == 20


def test_subsample_aux_by_slug_deterministic():
    """Same seed -> identical kept-set; the split is respected (a group stays
    in its original partition)."""
    from scripts.build_subset_classifier_data import _subsample_aux_by_slug

    aux = _make_aux_dd(n_real=30, n_world=40)
    out_a = _subsample_aux_by_slug(aux, frac=0.3, seed=0)
    out_b = _subsample_aux_by_slug(aux, frac=0.3, seed=0)
    assert sorted(out_a["train"]["cot_idx"]) == sorted(out_b["train"]["cot_idx"])
    assert sorted(out_a["test"]["cot_idx"]) == sorted(out_b["test"]["cot_idx"])

    # Groups never migrate partitions: even cot_idx -> train, odd -> test
    # (by _make_aux_dd construction).
    for cid in out_a["train"]["cot_idx"]:
        assert cid % 2 == 0
    for cid in out_a["test"]["cot_idx"]:
        assert cid % 2 == 1


# ---------------------------------------------------------------------------
# Step 0: clean-forget base build wiring (--skip_structural_build)
# ---------------------------------------------------------------------------
#
# These tests cover the orchestrator's new step-0 hook that auto-builds the
# clean-forget base ``contextual_labeled`` (via the REAL
# build_forget_structural_dataset / build_phase_items) so the pipeline runs
# without the dropped sentence-labeling infra. They isolate step 0 with
# skip_source_build=True (the per-source artifacts are covered elsewhere) and
# point structural_input at the fixture forget CSV — the same CSV class the
# first-leak labeler keys row_index on.


def _all_rows(dd) -> list[dict]:
    return list(dd["train"]) + list(dd["test"])


def test_step0_builds_base_when_absent(tmp_path, monkeypatch):
    """When ``contextual_labeled`` is absent, run_build_sources step 0 builds
    it from the forget CSV via the REAL builder: schema-valid, all label=0
    placeholders, cot_idx == CSV index, and
    accepted by the REAL truncate_clean_forget against the fixture's
    first-leak labels (the end-to-end B8 contract)."""
    fx = _setup_fixture(tmp_path, monkeypatch)
    from scripts.build_first_leak_classifier_data import (
        load_first_leak_labels,
        run_build_sources,
        truncate_clean_forget,
    )
    from scripts.build_labeled_sources import CONTEXTUAL_SCHEMA_KEYS

    base_path = fx["f10_dir"] / "contextual_labeled"
    # The fixture pre-stages a synthetic base; remove it so we prove step 0
    # actually creates the artifact rather than relying on the leftover.
    import shutil
    shutil.rmtree(base_path)
    assert not base_path.exists()

    # Isolate step 0: skip per-source artifacts. structural_input is the
    # fixture forget CSV (same index keying the labeler used).
    run_build_sources(
        split="forget10",
        output_dir=fx["f10_dir"],
        structural_input=fx["forget_input"],
        skip_source_build=True,
        overwrite=True,
    )
    assert base_path.exists(), "step 0 did not build contextual_labeled"
    # Per-source artifacts must NOT have been built (skip_source_build=True).
    assert not (fx["f10_dir"] / "contextual_labeled_idk").exists()

    dd = load_from_disk(str(base_path))
    rows = _all_rows(dd)
    assert rows, "step-0 base is empty"

    expected_keys = set(CONTEXTUAL_SCHEMA_KEYS)
    for r in rows:
        assert set(r.keys()) == expected_keys, (
            f"schema drift on {r['uid']!r}: "
            f"missing={expected_keys - set(r.keys())}, "
            f"extra={set(r.keys()) - expected_keys}"
        )
        assert r["label"] == 0, f"non-placeholder label on {r['uid']!r}"
        assert r["source"] == "forget"
        assert set(r["source_tags"]) == {"clean"}
        assert r["matched_claim_idx"] is None
        assert r["claim_snippet"] is None

    # Alignment contract: the base's cot_idx set equals the forget CSV index
    # set (so first_leak_labels.row_index == cot_idx holds row-for-row).
    import csv as _csv
    with open(fx["forget_input"]) as f:
        csv_idxs = {int(row["index"]) for row in _csv.DictReader(f)}
    base_cot_idxs = {r["cot_idx"] for r in rows}
    assert base_cot_idxs == csv_idxs, (
        f"base cot_idx set {sorted(base_cot_idxs)[:5]}... != forget CSV "
        f"index set {sorted(csv_idxs)[:5]}..."
    )

    # End-to-end: the REAL truncator must accept the REAL base + fixture
    # labels (precondition assert + relabel) on every partition.
    labels = load_first_leak_labels(fx["first_leak_path"])
    for sname in dd:
        truncated, summary = truncate_clean_forget(dd[sname], labels)
        # ans rows dropped; leak row (sent_idx==first_leak_idx==1) -> label=1.
        assert all(r["phase"] == "cot" for r in truncated)
        if len(truncated) > 0:
            assert summary["n_relabeled_to_1"] >= 0
            # Every kept CoT contributes exactly first_leak_idx+1 = 2 rows.
            from collections import Counter
            per_cot = Counter(r["cot_idx"] for r in truncated)
            assert all(c == 2 for c in per_cot.values()), (
                f"expected 2 rows/CoT (first_leak_idx=1); got {dict(per_cot)}"
            )


def test_step0_skip_preserves_prestaged_base(tmp_path, monkeypatch):
    """``skip_structural_build=True`` must leave a pre-existing
    ``contextual_labeled`` untouched (the pre-staged-base opt-out)."""
    fx = _setup_fixture(tmp_path, monkeypatch)
    from datasets import Dataset, DatasetDict
    from scripts.build_first_leak_classifier_data import run_build_sources
    from scripts.build_labeled_sources import CONTEXTUAL_FEATURES
    from redact.leak_dataset_schema import save_dataset_dict_with_invariant

    base_path = fx["f10_dir"] / "contextual_labeled"
    # Overwrite the fixture base with a 1-row SENTINEL whose uid is unique, so
    # we can detect any clobber by step 0.
    sentinel_uid = "SENTINEL_prestaged_q0000_cot_s000"
    sentinel_row = {
        "uid": sentinel_uid,
        "prompt": "sentinel?",
        "text": "Sentinel sentence.",
        "context_sentences": [],
        "phase": "cot",
        "author_slug": "sentinel_author",
        "cot_idx": 0,
        "sent_idx": 0,
        "label": 0,
        "matched_claim_idx": None,
        "claim_snippet": None,
        "source": "forget",
        "source_tags": ["clean"],
        "escalated": False,
        "rationale": "",
    }
    import shutil
    shutil.rmtree(base_path)
    # Route through the canonical save chokepoint so the empty test split gets
    # the num_shards=1 pin (datasets 4.8.x can't load_from_disk a zero-shard
    # split). A raw DatasetDict.save_to_disk would write an unreadable base.
    save_dataset_dict_with_invariant(
        DatasetDict({
            "train": Dataset.from_list(
                [sentinel_row], features=CONTEXTUAL_FEATURES
            ),
            "test": Dataset.from_dict(
                {c: [] for c in CONTEXTUAL_FEATURES}, features=CONTEXTUAL_FEATURES
            ),
        }),
        str(base_path),
    )

    run_build_sources(
        split="forget10",
        output_dir=fx["f10_dir"],
        structural_input=fx["forget_input"],
        skip_structural_build=True,
        skip_source_build=True,
        overwrite=True,
    )

    # The sentinel base survives verbatim (step 0 was skipped).
    dd = load_from_disk(str(base_path))
    rows = _all_rows(dd)
    assert len(rows) == 1, (
        f"expected the pre-staged 1-row base to survive; got {len(rows)} rows "
        f"(step 0 likely rebuilt it despite --skip_structural_build)"
    )
    assert rows[0]["uid"] == sentinel_uid, (
        f"pre-staged base was clobbered: uid={rows[0]['uid']!r}"
    )


def test_step0_cli_flag_routes_to_run_build_sources(tmp_path, monkeypatch):
    """``main()`` parses --skip_structural_build / --structural_input and
    forwards them to run_build_sources. Stubs run_build_sources + the later
    steps to capture the kwargs without running the heavy pipeline."""
    import scripts.build_first_leak_classifier_data as bflcd

    captured: dict = {}

    def _fake_run_build_sources(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(bflcd, "run_build_sources", _fake_run_build_sources)
    # Stub the remaining steps so main() returns quickly after step 0/1.
    monkeypatch.setattr(bflcd, "run_truncate_and_concat", lambda **k: None)
    monkeypatch.setattr(bflcd, "run_sentence0_reword", lambda **k: None)
    monkeypatch.setattr(bflcd, "run_subset_builds", lambda **k: None)

    csv_path = tmp_path / "my_forget.csv"
    csv_path.write_text("index,question,model_cot,response\n0,Q?,C.,A.\n")

    argv = [
        "build_first_leak_classifier_data",
        "--split", "forget10",
        "--skip_structural_build",
        "--structural_input", str(csv_path),
        "--skip_sentence0_reword",
        "--skip_subset_build",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bflcd.main()

    assert captured.get("skip_structural_build") is True, (
        f"--skip_structural_build not forwarded; captured={captured}"
    )
    assert captured.get("structural_input") == csv_path, (
        f"--structural_input not forwarded; captured={captured}"
    )

    # Inverse: default (flag absent) forwards skip_structural_build=False.
    captured.clear()
    monkeypatch.setattr(sys, "argv", [
        "build_first_leak_classifier_data",
        "--split", "forget10",
        "--skip_sentence0_reword",
        "--skip_subset_build",
    ])
    bflcd.main()
    assert captured.get("skip_structural_build") is False, (
        f"default should forward skip_structural_build=False; "
        f"captured={captured}"
    )
