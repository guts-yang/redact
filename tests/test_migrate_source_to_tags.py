"""Integration tests for ``scripts.migrate_source_to_tags``.

Builds a tiny synthetic DatasetDict with the legacy single-column
``source`` schema, runs the migration, and asserts the output has the
correct ``(source, source_tags)`` for every row — including the
``_aug_v4`` reword detection path (clean_forget + reword rationale →
``("forget", ["clean", "reword_s0"])``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, Features, load_from_disk

from scripts.build_labeled_sources import CONTEXTUAL_FEATURES
from scripts.migrate_source_to_tags import migrate_dataset_dict


# The legacy schema is the canonical contextual schema minus the
# ``source_tags`` column the migration adds. Derived from
# CONTEXTUAL_FEATURES rather than hand-copied so the fixture tracks
# schema changes (``migrate_dataset_dict`` casts its output to
# CONTEXTUAL_FEATURES, so the two must agree column-for-column).
_LEGACY_FEATURES = Features({
    name: feature
    for name, feature in CONTEXTUAL_FEATURES.items()
    if name != "source_tags"
})


def _row(
    *,
    uid: str,
    source: str,
    label: int = 0,
    sent_idx: int = 0,
    rationale: str = "",
    cot_idx: int = 0,
) -> dict:
    row = {
        "uid": uid,
        "prompt": "Q?",
        "text": "T",
        "context_sentences": [],
        "phase": "cot",
        "author_slug": "test_author",
        "cot_idx": cot_idx,
        "sent_idx": sent_idx,
        "label": label,
        "matched_claim_idx": 0 if label == 1 else None,
        "claim_snippet": "c0" if label == 1 else None,
        "source": source,
        "escalated": False,
        "rationale": rationale,
    }
    # Only keep the columns the (derived) legacy schema declares.
    return {k: v for k, v in row.items() if k in _LEGACY_FEATURES}


def _make_legacy_dd(rows: list[dict]) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_list(rows, features=_LEGACY_FEATURES),
    })


class TestSimpleMigration:
    def test_clean_forget(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_forget")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert rows[0]["source_tags"] == ["clean"]

    def test_clean_idk(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_idk")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "idk"
        assert rows[0]["source_tags"] == ["clean"]

    def test_clean_idk_paraphrase(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_idk_paraphrase")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "idk"
        assert sorted(rows[0]["source_tags"]) == ["clean", "paraphrase"]

    def test_clean_aux(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_aux")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "aux"
        assert rows[0]["source_tags"] == ["clean"]

    def test_clean_retain(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_retain")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "retain"
        assert rows[0]["source_tags"] == ["clean"]

    def test_clean_retain_xfer(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_retain_xfer")])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "retain"
        assert sorted(rows[0]["source_tags"]) == ["clean", "xfer"]

    def test_clean_answer_forget(self):
        dd = _make_legacy_dd([_row(uid="u0", source="clean_answer_forget",
                                    label=1)])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert sorted(rows[0]["source_tags"]) == ["answer", "clean"]

    def test_corrupted_scheme(self):
        dd = _make_legacy_dd([_row(uid="u0", source="corrupted_noise_topk",
                                    label=1)])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert sorted(rows[0]["source_tags"]) == ["corrupted", "noise_topk"]


class TestRewordDetection:
    def test_reword_row_detected(self):
        """A clean_forget row with the Augmented-replace-mode rationale
        should be mapped to (forget, [clean, reword_s0]).
        """
        rows = [
            _row(
                uid="u_reword",
                source="clean_forget",
                label=1,
                sent_idx=0,
                rationale=(
                    "Augmented (replace-mode, subagent rewording pattern X): "
                    "sentence-0 rewritten to introduce a male pronoun"
                ),
            ),
            _row(uid="u_plain", source="clean_forget"),
        ]
        dd = _make_legacy_dd(rows)
        out = migrate_dataset_dict(dd, detect_reword=True)
        out_rows = {r["uid"]: r for r in out["train"]}
        assert out_rows["u_reword"]["source"] == "forget"
        assert sorted(out_rows["u_reword"]["source_tags"]) == [
            "clean", "reword_s0"
        ]
        assert out_rows["u_plain"]["source"] == "forget"
        assert out_rows["u_plain"]["source_tags"] == ["clean"]

    def test_reword_disabled_treats_as_plain(self):
        dd = _make_legacy_dd([
            _row(
                uid="u_reword",
                source="clean_forget",
                label=1,
                sent_idx=0,
                rationale="Augmented (replace-mode, ...)",
            ),
        ])
        out = migrate_dataset_dict(dd, detect_reword=False)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert rows[0]["source_tags"] == ["clean"]

    def test_non_reword_label1_not_detected(self):
        """label=1 + sent_idx=0 alone shouldn't trigger reword detection
        without the rationale prefix.
        """
        dd = _make_legacy_dd([
            _row(
                uid="u_label1",
                source="clean_forget",
                label=1,
                sent_idx=0,
                rationale="leaked the author's birth year",
            ),
        ])
        out = migrate_dataset_dict(dd, detect_reword=True)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert rows[0]["source_tags"] == ["clean"]

    def test_reword_wrong_sent_idx_not_detected(self):
        dd = _make_legacy_dd([
            _row(
                uid="u_si1",
                source="clean_forget",
                label=1,
                sent_idx=2,
                rationale="Augmented (replace-mode, ...)",
            ),
        ])
        out = migrate_dataset_dict(dd, detect_reword=True)
        rows = list(out["train"])
        assert rows[0]["source"] == "forget"
        assert rows[0]["source_tags"] == ["clean"]


class TestMigrationRoundTripOnDisk:
    def test_full_round_trip_save_load(self, tmp_path: Path):
        rows = [
            _row(uid="u0", source="clean_forget", cot_idx=0),
            _row(uid="u1", source="clean_retain", cot_idx=1),
            _row(uid="u2", source="clean_idk", cot_idx=2),
            _row(uid="u3", source="clean_idk_paraphrase", cot_idx=3),
            _row(uid="u4", source="clean_aux", cot_idx=4),
            _row(uid="u5", source="clean_retain_xfer", cot_idx=5),
            _row(uid="u6", source="clean_answer_forget", label=1, cot_idx=6),
        ]
        dd = _make_legacy_dd(rows)
        out = migrate_dataset_dict(dd, detect_reword=False)
        out_path = tmp_path / "migrated"
        out.save_to_disk(str(out_path))

        reloaded = load_from_disk(str(out_path))
        assert set(reloaded["train"]["source"]) == {
            "forget", "retain", "idk", "aux"
        }
        for r in reloaded["train"]:
            tags = list(r["source_tags"])
            assert "clean" in tags
            assert "corrupted" not in tags


class TestMigrationValidation:
    def test_missing_source_column_raises(self):
        from datasets import Dataset, DatasetDict, Features, Value

        features = Features({"uid": Value("string"), "label": Value("int64")})
        ds = Dataset.from_list([{"uid": "u0", "label": 0}], features=features)
        dd = DatasetDict({"train": ds})
        with pytest.raises(ValueError, match="no 'source' column"):
            migrate_dataset_dict(dd)

    def test_unknown_source_raises(self):
        dd = _make_legacy_dd([_row(uid="u0", source="mystery_source")])
        with pytest.raises(ValueError, match="unknown legacy"):
            migrate_dataset_dict(dd, detect_reword=False)
