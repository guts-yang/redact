"""Runtime CSV loading, entity maps, and split-manifest tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upbd_impl.runtime.data import CHANNELS, load_training_data, normalize_question
from upbd_impl.runtime.paths import Paths


class RuntimeDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = Paths(root, root / "target", root / "unlearned", root / "results")
        self.env = {}
        self.entity_map = {}
        for channel in CHANNELS:
            path = root / f"{channel}.csv"
            lines = ["index,question,model_cot,response"]
            for index, entity in enumerate(("A", "B", "C", "D")):
                question = f"<｜User｜>{channel}-question-{entity}<｜Assistant｜>"
                lines.append(f'{index},"{question}",cot,answer')
                self.entity_map[f"{channel}-question-{entity}"] = entity
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.env[f"UPBD_IMPL_{channel.upper()}_CSV"] = str(path)
        self.map_path = root / "entities.json"
        self.map_path.write_text(json.dumps(self.entity_map), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_entity_mode_creates_scientifically_valid_disjoint_manifest(self):
        env = {**self.env, "UPBD_IMPL_ENTITY_MAP_JSON": str(self.map_path)}
        with patch.dict("os.environ", env, clear=False):
            data = load_training_data(self.paths, split_mode="entity")
        self.assertTrue(data.manifest["scientific_valid"])
        for channel in CHANNELS:
            manifest = data.manifest["channels"][channel]
            train = {entry["entity_key"] for entry in manifest["train"]}
            calibration = {entry["entity_key"] for entry in manifest["calibration"]}
            test = {entry["entity_key"] for entry in manifest["test"]}
            self.assertFalse(train & calibration)
            self.assertFalse(train & test)
            self.assertFalse(calibration & test)
            self.assertEqual(train | calibration | test, {"A", "B", "C", "D"})

    def test_question_mode_is_explicitly_non_scientific(self):
        with patch.dict("os.environ", self.env, clear=False):
            data = load_training_data(self.paths, split_mode="question")
        self.assertFalse(data.manifest["scientific_valid"])
        self.assertEqual(data.manifest["mode"], "question")

    def test_holdout_mode_marks_overlap(self):
        with patch.dict("os.environ", self.env, clear=False):
            data = load_training_data(self.paths, holdout=1, split_mode="holdout")
        self.assertTrue(data.manifest["holdout_test_overlap"])
        self.assertEqual(len(data.test["forget"]), 1)
        self.assertEqual(len(data.forget), 3)

    def test_missing_entity_map_fails_closed(self):
        with patch.dict("os.environ", self.env, clear=False):
            with self.assertRaises(FileNotFoundError):
                load_training_data(self.paths, split_mode="entity")

    def test_question_normalization(self):
        self.assertEqual(normalize_question("<｜User｜>q<｜Assistant｜>"), "q")


if __name__ == "__main__":
    unittest.main()
