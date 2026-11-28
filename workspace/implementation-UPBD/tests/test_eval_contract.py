"""Evaluation adapter contracts without launching generation or judges."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upbd_impl.runtime.artifacts import RunArtifacts
from upbd_impl.runtime.device import resolve_device
from upbd_impl.runtime.paths import Paths
from upbd_impl.runtime.rtofu_eval import generation_commands, run_eval, select_test_csvs


class EvalContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = Paths(root, root / "target", root / "unlearned", root / "results")
        self.artifacts = RunArtifacts.create(self.paths, "run")

    def tearDown(self):
        self.temp.cleanup()

    def test_generation_commands_are_run_local(self):
        config_dir = self.artifacts.path("model_config")
        commands = generation_commands(self.artifacts, self.paths, config_dir, 4)
        self.assertEqual(len(commands), 4)
        for command in commands:
            joined = " ".join(command)
            self.assertIn(str(self.artifacts.root), joined)
            self.assertIn(str(config_dir), joined)
            self.assertNotIn("eval_one_model.sh", joined)

    def test_missing_merged_checkpoint_fails_before_generation(self):
        with self.assertRaises(RuntimeError):
            run_eval(self.artifacts.root, stage="cpu")
        with self.assertRaises(FileNotFoundError):
            run_eval(self.artifacts.root, stage="gpu", num_examples=1)

    def test_manifest_selects_only_test_rows(self):
        channels = {}
        for split, channel in {"forget10": "forget", "forget10_retain": "retain",
                               "real_authors": "authors", "world_facts": "world"}.items():
            channels[channel] = {"test": [{"probe_id": "1", "question": "test-question"}]}
            self.artifacts.write_text(
                f"eval/raw/{split}.csv",
                "index,question\n0,train-question\n1,test-question\n")
        self.artifacts.write_json("split_manifest.json", {
            "mode": "question", "scientific_valid": False, "channels": channels})
        counts = select_test_csvs(self.artifacts)
        self.assertEqual(counts, {
            "forget10": 1, "forget10_retain": 1,
            "real_authors": 1, "world_facts": 1,
        })
        for split in ("forget10", "forget10_retain", "real_authors", "world_facts"):
            content = self.artifacts.path(f"eval/test/{split}.csv").read_text(encoding="utf-8")
            self.assertIn("test-question", content)
            self.assertNotIn("train-question", content)

    def test_cuda_requires_explicit_visibility(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
