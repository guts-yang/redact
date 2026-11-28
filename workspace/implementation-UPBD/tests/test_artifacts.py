"""Run artifact path and non-overwrite tests."""
import json
import tempfile
import unittest
from pathlib import Path

from upbd_impl.runtime.artifacts import RunArtifacts
from upbd_impl.runtime.paths import Paths, run_dir


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = Paths(root, root / "target", root / "unlearned", root / "results")

    def tearDown(self):
        self.temp.cleanup()

    def test_run_path_cannot_escape(self):
        with self.assertRaises(ValueError):
            run_dir(self.paths, "../escape")
        with self.assertRaises(ValueError):
            run_dir(self.paths, "nested/run")

    def test_atomic_summary_and_no_overwrite(self):
        artifacts = RunArtifacts.create(self.paths, "run")
        target = artifacts.write_summary({"status": "ok"})
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertTrue(payload["diagnostic_only"])
        self.assertEqual(payload["claim_level"], "non-claim")
        with self.assertRaises(FileExistsError):
            RunArtifacts.create(self.paths, "run")
        with self.assertRaises(ValueError):
            artifacts.path("../outside")


if __name__ == "__main__":
    unittest.main()
