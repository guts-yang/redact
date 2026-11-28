"""Registry and import-purity tests."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from upbd_impl.registry import PROPOSAL_ORDER, get_spec, list_methods, load_method


ROOT = Path(__file__).resolve().parents[1]


class RegistryTest(unittest.TestCase):
    def test_registry_contents(self):
        self.assertEqual(PROPOSAL_ORDER, ("tcr", "bless", "rcbf", "stream", "tmot", "crown"))
        self.assertEqual(len(list_methods()), 7)
        self.assertEqual(get_spec("tcr").idea_dir, "IDEA_01_TCR")
        with self.assertRaises(ValueError):
            get_spec("unknown")

    def test_lazy_method_loading(self):
        for spec in list_methods():
            self.assertEqual(load_method(spec.slug).spec.slug, spec.slug)

    def test_registry_import_has_no_heavy_side_effects(self):
        code = (
            "import sys; import upbd_impl.registry; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'peft' not in sys.modules; print('pure')"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            check=True, capture_output=True, text=True)
        self.assertIn("pure", result.stdout)


if __name__ == "__main__":
    unittest.main()
