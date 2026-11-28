"""Static dependency and device-isolation checks."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "upbd_impl" / "methods"


class IsolationTest(unittest.TestCase):
    def test_method_cores_have_no_runtime_or_cross_method_imports(self):
        for path in METHODS.glob("*/core.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("upbd_impl.runtime"), path)
                        self.assertFalse(alias.name.startswith("upbd_impl.methods"), path)
                    continue
                if module:
                    self.assertFalse(module.startswith("upbd_impl.runtime"), path)
                    self.assertFalse(module.startswith("upbd_impl.methods"), path)
                    self.assertNotIn("implementations-1127", module, path)

    def test_no_hardcoded_gpu_or_existing_queue(self):
        for path in (ROOT / "upbd_impl").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('.to("cuda")', text, path)
            self.assertNotIn("CUDA_VISIBLE_DEVICES=1", text, path)
            self.assertNotIn("queue_1127", text, path)
            self.assertNotIn("implementations-1127", text, path)


if __name__ == "__main__":
    unittest.main()
