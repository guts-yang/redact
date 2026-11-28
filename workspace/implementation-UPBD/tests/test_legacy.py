"""Legacy behavior-target parity and command-isolation tests."""
import unittest
from pathlib import Path

from upbd_impl.runtime.data import behavior_target
from upbd_impl.runtime.legacy_upbd import compare_behavior_targets, legacy_train_command


class LegacyTest(unittest.TestCase):
    def test_behavior_target_semantics(self):
        rows = [
            {"model_cot": "think", "response": "answer"},
            {"model_cot": "", "response": "answer"},
            {"model_cot": "", "response": "", "gold_answer": "gold"},
        ]
        self.assertEqual(behavior_target(rows[0]), "think\n</think>\n\nanswer")
        self.assertEqual(behavior_target(rows[1]), "answer")
        self.assertEqual(behavior_target(rows[2]), "gold")
        self.assertTrue(all(item["matches"] for item in compare_behavior_targets(rows)))

    def test_legacy_command_is_not_executed(self):
        command = legacy_train_command(
            python="python", output_dir=Path("new/adapter"), merged_dir=Path("new/merged"))
        self.assertIn("scripts.upbd.train_upbd", command)
        self.assertIn("new/merged", command)


if __name__ == "__main__":
    unittest.main()
