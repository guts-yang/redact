"""Entity-disjoint probe and attribution-control tests."""
import unittest

from upbd_impl.contracts import Probe
from upbd_impl.core.controls import assert_matched, make_matched_controls
from upbd_impl.core.probes import split_probes


class DataSplitTest(unittest.TestCase):
    def setUp(self):
        self.probes = [
            Probe(f"p{entity}-{variant}", f"entity-{entity}", "q", transformation=str(variant))
            for entity in range(8) for variant in range(2)
        ]

    def test_entity_partitions_are_disjoint_and_deterministic(self):
        first = split_probes(self.probes, seed=1001)
        second = split_probes(self.probes, seed=1001)
        self.assertEqual(first, second)
        train = {probe.entity_key for probe in first.train}
        calibration = {probe.entity_key for probe in first.calibration}
        test = {probe.entity_key for probe in first.test}
        self.assertFalse(train & calibration)
        self.assertFalse(train & test)
        self.assertFalse(calibration & test)
        self.assertTrue(train and calibration and test)
        self.assertEqual(len(first.train + first.calibration + first.test), len(self.probes))

    def test_too_few_entities_fail_closed(self):
        with self.assertRaises(ValueError):
            split_probes(self.probes[:2])

    def test_controls_are_matched(self):
        controls = make_matched_controls(
            unlearned_checkpoint="u", target_checkpoint="t",
            steps=8, lr=1e-4, seed=1001, max_len=256, lora_r=8)
        assert_matched(controls)
        self.assertNotEqual(controls["retain_only"].start_checkpoint,
                            controls["target_start"].start_checkpoint)


if __name__ == "__main__":
    unittest.main()
