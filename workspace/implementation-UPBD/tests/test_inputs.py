"""Proposal-specific JSON input and decisive fail-closed tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upbd_impl.contracts import BoundaryAsset
from upbd_impl.runtime.inputs import load_prepare_inputs, validate_decisive_inputs
from upbd_impl.runtime.paths import Paths


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = Paths(root, root / "target", root / "unlearned", root / "results")
        self.stats = root / "stats.json"
        self.stats.write_text(json.dumps({
            "representation_sensitivity": [[1, 0], [0, 1], [0, 0], [0, 0]],
            "parameter_weights": [1, 1, 1, 1],
            "benign_operator_drops": [0.01, 0.02],
            "aliases": {"a": "x"},
            "allow_concession": True,
        }), encoding="utf-8")
        self.reference = root / "reference.json"
        self.reference.write_text(json.dumps([[0] * 4, [1] * 4]), encoding="utf-8")
        self.requests = root / "requests.json"
        self.requests.write_text(json.dumps([
            {"request_id": "r1", "forget": ["a"], "protect": ["y"]},
        ]), encoding="utf-8")
        self.env = {
            "UPBD_IMPL_BOUNDARY_STATS_JSON": str(self.stats),
            "UPBD_IMPL_REFERENCE_HIDDEN_JSON": str(self.reference),
            "UPBD_IMPL_REQUESTS_JSON": str(self.requests),
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_json_inputs_load_without_gpu(self):
        import torch
        with patch.dict("os.environ", self.env, clear=False):
            boundary, requests, extras = load_prepare_inputs(
                self.paths, hidden_dim=4, device=torch.device("cpu"))
        self.assertEqual(boundary.representation_sensitivity.shape, (4, 2))
        self.assertEqual(boundary.parameter_weights.shape, (4,))
        self.assertEqual(extras["reference_hidden"].shape, (2, 4))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].forget, ("a",))
        for method in ("tcr", "bless", "rcbf", "stream", "tmot", "crown"):
            validate_decisive_inputs(method, boundary, requests, extras)

    def test_missing_required_inputs_fail_closed(self):
        import torch
        with patch.dict("os.environ", self.env, clear=False):
            _, requests, extras = load_prepare_inputs(
                self.paths, hidden_dim=4, device=torch.device("cpu"))
        empty = BoundaryAsset("synthetic", ("direct",))
        for method in ("tcr", "bless", "tmot"):
            with self.assertRaises(ValueError):
                validate_decisive_inputs(method, empty, (), {})
        with self.assertRaises(ValueError):
            validate_decisive_inputs("stream", empty, (), extras)
        with self.assertRaises(ValueError):
            validate_decisive_inputs("crown", empty, (), extras)


if __name__ == "__main__":
    unittest.main()
