"""CPU invariants for TCR projection and acceptance."""
import unittest

import torch

from upbd_impl.contracts import Candidate, DecisionKind, EvidenceBatch, PrepareContext
from upbd_impl.methods.tcr.core import (
    TCR, parameter_energy, project_repair_update, sensitive_component,
)


class TCRTest(unittest.TestCase):
    def test_projection_removes_sensitive_component(self):
        update = torch.tensor([1.0, 2.0, 0.5])
        basis = torch.tensor([[1.0], [0.0], [0.0]])
        repaired = project_repair_update(update, basis)
        self.assertAlmostEqual(repaired[0].item(), 0.0, places=5)
        self.assertAlmostEqual(repaired[1].item(), 2.0, places=5)
        self.assertLess(sensitive_component(repaired, basis).norm(), 1e-5)

    def test_parameter_energy(self):
        update = torch.tensor([1.0, 2.0])
        weights = torch.tensor([0.5, 2.0])
        self.assertAlmostEqual(parameter_energy(update, weights).item(), 8.5)
        with self.assertRaises(ValueError):
            parameter_energy(update, torch.ones(3))

    def test_acceptance_and_rollback(self):
        method = TCR()
        candidate = Candidate("test", 0, 1001)
        good = EvidenceBatch(0.01, 0.8, 0.0, 1.0, n=4)
        bad = EvidenceBatch(0.2, 0.8, 0.0, 1.0, n=4)
        state = method.prepare(PrepareContext())
        self.assertEqual(method.accept(candidate, good, state).kind, DecisionKind.ACCEPT)
        self.assertEqual(method.accept(candidate, bad, state).kind, DecisionKind.ROLLBACK)


if __name__ == "__main__":
    unittest.main()
