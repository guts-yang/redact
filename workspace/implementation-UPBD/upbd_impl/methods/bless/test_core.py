"""CPU invariants for BLESS margin composition and scope handling."""
import unittest

import torch

from upbd_impl.contracts import (
    Candidate, DecisionKind, EvidenceBatch, MethodState, PrepareContext,
)
from upbd_impl.methods.bless.core import (
    BLESS, compose_margin, margin_consumption, required_margin,
)


class BLESSTest(unittest.TestCase):
    def test_margin_consumption_and_composition(self):
        before = torch.tensor([2.0, 1.0])
        after = torch.tensor([1.5, 0.75])
        self.assertTrue(torch.equal(margin_consumption(before, after), torch.tensor([0.5, 0.25])))
        self.assertAlmostEqual(required_margin(1.0, 0.1, 3), 1.3)
        self.assertAlmostEqual(compose_margin(2.0, [0.2, 0.3]), 1.5)

    def test_outside_family_is_unknown(self):
        method = BLESS()
        state = method.prepare(PrepareContext(extras={"outside_operator_family": True}))
        evidence = EvidenceBatch(0.0, 1.0, 0.0, 1.0, n=2)
        decision = method.accept(Candidate("test", 0, 1001), evidence, state)
        self.assertEqual(decision.kind, DecisionKind.UNKNOWN)

    def test_insufficient_reserve_rolls_back(self):
        method = BLESS()
        state = MethodState(values={"last_required_margin": 1.5, "last_worst_margin": 1.0})
        evidence = EvidenceBatch(0.0, 1.0, 0.0, 1.0, n=2)
        decision = method.accept(Candidate("test", 0, 1001), evidence, state)
        self.assertEqual(decision.kind, DecisionKind.ROLLBACK)


if __name__ == "__main__":
    unittest.main()
