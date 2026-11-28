"""CPU invariants for STREAM dual state and infeasibility."""
import unittest

import torch

from upbd_impl.contracts import (
    Candidate, DecisionKind, EvidenceBatch, PrepareContext, RequestSpec,
)
from upbd_impl.methods.stream.core import (
    STREAM, cumulative_violation, direct_request_conflict, dual_update, proxy_regret,
)


class STREAMTest(unittest.TestCase):
    def test_dual_update_is_nonnegative(self):
        current = torch.tensor([0.0, 1.0])
        risk = torch.tensor([0.0, 0.2])
        budget = torch.tensor([0.05, 0.05])
        updated = dual_update(current, risk, budget, rate=0.1)
        self.assertTrue((updated >= 0).all())
        self.assertGreater(updated[1].item(), current[1].item())

    def test_metrics(self):
        self.assertAlmostEqual(cumulative_violation([0.1, 0.0], [0.05, 0.05]), 0.05)
        self.assertAlmostEqual(proxy_regret([1.0, 2.0], [0.5, 1.0]), 1.5)

    def test_conflict_is_infeasible(self):
        request = RequestSpec("r1", forget=("x",), protect=("x",))
        self.assertTrue(direct_request_conflict([request]))
        method = STREAM()
        state = method.prepare(PrepareContext(requests=(request,)))
        evidence = EvidenceBatch(0.0, 1.0, 0.0, 1.0, n=1)
        decision = method.accept(Candidate("test", 0, 1001), evidence, state)
        self.assertEqual(decision.kind, DecisionKind.INFEASIBLE)

    def test_feasible_request_updates_dual(self):
        request = RequestSpec("r1", forget=("x",), protect=("y",), risk_budget=0.05)
        method = STREAM()
        state = method.prepare(PrepareContext(requests=(request,)))
        state.values["last_margin_loss"] = torch.tensor(0.2)
        evidence = EvidenceBatch(0.1, 1.0, 0.0, 1.0, n=1)
        decision = method.accept(Candidate("test", 0, 1001), evidence, state)
        self.assertEqual(decision.kind, DecisionKind.ROLLBACK)
        self.assertGreater(state.values["dual"][0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
