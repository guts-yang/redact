"""CPU invariants for the RCBF barrier and co-adaptation interface."""
import unittest

import torch

from upbd_impl.contracts import PrepareContext
from upbd_impl.methods.rcbf.core import (
    RCBF, barrier_scores, invariance_violation, safety_loss,
)


class RCBFTest(unittest.TestCase):
    def test_barrier_score_gradient(self):
        hidden = torch.randn(2, 4, 8)
        weight = torch.zeros(8, requires_grad=True)
        loss = safety_loss(barrier_scores(hidden, weight), eta=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(weight.grad.abs().sum().item(), 0.0)

    def test_invariance_violation_direction(self):
        current = torch.tensor([1.0, 0.2])
        following = torch.tensor([1.0, 0.0])
        violation = invariance_violation(current, following, gamma=0.1)
        self.assertAlmostEqual(violation[0].item(), 0.0)
        self.assertGreater(violation[1].item(), 0.0)

    def test_barrier_parameter_group_is_isolated(self):
        method = RCBF()
        state = method.prepare(PrepareContext(extras={"hidden_dim": 8}))
        groups = method.parameter_groups(state)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "barrier")
        self.assertIs(groups[0].params[0], state.values["barrier_weight"])


if __name__ == "__main__":
    unittest.main()
