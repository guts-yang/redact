"""CPU invariants for torch-only Sinkhorn and forbidden support."""
import unittest

import torch

from upbd_impl.methods.tmot.core import (
    forbidden_mass, marginal_error, sinkhorn_transport,
)


class TMOTTest(unittest.TestCase):
    def test_sinkhorn_marginals(self):
        cost = torch.rand(4, 5)
        source = torch.full((4,), 0.25)
        target = torch.full((5,), 0.2)
        plan = sinkhorn_transport(cost, source, target, iterations=100)
        self.assertLess(marginal_error(plan, source, target).item(), 1e-3)
        self.assertAlmostEqual(plan.sum().item(), 1.0, places=4)

    def test_forbidden_support_has_zero_mass(self):
        cost = torch.rand(3, 3)
        forbidden = torch.tensor([
            [False, True, True],
            [True, False, True],
            [True, True, False],
        ])
        plan = sinkhorn_transport(cost, forbidden=forbidden, iterations=100)
        self.assertEqual(forbidden_mass(plan, forbidden).item(), 0.0)
        self.assertAlmostEqual(plan.sum().item(), 1.0, places=4)

    def test_fully_forbidden_row_fails(self):
        cost = torch.rand(2, 2)
        forbidden = torch.tensor([[True, True], [False, False]])
        with self.assertRaises(ValueError):
            sinkhorn_transport(cost, forbidden=forbidden)


if __name__ == "__main__":
    unittest.main()
