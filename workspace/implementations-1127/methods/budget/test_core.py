"""Independent CPU unit tests for the budget core (torch only).

Run: python methods/budget/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import BUDGET, BudgetConfig, monotonicity_loss, budget_certificate

def test_monotonicity_positive_on_increasing_rho():
    # rho increasing in b must be PENALIZED (positive L_mono)
    rho = torch.tensor([0.1, 0.2, 0.4, 0.8])
    assert monotonicity_loss(rho).item() > 0
    # decreasing rho -> zero penalty
    assert monotonicity_loss(torch.tensor([0.8, 0.4, 0.2, 0.1])).item() == 0

def test_bc_certificate():
    assert abs(budget_certificate(torch.tensor([0.5,0.4,0.3])).item()) < 1e-6
    assert abs(budget_certificate(torch.tensor([0.1,0.5,0.3])).item() - 0.4) < 1e-6

def test_loss_scalar_and_grad():
    out = BUDGET().loss(torch.tensor([0.3,0.2,0.1], requires_grad=True),
                        torch.tensor(0.0), torch.tensor(0.0))
    out.loss.backward()
    assert out.loss.dim() == 0 and torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_monotonicity_positive_on_increasing_rho(self):
        globals()['test_monotonicity_positive_on_increasing_rho']()
    def test_bc_certificate(self):
        globals()['test_bc_certificate']()
    def test_loss_scalar_and_grad(self):
        globals()['test_loss_scalar_and_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
