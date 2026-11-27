"""Independent CPU unit tests for the drift core (torch only).

Run: python methods/drift/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import DRIFT, drift_decomposition

def test_decomposition_masks_sum_to_one():
    cur = torch.randn(4,5); base = torch.randn(4,5)
    c = torch.tensor([0.1,0.9,0.2,0.8])
    _, free, load = drift_decomposition(cur, base, c, 0.5)
    assert torch.allclose(free + load, torch.ones(4,1))

def test_restoration_pulls_free_toward_base():
    torch.manual_seed(0)
    cur = torch.randn(4,5); base = cur.detach() + 1.0
    c = torch.tensor([0.9,0.9,0.1,0.1])  # last two are free
    delta = torch.zeros(4,5, requires_grad=True)
    out = DRIFT().loss(cur, base, c, delta)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    # free steps carry gradient for restoration, load steps are locked
    assert float(delta.grad[2].abs().sum()) + float(delta.grad[3].abs().sum()) > 0


class CoreTest(unittest.TestCase):
    def test_decomposition_masks_sum_to_one(self):
        globals()['test_decomposition_masks_sum_to_one']()
    def test_restoration_pulls_free_toward_base(self):
        globals()['test_restoration_pulls_free_toward_base']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
