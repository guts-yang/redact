"""Independent CPU unit tests for the layer core (torch only).

Run: python methods/layer/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import LAYER, soft_depth, depth_consistency, certificate

def test_soft_depth_shallow_when_early_leak():
    # leak high at shallow layers -> small depth
    leak = torch.tensor([0.9,0.0,0.0,0.0])
    assert soft_depth(leak).item() < 1.0

def test_dc_zero_for_identical():
    assert depth_consistency(torch.tensor([3.0,3.0,3.0])).item() < 1e-4

def test_cert_in_range():
    c = certificate(torch.tensor([9.,9.,9.]), 8.0); assert 0<=c.item()<=1

def test_loss_grad():
    torch.manual_seed(0)
    lr = torch.rand(4,12, requires_grad=True)
    out = LAYER().loss(lr, torch.tensor(0.0))
    out.loss.backward(); assert torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_soft_depth_shallow_when_early_leak(self):
        globals()['test_soft_depth_shallow_when_early_leak']()
    def test_dc_zero_for_identical(self):
        globals()['test_dc_zero_for_identical']()
    def test_cert_in_range(self):
        globals()['test_cert_in_range']()
    def test_loss_grad(self):
        globals()['test_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
