"""Independent CPU unit tests for the reverse core (torch only).

Run: python methods/reverse/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import REVERSE, worst_direction, low_rank_penalty, reversibility_radius

def test_worst_direction_unit_norm():
    g = torch.randn(16); v = worst_direction(g, 1.0)
    assert abs(v.norm().item() - 1.0) < 1e-5

def test_lowrank_penalty_rank_one_higher():
    rank1 = torch.zeros(8,8); rank1[:,0] = 1.0
    spread = torch.eye(8)
    assert low_rank_penalty(rank1).item() > low_rank_penalty(spread).item()

def test_radius_zero_when_no_leak():
    assert reversibility_radius(torch.zeros(4), torch.ones(4), 0.05).item() < 0.5


class CoreTest(unittest.TestCase):
    def test_worst_direction_unit_norm(self):
        globals()['test_worst_direction_unit_norm']()
    def test_lowrank_penalty_rank_one_higher(self):
        globals()['test_lowrank_penalty_rank_one_higher']()
    def test_radius_zero_when_no_leak(self):
        globals()['test_radius_zero_when_no_leak']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
