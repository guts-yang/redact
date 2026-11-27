"""Independent CPU unit tests for the attrib core (torch only).

Run: python methods/attrib/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import ATTRIB, keyword_attention_share

def test_ka_is_keyword_mass():
    att = torch.zeros(2,4); att[0,1]=0.3; att[0,2]=0.2; att[1,3]=0.9
    ka = keyword_attention_share(att, torch.tensor([1,2]), None)
    assert abs(ka[0].item()-0.5)<1e-6 and abs(ka[1].item()-0.0)<1e-6

def test_rr_dr_hinge():
    out = ATTRIB().loss(torch.tensor(0.9), torch.tensor(0.1),
                        torch.tensor(0.5), torch.tensor(0.4),
                        torch.sigmoid(torch.linspace(-1,1,4)), torch.full((4,),0.5),
                        torch.linspace(0,1,4), torch.tensor(0.0))
    assert out.diagnostics["attrib_RR"] > 0 and torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_ka_is_keyword_mass(self):
        globals()['test_ka_is_keyword_mass']()
    def test_rr_dr_hinge(self):
        globals()['test_rr_dr_hinge']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
