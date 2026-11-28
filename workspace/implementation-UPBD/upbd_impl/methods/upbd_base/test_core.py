"""CPU tests for the legacy UPBD reference objective."""
import unittest

import torch

from upbd_impl.methods.upbd_base.core import reference_sft_loss


class UPBDBaseTest(unittest.TestCase):
    def test_loss_backward(self):
        nll = torch.tensor([0.5, 1.0], requires_grad=True)
        out = reference_sft_loss(nll)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertIsNotNone(nll.grad)
        self.assertTrue(torch.isfinite(nll.grad).all())

    def test_diagnostics_are_finite(self):
        out = reference_sft_loss(torch.tensor([0.5]))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in out.float_diagnostics().values()))


if __name__ == "__main__":
    unittest.main()
