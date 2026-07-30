import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train import calculate_mean_rank_ic


class AuxiliaryDiagnosticsTest(unittest.TestCase):
    def test_rank_ic_ignores_padding_and_detects_direction(self):
        targets = torch.tensor([[0.1, 0.2, 0.3, 0.4, 99.0]])
        mask = torch.tensor([[True, True, True, True, False]])
        aligned = torch.tensor([[1.0, 2.0, 3.0, 4.0, -99.0]])
        reversed_prediction = torch.tensor([[4.0, 3.0, 2.0, 1.0, -99.0]])

        aligned_ic = calculate_mean_rank_ic(aligned, targets, mask)
        reversed_ic = calculate_mean_rank_ic(reversed_prediction, targets, mask)

        self.assertAlmostEqual(aligned_ic, 1.0)
        self.assertAlmostEqual(reversed_ic, -1.0)


if __name__ == "__main__":
    unittest.main()
