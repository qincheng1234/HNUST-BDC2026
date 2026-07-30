import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train import learning_rate_multiplier


class TrainingScheduleTest(unittest.TestCase):
    def test_warmup_then_cosine_decay(self):
        multipliers = [
            learning_rate_multiplier(epoch, total_epochs=10, warmup_epochs=3, min_ratio=0.2)
            for epoch in range(10)
        ]

        self.assertAlmostEqual(multipliers[0], 1.0 / 3.0)
        self.assertAlmostEqual(multipliers[1], 2.0 / 3.0)
        self.assertAlmostEqual(multipliers[2], 1.0)
        self.assertAlmostEqual(multipliers[3], 1.0)
        self.assertAlmostEqual(multipliers[-1], 0.2)
        self.assertTrue(all(left >= right for left, right in zip(multipliers[3:], multipliers[4:])))

    def test_single_epoch_schedule_stays_at_base_rate(self):
        multiplier = learning_rate_multiplier(0, total_epochs=1, warmup_epochs=3, min_ratio=0.2)

        self.assertEqual(multiplier, 1.0)

    def test_invalid_schedule_arguments_fail_fast(self):
        with self.assertRaises(ValueError):
            learning_rate_multiplier(0, total_epochs=0, warmup_epochs=1, min_ratio=0.2)
        with self.assertRaises(ValueError):
            learning_rate_multiplier(0, total_epochs=2, warmup_epochs=1, min_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
