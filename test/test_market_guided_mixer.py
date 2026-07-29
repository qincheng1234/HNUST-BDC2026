import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import MarketGuidedMixer


def model_config(sequence_length=12, d_model=32):
    return {
        "sequence_length": sequence_length,
        "d_model": d_model,
        "mixer_layers": 2,
        "time_mixer_hidden": 8,
        "mixer_expansion": 2,
        "dropout": 0.0,
    }


class MarketGuidedMixerTest(unittest.TestCase):
    def test_forward_shape_and_gradients(self):
        torch.manual_seed(7)
        model = MarketGuidedMixer(5, model_config(), num_stocks=10)
        inputs = torch.randn(2, 10, 12, 5, requires_grad=True)

        scores = model(inputs)
        self.assertEqual(scores.shape, (2, 10))
        self.assertTrue(torch.isfinite(scores).all())

        scores.mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_market_guidance_changes_other_stock_scores(self):
        torch.manual_seed(11)
        model = MarketGuidedMixer(5, model_config(), num_stocks=10).eval()
        inputs = torch.randn(1, 10, 12, 5)
        changed_inputs = inputs.clone()
        changed_inputs[:, 0] += 0.5

        with torch.no_grad():
            original_scores = model(inputs)
            changed_scores = model(changed_inputs)

        self.assertFalse(torch.allclose(original_scores[:, 1:], changed_scores[:, 1:]))

    def test_compact_default_configuration(self):
        config = model_config(sequence_length=60, d_model=128)
        config["time_mixer_hidden"] = 32
        model = MarketGuidedMixer(197, config, num_stocks=300)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(parameter_count, 500_000)


if __name__ == "__main__":
    unittest.main()
