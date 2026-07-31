import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import CrossSectionalResidualFactorMixer, build_model


def model_config(sequence_length=12, d_model=32):
    return {
        "model_type": "causal_factor_mixer_v2",
        "sequence_length": sequence_length,
        "d_model": d_model,
        "mixer_layers": 2,
        "market_mixer_layers": 1,
        "time_mixer_hidden": 8,
        "mixer_expansion": 2,
        "factor_count": 4,
        "dropout": 0.0,
        "cross_sectional_epsilon": 1e-6,
    }


class CausalFactorMixerTest(unittest.TestCase):
    def test_forward_shapes_and_gradients(self):
        torch.manual_seed(17)
        model = CrossSectionalResidualFactorMixer(5, model_config(), num_stocks=10)
        inputs = torch.randn(2, 10, 12, 5, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.bool)

        scores = model(inputs, mask=mask)
        self.assertEqual(scores.shape, (2, 10))
        self.assertTrue(torch.isfinite(scores).all())

        scores.mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_cross_stock_interaction(self):
        """Changing one stock alters other stock scores via factor mixer."""
        torch.manual_seed(19)
        model = CrossSectionalResidualFactorMixer(5, model_config(), num_stocks=10)
        model.eval()
        inputs = torch.randn(1, 10, 12, 5)
        changed = inputs.clone()
        changed[:, 0] += 0.5
        mask = torch.ones(1, 10, dtype=torch.bool)

        with torch.no_grad():
            original = model(inputs, mask=mask)
            new = model(changed, mask=mask)

        self.assertFalse(torch.allclose(original[:, 1:], new[:, 1:]))

    def test_padding_does_not_affect_valid_stocks(self):
        torch.manual_seed(23)
        model = CrossSectionalResidualFactorMixer(5, model_config(), num_stocks=6)
        model.eval()
        inputs = torch.randn(1, 6, 12, 5)
        mask = torch.tensor([[True, True, True, True, False, False]])
        changed = inputs.clone()
        changed[:, 4:] += 100.0

        with torch.no_grad():
            original = model(inputs, mask=mask)
            new = model(changed, mask=mask)

        self.assertTrue(torch.allclose(original[:, :4], new[:, :4], atol=1e-6))

    def test_build_model_and_parameter_budget(self):
        config = model_config(sequence_length=60, d_model=128)
        config["time_mixer_hidden"] = 32
        config["factor_count"] = 8
        model = build_model(197, config, num_stocks=300)
        param_count = sum(p.numel() for p in model.parameters())

        self.assertIsInstance(model, CrossSectionalResidualFactorMixer)
        self.assertLess(param_count, 800_000)

    def test_residual_encoder_is_invariant_to_common_offset(self):
        torch.manual_seed(29)
        config = model_config()
        model = CrossSectionalResidualFactorMixer(5, config, num_stocks=6)
        inputs = torch.randn(2, 6, 12, 5)
        common_offset = torch.randn(2, 1, 12, 5)
        mask = torch.ones(2, 6, dtype=torch.bool)

        original = model._prepare_stock_inputs(inputs, mask)
        shifted = model._prepare_stock_inputs(inputs + common_offset, mask)

        self.assertTrue(torch.allclose(original, shifted, atol=1e-6))

    def test_factory_produces_correct_type(self):
        config = model_config()
        model = build_model(5, config, num_stocks=6)
        self.assertIsInstance(model, CrossSectionalResidualFactorMixer)
        scores = model(
            torch.randn(1, 6, 12, 5),
            mask=torch.ones(1, 6, dtype=torch.bool),
        )
        self.assertEqual(scores.shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
