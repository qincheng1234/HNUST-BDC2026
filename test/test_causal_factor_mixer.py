import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import (
    CausalCrossSectionalFactorMixer,
    CrossSectionalResidualFactorMixer,
    build_model,
)


def model_config(sequence_length=12, d_model=32):
    return {
        "model_type": "causal_factor_mixer_v1",
        "sequence_length": sequence_length,
        "d_model": d_model,
        "mixer_layers": 2,
        "market_mixer_layers": 1,
        "time_mixer_hidden": 8,
        "mixer_expansion": 2,
        "factor_count": 4,
        "dropout": 0.0,
    }


class CausalFactorMixerTest(unittest.TestCase):
    def test_forward_shapes_and_gradients(self):
        torch.manual_seed(17)
        model = CausalCrossSectionalFactorMixer(5, model_config(), num_stocks=10)
        inputs = torch.randn(2, 10, 12, 5, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.bool)

        outputs = model(inputs, mask=mask)

        self.assertEqual(outputs["ranking_score"].shape, (2, 10))
        self.assertEqual(outputs["horizon_return"].shape, (2, 10, 3))
        self.assertEqual(outputs["downside_risk"].shape, (2, 10))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

        outputs["ranking_score"].mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_factor_context_changes_other_stock_scores(self):
        torch.manual_seed(19)
        model = CausalCrossSectionalFactorMixer(5, model_config(), num_stocks=10).eval()
        inputs = torch.randn(1, 10, 12, 5)
        changed_inputs = inputs.clone()
        changed_inputs[:, 0] += 0.5
        mask = torch.ones(1, 10, dtype=torch.bool)

        with torch.no_grad():
            original = model(inputs, mask=mask)["ranking_score"]
            changed = model(changed_inputs, mask=mask)["ranking_score"]

        self.assertFalse(torch.allclose(original[:, 1:], changed[:, 1:]))

    def test_padding_does_not_change_valid_stock_scores(self):
        torch.manual_seed(23)
        model = CausalCrossSectionalFactorMixer(5, model_config(), num_stocks=6).eval()
        inputs = torch.randn(1, 6, 12, 5)
        mask = torch.tensor([[True, True, True, True, False, False]])
        changed_inputs = inputs.clone()
        changed_inputs[:, 4:] += 100.0

        with torch.no_grad():
            original = model(inputs, mask=mask)["ranking_score"]
            changed = model(changed_inputs, mask=mask)["ranking_score"]

        self.assertTrue(torch.allclose(original[:, :4], changed[:, :4], atol=1e-6))

    def test_build_model_and_default_scale(self):
        config = model_config(sequence_length=60, d_model=128)
        config["time_mixer_hidden"] = 32
        config["factor_count"] = 8
        model = build_model(197, config, num_stocks=300)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertIsInstance(model, CausalCrossSectionalFactorMixer)
        self.assertLess(parameter_count, 800_000)

    def test_residual_encoder_is_invariant_to_market_common_offsets(self):
        torch.manual_seed(29)
        config = model_config()
        model = CrossSectionalResidualFactorMixer(5, config, num_stocks=6)
        inputs = torch.randn(2, 6, 12, 5)
        common_offset = torch.randn(2, 1, 12, 5)
        mask = torch.ones(2, 6, dtype=torch.bool)

        original = model._prepare_stock_inputs(inputs, mask)
        shifted = model._prepare_stock_inputs(inputs + common_offset, mask)

        self.assertTrue(torch.allclose(original, shifted, atol=1e-6))

    def test_residual_model_factory_and_outputs(self):
        torch.manual_seed(31)
        config = model_config()
        config["model_type"] = "causal_factor_mixer_v2"
        model = build_model(5, config, num_stocks=6)
        inputs = torch.randn(1, 6, 12, 5)

        outputs = model(inputs, mask=torch.ones(1, 6, dtype=torch.bool))

        self.assertIsInstance(model, CrossSectionalResidualFactorMixer)
        self.assertEqual(outputs["ranking_score"].shape, (1, 6))

if __name__ == "__main__":
    unittest.main()
