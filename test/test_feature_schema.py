import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils import CROSS_SECTIONAL_FEATURES, add_cross_sectional_market_features
from utils import build_model_feature_columns, engineer_features


class FeatureSchemaTests(unittest.TestCase):
    def test_model_features_exclude_continuous_stock_identifier(self):
        features = build_model_feature_columns(["instrument", "return_1", "换手率"])

        self.assertNotIn("instrument", features)
        self.assertEqual(features[:2], ["return_1", "换手率"])
        self.assertEqual(features[-len(CROSS_SECTIONAL_FEATURES):], CROSS_SECTIONAL_FEATURES)

    def test_cross_sectional_features_use_only_same_day_values(self):
        dates = pd.to_datetime(["2026-07-01"] * 3 + ["2026-07-02"] * 3)
        frame = pd.DataFrame(
            {
                "日期": dates,
                "成交额": [100, 200, 300, 100, 200, 300],
                "换手率": [1, 2, 3, 1, 2, 3],
                "return_1": [-0.01, 0.0, 0.02, -0.03, 0.01, 0.04],
                "return_5": [-0.02, 0.01, 0.03, -0.01, 0.02, 0.05],
                "return_10": [-0.04, 0.02, 0.06, -0.02, 0.03, 0.08],
                "volatility_20": [0.3, 0.2, 0.1, 0.4, 0.2, 0.1],
                "high_low_spread": [1, 2, 3, 1, 2, 3],
                "开盘": [10, 10, 10, 10, 10, 10],
            }
        )

        featured = add_cross_sectional_market_features(frame)

        self.assertTrue(np.isfinite(featured[CROSS_SECTIONAL_FEATURES].to_numpy()).all())
        self.assertEqual(featured.loc[0, "market_return_1"], 0.0)
        self.assertEqual(featured.loc[3, "market_return_1"], 0.01)
        self.assertLess(featured.loc[0, "cs_return_1_rank"], featured.loc[2, "cs_return_1_rank"])
        self.assertAlmostEqual(featured.loc[2, "relative_return_1"], 0.02)

    def test_rsqr_remains_available_after_its_rolling_window(self):
        try:
            import talib
        except ImportError:
            self.skipTest("TA-Lib is not installed")

        self.assertIsNotNone(talib)

        periods = 80
        close = np.arange(10.0, 10.0 + periods)
        source = pd.DataFrame(
            {
                "开盘": close - 0.1,
                "收盘": close,
                "最高": close + 0.2,
                "最低": close - 0.2,
                "成交量": np.full(periods, 1_000.0),
                "成交额": close * 1_000.0,
            }
        )

        featured = engineer_features(source)

        self.assertGreater(featured["RSQR5"].iloc[-1], 0.99)
        self.assertGreater(featured["RSQR60"].iloc[-1], 0.99)
