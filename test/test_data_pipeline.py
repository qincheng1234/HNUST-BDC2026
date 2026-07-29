import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data_io import load_contest_stock_data, load_prediction_data, load_training_data
from splits import build_walk_forward_folds
from utils import create_labeled_ranking_dataset


class DataPipelineTest(unittest.TestCase):
    def test_loader_normalizes_codes_and_honors_as_of_date(self):
        rows = []
        for stock_code in ("1", "000002"):
            for date in pd.bdate_range("2026-01-01", periods=3):
                rows.append(
                    {
                        "股票代码": stock_code,
                        "日期": date.strftime("%Y-%m-%d"),
                        "开盘": 10.0,
                        "收盘": 10.1,
                        "最高": 10.2,
                        "最低": 9.9,
                        "成交量": 1000.0,
                        "成交额": 10000.0,
                        "换手率": 1.0,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(rows).to_csv(os.path.join(directory, "stock_data.csv"), index=False)
            pd.DataFrame(rows).to_csv(os.path.join(directory, "train.csv"), index=False)
            pd.DataFrame(rows).to_csv(os.path.join(directory, "test.csv"), index=False)
            data, codes, _ = load_contest_stock_data(
                directory,
                expected_stock_count=2,
                as_of_date="2026-01-02",
            )
            local_training, _, local_source = load_training_data(
                directory,
                data_mode="local_split",
                expected_stock_count=2,
            )
            local_prediction, _, prediction_source = load_prediction_data(
                directory,
                data_mode="local_split",
                expected_stock_count=2,
            )

        self.assertEqual(codes, frozenset({"000001", "000002"}))
        self.assertEqual(data["日期"].max(), pd.Timestamp("2026-01-02"))
        self.assertEqual(local_source.name, "train.csv")
        self.assertEqual(prediction_source.name, "train.csv")
        self.assertEqual(len(local_training), len(local_prediction))

    def test_walk_forward_folds_purge_forward_label_window(self):
        dates = pd.bdate_range("2025-01-01", periods=260)
        folds = build_walk_forward_folds(
            dates,
            label_horizon=5,
            embargo_days=5,
            validation_days=5,
            min_train_days=180,
            cv_folds=3,
            warmup_days=120,
        )
        positions = {date: index for index, date in enumerate(dates)}

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[-1]["validation_data_end"], dates[-1])
        for fold in folds:
            train_end = positions[fold["train_sample_end"]]
            validation_start = positions[fold["validation_start"]]
            self.assertLessEqual(train_end + 5, validation_start - 1)
            self.assertLess(fold["train_data_end"], fold["validation_start"])

    def test_labeled_dataset_keeps_sessions_across_weekends(self):
        dates = pd.bdate_range("2026-01-01", periods=8)
        rows = []
        for instrument in range(10):
            for date_index, date in enumerate(dates):
                rows.append(
                    {
                        "instrument": instrument,
                        "日期": date,
                        "feature": float(instrument + date_index),
                        "label": float(instrument - date_index),
                    }
                )
        sequences, _, _, _ = create_labeled_ranking_dataset(
            pd.DataFrame(rows),
            features=["feature"],
            sequence_length=3,
        )

        self.assertEqual(len(sequences), 6)
        self.assertTrue(all(sequence.shape == (10, 3, 1) for sequence in sequences))

    def test_production_entrypoints_use_mode_specific_loaders(self):
        with open(os.path.join(ROOT, "code/src/train.py"), encoding="utf-8") as handle:
            train_source = handle.read()
        with open(os.path.join(ROOT, "code/src/predict.py"), encoding="utf-8") as handle:
            predict_source = handle.read()

        self.assertIn("load_training_data", train_source)
        self.assertIn("load_prediction_data", predict_source)
        self.assertNotIn("test.csv", train_source)
        self.assertNotIn("test.csv", predict_source)


if __name__ == "__main__":
    unittest.main()
