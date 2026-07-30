import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_reporting import build_ablation_row, write_ablation_summary
from utils import CROSS_SECTIONAL_FEATURES, select_feature_experiment_columns


FEATURE_MAP = {
    "39": ["instrument", "sma_5", "return_1", "volume_ratio"],
    "158+39": [
        "instrument",
        "开盘",
        "收盘",
        "成交量",
        "换手率",
        "ROC5",
        "MA5",
        "VMA5",
        "OPEN0",
        "sma_5",
        "ema_12",
        "return_1",
        "volume_ratio",
        "high_low_spread",
    ],
}


def make_metadata(experiment, scores):
    return {
        "feature_experiment": experiment,
        "model_type": "causal_factor_mixer_v2",
        "source_feature_set": "39" if experiment == "A1" else "158+39",
        "feature_count": 53,
        "features": list(CROSS_SECTIONAL_FEATURES),
        "selected_epochs": 2,
        "oof_top5_mean_return": sum(scores) / len(scores),
        "oof_top5_std": 0.01,
        "epoch_selection": {"robust_score": 0.004},
        "oof_folds": [{"epoch_scores": [0.0, score]} for score in scores],
    }


class FeatureExperimentTests(unittest.TestCase):
    def test_feature_experiments_are_schema_based_and_deterministic(self):
        self.assertEqual(
            select_feature_experiment_columns(FEATURE_MAP, "158+39", "A1"),
            FEATURE_MAP["39"],
        )
        self.assertEqual(
            select_feature_experiment_columns(FEATURE_MAP, "158+39", "A2"),
            FEATURE_MAP["158+39"][:9],
        )

        a3_columns = select_feature_experiment_columns(FEATURE_MAP, "158+39", "A3")
        self.assertEqual(a3_columns, ["instrument", "换手率", "ROC5", "return_1", "volume_ratio"])

    def test_a0_preserves_the_requested_source_schema(self):
        self.assertEqual(
            select_feature_experiment_columns(FEATURE_MAP, "39", "A0"),
            FEATURE_MAP["39"],
        )

    def test_summary_uses_selected_epoch_and_records_missing_experiments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_root = Path(temporary_directory) / "model"
            artifact_dir = model_root / "a1"
            artifact_dir.mkdir(parents=True)
            metadata_path = artifact_dir / "model_meta.json"
            metadata_path.write_text(
                json.dumps(make_metadata("A1", [0.02, 0.01])),
                encoding="utf-8",
            )

            row = build_ablation_row(metadata_path)
            self.assertEqual(row["worst_fold_return"], 0.01)
            self.assertEqual(row["positive_fold_rate"], 1.0)
            self.assertTrue(row["passes_a_gate"])

            output_path = Path(temporary_directory) / "summary.csv"
            rows = write_ablation_summary(model_root, output_path)
            self.assertEqual(
                [row["status"] for row in rows],
                ["completed", "missing_artifact", "missing_artifact"],
            )
            with output_path.open(encoding="utf-8", newline="") as handle:
                written_rows = list(csv.DictReader(handle))
            self.assertEqual(written_rows[0]["experiment"], "A1")
            self.assertEqual(written_rows[1]["status"], "missing_artifact")


if __name__ == "__main__":
    unittest.main()
