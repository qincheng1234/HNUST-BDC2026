import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "split_train_test.py"


class SplitTrainTestScriptTests(unittest.TestCase):
    def test_auto_last_sessions_are_written_to_test_set(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "stock_data.csv"
            output_path = temporary_path / "output"
            source = pd.DataFrame(
                {
                    "股票代码": ["000001", "000001", "000001", "000002", "000002", "000002"],
                    "日期": [
                        "2026-07-21",
                        "2026-07-22",
                        "2026-07-23",
                        "2026-07-21",
                        "2026-07-22",
                        "2026-07-23",
                    ],
                    "开盘": [1, 2, 3, 4, 5, 6],
                }
            )
            source.to_csv(input_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_path),
                    "--auto-last-days",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            train = pd.read_csv(output_path / "train.csv", dtype={"股票代码": str})
            test = pd.read_csv(output_path / "test.csv", dtype={"股票代码": str})

        self.assertEqual(sorted(train["日期"].unique()), ["2026-07-21", "2026-07-22"])
        self.assertEqual(sorted(test["日期"].unique()), ["2026-07-23"])
        self.assertEqual(test["股票代码"].tolist(), ["000001", "000002"])
