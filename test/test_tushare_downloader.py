import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import get_stock_data


class TushareDownloaderTest(unittest.TestCase):
    def test_token_file_is_used_without_environment_token(self):
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.txt"
            token_path.write_text("test-token\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"TUSHARE_TOKEN": "", "TUSHARE_MCP_URL": "", "TUSHARE_TOKEN_FILE": str(token_path)},
                clear=False,
            ):
                self.assertEqual(get_stock_data.get_tushare_token(), "test-token")

    def test_missing_ranges_only_append_without_backfill(self):
        ranges = get_stock_data.missing_ranges(
            (pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-20")),
            "2025-01-01",
            "2025-01-31",
            backfill=False,
        )
        self.assertEqual(ranges, [(pd.Timestamp("2025-01-21"), pd.Timestamp("2025-01-31"))])

    def test_normalize_output_deduplicates_and_orders_rows(self):
        rows = [
            ["000002", "2026/01/03", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ["000001", "2026/01/02", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ["000001", "2026/01/02", 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]
        normalized = get_stock_data.normalize_output(
            pd.DataFrame(rows, columns=get_stock_data.OUTPUT_COLUMNS)
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["股票代码"].tolist(), ["000001", "000002"])
        self.assertEqual(normalized.iloc[0]["开盘"], 2)

    def test_downloader_has_no_fallback_provider_imports(self):
        source = Path(ROOT, "get_stock_data.py").read_text(encoding="utf-8")
        self.assertNotIn("import akshare", source)
        self.assertNotIn("import baostock", source)


if __name__ == "__main__":
    unittest.main()
