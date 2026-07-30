"""Create a local holdout split from the latest trading sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source stock_data.csv path")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv and test.csv")
    parser.add_argument(
        "--auto-last-days",
        type=int,
        default=5,
        help="Number of latest trading sessions reserved for test.csv (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.auto_last_days <= 0:
        raise ValueError("--auto-last-days must be positive")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frame = pd.read_csv(input_path, dtype={"股票代码": str})
    if "日期" not in frame.columns:
        raise ValueError("Input data must contain a 日期 column")

    frame["日期"] = pd.to_datetime(frame["日期"])
    trading_dates = pd.DatetimeIndex(sorted(frame["日期"].dropna().unique()))
    if len(trading_dates) <= args.auto_last_days:
        raise ValueError("Input data does not contain enough trading sessions to split")

    first_test_date = trading_dates[-args.auto_last_days]
    train_frame = frame.loc[frame["日期"] < first_test_date].copy()
    test_frame = frame.loc[frame["日期"] >= first_test_date].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("Split produced an empty train or test dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_frame["日期"] = train_frame["日期"].dt.strftime("%Y-%m-%d")
    test_frame["日期"] = test_frame["日期"].dt.strftime("%Y-%m-%d")
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    print(f"Training data: {train_path} ({train_frame['日期'].min()} ~ {train_frame['日期'].max()})")
    print(f"Holdout data: {test_path} ({test_frame['日期'].min()} ~ {test_frame['日期'].max()})")


if __name__ == "__main__":
    main()
