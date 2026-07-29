"""Split raw stock data into local training and held-out scoring files."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按交易日将 stock_data.csv 划分为训练集和本地评分测试集"
    )
    parser.add_argument("--input", default="data/stock_data.csv", help="原始行情文件路径")
    parser.add_argument("--output-dir", default="data", help="train.csv 与 test.csv 输出目录")
    parser.add_argument(
        "--auto-last-days",
        type=int,
        default=5,
        help="留作测试集的最后交易日数量；设为 0 时使用显式日期参数",
    )
    parser.add_argument("--train-start", default=None, help="训练集起始日期，可选")
    parser.add_argument("--train-end", default=None, help="训练集结束日期")
    parser.add_argument("--test-start", default=None, help="测试集起始日期")
    parser.add_argument("--test-end", default=None, help="测试集结束日期")
    return parser.parse_args()


def parse_date(value: str | None, argument_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"{argument_name} 不是有效日期: {value}")
    return date.normalize()


def select_date_range(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError(f"开始日期晚于结束日期: {start_date.date()} > {end_date.date()}")
    return frame.loc[frame["日期"].between(start_date, end_date)].copy()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    frame = pd.read_csv(input_path, dtype={"股票代码": str})
    required_columns = {"股票代码", "日期"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"输入文件缺少必要列: {sorted(missing_columns)}")

    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    if frame["日期"].isna().any():
        raise ValueError("输入文件包含无法解析的日期")

    sessions = pd.Index(frame["日期"].drop_duplicates().sort_values())
    train_start = parse_date(args.train_start, "--train-start")

    if args.auto_last_days < 0:
        raise ValueError("--auto-last-days 不能小于 0")
    if args.auto_last_days > 0:
        if len(sessions) <= args.auto_last_days:
            raise ValueError("交易日不足，无法留出测试集")
        test_sessions = sessions[-args.auto_last_days :]
        train_end = sessions[-args.auto_last_days - 1]
        train_start = train_start or sessions[0]
        train_frame = select_date_range(frame, train_start, train_end)
        test_frame = frame.loc[frame["日期"].isin(test_sessions)].copy()
    else:
        train_end = parse_date(args.train_end, "--train-end")
        test_start = parse_date(args.test_start, "--test-start")
        test_end = parse_date(args.test_end, "--test-end")
        if None in (train_start, train_end, test_start, test_end):
            raise ValueError("显式划分时必须提供 train/test 的起止日期")
        train_frame = select_date_range(frame, train_start, train_end)
        test_frame = select_date_range(frame, test_start, test_end)
        test_sessions = pd.Index(test_frame["日期"].drop_duplicates().sort_values())

    if train_frame.empty or test_frame.empty:
        raise ValueError("训练集或测试集为空，请检查日期参数")

    output_dir.mkdir(parents=True, exist_ok=True)
    for output_frame in (train_frame, test_frame):
        output_frame.sort_values(["股票代码", "日期"], inplace=True)
        output_frame["日期"] = output_frame["日期"].dt.strftime("%Y-%m-%d")

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    test_dates = ", ".join(test_sessions.strftime("%Y-%m-%d"))
    print(f"训练集: {train_path}，共 {len(train_frame)} 行，股票数 {train_frame['股票代码'].nunique()}")
    print(f"测试集: {test_path}，共 {len(test_frame)} 行，股票数 {test_frame['股票代码'].nunique()}")
    print(f"训练集日期范围: {train_frame['日期'].min()} ~ {train_frame['日期'].max()}")
    print(f"测试集交易日: {test_dates}")


if __name__ == "__main__":
    main()
