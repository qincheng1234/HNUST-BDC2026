"""Load the contest-mounted raw market data without split-file dependencies."""

from pathlib import Path
import re

import pandas as pd


STOCK_DATA_FILENAMES = ("stock_data.csv", "stock_data")
LOCAL_TRAIN_FILENAMES = ("train.csv",)
DATA_MODES = {"local_split", "stock_data"}
REQUIRED_COLUMNS = {
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "换手率",
}


def normalize_stock_code(value):
    text = str(value).strip()
    numeric_match = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    if numeric_match is not None:
        return numeric_match.group(1).zfill(6)

    code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return code_match.group(1) if code_match is not None else None


def resolve_stock_data_path(data_path, filenames=STOCK_DATA_FILENAMES, model_dir=None):
    """Find the stock data file, checking model_dir first (Docker-safe).

    When *model_dir* is provided it takes priority over *data_path* so that
    a copy of stock_data.csv stored alongside the model survives the Docker
    mounts (data/, output/, temp/ are overwritten at verification time).
    """
    search_dirs = []
    if model_dir is not None:
        search_dirs.append(Path(model_dir))
    search_dirs.append(Path(data_path))

    checked = []
    for base in search_dirs:
        for filename in filenames:
            candidate = base / filename
            checked.append(str(candidate))
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

    raise FileNotFoundError(f"Missing market-data file; checked: {', '.join(checked)}")


def load_contest_stock_data(
    data_path,
    expected_stock_count=300,
    as_of_date=None,
    filenames=STOCK_DATA_FILENAMES,
    model_dir=None,
):
    """Load normalized market data, optionally cutting off future local rows."""
    source_path = resolve_stock_data_path(data_path, filenames=filenames, model_dir=model_dir)
    frame = pd.read_csv(source_path, dtype={"股票代码": str})
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"stock_data is missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("stock_data must not be empty")

    data = frame.copy()
    data["股票代码"] = data["股票代码"].map(normalize_stock_code)
    if data["股票代码"].isna().any():
        raise ValueError("stock_data contains invalid stock codes")
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    if data["日期"].isna().any():
        raise ValueError("stock_data contains invalid trading dates")

    if as_of_date:
        cutoff = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(cutoff):
            raise ValueError(f"Invalid AS_OF_DATE: {as_of_date}")
        data = data[data["日期"] <= cutoff].copy()
        if data.empty:
            raise ValueError(f"stock_data has no rows on or before {cutoff.date()}")

    if data.duplicated(["股票代码", "日期"]).any():
        raise ValueError("stock_data has duplicate stock-code/date rows")

    stock_codes = frozenset(data["股票代码"].unique())
    if expected_stock_count is not None and len(stock_codes) != int(expected_stock_count):
        raise ValueError(
            f"stock_data must contain {expected_stock_count} stocks, found {len(stock_codes)}"
        )

    return data.sort_values(["股票代码", "日期"]).reset_index(drop=True), stock_codes, source_path


def load_training_data(data_path, data_mode, expected_stock_count=300, as_of_date=None,
                      model_dir=None):
    """Load the model-training history for local scoring or final submission."""
    if data_mode not in DATA_MODES:
        raise ValueError(f"Unsupported DATA_MODE: {data_mode}; expected one of {sorted(DATA_MODES)}")
    filenames = LOCAL_TRAIN_FILENAMES if data_mode == "local_split" else STOCK_DATA_FILENAMES
    return load_contest_stock_data(
        data_path,
        expected_stock_count=expected_stock_count,
        as_of_date=as_of_date,
        filenames=filenames,
        model_dir=model_dir,
    )


def load_prediction_data(data_path, data_mode, expected_stock_count=300, as_of_date=None,
                         model_dir=None):
    """Load prediction history without ever consuming the local held-out test week."""
    return load_training_data(
        data_path,
        data_mode=data_mode,
        expected_stock_count=expected_stock_count,
        as_of_date=as_of_date,
        model_dir=model_dir,
    )
