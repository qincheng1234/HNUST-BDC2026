#!/usr/bin/env python3
"""Download a fixed HS300 universe with Tushare Pro only."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    import tushare as ts
except ImportError:
    ts = None


OUTPUT_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def get_tushare_token():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    mcp_url = os.environ.get("TUSHARE_MCP_URL", "").strip()
    if mcp_url:
        token_values = parse_qs(urlparse(mcp_url).query).get("token", [])
        if token_values and token_values[0].strip():
            return token_values[0].strip()

    token_path = Path(os.environ.get("TUSHARE_TOKEN_FILE", "./temp/tushare_token.txt"))
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def get_tushare_client():
    if ts is None:
        raise ImportError("缺少 tushare 依赖，请先运行 uv sync")
    token = get_tushare_token()
    if not token:
        raise ValueError("未找到 Tushare token；请设置 TUSHARE_TOKEN 或 TUSHARE_TOKEN_FILE")
    return ts.pro_api(token)


def call_with_retry(operation, description):
    retries = int(os.environ.get("TUSHARE_RETRIES", "4"))
    retry_seconds = float(os.environ.get("TUSHARE_RETRY_SECONDS", "3"))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt < retries:
                print(f"  {description} 失败，{retry_seconds:.1f} 秒后重试 ({attempt}/{retries})：{error}")
                time.sleep(retry_seconds)
    raise last_error


def to_trade_date(value):
    return pd.to_datetime(value).strftime("%Y%m%d")


def to_stock_id(ts_code):
    return str(ts_code).split(".")[0].zfill(6)


def get_hs300_universe(pro, universe_date):
    universe_timestamp = pd.Timestamp(universe_date).normalize()
    weights = pd.DataFrame()
    for years in (1, 3, 6):
        start_date = (universe_timestamp - pd.DateOffset(years=years)).strftime("%Y%m%d")
        weights = call_with_retry(
            lambda: pro.index_weight(
                index_code="000300.SH",
                start_date=start_date,
                end_date=universe_timestamp.strftime("%Y%m%d"),
                fields="index_code,con_code,trade_date,weight",
            ),
            "获取沪深300成分股",
        )
        if weights is not None and not weights.empty:
            break
    if weights is None or weights.empty:
        raise ValueError(f"Tushare 未返回 {universe_timestamp:%Y-%m-%d} 前的沪深300成分股")

    weights = weights.copy()
    weights["trade_date_dt"] = pd.to_datetime(weights["trade_date"], format="%Y%m%d", errors="coerce")
    valid_dates = weights.loc[weights["trade_date_dt"] <= universe_timestamp, "trade_date_dt"].dropna()
    if valid_dates.empty:
        raise ValueError("Tushare 返回的沪深300成分股日期无效")
    effective_date = valid_dates.max().strftime("%Y%m%d")
    universe = weights.loc[weights["trade_date"].astype(str) == effective_date].copy()
    universe = universe.drop_duplicates(subset=["con_code"], keep="first")

    basic = call_with_retry(
        lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,name"),
        "获取股票名称",
    )
    if basic is None:
        basic = pd.DataFrame(columns=["ts_code", "name"])
    universe = universe.merge(
        basic.rename(columns={"ts_code": "con_code", "name": "stock_name"}),
        how="left",
        on="con_code",
    )
    universe["stock_id"] = universe["con_code"].map(to_stock_id)
    universe["stock_name"] = universe["stock_name"].fillna(universe["con_code"])
    universe = universe.sort_values("stock_id").reset_index(drop=True)
    if len(universe) != 300:
        raise ValueError(f"固定股票池应为 300 只，实际为 {len(universe)} 只")

    print(f"沪深300成分股：{len(universe)} 只，有效日期：{pd.Timestamp(effective_date):%Y-%m-%d}")
    return universe[["stock_id", "stock_name", "con_code", "trade_date", "weight"]]


def fetch_turnover(pro, ts_code, start_date, end_date):
    try:
        data = call_with_retry(
            lambda: pro.daily_basic(
                ts_code=ts_code,
                start_date=to_trade_date(start_date),
                end_date=to_trade_date(end_date),
                fields="ts_code,trade_date,turnover_rate",
            ),
            f"获取 {ts_code} 换手率",
        )
    except Exception as error:
        print(f"  {ts_code} 换手率不可用，使用 0 填充：{error}")
        return pd.DataFrame(columns=["trade_date", "turnover_rate"])
    if data is None or data.empty:
        return pd.DataFrame(columns=["trade_date", "turnover_rate"])
    return data[["trade_date", "turnover_rate"]].drop_duplicates("trade_date", keep="last")


def fetch_stock_history(pro, ts_code, start_date, end_date):
    bars = call_with_retry(
        lambda: ts.pro_bar(
            api=pro,
            ts_code=ts_code,
            adj="hfq",
            freq="D",
            start_date=to_trade_date(start_date),
            end_date=to_trade_date(end_date),
        ),
        f"获取 {ts_code} 日线",
    )
    if bars is None or bars.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"{ts_code} 日线缺少字段：{sorted(missing)}")

    bars = bars.copy()
    for column in ("pre_close", "change", "pct_chg"):
        if column not in bars:
            bars[column] = pd.NA
    turnover = fetch_turnover(pro, ts_code, start_date, end_date)
    bars = bars.merge(turnover, how="left", on="trade_date")
    for column in ("open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "turnover_rate"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    bars["date_dt"] = pd.to_datetime(bars["trade_date"], format="%Y%m%d", errors="coerce")
    bars = bars.dropna(subset=["date_dt", "open", "high", "low", "close"]).sort_values("date_dt")
    previous_close = bars["pre_close"].where(bars["pre_close"].notna(), bars["close"].shift(1))
    change = bars["change"].where(bars["change"].notna(), bars["close"] - previous_close)
    pct_change = bars["pct_chg"].where(
        bars["pct_chg"].notna(),
        (bars["close"] / (previous_close + 1e-12) - 1.0) * 100.0,
    )
    return pd.DataFrame(
        {
            "股票代码": to_stock_id(ts_code),
            "日期": bars["date_dt"].dt.strftime("%Y/%m/%d"),
            "开盘": bars["open"],
            "收盘": bars["close"],
            "最高": bars["high"],
            "最低": bars["low"],
            "成交量": (bars["vol"].fillna(0.0) * 100.0).round(0),
            "成交额": (bars["amount"].fillna(0.0) * 1000.0).round(2),
            "振幅": ((bars["high"] - bars["low"]) / (previous_close + 1e-12) * 100.0).fillna(0.0).round(2),
            "涨跌额": change.fillna(0.0).round(2),
            "换手率": bars["turnover_rate"].fillna(0.0).round(4),
            "涨跌幅": pct_change.fillna(0.0).round(4),
        }
    )[OUTPUT_COLUMNS]


def normalize_output(data):
    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    missing = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"数据文件缺少字段：{sorted(missing)}")
    normalized = data[OUTPUT_COLUMNS].copy()
    normalized["股票代码"] = normalized["股票代码"].astype(str).str.extract(r"(\d{6})", expand=False)
    normalized["日期_dt"] = pd.to_datetime(normalized["日期"], errors="coerce")
    normalized = normalized.dropna(subset=["股票代码", "日期_dt"])
    normalized = normalized.drop_duplicates(["股票代码", "日期_dt"], keep="last")
    normalized = normalized.sort_values(["股票代码", "日期_dt"]).reset_index(drop=True)
    normalized["日期"] = normalized.pop("日期_dt").dt.strftime("%Y/%m/%d")
    return normalized[OUTPUT_COLUMNS]


def load_existing_output(output_path, selected_stock_ids, resume):
    if not resume or not output_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = pd.read_csv(output_path, dtype={"股票代码": str})
    data = normalize_output(data)
    return data[data["股票代码"].isin(selected_stock_ids)].reset_index(drop=True)


def coverage_by_stock(data):
    if data.empty:
        return {}
    dated = data.copy()
    dated["日期"] = pd.to_datetime(dated["日期"])
    return {
        stock_id: (frame["日期"].min(), frame["日期"].max())
        for stock_id, frame in dated.groupby("股票代码", sort=False)
    }


def missing_ranges(existing_range, start_date, end_date, backfill):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError(f"开始日期晚于结束日期：{start:%Y-%m-%d} > {end:%Y-%m-%d}")
    if existing_range is None:
        return [(start, end)]
    existing_start, existing_end = existing_range
    ranges = []
    if backfill and existing_start > start:
        ranges.append((start, min(end, existing_start - pd.Timedelta(days=1))))
    if existing_end < end:
        ranges.append((max(start, existing_end + pd.Timedelta(days=1)), end))
    return [(range_start, range_end) for range_start, range_end in ranges if range_start <= range_end]


def write_output(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    normalize_output(data).to_csv(temporary_path, index=False, encoding="utf-8-sig")
    os.replace(temporary_path, output_path)


def download_data(save_dir, start_date, end_date, universe_date):
    pro = get_tushare_client()
    universe = get_hs300_universe(pro, universe_date)
    save_dir.mkdir(parents=True, exist_ok=True)
    universe.to_csv(save_dir / "hs300_stock_list.csv", index=False, encoding="utf-8-sig")

    output_path = save_dir / "stock_data.csv"
    resume = os.environ.get("TUSHARE_RESUME", "").strip().lower() in {"1", "true", "yes"}
    backfill = os.environ.get("TUSHARE_BACKFILL", "").strip().lower() in {"1", "true", "yes"}
    pause_seconds = float(os.environ.get("TUSHARE_SLEEP_SECONDS", "0.25"))
    save_every = max(1, int(os.environ.get("TUSHARE_SAVE_EVERY", "10")))
    stock_ids = set(universe["stock_id"])
    result = load_existing_output(output_path, stock_ids, resume)
    coverage = coverage_by_stock(result)
    pending = []
    failures = []

    if resume and not result.empty:
        print(f"增量更新：已加载 {len(coverage)} 只股票的历史数据")
    for index, row in universe.iterrows():
        stock_id = row["stock_id"]
        ranges = missing_ranges(coverage.get(stock_id), start_date, end_date, backfill)
        if not ranges:
            print(f"[{index + 1:03d}/300] {stock_id} 已是最新，跳过")
            continue
        print(f"[{index + 1:03d}/300] {stock_id} {row['stock_name']}")
        try:
            frames = [
                fetch_stock_history(pro, row["con_code"], range_start, range_end)
                for range_start, range_end in ranges
            ]
            frames = [frame for frame in frames if not frame.empty]
            if frames:
                downloaded = pd.concat(frames, ignore_index=True)
                pending.append(downloaded)
                dates = pd.to_datetime(downloaded["日期"])
                coverage[stock_id] = (dates.min(), dates.max())
                print(f"  新增 {len(downloaded)} 行")
            else:
                print("  目标区间无交易数据")
        except Exception as error:
            print(f"  下载失败：{error}")
            failures.append({"stock_id": stock_id, "stock_name": row["stock_name"], "error": str(error)})

        if len(pending) >= save_every:
            result = normalize_output(pd.concat([result, *pending], ignore_index=True))
            write_output(result, output_path)
            pending.clear()
            print(f"  已保存断点：{len(result)} 行")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if pending:
        result = normalize_output(pd.concat([result, *pending], ignore_index=True))
    if result.empty:
        raise RuntimeError("未下载到任何行情数据")
    write_output(result, output_path)
    failure_path = save_dir / "failed_stocks.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
    elif failure_path.exists():
        failure_path.unlink()

    print("=" * 60)
    print(f"输出文件：{output_path}")
    print(f"记录数：{len(result)}")
    print(f"股票数：{result['股票代码'].nunique()}")
    print(f"日期范围：{result['日期'].min()} 至 {result['日期'].max()}")
    print(f"失败股票数：{len(failures)}")


def main():
    save_dir = Path(os.environ.get("STOCK_DATA_SAVE_DIR", "./data"))
    start_date = os.environ.get("STOCK_DATA_START_DATE", "2024-01-01")
    end_date = os.environ.get("STOCK_DATA_END_DATE", datetime.today().strftime("%Y-%m-%d"))
    universe_date = os.environ.get("STOCK_UNIVERSE_DATE", end_date)
    print(f"Tushare-only 下载：{start_date} 至 {end_date}")
    print(f"沪深300固定成分股基准日：{universe_date}")
    download_data(save_dir, start_date, end_date, universe_date)


if __name__ == "__main__":
    main()
