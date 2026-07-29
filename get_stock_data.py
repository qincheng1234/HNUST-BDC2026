#!/usr/bin/env python3
"""
获取沪深300指数成分股历史数据
- 获取最近一期沪深300的300个成分股
- 抓取每只股票从指定开始日至今的历史量价数据
- 默认使用Tushare Pro，保留baostock/akshare兜底
- 保存格式: 股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅
"""

import baostock as bs
import akshare as ak
import pandas as pd
from datetime import datetime
import os
import time
import multiprocessing as mp
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import tushare as ts
except ImportError:
    ts = None


OUTPUT_COLUMNS = [
    '股票代码', '日期', '开盘', '收盘', '最高', '最低',
    '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
]


def login():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"登录失败: {lg.error_msg}")
    print("baostock登录成功")
    return lg


def logout():
    """登出baostock"""
    bs.logout()
    print("baostock已登出")


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    print("正在获取沪深300成分股列表...")
    
    rs = bs.query_hs300_stocks()
    
    if rs.error_code != '0':
        raise Exception(f"获取成分股失败: {rs.error_msg}")
    
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    print(f"获取到 {len(df)} 只沪深300成分股")
    return df


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据"""
    rs = bs.query_history_k_data_plus(bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="1")  # adjustflag="1"表示后复权
    
    if rs.error_code != '0':
        raise Exception(f"查询失败: {rs.error_msg}")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算振幅和涨跌额
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    
    # 转换日期格式 YYYY/MM/DD，避免 Windows 下 strftime("%-m") 不兼容
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%m/%d')
    
    # 提取纯数字股票代码（统一为6位格式，不足前面补0）
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
    df['code'] = df['code'].str.zfill(6)
    
    # 重命名列
    df = df.rename(columns={
        'code': '股票代码',
        'date': '日期',
        'open': '开盘',
        'close': '收盘',
        'high': '最高',
        'low': '最低',
        'volume': '成交量',
        'amount': '成交额',
        'turn': '换手率',
        'pctChg': '涨跌幅'
    })
    
    columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', 
               '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']
    df = df[columns]
    
    return df


def get_existing_stocks(output_path):
    """获取已经保存的股票代码列表"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, dtype={'股票代码': str})
        if '股票代码' in df.columns and len(df) > 0:
            return set(df['股票代码'].astype(str).str.zfill(6).unique())
    except:
        pass
    return set()


def get_stock_date_range(output_path, stock_code, start_date=None, end_date=None):
    """获取某只股票在现有数据中的日期范围（可限定目标时间窗）"""
    if not os.path.exists(output_path):
        return None, None
    try:
        df = pd.read_csv(output_path)
        if '股票代码' not in df.columns or '日期' not in df.columns:
            return None, None
        stock_df = df[df['股票代码'].astype(str).str.zfill(6) == stock_code].copy()
        if len(stock_df) == 0:
            return None, None

        # 解析日期
        stock_df.loc[:, '日期_dt'] = pd.to_datetime(stock_df['日期'], format='%Y/%m/%d', errors='coerce')
        stock_df = stock_df.dropna(subset=['日期_dt'])
        if len(stock_df) == 0:
            return None, None

        # 若设置了目标时间窗，仅统计目标区间内的数据覆盖情况
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            stock_df = stock_df[stock_df['日期_dt'] >= start_dt]
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            stock_df = stock_df[stock_df['日期_dt'] <= end_dt]
        if len(stock_df) == 0:
            return None, None

        return stock_df['日期_dt'].min().strftime('%Y-%m-%d'), stock_df['日期_dt'].max().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"  警告: 读取股票 {stock_code} 现有日期范围失败: {e}")
        return None, None


def parse_api_date(date_str):
    """将API返回的日期 YYYY-MM-DD 转为 datetime"""
    return datetime.strptime(date_str, '%Y-%m-%d')


def format_api_date(dt):
    """将datetime转为API日期格式 YYYY-MM-DD"""
    return dt.strftime('%Y-%m-%d')


def normalize_provider(provider):
    return (provider or "tushare").strip().lower()


def get_tushare_token():
    """Read a Tushare token without exposing it in source code or logs."""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    mcp_url = os.environ.get("TUSHARE_MCP_URL", "").strip()
    if mcp_url:
        query = parse_qs(urlparse(mcp_url).query)
        token_values = query.get("token")
        if token_values and token_values[0].strip():
            return token_values[0].strip()

    token_file = os.environ.get("TUSHARE_TOKEN_FILE", "./temp/tushare_token.txt").strip()
    if token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            return token

    return ""


def get_tushare_client():
    if ts is None:
        raise ImportError("未安装 tushare，请先运行 uv add tushare 或 uv sync 更新依赖")

    token = get_tushare_token()
    if not token:
        raise ValueError("未设置 TUSHARE_TOKEN；也可以设置 TUSHARE_MCP_URL 并在URL中带 token 参数")

    return ts.pro_api(token)


def pure_to_ts_code(stock_code):
    stock_code = str(stock_code).zfill(6)
    suffix = "SH" if stock_code.startswith("6") else "SZ"
    return f"{stock_code}.{suffix}"


def ts_code_to_pure(ts_code):
    return str(ts_code).split(".")[0].zfill(6)


def ts_code_to_baostock_code(ts_code):
    pure_code = ts_code_to_pure(ts_code)
    prefix = "sh" if str(ts_code).upper().endswith(".SH") else "sz"
    return f"{prefix}.{pure_code}"


def tushare_date(date_str):
    return pd.to_datetime(date_str).strftime("%Y%m%d")


def format_output_date(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y/%m/%d")


def call_tushare(func, *, max_retries=3, retry_sleep=3, **kwargs):
    """带简单重试的Tushare调用，避免偶发网络/限流导致整批中断。"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            print(f"  Tushare调用失败，{retry_sleep}秒后重试({attempt}/{max_retries}): {e}")
            time.sleep(retry_sleep)
    raise last_error


def get_hs300_stocks_tushare(pro, universe_date):
    """获取不晚于指定基准日的最近一期沪深300成分股。"""
    universe_dt = pd.to_datetime(universe_date)
    print(f"正在通过 Tushare 获取沪深300成分股列表（基准日: {universe_dt:%Y-%m-%d}）...")

    weight_df = pd.DataFrame()
    for years in (1, 3, 6):
        start_ymd = (universe_dt - pd.DateOffset(years=years)).strftime("%Y%m%d")
        end_ymd = universe_dt.strftime("%Y%m%d")
        weight_df = call_tushare(
            pro.index_weight,
            index_code="000300.SH",
            start_date=start_ymd,
            end_date=end_ymd,
            fields="index_code,con_code,trade_date,weight",
        )
        if weight_df is not None and not weight_df.empty:
            break

    if weight_df is None or weight_df.empty:
        raise ValueError("Tushare index_weight 未返回沪深300成分股数据")

    weight_df = weight_df.copy()
    weight_df["trade_date_dt"] = pd.to_datetime(weight_df["trade_date"], format="%Y%m%d", errors="coerce")
    valid_dates = weight_df.loc[weight_df["trade_date_dt"] <= universe_dt, "trade_date_dt"].dropna()
    if valid_dates.empty:
        raise ValueError(f"Tushare 未返回不晚于 {universe_dt:%Y-%m-%d} 的沪深300成分股数据")
    latest_trade_date = valid_dates.max().strftime("%Y%m%d")
    latest_df = (
        weight_df[weight_df["trade_date"].astype(str) == latest_trade_date]
        .drop_duplicates(subset=["con_code"], keep="first")
        .copy()
    )

    basic_df = call_tushare(
        pro.stock_basic,
        exchange="",
        list_status="L",
        fields="ts_code,name",
    )
    if basic_df is None:
        basic_df = pd.DataFrame(columns=["ts_code", "name"])

    latest_df = latest_df.merge(
        basic_df.rename(columns={"ts_code": "con_code", "name": "code_name"}),
        on="con_code",
        how="left",
    )
    latest_df["纯代码"] = latest_df["con_code"].map(ts_code_to_pure)
    latest_df["code"] = latest_df["con_code"].map(ts_code_to_baostock_code)
    latest_df["code_name"] = latest_df["code_name"].fillna(latest_df["con_code"])
    latest_df = latest_df.sort_values("纯代码").reset_index(drop=True)

    print(
        f"Tushare 获取到 {len(latest_df)} 只沪深300成分股，"
        f"成分日期: {pd.to_datetime(latest_trade_date).strftime('%Y-%m-%d')}"
    )
    return latest_df[["code", "code_name", "纯代码", "con_code", "trade_date", "weight"]]


def get_stock_turnover_tushare(pro, ts_code, start_ymd, end_ymd):
    """单独获取换手率；如果接口不可用则返回空表，后续用0兜底。"""
    try:
        basic_df = call_tushare(
            pro.daily_basic,
            ts_code=ts_code,
            start_date=start_ymd,
            end_date=end_ymd,
            fields="ts_code,trade_date,turnover_rate",
        )
    except Exception as e:
        print(f"  警告: Tushare daily_basic 获取换手率失败: {e}")
        return pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate"])

    if basic_df is None or basic_df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate"])
    return basic_df


def get_stock_history_tushare(pro, ts_code, start_date, end_date):
    """使用Tushare Pro获取单只股票后复权日线，字段对齐训练数据格式。"""
    ts_code = str(ts_code).upper()
    pure_code = ts_code_to_pure(ts_code)
    start_ymd = tushare_date(start_date)
    end_ymd = tushare_date(end_date)

    bar_df = call_tushare(
        ts.pro_bar,
        api=pro,
        ts_code=ts_code,
        adj="hfq",
        freq="D",
        start_date=start_ymd,
        end_date=end_ymd,
    )

    if bar_df is None or bar_df.empty:
        return None

    bar_df = bar_df.copy()
    required_columns = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    missing = required_columns - set(bar_df.columns)
    if missing:
        raise ValueError(f"Tushare pro_bar 缺少必要列: {sorted(missing)}")

    if "pre_close" not in bar_df.columns:
        bar_df["pre_close"] = pd.NA
    if "change" not in bar_df.columns:
        bar_df["change"] = pd.NA
    if "pct_chg" not in bar_df.columns:
        bar_df["pct_chg"] = pd.NA

    turnover_df = get_stock_turnover_tushare(pro, ts_code, start_ymd, end_ymd)
    if not turnover_df.empty:
        bar_df = bar_df.merge(
            turnover_df[["trade_date", "turnover_rate"]],
            on="trade_date",
            how="left",
        )
    else:
        bar_df["turnover_rate"] = 0.0

    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "turnover_rate"]
    for col in numeric_cols:
        bar_df[col] = pd.to_numeric(bar_df[col], errors="coerce")

    bar_df["date_dt"] = pd.to_datetime(bar_df["trade_date"], format="%Y%m%d", errors="coerce")
    bar_df = bar_df.dropna(subset=["date_dt", "open", "high", "low", "close"]).sort_values("date_dt").reset_index(drop=True)

    previous_close = bar_df["pre_close"].where(bar_df["pre_close"].notna(), bar_df["close"].shift(1))
    change = bar_df["change"].where(bar_df["change"].notna(), bar_df["close"] - previous_close)
    pct_chg = bar_df["pct_chg"].where(
        bar_df["pct_chg"].notna(),
        (bar_df["close"] / (previous_close + 1e-12) - 1.0) * 100.0,
    )

    out = pd.DataFrame(
        {
            "股票代码": pure_code,
            "日期": bar_df["date_dt"].dt.strftime("%Y/%m/%d"),
            "开盘": bar_df["open"],
            "收盘": bar_df["close"],
            "最高": bar_df["high"],
            "最低": bar_df["low"],
            "成交量": (bar_df["vol"].fillna(0.0) * 100.0).round(0),
            "成交额": (bar_df["amount"].fillna(0.0) * 1000.0).round(2),
            "振幅": ((bar_df["high"] - bar_df["low"]) / (previous_close + 1e-12) * 100.0).fillna(0.0).round(2),
            "涨跌额": change.fillna(0.0).round(2),
            "换手率": bar_df["turnover_rate"].fillna(0.0).round(4),
            "涨跌幅": pct_chg.fillna(0.0).round(4),
        }
    )
    return out[OUTPUT_COLUMNS]


def clean_tushare_output_frame(data):
    """Return valid rows with canonical code and parsed date columns."""
    code_column, date_column = OUTPUT_COLUMNS[:2]
    raw_codes = data[code_column].astype(str).str.strip()
    parsed_dates = pd.to_datetime(
        data[date_column].astype(str),
        format="%Y/%m/%d",
        errors="coerce",
    )
    valid_rows = raw_codes.str.fullmatch(r"\d{6}").fillna(False) & parsed_dates.notna()
    cleaned = data.loc[valid_rows].copy()
    cleaned[code_column] = raw_codes.loc[valid_rows]
    cleaned[date_column] = parsed_dates.loc[valid_rows]
    return cleaned, int((~valid_rows).sum())


def get_existing_tushare_coverage(output_path):
    """Return each locally stored stock's available date range in one file pass."""
    if not os.path.exists(output_path):
        return {}

    try:
        existing = pd.read_csv(
            output_path,
            dtype={OUTPUT_COLUMNS[0]: str},
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Unable to read existing Tushare coverage: {exc}")
        return {}

    existing, ignored_rows = clean_tushare_output_frame(existing)
    if ignored_rows:
        print(f"增量覆盖检查忽略 {ignored_rows} 条无效历史记录")
    if existing.empty:
        return {}

    code_column, date_column = OUTPUT_COLUMNS[:2]
    coverage = {}
    for stock_code, frame in existing.groupby(code_column, sort=False):
        coverage[stock_code] = (frame[date_column].min(), frame[date_column].max())
    return coverage


def get_incremental_tushare_ranges(existing_range, start_date, end_date, allow_early_backfill=False):
    """Build missing ranges; normal incremental runs append only the late tail."""
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    if start_dt > end_dt:
        raise ValueError(f"Invalid date range: {start_date} > {end_date}")
    if existing_range is None:
        return [(start_dt, end_dt)]

    existing_min, existing_max = (pd.Timestamp(value).normalize() for value in existing_range)
    ranges = []
    if allow_early_backfill and existing_min > start_dt:
        early_end = min(end_dt, existing_min - pd.Timedelta(days=1))
        if start_dt <= early_end:
            ranges.append((start_dt, early_end))
    if existing_max < end_dt:
        late_start = max(start_dt, existing_max + pd.Timedelta(days=1))
        if late_start <= end_dt:
            ranges.append((late_start, end_dt))
    return ranges


def deduplicate_tushare_output(output_path):
    """Keep one sorted record per stock/date after incremental appends."""
    if not os.path.exists(output_path):
        return 0

    code_column, date_column = OUTPUT_COLUMNS[:2]
    data = pd.read_csv(output_path, dtype={code_column: str})
    missing_columns = set(OUTPUT_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Existing output is missing required columns: {sorted(missing_columns)}")

    data, ignored_rows = clean_tushare_output_frame(data)
    if ignored_rows:
        print(f"增量去重忽略 {ignored_rows} 条无效记录")
    data = data.drop_duplicates([code_column, date_column], keep="last")
    data = data.sort_values([code_column, date_column]).reset_index(drop=True)
    data[date_column] = data[date_column].dt.strftime("%Y/%m/%d")
    temporary_path = f"{output_path}.dedup.tmp"
    try:
        data.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return len(data)


def download_with_tushare(save_dir, start_date, end_date, output_path, universe_date=None):
    """Download a fixed HS300 universe, optionally appending only missing dates."""
    pro = get_tushare_client()
    hs300_df = get_hs300_stocks_tushare(pro, universe_date or end_date)

    hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
    hs300_df.to_csv(hs300_list_path, index=False, encoding="utf-8-sig")

    resume_download = os.environ.get("TUSHARE_RESUME", "").strip().lower() in {"1", "true", "yes"}
    existing_coverage = get_existing_tushare_coverage(output_path) if resume_download else {}
    if resume_download and existing_coverage:
        print(f"增量模式：已检测到 {len(existing_coverage)} 只股票数据，将只下载缺失日期")
    elif os.path.exists(output_path):
        os.remove(output_path)

    failed_stocks = []
    total_records = 0
    total = len(hs300_df)
    sleep_seconds = float(os.environ.get("TUSHARE_SLEEP_SECONDS", "0.25"))
    allow_early_backfill = os.environ.get("TUSHARE_BACKFILL", "").strip().lower() in {"1", "true", "yes"}

    for idx, row in hs300_df.iterrows():
        ts_code = row["con_code"]
        stock_code = row["纯代码"]
        stock_name = row["code_name"]
        fetch_ranges = get_incremental_tushare_ranges(
            existing_coverage.get(stock_code),
            start_date,
            end_date,
            allow_early_backfill=allow_early_backfill,
        )
        if not fetch_ranges:
            print(f"\n[{idx + 1}/{total}] {stock_code} {stock_name} - 日期已完整，跳过")
            continue
        print(f"\n[{idx + 1}/{total}] {stock_code} {stock_name} - Tushare 增量获取")

        try:
            new_frames = []
            for fetch_start, fetch_end in fetch_ranges:
                stock_data = get_stock_history_tushare(
                    pro,
                    ts_code,
                    fetch_start.strftime("%Y-%m-%d"),
                    fetch_end.strftime("%Y-%m-%d"),
                )
                if stock_data is not None and not stock_data.empty:
                    new_frames.append(stock_data)
            if not new_frames:
                print("  无新增数据")
                continue

            stock_data = pd.concat(new_frames, ignore_index=True)
            stock_data.to_csv(
                output_path,
                mode="a",
                header=not os.path.exists(output_path),
                index=False,
                encoding="utf-8-sig",
            )
            total_records += len(stock_data)
            print(f"  获取成功，新增 {len(stock_data)} 条记录")
        except Exception as e:
            print(f"  失败: {e}")
            failed_stocks.append((stock_code, stock_name))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if (idx + 1) % 20 == 0:
            print(f"\n  --- 已处理 {idx + 1} 只 ---")

    if resume_download and os.path.exists(output_path):
        final_records = deduplicate_tushare_output(output_path)
        print(f"增量去重完成，当前共有 {final_records} 条记录")
    print_download_summary("Tushare 下载完成", save_dir, output_path, total_records, failed_stocks)


def filter_data_by_date_range(df, start_date, end_date):
    """过滤DataFrame，仅保留目标时间窗内的数据"""
    if df is None or df.empty:
        return df

    if '日期' not in df.columns:
        return df

    filtered = df.copy()
    filtered.loc[:, '日期_dt'] = pd.to_datetime(filtered['日期'], format='%Y/%m/%d', errors='coerce')
    filtered = filtered.dropna(subset=['日期_dt'])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered = filtered[(filtered['日期_dt'] >= start_dt) & (filtered['日期_dt'] <= end_dt)].copy()
    filtered = filtered.drop(columns=['日期_dt'])
    return filtered


def get_hs300_stocks_akshare():
    """使用 akshare 获取沪深300成分股列表，作为 baostock 不可用时的兜底。"""
    print("正在通过 akshare 中证指数接口获取沪深300成分股列表...")
    df = ak.index_stock_cons_csindex(symbol="000300")
    required_columns = {"成分券代码", "成分券名称"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"akshare 成分股接口缺少必要列: {sorted(missing)}")

    df = df.copy()
    df["纯代码"] = df["成分券代码"].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=["纯代码"], keep="first").reset_index(drop=True)
    df["code"] = df["纯代码"].apply(lambda code: f"sh.{code}" if code.startswith("6") else f"sz.{code}")
    df["code_name"] = df["成分券名称"].astype(str)
    print(f"akshare 获取到 {len(df)} 只沪深300成分股")
    return df


def get_stock_history_akshare(stock_code, start_date, end_date):
    """使用 akshare 腾讯日线接口获取单只股票历史数据，字段对齐训练数据格式。"""
    stock_code = str(stock_code).zfill(6)
    tx_symbol = f"sh{stock_code}" if stock_code.startswith("6") else f"sz{stock_code}"
    df = ak.stock_zh_a_hist_tx(
        symbol=tx_symbol,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="hfq",
    )

    if df is None or df.empty:
        return None

    required_columns = ["date", "open", "close", "high", "low", "amount"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"akshare 日线接口缺少必要列: {sorted(missing)}")

    df = df[required_columns].copy()
    for col in ["open", "close", "high", "low", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "close", "high", "low", "amount"]).copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)

    prev_close = df["close"].shift(1)
    volume = df["amount"]
    amount_proxy = volume * df["close"] * 100.0
    turnover_proxy = volume / (volume.rolling(20, min_periods=1).mean() + 1e-12) * 100.0

    out = pd.DataFrame(
        {
            "股票代码": stock_code,
            "日期": df["date_dt"].dt.strftime("%Y/%m/%d"),
            "开盘": df["open"],
            "收盘": df["close"],
            "最高": df["high"],
            "最低": df["low"],
            "成交量": volume,
            "成交额": amount_proxy,
            "振幅": ((df["high"] - df["low"]) / (prev_close + 1e-12) * 100.0).fillna(0.0).round(2),
            "涨跌额": (df["close"] - prev_close).fillna(0.0).round(2),
            "换手率": turnover_proxy.fillna(0.0).round(4),
            "涨跌幅": ((df["close"] / (prev_close + 1e-12) - 1.0) * 100.0).fillna(0.0).round(4),
        }
    )
    return out


def print_download_summary(title, save_dir, output_path, total_records, failed_stocks):
    print("\n" + "=" * 60)
    print(title)
    print(f"  - 新增记录: {total_records}")
    print(f"  - 失败股票: {len(failed_stocks)}")

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        date_series = pd.to_datetime(df["日期"], errors="coerce")
        print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"  - 总行数: {len(df)}")
        print(f"  - 股票数量: {df['股票代码'].astype(str).str.zfill(6).nunique()}")
        if len(date_series.dropna()) > 0:
            print(f"  - 时间范围: {date_series.min().date()} 至 {date_series.max().date()}")

    if failed_stocks:
        failed_df = pd.DataFrame(failed_stocks, columns=["股票代码", "股票名称"])
        failed_path = os.path.join(save_dir, "failed_stocks.csv")
        failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")
        print(f"失败股票列表已保存至: {failed_path}")


def download_with_akshare(save_dir, start_date, end_date, output_path):
    """通过 akshare 全量下载数据，用于 baostock 网络不可用时继续完成流程。"""
    hs300_df = get_hs300_stocks_akshare()

    hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
    hs300_df[["code", "code_name", "纯代码"]].to_csv(hs300_list_path, index=False, encoding="utf-8-sig")

    if os.path.exists(output_path):
        os.remove(output_path)

    failed_stocks = []
    total_records = 0
    total = len(hs300_df)

    for idx, row in hs300_df.iterrows():
        stock_code = row["纯代码"]
        stock_name = row["code_name"]
        print(f"\n[{idx + 1}/{total}] {stock_code} {stock_name} - akshare 全量获取")

        try:
            stock_data = get_stock_history_akshare(stock_code, start_date, end_date)
            if stock_data is None or stock_data.empty:
                print("  无数据")
                continue

            stock_data.to_csv(
                output_path,
                mode="a",
                header=not os.path.exists(output_path),
                index=False,
                encoding="utf-8-sig",
            )
            total_records += len(stock_data)
            print(f"  获取成功，新增 {len(stock_data)} 条记录")
        except Exception as e:
            print(f"  失败: {e}")
            failed_stocks.append((stock_code, stock_name))

        if (idx + 1) % 10 == 0:
            print(f"\n  --- 已处理 {idx + 1} 只，暂停2秒 ---")
            time.sleep(2)

    print_download_summary("akshare 全量下载完成", save_dir, output_path, total_records, failed_stocks)


def _baostock_login_probe(queue):
    try:
        lg = bs.login()
        queue.put((lg.error_code, lg.error_msg))
        if lg.error_code == "0":
            bs.logout()
    except Exception as e:
        queue.put(("exception", str(e)))


def can_login_baostock(timeout_seconds=30):
    """在子进程中探测 baostock，避免 login 长时间挂住阻断备用源。"""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_baostock_login_probe, args=(queue,))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return False, f"baostock.login() 超过 {timeout_seconds} 秒未返回"
    if queue.empty():
        return False, "baostock.login() 未返回可用状态"
    error_code, error_msg = queue.get()
    return error_code == "0", error_msg


def merge_stock_data(existing_df, new_df, stock_code):
    """合并现有数据和新数据，保持同一股票数据相邻"""
    if new_df is None or new_df.empty:
        return existing_df
    
    # 统一股票代码为6位字符串格式用于比较
    existing_df['股票代码_str'] = existing_df['股票代码'].astype(str).str.zfill(6)
    
    # 从现有数据中移除该股票的旧数据
    other_df = existing_df[existing_df['股票代码_str'] != stock_code].drop(columns=['股票代码_str'])
    
    # 获取该股票的现有数据
    stock_existing = existing_df[existing_df['股票代码_str'] == stock_code].drop(columns=['股票代码_str']) if stock_code in existing_df['股票代码_str'].values else pd.DataFrame()
    
    # 合并该股票的新旧数据
    if not stock_existing.empty:
        # 将日期转为datetime用于比较和去重
        stock_existing_copy = stock_existing.copy()
        new_df_copy = new_df.copy()
        stock_existing_copy['日期_dt'] = pd.to_datetime(stock_existing_copy['日期'], format='%Y/%m/%d')
        new_df_copy['日期_dt'] = pd.to_datetime(new_df_copy['日期'], format='%Y/%m/%d')
        
        # 合并并去重
        combined = pd.concat([stock_existing_copy, new_df_copy], ignore_index=True)
        combined = combined.drop_duplicates(subset=['日期_dt'], keep='last')
        combined = combined.sort_values('日期_dt')
        
        # 删除临时列
        combined = combined.drop(columns=['日期_dt'])
    else:
        combined = new_df
    
    # 重新组装：其他股票数据 + 该股票合并后的数据
    result = pd.concat([other_df, combined], ignore_index=True)
    return result


def main():
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)
    
    start_date = os.environ.get("STOCK_DATA_START_DATE", "2024-01-01")
    end_date = os.environ.get("STOCK_DATA_END_DATE", datetime.today().strftime("%Y-%m-%d"))
    universe_date = os.environ.get("STOCK_UNIVERSE_DATE", end_date)
    
    output_path = os.path.join(save_dir, "stock_data.csv")
    
    print(f"目标数据时间范围: {start_date} 至 {end_date}")
    print(f"沪深300成分股基准日: {universe_date}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    
    # 检查已有的数据
    existing_stocks = get_existing_stocks(output_path)
    if existing_stocks:
        print(f"发现已有数据，包含 {len(existing_stocks)} 只股票，将检查每只股票是否需要增量更新")
    
    provider = normalize_provider(os.environ.get("STOCK_DATA_PROVIDER", "tushare"))
    if provider in {"tushare", "ts"}:
        print("已选择 Tushare Pro 下载数据")
        download_with_tushare(save_dir, start_date, end_date, output_path, universe_date)
        return

    if provider in {"akshare", "akshare_tx", "tx"}:
        print("已选择 akshare 腾讯日线接口下载数据")
        download_with_akshare(save_dir, start_date, end_date, output_path)
        return

    if provider == "auto":
        try:
            print("自动模式：优先尝试 Tushare Pro 下载数据")
            download_with_tushare(save_dir, start_date, end_date, output_path, universe_date)
            return
        except Exception as e:
            print(f"Tushare 下载失败，继续尝试 baostock/akshare: {e}")

        can_login, probe_msg = can_login_baostock(timeout_seconds=30)
        if not can_login:
            print(f"baostock 探测失败，改用 akshare 腾讯日线接口下载: {probe_msg}")
            download_with_akshare(save_dir, start_date, end_date, output_path)
            return
    elif provider not in {"baostock", "bs"}:
        raise ValueError(f"未知 STOCK_DATA_PROVIDER: {provider}")

    # 登录baostock；若不可用，则改用 akshare 通过 HTTP/HTTPS 下载
    try:
        login()
    except Exception as e:
        print(f"baostock 登录失败，改用 akshare 腾讯日线接口下载: {e}")
        download_with_akshare(save_dir, start_date, end_date, output_path)
        return
    
    try:
        # 获取沪深300成分股
        hs300_df = get_hs300_stocks()
        
        # 保存成分股列表
        hs300_list_path = os.path.join(save_dir, "hs300_stock_list.csv")
        hs300_df.to_csv(hs300_list_path, index=False, encoding='utf-8-sig')
        
        # 读取现有数据（用于增量合并）
        existing_df = None
        if os.path.exists(output_path) and len(existing_stocks) > 0:
            try:
                existing_df = pd.read_csv(output_path)
                raw_len = len(existing_df)
                existing_df = filter_data_by_date_range(existing_df, start_date, end_date)
                filtered_len = len(existing_df)
                print(f"  已加载现有数据: {len(existing_df)} 条记录")
                if filtered_len != raw_len:
                    print(f"  已按目标区间过滤旧数据: {raw_len} -> {filtered_len}")
            except Exception as e:
                print(f"  警告: 读取现有数据失败: {e}")
        
        # 准备处理所有股票（统一为6位字符串格式）
        hs300_df['纯代码'] = hs300_df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        
        # 统计信息
        failed_stocks = []
        total = len(hs300_df)
        success_count = 0
        new_stock_count = 0
        incremental_count = 0
        total_new_records = 0
        
        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            stock_name = row.get('code_name', '')
            pure_code = row.get('纯代码', '')
            
            # 检查该股票是否已存在数据
            existing_min_date, existing_max_date = get_stock_date_range(output_path, pure_code, start_date, end_date)
            
            if existing_min_date and existing_max_date:
                # 已有数据，检查是否需要增量
                need_early = existing_min_date > start_date
                need_late = existing_max_date < end_date
                
                if not need_early and not need_late:
                    print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 数据已完整 ({existing_min_date} 至 {existing_max_date})，跳过")
                    continue
                
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 增量更新")
                print(f"  现有数据范围: {existing_min_date} 至 {existing_max_date}")
                
                # 计算需要获取的日期范围
                fetch_ranges = []
                if need_early:
                    fetch_start = start_date
                    fetch_end = (datetime.strptime(existing_min_date, '%Y-%m-%d') - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    fetch_ranges.append((fetch_start, fetch_end, "早期"))
                if need_late:
                    late_start = datetime.strptime(existing_max_date, '%Y-%m-%d') + pd.Timedelta(days=1)
                    fetch_start = max(pd.to_datetime(start_date), pd.to_datetime(late_start)).strftime('%Y-%m-%d')
                    fetch_end = end_date
                    fetch_ranges.append((fetch_start, fetch_end, "近期"))
            else:
                # 全新股票
                print(f"\n[{idx+1}/{total}] {bs_code} {stock_name} - 全新获取")
                fetch_ranges = [(start_date, end_date, "全量")]
            
            try:
                all_new_data = []
                for fetch_start, fetch_end, period_name in fetch_ranges:
                    print(f"  获取{period_name}数据: {fetch_start} 至 {fetch_end}")
                    stock_data = get_stock_history(bs_code, fetch_start, fetch_end)
                    if stock_data is not None and not stock_data.empty:
                        all_new_data.append(stock_data)
                
                if all_new_data:
                    new_data = pd.concat(all_new_data, ignore_index=True)
                    
                    if existing_df is not None and len(existing_df) > 0:
                        # 增量更新：合并数据并保持同一股票相邻
                        existing_df = merge_stock_data(existing_df, new_data, pure_code)
                        # 立即写回文件
                        existing_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                        incremental_count += 1
                    else:
                        # 首次写入
                        new_data.to_csv(output_path, index=False, encoding='utf-8-sig')
                        existing_df = new_data
                        new_stock_count += 1
                    
                    total_new_records += len(new_data)
                    success_count += 1
                    print(f"  ✓ 获取成功，新增 {len(new_data)} 条记录")
                else:
                    print(f"  ✗ 无新数据")
                    
            except Exception as e:
                print(f"  ✗ 失败: {e}")
                failed_stocks.append((bs_code, stock_name))
            
            # 每10只成功获取的股票暂停一下
            if success_count > 0 and success_count % 10 == 0:
                print(f"\n  --- 已处理 {success_count} 只，暂停2秒 ---")
                time.sleep(2)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("本次运行完成!")
        print(f"  - 全新获取: {new_stock_count} 只股票")
        print(f"  - 增量更新: {incremental_count} 只股票")
        print(f"  - 失败: {len(failed_stocks)} 只股票")
        print(f"  - 新增记录: {total_new_records}")
        
        # 验证总数据
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"\n文件总览:")
            print(f"  - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            print(f"  - 总行数: {len(df)}")
            print(f"  - 股票数量: {df['股票代码'].nunique()}")
            if len(df) > 0:
                print(f"  - 时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
                
                # 验证同一股票数据是否相邻
                stock_blocks = df.groupby('股票代码').apply(lambda x: x.index.max() - x.index.min() + 1).sum()
                if stock_blocks == len(df):
                    print("  - 数据组织: ✓ 同一股票数据相邻")
                else:
                    print(f"  - 数据组织: 警告，股票数据块总长度({stock_blocks})与总行数({len(df)})不一致")
                
                print("\n前3行数据预览:")
                print(df.head(3).to_string(index=False))
                print("\n最后3行数据预览:")
                print(df.tail(3).to_string(index=False))
        
        # 保存失败列表
        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=['股票代码', '股票名称'])
            failed_path = os.path.join(save_dir, "failed_stocks.csv")
            failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
            print(f"\n失败股票列表已保存至: {failed_path}")
    
    finally:
        logout()


if __name__ == "__main__":
    main()
