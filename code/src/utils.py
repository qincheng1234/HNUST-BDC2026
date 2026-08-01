"""Feature engineering and cross-sectional dataset construction."""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# cross-sectional / market features
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_FEATURES = [
    "cs_return_1_rank",
    "cs_return_5_rank",
    "cs_return_10_rank",
    "cs_volatility_20_rank",
    "cs_turnover_rank",
    "cs_amount_rank",
    "cs_range_rank",
    "market_return_1",
    "market_return_5",
    "market_breadth_1",
    "market_dispersion_1",
    "market_volatility_20",
    "market_turnover_median",
    "relative_return_1",
    "relative_return_5",
    # regime-change indicators (v7)
    "market_skewness_1",
    "market_vol_ratio",
    "cs_reversal_5",
]


def build_model_feature_columns(base_columns):
    """Drop the instrument identifier and append cross-sectional features."""
    return [
        col for col in base_columns if col != "instrument"
    ] + CROSS_SECTIONAL_FEATURES


def add_cross_sectional_market_features(frame):
    """Add same-day market-state and stock-relative features (no future leak)."""
    required = {
        "日期", "成交额", "换手率", "return_1", "return_5", "return_10",
        "volatility_20", "high_low_spread", "开盘",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Cannot build cross-sectional features; missing columns: {sorted(missing)}"
        )

    data = frame.copy()
    data["日期"] = pd.to_datetime(data["日期"])
    grouped = data.groupby("日期", sort=False)

    data["cs_return_1_rank"] = grouped["return_1"].rank(pct=True)
    data["cs_return_5_rank"] = grouped["return_5"].rank(pct=True)
    data["cs_return_10_rank"] = grouped["return_10"].rank(pct=True)
    data["cs_volatility_20_rank"] = grouped["volatility_20"].rank(pct=True)
    data["cs_turnover_rank"] = grouped["换手率"].rank(pct=True)
    data["cs_amount_rank"] = (
        np.log1p(data["成交额"].clip(lower=0))
        .groupby(data["日期"])
        .rank(pct=True)
    )
    data["cs_range_rank"] = (
        (data["high_low_spread"] / data["开盘"].abs().clip(lower=1e-12))
        .groupby(data["日期"])
        .rank(pct=True)
    )

    data["market_return_1"] = grouped["return_1"].transform("median")
    data["market_return_5"] = grouped["return_5"].transform("median")
    data["market_breadth_1"] = grouped["return_1"].transform(
        lambda v: (v > 0).mean(),
    )
    data["market_dispersion_1"] = grouped["return_1"].transform("std").fillna(0.0)
    data["market_volatility_20"] = grouped["volatility_20"].transform("median")
    data["market_turnover_median"] = grouped["换手率"].transform("median")
    data["relative_return_1"] = data["return_1"] - data["market_return_1"]
    data["relative_return_5"] = data["return_5"] - data["market_return_5"]

    # ---- regime-change features (v7) ----
    # market_skewness_1: cross-sectional return skew → fat-tail / crash signal
    data["market_skewness_1"] = (
        grouped["return_1"]
        .transform(lambda v: v.skew() if len(v) >= 10 else 0.0)
        .fillna(0.0)
    )

    # market_vol_ratio:  mean return of top-20%-vol stocks / bottom-20%-vol stocks
    # >1 = risk-taking rewarded (momentum regime), <1 = defense dominates
    def _vol_ratio(grp):
        if len(grp) < 20:
            return 0.0
        hi = grp["return_1"][grp["volatility_20"] >= grp["volatility_20"].quantile(0.8)]
        lo = grp["return_1"][grp["volatility_20"] <= grp["volatility_20"].quantile(0.2)]
        hi_mean = hi.mean() if len(hi) > 0 else 0.0
        lo_mean = lo.mean() if len(lo) > 0 else 0.0
        return hi_mean / (abs(lo_mean) + 1e-6)

    vol_ratios = []
    for _, grp in data.groupby("日期", sort=False):
        val = _vol_ratio(grp)
        vol_ratios.extend([val] * len(grp))
    data["market_vol_ratio"] = vol_ratios

    # cs_reversal_5: cross-sectional rank of reversal signal
    #   reversal = today's return / (abs(5-day return) + ε)
    # High rank → stock that bounced today after being weak for 5 days
    reversal_raw = data["return_1"] / (data["return_5"].abs() + 1e-6)
    data["cs_reversal_5"] = (
        reversal_raw.groupby(data["日期"]).rank(pct=True).fillna(0.5)
    )

    data[CROSS_SECTIONAL_FEATURES] = (
        data[CROSS_SECTIONAL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return data


# ---------------------------------------------------------------------------
# Alpha158 feature engineering
# ---------------------------------------------------------------------------

def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta)) ** 2)


def engineer_features(df):
    """Alpha158-style features accelerated with TA-Lib."""
    try:
        import talib
    except ImportError:
        print("Please install TA-Lib: pip install TA-Lib")
        raise

    df = df.copy()
    open_ = df["开盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)
    vwap = df["成交额"] / (volume + 1e-12)

    features = []
    feature_names = []

    # 1. K-line features (9)
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12),
    ])
    feature_names.extend([
        "KMID", "KLEN", "KMID2", "KUP", "KUP2",
        "KLOW", "KLOW2", "KSFT", "KSFT2",
    ])

    # 2. Price-related (4)
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12),
    ])
    feature_names.extend(["OPEN0", "HIGH0", "LOW0", "VWAP0"])

    windows = [5, 10, 20, 30, 60]

    # 3. ROC (5)
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f"ROC{w}")

    # 4. Moving averages (5)
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MA{w}")

    # 5. Std dev (5)
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"STD{w}")

    # 6. Regression (15)
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f"BETA{w}")
        rolling_corr = talib.CORREL(
            close,
            np.arange(len(close), dtype=float),
            timeperiod=w,
        )
        features.append(rolling_corr ** 2)
        feature_names.append(f"RSQR{w}")
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        features.append((close - predicted) / (close + 1e-12))
        feature_names.append(f"RESI{w}")

    # 7. Max/Min (10)
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MAX{w}")
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MIN{w}")

    # 8. Quantile (10)
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f"QTLU{w}")
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f"QTLD{w}")

    # 9. Rank (5)
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f"RANK{w}")

    # 10. Stochastic oscillator (5)
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f"RSV{w}")

    # 11. Index of Max/Min (15)
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f"IMAX{w}")
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f"IMIN{w}")
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f"IMXD{w}")

    # 12. Correlation (10)
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f"CORR{w}")
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(
            talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w),
        )
        feature_names.append(f"CORD{w}")

    # 13. Count (15)
    close_diff_pos = close > close.shift(1)
    close_diff_neg = close < close.shift(1)
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f"CNTP{w}")
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f"CNTN{w}")
    for w in windows:
        features.append(
            close_diff_pos.rolling(w).mean() - close_diff_neg.rolling(w).mean(),
        )
        feature_names.append(f"CNTD{w}")

    # 14. Sum of price changes (15)
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        features.append(
            close_diff_up.rolling(w).sum()
            / (close_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"SUMP{w}")
    for w in windows:
        features.append(
            close_diff_down.rolling(w).sum()
            / (close_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"SUMN{w}")
    for w in windows:
        features.append(
            (close_diff_up.rolling(w).sum() - close_diff_down.rolling(w).sum())
            / (close_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"SUMD{w}")

    # 15. Volume MA / std (10)
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f"VMA{w}")
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f"VSTD{w}")

    # 16. Weighted volume (5)
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vwr = vol_weighted_ret.rolling(w).mean()
        std_vwr = vol_weighted_ret.rolling(w).std()
        features.append(std_vwr / (mean_vwr + 1e-12))
        feature_names.append(f"WVMA{w}")

    # 17. Volume change sum (15)
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        features.append(
            volume_diff_up.rolling(w).sum()
            / (volume_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"VSUMP{w}")
    for w in windows:
        features.append(
            volume_diff_down.rolling(w).sum()
            / (volume_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"VSUMN{w}")
    for w in windows:
        features.append(
            (volume_diff_up.rolling(w).sum() - volume_diff_down.rolling(w).sum())
            / (volume_diff_abs.rolling(w).sum() + 1e-12),
        )
        feature_names.append(f"VSUMD{w}")

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    df = pd.concat([df, feature_df], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def engineer_features_39(df):
    """39 technical indicators (SMA, EMA, MACD, RSI, KDJ, Bollinger, etc.)."""
    try:
        import talib
    except ImportError:
        print("Please install TA-Lib: pip install TA-Lib")
        raise

    df = df.copy()
    open_ = df["开盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)

    df["sma_5"] = talib.SMA(close, timeperiod=5)
    df["sma_20"] = talib.SMA(close, timeperiod=20)
    df["ema_12"] = talib.EMA(close, timeperiod=12)
    df["ema_26"] = talib.EMA(close, timeperiod=26)
    df["ema_60"] = talib.EMA(close, timeperiod=60)

    df["macd"], df["macd_signal"], _ = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9,
    )
    df["rsi"] = talib.RSI(close, timeperiod=14)
    df["kdj_k"], df["kdj_d"] = talib.STOCH(
        high, low, close, fastk_period=9, slowk_period=3, slowd_period=3,
    )
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    upper, middle, lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
    )
    df["boll_mid"] = middle
    df["boll_std"] = (upper - middle) / 2

    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["obv"] = talib.OBV(close, volume)
    df["volume_change"] = volume.pct_change()
    df["volume_ma_5"] = talib.SMA(volume, timeperiod=5)
    df["volume_ma_20"] = talib.SMA(volume, timeperiod=20)
    df["volume_ratio"] = df["volume_ma_5"] / df["volume_ma_20"]

    df["return_1"] = close.pct_change(1)
    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    df["high_low_spread"] = high - low
    df["open_close_spread"] = open_ - close
    df["high_close_spread"] = high - close
    df["low_close_spread"] = low - close

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def engineer_features_158plus39(df):
    """Concatenate Alpha158 + 39 technical indicator features."""
    df_158 = engineer_features(df.copy())
    df_39 = engineer_features_39(df.copy())

    feature_cols_39 = [
        "sma_5", "sma_20", "ema_12", "ema_26", "rsi", "macd", "macd_signal",
        "volume_change", "obv", "volume_ma_5", "volume_ma_20", "volume_ratio",
        "kdj_k", "kdj_d", "kdj_j", "boll_mid", "boll_std", "atr_14", "ema_60",
        "volatility_10", "volatility_20", "return_1", "return_5", "return_10",
        "high_low_spread", "open_close_spread", "high_close_spread",
        "low_close_spread",
    ]
    feature_cols_39_exist = [c for c in feature_cols_39 if c in df_39.columns]

    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    return df_final


# ---------------------------------------------------------------------------
# ranking dataset construction
# ---------------------------------------------------------------------------

def create_labeled_ranking_dataset(
    data,
    features,
    sequence_length,
    min_window_end_date=None,
    max_window_end_date=None,
):
    """Build date-level ranking samples from rows with pre-computed labels."""
    prepared = data.copy()
    prepared["日期"] = pd.to_datetime(prepared["日期"])
    prepared = prepared.dropna(subset=["label"])
    prepared = prepared.sort_values(["instrument", "日期"]).reset_index(drop=True)

    min_date = (
        pd.Timestamp(min_window_end_date) if min_window_end_date is not None else None
    )
    max_date = (
        pd.Timestamp(max_window_end_date) if max_window_end_date is not None else None
    )
    windows = []

    for instrument, group in prepared.groupby("instrument", sort=False):
        if len(group) < sequence_length:
            continue
        feature_values = group[features].to_numpy(dtype=np.float32, copy=True)
        labels = group["label"].to_numpy(dtype=np.float32, copy=True)
        dates = group["日期"].to_numpy()

        for start in range(len(group) - sequence_length + 1):
            end = start + sequence_length - 1
            end_date = pd.Timestamp(dates[end])
            if min_date is not None and end_date < min_date:
                continue
            if max_date is not None and end_date > max_date:
                continue
            windows.append((
                end_date,
                int(instrument),
                feature_values[start : end + 1],
                labels[end],
            ))

    if not windows:
        raise ValueError("No labeled ranking samples available for this period")

    window_frame = pd.DataFrame(
        windows, columns=["date", "instrument", "sequence", "target"],
    )
    sequences, targets, relevance_scores, stock_indices = [], [], [], []
    for _, group in window_frame.groupby("date", sort=True):
        if len(group) < 10:
            continue
        day_targets = group["target"].to_numpy(dtype=np.float32, copy=True)
        order = np.argsort(day_targets)[::-1]
        relevance = np.empty(len(day_targets), dtype=np.float32)
        relevance[order] = np.arange(len(day_targets), 0, -1, dtype=np.float32)
        sequences.append(np.stack(group["sequence"].to_numpy()))
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(
            group["instrument"].to_numpy(dtype=np.int64, copy=True),
        )

    if not sequences:
        raise ValueError("No complete cross-sectional ranking samples available")
    return sequences, targets, relevance_scores, stock_indices
