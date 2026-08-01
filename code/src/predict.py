import json
import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from data_io import load_prediction_data
from industry import load_industry_map
from model import build_model
from utils import (
    add_cross_sectional_market_features,
    build_model_feature_columns,
    engineer_features_39,
    engineer_features_158plus39,
)


FEATURE_COLUMNS_MAP = {
    "39": [
        "instrument", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅",
        "涨跌额", "换手率", "涨跌幅", "sma_5", "sma_20", "ema_12", "ema_26",
        "rsi", "macd", "macd_signal", "volume_change", "obv", "volume_ma_5",
        "volume_ma_20", "volume_ratio", "kdj_k", "kdj_d", "kdj_j", "boll_mid",
        "boll_std", "atr_14", "ema_60", "volatility_10", "volatility_20",
        "return_1", "return_5", "return_10", "high_low_spread",
        "open_close_spread", "high_close_spread", "low_close_spread",
    ],
    "158+39": [
        "instrument", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅",
        "涨跌额", "换手率", "涨跌幅", "KMID", "KLEN", "KMID2", "KUP", "KUP2",
        "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0", "HIGH0", "LOW0", "VWAP0",
        "ROC5", "ROC10", "ROC20", "ROC30", "ROC60", "MA5", "MA10", "MA20",
        "MA30", "MA60", "STD5", "STD10", "STD20", "STD30", "STD60", "BETA5",
        "BETA10", "BETA20", "BETA30", "BETA60", "RSQR5", "RSQR10", "RSQR20",
        "RSQR30", "RSQR60", "RESI5", "RESI10", "RESI20", "RESI30", "RESI60",
        "MAX5", "MAX10", "MAX20", "MAX30", "MAX60", "MIN5", "MIN10", "MIN20",
        "MIN30", "MIN60", "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
        "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60", "RANK5", "RANK10",
        "RANK20", "RANK30", "RANK60", "RSV5", "RSV10", "RSV20", "RSV30",
        "RSV60", "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60", "IMIN5",
        "IMIN10", "IMIN20", "IMIN30", "IMIN60", "IMXD5", "IMXD10", "IMXD20",
        "IMXD30", "IMXD60", "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
        "CORD5", "CORD10", "CORD20", "CORD30", "CORD60", "CNTP5", "CNTP10",
        "CNTP20", "CNTP30", "CNTP60", "CNTN5", "CNTN10", "CNTN20", "CNTN30",
        "CNTN60", "CNTD5", "CNTD10", "CNTD20", "CNTD30", "CNTD60", "SUMP5",
        "SUMP10", "SUMP20", "SUMP30", "SUMP60", "SUMN5", "SUMN10", "SUMN20",
        "SUMN30", "SUMN60", "SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
        "VMA5", "VMA10", "VMA20", "VMA30", "VMA60", "VSTD5", "VSTD10",
        "VSTD20", "VSTD30", "VSTD60", "WVMA5", "WVMA10", "WVMA20", "WVMA30",
        "WVMA60", "VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60",
        "VSUMN5", "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60", "VSUMD5",
        "VSUMD10", "VSUMD20", "VSUMD30", "VSUMD60", "sma_5", "sma_20",
        "ema_12", "ema_26", "rsi", "macd", "macd_signal", "volume_change",
        "obv", "volume_ma_5", "volume_ma_20", "volume_ratio", "kdj_k", "kdj_d",
        "kdj_j", "boll_mid", "boll_std", "atr_14", "ema_60", "volatility_10",
        "volatility_20", "return_1", "return_5", "return_10",
        "high_low_spread", "open_close_spread", "high_close_spread",
        "low_close_spread",
    ],
}

FEATURE_ENGINEER_MAP = {
    "39": engineer_features_39,
    "158+39": engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
    assert config["feature_num"] in FEATURE_ENGINEER_MAP, (
        f"Unsupported feature_num: {config['feature_num']}"
    )
    feature_engineer = FEATURE_ENGINEER_MAP[config["feature_num"]]
    feature_columns = build_model_feature_columns(
        FEATURE_COLUMNS_MAP[config["feature_num"]],
    )

    df = df.copy()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    groups = [group for _, group in df.groupby("股票代码", sort=False)]
    if len(groups) == 0:
        raise ValueError("Input data is empty, cannot predict")

    num_processes = min(10, mp.cpu_count())
    print(f"Feature engineering processes: {num_processes}")
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(
            tqdm(
                pool.imap(feature_engineer, groups),
                total=len(groups),
                desc="Prediction feature engineering",
            )
        )

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed["instrument"] = processed["股票代码"].map(stockid2idx)
    processed = processed.dropna(subset=["instrument"]).copy()
    processed["instrument"] = processed["instrument"].astype(np.int64)
    processed["日期"] = pd.to_datetime(processed["日期"])
    processed = add_cross_sectional_market_features(processed)
    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data["股票代码"] == stock_id)
            & (data["日期"] <= latest_date)
        ].sort_values("日期").tail(sequence_length)
        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError(
            "No valid stock sequences for prediction — check data and sequence_length"
        )
    return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_industry_path():
    """Return the best industry csv path, checking model dir first (Docker-safe)."""
    primary = os.path.join(config["output_dir"], "sw_industry.csv")
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(config["data_path"], "sw_industry.csv")
    if os.path.exists(fallback):
        return fallback
    return primary  # let load_industry_map report missing file


def sector_diversified_top_k(scores, stock_ids, industry_map, k=5,
                             max_per_sector=2, min_sectors=3):
    """Select top-k stocks with soft sector diversity.

    * At most *max_per_sector* picks from any one sector.
    * Aim for at least *min_sectors* distinct sectors (best-effort;
      does NOT force picking bad stocks from weak sectors).

    If min_sectors cannot be reached with the remaining candidate pool
    the constraint is silently relaxed — diversity is a preference,
    not a hard rule that would sacrifice return.

    Returns list of k stock_ids.
    """
    if not industry_map or len(industry_map) < 5:
        order = np.argsort(scores)[::-1]
        return [stock_ids[i] for i in order[:k]]

    industries = [industry_map.get(sid, f"__unknown_{i}__") for i, sid in enumerate(stock_ids)]
    order = np.argsort(scores)[::-1]

    selected = []
    sector_counts: dict[str, int] = {}

    # Phase 1 — greedy with per-sector cap
    for idx in order:
        sector = industries[idx]
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == k:
            break

    # Phase 2 — fill remaining slots if cap prevented k selections
    if len(selected) < k:
        for idx in order:
            if idx in selected:
                continue
            selected.append(idx)
            sector_counts[industries[idx]] = sector_counts.get(industries[idx], 0) + 1
            if len(selected) == k:
                break

    # Phase 3 — best-effort sector diversity improvement
    for _ in range(k):  # at most k swap attempts
        unique_now = len(sector_counts)
        if unique_now >= min_sectors:
            break

        # Find all sectors NOT yet represented
        present_sectors = set(sector_counts.keys())
        missing = {industries[i] for i in order if i not in selected} - present_sectors
        if not missing:
            break  # no new sectors available, accept current selection

        # Find the most replaceable selected stock:
        # lowest-scoring pick from a sector that has >1 representation
        candidates_to_swap = [
            (i, pos) for pos, i in enumerate(selected)
            if sector_counts[industries[i]] > 1
        ]
        if not candidates_to_swap:
            break  # every sector has exactly 1, can't improve

        # Sort by score ascending — we want to swap out the weakest duplicate
        candidates_to_swap.sort(key=lambda x: scores[x[0]])
        weakest_idx, weakest_pos = candidates_to_swap[0]
        weakest_sector = industries[weakest_idx]

        # Find the best unpicked stock from a missing sector
        best_new = None
        for idx in order:
            if idx in selected:
                continue
            if industries[idx] in missing:
                best_new = idx
                break

        if best_new is None:
            break  # no suitable replacement

        # Execute swap
        sector_counts[weakest_sector] -= 1
        selected[weakest_pos] = best_new
        new_sector = industries[best_new]
        sector_counts[new_sector] = sector_counts.get(new_sector, 0) + 1

    result = [stock_ids[i] for i in selected]
    unique_sectors = len({industries[i] for i in selected})
    print(f"  Sector-diversified top-{k}: "
          f"{unique_sectors} sectors, "
          f"max {max(sector_counts.values())}/sector")
    return result


def main():
    model_path = os.path.join(config["output_dir"], "best_model.pth")
    scaler_path = os.path.join(config["output_dir"], "scaler.pkl")
    metadata_path = os.path.join(config["output_dir"], "model_meta.json")
    output_path = os.path.join("./output/", "result.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for path, name in [
        (model_path, "model"),
        (scaler_path, "scaler"),
        (metadata_path, "metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    raw_df, available_codes, data_file = load_prediction_data(
        config["data_path"],
        data_mode=config["data_mode"],
        expected_stock_count=config.get("competition_stock_count", 300),
        as_of_date=config.get("data_as_of_date"),
    )
    latest_date = raw_df["日期"].max()

    # Load industry mapping for sector-diversified prediction
    industry_map = load_industry_map(_resolve_industry_path())

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("data_mode") != config["data_mode"]:
        raise ValueError(
            "Model data_mode does not match current DATA_MODE — re-train first"
        )
    if metadata.get("model_type") != config["model_type"]:
        raise ValueError(
            "Model type does not match current config — re-train first"
        )
    if metadata.get("feature_schema_version") != config["feature_schema_version"]:
        raise ValueError(
            "Feature schema version mismatch — re-train first"
        )
    if metadata.get("source_feature_set", config["feature_num"]) != config["feature_num"]:
        raise ValueError(
            "Feature source set mismatch — re-train first"
        )
    stock_ids = metadata["stock_ids"]
    if set(stock_ids) != set(available_codes):
        raise ValueError(
            "Model stock universe does not match current data — re-train"
        )
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, _ = preprocess_predict_data(raw_df, stockid2idx)
    features = metadata["features"]
    missing_features = sorted(set(features) - set(processed.columns))
    if missing_features:
        raise ValueError(f"Prediction features missing: {missing_features[:5]}")
    processed[features] = (
        processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    sequence_length = config["sequence_length"]
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed, features, sequence_length, stock_ids, latest_date,
    )

    device = resolve_device()
    model = build_model(
        input_dim=len(features), config=config, num_stocks=len(stock_ids),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        if getattr(model, "supports_cross_sectional_mask", False):
            scores = model(
                x,
                mask=torch.ones(x.shape[:2], dtype=torch.bool, device=device),
            )
        else:
            scores = model(x)
        scores = scores.squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]

    if len(ranked_stock_ids) < 5:
        raise ValueError(
            f"Insufficient stocks for prediction: {len(ranked_stock_ids)} < 5"
        )

    # Sector-diversified top-5 (falls back to unconstrained if no industry data)
    top5 = sector_diversified_top_k(
        scores, sequence_stock_ids, industry_map, k=5,
    )
    output_df = pd.DataFrame({
        "stock_id": top5,
        "weight": [0.2] * len(top5),
    })
    output_df.to_csv(output_path, index=False)

    print(f"Data mode: {config['data_mode']}")
    print(f"Prediction date: {latest_date.date()}")
    print(f"Stocks ranked: {len(ranked_stock_ids)}")
    print(f"Result written to: {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
