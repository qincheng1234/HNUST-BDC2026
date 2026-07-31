import copy
import json
import multiprocessing as mp
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from data_io import load_training_data
from model import build_model
from splits import build_walk_forward_folds, select_robust_epoch, trading_dates_from_frame
from utils import (
    add_cross_sectional_market_features,
    build_model_feature_columns,
    create_labeled_ranking_dataset,
    engineer_features_39,
    engineer_features_158plus39,
)


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# feature / column maps
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# data preprocessing
# ---------------------------------------------------------------------------

def _build_label_and_clean(processed, drop_small_open=True):
    horizon = int(config.get("label_horizon_days", 5))
    grouped = processed.groupby("股票代码", sort=False)
    processed["open_t1"] = grouped["开盘"].shift(-1)
    processed["open_th"] = grouped["开盘"].shift(-horizon)
    processed = processed[processed["open_t1"] > 1e-4] if drop_small_open else processed
    processed["label"] = (
        (processed["open_th"] - processed["open_t1"]) / (processed["open_t1"] + 1e-12)
    )
    processed = processed.dropna(subset=["label"])
    processed.drop(columns=["open_t1", "open_th"], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config["feature_num"] in FEATURE_ENGINEER_MAP, (
        f"Unsupported feature_num: {config['feature_num']}"
    )
    assert stockid2idx is not None, "stockid2idx cannot be empty"

    feature_engineer = FEATURE_ENGINEER_MAP[config["feature_num"]]
    feature_columns = build_model_feature_columns(
        FEATURE_COLUMNS_MAP[config["feature_num"]],
    )

    df = df.copy()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    print(f"Running {desc} feature engineering (multi-process)...")
    groups = [group for _, group in df.groupby("股票代码", sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc} input is empty")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(
            tqdm(
                pool.imap(feature_engineer, groups),
                total=len(groups),
                desc=desc,
            )
        )

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed["instrument"] = processed["股票代码"].map(stockid2idx)
    processed = processed.dropna(subset=["instrument"]).copy()
    processed["instrument"] = processed["instrument"].astype(np.int64)
    processed = add_cross_sectional_market_features(processed)
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    return _preprocess_common(
        df, stockid2idx,
        desc="Feature engineering",
        drop_small_open=is_train,
    )


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(
        df, stockid2idx,
        desc="Validation feature engineering",
        drop_small_open=True,
    )


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

class WeightedRankingLoss(nn.Module):
    """Combined listwise + pairwise ranking loss weighted toward top-k."""

    def __init__(self, temperature=1.0, k=5, weight_factor=2.0,
                 pairwise_weight=1, base_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        return (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()

    def pairwise_loss(self, y_pred, y_true, weights):
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted = pairwise * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        return (weighted.sum(dim=[1, 2]) / num_pairs).mean()

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, y_true[i].topk(k).indices] = self.weight_factor
        return (
            self.listwise_loss(y_pred, y_true, weights)
            + self.pairwise_weight * self.pairwise_loss(y_pred, y_true, weights)
        )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    batch_size = y_pred.size(0)
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    final_score_list = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        pred_return_sum = valid_true[pred_indices].sum().item()

        _, true_indices = torch.topk(valid_true, k)
        max_return_sum = valid_true[true_indices].sum().item()

        random_return_sum = k * valid_true.mean().item()

        ratio_pred = (
            pred_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9
            else 0.0
        )
        denominator = max_return_sum - random_return_sum
        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-6
            else 0.0
        )

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        final_score_list.append(final_score)

    return {
        "pred_return_sum": np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        "max_return_sum": np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        "random_return_sum": np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        "ratio_pred": np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        "final_score": np.mean(final_score_list) if final_score_list else 0.0,
    }


# ---------------------------------------------------------------------------
# dataset / collation
# ---------------------------------------------------------------------------

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequences": torch.FloatTensor(self.sequences[idx]),
            "targets": torch.FloatTensor(self.targets[idx]),
            "relevance": torch.LongTensor(self.relevance_scores[idx]),
            "stock_indices": torch.LongTensor(self.stock_indices[idx]),
        }


def collate_fn(batch):
    sequences = [item["sequences"] for item in batch]
    targets = [item["targets"] for item in batch]
    relevance = [item["relevance"] for item in batch]
    stock_indices = [item["stock_indices"] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, sid in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq = torch.cat([seq, torch.zeros(pad_size, seq.size(1), seq.size(2))], dim=0)
            tgt = torch.cat([tgt, torch.zeros(pad_size)], dim=0)
            rel = torch.cat([rel, torch.zeros(pad_size, dtype=torch.long)], dim=0)
            sid = torch.cat([sid, torch.zeros(pad_size, dtype=torch.long)], dim=0)
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(sid)
        masks.append(mask)

    return {
        "sequences": torch.stack(padded_sequences),
        "targets": torch.stack(padded_targets),
        "relevance": torch.stack(padded_relevance),
        "stock_indices": torch.stack(padded_stock_indices),
        "masks": torch.stack(masks),
    }


# ---------------------------------------------------------------------------
# training / eval loops
# ---------------------------------------------------------------------------

def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        sequences = batch["sequences"].to(device)
        targets = batch["targets"].to(device)
        relevance = batch["relevance"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        # model forward — returns [batch, stocks] ranking scores
        if getattr(model, "supports_cross_sectional_mask", False):
            outputs = model(sequences, mask=masks)
        else:
            outputs = model(sequences)

        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks

        batch_loss = None
        for i in range(sequences.size(0)):
            valid_indices = masks[i].nonzero().squeeze()
            if valid_indices.numel() == 0:
                continue
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            if len(valid_pred) > 1:
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / sequences.size(0)
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"],
            )
            optimizer.step()
            total_loss += batch_loss.item()

            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
            local_step += 1

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            sequences = batch["sequences"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)

            if getattr(model, "supports_cross_sectional_mask", False):
                outputs = model(sequences, mask=masks)
            else:
                outputs = model(sequences)

            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            for i in range(sequences.size(0)):
                valid_indices = masks[i].nonzero().squeeze()
                if valid_indices.numel() == 0:
                    continue
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(
                        len(valid_true), 0, -1, device=device, dtype=torch.float32,
                    )
                    relevance_scores = relevance_scores.detach()
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if not isinstance(batch_loss, torch.Tensor) else loss

            if batch_loss is not None:
                batch_loss = batch_loss / sequences.size(0)
                total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    return avg_loss, total_metrics


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scale_processed_data(processed, features, fit_end_date):
    scaled = processed.copy()
    scaled[features] = scaled[features].replace([np.inf, -np.inf], np.nan)
    fit_rows = scaled[
        scaled["日期"] <= pd.Timestamp(fit_end_date)
    ].dropna(subset=features)
    if fit_rows.empty:
        raise ValueError("No valid training rows available to fit the feature scaler")
    scaler = StandardScaler()
    scaler.fit(fit_rows[features])
    scaled = scaled.dropna(subset=features).copy()
    scaled[features] = scaler.transform(scaled[features])
    return scaled, scaler


def build_ranking_dataset(processed, features, sequence_length,
                          start_date=None, end_date=None):
    sequences, targets, relevance, stock_indices = create_labeled_ranking_dataset(
        processed, features, sequence_length,
        min_window_end_date=start_date,
        max_window_end_date=end_date,
    )
    return RankingDataset(sequences, targets, relevance, stock_indices)


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )


def fit_model(train_dataset, validation_dataset, input_dim, stock_count,
              device, epochs):
    model = build_model(
        input_dim=input_dim, config=config, num_stocks=stock_count,
    ).to(device)
    criterion = WeightedRankingLoss(
        k=5, temperature=1.0, weight_factor=config["top5_weight"],
        pairwise_weight=config["pairwise_weight"],
        base_weight=config.get("base_weight", 1.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5,
    )
    train_loader = build_loader(train_dataset, shuffle=True)
    validation_loader = (
        build_loader(validation_dataset, shuffle=False)
        if validation_dataset is not None else None
    )

    best_score = -float("inf")
    best_epoch = 0
    best_metrics = {}
    best_state = None
    epoch_scores = []

    for epoch in range(int(epochs)):
        train_ranking_model(model, train_loader, criterion, optimizer, device, epoch)
        if validation_loader is None:
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            continue

        _, metrics = evaluate_ranking_model(
            model, validation_loader, criterion, device, epoch,
        )
        score = float(metrics.get("pred_return_sum", 0.0))
        epoch_scores.append(score)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, best_score, epoch_scores


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward_validation(processed, features, stock_count, folds, device):
    records = []
    for fold_index, fold in enumerate(folds):
        set_seed(int(config.get("seed", 42)) + fold_index)
        scaled, _ = scale_processed_data(
            processed, features, fold["train_sample_end"],
        )
        train_dataset = build_ranking_dataset(
            scaled, features, config["sequence_length"],
            end_date=fold["train_sample_end"],
        )
        validation_dataset = build_ranking_dataset(
            scaled, features, config["sequence_length"],
            start_date=fold["validation_start"],
            end_date=fold["validation_end"],
        )
        _, best_epoch, metrics, score, epoch_scores = fit_model(
            train_dataset, validation_dataset,
            input_dim=len(features), stock_count=stock_count,
            device=device,
            epochs=config.get("cv_num_epochs", config["num_epochs"]),
        )
        records.append({
            "fold": fold["name"],
            "train_sample_end": fold["train_sample_end"].date().isoformat(),
            "validation_start": fold["validation_start"].date().isoformat(),
            "validation_end": fold["validation_end"].date().isoformat(),
            "best_epoch": best_epoch,
            "top5_mean_return": score,
            "epoch_scores": epoch_scores,
            "metrics": {name: float(value) for name, value in metrics.items()},
        })
        print(
            f"{fold['name']}: train<= {fold['train_sample_end'].date()}, "
            f"val={fold['validation_start'].date()}~{fold['validation_end'].date()}, "
            f"top5_mean_return={score:.6f}",
        )
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    set_seed(config.get("seed", 42))
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    full_df, stock_codes, data_file = load_training_data(
        config["data_path"], data_mode=config["data_mode"],
        expected_stock_count=config.get("competition_stock_count", 300),
        as_of_date=config.get("data_as_of_date"),
    )
    print(f"Data mode: {config['data_mode']}")
    print(f"Training input: {data_file}")
    print(
        f"Training data range: "
        f"{full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}",
    )

    stock_ids = sorted(stock_codes)
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    processed, features = preprocess_data(
        full_df, is_train=True, stockid2idx=stockid2idx,
    )
    processed["日期"] = pd.to_datetime(processed["日期"])

    folds = build_walk_forward_folds(
        trading_dates_from_frame(full_df),
        label_horizon=int(config["label_horizon_days"]),
        embargo_days=int(config["cv_embargo_days"]),
        validation_days=int(config["cv_validation_days"]),
        min_train_days=int(config["cv_min_train_days"]),
        cv_folds=int(config["cv_folds"]),
        warmup_days=int(config["feature_warmup_days"]),
    )
    device = resolve_device()
    oof_records = run_walk_forward_validation(
        processed, features, len(stock_ids), folds, device,
    )
    selected_epochs, epoch_selection = select_robust_epoch(
        [record["epoch_scores"] for record in oof_records],
        risk_penalty=float(config["cv_epoch_risk_penalty"]),
    )

    # final re-fit on all labelled history
    final_fit_end = processed["日期"].max()
    final_scaled, final_scaler = scale_processed_data(
        processed, features, final_fit_end,
    )
    final_dataset = build_ranking_dataset(
        final_scaled, features, config["sequence_length"],
    )
    set_seed(int(config.get("seed", 42)) + 10_000)
    final_model, _, _, _, _ = fit_model(
        final_dataset, validation_dataset=None,
        input_dim=len(features), stock_count=len(stock_ids),
        device=device, epochs=selected_epochs,
    )

    torch.save(final_model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    joblib.dump(final_scaler, os.path.join(output_dir, "scaler.pkl"))

    selected_oof_scores = np.asarray(
        [record["epoch_scores"][selected_epochs - 1] for record in oof_records],
        dtype=float,
    )
    oof_mean = float(selected_oof_scores.mean())
    metadata = {
        "input_file": str(data_file),
        "data_mode": config["data_mode"],
        "model_type": config["model_type"],
        "feature_schema_version": config["feature_schema_version"],
        "source_feature_set": config["feature_num"],
        "feature_count": len(features),
        "data_as_of_date": config.get("data_as_of_date"),
        "stock_ids": stock_ids,
        "features": features,
        "label_horizon_days": int(config["label_horizon_days"]),
        "cv_embargo_days": int(config["cv_embargo_days"]),
        "selected_epochs": selected_epochs,
        "oof_top5_mean_return": oof_mean,
        "oof_top5_std": float(selected_oof_scores.std()),
        "epoch_selection": epoch_selection,
        "oof_folds": oof_records,
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "final_score.txt"), "w", encoding="utf-8") as f:
        f.write(f"OOF top-5 mean return: {oof_mean:.6f}\n")
        f.write(f"Selected epochs: {selected_epochs}\n")

    print(f"Walk-forward OOF top-5 mean return: {oof_mean:.6f}")
    print(f"Final epochs: {selected_epochs}")
    return oof_mean


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    best_score = main()
    print(f"\n########## Training complete! OOF mean: {best_score:.6f} ##########")
