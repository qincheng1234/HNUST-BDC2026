import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import build_model
from utils import add_cross_sectional_market_features, build_model_feature_columns
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_labeled_ranking_dataset, create_multitask_ranking_dataset, create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random
import copy
from data_io import load_training_data
from splits import build_walk_forward_folds, select_robust_epoch, trading_dates_from_frame
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    horizon = int(config.get('label_horizon_days', 5))
    grouped = processed.groupby('股票代码', sort=False)
    processed['open_t1'] = grouped['开盘'].shift(-1)
    processed['open_th'] = grouped['开盘'].shift(-horizon)
    processed['close_t1'] = grouped['收盘'].shift(-1)
    processed['open_t3'] = grouped['开盘'].shift(-3)
    future_open_prices = pd.concat(
        [grouped['开盘'].shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    )

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_th'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label_h1'] = (processed['close_t1'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label_h3'] = (processed['open_t3'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label_h5'] = processed['label']
    processed['downside_5'] = (
        1.0 - future_open_prices.min(axis=1) / (processed['open_t1'] + 1e-12)
    ).clip(lower=0.0)
    processed = processed.dropna(subset=['label', 'label_h1', 'label_h3', 'label_h5', 'downside_5'])

    processed.drop(columns=['open_t1', 'open_th', 'close_t1', 'open_t3'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = build_model_feature_columns(feature_cloums_map[config['feature_num']])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = add_cross_sectional_market_features(processed)
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失 (KL散度 + Cross Entropy)"""
        
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)

        # 加权 Cross Entropy（原实现未使用 weights）
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()
        
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        
        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()
        
        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)
        
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        
        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix
        
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        
        return loss
        
    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        _, top_indices = torch.topk(y_true, k, dim=1)
        
        # 2. 创建权重向量
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
            
        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        
        # 组合两种损失
        total_loss = listwise + self.pairwise_weight * pairwise
        
        return total_loss


class AdaptiveMultiTaskRankingLoss(nn.Module):
    """Combine ranking, multi-horizon return and downside supervision adaptively."""

    def __init__(self, ranking_loss):
        super().__init__()
        self.ranking_loss = ranking_loss
        self.log_task_scales = nn.Parameter(torch.zeros(3))

    def rank_loss(self, predictions, relevance):
        return self.ranking_loss(predictions, relevance)

    def combine(self, ranking_loss, horizon_prediction, horizon_target, downside_prediction, downside_target, mask):
        valid_mask = mask.bool()
        if not valid_mask.any():
            return ranking_loss
        horizon_loss = F.smooth_l1_loss(
            horizon_prediction[valid_mask],
            horizon_target[valid_mask],
        )
        downside_loss = F.smooth_l1_loss(
            downside_prediction[valid_mask],
            downside_target[valid_mask],
        )
        task_losses = torch.stack([ranking_loss, horizon_loss, downside_loss])
        log_scales = self.log_task_scales.clamp(min=-3.0, max=3.0)
        return torch.sum(torch.exp(-log_scales) * task_losses + log_scales)

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(
        self,
        sequences,
        targets,
        relevance_scores,
        stock_indices,
        auxiliary_returns=None,
        downside_targets=None,
    ):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.auxiliary_returns = auxiliary_returns
        self.downside_targets = downside_targets
        if (auxiliary_returns is None) != (downside_targets is None):
            raise ValueError('Auxiliary returns and downside targets must be provided together')
        if auxiliary_returns is not None and len(auxiliary_returns) != len(sequences):
            raise ValueError('Auxiliary target count must match sequence count')
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(self.relevance_scores[idx]),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }
        if self.auxiliary_returns is not None:
            item['auxiliary_returns'] = torch.FloatTensor(self.auxiliary_returns[idx])
            item['downside_targets'] = torch.FloatTensor(self.downside_targets[idx])
        return item

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    padded_auxiliary_returns = []
    padded_downside_targets = []
    masks = []
    
    for item, seq, tgt, rel, stock_idx in zip(batch, sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
        if 'auxiliary_returns' in batch[0]:
            auxiliary_returns = item['auxiliary_returns']
            downside_targets = item['downside_targets']
            if num_stocks < max_stocks:
                auxiliary_pad = torch.zeros(pad_size, auxiliary_returns.size(1))
                downside_pad = torch.zeros(pad_size)
                auxiliary_returns = torch.cat([auxiliary_returns, auxiliary_pad], dim=0)
                downside_targets = torch.cat([downside_targets, downside_pad], dim=0)
            padded_auxiliary_returns.append(auxiliary_returns)
            padded_downside_targets.append(downside_targets)
    
    collated = {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }
    if padded_auxiliary_returns:
        collated['auxiliary_returns'] = torch.stack(padded_auxiliary_returns)
        collated['downside_targets'] = torch.stack(padded_downside_targets)
    return collated


def _forward_ranking_model(model, sequences, masks):
    """Normalize legacy and multi-task model outputs for the training loop."""
    if getattr(model, 'supports_cross_sectional_mask', False):
        outputs = model(sequences, mask=masks)
    else:
        outputs = model(sequences)
    if isinstance(outputs, dict):
        return outputs['ranking_score'], outputs['horizon_return'], outputs['downside_risk']
    return outputs, None, None


def _ranking_loss(criterion, predictions, relevance):
    if isinstance(criterion, AdaptiveMultiTaskRankingLoss):
        return criterion.rank_loss(predictions, relevance)
    return criterion(predictions, relevance)

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs, horizon_outputs, downside_outputs = _forward_ranking_model(model, sequences, masks)
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 直接使用预处理好的相关性得分，无需重新计算
                loss = _ranking_loss(criterion, valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            if isinstance(criterion, AdaptiveMultiTaskRankingLoss):
                auxiliary_returns = batch['auxiliary_returns'].to(device)
                downside_targets = batch['downside_targets'].to(device)
                batch_loss = criterion.combine(
                    batch_loss,
                    horizon_outputs,
                    auxiliary_returns,
                    downside_outputs,
                    downside_targets,
                    masks,
                )
            batch_loss.backward()
            if not config.get('drop_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs, horizon_outputs, downside_outputs = _forward_ranking_model(model, sequences, masks)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    
                    loss = _ranking_loss(criterion, valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                if isinstance(criterion, AdaptiveMultiTaskRankingLoss):
                    auxiliary_returns = batch['auxiliary_returns'].to(device)
                    downside_targets = batch['downside_targets'].to(device)
                    batch_loss = criterion.combine(
                        batch_loss,
                        horizon_outputs,
                        auxiliary_returns,
                        downside_outputs,
                        downside_targets,
                        masks,
                    )
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        masks = torch.ones(sequences.shape[:2], device=device)
        outputs, _, _ = _forward_ranking_model(model, sequences, masks)
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

# 主程序
def legacy_train_main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 1. 数据加载
    data_file = os.path.join(data_path, 'stock_data.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    
    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = build_model(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 保存最佳模型（基于final score）
            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - final score: {best_score:.4f}")
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\\nBest final_score: {best_score:.6f}\\n")

        if writer:
            writer.close()

        return best_score

def resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def scale_processed_data(processed, features, fit_end_date):
    scaled = processed.copy()
    scaled[features] = scaled[features].replace([np.inf, -np.inf], np.nan)
    fit_rows = scaled[scaled['日期'] <= pd.Timestamp(fit_end_date)].dropna(subset=features)
    if fit_rows.empty:
        raise ValueError('No valid training rows are available to fit the feature scaler')

    scaler = StandardScaler()
    scaler.fit(fit_rows[features])
    scaled = scaled.dropna(subset=features).copy()
    scaled[features] = scaler.transform(scaled[features])
    return scaled, scaler


def build_ranking_dataset(processed, features, sequence_length, start_date=None, end_date=None):
    if config['model_type'] == 'causal_factor_mixer_v1':
        (
            sequences,
            targets,
            relevance,
            stock_indices,
            auxiliary_returns,
            downside_targets,
        ) = create_multitask_ranking_dataset(
            processed,
            features,
            sequence_length,
            min_window_end_date=start_date,
            max_window_end_date=end_date,
        )
        return RankingDataset(
            sequences,
            targets,
            relevance,
            stock_indices,
            auxiliary_returns=auxiliary_returns,
            downside_targets=downside_targets,
        )
    sequences, targets, relevance, stock_indices = create_labeled_ranking_dataset(
        processed,
        features,
        sequence_length,
        min_window_end_date=start_date,
        max_window_end_date=end_date,
    )
    return RankingDataset(sequences, targets, relevance, stock_indices)


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )


def fit_model(train_dataset, validation_dataset, input_dim, stock_count, device, epochs):
    model = build_model(input_dim=input_dim, config=config, num_stocks=stock_count).to(device)
    ranking_loss = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0),
    )
    criterion = (
        AdaptiveMultiTaskRankingLoss(ranking_loss)
        if config['model_type'] == 'causal_factor_mixer_v1'
        else ranking_loss
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-5,
    )
    train_loader = build_loader(train_dataset, shuffle=True)
    validation_loader = build_loader(validation_dataset, shuffle=False) if validation_dataset is not None else None

    best_score = -float('inf')
    best_epoch = 0
    best_metrics = {}
    best_state = None
    epoch_scores = []
    for epoch in range(int(epochs)):
        train_ranking_model(model, train_loader, criterion, optimizer, device, epoch, writer=None)
        if validation_loader is None:
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            continue

        _, metrics = evaluate_ranking_model(model, validation_loader, criterion, device, writer=None, epoch=epoch)
        score = float(metrics.get('pred_return_sum', 0.0)) / float(config.get('top_k', 5))
        epoch_scores.append(score)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError('Transformer training did not produce a model state')
    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, best_score, epoch_scores


def run_walk_forward_validation(processed, features, stock_count, folds, device):
    records = []
    for fold_index, fold in enumerate(folds):
        set_seed(int(config.get('seed', 42)) + fold_index)
        scaled, _ = scale_processed_data(processed, features, fold['train_sample_end'])
        train_dataset = build_ranking_dataset(
            scaled,
            features,
            config['sequence_length'],
            end_date=fold['train_sample_end'],
        )
        validation_dataset = build_ranking_dataset(
            scaled,
            features,
            config['sequence_length'],
            start_date=fold['validation_start'],
            end_date=fold['validation_end'],
        )
        _, best_epoch, metrics, score, epoch_scores = fit_model(
            train_dataset,
            validation_dataset,
            input_dim=len(features),
            stock_count=stock_count,
            device=device,
            epochs=config.get('cv_num_epochs', config['num_epochs']),
        )
        records.append(
            {
                'fold': fold['name'],
                'train_sample_end': fold['train_sample_end'].date().isoformat(),
                'validation_start': fold['validation_start'].date().isoformat(),
                'validation_end': fold['validation_end'].date().isoformat(),
                'best_epoch': best_epoch,
                'top5_mean_return': score,
                'epoch_scores': epoch_scores,
                'metrics': {name: float(value) for name, value in metrics.items()},
            }
        )
        print(
            f"{fold['name']}: train<= {fold['train_sample_end'].date()}, "
            f"val={fold['validation_start'].date()}~{fold['validation_end'].date()}, "
            f"top5_mean_return={score:.6f}"
        )
    return records


def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    full_df, stock_codes, data_file = load_training_data(
        config['data_path'],
        data_mode=config['data_mode'],
        expected_stock_count=config.get('competition_stock_count', 300),
        as_of_date=config.get('data_as_of_date'),
    )
    print(f"Data mode: {config['data_mode']}")
    print(f"Training input: {data_file}")
    print(f"Training data range: {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    stock_ids = sorted(stock_codes)
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}
    processed, features = preprocess_data(full_df, is_train=True, stockid2idx=stockid2idx)
    processed['日期'] = pd.to_datetime(processed['日期'])

    folds = build_walk_forward_folds(
        trading_dates_from_frame(full_df),
        label_horizon=int(config['label_horizon_days']),
        embargo_days=int(config['cv_embargo_days']),
        validation_days=int(config['cv_validation_days']),
        min_train_days=int(config['cv_min_train_days']),
        cv_folds=int(config['cv_folds']),
        warmup_days=int(config['feature_warmup_days']),
    )
    device = resolve_device()
    oof_records = run_walk_forward_validation(processed, features, len(stock_ids), folds, device)
    selected_epochs, epoch_selection = select_robust_epoch(
        [record['epoch_scores'] for record in oof_records],
        risk_penalty=float(config['cv_epoch_risk_penalty']),
    )

    final_fit_end = processed['日期'].max()
    final_scaled, final_scaler = scale_processed_data(processed, features, final_fit_end)
    final_dataset = build_ranking_dataset(final_scaled, features, config['sequence_length'])
    set_seed(int(config.get('seed', 42)) + 10_000)
    final_model, _, _, _, _ = fit_model(
        final_dataset,
        validation_dataset=None,
        input_dim=len(features),
        stock_count=len(stock_ids),
        device=device,
        epochs=selected_epochs,
    )

    torch.save(final_model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    joblib.dump(final_scaler, os.path.join(output_dir, 'scaler.pkl'))
    selected_oof_scores = np.asarray(
        [record['epoch_scores'][selected_epochs - 1] for record in oof_records],
        dtype=float,
    )
    oof_mean = float(selected_oof_scores.mean())
    metadata = {
        'input_file': str(data_file),
        'data_mode': config['data_mode'],
        'model_type': config['model_type'],
        'feature_schema_version': config['feature_schema_version'],
        'data_as_of_date': config.get('data_as_of_date'),
        'stock_ids': stock_ids,
        'features': features,
        'label_horizon_days': int(config['label_horizon_days']),
        'cv_embargo_days': int(config['cv_embargo_days']),
        'selected_epochs': selected_epochs,
        'oof_top5_mean_return': oof_mean,
        'oof_top5_std': float(selected_oof_scores.std()),
        'epoch_selection': epoch_selection,
        'oof_folds': oof_records,
    }
    with open(os.path.join(output_dir, 'model_meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'final_score.txt'), 'w', encoding='utf-8') as handle:
        handle.write(f"OOF top-5 mean return: {oof_mean:.6f}\n")
        handle.write(f"Selected epochs: {selected_epochs}\n")

    print(f"Walk-forward OOF top-5 mean return: {oof_mean:.6f}")
    print(f"Final epochs: {selected_epochs}")
    return oof_mean


if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
