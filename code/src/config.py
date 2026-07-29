# 配置参数
import os


sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
    'data_mode': os.environ.get('DATA_MODE', 'local_split').strip().lower(),
    'data_as_of_date': os.environ.get('AS_OF_DATE') or None,
    'competition_stock_count': 300,
    'label_horizon_days': 5,
    'cv_embargo_days': 5,
    'cv_validation_days': 5,
    'cv_min_train_days': 180,
    'cv_folds': 3,
    'cv_num_epochs': int(os.environ.get('CV_NUM_EPOCHS', '50')),
    'feature_warmup_days': 120,
}
