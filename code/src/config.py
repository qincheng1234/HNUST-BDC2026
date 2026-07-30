# 配置参数
import os


sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'model_type': 'causal_factor_mixer_v3',
    'feature_schema_version': 'risk_calibrated_residual_v4',
    'd_model': 128,
    'mixer_layers': 2,
    'time_mixer_hidden': 32,
    'mixer_expansion': 2,
    'factor_count': 8,
    'market_mixer_layers': 1,
    'cross_sectional_epsilon': 1e-6,
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}_risk_calibrated_factor_mixer',
    'data_path': './data',
    'data_mode': os.environ.get('DATA_MODE', 'local_split').strip().lower(),
    'data_as_of_date': os.environ.get('AS_OF_DATE') or None,
    'competition_stock_count': 300,
    'label_horizon_days': 5,
    'cv_embargo_days': 5,
    'cv_validation_days': int(os.environ.get('CV_VALIDATION_DAYS', '10')),
    'cv_min_train_days': 180,
    'cv_folds': int(os.environ.get('CV_FOLDS', '6')),
    'cv_num_epochs': int(os.environ.get('CV_NUM_EPOCHS', '40')),
    'cv_epoch_risk_penalty': float(os.environ.get('CV_EPOCH_RISK_PENALTY', '0.25')),
    'feature_warmup_days': 120,
}
