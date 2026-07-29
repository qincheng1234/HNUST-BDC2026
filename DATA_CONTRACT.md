# 数据与验证约定

## 数据来源

- 本地默认 `DATA_MODE=local_split`：训练和预测都读取 `data/train.csv`；`data/test.csv` 只由 `test/score_self.py` 读取，用于留出周评分，绝不进入训练或预测。
- 最终提交使用 `DATA_MODE=stock_data`：训练和预测只读取 `data/stock_data.csv`（也兼容无扩展名的 `data/stock_data`），可直接使用赛事最终挂载的 `data` 目录。
- 输入必须包含 300 只股票，以及股票代码、日期、开高低收、成交量、成交额和换手率列。
- `AS_OF_DATE=YYYY-MM-DD` 可用于历史模拟：训练和预测都会先截断至该日期，不会读取其后的本地数据。

## 标签与验证

- 标签为决策日后第 1 个交易日开盘至第 5 个交易日开盘的收益率。
- 使用最近三个、每个五个交易日的扩展窗口验证折叠。
- 每个折叠在验证期前保留五个交易日的 embargo；训练样本的标签结束日严格早于验证决策日。
- 每个折叠的标准化器只在该折叠训练期拟合，验证期和最终预测仅调用 `transform`。
- 最终模型在完整可标注历史上重新训练；验证结果和训练元数据写入 `model/.../model_meta.json`。

## 运行

```powershell
python code/src/train.py
python code/src/predict.py
```

最终提交前，在同一终端执行：

```powershell
$env:DATA_MODE = 'stock_data'
python code/src/train.py
python code/src/predict.py
```

本地快速检查可临时设置 `CV_NUM_EPOCHS`；正式训练应使用完整轮数，且不得根据 `score_self` 或未来测试周调参。
