# 项目交接文档（2026-07-31）

> 项目：`HNUST-BDC2026` / 2026 中国高校计算机大赛大数据挑战赛。  
> 本文记录当前可用基线、已证伪实验、数据与运行约束，以及本地和 AutoDL 的协作流程。  
> **不要把本文件中的本地 `score_self` 当成调参目标。模型选择一律以训练数据内的 walk-forward OOF 为准。**

---

## 1. 当前状态与结论

### 1.1 当前应使用的模型

当前代码已回退至唯一推荐基线：

| 项目 | 当前值 |
| --- | --- |
| 模型 | `causal_factor_mixer_v2` / `CrossSectionalResidualFactorMixer` |
| 特征 | `158+39` 原始工程特征 + 15 个横截面/市场特征，共 **211** 个模型特征 |
| 序列长度 | 60 个交易日 |
| 标签 | 决策日后的第 1 个交易日开盘到第 5 个交易日开盘收益率 |
| 验证 | 6 折 walk-forward；每折 10 个交易日验证；标签期/embargo 均为 5 个交易日 |
| epoch 选择 | 所有折同一 epoch 的均值 `- 0.25 × 标准差` 最大者 |
| 推荐产物目录 | `model/60_158+39_causal_residual_factor_mixer/` |

该基线的已同步 OOF 结果：

| OOF Top-5 均值 | OOF 标准差 | 稳健分 | 选择 epoch |
| ---: | ---: | ---: | ---: |
| `0.011834` | `0.036122` | `0.002803` | `4` |

它是目前所有已完成候选中唯一稳健分为正、且 OOF 均值最高的模型。当前配置位于 `code/src/config.py`，模型实现在 `code/src/model.py`。

### 1.2 当前工作区提醒

截至本文写入时，以下 v2 恢复变更**尚未提交**：

```text
M code/src/config.py
M code/src/model.py
M code/src/train.py
M test/test_causal_factor_mixer.py
```

这些变更做了两件事：恢复 v2 为默认模型、删除失败的多尺度市场 Mixer。同步到 AutoDL 前必须先提交并推送这四个文件，见第 8 节。

Git 仓库不承载实际行情数据；当前或新的克隆目录在运行训练前都需要重新下载 `data/stock_data.csv`，或由赛事/容器挂载提供该文件。

### 1.3 合规边界

- 训练和预测都真实依赖输入行情及模型参数；`result.csv` 由预测分数排序后产生。
- 代码中没有预置股票代码、预置最终持仓名单或预计算结果。
- 当前输出为预测分数前 5 名股票的通用等权 `0.2`；股票代码随输入数据和模型预测变化。
- 本地 `data/test.csv` 只允许给 `test/score_self.py` 做留出观察；不得被 `train.py` 或 `predict.py` 读取，更不能用其结果选择参数。
- 未加载、微调或复用任何公开预训练模型权重；所有候选均从随机初始化开始训练。

---

## 2. 当前模型与训练流程

### 2.1 特征与样本

1. `engineer_features_158plus39()` 生成 Alpha158 与 39 个技术指标；基础特征列表包含 `instrument`，但模型输入会移除它。
2. `add_cross_sectional_market_features()` 追加 15 个只使用同日截面数据计算的特征：收益/波动/换手率/成交额/振幅的截面排名、市场收益/广度/离散度/波动/换手率，以及相对市场收益。
3. 每个样本为一个交易日的全部可用股票：输入形状为 `[batch, stocks, 60, 211]`，训练时通过 `collate_fn` 补齐并使用 mask。
4. 每个折的 `StandardScaler` 只在该折训练截止日前拟合；验证和最终预测只调用 `transform`。

### 2.2 `causal_factor_mixer_v2`

模型的主要路径如下：

1. 对每个日期、每个特征在股票截面内去均值并按截面标准差标准化，形成个股相对市场的残差路径。
2. 经过两个轻量 `TemporalMixBlock`，以趋势/残差时间混合与特征 MLP 混合建模 60 日路径。
3. `MultiScalePool` 聚合最近、5 日、20 日和全窗口状态。
4. 市场分支使用全截面均值路径编码市场状态；`DynamicFactorMixer` 用 8 个可学习因子 token 做 O(NK) 的个股交互，不使用 300×300 全注意力或静态行业图。
5. 学习三路融合门控（个股、动态因子、市场状态），`score_head` 输出每只股票的排序分数。

训练目标为 `WeightedRankingLoss`：listwise 交叉熵与 pairwise 损失之和，真实 Top-5 的权重为 2，其余样本为 1。当前 v2 仅以排序损失训练；模型中保留的辅助输出不参与 v2 损失。

### 2.3 训练、验证、重训

`code/src/train.py` 的实际流程：

1. 依据 `DATA_MODE` 从指定数据文件加载 300 只股票；代码会验证股票数、代码格式、日期和必要列。
2. 生成特征、标签与按日期组织的排名数据集。
3. 依据 `code/src/splits.py` 构建 6 个滚动折，训练每折并记录每个 epoch 的 Top-5 收益。
4. 在所有折上选择同一个稳健 epoch，而不是选择单折最佳 epoch。
5. 使用全部可标注历史重新拟合 scaler 和模型，写入：
   - `best_model.pth`
   - `scaler.pkl`
   - `model_meta.json`（数据模式、股票池、特征、OOF 折和选中 epoch）
   - `final_score.txt`

`code/src/predict.py` 会验证模型元数据中的数据模式、模型类型、特征版本和股票池，随后仅使用历史末日生成每只股票的模型分数，输出 `output/result.csv`。

---

## 3. 数据模式与赛事挂载规则

### 3.1 本地留出模式：`DATA_MODE=local_split`

| 程序 | 读取文件 | 用途 |
| --- | --- | --- |
| `train.py` | `data/train.csv` | 特征、标签、walk-forward OOF、最终拟合 |
| `predict.py` | `data/train.csv` | 用训练集最后一个交易日生成预测 |
| `test/score_self.py` | `data/test.csv` + `output/result.csv` | 仅做本地留出周参考评分 |

本地划分命令（最后 5 个交易日为留出集）：

```bash
python scripts/split_train_test.py \
  --input data/stock_data.csv \
  --output-dir data \
  --auto-last-days 5
```

划分完成后，先训练、预测，最后才运行：

```bash
python test/score_self.py
```

**注意：** 下载更新后，留出周日期会移动；不同下载批次的 `score_self` 不能横向比较，也不能用来回选模型。

### 3.2 最终提交模式：`DATA_MODE=stock_data`

赛事最终会将 `data/` 覆盖挂载，仅提供截至提交日前约三年的 `stock_data`，不提供 `train.csv`、`test.csv`。最终复现必须由代码自行训练、OOF 和预测：

```bash
export DATA_MODE=stock_data
python code/src/train.py
python code/src/predict.py
```

Windows PowerShell：

```powershell
$env:DATA_MODE = 'stock_data'
.\.venv\Scripts\python.exe code/src/train.py
.\.venv\Scripts\python.exe code/src/predict.py
```

禁止把依赖数据放进 `data/`、`output/` 或 `temp/` 并期待镜像保留；这些目录会被赛事挂载覆盖。当前项目的 `.gitignore` 也刻意忽略这些本地数据和产物。

### 3.3 历史模拟截断

可以用 `AS_OF_DATE` 截断训练和预测可见的原始历史，用于不读取后续行情的历史模拟：

```bash
export DATA_MODE=stock_data
export AS_OF_DATE=2026-07-21
python code/src/train.py
python code/src/predict.py
unset AS_OF_DATE
```

随后若有单独保存的未来五日行情，才能离线计算对应持有期表现。不要把后续数据混入该次训练文件。

---

## 4. Tushare 数据下载与增量更新

下载脚本是 `get_stock_data.py`，已改为 **Tushare-only**，不依赖 AkShare 或 Baostock。

### 4.1 Token

推荐将 token 放在未跟踪文件：

```text
temp/tushare_token.txt
```

或设置环境变量 `TUSHARE_TOKEN`。不要将 token 写入 Git、文档、日志或命令历史。

### 4.2 固定股票池

项目曾确认应以 **2026-02-20** 的沪深 300 成分股为基准，而非把最新一期成分股回填所有历史。脚本会获取不晚于 `STOCK_UNIVERSE_DATE` 的可用 Tushare 成分记录；此前一次有效记录日期为 2026-01-30。

全量/增量下载示例（日期按实际需要修改）：

```bash
export TUSHARE_TOKEN_FILE="$PWD/temp/tushare_token.txt"
export STOCK_UNIVERSE_DATE=2026-02-20
export STOCK_DATA_START_DATE=2024-01-01
export STOCK_DATA_END_DATE=2026-07-31
export TUSHARE_RESUME=1
python get_stock_data.py
```

可选控制项：

| 环境变量 | 含义 |
| --- | --- |
| `TUSHARE_RESUME=1` | 读取已有 `data/stock_data.csv`，仅补下载缺失尾部区间 |
| `TUSHARE_BACKFILL=1` | 同时补齐已有数据前的缺口 |
| `TUSHARE_RETRIES` | 单请求重试次数，默认 4 |
| `TUSHARE_RETRY_SECONDS` | 重试等待秒数，默认 3 |
| `TUSHARE_SLEEP_SECONDS` | 股票请求间隔，默认 0.25 秒 |
| `TUSHARE_SAVE_EVERY` | 每下载多少只股票落盘一次，默认 10 |

输出包括 `data/stock_data.csv`、`data/hs300_stock_list.csv`，失败时还有 `data/failed_stocks.csv`。下载后应检查：恰有 300 个唯一股票、日期连续性合理、无 `股票代码+日期` 重复，再进行本地划分。

---

## 5. 实验记录与已证伪方向

所有下表 OOF 都来自相同的本地 `local_split` 训练数据、6 折验证和 211 特征口径（A1/A2/A3 除外）。稳健分为 `mean - 0.25 × std`；数值越高越好。

| 候选 | 特征数 | epoch | OOF 均值 | OOF 标准差 | 稳健分 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `market_guided_mixer_v1` | 211 | 8 | 0.006301 | 0.030312 | -0.001277 | 淘汰 |
| `causal_factor_mixer_v1` | 211 | 1 | -0.003526 | 0.009905 | -0.006002 | 淘汰 |
| **`causal_factor_mixer_v2`** | **211** | **4** | **0.011834** | **0.036122** | **0.002803** | **当前基线** |
| `causal_factor_mixer_v3` 风险校准头 | 211 | 14 | 0.006938 | 0.029226 | -0.000369 | 淘汰，辅助收益 RankIC 为负 |
| 因果多尺度市场 token Mixer | 211 | 1 | 0.009963 | 0.055648 | -0.003949 | 淘汰，波动和尾部风险明显变差 |

### 5.1 特征消融 A1/A2/A3

为检验高相关技术指标是否是噪声，曾完成 A 阶段消融。脚本与开关已按结论删除，历史元数据仅保留作记录。

| 实验 | 输入 | 特征数 | epoch | OOF 均值 | OOF 标准差 | 稳健分 | 最差折 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | 39 个技术指标 + 截面/市场特征 | 53 | 1 | 0.011149 | 0.036973 | 0.001906 | -0.018788 | 接近 v2，但均值和稳健分不足；不替换 |
| A2 | Alpha158 主体（无 39 扩展） | 183 | 14 | 0.003884 | 0.038907 | -0.005843 | -0.063394 | 39 指标具有补充信号，不能删 |
| A3 | 移除绝对价格/成交量/均线层级 | 175 | 4 | 0.008401 | 0.048766 | -0.003790 | -0.073079 | 删除层级特征反而更差 |

结论：**完整 158+39 特征是当前 v2 最佳输入；瓶颈不是简单删特征。** 后续如探索新模型，必须以 v2 的完整特征和同一 OOF 协议作单一候选比较，禁止一次并行叠加多种策略、模型和目标。

### 5.2 多尺度市场 token 失败复盘

失败模型把个股序列拆成短期/中期/长期路径，并引入 1/5/20 日市场 token 门控。它在 6 折上的收益为：

```text
0.027338, 0.052471, 0.000571, 0.090272, -0.029176, -0.081698
```

模型只选中第 1 个 epoch，说明额外容量很快过拟合；最后一折 `-8.17%` 使标准差上升至 `5.56%`。本地一次留出周 `score_self=-0.017774` 与 OOF 的风险结论一致，但该分数**没有**用于回调参数。

其源码、配置、工厂分支和专项测试已删除；当前代码不会再训练该模型。

### 5.3 更早阶段（已重置的策略分支）

项目早期使用过 LightGBM/LambdaRank、压力因子覆盖和策略融合。诊断发现压力覆盖在触发时会绕过模型分数，仅按风险因子挑选股票，存在“主要贡献不是模型预测”的合规风险，且曾得到 `score_self=-0.000868`。同一已知周的模型常规选股反事实为 `+0.027600`，但该信息**未**被用于调参。该整套分支已经从当前源码重置移除，不应恢复。

### 5.4 已观察到的本地评分（仅留档）

不同评分运行可能对应不同下载截止日和不同留出周，因此不能比较或用于选模型：

| 阶段 | 已报告 `score_self` | 说明 |
| --- | ---: | --- |
| 早期市场引导模型运行 | 0.002786、另一次 0.026023 | 历史日志，数据批次不同 |
| v2 基线一次运行 | 0.007374 | 仅留出周参考 |
| 风险校准 v3 | 0.006802 | OOF 已淘汰 |
| 多尺度市场 token | -0.017774 | OOF 已淘汰 |

---

## 6. 下一步研究原则

1. 先确保当前 v2 在同一 AutoDL 环境、同一数据快照可复现，再评估任何新候选。
2. 不要再实施“大分支叠加”的多尺度/多头/策略覆盖；样本规模下容易放大折间方差。
3. 若继续模型创新，只做**一个**预先定义的轻量候选，例如在 v2 现有时序编码器上替换固定多尺度池化为小型可学习时间权重池化；不要同时更改特征、损失、选股策略和权重规则。
4. 候选至少同时满足以下条件才可替换 v2：
   - OOF 均值不低于 `0.011834`；
   - 稳健分高于 `0.002803`；
   - 最差折不差于 v2 的 `-0.037439`。
5. 仅在上述 OOF 条件达标后，查看一次固定留出周的 `score_self`；它只用于发现明显流程问题，而非继续调参。

---

## 7. 环境、依赖与常用命令

### 7.1 依赖

项目使用 Python `>=3.10,<3.13`，核心依赖定义在 `pyproject.toml`：PyTorch、pandas、scikit-learn、TA-Lib、tensorboardX、Tushare、joblib、tqdm。

本地 Windows：

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
```

AutoDL 推荐使用已激活的 `(.venv-autodl)` 环境；容器系统 Python 曾确认可用 CUDA，但不要重新安装/覆盖系统已可用的 PyTorch，除非环境确实损坏。

历史上 AutoDL 的镜像源下载 `scipy`、`triton` 时出现过超时、hash mismatch，以及因索引策略导致依赖重下。若当前 `python -c "import torch"` 已成功且 CUDA 可用，优先直接复用现有环境，不要为了同步源码重复执行 `uv sync` 或重装 Torch。

基础检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s test -p "test_*.py"
```

### 7.2 本地/AutoDL 训练命令

本地 Windows（仅在需要本地运行时）：

```powershell
$env:DATA_MODE = 'local_split'
.\.venv\Scripts\python.exe code/src/train.py
.\.venv\Scripts\python.exe code/src/predict.py
.\.venv\Scripts\python.exe test/score_self.py
```

AutoDL（已激活虚拟环境）：

```bash
export DATA_MODE=local_split
python code/src/train.py
python code/src/predict.py
python test/score_self.py
```

快速排障可临时缩短 OOF epoch，例如 `export CV_NUM_EPOCHS=5`；正式结果必须取消该临时设置或显式使用完整设置。可用 `CV_FOLDS`、`CV_VALIDATION_DAYS`、`CV_NUM_EPOCHS` 调整验证流程，但这属于模型选择参数，必须在训练 OOF 内固定和记录。

---

## 8. Git、GitHub 与 AutoDL 协作

远程仓库：`https://github.com/qincheng1234/HNUST-BDC2026.git`。默认分支：`main`。

### 8.1 本地推送源码

始终先查看变化，明确添加源码；不要用 `git add .` 把数据、token 或产物意外加入暂存区。

```powershell
git status --short
git add code/src/config.py code/src/model.py code/src/train.py test/test_causal_factor_mixer.py HANDOVER.md
git commit -m "revert: restore validated v2 factor mixer"
git pull --rebase origin main
git push origin main
```

如果 `git pull --rebase` 有冲突：先解决冲突并运行测试；不要未经检查使用 `git push --force`。历史上曾因远程初始提交不同使用过 `--force-with-lease`，那只适用于明确确认远程分支应被本地历史替换的情况。

### 8.2 AutoDL 拉取源码

AutoDL 偶发以下网络错误：

```text
GnuTLS recv error (-110): The TLS connection was non-properly terminated
```

优先使用平台网络加速，并强制 HTTP/1.1：

```bash
source /etc/network_turbo
git config --local http.version HTTP/1.1
git fetch --prune origin
git pull --ff-only origin main
```

若仍失败，等待后重试；该错误通常是 GitHub 网络/TLS 路由中断，不是本地代码冲突。避免在网络不稳时混用强制推送。

### 8.3 从 AutoDL 同步训练结果

默认 `.gitignore` 会忽略 `model/` 和 `output/`。只为分析同步小型元数据与结果 CSV，不同步数据、token、`best_model.pth` 或 `scaler.pkl`：

```bash
source /etc/network_turbo

git add -f \
  model/60_158+39_causal_residual_factor_mixer/model_meta.json \
  model/60_158+39_causal_residual_factor_mixer/final_score.txt \
  output/result.csv

git commit -m "artifacts: sync v2 training results"
git push origin main
```

本地拉取：

```powershell
git pull --ff-only origin main
```

如果 AutoDL 提示 `Author identity unknown`，仅在该仓库设置身份后重试：

```bash
git config user.name "<你的 GitHub 用户名>"
git config user.email "<你的 GitHub 邮箱>"
```

### 8.4 已跟踪历史产物

虽然 `.gitignore` 忽略产物目录，但此前为分析曾以 `git add -f` 提交过部分 `model_meta.json`、`final_score.txt` 和 `output/result.csv`。它们是历史记录，不是训练或最终 Docker 复现的依赖。提交前可评估是否从版本控制中移除这些历史小文件；**不要删除本地唯一的数据或模型备份**。

---

## 9. Docker 与最终提交检查

### 9.1 当前 Docker 结构

- `Dockerfile` 以 Python 3.12 slim 为基底，编译 TA-Lib C 库，使用 `uv sync --frozen` 安装依赖。
- `docker-compose.yml` 将本地 `data/`、`test/output/`、`temp/` 挂载到容器。
- 赛事最终挂载会覆盖 `data/`，因此最终流程必须依赖 `data/stock_data`，且以 `DATA_MODE=stock_data` 运行。

### 9.2 提交前阻塞项

当前 `docker-compose.yml` 的命令是：

```text
/bin/bash /app/data/run.sh
```

但赛事说明表示最终挂载的 `data/` 只保证提供 `stock_data`，未保证存在 `run.sh`。因此在最终打包前必须把 compose/入口改为镜像内的脚本（例如根目录 `init.sh` 或直接执行训练、预测），且该脚本不得依赖 `data/run.sh`、`train.csv` 或 `test.csv`。

最终提交前最低检查清单：

1. 设置 `DATA_MODE=stock_data`，删除或忽略本地 `train.csv`、`test.csv` 后可完整训练和预测。
2. 新容器仅挂载赛事 `data/stock_data` 时可从零运行并在 `output/result.csv` 生成 5 行、代码唯一、权重和为 1 的结果。
3. 运行 `python -m unittest discover -s test -p "test_*.py"`。
4. 不将 Tushare token、真实数据、测试数据或临时日志打包进镜像。
5. 再构建并检查镜像：

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
docker compose up
docker save -o team_name.tar bdc2026:latest
```

---

## 10. 关键文件索引

| 路径 | 作用 |
| --- | --- |
| `get_stock_data.py` | Tushare-only 下载、固定沪深 300 股票池、增量续传 |
| `scripts/split_train_test.py` | 从 `stock_data.csv` 生成本地 `train.csv` / `test.csv` |
| `code/src/config.py` | 当前模型、特征、OOF、数据模式和输出目录配置 |
| `code/src/data_io.py` | 严格区分 `local_split` 与 `stock_data` 的读数契约 |
| `code/src/utils.py` | 特征工程、截面/市场特征和数据集构建工具 |
| `code/src/splits.py` | walk-forward 折与稳健 epoch 选择 |
| `code/src/model.py` | v2 残差动态因子 Mixer |
| `code/src/train.py` | 训练、OOF、最终重新拟合、模型元数据写入 |
| `code/src/predict.py` | 加载模型并生成 `output/result.csv` |
| `test/score_self.py` | 本地留出周参考评分；不参与训练 |
| `DATA_CONTRACT.md` | 数据使用和复现约束摘要 |
| `MODEL_OPTIMIZATION_PLAN.md` | 历史研究动机和实验计划；其中已完成/淘汰部分以本文为准 |

---

## 11. 交接时第一步

1. 阅读本文件第 1、3、5 节，确认当前可用模型是 v2，失败候选不应恢复。
2. 执行 `git status --short`，先提交第 1.2 节列出的 v2 恢复变更及本文件。
3. AutoDL 拉取后，执行 `python -c "import sys; sys.path.insert(0, 'code/src'); from config import config; print(config['model_type'])"`，确认输出为 `causal_factor_mixer_v2`。
4. 下载/更新数据、执行本地划分、训练和预测；只将 `model_meta.json`、`final_score.txt`、`result.csv` 同步回来做 OOF 审查。
5. 在最终提交前单独完成第 9 节的 Docker 入口修复与挂载复现，不要把本地评分脚本流程当成赛事最终流程。
