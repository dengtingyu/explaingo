# GoQueen：19 路围棋研究源码

当前代码入口为 **[Qwen3.8-27B / RTX 5090 训练代码与复现配置](reproduction/qwen38-27b-20261014/README.md)**。冻结官方 Qwen 底座和 KataGo，使用单卡 NF4 与 FP32、宽度 1024 的棋盘桥接模块；配方为固定 126,561 训练题、14,499 验证题、5 个完整 epoch、batch16、peak lr `1e-4`。这些数目是配置说明，实际训练集、棋谱、缓存、权重和运行结果均不随仓库发布，须从维护者另行取得。

- 主入口：`scripts/run_qwen38_expanded.py` → `scripts/train_qwen38_expanded.py`。
- 当前环境、固定配方和实际训练源码摘要：[代码交接说明](reproduction/qwen38-27b-20261014/README.md)、[配置](reproduction/qwen38-27b-20261014/configuration.json)、[源码摘要](reproduction/qwen38-27b-20261014/source-pins.json)。
- `sources.json` 仅含官方 KataGo 模型来源、版本与文件摘要，供当前特征生成入口核验；不含数据集、棋谱或标签。

下文保留历史小模型的辅助选点（APS）、从 APS 起步的自然 Q/K 训练及固定权重训练题评估。目标仍是让语言模型按文字坐标读取棋盘；可靠自然选点、通用围棋理解与棋力尚未建立。

## 历史 APS / QK 结果

| 路径 | 已保存的观察结果 | 边界 |
|---|---|---|
| APS 外部指定点位 | 21 道训练题全部正确；固定 54 题辅助评估全部正确 | 不证明模型自行找到所问点 |
| APS 起步自然 Q/K，1,000 次更新 | 同一 54 题从 18/54 到 18/54；三类 CE 从 7.9614 降至 1.1670 | 分数变化，正确率未提高 |
| 固定最终权重评估实际训练题 | 1,256/2,702（46.4841%）；宏平均召回率 43.2578%；按 4,000 次曝光加权为 43.4% | 训练题仍未充分拟合，不能唯一定位训练预算或接入结构 |

这些是项目保存记录的摘要。本目录提供源码、测试源码、配置和技术说明，**不包含模型权重、专家特征缓存、数据清单或原始实验输出**，不能仅凭本目录独立复现上述历史结果；本次源码整理没有重新执行测试或模型。此处为历史 APS / QK 阶段摘要，不能用来代替上方 Qwen3.8-27B 配方的状态或效果。

## 历史 APS / QK 代码入口

| 内容 | 入口 |
|---|---|
| 基础融合与权重加载 | `src/goqueen/fusion.py`、`checkpoints.py` |
| APS 单点选择与评分 | `src/goqueen/assisted_point.py`、`assisted_point_eval.py` |
| APS 拟合入口 | `scripts/train_grounding_assisted_point.py`、`run_grounding_assisted_point.py` |
| 固定 APS 权重的 54 题评估 | `scripts/eval_assisted_point_transfer.py`、`run_assisted_point_transfer.py` |
| 自然 Q/K 状态转换及完整保存 | `src/goqueen/warm_qk.py` |
| 自然 Q/K 训练入口 | `scripts/train_aps_warm_qk.py`、`run_aps_warm_qk.py` |
| 实际训练题选择与曝光加权 | `src/goqueen/train_eval_selection.py`、`transfer_metrics.py` |
| 固定训练题评估入口 | `scripts/eval_aps_warm_qk_train.py`、`run_aps_warm_qk_train.py` |
| 保存记录核对与分层描述 | `scripts/inspect_aps_warm_qk_train.py`、`inspect_warm_qk_training_strata.py` |
| 相关测试 | `tests/`；包括纯算术、模拟进程和合成小模型检查，性质各不相同 |

APS 实现复用旧 M/M-Key 脚本中的数据、摘要和报告函数，因此本目录保留这些导入依赖；它们不是建议重开旧实验。配置 `configs/mixed-train.json` 是既有数据／接口背景，当前 APS/QK 的固定运行合同见各入口及设计文件。

## 技术说明

- [APS 设计](docs/grounding-assisted-point-design.md)、[工程](docs/assisted-point-engineering.md)、[拟合](docs/assisted-point-fit.md)。
- [固定 APS 评估设计](docs/assisted-point-transfer-design.md)与[结果](docs/assisted-point-transfer-results.md)。
- [Q/K 设计](docs/aps-warm-qk-design.md)、[工程](docs/aps-warm-qk-engineering.md)、[训练结果](docs/aps-warm-qk-fit.md)。
- [固定最终权重的训练题评估与分层解释](docs/aps-warm-qk-train-eval.md)。
- [已停止的注意力观察记录](docs/aps-warm-qk-attention-results.md)仅为历史说明；该观察器代码不属于本次最小发布闭包，没有配对结论或自动重跑计划。

文档中的 `outputs/`、缓存和权重路径指原始项目的依赖或记录，不代表这些文件随本源码发布。历史工程阶段中的未执行说明保留其当时语境，后续实测以对应结果页为准。

## 历史 APS / QK 环境与运行边界

以下依赖属于历史小模型；Qwen3.8-27B 使用上方交接目录的云端依赖。历史项目使用 Python 3.12、PyTorch、Transformers、NumPy、safetensors 和 sgfmill；直接依赖及原环境锁定版本分别见 `requirements.txt`、`requirements-lock.txt`。第一方模块在 `src/goqueen/`。本目录没有打包第三方实现。

模型实验还需另行准备 Qwen3-0.6B 基础权重／tokenizer、相应适配器权重、19×19 专家特征及带固定摘要的数据／历史记录。入口会检查精确来源、恢复容差和排他输出；这些前提缺失时不能把运行失败解释为模型学习失败，也不应为了直接启动而删除检查。说明中的历史调用顺序不是批量重跑指令。
