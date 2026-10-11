# GoQueen：19路围棋桥接实验

## 总览

目标是把冻结的KataGo棋盘特征接入冻结的Qwen3-0.6B，训练桥接参数回答棋盘问题。当前发布子目录提供M-APS辅助点位选择模块、实际批次路由、变体权重加载及必要的第一方源码。

| 项目 | 当前状态 |
|---|---|
| M-APS模块 | 根据题目已有坐标限定一个棋盘token；不使用答案标签选点 |
| 已有工程结果 | 64项小型检查、130项真实模型工程检查通过；2次反传、0次优化器更新 |
| 已有实测范围 | 21题初始logP与原M差异0；231次指定点及所选value核对通过；外部15.550秒、退出码0 |
| 完整拟合 | 单次拟合入口正在实现，尚未执行APS拟合；下一阶段最多180更新、720采样槽、300秒软总预算 |
| 研究结论 | 工程可用不表示已经拟合、学会自然寻址、具备可靠棋盘理解或达到正式M1标准 |

本目录是源码发布子集。模型权重、专家特征、原始实验输出和运行记录均另行保存，未包含在仓库中。文档引用的已有结果是项目记录摘要，源码副本本身不构成独立运行复现。

## 实现

- `src/goqueen/assisted_point.py`：保持原M的Q/K/V、共享行列编码、归一化及门控公式；每个样本只允许题目所问的一个棋盘token参与注意力。
- `src/goqueen/assisted_point_eval.py`：在实际前向批次内核对坐标和规范提示、按对应顺序绑定点位；路由不读取占用标签。
- `src/goqueen/checkpoints.py`：先核对变体、元数据、完整参数名称与形状，再复制权重；拒绝M、M-Key和M-APS之间误加载。
- `scripts/check_assisted_point_engineering.py`：已有零更新工程入口，覆盖初始评分、实际mask/value、反传、恢复及清理；拒绝覆盖既有输出。

有限分数、math SDPA及dropout=0条件下，单点注意力直接返回所选value。同一层给该样本全部文本位置加入同一个桥接向量，整个冻结语言模型仍处理文本。44个Q/K及query归一化张量在这一路径中不起输出作用，保留它们以匹配原101个参数张量、40,438,795个参数值。

## 源码检查

在本目录运行，使用Python 3.12环境安装固定依赖后，可执行不需要预训练权重的小型检查：

```sh
python -m pip install -r requirements.txt -r requirements-lock.txt
python -m unittest discover -s tests -p 'test_assisted_point*.py' -v
python -m unittest discover -s tests -p 'test_adapter_checkpoints.py' -v
python -m unittest discover -s tests -p 'test_m_key.py' -v
python -m unittest discover -s tests -p 'test_grounding_interaction.py' -v
```

上述检查使用小型张量或假模型，不等价于真实模型工程、完整拟合或一般棋盘能力验证。必要的M-Key模块是当前加载器的依赖；历史自然路由M脚本提供固定采样、参数摘要和来源复现辅助函数，不表示应再次运行旧实验。

## 真实模型运行前提

工程和后续拟合需要另行准备原始且摘要匹配的固定资料：

- `models/Qwen3-0.6B/`：固定版本的官方Qwen模型和tokenizer。
- `outputs/grounding-data/manifest.json`与对应5份专家特征缓存。
- `outputs/grounding-conflict-audit/report.json`：固定21题来源选择资料。
- `outputs/grounding-microfit/report.json`及其来源记录：原M初始化、采样和评分复现参考。

入口会校验真实文件及摘要；缺少资料时不能把这份源码当作独立可执行的真实模型实验。当前APS真实工程已完成，不需要为发布重复执行。

后续单次拟合保持真实21题、fresh seed42、Random(42)固定720槽、batch4、AdamW学习率0.001及weight_decay=0、CPU4线程、FP32与math SDPA。训练采用全词表值token CE，三类CE只是报告指标；固定180终点与中途评分分别记录。300秒是含加载、核对、训练、评分、保存和报告的软总预算，不能保证实际外部时长不超出。不会因为未拟合或超时自动重跑、延长预算或追加调参分支。

详细协议见[APS设计](docs/grounding-assisted-point-design.md)，已有结果及覆盖限制见[APS工程核验](docs/assisted-point-engineering.md)。辅助选点条件下成功只说明下游整体路径能有限拟合这21题；失败不能据此确定自然寻址失败原因或排除其他架构。
