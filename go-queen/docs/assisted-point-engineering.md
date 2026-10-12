# M-APS：辅助点位选择工程核验

## 总览

| 项目 | 实际结果 |
|---|---|
| 当前状态 | 本文保留一次已完成的零更新工程核验；拟合runner及后续唯一一次180更新拟合已完成，固定终点21/21，另见[单次拟合结果](assisted-point-fit.md) |
| 小型检查 | 64项通过，0失败、0错误；其中35项APS检查、29项既有M-Key及加载器回归检查，覆盖小型torch／假Qwen、合成权重文件和实际批次接线，不加载预训练模型 |
| 真实工程 | Qwen3-0.6B，CPU4线程、FP32、math SDPA；130项检查通过，两次反传，未构造优化器，0次更新 |
| 实际结束 | 外部退出码0、无终止信号；外部时长15.550164秒，脚本报告15.150669秒，低于240秒软上限 |
| 核心结果 | 21题初始评分与原M差异0；231次实际注意力调用均只允许正确索引，输出与所选点value差异0；人工非零门控时44个Q／K及query归一化张量梯度为0 |
| 工程结论 | 辅助点位接线、有限反传、有限保存恢复与缓存路径可用；本文零更新结果不承担拟合、自然坐标寻址或泛化结论 |

## 实现与输入边界

`src/goqueen/assisted_point.py`定义独立`AssistedPointGoQueen`，保留原M的Q／K／V、LayerNorm、共享行列编码和门控公式。每个实际样本提供`torch.long [B]`点位索引；布尔mask为`[B,1,1,361]`，仅所问点为`True`，在全部head、query位置与11个adapter中使用。注意力仍处理完整361个棋盘token；此变体没有M-Key的额外位置副本。

有限分数、math SDPA、dropout0且只允许一个点时，注意力结果就是该点的value。对给定样本与层，辅助分支为`tanh(alpha)·W_O·W_V·LN_expert(f_l[point]+p[point])`，与该层文本query无关，在每个文本token位置加入同一个向量；不同层的向量可以不同。外部已确定坐标，Q／K不再承担读懂坐标并选择点的任务。44个Q／K及query归一化张量合计20,207,616／40,438,795个参数值，在这一路径中不影响输出；保留它们只是维持原参数与初始化合同。整个冻结语言模型仍处理文本，不能把桥接query无关扩写成整模型不使用文本。

`src/goqueen/assisted_point_eval.py`只从`point_index`、`point`、`to_play`和规范化原提示核对坐标，不读取答案。它在原评分器的每个实际内部批次边界绑定该批次的点位，避免把21题索引误用于单题或其他批次。直接前向也核对缓存前缀与原提示一致。成功和异常退出后，专家特征、mask及索引均由`finally`清空；不支持同一模型的并发或嵌套绑定。

构造后的101个张量共40,438,795个参数值，其名称、形状、顺序、初始字节摘要与原M构造相同；本次并列构造的RNG状态相同。历史M未保存RNG状态，不能将本次比较扩写成历史RNG复核。21题初始label logP最大差异为0，分类和全词表argmax均与原M初始记录一致。零门控下的一致性不证明模型已经读取棋盘。

## 实际核验范围

| 检查 | 覆盖与结果 |
|---|---|
| 坐标与特征 | 原21题、5份缓存及来源哈希核对；输入棋盘恢复的1,805个占用标签核对；答案不进入前向元数据 |
| 单点注意力 | 21题逐题评分，每题11层，共231次实际SDPA调用；展开后的每个样本／head／query只允许规范坐标，返回所选value最大差异0 |
| 零门控反传 | 真实batch2；门控获得非零梯度，其余连接参数梯度为0；全部梯度有限 |
| 人工非零门控反传 | 将11个alpha人工设为0.01；真实batch2的value、输出、gate、专家LayerNorm及共享行列参数梯度组有限且非零；44个Q／K及query归一化张量梯度精确为0 |
| 冻结与清理 | 基础模型与缓存前后字节摘要相同，均无梯度；44个张量最终字节摘要相同；成功与异常退出后绑定状态清空 |
| 批量与左填充 | 实际两题完整前缀长度为74／74；batch与逐题label logP最大差异`2.121925354e-5`，小于`1e-4`；额外添加两列左pad后差异0 |
| 保存恢复 | 保存人工alpha=0.01的工程权重，清零全部101张量后通过APS加载器恢复；正确重新绑定后的logits最大差异0 |
| 生成缓存 | 单题无padding前缀、贪心生成两个新token；cache开启／关闭序列一致；未覆盖长生成、beam search或填充批量生成 |

本文实际梯度核验仅用两题、两次反传，不是完整batch4训练，也没有构造或调用AdamW。44个张量当时梯度为零、字节未变，不能单凭这些工程记录声称180次更新的合同已通过；后续单次拟合的实际检查另见[拟合结果](assisted-point-fit.md)。字节未变本身也不能证明索引正确，因此索引与所选value另行核对。

生成日志保留了无效采样参数可能被忽略、`pad_token_id=eos_token_id`且未显式传入生成`attention_mask`的提示。本次缓存比较使用无padding单题，只说明这一有限输入的两token结果相同；不据此声称一般生成接口或带padding生成已验证。左填充检查是另一个显式mask／position输入的前向检查。

## 权重合同与产物

`src/goqueen/checkpoints.py`已识别实际APS模型类型和`variant=m_assisted_point`，要求`routing_contract=single_point_bool_mask_v1`、工程／拟合标记、已完成更新数及题目／manifest／特征摘要。先检查全部名称、形状及APS张量有限性，再复制任何参数；小型检查覆盖APS与M／M-Key双向拒绝跨变体。摘要字段在加载器中检查格式，实际来源一致性仍须由运行入口核对，不能把摘要存在等同于当前输入已匹配。

`outputs/assisted-point-engineering/adapter.safetensors`只用于人工门控的工程恢复，元数据为`engineering_only=true`、`updates=0`，不是拟合任务权重。报告保存11份实际导入的第一方源码副本与各自哈希；进程记录保存外部退出、计时与报告哈希。公开进程生产器副本只做项目根路径归一化及私有目录省略，保留原始／副本哈希，未重新执行副本。`process-source-provenance.json`另保留作者对既有completion、stdout、stderr与原始文件逐字节相同的比对声明；这是文件来源核对，非新运行或独立运行复现。

| 相对产物路径 | SHA-256 |
|---|---|
| `outputs/assisted-point-unit-check/report.json` | `d45ab004dab9f390c85d4791d5cedc4b1258b59cce4ae92460603e4a7b18115c` |
| `outputs/assisted-point-unit-check/test.log` | `326d0d102dd8286172a75a5efe46a74dd9ea5804188c88e9a4cb194812a02d35` |
| `outputs/assisted-point-engineering/report.json` | `5c3f0a092628d6c0dbfbcfa0a09eaae71b965c4fda9707ce87d1b8ad1c00337b` |
| `outputs/assisted-point-engineering/process-completion.json` | `c466f1486d661ea2203ca4e25c96d19193099506cb7312067855cb380cf10850` |
| `outputs/assisted-point-engineering/process-source-provenance.json` | `b072f21b7065e95293f15ae59680b5d08157ecf7690d251c5aa511ae5637fc4c` |
| `outputs/assisted-point-engineering/adapter.safetensors` | `d78b5f3ce6a97d02a39a2cf0828251d67b9455f487aaaf3f84fb0e3d17be9371` |
| `scripts/check_assisted_point_engineering.py` | `e2eb13ebe44708d0c9fde8fe9f0c45aa2efe2e8e5478c0ed09b22572dcf523b1` |
| `src/goqueen/assisted_point.py` | `00155cbcfae7052aec9306cf95bbc6815d998e38c96220bf45998e1a4414f050` |
| `src/goqueen/assisted_point_eval.py` | `de3af14e363a91bda1d77e1b22eb5a4910309194c5b0eaa5fc7d84400f6b5a3b` |
| `src/goqueen/checkpoints.py` | `e0590f76d358cd3027f4850ec4bcecbef225eccc5c2c34e87bc7e67de08ca509` |

工程脚本入口为`scripts/check_assisted_point_engineering.py`，参数`--seconds`最多240，默认输出上述目录；已有目录拒绝覆盖。本次已完成且不重复。小型测试日志的套件时间为0.739秒，生产器总记录为1.828126秒，两者计时范围不同。

## 当前下一步

完整APS拟合runner及必要检查已实现，并已按[固定设计](grounding-assisted-point-design.md)完成唯一一次21题、720槽、180更新拟合：固定终点21/21，外部209.861817秒，小于300秒软预算，OS退出码0。本页15.55秒工程记录保留不变，新拟合记录见[单次拟合结果](assisted-point-fit.md)。本阶段停止，下一步先确定一个最小后继问题；工程及辅助拟合均不证明自然路由学会选点、原M失败的唯一原因或泛化。

## 简化选择

本次复用模块、点位接线与工程恢复记录，没有重复这套模型检查；仅为完整runner新增的采样、batch4训练、优化器和拟合保存／预算边界补必要检查。辅助拟合已成功，只回答外部选点后的下游路径能否完成这21题，不承担自然寻址的验证，不自动追加探针、调参或对照分支。
