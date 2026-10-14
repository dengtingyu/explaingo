# Qwen3.8-27B / GoQueen 代码交接

本目录对应冻结官方 Qwen3.8-27B、冻结 KataGo、训练棋盘桥接模块的单张 RTX 5090 配方。仓库仅提供代码、依赖与配置。实际固定训练集、验证集、棋谱、特征缓存、权重、逐题评估和训练日志均另行提供，不在 GitHub 中。

实际运行的 10 个训练源码文件保持原字节；`source-pins.json` 列出来源与 SHA-256。源码个别开头仍保留实施前的“draft / not been run”注释，它们是原始版本文字，不是当前运行状态声明。本交接包不修改历史实验源码，也没有重新训练或运行测试。当前训练进展的简要汇总见[仓库首页](../../../README.md)。逐题评分、完整诊断及原始记录由维护者私下交付，不存入本代码库。

## 配方与代码入口

- 主训练：`scripts/run_qwen38_expanded.py` → `scripts/train_qwen38_expanded.py`。
- 数据及输入：`src/goqueen/expanded_training_data.py`、`qwen38_training_runtime.py`。
- 桥接：`src/goqueen/qwen38_fusion.py`；64 层中在零起始 `[0,6,13,19,25,32,38,44,50,57,63]` 层前注入。KataGo 11 个外层块结束特征，每层 `[361,768]`。
- 训练桥接宽度 1024、8 头、FP32，101 个张量共 132,803,595 参数；seed42、新桥接、零门控。NF4 冻结底座、BF16 compute、double quantization 关闭，`lm_head/in_proj_a/in_proj_b` 保留 BF16。
- 固定 126,561 训练题、14,499 验证题，分别来自 210/24 盘及 1260/144 个局面；按整盘拆分。五个完整 epoch、632,805 次曝光、39,555 次更新；microbatch/effective batch 均16，每轮末尾1题仍除以16。
- AdamW peak lr `1e-4`、396步warmup、cosine floor `0.1`、betas `(0.9,0.999)`、eps `1e-8`、weight decay0、clip1；non-reentrant gradient checkpointing。完整机器可读配方见 `configuration.json`。
- 每500步、初始、每轮末保存完整 adapter/Adam/RNG/cursor；同步持久回执后继续。`cloud_training_state.py`、`cloud_checkpoint_transport.py`、`scripts/receive_cloud_checkpoint.py` 给出实现。连接资料由部署者私下配置，不进仓库。

## 准备环境与外部数据

以下命令均在仓库 `go-queen/` 下执行。仅代码检查不需要下载模型；27B 权重仅在 `/workspace` 下的云端 Linux 主机下载或运行。已验证的硬件范围是单张 32GB RTX5090 的 NF4 配方，多卡 BF16 入口不是已验证训练方案。训练入口硬性检查 Linux、项目位于 `/workspace/` 内、可见 GPU 数量与硬件模式一致且型号含 `5090`；不要自行解除这些校验来冒充原环境复现。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r reproduction/qwen38-27b-20261014/requirements-cloud.txt
.venv/bin/python -m pip install causal-conv1d==1.7.0 --no-build-isolation --no-deps
git clone https://github.com/lightvector/KataGo.git vendor/KataGo
git -C vendor/KataGo checkout d91ea855110dae533f0aada947b2b7d78cc8a4e1
.venv/bin/python scripts/download_qwen38_assets.py
```

`requirements-cloud.txt` 是本次主要运行依赖的固定版本，本次环境以此文件为准。`environment-packages-observed.json` 是当时已安装包版本的只读清单，不是完整虚拟环境镜像；系统 CUDA 编译工具与二进制构建仍需匹配。新的下载辅助脚本仅作分发准备、未在本次打包时执行，不是实际训练的10份源码之一；下载和训练是两个明确步骤。

**先从项目维护者获取原始固定数据，按 `configuration.json.required_external_inputs` 的相对路径放置。** 需要完整主数据 manifest/questions、1,404份特征及 manifest，以及另列历史54题与18份特征；仅有本仓库不能启动同一数据实验。不要重新随机生成一套数据冒充这次固定集合。

`sources.json` 是当前缓存提取器使用的官方 KataGo 模型来源配置，只有模型版本、下载 URL、大小及 SHA，不含任何训练数据。

保留了 `prepare_katago_training_sources.py`、`extract_features_cuda_batch.py` 等原生成代码供核查。若从源头完整重建，还需要外置的 `data/pilot-sources.json`、`data/expanded-sources.json`、它们引用的历史 SGF，以及固定8个源归档的完整清单与 SHA；这些是旧棋谱排除和来源核验的输入，均向维护者另行取得。本库不是独立的数据复现包。若缺缓存，先取得原固定 SGF/manifest 和相应历史输入，再按原生成协议提取到一个新目录；新提取的运行元数据或数值摘要可能不同，不能直接宣称与原存档完全同一身份，也不能绕过校验续训。

```bash
# 可选：已有原固定源数据且需重新提取主缓存时，单独执行；不是生成另一套训练集。
PYTHONPATH=src .venv/bin/python scripts/extract_features_cuda_batch.py \
  --sources outputs/katago-training-sources-20261013/manifest.json \
  --output outputs/katago-training-features-rebuilt \
  --checkpoint models/kata1-tf3-b11c768-s11003M-d5973M-7gres.ckpt \
  --batch-size 4 --threads 4 --seconds 1800
```

## 复现启动与排查

先部署独立持久接收端并私下准备连接配置（schema 见 `cloud_checkpoint_transport.load_config`）。必填字段为 `host`、`port`、`identity_file`、`known_hosts`、`run_id`、`receiver_sha256`；可选字段为 `user`、`target_id`、`timeout_seconds`、`expected_run_identity_sha256`。密钥文件与 known-hosts 文件必须是私下配置的绝对路径常规文件，接收端源码摘要须匹配；这些实际连接值不进入仓库。原训练入口强制要求此接收端，缺少时不能跳过持久化约束。以下是**新运行**命令模板，不是自动续跑指令；输出目录必须全新。要复现原实验，应使用另行获取的原主缓存和历史缓存。

```bash
PYTHONPATH=src .venv/bin/python scripts/run_qwen38_expanded.py \
  --hardware nf4-single-gpu --microbatch 16 --cache-capacity 1404 \
  --model-dir models/Qwen3.8-27B \
  --asset-record models/qwen-assets.local.json \
  --asset-hashes models/qwen-asset-hashes.local.json \
  --source-manifest outputs/katago-training-sources-20261013/manifest.json \
  --feature-manifest outputs/katago-training-features-cuda-20261013/manifest.json \
  --legacy-manifest outputs/qwen38-legacy-reference/manifest.json \
  --expected-train 126561 --output outputs/qwen38-expanded-reproduction \
  --persistence-config config/persistence.local.json
```

排查优先看：最后真实 token 的选取与 loss、空/黑/白 token 映射、KataGo 点位展开顺序、共享位置嵌入、零门控与反传绑定生命周期、完整 epoch 顺序/尾步缩放。`tests/` 中对应标准库夹具可检查存档/调度/资源边界，但不是学习效果测试。没有在本次交接中执行这些测试；原模型、输入与恢复身份仍严格核验，不自动改变配方或重启。

## 已训练权重的读取与评分接口

本库尚无单独加载最终27B权重并评估的命令。训练入口的 `--resume` 遇到已完成的最终边界会主动拒绝继续，它不是最终 checkpoint 的评分入口。以下仅说明已有函数的复用顺序，不代表新增评估脚本已实现或验证：

1. 向维护者取得该次 `RUN.json` 及完整 checkpoint。`cloud_training_state.resolve_checkpoint(checkpoint_root, expected_identities)` 可核验最新提交边界的 `MANIFEST.json`、`READY`、载荷文件集合/大小/SHA及运行身份；调用方仍须核对官方模型 revision、量化配方、源码和输入摘要确属目标运行。另核对 `architecture.json` 及 safetensors metadata 中的 `model`、`global_step`、`adapter_attention_width`，不能仅凭文件名认定是最终权重。
2. 按原 `train_qwen38_expanded.load_base` 配方加载底座，保留来源与实际底座摘要检查。构造相同 `Qwen38GoQueen` 后，先用 `adapter_named_parameters()` 获取全部101个参数；使用 `safetensors.torch.load_file` 读 adapter，先完整核对名称、形状、FP32 dtype和有限性，再按训练源码已有的 `torch.no_grad()` / `parameter.copy_()` 模式加载，不能把仅含 adapter 的文件当作完整语言模型 state_dict。
3. 全部加载完成后调用 `set_trainable_adapter_names([])` 冻结底座与全部 adapter，并调用 `eval()`。`adapter_named_parameters()` 不依赖 `requires_grad` 过滤，冻结后仍返回完整101个参数；仅评分不需要创建优化器或恢复 Adam 状态。
4. 组员可复用 `qwen38_training_runtime.encode_questions`、`feature_batch`、`forward_logits`、`score_batch`、`aggregate`；一次前向需处于相应的 `model.bind_expert(features)` 上下文中，并保持原 tokenizer、提示格式、点位顺序与标签映射。实际重载和评分应另有明确输入与输出范围，不应通过重开训练来代替。

`lm_head` 保留 BF16。`forward_logits` 在输出后转为 FP32，这不会恢复 BF16 输出已经损失的精度。`score_batch` 的三类顺序为 `empty, black, white`，`argmax` 同分优先靠前类别；全词表则优先最小 token ID。本次映射为 empty=3092、white=5603、black=11124，因此黑白在最高分处并列时，两种评分可能选择不同颜色。排查时应区分同分规则与颜色偏好，结合原始分数、CE和并列情况解释；不改写原已保存指标。

坐标先验脚本 `inspect_qwen38_coordinate_prior.py` 及接线检查脚本 `check_qwen38_narrow_cloud.py`、`check_qwen38_fusion_cloud.py` 未列入本最小发布包，可向维护者取得。旧工程接线检查不等于最终27B模型的效果验证；记录中的学习现象与原因讨论请结合另行提供的完整诊断，而不是只看代码接线清单。

第三方出处及许可见 `THIRD_PARTY_NOTICES.md` 与 `KataGo-LICENSE.txt`。Qwen 与 KataGo 权重只给官方下载出处和固定 SHA，不随仓库分发；数据及结果不随本代码包上传。
