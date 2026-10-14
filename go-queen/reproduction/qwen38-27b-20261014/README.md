# Qwen3.8-27B / GoQueen 代码交接

本目录对应冻结官方 Qwen3.8-27B、冻结 KataGo、训练棋盘桥接模块的单张 RTX 5090 配方。仓库仅提供代码、依赖与配置。实际固定训练集、验证集、棋谱、特征缓存、权重、逐题评估和训练日志均另行提供，不在 GitHub 中。

实际运行的 10 个训练源码文件保持原字节；`source-pins.json` 列出来源与 SHA-256。源码个别开头仍保留实施前的“draft / not been run”注释，它们是原始版本文字，不是当前运行状态声明。本交接包不修改历史实验源码，也没有重新训练或运行测试。学习效果与原因分析另行记录，不能由代码包推断。

## 配方与代码入口

- 主训练：`scripts/run_qwen38_expanded.py` → `scripts/train_qwen38_expanded.py`。
- 数据及输入：`src/goqueen/expanded_training_data.py`、`qwen38_training_runtime.py`。
- 桥接：`src/goqueen/qwen38_fusion.py`；64 层中在零起始 `[0,6,13,19,25,32,38,44,50,57,63]` 层前注入。KataGo 11 个外层块结束特征，每层 `[361,768]`。
- 训练桥接宽度 1024、8 头、FP32，101 个张量共 132,803,595 参数；seed42、新桥接、零门控。NF4 冻结底座、BF16 compute、double quantization 关闭，`lm_head/in_proj_a/in_proj_b` 保留 BF16。
- 固定 126,561 训练题、14,499 验证题，分别来自 210/24 盘及 1260/144 个局面；按整盘拆分。五个完整 epoch、632,805 次曝光、39,555 次更新；microbatch/effective batch 均16，每轮末尾1题仍除以16。
- AdamW peak lr `1e-4`、396步warmup、cosine floor `0.1`、betas `(0.9,0.999)`、eps `1e-8`、weight decay0、clip1；non-reentrant gradient checkpointing。完整机器可读配方见 `configuration.json`。
- 每500步、初始、每轮末保存完整 adapter/Adam/RNG/cursor；同步持久回执后继续。`cloud_training_state.py`、`cloud_checkpoint_transport.py`、`scripts/receive_cloud_checkpoint.py` 给出实现。连接资料由部署者私下配置，不进仓库。

## 准备环境与外部数据

以下命令均在仓库 `go-queen/` 下执行。仅代码检查不需要下载模型；27B 权重仅在 `/workspace` 下的云端 Linux 主机下载或运行。已验证的硬件范围是单张 32GB RTX5090 的 NF4 配方，多卡 BF16 入口不是已验证训练方案。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r reproduction/qwen38-27b-20261014/requirements-cloud.txt
.venv/bin/python -m pip install causal-conv1d==1.7.0 --no-build-isolation --no-deps
git clone https://github.com/lightvector/KataGo.git vendor/KataGo
git -C vendor/KataGo checkout d91ea855110dae533f0aada947b2b7d78cc8a4e1
.venv/bin/python scripts/download_qwen38_assets.py
```

`requirements-cloud.txt` 是本次主要运行依赖的固定版本，不能使用项目旧小模型的 `requirements.txt` 替代。`environment-packages-observed.json` 是当时已安装包版本的只读清单，不是完整虚拟环境镜像；系统 CUDA 编译工具与二进制构建仍需匹配。新的下载辅助脚本仅作分发准备、未在本次打包时执行，不是实际训练的10份源码之一；下载和训练是两个明确步骤。

**先从项目维护者获取原始固定数据，按 `configuration.json.required_external_inputs` 的相对路径放置。** 需要完整主数据 manifest/questions、1,404份特征及 manifest，以及另列历史54题与18份特征；仅有本仓库不能启动同一数据实验。不要重新随机生成一套数据冒充这次固定集合。

`sources.json` 是当前缓存提取器使用的官方 KataGo 模型来源配置，只有模型版本、下载 URL、大小及 SHA，不含任何训练数据。

保留了 `prepare_katago_training_sources.py`、`extract_features_cuda_batch.py` 等原生成代码供核查。若缺缓存，先取得原固定 SGF/manifest 和相应历史输入，再按原生成协议提取到一个新目录；新提取的运行元数据或数值摘要可能不同，不能直接宣称与原存档完全同一身份，也不能绕过校验续训。

```bash
# 可选：已有原固定源数据且需重新提取主缓存时，单独执行；不是生成另一套训练集。
PYTHONPATH=src .venv/bin/python scripts/extract_features_cuda_batch.py \
  --sources outputs/katago-training-sources-20261013/manifest.json \
  --output outputs/katago-training-features-rebuilt \
  --checkpoint models/kata1-tf3-b11c768-s11003M-d5973M-7gres.ckpt \
  --batch-size 4 --threads 4 --seconds 1800
```

## 复现启动与排查

先部署独立持久接收端并私下准备连接配置（schema 见 `cloud_checkpoint_transport.load_config`）。以下是**新运行**命令模板，不是自动续跑指令；输出目录必须全新。要复现原实验，应使用另行获取的原主缓存和历史缓存。

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

第三方出处及许可见 `THIRD_PARTY_NOTICES.md` 与 `KataGo-LICENSE.txt`。Qwen 与 KataGo 权重只给官方下载出处和固定 SHA，不随仓库分发；数据及结果不随本代码包上传。
