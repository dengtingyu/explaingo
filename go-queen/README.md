# 当前训练代码地图

这里是 Qwen3.8-27B 版本。项目状态见[仓库首页](../README.md)，完整环境、参数、启动与权重读取说明见[固定复现入口](reproduction/qwen38-27b-20261014/README.md)。以下路径均相对本目录；不需要先阅读旧实验。

## 接入是怎样工作的

```text
棋盘 → 冻结KataGo → 11组特征，每组361点×768维
                              ↓ 加共享行列位置嵌入
文字问题 → 冻结Qwen → 11处FP32跨注意力桥接 → Qwen输出头 → 空/黑/白评分
                         ↑ query来自Qwen；key/value来自全部361点
```

当前取 `trunk_after_block0` 至 `trunk_after_block10`，包括第一个外层块结束的特征；每个外层块内部有3个Transformer块。它不是只输入题目指定点：正确坐标写在文字提示中，桥接要从全部361点学习读取。

Qwen共64层，在零起始 `[0,6,13,19,25,32,38,44,50,57,63]` 层前注入。桥接宽1024、8头，含归一化、Q/K/V/O投影、门控及共享位置嵌入。底座冻结，**全部101个桥接张量可训**，不是旧方案的44个Q/K张量。桥接以FP32计算，残差转回语言模型的BF16。

## 按这个顺序读代码

1. **[qwen38_fusion.py](src/goqueen/qwen38_fusion.py)**：`NarrowBoardCrossAttention` 是桥接；`Qwen38ExpertDecoderLayer` 是注入点；`Qwen38GoQueen` 管理冻结、特征绑定与101参数枚举。
2. **[qwen38_training_runtime.py](src/goqueen/qwen38_training_runtime.py)**：`encode_questions` 构造聊天提示和答案token；`make_inputs` 处理padding；`forward_logits` 取最后真实位置；`score_batch` 给出两种首选与CE。
3. **[train_qwen38_expanded.py](scripts/train_qwen38_expanded.py)**：`load_base` 加载NF4/BF16底座；主循环构造桥接/AdamW、绑定特征、计算loss、反传、裁剪、保存与评估。
4. **[run_qwen38_expanded.py](scripts/run_qwen38_expanded.py)**：训练子进程入口，管理日志、退出状态与持久化收尾。启动命令见复现说明。

## 按问题找文件

- **坐标、黑白、题目与棋盘有没有对上**：[`prepare_katago_training_sources.py`](scripts/prepare_katago_training_sources.py) 构造题目与整盘拆分；[`expanded_training_data.py`](src/goqueen/expanded_training_data.py) 校验点位、prompt、标签与特征身份。
- **KataGo取了什么、361点顺序是否一致**：[`extract_features_cuda_batch.py`](scripts/extract_features_cuda_batch.py) 是当前提取入口；它仍调用 [`extract_features.py`](scripts/extract_features.py)、[`build_batch_data.py`](scripts/build_batch_data.py)、[`prepare_expanded_sources.py`](scripts/prepare_expanded_sources.py)，所以这三份依赖保留。
- **特征是否进入了这一次前向/反传**：查看 `qwen38_fusion.py` 的 `bind_expert` 和训练循环对应上下文。绑定需覆盖forward至backward，不能提前清除。
- **loss和评分是否一致**：训练用答案token的全词表CE。三类强制首选按 `empty, black, white` 同分取前者；全词表按token ID。`lm_head`先以BF16计算，再转FP32，不会恢复已丢失的精度。这些是排查入口，不是已定位的失败原因。
- **epoch、尾批与梯度**：[`epoch_schedule.py`](src/goqueen/epoch_schedule.py) 管完整遍历；每轮尾批1题的loss仍除以16；训练入口检查底座无梯度，桥接裁剪范数1。
- **缓存与恢复**：[`cached_features.py`](src/goqueen/cached_features.py) 管特征缓存；[`cloud_training_state.py`](src/goqueen/cloud_training_state.py) 保存adapter/Adam/RNG/cursor；[`cloud_checkpoint_transport.py`](src/goqueen/cloud_checkpoint_transport.py) 和 [`receive_cloud_checkpoint.py`](scripts/receive_cloud_checkpoint.py) 实现完整单元回执。

## 测试与交接边界

[`tests/`](tests/) 只保留当前链路相关测试：题目构造、缓存、epoch调度、完整保存/传送、launcher终态与内存边界。其中部分使用NumPy/Torch，小型夹具通过不代表学习效果达标。

独立的最终27B权重评估命令尚未实现；不要把 `--resume` 当评分命令。已有加载与评分API的准确调用顺序见[复现说明中的权重读取部分](reproduction/qwen38-27b-20261014/README.md#已训练权重的读取与评分接口)。维护者另行提供同一运行的固定数据、RUN、完整checkpoint和原始评估记录。

环境依赖只看 [`requirements-cloud.txt`](reproduction/qwen38-27b-20261014/requirements-cloud.txt)，参数只看 [`configuration.json`](reproduction/qwen38-27b-20261014/configuration.json)；不使用旧小模型配置。历史源码头部的draft注释属于原运行版本，当前状态以首页为准。
