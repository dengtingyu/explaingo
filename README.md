# GoQueen：KataGo → Qwen

让 Qwen 读取 KataGo 的棋盘特征，回答 19 路棋盘某个坐标是**空、黑棋还是白棋**。当前版本冻结 KataGo 与 Qwen3.8-27B，只训练两者之间的桥接模块。

## 组员从这里开始

1. **看接入和排错位置**：[代码地图](go-queen/README.md)。先看 `qwen38_fusion.py`，再看 `qwen38_training_runtime.py` 和训练循环。
2. **看训练参数、环境和启动方法**：[固定配方与复现说明](go-queen/reproduction/qwen38-27b-20261014/README.md)。机器可读参数在 [configuration.json](go-queen/reproduction/qwen38-27b-20261014/configuration.json)。
3. **核对排查的源码版本**：[source-pins.json](go-queen/reproduction/qwen38-27b-20261014/source-pins.json)。10份正式运行源码保持原字节。

## 当前进展（2026-10-14）

- 五个完整 epoch 已完成：126,561道训练题，39,555次更新，batch16，AdamW 峰值学习率 `1e-4`；共训练132,803,595个桥接参数。
- 固定终点验证集三类首选正确 **7,033 / 14,499（48.51%）**；全词表首选为 **7,140 / 14,499（49.24%）**。两者同分规则不同，不能混用。
- 目前在排查识别效果偏低。训练完成不代表可靠棋盘读取已经建立；模型大小、特征接入、监督与数值精度的影响尚未被分别确定。
- 保存分数中6,473题出现三类最高分并列。这是已观察现象，尚不能确定为效果不佳的唯一原因。最终权重在完整训练集上的固定评估尚未完成。

## 目录分类

```text
go-queen/
├── src/goqueen/       KataGo→Qwen桥接、token评分、数据接线、调度与保存
├── scripts/          训练入口、特征提取、来源准备、checkpoint接收
├── tests/            上述代码的单元与边界测试
├── reproduction/     当前27B版本的固定参数、依赖、模型来源、许可和复现说明
└── sources.json      官方KataGo模型来源配置
```

本分支只保留当前训练和接入所需代码及说明。训练集、棋谱、特征缓存、权重、逐题结果、日志和连接资料**不放 GitHub**，由维护者另行交付；仅克隆代码不能复现同一实验。

旧0.6B、APS和warm-QK实验已移出当前目录，仍可在[整理前的历史提交](https://github.com/dengtingyu/explaingo/tree/ac10909a964615bcbe6553b1b1fa130e9cc7988a)回看。
