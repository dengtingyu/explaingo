# ExplainGo / GoQueen

19 路围棋版 QUEEN 研究项目。当前代码交接入口为 **[Qwen3.8-27B 单卡训练代码与复现配置](go-queen/reproduction/qwen38-27b-20261014/README.md)**。包含实际训练源码、依赖版本、模型来源配置和检查用测试源码；训练集、棋谱、缓存、权重及运行结果均另行提供，不放入本仓库。

历史 APS / QK 代码与技术说明保留在 **[go-queen/](go-queen/README.md)**。

- [项目总览与代码入口](go-queen/README.md)
- [固定最终权重的训练题评估与分层结果](go-queen/docs/aps-warm-qk-train-eval.md)
- [APS 起步 Q/K 训练结果](go-queen/docs/aps-warm-qk-fit.md)
- [辅助选点评估结果](go-queen/docs/assisted-point-transfer-results.md)

仓库提供源码、测试源码、配置及已有实验的技术说明；**不包含模型权重、专家缓存、数据清单或原始实验输出**。运行模型实验需要另行准备这些依赖，不能仅凭本仓库独立复现历史成绩。

历史小模型固定训练题评估为 1,256/2,702（46.4841%）；自然按坐标读取棋盘、一般泛化与棋力仍未建立。结果及适用范围见上述说明。
