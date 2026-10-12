# ExplainGo / GoQueen

19 路围棋版 QUEEN 研究项目。最新第一方代码和技术说明位于 **[go-queen/](go-queen/README.md)**。

- [项目总览与代码入口](go-queen/README.md)
- [固定最终权重的训练题评估与分层结果](go-queen/docs/aps-warm-qk-train-eval.md)
- [APS 起步 Q/K 训练结果](go-queen/docs/aps-warm-qk-fit.md)
- [辅助选点评估结果](go-queen/docs/assisted-point-transfer-results.md)

仓库提供源码、测试源码、配置及已有实验的技术说明；**不包含模型权重、专家缓存、数据清单或原始实验输出**。运行模型实验需要另行准备这些依赖，不能仅凭本仓库独立复现历史成绩。

当前固定训练题评估为 1,256/2,702（46.4841%）；自然按坐标读取棋盘、一般泛化与棋力仍未建立。结果及适用范围见上述说明。
