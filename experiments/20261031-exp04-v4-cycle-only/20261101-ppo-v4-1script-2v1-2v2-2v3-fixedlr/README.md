# V4 新历史对手训练

创建于 2026-11-01T00:02:31.097928+08:00（北京时间）。实例 `20261031-v456-history-02`。

1 scripted + 2 fixed V1 + 2 fixed V2 + 2 fixed V3 + 1 learner; rotate physical seats every game。模型对手始终随机采样、BF16。
学习者随机／贪心各64局，PPO0/50/100验证；仅随机平均名次严格降低选best。
独立 initial/best/final 每模式各128局，不选模、不排弱、不决定扩展。
100 PPO、13,107,200学习者样本、最多1600新Adam，LR全程1e-5，继承来源Adam。
V1/V2/V3冻结对手不加新提前判负；V4+按来源实验协议逐席判负。

未来来源在前阶段best完成后绑定，未绑定不能启动。
