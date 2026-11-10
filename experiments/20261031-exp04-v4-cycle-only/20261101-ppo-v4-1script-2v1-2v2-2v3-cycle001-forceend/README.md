# V4 固定循环成本与强制End

创建于2026-11-01T00:26:48.008161+08:00（北京时间），实例`20261101-v456-history-03`。

1 scripted + 2 fixed V1 + 2 fixed V2 + 2 fixed V3 + 1 learner; rotate physical seats every game。每次返回已见完整可见状态固定扣0.01，不单次递增；操作成本0。
两种提前判负均关闭，原生60步上限保持强制End并正常战斗/继续对局。
来源V1/V2/V3不套新规则；未来冻结V4+按本链路真实best协议。
100PPO、13,107,200学习者样本、最多1600新Adam，固定LR1e-5、继承来源Adam。
PPO0/50/100随机/贪心各64局，仅随机平均名次严格降低选best；独立每模型每模式128局不选模。
重复指标：cycle_returns、cycle_return_fraction、max_consecutive_cycle_returns、cycle_penalty。

后续来源待真实前阶段best完成后绑定；未绑定不能运行。
