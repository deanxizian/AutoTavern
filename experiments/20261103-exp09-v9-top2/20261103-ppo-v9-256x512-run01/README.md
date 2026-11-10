# V9

Authorized finite stage: 100 PPO, 256 environments × 512 steps, max1600 Adam,16h. Full predecessor best and Adam inherited; fixed LR1e-5; BF16.

development rank 1 x4 + rank 2 x3; no V0. Opponents frozen and stochastic.

Fixed seven frozen stochastic opponents; learner sampled/greedy64 each at0/25/50/75/100; sampled strict decrease includes step0; independent test128 per mode initial/best/final never selects.

No global champion promotion or deployment.

## 实际启动记录

V9已锁定4V8+3V3，七个对手全部冻结随机、无V0。学习者完整继承V8所选PPO50 best及Adam9600，阶段计数归零；固定LR1e-5，BF16。起点随机/贪心各64局，均名4.84375/3.90625。

[开发排名摘要](ranking-summary.json)、[实际384局共同调度证据](ranking-scheduler-summary.json)。PPO25随机64局均名4.234375严格降低，已暂选本阶段PPO25 best/完整Adam10000；后续以同协议val严格降低更新，独立test不选模。实时进度和后续best以[总览](../../EXPERIMENT_OVERVIEW.md)为准。

## 最终完成记录

V9已完成100/100PPO、13,107,200有效样本、1600新增Adam，最后完整checkpoint累计Adam11200。最终仍按随机val严格降低选择PPO50 best（累计Adam10400）；模型生命周期active，训练与收尾状态completed，控制器已退出。

固定4V8+3V3、0V0，BF16；所选best独立随机128局Top1=7.8125%、Top4=46.875%、均名4.59375；贪心128局Top1=15.625%、Top4=60.15625%、均名3.84375。独立test不选择checkpoint或训练池。[结果摘要及来源SHA](results-summary.json)。

本有限V8→V9链路已结束，未进入V10、未部署、未提交或推送Git。
