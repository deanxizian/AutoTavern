# V8

Authorized finite stage: 100 PPO, 256 environments × 512 steps, max1600 Adam,16h. Full predecessor best and Adam inherited; fixed LR1e-5; BF16.

3 V5 + 4 V7; no V0. Opponents frozen and stochastic.

Fixed seven frozen stochastic opponents; learner sampled/greedy64 each at0/25/50/75/100; sampled strict decrease includes step0; independent test128 per mode initial/best/final never selects.

No global champion promotion or deployment.

## 实际完成记录

V8已完成100/100 PPO、13,107,200有效样本、1600新增Adam，最终完整checkpoint累计Adam10400。随机val各64局的均名为4.171875→4.265625→4.109375→4.3125→4.28125，按严格下降选PPO50 best（累计Adam9600），并完整继承至V9。

固定3V5+4V7、0V0，BF16；所选best收尾随机128局Top1=13.28125%、Top4=54.6875%、均名4.3046875；贪心128局Top1=13.28125%、Top4=52.34375%、均名4.3046875。独立test未选模或选池，不宣称全局冠军晋级。

[结果摘要与原始来源SHA](results-summary.json)，原始逐局与W&B曲线保留本地。完整best通过生命周期核验，登记为有效canonical V8；只保留该完整best并保留所有历史来源/对手。
