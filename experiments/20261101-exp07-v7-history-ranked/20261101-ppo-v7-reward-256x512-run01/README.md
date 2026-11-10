# V7 独立启动实例

用户授权“好的，用这个，然后启动V7”，只运行V7，不恢复旧停止或暂停实例，不继续V8/V9。
奖励逐字段沿用已审核256×512准备稿，包含move.0005/cycle.01/前期逐步费/三个开局奖励/12英雄top1先验（scale.01 cap.005）；action_reward只分组，无数值变化。

预算100 PPO ×131072=13107200新样本、最多1600新Adam；LR固定1e-5，继承V5 best完整Adam7200；16h总上限，意外恢复最多1次。
最终val点PPO0/25/50/75/100，每点随机/贪心各64，仅随机均名严格下降选best，平局保留较早；checkpoint每25。
开发排名V1/V2/V3/V5各64，共256局同时推进，同权重跨阵容合批，参考2V1+2V2+2V3+1V0；选前三各2席+V0，V4/V6不入池。
收尾initial/best/final每模式128独立test不选模。完成后按实际策略/完整Adam核验登记active或retired；不自动晋级全局冠军。

启动前通过必要正确性检查并等待父任务monitor-ready。旧V7–V9三版草稿仍暂停；本目录为新实例，未绑定来源/排名前不直接运行experiment。

## 实际完成（2026-11-02 02:56）

完成100 PPO、13,107,200有效训练样本、1600新Adam，继承V5累计7200→8800，LR固定1e-5。冻结阵容为2V3+2V5+2V2+1V0；随机/贪心val每组64，0/25/50/75/100全部完成，严格随机均名4.125→3.453125选择PPO100 best。完整策略和Adam与iteration0100已核验，保留`20261101-115350-best_model.zip`（SHA-256 `cdb10c8bb08587b52dd2100ba71302920c549561747f47a7eb1e72fa7333f354`），V7登记active。

独立initial/best/final每模式128结果完整，best与final同policy复用，实际512局。best/final随机Top1 11.71875%、Top4 60.15625%、均名3.90625；贪心16.40625%/72.65625%/3.4375。test不选模，不自动晋级全球冠军。

用户两次人工继续均单独封存；旧自动恢复1/1未重置。75边界补val/metrics为0样本0Adam，原rollout行为缺失明确标记。旧丢弃采样至少125,164另列，实际训练采样支出下限13,232,364；诊断64采样/0Adam独立计账。原03:50:57截止未延长，02:56已全部完成，所有进程退出、授权关闭，不推进V8/V9或V4/V6清理。

证据：[results](results-summary.json)、[retention](retention.json)、[lifecycle](lifecycle.json)、[最终核验](../../20261031-infra/20261031-training-agents/instances/20261101-v7-reward-256x512-01/final-verification.json)、[W&B](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/iz2flcie)。
