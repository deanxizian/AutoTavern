# 2026-10-30 训练暂停与恢复交接

本次按用户要求为 RTX 5090 停机收尾。训练于 **2026-10-30 23:12:05
Asia/Shanghai** 保存 u23 后退出，原因是 `stop_file`。23:29:31 的最终核验确认
训练、完整性检查及本地备份/评测监视进程均已退出，GPU 没有计算进程。
没有执行机器关机，也不会自动恢复训练。

## 保存进度与验证边界

| 项目 | 已保存状态 |
| --- | --- |
| 实验 | `ppo-continuation-v2`，按用户要求暂停 |
| 完成更新 | u23 |
| 谱系累计采样 | 6,029,312 次学习席位决策 |
| 本阶段新增 | 从 u19 开始，4 次更新、1,048,576 次决策、5,363 局完成对局 |
| 每次更新 | 1,024 个环境 × 256 步 = 262,144 次决策 |
| 本阶段完整性 | u20–u23 检查通过，无训练截断或非法动作拒绝 |
| 阶段末开发评测 | 计划中的 512 局尚未运行 |

这些是保存与运行完整性结果，不能推出 u23 的吃鸡率、前四率或实战水平。
最近完成的正式开发评测仍为 u19；客户端单独固定使用通过回放适配验证的 u20，
不会随训练检查点自动升级。客户端记录见 [实战问题复盘](CLIENT_MATCH_20261030.md)，
模型准入标准见 [MODEL_READINESS.md](MODEL_READINESS.md)。

原冻结控制器只接受 `runtime_only` 停止原因，因此在 learner 已完成保存退出后，
对本次 `user_requested_server_shutdown` 记录了 `AssertionError`。原始记录保留，
本次状态记为用户主动暂停，不是训练崩溃，也不是正常完成整个实验预算。

## 本地恢复材料

以下路径相对于项目根目录，均在忽略的 `runs/` 中；Git 提交不包含模型、原始日志、
私有策略数据或运行环境。仅克隆仓库不足以恢复本次训练，仍需这些私有备份。

- `runs/shutdown/20261030-5090/latest-full.pt`：首选恢复检查点，344,229,195 字节，
  含模型、Adam、环境、CPU/CUDA/NumPy RNG、对手身份和累计计数。
  SHA256：`775618c94cdc1760737d10469df64962ff4b3d78c7f6bd35860f593d542205ac`。
- `runs/remote-backup/ppo-continuation-v2/`：u20–u23 的完整/推理检查点、原计划、
  `evidence.tar.gz` 和 manifest；归档及其 149 个成员均已校验。
- `runs/shutdown/20261030-5090/`：`RECOVERY.md`、`environment.json`、
  `repository.bundle`、`recovery-support.tar.gz`、`backup-review.json`、
  `final-state.json` 及完整性、状态等价性证据。
- `runs/research/reference-bank-strategy-api-v1/bundles/`：3 个参考对手的
  actor/encoder/ridge，共 9 个文件；校验记录见停机目录的 `reference-backup-review.json`。

最终 `latest-full.pt` 与不可变 `u23-full.pt` 文件哈希不同，因为累计耗时与 gzip
时间戳变化。实际加载比较确认模型、优化器、随机数状态及解压后的环境完全一致；
证据为 `final-checkpoint-equivalence.json`。精确续训使用完整检查点，不能以推理用
`u23.pt` 代替优化器和环境状态。

## 经用户要求恢复时

1. 先确认机器、GPU、磁盘、环境和现存进程，避免重复启动。目标模式保持暂停，
   本次提交不授权恢复长期训练。
2. 使用实际训练的冻结源码提交 `a14664c56fd30e35117172fccd82b179f35cb271`。
   训练 source hash 为 `e262adc9d3072fc293c6416b9664a0613330cc23cb10fcf465bf7879498dd985`，
   engine hash 为 `35af4beacd81af84`。按 `environment.json` 恢复依赖并检查指纹，
   不要误用服务器根目录的旧 checkout，也不要修改元数据绕过检查。
3. 保留旧 run 的 `training/STOP` 与终止记录，创建新的计划和输出目录，以
   `latest-full.pt` 作为 `--resume`。从原计划与检查点参数恢复对手、规则、策略、
   环境、种子及优化参数，包括英雄二选一和每回合 128 动作上限。
   `strategy_coef=0` 不代表关闭推理时的策略先验。
4. 上限按累计值设置：`--max-updates = 23 + 新增更新数`；
   `--max-hours = 32901.461649894714 / 3600 + 新预算小时数`。
   新预算在恢复时确定。
5. 不直接重跑旧 `train_ppo_continuation_v2.py` 控制器：它绑定 u19 起点与旧输出
   路径。先验证新计划的恢复状态，再启动续训或补充评测。

通用方法与命令见 [TRAINING.md](TRAINING.md)。本次客户端提交没有修改冻结训练
源码或引擎，也没有替换客户端已选定的模型。
