# Eric 37.0.2 随机初始化单阶段

实例 `eric-3702-fresh-v1`。本任务在独立服务器 checkout 将 Eric
`719a3491eb0b95c16f5d028d549d944cbc7cd286` 与 upstream
`1c2086a997afde46d68c6c5992cab194ddcfce88` 真正合并，规则为
37.0.2.256589 / server 37.0.2；保留 Eric 训练、PyO3 和批量采样实现。
此新谱系从随机初始化开始，不是 V10，也不继承任何历史权重、Adam 或文本缓存。
当前可执行状态见服务器 `/root/autodl-tmp/bootstrap/eric-3702-status/` 的状态账本。

冻结预算：[config.json](config.json)。一个学习者与七个原生 V0，显式
`simulator_config.hero_choices=2`、`turn_action_caps=[60]`；无循环/动作提前判负。
256 环境×1024步，每 PPO 262144 样本，最多100个完整PPO / 26214400样本；
有效 batch8192、microbatch1024、epoch1，LR3e-4线性到3e-6，无warmup。
BF16前向，FP32参数/累积梯度/Adam，operation_cost=0、cycle_cost=.0002、all_actions。
训练/开发/独立测试种子分别为37020001/37022001/37023001。

开发起点及 PPO25/50/75/100，学习者随机/贪心各64局；只有随机均名严格降低才换best，
平手保留早者。结束仅对已选best使用独立种子随机/贪心各128局；不使用test选模型。
沿用Eric训练内评估入口的32环境与逐环境计分quota；64/128是计分总局数，
不声称64/128全量并行。已达quota环境可能推进未计分片段，不纳入指标或选模。
即使best仍是PPO0，也如实保留随机初始化及空Adam，不从别的模型补入权重。
不同补丁/不同对手阵容的历史指标不横比。

本轮最小兼容改动：显式新谱系允许封存的七分区schema v2；全V0也能进行两模式评估；
收尾可显式选择best；首轮额外保存完整边界；初始化原生配置显式透传；
新谱系总览与旧V1–V9记录分开；非有限loss/KL不会覆盖干净的latest边界。
历史配置未选择这些新选项时保持原协议。

[来源与引擎指纹](source-manifest.json)记录卡表/引擎/锁摘要。3983个基础ID顺序不变，
含padding的N_CARDS=3984，245动作、3227数值、124卡索引；卡表已包含37.0.2数值、
卡池、Dark Gift和饰品更新。卡表英文文本79项变化，新的文本资产必须重新生成。
静态向量由当前原生`card_static()`重新构造；文本使用固定MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，资产放新谱系的runs目录。

`cuda_preflight.py`在独立37024001种子运行8192采样与1次Adam，使用相同小模型和microbatch1024；
其检查点与计数只属于预检，正式初始化ZIP哈希保持不变。
完整Rust/Python门禁与预检通过后才写validation-gates.json passed，才允许`supervise.py`启动。

正式命令由`supervise.py`记录在状态目录training-launch.json；监督进程和GPU训练各自
脱离SSH进程组，标准输入为/dev/null。最多6小时，到时SIGTERM请求保存退出；
最近已完成PPO的latest与编号检查点不被中断rollout覆盖。STOP文件或向守护发送SIGTERM可停，
磁盘低于2GiB时停，故障暂停；不自动循环重启，不自动新增阶段。
最多一次经过诊断的环境故障恢复也必须核验完整边界、记录来源、扣除已用累计时间/样本。
本入口主动拒绝已有run目录，不隐式恢复。

W&B使用offline，entity=null，不上传Eric组织；七分区本地journal和SDK硬件记录保留本地。
权重、原始曲线、逐局结果、W&B缓存和临时输出均不提交。此checkout初始没有旧模型文件，
未删除任何历史模型，也未从其他工作区导入；Git bundle仅备份仓库历史，不能称作权重备份。
