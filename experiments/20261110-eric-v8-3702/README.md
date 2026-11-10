# Eric V8 · TavernGym 37.0.2

本轮交付按用户最新指令命名为 **V8**，在 `deanxizian/eric-3702` 谱系中对应内部 **C007 best75**、累计 Adam **11600**。命名不改变权重：ZIP SHA256 为 `10da2508386bf38472bdbb13e3d1e32cbbcd6638640cd99ed686b6c5d0b21c10`。Eric 上游历史 V8/V9 保持原记录；C004 曾使用的本地 V8 称呼保留为来源历史，不代表当前交付。

来源为 Eric V7 完整学习参数和 Adam8800，审计适配37.0.2静态/文本buffers，经过C004 best100/Adam10400，再续训获得本轮best75/Adam11600；未随机初始化、未导入旧自建PPO模型。实际阶段完成100PPO/13,107,200新样本/1600新Adam，所选best是75PPO/1200新Adam。完整ZIP保留Adam；配套RNG和PPO边界保留在本地，NativeGame原世界历史未序列化，不声称能逐动作重放中断对局。

## 独立验证

37.0.2、英雄二选一、native60步仅强制End、BF16 forward/FP32参数。每局1个被测模型，冻结对手相同：**2V1+2V2+2V7+1V0**。随机各768局，3个独立种子各256局；贪心各256局，2个种子各128局。总2048局全部在第一步前初始化，5组10分片跨角色按同模型合并推理，无补局；V0行为统计未知。

随机模式（各768局）：

| 模型 | Top1 ↑ | Top4 ↑ | 平均名次 ↓ |
|---|---:|---:|---:|
| 适配 V7 | **21.09375%** | 68.75000% | 3.51171875 |
| V8 / C007 best75 | 19.921875% | **75.00000%** | **3.27343750** |

名次差(V8−V7)=-0.23828125，整局成对bootstrap10000次95%区间[-0.421875,-0.04947917]；三个随机seed的点估计均改善。满足预登记门槛：改善≥0.1、合并区间上界<0、每随机批次改善、贪心回退≤0.1。不是每个seed均显著，也不是所有指标占优：随机Top1略低。贪心各256局的平均名次为V8 3.33203125、V7 3.6171875，其差值区间包含0，不单独宣称显著。

开发晋级已在独立共同开发1280局中完成；此后模型冻结，独立test未改学习者、冠军、超参数或池。第一次独立block1因守护基础设施故障失败，1936/2048原终局保留且不统计部分质量；在读新test前预登记了唯一基础设施恢复例外，以同模型/同门槛和新种子执行block2，不复用原时钟、不重试至通过。完整raw、14,336模型玩家局和213,719回合已无损归档并SHA核对，Git仅保存汇总和来源SHA。

完整数值、种子、判定与来源哈希见 [summary.json](summary.json)，实际奖励/学习预算见 [training-recipe.json](training-recipe.json)，动作/循环/触顶分布见 [behavior-summary.json](behavior-summary.json)。不同阵容的旧val胜率不能横向排名；模拟器成绩不等于真实炉石胜率。

## 代码与权重

[configs/eric-3702-models.json](../../configs/eric-3702-models.json) 将当前V8明确绑定到完整权重SHA/Adam/规则。权重和RNG不提交Git；取得原始ZIP后放入 `runs/eric-3702-v8/20261110-022609-best_model.zip`，校验SHA再加载，不能绕过卡表哈希或复用旧效果文本缓存。

本提交包含Eric训练实现、37.0.2平衡规则、native观察传输和冻结模型批量评估。修复了嵌套Discover/付费饰品选择结束时的共同清理路径，避免无合法动作的残留选择状态；3983卡ID顺序、245动作和观察契约保留。采样观察使用每次新分配的bytearray所有者，Rust编码直接写入，公开observe接口保持原样；定种子Rust/Python测试核对字节、状态及所有权。

服务器的Goal/进程总控、路径绑定的准入脚本和其部署上下文测试保留本地，不作为通用客户端/库API发布；完整部署源码快照随本地交付保存。历史服务器全套Python曾为693通过/17既有上下文失败/4跳过，17项为13个bare-import CPU门禁和4个Main/runtime路径守卫，未声称它们已修复。这里对实际提交的整个测试目录、安装train依赖后重新验证；CI另外显式跑locked训练依赖测试，避免基础CI跳过训练测试冒充全量。CUDA在无GPU的CI可跳过，本机另有实际CUDA前向/采样/更新检查。

```bash
uv sync --locked --python 3.12 --group train --extra test
uv run --locked --group train --extra test python -m pytest -q
cargo fmt --all -- --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
uv run --locked --group train python -m taverngym.learning.experiment --help
```

本轮使用本地/offline七分区schema2，不上传Eric的W&B组织。没有修改Mac或删除旧模型、配置与日志。
