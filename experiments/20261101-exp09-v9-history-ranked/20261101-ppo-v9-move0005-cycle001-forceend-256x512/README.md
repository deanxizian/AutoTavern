# V9 256×512 前期操作与开局奖励准备

2026-11-01：用户要求准备Reward改进；本实例仍处于 `paused_pending_reward_design`，禁止启动或自动恢复。
训练、开发排名与线上run均未执行，当前没有本版checkpoint或best。

## 奖励

- 每次实际成功的非End操作：第1/2回合扣0.001，第3/4/5回合扣0.0005；第6回合起不扣，无免费次数或超额阈值。
- 本项每回合最多扣0.01，五回合最多0.05；同回合跨rollout保留费用台账。
- 每局每项只奖一次：第1回合己方board从空变非空（含技能召唤、购买不算）+0.01；第2回合实际Upgrade从tier1到2 +0.005；第3回合完成本次付费DarkGift对应Discover选择 +0.005。
- DarkGift按钮本身不奖；匹配uses序号、pending_source和实际Choose。三个奖励总和最多0.02。
- 统计全部成功操作，包含主动/强制End与菜单；费用排除End。英雄选择turn0、deferred、rejected、未执行预算拦截不计；真实动作后终局仍收成本。
- Move独立扣0.0005，循环返回独立固定扣0.01；早期费用可叠加。循环总罚仍线性无上限；不改60步强制End，两项forfeit=null。
- 不新增刷新/空场费用，不强制动作或开局；收益未经训练验证，可能诱导过早End，仍需既定val观察。

## 来源与未执行预算

来源 `V8` 所选best完整PPO/Adam；V7直接取V5 best完整Adam7200。
固定LR=1e-5；256环境×512步，每PPO131072样本，计划100 PPO、最多1600新Adam；均保持未执行。
冻结对手由统一开发排名取前三版各2席加1 V0，V4/V6及未来废弃别名不入新池。
val仍为PPO0/50/100随机/贪心各64局，只有随机mean_placement严格降低替换best，起点参加；独立test不选模。

新实例 `20261101-v789-move0005-256x512-01`；旧V1–V6、旧V7准备和旧停止实例不改。
修改前快照保存在实例 `preparation-history/20261101-112007-before-early-action-reward/`。
本轮只做CPU/原生正确性检查，禁止真实Adam、训练、排名和线上日志。

验证：228项相关CPU／原生检查通过，排除2项真实Adam更新用例；全局Adam.step失败守卫记录更新尝试0。包括原生DarkGift跨rollout与unified_actor跨collect、GAE、默认关闭轨迹兼容及mock日志。父任务只读复核通过；CUDA未跑。准备验证不代表已训练或已解除启动暂停。

## 仅选英雄偏好追加（2026-11-01，继续暂停）

用户仅授权接入初始选英雄偏好；不接入116英雄操作指南或31流派规则。完整原文保留在docs/references。结构化资料configs/hero-selection-priors-hsreplay-20261029.json仅含原文12英雄的前50%/前1%双分段均名。此稿内嵌前1%均名，source/data路径与SHA、cohort和缺失每英雄样本量的限制另封存，不在运行时重新读分数文件。

只有成功完成初始HeroSelect才可能每局奖励一次：所有实际合法可选英雄均有同分段数值且至少2种不同英雄时，bonus=min(0.005,0.01×(最差可选均名−已选均名))。任一未知、同分、单候选或选最差均不给正奖，不补未知值、不将未知当最差；其它Discover/技能选择不奖。稳定Blizzard ID按catalog解析cid+1并排除教程别名。先验是人类分段偏好，未经模拟器强度验证。原有费用、三个开局奖励与随机val严格降均名选best规则保持。

新增配置接入不是训练启动授权；start=false、pause=true、automatic_resume=false继续生效。

选英雄Reward验证：37项新增检查通过，合并相关CPU／原生265项通过，排除2项真实Adam日志用例；全局Adam.step失败守卫记录更新尝试0，CUDA未跑，无线上请求／run。父任务独立核对12英雄两个分段原始数值、ID、公式、来源SHA、三版嵌入配置与暂停状态通过。准备验证不代表已经训练或确认收益。

## 操作奖励结构合并（数值和协议不变，仍暂停）

操作正负项统一由_action_step计算，action_reward=旧联合费+循环费+移动费+前期逐步费+三个操作里程碑奖励。总reward的分解为placement_reward+hero_selection_bonus+action_reward。action_reward是派生小计，不再额外发一次；opening_bonus已包含三个开局奖励，细分指标均保留，旧operation_penalty仍只表示旧联合费。实际加法保留历史浮点顺序，0.01 cap仅限early项，cycle累计仍线性无封顶。所有配置字段/费用/触发不变，config文件SHA也未改。

274项相关CPU／原生检查通过（9项新增分组回归），排除2项真实Adam日志用例；更新尝试0，CUDA未跑。对实际归档旧shaper做固定种子回放，float32奖励与GAE数组逐位一致，原生状态/掩码/名次及旧分项一致；统一actor跨collect与finish路径也等价。训练/排名/线上run未启动，暂停授权不变。
