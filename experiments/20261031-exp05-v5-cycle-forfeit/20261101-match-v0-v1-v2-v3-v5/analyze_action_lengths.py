"""Describe saved player action totals; do not infer missing cycles or turn lengths."""

import hashlib
import json
from pathlib import Path

import numpy as np


FOLDER = Path(__file__).resolve().parent
GROUPS = ("v1", "v2", "v3", "v5")
PERCENTILES = (50, 90, 95, 99)


def main():
    source = FOLDER / "games.jsonl"
    data = source.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 320
    assert len({(r["lineup_id"], r["lane"], r["episode_index"]) for r in rows}) == 320
    result = {
        "source": "games.jsonl",
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "lobbies": 320,
        "protocol": "Eight-player lobbies; V0/V1/V2/V3/V5 each 1-2 seats in ten balanced lineups; "
                    "32 lobbies per lineup; stochastic BF16 neural inference; frozen weights.",
        "unit": "One model seat from episode start to its terminal placement (player-game).",
        "definition": "model_actions counts ready actions excluding native deferred actions; "
                      "native rejected actions were zero. Includes hero/menu/End decisions; "
                      "excludes automatic combat and native V0 actions.",
        "quantile_method": "numpy.percentile(method='linear'); sorted index (n-1)*p; "
                           "fractional percentiles are interpolated action counts.",
        "groups": {},
        "unavailable": {
            "v0_action_counts": "V0 is advanced internally by native code; no V0 action counters saved.",
            "cycle_returns": "No per-transition observations, visible-state hashes or cycle counters saved.",
            "zero_cycle_player_games": "Cannot identify player-games without repeated visible states.",
            "actions_per_turn": "No per-turn action counts or turn boundaries saved.",
            "cap_end_rate": "No per-turn action masks or forced-End events saved.",
            "action_id_repetition": "No ordered action IDs saved; distinct from visible-state cycles.",
        },
        "limitations": [
            "Shorter total action counts do not establish fewer cycles or zero repetition.",
            "Total actions also depend on survival duration; no per-turn normalization is possible here.",
            "This is post-hoc analysis of existing records, not a new match or training run.",
        ],
    }
    for name in GROUPS:
        for row in rows:
            assert len(row["model_actions"][name]) == len(row["placements"][name])
        values = np.array([v for row in rows for v in row["model_actions"][name]], dtype=np.int64)
        assert len(values) == 512 and (values >= 0).all()
        quantiles = np.percentile(values, PERCENTILES, method="linear")
        result["groups"][name] = {
            "player_games": int(values.size), "total_actions": int(values.sum()),
            "mean": float(values.mean()), "min": int(values.min()), "max": int(values.max()),
            **{f"p{p}": float(q) for p, q in zip(PERCENTILES, quantiles)},
        }
    result["v5_mean_action_reduction_vs_v3"] = (
        1 - result["groups"]["v5"]["mean"] / result["groups"]["v3"]["mean"])
    (FOLDER / "action-length-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 已保存 320 局的动作总数分析", "",
        "八人局，V0／V1／V2／V3／V5 每版各 1–2 席，十种平衡阵容各 32 局；",
        "V1／V2／V3／V5 随机采样、BF16、冻结模型。每版 512 次玩家出场。", "",
        "原记录并非逐步轨迹：保存了终局名次与各模型每次出场的动作总数，没有动作序列、状态或回合边界。",
        "因此只能计算**每个模型玩家每局的总动作数**，无法从本轮还原重复次数、无重复局数、每回合步数或触顶率。", "",
        "| 版本 | 样本数 | 均值 | P50 | P90 | P95 | P99 | 最小 | 最大 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| V0（脚本） | 未记录动作 | — | — | — | — | — | — | — |",
    ]
    for name, group in result["groups"].items():
        columns = [name.upper(), str(group["player_games"])]
        columns += [f"{group[key]:.1f}" for key in ("mean", "p50", "p90", "p95", "p99")]
        columns += [str(group["min"]), str(group["max"])]
        lines.append("| " + " | ".join(columns) + " |")
    reduction = result["v5_mean_action_reduction_vs_v3"]
    lines += [
        "", f"V5 的每局平均动作总数比 V3 少 {reduction:.1%}。但总动作数受存活回合数影响，不能据此断言重复减少或没有重复。",
        "", "## 口径与缺失", "",
        "- 计数含英雄选择、菜单、End 等模型决策；不含自动战斗或 V0 内部脚本动作。已排除 deferred，本轮 rejected=0。",
        "- 分位数使用线性插值，位置为 `(n-1)*p`；小数来自插值，并非单次真实动作数为小数。",
        "- P90 反映约 90% 样本的动作总数落在该值以内；不是第 90 回合，也不是 PPO 步数。",
        "- 重复按训练中同一招募回合的完整可见状态返回定义；动作 ID 重复不一定是无进展循环。",
        "- 未记录不等于零，无法列出本轮哪些模型从未重复。此前训练的循环均值属于不同阵容和数据，不能补作本轮观测。",
        "- 要计算每回合分位数及无重复比例，下一次测评须记录逐回合动作计数、状态返回次数和连续返回峰值；完整轨迹可记录动作及前后可见状态哈希。",
        "", "来源：[games.jsonl](games.jsonl)，机器可读结果：[action-length-analysis.json](action-length-analysis.json)。",
        f"原记录 SHA-256：`{result['source_sha256']}`。本分析未重跑比赛、未改变封存结果。", "",
    ]
    (FOLDER / "action-length-analysis.md").write_text("\n".join(lines))
    assert source.read_bytes() == data
    print(json.dumps(result["groups"]))


if __name__ == "__main__":
    main()
