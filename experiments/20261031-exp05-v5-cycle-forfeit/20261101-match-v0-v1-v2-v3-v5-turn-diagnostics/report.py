"""Sealed-record summaries. No model loading, training, or simulation."""
from collections import Counter, defaultdict
import math
import statistics

GROUPS = ('v0', 'v1', 'v2', 'v3', 'v5')
MODELS = GROUPS[1:]


def distribution(values):
    a = sorted(values)
    def quantile(p):
        index = (len(a) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return a[low] + (a[high] - a[low]) * (index - low)
    return {'n': len(a), 'mean': statistics.mean(a), 'min': min(a), 'max': max(a),
            **{f'p{int(p * 100)}': quantile(p) for p in (.5, .9, .95, .99)}}


def summarize(games, players, turns):
    out = {'groups': {}, 'behavior': {}, 'by_turn': {}}
    for name in GROUPS:
        ranks = [v for row in games for v in row['placements'][name]]
        out['groups'][name] = {
            'player_games': len(ranks), 'first_rate': ranks.count(1) / len(ranks),
            'top4_rate': sum(x <= 4 for x in ranks) / len(ranks),
            'mean_placement': statistics.mean(ranks),
            'placement_counts': dict(sorted(Counter(ranks).items()))}
    for name in MODELS:
        ps = [p for p in players if p['model'] == name]
        ts = [t for t in turns if t['model'] == name and t['turn'] > 0]
        total_actions, total_returns = sum(t['actions'] for t in ts), sum(t['cycle_returns'] for t in ts)
        out['behavior'][name] = {
            'player_games': len(ps), 'player_turns': len(ts),
            'recruit_actions': total_actions,
            'pooled_actions_per_turn': total_actions / len(ts),
            'player_turn_action_distribution': distribution([t['actions'] for t in ts]),
            'player_mean_actions_per_turn_distribution': distribution(
                [p['mean_actions_per_turn'] for p in ps]),
            'recruit_turns_per_player_distribution': distribution([p['recruit_turns'] for p in ps]),
            'actions_per_player_game_distribution': distribution([p['actions'] for p in ps]),
            'cycle_returns': total_returns,
            'cycle_returns_per_player_distribution': distribution([p['cycle_returns'] for p in ps]),
            'pooled_cycle_return_fraction': total_returns / max(total_actions, 1),
            'player_cycle_return_fraction_distribution': distribution(
                [p['cycle_return_fraction'] for p in ps]),
            'zero_cycle_player_games': sum(p['cycle_returns'] == 0 for p in ps),
            'zero_cycle_player_game_fraction': sum(p['cycle_returns'] == 0 for p in ps) / len(ps),
            'max_consecutive_cycle_returns': max(p['max_consecutive_cycle_returns'] for p in ps),
            'forced_ends': sum(p['forced_ends'] for p in ps),
            'forced_end_turn_fraction': sum(t['forced_ends'] > 0 for t in ts) / len(ts),
            'observed_zero_action_turns': sum(t['actions'] == 0 for t in ts),
            'same_action_id_repeats_within_turn': sum(t['same_action_id_repeats'] for t in ts),
            'hero_actions': sum(p['hero_actions'] for p in ps),
        }
        out['by_turn'][name] = {
            str(turn): distribution([t['actions'] for t in ts if t['turn'] == turn])
            for turn in sorted({t['turn'] for t in ts})}
    out['behavior']['v0'] = {'status': 'unknown; native script actions are not exposed to Python'}
    return out
