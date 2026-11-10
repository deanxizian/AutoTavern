"""Independent sealed-trace recount; no report helpers, inference, or simulation."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import math

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / 'pyproject.toml').is_file())
MODELS = ('v1', 'v2', 'v3', 'v5')


def key(row):
    return row['lineup_id'], row['lane'], row['seat'], row['model']


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def check_dist(values, target):
    assert len(values) == target['n']
    close(statistics.mean(values), target['mean'])
    assert min(values) == target['min'] and max(values) == target['max']
    quantiles = (statistics.quantiles(values, n=100, method='inclusive')
                 if len(values) > 1 else [values[0]] * 99)
    for p in (50,90,95,99):
        close(quantiles[p-1], target[f'p{p}'])


def main():
    config = json.loads((FOLDER / 'config.json').read_text())
    result = json.loads((FOLDER / 'results.json').read_text())
    games = [json.loads(l) for l in (FOLDER / 'games.jsonl').read_text().splitlines()]
    players = [json.loads(l) for l in (FOLDER / 'players.jsonl').read_text().splitlines()]
    assert result['config'] == config and result['status'] == 'completed'
    assert len(games) == 320 and len(players) == 2048
    assert len({(g['lineup_id'],g['lane']) for g in games}) == 320
    assert Counter(g['lineup_id'] for g in games) == dict.fromkeys(range(10),32)
    player_map = {key(p):p for p in players}
    assert len(player_map) == len(players)
    for g in games:
        lid, lane = g['lineup_id'], g['lane']
        assert g['lineup'] == config['lineups'][lid] and g['episode_index'] == 0
        assert 0 <= lane < 32
        assert sorted(sum(g['placements'].values(),[])) == list(range(1,9))
        assert sorted(sum(g['model_seats'].values(),[]) + g['v0_seats']) == list(range(8))
        for n in ('v0',)+MODELS:
            assert len(g['placements'][n]) == g['lineup'][n]
        for n in MODELS:
            for seat, rank in zip(g['model_seats'][n], g['placements'][n], strict=True):
                assert player_map[(lid,lane,seat,n)]['placement'] == rank
    turn_map = defaultdict(list)
    all_turns = []
    trace_actions = 0
    with gzip.open(FOLDER / 'turns.jsonl.gz','rt') as stream:
        for line in stream:
            t = json.loads(line)
            trace = t.pop('trace')
            assert key(t) in player_map
            assert t['actions'] == len(trace)
            seen, last, consecutive = set(), None, 0
            returns, peak, forced_ends = 0, 0, 0
            for i, row in enumerate(trace):
                rnd, action, before, after, after_turn, cycle, run, forced, done, onlyend = row
                assert len(before) == 64
                assert (after is None) == done
                if after is not None:
                    assert len(after) == 64
                if i:
                    assert rnd > trace[i-1][0]
                inferred = t['turn'] > 0 and action == 0 and i >= 60 and onlyend
                assert bool(forced) == bool(inferred)
                forced_ends += forced
                if done or action == 0 or after_turn != t['turn']:
                    seen, last, consecutive = set(), None, 0
                    detected = False
                else:
                    if last != before:
                        seen = set()
                        consecutive = 0
                    seen.add(before)
                    detected = after in seen
                    seen.add(after)
                    last = after
                    consecutive = consecutive + 1 if detected else 0
                assert cycle == int(detected) and run == consecutive
                returns += cycle
                peak = max(peak, consecutive)
            assert t['cycle_returns'] == returns
            assert t['max_consecutive_cycle_returns'] == peak
            assert t['forced_ends'] == forced_ends
            assert t['same_action_id_repeats'] == len(trace)-len({a[1] for a in trace})
            assert all(t['turn'] != prior['turn'] for prior in turn_map[key(t)])
            turn_map[key(t)].append(t)
            all_turns.append(t)
            trace_actions += len(trace)
    for pk,p in player_map.items():
        ts = sorted(turn_map[pk], key=lambda t:t['turn'])
        recruit = [t for t in ts if t['turn']>0]
        assert p['recruit_turn_numbers'] == [t['turn'] for t in recruit]
        assert p['recruit_turns'] == len(recruit)
        assert p['actions'] == sum(t['actions'] for t in ts)
        assert p['hero_actions'] == sum(t['actions'] for t in ts if t['turn']==0)
        assert p['recruit_actions'] == sum(t['actions'] for t in recruit)
        close(p['mean_actions_per_turn'],p['recruit_actions']/len(recruit))
        assert p['cycle_returns'] == sum(t['cycle_returns'] for t in recruit)
        close(p['cycle_return_fraction'],p['cycle_returns']/max(p['recruit_actions'],1))
        assert p['max_consecutive_cycle_returns'] == max(t['max_consecutive_cycle_returns'] for t in recruit)
        assert p['forced_ends'] == sum(t['forced_ends'] for t in recruit)
    for n in ('v0',)+MODELS:
        ranks=[r for g in games for r in g['placements'][n]]
        target=result['groups'][n]
        assert len(ranks)==target['player_games']==512
        close(statistics.mean(ranks),target['mean_placement'])
        close(ranks.count(1)/512,target['first_rate'])
        close(sum(r<=4 for r in ranks)/512,target['top4_rate'])
    for n in MODELS:
        ps=[p for p in players if p['model']==n]
        ts=[t for t in all_turns if t['model']==n and t['turn']>0]
        b=result['behavior'][n]
        assert len(ts)==b['player_turns'] and len(ps)==512
        close(sum(t['actions'] for t in ts)/len(ts),b['pooled_actions_per_turn'])
        assert sum(t['actions'] for t in ts)==b['recruit_actions']
        for values, name in (([t['actions'] for t in ts],'player_turn_action_distribution'),
            ([p['mean_actions_per_turn'] for p in ps],'player_mean_actions_per_turn_distribution'),
            ([p['recruit_turns'] for p in ps],'recruit_turns_per_player_distribution'),
            ([p['actions'] for p in ps],'actions_per_player_game_distribution'),
            ([p['cycle_returns'] for p in ps],'cycle_returns_per_player_distribution'),
            ([p['cycle_return_fraction'] for p in ps],'player_cycle_return_fraction_distribution')):
            check_dist(values,b[name])
        assert b['cycle_returns']==sum(p['cycle_returns'] for p in ps)
        close(b['pooled_cycle_return_fraction'],b['cycle_returns']/b['recruit_actions'])
        assert b['zero_cycle_player_games']==sum(p['cycle_returns']==0 for p in ps)
        close(b['zero_cycle_player_game_fraction'],b['zero_cycle_player_games']/512)
        assert b['max_consecutive_cycle_returns']==max(p['max_consecutive_cycle_returns'] for p in ps)
        assert b['forced_ends']==sum(t['forced_ends'] for t in ts)
        close(b['forced_end_turn_fraction'],sum(t['forced_ends']>0 for t in ts)/len(ts))
        assert b['observed_zero_action_turns']==sum(t['actions']==0 for t in ts)
        assert b['same_action_id_repeats_within_turn']==sum(t['same_action_id_repeats'] for t in ts)
        assert b['hero_actions']==sum(p['hero_actions'] for p in ps)
        for turn,target in result['by_turn'][n].items():
            check_dist([t['actions'] for t in ts if t['turn']==int(turn)],target)
    batches=json.loads((FOLDER/'batches.json').read_text())
    predicted_total = 0
    for n,bs in batches.items():
        assert len({r[0] for r in bs})==len(bs), 'more than one call/model/global round'
        assert bs[0][1:]==[512,2], 'all initial model seats must join merged call'
        assert sum(r[1] for r in bs)>=sum(p['actions'] for p in players if p['model']==n)
        predicted_total += sum(r[1] for r in bs)
        assert sum(r[2]==2 for r in bs)==result['cross_shard_calls'][n]
        assert Counter(r[1] for r in bs)=={int(k):v for k,v in result['batch_distributions'][n].items()}
    assert predicted_total == trace_actions + result['deferred_actions']
    initial=json.loads((FOLDER/'schedule-initial.json').read_text())
    assert initial['actual_initial_concurrency']==result['actual_initial_concurrency']==320
    assert [(s['roles'],s['games']) for s in initial['shards']]==[(6,192),(7,128)]
    assert result['training_started'] is False and result['optimizer_updates']==0
    assert result['rejected_actions']==0
    for spec in config['models'].values():
        assert hashlib.sha256((ROOT/spec['path']).read_bytes()).hexdigest()==spec['sha256']
    files=['config.json','run.py','report.py','preflight.py','games.jsonl','players.jsonl',
           'turns.jsonl.gz','results.json','batches.json','schedule-initial.json']
    report={'status':'passed','completed_lobbies':320,'player_games_per_version':512,
            'model_player_games':len(players),'player_turn_records_including_hero':len(all_turns),
            'executed_actions_recounted':trace_actions,'exact_ranks_and_quotas':True,
            'turn_counts_and_both_weighting_methods_independently_recounted':True,
            'state_hash_cycles_independently_recounted':True,'source_weights_unchanged':True,
            'actual_initial_concurrency':320,'cross_shard_merged_gpu_batches_verified':True,
            'file_sha256':{f:hashlib.sha256((FOLDER/f).read_bytes()).hexdigest() for f in files}}
    (FOLDER/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
