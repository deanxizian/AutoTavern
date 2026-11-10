"""Synthetic recording checks and two unfinished native lanes (five global rounds)."""
import io
import json
import numpy as np
from run import FOLDER, Recorder, make_shards, merged_decisions, advance_shard
from report import MODELS


def obs(turn, state=0, end_only=False):
    f = np.zeros(16, dtype=np.float32)
    f[0], f[1] = turn / 20, state
    return {'features': f, 'cards': np.zeros(3, dtype=np.int32),
            'action_mask': np.array([True, not end_only, not end_only])}


class LegalFirst:
    def __init__(self):
        self.calls = []

    def predict(self, inputs, deterministic=False):
        self.calls.append(len(inputs['features']))
        return np.argmax(inputs['action_mask'], axis=1), None


def main():
    config = json.loads((FOLDER / 'config.json').read_text())
    stream = io.StringIO()
    r = Recorder({'model': 'v1'}, stream, config['diagnostics'])
    r.observe_ready(0)
    r.execute(0, 1, obs(0), obs(1), False)
    r.observe_ready(1)
    r.execute(1, 1, obs(1), obs(1, 1), False)
    r.execute(2, 1, obs(1, 1), obs(1), False)
    assert r.current['cycle_returns'] == 1
    r.execute(3, 0, obs(1), obs(2), False)
    r.observe_ready(2)
    # Observed alive/ready, but deferred and then eliminated: retain zero-action turn.
    row = r.finish(8)
    assert row['actions'] == 4 and row['hero_actions'] == 1
    assert row['recruit_actions'] == 3 and row['recruit_turns'] == 2
    assert row['mean_actions_per_turn'] == 1.5 and row['cycle_returns'] == 1
    turns = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert turns[-1]['actions'] == 0 and turns[-1]['turn'] == 2
    cap = Recorder({'model':'v5'}, io.StringIO(), config['diagnostics'])
    cap.observe_ready(1)
    for i in range(60):
        cap.execute(i, 1, obs(1, i), obs(1, i+1), False)
    terminal = cap.execute(60, 0, obs(1, 60, True), obs(2), True)
    assert cap.current['forced_ends'] == 1
    assert cap.finish(1, terminal)['recruit_actions'] == 61
    config['lineups'] = [config['lineups'][0], config['lineups'][6]]
    turn_output, game_output, player_output = io.StringIO(), io.StringIO(), io.StringIO()
    shards = make_shards(config, turn_output, games_per_lineup=1)
    assert [(s['roles'], s['env'].n_games) for s in shards] == [(6, 1), (7, 1)]
    policies = {n: LegalFirst() for n in MODELS}
    stats = {n: [] for n in MODELS}
    games, players = [], []
    executed = 0
    for step in range(5):
        merged_decisions(shards, policies, stats, step)
        for s in shards:
            _, deferred = advance_shard(s, step, game_output, player_output,
                                        games, players, config)
            assert deferred == 0
            executed += int(s['ready'].sum())
    assert not games and not players, 'Preflight must not complete extra matches'
    for s in shards:
        assert s['env'].episode_index(0) == 0
        for r in s['records']:
            r.flush()
    ts = [json.loads(line) for line in turn_output.getvalue().splitlines()]
    assert sum(t['actions'] for t in ts) == executed
    assert sum(t['actions'] for t in ts if t['turn'] == 0) == 13
    assert any(t['turn'] > 0 for t in ts)
    assert all(all(call[2] == 2 for call in stats[n]) for n in MODELS)
    assert all(len(policies[n].calls) == 5 for n in MODELS)
    report = {'status': 'passed', 'synthetic': ['hero/recruit split', 'A-B-A return',
              'End resets history', 'zero-action observed turn retained',
              '60 prior actions + End under single-End mask counted as forced',
              'terminal action included, reset observation excluded'],
              'native_preflight': {'lanes': 2, 'roles': [6,7], 'global_rounds':5,
                  'completed_matches':0, 'executed_actions':executed,
                  'cross_shard_merged_calls':stats}, 'training_started':False}
    (FOLDER / 'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
