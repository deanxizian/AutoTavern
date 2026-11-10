"""Bounded frozen match: 320 lanes, two shards, cross-shard GPU batches, turn traces."""
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import taverngym as tg
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.rewards import ActionRewardShaper, RewardShapingConfig
from taverngym.learning.tracking import WandbTracker
from report import GROUPS, MODELS, summarize

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / 'pyproject.toml').is_file())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(stream, row):
    stream.write(json.dumps(row, separators=(',', ':')) + '\n')


class Recorder:
    def __init__(self, identity, stream, config):
        self.identity, self.stream = identity, stream
        self.shaper = ActionRewardShaper(RewardShapingConfig(**config))
        self.current = None
        self.turns = []
        self.finished = False

    def observe_ready(self, turn):
        assert not self.finished
        if self.current is None or self.current['turn'] != turn:
            if self.current is not None:
                assert turn > self.current['turn'], 'turn regressed or next episode leaked'
                self.flush()
            self.current = {**self.identity, 'turn': turn, 'actions': 0, 'cycle_returns': 0,
                            'max_consecutive_cycle_returns': 0, 'forced_ends': 0,
                            'same_action_id_repeats': 0, 'trace': []}

    def execute(self, global_round, action, before, after, done):
        turn = int(round(float(before['features'][0]) * 20))
        assert self.current is not None and self.current['turn'] == turn
        row = self.current
        forced = (turn > 0 and action == 0 and row['actions'] >= 60
                  and int(before['action_mask'].sum()) == 1)
        old_returns = self.shaper._totals['cycle_returns']
        _, metrics = self.shaper.step(action, before, after, 0, done=done)
        totals = metrics if metrics is not None else self.shaper._totals
        cycle = int(totals['cycle_returns'] - old_returns)
        consecutive = 0 if done else self.shaper.consecutive_returns
        row['actions'] += 1
        row['cycle_returns'] += cycle
        row['forced_ends'] += int(forced)
        row['max_consecutive_cycle_returns'] = max(row['max_consecutive_cycle_returns'], consecutive)
        row['trace'].append([global_round, action, self.shaper._visible(before).hex(),
                             None if done else self.shaper._visible(after).hex(),
                             None if done else int(round(float(after['features'][0]) * 20)),
                             cycle, consecutive, int(forced), bool(done),
                             bool(before['action_mask'][0] and before['action_mask'].sum() == 1)])
        return metrics

    def flush(self):
        row = self.current
        if row is None:
            return
        row['same_action_id_repeats'] = row['actions'] - len({a[1] for a in row['trace']})
        emit(self.stream, row)
        self.turns.append({k: v for k, v in row.items() if k != 'trace'})
        self.current = None

    def finish(self, placement, terminal_metrics=None):
        assert not self.finished
        self.flush()
        totals = (terminal_metrics if terminal_metrics is not None
                  else self.shaper.finish_terminal(0, placement))
        assert int(totals['actions_per_game']) == sum(t['actions'] for t in self.turns)
        assert int(totals['cycle_returns']) == sum(t['cycle_returns'] for t in self.turns)
        recruit = [t for t in self.turns if t['turn'] > 0]
        count, actions = len(recruit), sum(t['actions'] for t in recruit)
        returns = sum(t['cycle_returns'] for t in recruit)
        self.finished = True
        return {**self.identity, 'placement': placement,
                'actions': sum(t['actions'] for t in self.turns), 'recruit_actions': actions,
                'hero_actions': sum(t['actions'] for t in self.turns if t['turn'] == 0),
                'recruit_turns': count, 'recruit_turn_numbers': [t['turn'] for t in recruit],
                'mean_actions_per_turn': actions / count if count else 0,
                'cycle_returns': returns, 'cycle_return_fraction': returns / max(actions, 1),
                'max_consecutive_cycle_returns': max(
                    (t['max_consecutive_cycle_returns'] for t in recruit), default=0),
                'forced_ends': sum(t['forced_ends'] for t in recruit)}


def make_shards(config, turn_stream, games_per_lineup=32):
    shards = []
    for spec in config['shards']:
        roles = spec['roles']
        mappings, identities = [], []
        for lid, lineup in enumerate(config['lineups']):
            if sum(lineup[n] for n in MODELS) != roles:
                continue
            role_ids = [j for repeat in range(2) for j, name in enumerate(MODELS)
                        if lineup[name] > repeat]
            for lane in range(games_per_lineup):
                mappings.append((lid, lane))
                identities.append(np.roll(role_ids, -lane))
        env = tg.BatchEnv(len(mappings), seed=spec['seed'], controlled=range(roles),
                          threads=config['threads'], config=config['simulator_config'],
                          rotate_seats=True)
        identity = np.array(identities, dtype=np.int64)
        seats = env.slot_seats.reshape(len(mappings), roles)
        records = []
        for i, (lid, lane) in enumerate(mappings):
            for j in range(roles):
                records.append(Recorder({'lineup_id': lid, 'lane': lane, 'shard_roles': roles,
                                         'shard_lane': i, 'seat': int(seats[i, j]),
                                         'model': MODELS[int(identity[i, j])]},
                                        turn_stream, config['diagnostics']))
        shards.append({'env': env, 'roles': roles, 'mapping': mappings, 'identity': identity,
                       'seats': seats, 'records': records,
                       'active': np.ones(len(mappings), dtype=bool),
                       'places': np.zeros_like(identity)})
    return shards


def merged_decisions(shards, policies, batch_stats, global_round):
    for s in shards:
        s['obs'] = s['env'].observe()
        s['ready'] = s['obs']['ready'] & np.repeat(s['active'], s['roles'])
        s['actions'] = np.zeros(s['env'].n_slots, dtype=np.int64)
        for slot in np.flatnonzero(s['ready']):
            turn = int(round(float(s['obs']['features'][slot, 0]) * 20))
            s['records'][slot].observe_ready(turn)
    for j, name in enumerate(MODELS):
        chunks = [(s, np.flatnonzero(s['ready'] & (s['identity'].reshape(-1) == j)))
                  for s in shards]
        chunks = [(s, ids) for s, ids in chunks if len(ids)]
        if not chunks:
            continue
        inputs = {k: np.concatenate([s['obs'][k][ids] for s, ids in chunks])
                  for k in ('features', 'cards', 'action_mask')}
        with torch.inference_mode():
            predicted, _ = policies[name].predict(inputs, deterministic=False)
        predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
        assert inputs['action_mask'][np.arange(len(predicted)), predicted].all()
        offset = 0
        for s, ids in chunks:
            s['actions'][ids] = predicted[offset:offset + len(ids)]
            offset += len(ids)
        assert offset == len(predicted)
        batch_stats[name].append([global_round, len(predicted), len(chunks)])


def advance_shard(s, global_round, game_stream, player_stream, games, players, config):
    started = time.monotonic()
    env, roles = s['env'], s['roles']
    _, dones, info = env.step(s['actions'], active=s['active'])
    env_seconds = time.monotonic() - started
    if info['rejected'].any() or ((info['placement'] == 0) & dones).any():
        for lane in np.unique(np.flatnonzero(info['rejected'] | dones) // roles):
            write_json(FOLDER / f'guard-{roles}-{lane}.json', json.loads(env.save(int(lane))))
        raise RuntimeError('Native rejection or terminal without a real placement')
    after = env.observe()
    terminal_metrics = {}
    for slot in np.flatnonzero(s['ready'] & ~info['deferred'] & ~info['rejected']):
        before_one = {k: s['obs'][k][slot] for k in ('features', 'cards', 'action_mask')}
        after_one = {k: after[k][slot] for k in ('features', 'cards', 'action_mask')}
        terminal_metrics[slot] = s['records'][slot].execute(
            global_round, int(s['actions'][slot]), before_one, after_one, bool(dones[slot]))
    for slot in np.flatnonzero(dones):
        i, j = divmod(int(slot), roles)
        assert s['active'][i] and s['places'][i, j] == 0
        placement = int(info['placement'][slot])
        s['places'][i, j] = placement
        row = s['records'][slot].finish(placement, terminal_metrics.get(slot))
        emit(player_stream, row)
        players.append(row)
    for i in np.flatnonzero(s['active'] & (s['places'] > 0).all(axis=1)):
        lid, lane = s['mapping'][i]
        places = s['places'][i]
        identity, seats = s['identity'][i], s['seats'][i]
        row = {'lineup_id': lid, 'lane': lane, 'shard_roles': roles, 'shard_lane': int(i),
               'episode_index': 0, 'lineup': config['lineups'][lid],
               'placements': {n: places[identity == j].tolist() for j, n in enumerate(MODELS)},
               'model_seats': {n: seats[identity == j].tolist() for j, n in enumerate(MODELS)},
               'v0_seats': sorted(set(range(8)) - set(seats.tolist()))}
        row['placements']['v0'] = sorted(set(range(1, 9)) - set(places.tolist()))
        assert sorted(sum(row['placements'].values(), [])) == list(range(1, 9))
        assert env.episode_index(int(i)) == 1
        emit(game_stream, row)
        games.append(row)
        s['active'][i] = False
    return env_seconds, int(info['deferred'].sum())


def main():
    config = json.loads((FOLDER / 'config.json').read_text())
    assert not (FOLDER / 'experiment.json').exists(), 'Refusing to rerun an existing official attempt'
    assert not (FOLDER / 'games.jsonl').exists()
    assert config['games'] == config['parallel_games'] == 320
    assert json.loads((FOLDER / 'preflight.json').read_text())['status'] == 'passed'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(config['seed'])
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert tg.build_is_current()
    models = {}
    for name, spec in config['models'].items():
        path = ROOT / spec['path']
        assert digest(path) == spec['sha256']
        models[name] = load_frozen_policy(path, 'cuda', forward_precision='bf16')
        assert {str(p.dtype) for p in models[name].parameters()} == {'torch.float32'}
        assert not any(p.requires_grad for p in models[name].parameters())
    torch.cuda.reset_peak_memory_stats()
    metadata = {'status': 'running', 'started_at': now(), 'training_started': False,
                'optimizer_updates': 0, 'models': config['models'], 'native_build': tg.build_info(),
                'requested_concurrency': 320, 'actual_initial_concurrency': 320}
    write_json(FOLDER / 'experiment.json', metadata)
    tracker = None
    started, last_report = time.monotonic(), time.monotonic()
    games, players = [], []
    batches = {name: [] for name in MODELS}
    inference_seconds = env_seconds = 0
    deferred = 0
    exit_code = 1
    try:
        try:
            tracker = WandbTracker(FOLDER, config, kind='frozen-all-lanes-turn-diagnostics')
            tracker.run.define_metric('evaluated_lobbies')
            tracker.run.define_metric('test/*', step_metric='evaluated_lobbies')
            tracker.run.define_metric('diagnostics/*', step_metric='evaluated_lobbies')
            meta = json.loads((FOLDER / 'wandb.json').read_text())
            meta['x_axis'] = 'evaluated_lobbies'
            write_json(FOLDER / 'wandb.json', meta)
        except Exception as exc:
            write_json(FOLDER / 'tracking-error.json', {'error': repr(exc), 'match_continues': True})
        with ExitStack() as stack:
            gs = stack.enter_context((FOLDER / 'games.jsonl').open('x', buffering=1))
            ps = stack.enter_context((FOLDER / 'players.jsonl').open('x', buffering=1))
            ts = stack.enter_context(gzip.open(FOLDER / 'turns.jsonl.gz', 'xt', compresslevel=1))
            shards = make_shards(config, ts)
            assert [(s['roles'], s['env'].n_games) for s in shards] == [(6, 192), (7, 128)]
            initial = sum(int(s['active'].sum()) for s in shards)
            assert initial == 320
            write_json(FOLDER / 'schedule-initial.json', {
                'actual_initial_concurrency': initial, 'shards': [
                    {'roles': s['roles'], 'games': s['env'].n_games, 'mapping': s['mapping']}
                    for s in shards]})
            for global_round in range(config['max_global_rounds']):
                if not any(s['active'].any() for s in shards):
                    break
                if time.monotonic() - started > config['max_match_seconds']:
                    raise TimeoutError('Bounded match wall time exhausted; partial records retained')
                t = time.monotonic()
                merged_decisions(shards, models, batches, global_round)
                torch.cuda.synchronize()
                inference_seconds += time.monotonic() - t
                for s in shards:
                    if not s['active'].any():
                        continue
                    duration, held = advance_shard(s, global_round, gs, ps, games, players, config)
                    env_seconds += duration
                    deferred += held
                if time.monotonic() - last_report > 15:
                    progress = {'status': 'running', 'global_round': global_round,
                                'completed_lobbies': len(games), 'elapsed_seconds': time.monotonic()-started,
                                'active_lanes': sum(int(s['active'].sum()) for s in shards)}
                    write_json(FOLDER / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    if tracker:
                        try:
                            tracker.log({'evaluated_lobbies': len(games)}, 0)
                        except Exception as exc:
                            write_json(FOLDER / 'tracking-error.json', {'error': repr(exc)})
                    last_report = time.monotonic()
            else:
                raise RuntimeError('Bounded global-round budget exceeded')
            assert len(games) == 320 and len(players) == 2048
            assert all(r.finished for s in shards for r in s['records'])
            turns = [t for s in shards for r in s['records'] for t in r.turns]
        assert all(digest(ROOT / v['path']) == v['sha256'] for v in config['models'].values())
        elapsed = time.monotonic() - started
        results = {'status': 'completed', 'completed_at': now(), 'config': config,
                   **summarize(games, players, turns), 'completed_lobbies': len(games),
                   'elapsed_seconds': elapsed, 'games_per_second': len(games) / elapsed,
                   'inference_and_observe_seconds': inference_seconds,
                   'native_step_seconds': env_seconds, 'global_rounds': global_round,
                   'actual_initial_concurrency': initial, 'microbatch': None,
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'batch_distributions': {n: dict(Counter(row[1] for row in bs)) for n, bs in batches.items()},
                   'cross_shard_calls': {n: sum(row[2] == 2 for row in bs) for n, bs in batches.items()},
                   'rejected_actions': 0, 'deferred_actions': deferred, 'weights_unchanged': True,
                   'training_started': False, 'optimizer_updates': 0,
                   'limitations': ['V0 behavioral metrics unknown; only rank complement is available.',
                                   'Fresh match with same frozen models; not a replay of historical 320 games.',
                                   'Turn denominator includes every recruit turn observed ready, even if zero executed actions.',
                                   'Within-batch other-seat changes enter post-action public observations.',
                                   'Hashed traces support independent cycle/count audits, not full state rendering.']}
        write_json(FOLDER / 'batches.json', batches)
        write_json(FOLDER / 'results.json', results)
        metadata.update(status='completed', completed_at=now(), completed_lobbies=len(games),
                        elapsed_seconds=elapsed)
        write_json(FOLDER / 'experiment.json', metadata)
        if tracker:
            payload = {'evaluated_lobbies': len(games)}
            for n, g in results['groups'].items():
                payload.update({f'test/{n}_{k}': g[k] for k in ('first_rate','top4_rate','mean_placement')})
            for n in MODELS:
                b = results['behavior'][n]
                payload.update({f'diagnostics/{n}_{k}': b[k] for k in (
                    'pooled_actions_per_turn','cycle_returns','pooled_cycle_return_fraction',
                    'zero_cycle_player_game_fraction','max_consecutive_cycle_returns','forced_ends')})
                for metric in ('mean','p50','p90','p95','p99'):
                    payload[f'diagnostics/{n}_turn_actions_{metric}'] = b['player_turn_action_distribution'][metric]
            try:
                tracker.log(payload, 0)
            except Exception as exc:
                write_json(FOLDER / 'tracking-error.json', {'error': repr(exc)})
        print(json.dumps(results['behavior']), flush=True)
        exit_code = 0
    except BaseException as exc:
        metadata.update(status='failed', failed_at=now(), error=repr(exc), completed_lobbies=len(games))
        write_json(FOLDER / 'experiment.json', metadata)
        raise
    finally:
        if tracker:
            try:
                tracker.finish(exit_code)
            except Exception as exc:
                write_json(FOLDER / 'tracking-error.json', {'error': repr(exc)})


if __name__ == '__main__':
    main()
