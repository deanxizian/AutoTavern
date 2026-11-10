"""Fixed strong-pool evaluation with lane pairing and one forward batch per checkpoint.

This is an inference-only scheduler. Placements never enter a policy observation;
all summaries are recomputed after the complete per-lane records are sealed.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file

import taverngym as tg
from .experiment import EVALUATION_METRICS, load_frozen_policy, now, write_json
from .model import EntityActorCritic, ModelConfig
from .rewards import ActionRewardShaper, RewardShapingConfig
from .stage_chain import tensor_digest
from .text import card_manifest, load_text_cache

KEYS = ('features', 'cards', 'action_mask')
DEFAULT_REWARD = {'operation_cost': 0, 'move_cost': .0005, 'cycle_cost': .01,
                  'cycle_detection': 'all_actions', 'cycle_forfeit_repetitions': None,
                  'turn_action_forfeit_limit': None}
ORIGINALS = {
    '7173d82343fcc79742838c72725c582966a3c5b7c84e15014debea292f32de24':
        ('20261031-131146-best_model.zip', 3200),
    'e4a748e933af5b642c50c223502d127ed1742599d841ab7d3838cf26df7ea6b4':
        ('20261031-165718-best_model.zip', 3200),
    'cdb10c8bb08587b52dd2100ba71302920c549561747f47a7eb1e72fa7333f354':
        ('20261101-115350-best_model.zip', 8800),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _name(spec):
    return str(spec.get('name') or spec.get('model_version') or spec['sha256'])


def _splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def native_lane_seed(seed, lane):
    """The native BatchEnv::game_seed(lane, episode_index=0), without rounding uint64."""
    return _splitmix64(_splitmix64(int(seed) ^ _splitmix64(int(lane))))


@contextmanager
def _isolated_rng(seed, device):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = []
    if str(device).startswith('cuda'):
        devices = [torch.device(device).index or 0]
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _inspect_source(root, spec):
    """Accept real trained boundaries and audited stage-zero adaptations, not shape aliases."""
    root, path = Path(root), Path(root) / spec['model']
    _require(_digest(path) == spec['sha256'], 'Sealed evaluation checkpoint SHA changed')
    data, parameters, _ = load_from_zip_file(path, device='cpu')
    current_card_sha = card_manifest()[2]
    _require(data['policy_kwargs'].get('card_data_sha256') == current_card_sha,
             'Evaluation checkpoint has incompatible card provenance')
    _require(tg.build_is_current() and tg.server_rules_version() == '37.0.2',
             'Evaluation native build/rules differ')
    _require((tg.N_CARDS, tg.N_ACTIONS, tg.OBS_FLOATS, tg.OBS_CARDS)
             == (3984, 245, 3227, 124), 'Evaluation native contracts differ')
    with torch.random.fork_rng(devices=[]):
        network = EntityActorCritic(ModelConfig(**data['policy_kwargs']['model_config']))
    names = ['network.' + name for name, _ in network.named_parameters()]
    policy, optimizer = parameters['policy'], parameters['policy.optimizer']
    _require(len(names) == 91 and [key for key in policy if key in names] == names,
             'Checkpoint learned-parameter registration order differs')
    _require(len(optimizer['param_groups']) == 1
             and optimizer['param_groups'][0]['params'] == list(range(91))
             and set(optimizer['state']) == set(range(91)), 'Incomplete/misordered91-parameter Adam')
    clocks = []
    for index, name in enumerate(names):
        value, state = policy[name], optimizer['state'][index]
        _require(value.dtype == torch.float32 and torch.isfinite(value).all(),
                 'Nonfinite/non-FP32 learned weights')
        for field in ('exp_avg', 'exp_avg_sq'):
            moment = state[field]
            _require(moment.shape == value.shape and moment.dtype == torch.float32
                     and torch.isfinite(moment).all(), 'Invalid Adam parameter mapping/moments')
        clock = float(state['step'])
        _require(np.isfinite(clock) and clock == int(clock) and clock > 0, 'Invalid Adam clock')
        clocks.append(int(clock))
    _require(len(set(clocks)) == 1, 'Adam clocks differ across learned parameters')
    template = dict(network.named_buffers())
    for name, value in template.items():
        if name.startswith('action_') or name == 'static_features':
            _require(torch.equal(policy['network.' + name], value),
                     'Action/static buffers do not match the current native engine')
    assets = json.loads((root / 'experiments/20261108-eric-3702-fresh-v1/assets-summary.json')
                        .read_text())
    text_path = Path(assets['text_asset'])
    _require(_digest(text_path) == assets['text_asset_sha256'], 'Current text asset SHA changed')
    vectors, _ = load_text_cache(text_path)
    _require(torch.equal(policy['network.text_features'], vectors)
             and bool(policy['network.text_ready']), 'Stale/unready rule-text vectors')
    _require(all(torch.isfinite(value).all() for value in policy.values()),
             'Nonfinite checkpoint buffer')
    config = data.get('experiment_config', {})
    history = config.get('source_history', {})
    origin = history.get('source_sha256')
    if data['num_timesteps'] == 0:
        _require(config.get('migration_only') is True and origin in ORIGINALS,
                 'Stage-zero evaluation requires an audited migration marker')
        filename, expected_clock = ORIGINALS[origin]
        original = root / 'runs/imported/eric-local-20261108' / filename
        _require(_digest(original) == origin and clocks[0] == expected_clock
                 and data.get('_optimizer_step_offset') == expected_clock
                 and data.get('_n_updates') == 0, 'Stage-zero source/Adam provenance differs')
        _, raw, _ = load_from_zip_file(original, device='cpu')
        _require(all(torch.equal(policy[name], raw['policy'][name]) for name in names),
                 'Adaptation did not preserve original learned parameters')
        _require(tensor_digest(optimizer) == tensor_digest(raw['policy.optimizer']),
                 'Adaptation did not preserve complete source Adam')
    else:
        budget = config.get('decisions_per_iteration')
        _require(budget and data['num_timesteps'] % budget == 0
                 and data['_n_updates'] == data['num_timesteps'] // budget * data['n_epochs'],
                 'Evaluation source is not a fully updated PPO boundary')
    declared = config.get('model_version')
    if declared is not None and spec.get('model_version') is not None:
        _require(declared == spec['model_version'], 'Evaluation model-version binding differs')
    return {'sha256': spec['sha256'], 'path': str(path), 'source_origin_sha256': origin,
            'stage_samples': data['num_timesteps'], 'adam_total': clocks[0],
            'card_data_sha256': current_card_sha, 'policy_sha256': tensor_digest(policy),
            'learned_sha256': tensor_digest({name: policy[name] for name in names}),
            'adam_sha256': tensor_digest(optimizer),
            'adam_moments_sha256': tensor_digest(optimizer['state']),
            'fingerprint_scheme': 'stage_chain.tensor_digest',
            'parameter_mapping_verified': True}


def _one(obs, slot):
    return {key: obs[key][slot] for key in KEYS}


def _actions_from_probabilities(probabilities, greedy, uniforms):
    """Use F(k)>u, so u=0 and exact CDF plateaus cannot select zero-mass actions."""
    _require(torch.isfinite(probabilities).all(), 'Nonfinite evaluation probabilities')
    _require((probabilities >= 0).all(), 'Negative evaluation probabilities')
    cdf = probabilities.double().cumsum(dim=1)
    _require((cdf[:, -1] > 0).all(), 'Empty evaluation action distribution')
    cdf /= cdf[:, -1:].clone()
    cdf[:, -1] = 1.
    draws = torch.as_tensor(uniforms, device=cdf.device).unsqueeze(1)
    _require(torch.isfinite(draws).all() and (draws >= 0).all() and (draws < 1).all(),
             'Categorical uniforms must be finite and in[0,1)')
    sampled = torch.searchsorted(cdf.contiguous(), draws, right=True).squeeze(1)
    modes = torch.as_tensor(greedy, device=cdf.device)
    return torch.where(modes, probabilities.argmax(dim=1), sampled)


def _merged_actions(shards, policies, statistics, inference_micro_batch_size=None, global_round=0):
    """One forward per SHA; mode and common-random uniforms belong to individual roles."""
    for shard in shards:
        shard['obs'] = shard['env'].observe()
        shard['ready'] = shard['obs']['ready'] & np.repeat(shard['active'], 7)
        shard['actions'] = np.zeros(shard['env'].n_slots, dtype=np.int64)
    for identity, policy in policies.items():
        chunks = [(shard, np.flatnonzero(shard['ready'] & (shard['identities'] == identity)))
                  for shard in shards]
        chunks = [(shard, slots) for shard, slots in chunks if len(slots)]
        if not chunks:
            continue
        obs = {key: np.concatenate([shard['obs'][key][slots] for shard, slots in chunks])
               for key in KEYS}
        greedy = np.concatenate([shard['deterministic'][slots] for shard, slots in chunks])
        uniforms = np.array([.5 if shard['deterministic'][slot]
                             else shard['rng'][slot].random()
                             for shard, slots in chunks for slot in slots], dtype=np.float64)
        count = len(greedy)
        micro = inference_micro_batch_size or count
        actions, forward_sizes = [], []
        for begin in range(0, count, micro):
            end = min(begin + micro, count)
            with torch.inference_mode():
                tensors, _ = policy.obs_to_tensor({key: value[begin:end]
                                                   for key, value in obs.items()})
                probabilities = policy.get_distribution(tensors).distribution.probs
                actions.append(_actions_from_probabilities(
                    probabilities, greedy[begin:end], uniforms[begin:end]).cpu().numpy())
            forward_sizes.append(end - begin)
        actions = np.concatenate(actions)
        _require(obs['action_mask'][np.arange(len(actions)), actions].all(),
                 'Evaluation policy selected an illegal action')
        offset = 0
        for shard, slots in chunks:
            shard['actions'][slots] = actions[offset:offset + len(slots)]
            offset += len(slots)
        statistics[identity].append({'requests': len(actions), 'shards': len(chunks),
                                     'greedy_roles': int(greedy.sum()),
                                     'stochastic_roles': int((~greedy).sum()),
                                     'global_round': global_round,
                                     'groups': sorted({shard['batch']['name']
                                                       for shard, _ in chunks}),
                                     'network_forward_sizes': forward_sizes,
                                     'network_forwards': len(forward_sizes)})


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'p50': None, 'p90': None, 'p95': None, 'p99': None}
    quantiles = np.quantile(values, [.5, .9, .95, .99])
    return {'count': len(values), 'mean': float(values.mean()),
            **{name: float(value) for name, value in
               zip(('p50', 'p90', 'p95', 'p99'), quantiles)}}


def _summary(records):
    records = sorted(records, key=lambda item: item['lane'])
    _require([row['lane'] for row in records] == list(range(len(records))),
             'Primary results require one complete record per original lane')
    placements = [row['placement'] for row in records]
    n = len(placements)
    result = {'games': n, 'seed': records[0]['seed'],
              'deterministic': records[0]['deterministic'],
              'placements': placements, 'lane_records': records,
              'first_rate': placements.count(1) / n,
              'top4_rate': sum(place <= 4 for place in placements) / n,
              'mean_placement': sum(placements) / n,
              'actions_per_game_distribution': _distribution([
                  row['reward_components']['actions_per_game'] for row in records]),
              'actions_per_recruit_turn_distribution': _distribution([
                  count for row in records for count in row['recruit_turn_actions']]),
              'no_visible_returns_player_game_fraction': float(np.mean([
                  row['reward_components']['cycle_returns'] == 0 for row in records]))}
    for metric in EVALUATION_METRICS:
        result[metric] = float(np.mean([row['reward_components'][metric] for row in records]))
    return result


def evaluate_fixed_pool(root, folder, candidates, reference, games, seed, device='cuda',
                        deterministic=False, threads=20, reward_shaping=None,
                        simulator_config=None):
    """Compatibility entry point returning candidate summaries for a single seed/mode."""
    results = evaluate_fixed_pool_batches(root, folder, candidates, reference,
        [{'name': 'single', 'games': games, 'seed': seed, 'deterministic': deterministic}],
        device=device, threads=threads, reward_shaping=reward_shaping,
        simulator_config=simulator_config)
    return results['single']


def evaluate_fixed_pool_batches(root, folder, candidates, reference, batches, device='cuda',
                               threads=20, reward_shaping=None, simulator_config=None,
                               inference_micro_batch_size=None):
    """Run every candidate×batch lane concurrently, batching ready roles globally by SHA.

    ``batches`` contains unique ``name``, integer ``seed``/``games`` and boolean
    ``deterministic`` fields. Return ``{batch_name: {candidate_name: summary}}``.
    All shards exist before the first step; inference microbatches never serialize envs.
    The original JSONL records carry group/seed/lane/native-seed/seat identities.
    """
    root, folder = Path(root), Path(folder)
    _require(batches and candidates, 'Positive fixed game budget required')
    _require(all(isinstance(batch.get('name'), str) and batch['name']
                 and type(batch.get('seed')) is int
                 and type(batch.get('games')) is int and batch['games'] > 0
                 and type(batch.get('deterministic')) is bool for batch in batches),
             'Each batch requires a name, integer seed/positive games and boolean mode')
    _require(len({batch['name'] for batch in batches}) == len(batches), 'Duplicate batch names')
    _require(inference_micro_batch_size is None
             or type(inference_micro_batch_size) is int and inference_micro_batch_size > 0,
             'Inference microbatch size must be positive or None')
    batches = [dict(batch) for batch in batches]
    _require(type(threads) is int and threads > 0, 'Positive native thread count required')
    _require(len(reference) == 3 and all(row['count'] == 2 for row in reference)
             and len({row['sha256'] for row in reference}) == 3, 'Reference must be three versions×2')
    _require(len({_name(spec) for spec in candidates}) == len(candidates), 'Duplicate candidate names')
    simulator = dict(simulator_config or {'hero_choices': 2, 'turn_action_caps': [60]})
    _require(simulator.get('hero_choices') == 2 and simulator.get('turn_action_caps') == [60],
             'Evaluation requires explicit hero2/native60-End protocol')
    reward = asdict(RewardShapingConfig(**(reward_shaping or DEFAULT_REWARD)))
    _require(reward['cycle_cost'] == .01 and reward['move_cost'] == .0005
             and reward['cycle_detection'] == 'all_actions'
             and reward['cycle_forfeit_repetitions'] is None
             and reward['turn_action_forfeit_limit'] is None, 'Unexpected evaluation reward/forfeit protocol')
    folder.mkdir(parents=True, exist_ok=True)
    outputs = ('games.jsonl', 'players.jsonl', 'turns.jsonl', 'batches.json', 'results.json',
               'evaluation-scheduler.json', 'source-manifest.json')
    _require(not any((folder / name).exists() for name in outputs),
             'Preserve previous/partial evaluation; choose a new output folder')
    all_specs = [*candidates, *reference]
    protocol = {'rules': '37.0.2', 'reference': [
        {'sha256': row['sha256'], 'count': row['count']} for row in reference],
        'reward_shaping': reward, 'simulator_config': simulator,
        'reference_deterministic': False,
        'forward_precision': 'bf16'}
    for batch in batches:
        batch['protocol'] = {**protocol, 'candidate_deterministic': batch['deterministic']}
        batch['protocol_sha256'] = hashlib.sha256(
            json.dumps(batch['protocol'], sort_keys=True).encode()).hexdigest()
    statistics, policies, sources, shards = defaultdict(list), {}, {}, []
    started = time.monotonic()
    requested = sum(batch['games'] for batch in batches) * len(candidates)
    metadata = {'status': 'running', 'started_at': now(), 'requested_concurrency': requested,
                'actual_initial_concurrency': 0,
                'games_per_candidate': sum(batch['games'] for batch in batches),
                'batches': batches, 'batch_groups': len(batches),
                'environment_shards': len(batches) * len(candidates),
                'protocol': protocol,
                'inference_micro_batch_size': inference_micro_batch_size,
                'controlled_roles': 7, 'v0_roles': 1, 'v0_behavior': 'unknown_uncontrolled',
                'replacement_games': 0, 'automatic_terminal_resets': 0,
                'inference_grouping': 'one_network_forward_per_checkpoint_sha_all_role_modes',
                'sampling_rng': 'independent_shared_seed_lane_role_inverse_cdf',
                'environment_shards_stepped_once_each_per_global_round': True,
                'all_groups_initialized_before_first_step': True}
    if len(batches) == 1:
        metadata.update(seed=batches[0]['seed'], protocol=batches[0]['protocol'],
                        protocol_sha256=batches[0]['protocol_sha256'])
    write_json(folder / 'evaluation-scheduler.json', metadata)
    rounds = deferred = 0
    try:
        with _isolated_rng(min(batch['seed'] for batch in batches), device):
            for spec in all_specs:
                _require(_digest(root / spec['model']) == spec['sha256'],
                         'Evaluation source SHA changed before loading')
                if spec['sha256'] not in sources:
                    sources[spec['sha256']] = _inspect_source(root, spec)
            _require({sources[spec['sha256']]['source_origin_sha256'] for spec in reference}
                     == set(ORIGINALS), 'Reference provenance must be the frozen V1/V2/V7 originals')
            write_json(folder / 'source-manifest.json', sources)
            for spec in all_specs:
                if spec['sha256'] not in policies:
                    policies[spec['sha256']] = load_frozen_policy(
                        root / spec['model'], device, forward_precision='bf16')
            for batch in batches:
                games, seed = batch['games'], batch['seed']
                initial_seats = None
                for candidate in candidates:
                    roles = [candidate, *(row for row in reference for _ in range(row['count']))]
                    env = tg.BatchEnv(games, seed=seed, controlled=range(7), threads=threads,
                                      config=simulator, rotate_seats=True)
                    _require(env.n_games == games and env.n_slots == games * 7,
                             'Native environment did not create every requested lane/role')
                    seats = env.slot_seats.copy()
                    if initial_seats is None:
                        initial_seats = seats
                    _require(np.array_equal(seats, initial_seats), 'Candidate lane/seat rotations differ')
                    shards.append({'batch': batch, 'spec': candidate, 'roles': roles,
                        'env': env, 'seats': seats,
                        'active': np.ones(games, dtype=bool),
                        'identities': np.tile([row['sha256'] for row in roles], games),
                        'deterministic': np.tile([batch['deterministic'], *([False] * 6)], games),
                        'rng': [np.random.default_rng(_splitmix64(native_lane_seed(seed, lane)
                                                                 ^ _splitmix64(role + 71)))
                                for lane in range(games) for role in range(7)],
                        'places': np.zeros((games, 7), dtype=np.int64), 'records': [None] * games,
                        'shapers': [ActionRewardShaper(RewardShapingConfig(**reward))
                                    for _ in range(games * 7)],
                        'totals': [None] * (games * 7),
                        'turns': [defaultdict(lambda: {'actions': 0, 'cycle_returns': 0,
                            'max_consecutive_cycle_returns': 0, 'forced_ends': 0})
                            for _ in range(games * 7)]})
            metadata['actual_initial_concurrency'] = sum(len(s['active']) for s in shards)
            _require(metadata['actual_initial_concurrency'] == requested,
                     'Actual evaluation concurrency differs from the fixed budget')
            write_json(folder / 'evaluation-scheduler.json', metadata)
            with (folder / 'games.jsonl').open('x') as games_stream, \
                 (folder / 'players.jsonl').open('x') as player_stream, \
                 (folder / 'turns.jsonl').open('x') as turn_stream:
                while any(shard['active'].any() for shard in shards):
                    rounds += 1
                    _merged_actions(shards, policies, statistics, inference_micro_batch_size, rounds)
                    _require(rounds <= 100000, 'Evaluation exceeded finite scheduling guard')
                    for shard in shards:
                        if not shard['active'].any():
                            continue
                        batch = shard['batch']
                        seed = batch['seed']
                        rewards, dones, info = shard['env'].step(shard['actions'], active=shard['active'])
                        _require(not info['rejected'].any(), 'Native evaluation rejected a masked action')
                        _require(not ((info['placement'] == 0) & dones).any(), 'Missing real terminal placement')
                        deferred += int(info['deferred'].sum())
                        after = shard['env'].observe()
                        for slot in np.flatnonzero(shard['ready'] & ~info['deferred']):
                            before_one, after_one = _one(shard['obs'], slot), _one(after, slot)
                            turn = int(round(float(before_one['features'][0]) * 20))
                            row, shaper = shard['turns'][slot][turn], shard['shapers'][slot]
                            forced = (turn > 0 and shard['actions'][slot] == 0
                                      and row['actions'] >= 60 and before_one['action_mask'].sum() == 1)
                            previous = shaper._totals['cycle_returns']
                            _, completed = shaper.step(shard['actions'][slot], before_one, after_one,
                                                       float(rewards[slot]), done=bool(dones[slot]))
                            total = completed or shaper._totals
                            row['actions'] += 1
                            row['forced_ends'] += int(forced)
                            row['cycle_returns'] += int(total['cycle_returns'] - previous)
                            row['max_consecutive_cycle_returns'] = max(
                                row['max_consecutive_cycle_returns'], shaper.consecutive_returns)
                            if completed is not None:
                                shard['totals'][slot] = completed
                        for slot in np.flatnonzero(dones):
                            lane, role = divmod(int(slot), 7)
                            _require(shard['places'][lane, role] == 0, 'Duplicate neural terminal')
                            placement = int(info['placement'][slot])
                            _require(1 <= placement <= 8, 'Invalid neural placement')
                            shard['places'][lane, role] = placement
                            if shard['totals'][slot] is None:
                                shard['totals'][slot] = shard['shapers'][slot].finish_terminal(
                                    float(rewards[slot]), placement)
                        finished = shard['active'] & (shard['places'] > 0).all(axis=1)
                        for lane in np.flatnonzero(finished):
                            places = shard['places'][lane].tolist()
                            _require(len(set(places)) == 7, 'Duplicate placements in an eight-seat game')
                            v0 = list(set(range(1, 9)) - set(places))
                            _require(len(v0) == 1, 'Uncontrolled V0 placement is not a unique complement')
                            players = []
                            for role, role_spec in enumerate(shard['roles']):
                                slot = int(lane) * 7 + role
                                turns = shard['turns'][slot]
                                recruit = [row for turn, row in turns.items() if turn > 0]
                                components = dict(shard['totals'][slot])
                                components['cap_end_rate'] = (sum(row['forced_ends'] for row in recruit)
                                                              / max(len(recruit), 1))
                                player = {'role': role, 'checkpoint_sha256': role_spec['sha256'],
                                    'seat': int(shard['seats'][slot]), 'placement': places[role],
                                    'deterministic': bool(shard['deterministic'][slot]),
                                    'reward_components': components, 'turns': dict(turns)}
                                players.append(player)
                                common = {'group': batch['name'], 'candidate': _name(shard['spec']),
                                          'seed': int(seed),
                                          'lane': int(lane), 'native_seed': str(native_lane_seed(seed, lane))}
                                player_stream.write(json.dumps({**common, **player}) + '\n')
                                for turn, row in turns.items():
                                    turn_stream.write(json.dumps({**common, 'role': role,
                                        'checkpoint_sha256': role_spec['sha256'],
                                        'seat': player['seat'], 'turn': turn, **row}) + '\n')
                            record = {'group': batch['name'], 'candidate': _name(shard['spec']),
                                'lane': int(lane),
                                'seed': int(seed), 'native_seed': str(native_lane_seed(seed, lane)),
                                'learner_seat': players[0]['seat'], 'placement': places[0],
                                'deterministic': batch['deterministic'],
                                'protocol_sha256': batch['protocol_sha256'],
                                'checkpoint_sha256': shard['spec']['sha256'],
                                'reward_components': players[0]['reward_components'],
                                'recruit_turn_actions': [row['actions']
                                    for turn, row in shard['turns'][int(lane) * 7].items()
                                    if turn > 0]}
                            shard['records'][lane] = record
                            games_stream.write(json.dumps({**record, 'players': players,
                                'v0_placement': v0[0], 'v0_behavior': 'unknown_uncontrolled'}) + '\n')
                            games_stream.flush()
                            _require(shard['env'].episode_index(int(lane)) == 1,
                                     'Native lane was replayed or replaced')
                            shard['active'][lane] = False
            _require(all(_digest(root / spec['model']) == spec['sha256'] for spec in all_specs),
                     'Checkpoint file changed during evaluation')
        results = {batch['name']: {} for batch in batches}
        for shard in shards:
            batch = shard['batch']
            result = _summary(shard['records'])
            result.update(group=batch['name'], rules='37.0.2', forward_precision='bf16',
                          reward_shaping=reward, protocol_sha256=batch['protocol_sha256'],
                          v0_opponents=1,
                          opponent_groups=[{'name': _name(row), 'count': row['count'],
                                            'checkpoint_sha256': row['sha256'],
                                            'deterministic': False} for row in reference])
            results[batch['name']][_name(shard['spec'])] = result
        # Preserve the historical single-group artifact shape for the compatibility wrapper.
        output = results['single'] if len(batches) == 1 and batches[0]['name'] == 'single' else results
        write_json(folder / 'results.json', output)
        write_json(folder / 'batches.json', dict(statistics))
        metadata.update(status='completed', completed_at=now(), completed_games=requested,
                        rounds=rounds, rejected_actions=0, deferred_actions=deferred,
                        automatic_terminal_resets=requested, replacement_games=0,
                        elapsed_seconds=time.monotonic() - started,
                        predictions_finalized_before_statistics=True,
                        hashes_verified_after=True,
                        games_sha256=_digest(folder / 'games.jsonl'))
        write_json(folder / 'evaluation-scheduler.json', metadata)
        return results
    except BaseException as error:
        metadata.update(status='failed_preserved', error=f'{type(error).__name__}: {error}',
                        rounds=rounds, elapsed_seconds=time.monotonic() - started)
        write_json(folder / 'evaluation-scheduler.json', metadata)
        raise
    finally:
        for shard in shards:
            if hasattr(shard['env'], 'close'):
                shard['env'].close()
            shard['env'] = None


def _records(value):
    if isinstance(value, dict) and 'lane_records' in value:
        records = list(value['lane_records'])
        if 'placements' in value:
            ordered = sorted(records, key=lambda row: row['lane'])
            _require(value['placements'] == [row['placement'] for row in ordered],
                     'Summary placements disagree with original lane records')
        return records
    if isinstance(value, (list, tuple)):
        if all(isinstance(row, dict) and 'lane' in row for row in value):
            return list(value)
        return [record for item in value for record in _records(item)]
    raise ValueError('Paired statistics require explicit lane records, not completion-order placements')


def paired_stats(reference, candidate, seed=0, bootstrap_samples=10000):
    """Bootstrap counterfactual game pairs; seven reference roles are never extra samples."""
    def indexed(value):
        result = {}
        for row in _records(value):
            key = (int(row['seed']), int(row['lane']), int(row['learner_seat']), str(row['native_seed']))
            _require(key not in result, 'Duplicate seed/lane/seat pair')
            _require(1 <= row['placement'] <= 8 and int(row['placement']) == row['placement'],
                     'Invalid paired placement')
            _require(row.get('protocol_sha256'), 'Missing fixed-pool protocol provenance')
            result[key] = row
        return result
    baseline, proposed = indexed(reference), indexed(candidate)
    _require(baseline and baseline.keys() == proposed.keys(), 'Paired lane/seed/seat sets differ')
    keys = sorted(baseline)
    for key in keys:
        _require(baseline[key]['protocol_sha256'] == proposed[key]['protocol_sha256']
                 and baseline[key]['deterministic'] == proposed[key]['deterministic'],
                 'Paired evaluation protocol/mode differs')
    old = np.array([baseline[key]['placement'] for key in keys], dtype=np.float64)
    new = np.array([proposed[key]['placement'] for key in keys], dtype=np.float64)
    differences = new - old
    _require(type(bootstrap_samples) is int and bootstrap_samples > 0, 'Positive bootstrap budget required')
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for begin in range(0, bootstrap_samples, 256):
        count = min(256, bootstrap_samples - begin)
        means[begin:begin + count] = differences[rng.integers(0, len(keys),
                                                           size=(count, len(keys)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'paired_games': len(keys), 'reference_first_rate': float((old == 1).mean()),
            'candidate_first_rate': float((new == 1).mean()),
            'reference_top4_rate': float((old <= 4).mean()),
            'candidate_top4_rate': float((new <= 4).mean()),
            'reference_mean_placement': float(old.mean()),
            'candidate_mean_placement': float(new.mean()),
            'mean_difference_candidate_minus_reference': float(differences.mean()),
            'difference_ci95': [float(low), float(high)], 'bootstrap_seed': int(seed),
            'bootstrap_samples': bootstrap_samples, 'unit': 'whole native seed/lane game pair',
            'negative_difference_is_better': True}
