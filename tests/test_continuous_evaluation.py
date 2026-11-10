"""Fixed-pool scheduling and paired statistics use lane identity, never finish order."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')

import taverngym as tg
from taverngym.learning import continuous_evaluation as evaluation


class FixtureEnv:
    instances = []

    def __init__(self, games, seed, controlled, threads, config, rotate_seats):
        assert tuple(controlled) == tuple(range(7)) and rotate_seats
        assert config == {'hero_choices': 2, 'turn_action_caps': [60]}
        self.index = len(self.instances)
        self.instances.append(self)
        self.n_games, self.n_slots = games, games * 7
        self.counts = np.zeros(games, dtype=int)
        self.indices = np.zeros(games, dtype=int)
        self.actions_seen = []
        self.slot_seats = np.array([(lane + seed + role) % 8
                                    for lane in range(games) for role in range(7)])

    def observe(self):
        features = np.zeros((self.n_slots, tg.OBS_FLOATS), dtype=np.float32)
        features[:, 0] = .05
        masks = np.zeros((self.n_slots, tg.N_ACTIONS), dtype=bool)
        masks[:, :2] = True
        return {'features': features, 'cards': np.zeros((self.n_slots, tg.OBS_CARDS), dtype=np.int32),
                'action_mask': masks, 'ready': np.ones(self.n_slots, dtype=bool)}

    def step(self, actions, active):
        rewards = np.zeros(self.n_slots, dtype=np.float32)
        dones = np.zeros(self.n_slots, dtype=bool)
        places = np.zeros(self.n_slots, dtype=int)
        for lane in np.flatnonzero(active):
            assert self.indices[lane] == 0, 'Replacement game was stepped'
            self.counts[lane] += 1
            self.actions_seen.extend((int(lane), role, int(actions[lane * 7 + role]))
                                     for role in range(7))
            if self.counts[lane] == 1 + (lane + 1) % self.n_games:
                candidate = ([7, 4, 1] if self.index == 0 else [6, 3, 1])[lane]
                result = [candidate, *[p for p in range(1, 8) if p != candidate]]
                places[lane * 7:(lane + 1) * 7] = result
                dones[lane * 7:(lane + 1) * 7] = True
                rewards[lane * 7:(lane + 1) * 7] = (9 - 2 * np.array(result)) / 7
                self.indices[lane] = 1
        return rewards, dones, {'placement': places, 'rejected': np.zeros(self.n_slots, dtype=bool),
                                'deferred': np.zeros(self.n_slots, dtype=bool)}

    def episode_index(self, lane):
        return self.indices[lane]


class FixturePolicy:
    def obs_to_tensor(self, obs):
        assert set(obs) == {'features', 'cards', 'action_mask'}, 'Placement leaked into observations'
        return {key: torch.as_tensor(value) for key, value in obs.items()}, True

    def get_distribution(self, obs):
        # All candidate×games environments must exist before the first neural inference.
        assert len(FixtureEnv.instances) == 2
        probabilities = torch.zeros((len(obs['features']), tg.N_ACTIONS))
        probabilities[:, :2] = .5  # Greedy is0; reference random must include1.
        return SimpleNamespace(distribution=SimpleNamespace(probs=probabilities))


def fixed_fixture(tmp_path, monkeypatch):
    FixtureEnv.instances = []
    origins = list(evaluation.ORIGINALS)
    reference = []
    origin_for_hash = {}
    for index, label in enumerate(('v1', 'v2', 'v7')):
        file = tmp_path / f'{label}.zip'
        file.write_bytes(label.encode())
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        origin_for_hash[digest] = origins[index]
        reference.append({'name': label, 'model': file.name, 'sha256': digest, 'count': 2})
    candidates = [{**reference[-1], 'name': name} for name in ('baseline', 'candidate')]
    monkeypatch.setattr(evaluation.tg, 'BatchEnv', FixtureEnv)
    monkeypatch.setattr(evaluation, '_inspect_source', lambda root, spec: {
        'sha256': spec['sha256'], 'source_origin_sha256': origin_for_hash[spec['sha256']]})
    monkeypatch.setattr(evaluation, 'load_frozen_policy', lambda *args, **kwargs: FixturePolicy())
    return candidates, reference


def test_full_concurrency_lane_order_and_greedy_reference_modes_are_independent(tmp_path, monkeypatch):
    candidates, reference = fixed_fixture(tmp_path, monkeypatch)
    before = torch.random.get_rng_state().clone()
    result = evaluation.evaluate_fixed_pool(tmp_path, tmp_path / 'evaluation', candidates,
                                            reference, 3, 123, device='cpu', deterministic=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert result['baseline']['placements'] == [7, 4, 1]
    assert result['candidate']['placements'] == [6, 3, 1]
    raw = [json.loads(line) for line in (tmp_path / 'evaluation/games.jsonl').read_text().splitlines()]
    assert [row['lane'] for row in raw if row['candidate'] == 'baseline'] == [2, 0, 1]
    assert len(raw) == 6 and all(row['v0_behavior'] == 'unknown_uncontrolled' for row in raw)
    for env in FixtureEnv.instances:
        assert env.counts.tolist() == [2, 3, 1]
        assert all(action == 0 for _, role, action in env.actions_seen if role == 0)
        assert any(action == 1 for _, role, action in env.actions_seen if role > 0)
    scheduler = json.loads((tmp_path / 'evaluation/evaluation-scheduler.json').read_text())
    assert scheduler['actual_initial_concurrency'] == scheduler['completed_games'] == 6
    assert scheduler['replacement_games'] == 0
    assert scheduler['inference_grouping'] == 'one_network_forward_per_checkpoint_sha_all_role_modes'
    rows = result['baseline']['lane_records']
    assert [row['lane'] for row in rows] == [0, 1, 2]
    assert [row['learner_seat'] for row in rows] == [123 % 8, 124 % 8, 125 % 8]
    paired = evaluation.paired_stats(result['baseline'], result['candidate'], seed=9,
                                     bootstrap_samples=2000)
    assert paired['paired_games'] == 3
    assert paired['mean_difference_candidate_minus_reference'] == pytest.approx(-2 / 3)
    assert paired['difference_ci95'][1] <= 0
    assert len((tmp_path / 'evaluation/players.jsonl').read_text().splitlines()) == 42


def paired_rows(placements, seed=100):
    return [{'seed': seed, 'lane': lane, 'native_seed': str(900 + lane),
             'learner_seat': lane % 8, 'placement': place, 'deterministic': False,
             'protocol_sha256': 'fixed'} for lane, place in enumerate(placements)]


def test_pairing_matches_lane_seed_seat_and_bootstraps_whole_games():
    reference = paired_rows([8, 2, 5])
    candidate = list(reversed(paired_rows([7, 2, 3])))
    first = evaluation.paired_stats(reference, candidate, seed=17, bootstrap_samples=2000)
    assert first == evaluation.paired_stats(reference, candidate, seed=17, bootstrap_samples=2000)
    assert first['paired_games'] == 3
    assert first['mean_difference_candidate_minus_reference'] == -1
    reference2, candidate2 = paired_rows([4, 4], 101), paired_rows([3, 3], 101)
    combined = evaluation.paired_stats([{'lane_records': reference}, {'lane_records': reference2}],
                                      [{'lane_records': candidate}, {'lane_records': candidate2}],
                                      bootstrap_samples=100)
    assert combined['paired_games'] == 5
    assert combined['unit'] == 'whole native seed/lane game pair'


@pytest.mark.parametrize('corruption', ['seat', 'protocol', 'mode', 'duplicate', 'summary_order'])
def test_pairing_rejects_unmatched_or_ambiguous_records(corruption):
    baseline = paired_rows([7, 3])
    candidate = copy.deepcopy(baseline)
    if corruption == 'seat':
        candidate[0]['learner_seat'] = 6
    elif corruption == 'protocol':
        candidate[0]['protocol_sha256'] = 'other pool'
    elif corruption == 'mode':
        candidate[0]['deterministic'] = True
    elif corruption == 'duplicate':
        candidate.append(copy.deepcopy(candidate[0]))
    else:
        candidate = {'lane_records': candidate, 'placements': [3, 7]}
    with pytest.raises(ValueError):
        evaluation.paired_stats(baseline, candidate, bootstrap_samples=100)


@pytest.mark.parametrize('version', ['v1', 'v2', 'v7'])
def test_private_stage_zero_migration_is_verified_without_old_budget_requirement(version):
    root = Path(__file__).resolve().parents[1]
    filename = 'eric-v7-3702-adapted.zip' if version == 'v7' else f'{version}-3702-adapted.zip'
    path = root / 'runs/imported/eric-local-20261108-adapted' / filename
    if not path.exists():
        pytest.skip('Private artifacts are intentionally not in Git')
    info = evaluation._inspect_source(root, {'name': version, 'model': str(path),
        'model_version': f'eric-3702-{version}-adapted', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert info['stage_samples'] == 0 and info['adam_total'] == (8800 if version == 'v7' else 3200)
    assert info['parameter_mapping_verified'] is True


@pytest.mark.parametrize('damage', ['nan', 'illegal'])
def test_nonfinite_or_illegal_inference_preserves_partial_failure(tmp_path, monkeypatch, damage):
    candidates, reference = fixed_fixture(tmp_path, monkeypatch)

    class InvalidPolicy(FixturePolicy):
        def get_distribution(self, obs):
            value = super().get_distribution(obs)
            probabilities = value.distribution.probs
            if damage == 'nan':
                probabilities[0, 0] = float('nan')
            else:
                probabilities.zero_()
                probabilities[:, 2] = 1.  # The fixture mask only allows0/1.
            return value

    monkeypatch.setattr(evaluation, 'load_frozen_policy', lambda *a, **k: InvalidPolicy())
    folder = tmp_path / 'failure'
    with pytest.raises(ValueError, match='Nonfinite|illegal'):
        evaluation.evaluate_fixed_pool(tmp_path, folder, candidates, reference,
                                        3, 123, device='cpu', deterministic=True)
    metadata = json.loads((folder / 'evaluation-scheduler.json').read_text())
    assert metadata['status'] == 'failed_preserved'
    assert not (folder / 'results.json').exists()
    assert all(not env.counts.any() for env in FixtureEnv.instances)


def test_inverse_cdf_zero_and_plateau_boundaries_never_select_masked_actions():
    probabilities = torch.zeros((4, tg.N_ACTIONS))
    probabilities[:, 1] = .5
    probabilities[:, 4] = .5
    uniforms = np.array([0., .5, np.nextafter(1., 0.), .5])
    modes = np.array([False, False, False, True])
    actions = evaluation._actions_from_probabilities(probabilities, modes, uniforms)
    assert actions.tolist() == [1, 4, 4, 1]
    assert all(action in (1, 4) for action in actions.tolist())
    with pytest.raises(ValueError, match=r'in\[0,1\)'):
        evaluation._actions_from_probabilities(probabilities[:1], [False], np.array([1.]))


def batch_fixture(tmp_path, monkeypatch, expected_shards):
    candidates, reference = fixed_fixture(tmp_path, monkeypatch)
    events, clock = [], {'round': 0}

    class BatchFixtureEnv(FixtureEnv):
        def __init__(self, games, seed, **kwargs):
            super().__init__(games, seed, **kwargs)
            self.seed = seed
            events.append(('init', self.index))

        def step(self, actions, active):
            events.append(('step', clock['round'], self.index, tuple(np.flatnonzero(active))))
            return super().step(actions, active)

    class BatchFixturePolicy(FixturePolicy):
        def get_distribution(self, obs):
            assert len(FixtureEnv.instances) == expected_shards
            events.append(('forward', clock['round'], len(obs['features'])))
            probabilities = torch.zeros((len(obs['features']), tg.N_ACTIONS))
            probabilities[:, :2] = .5
            return SimpleNamespace(distribution=SimpleNamespace(probs=probabilities))

    merged_actions = evaluation._merged_actions

    def merged(shards, policies, statistics, inference_micro_batch_size, global_round):
        clock['round'] = global_round
        events.append(('round', global_round))
        return merged_actions(shards, policies, statistics, inference_micro_batch_size, global_round)

    monkeypatch.setattr(evaluation.tg, 'BatchEnv', BatchFixtureEnv)
    monkeypatch.setattr(evaluation, 'load_frozen_policy', lambda *a, **k: BatchFixturePolicy())
    monkeypatch.setattr(evaluation, '_merged_actions', merged)
    return candidates, reference, events


def five_batches():
    return [{'name': f'group{index}', 'seed': 100 + index, 'games': games,
             'deterministic': index >= 3} for index, games in enumerate((3, 3, 2, 2, 1))]


def test_all_five_groups_initialize_and_advance_together_with_cross_group_sha_inference(
        tmp_path, monkeypatch):
    candidates, reference, events = batch_fixture(tmp_path, monkeypatch, expected_shards=10)
    folder, batches = tmp_path / 'all-groups', five_batches()
    results = evaluation.evaluate_fixed_pool_batches(tmp_path, folder, candidates, reference,
                                                     batches, device='cpu')
    assert [event[0] for event in events[:10]] == ['init'] * 10
    assert events[10] == ('round', 1)
    steps = [event for event in events if event[0] == 'step']
    keys = [(event[1], event[2]) for event in steps]
    assert len(keys) == len(set(keys)), 'A shard advanced twice in a global round'
    for env in FixtureEnv.instances:
        assert [event[1] for event in steps if event[2] == env.index] == list(
            range(1, env.n_games + 1))
        assert env.counts.tolist() == [1 + (lane + 1) % env.n_games
                                        for lane in range(env.n_games)]
    statistics = json.loads((folder / 'batches.json').read_text())
    assert len(statistics) == 3
    assert all(len(rows[0]['groups']) == 5 and rows[0]['shards'] == 10
               and rows[0]['network_forwards'] == 1 for rows in statistics.values())
    v7 = statistics[reference[-1]['sha256']][0]
    assert v7['greedy_roles'] > 0 and v7['stochastic_roles'] > v7['greedy_roles']
    scheduler = json.loads((folder / 'evaluation-scheduler.json').read_text())
    assert scheduler['actual_initial_concurrency'] == scheduler['completed_games'] == 22
    assert scheduler['batch_groups'] == 5 and scheduler['environment_shards'] == 10
    assert scheduler['replacement_games'] == 0 and scheduler['inference_micro_batch_size'] is None
    for stream in ('games.jsonl', 'players.jsonl', 'turns.jsonl'):
        records = [json.loads(line) for line in (folder / stream).read_text().splitlines()]
        assert {row['group'] for row in records} == {batch['name'] for batch in batches}
        assert all({'seed', 'lane', 'native_seed'} <= row.keys() for row in records)
        assert all(('learner_seat' if stream == 'games.jsonl' else 'seat') in row
                   for row in records)
    for batch in batches:
        for result in results[batch['name']].values():
            assert result['games'] == batch['games'] and result['seed'] == batch['seed']
            assert result['deterministic'] == batch['deterministic']
            assert [row['lane'] for row in result['lane_records']] == list(range(batch['games']))
    raw_players = [json.loads(line) for line in (folder / 'players.jsonl').read_text().splitlines()]
    assert all(not row['deterministic'] for row in raw_players if row['role'] > 0)


def test_batch_order_and_inference_microbatches_do_not_change_lane_role_random_streams(
        tmp_path, monkeypatch):
    candidates, reference, _ = batch_fixture(tmp_path, monkeypatch, expected_shards=10)
    batches = five_batches()
    first = evaluation.evaluate_fixed_pool_batches(tmp_path, tmp_path / 'first', candidates,
                                                    reference, batches, device='cpu')
    def observed_actions():
        return {(env.seed, env.index % 2): env.actions_seen for env in FixtureEnv.instances}
    original_actions = observed_actions()
    FixtureEnv.instances = []
    second = evaluation.evaluate_fixed_pool_batches(tmp_path, tmp_path / 'reversed', candidates,
        reference, list(reversed(batches)), device='cpu', inference_micro_batch_size=5)
    assert observed_actions() == original_actions
    for batch in batches:
        for candidate in ('baseline', 'candidate'):
            before, after = first[batch['name']][candidate], second[batch['name']][candidate]
            assert before['protocol_sha256'] == after['protocol_sha256']
            assert [row['native_seed'] for row in before['lane_records']] == [
                row['native_seed'] for row in after['lane_records']]
    statistics = json.loads((tmp_path / 'reversed/batches.json').read_text())
    assert any(row['network_forwards'] > 1 for rows in statistics.values() for row in rows)
    assert all(0 < size <= 5 for rows in statistics.values() for row in rows
               for size in row['network_forward_sizes'])
