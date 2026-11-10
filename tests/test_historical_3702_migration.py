"""Frozen historical artifacts preserve learned state and discard legacy stage protocols."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('stable_baselines3')
from stable_baselines3.common.save_util import load_from_zip_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'historical_3702_migration', ROOT / 'tools/migrate_eric_historical_3702.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_historical_metadata_resets_local_clock_and_removes_old_training_protocol(version):
    original = {'policy_kwargs': {'card_data_sha256': 'old', 'model_config': {}},
                'experiment_config': {'iterations': 100, 'opponent_pool': ['old'],
                                      'reward_shaping': {'cycle_cost': .01},
                                      'tracking': {'entity': 'legacy'}},
                'num_timesteps': 26214400, '_n_updates': 100, 'lr_schedule': 'old schedule'}
    data, removed = migration.prepared_metadata(original, version, {'source_adam': 3200}, {})
    assert data['num_timesteps'] == data['_n_updates'] == 0
    assert data['_optimizer_step_offset'] == 3200
    protocol = data['experiment_config']
    assert protocol['model_version'] == f'eric-3702-{version}-adapted'
    assert protocol['reward_shaping'] is None and protocol['training_start_authorized'] is False
    assert not {'iterations', 'opponent_pool', 'tracking'} & protocol.keys()
    assert set(removed) == {'iterations', 'opponent_pool', 'reward_shaping', 'tracking'}
    assert 'lr_schedule' not in data
    assert original['num_timesteps'] == 26214400  # Source metadata must remain untouched.


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_private_historical_artifact_has_exact_learned_parameters_and_complete_adam(version):
    source = ROOT / 'runs/imported/eric-local-20261108' / migration.SOURCES[version][0]
    target = ROOT / f'runs/imported/eric-local-20261108-adapted/{version}-3702-adapted.zip'
    if not source.exists() or not target.exists():
        pytest.skip('Private historical model files are intentionally not stored in Git')
    source_data, source_parameters, _ = load_from_zip_file(source, device='cpu')
    data, parameters, _ = load_from_zip_file(target, device='cpu')
    changes = {name for name in source_parameters['policy']
               if not torch.equal(source_parameters['policy'][name], parameters['policy'][name])}
    assert changes == migration.helpers.REFRESHED_BUFFERS
    assert (migration.helpers.fingerprint(source_parameters['policy.optimizer'])
            == migration.helpers.fingerprint(parameters['policy.optimizer']))
    assert len(parameters['policy.optimizer']['state']) == 91
    assert all(int(state['step']) == 3200
               for state in parameters['policy.optimizer']['state'].values())
    assert data['num_timesteps'] == data['_n_updates'] == 0
    assert data['_optimizer_step_offset'] == 3200
    assert data['policy_kwargs']['card_data_sha256'] == migration.helpers.CURRENT_CARDS_SHA
    assert source_data['policy_kwargs']['card_data_sha256'] == migration.helpers.OLD_CARDS_SHA
    assert data['experiment_config']['model_version'] == f'eric-3702-{version}-adapted'
    assert data['experiment_config']['reward_shaping'] is None
