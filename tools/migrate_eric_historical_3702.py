"""Strict CPU-only V1/V2 adaptation for frozen 37.0.2 historical opponents."""

import argparse
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import torch
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

import taverngym as tg
from taverngym.learning.model import EntityActorCritic, ModelConfig
from taverngym.learning.ppo import EntityPPO, PlacementBuffer

_SPEC = importlib.util.spec_from_file_location(
    'eric_3702_migration_helpers', Path(__file__).with_name('migrate_eric_3702.py'))
helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helpers)

SOURCES = {
    'v1': ('20261031-131146-best_model.zip',
           '7173d82343fcc79742838c72725c582966a3c5b7c84e15014debea292f32de24'),
    'v2': ('20261031-165718-best_model.zip',
           'e4a748e933af5b642c50c223502d127ed1742599d841ab7d3838cf26df7ea6b4'),
}


def prepared_metadata(source, version, source_history, text_metadata):
    """Discard legacy stage protocol; retain source history separately from new clocks."""
    helpers.require(version in SOURCES, 'Unexpected historical version')
    data = copy.deepcopy(source)
    removed = sorted(data.get('experiment_config', {}))
    data['policy_kwargs'] = {**data['policy_kwargs'],
                             'card_data_sha256': helpers.CURRENT_CARDS_SHA}
    data['rule_text_metadata'] = copy.deepcopy(text_metadata)
    data['experiment_config'] = {
        'name': f'eric-3702-{version}-adapted-frozen-opponent',
        'model_version': f'eric-3702-{version}-adapted',
        'reward_shaping': None,
        'migration_only': True,
        'training_start_authorized': False,
        'frozen_opponent_candidate': True,
        'source_history': copy.deepcopy(source_history),
        'current_card_data_sha256': helpers.CURRENT_CARDS_SHA,
        'inspection_rollout_dimensions': {'n_envs': 1, 'n_steps': 2, 'batch_size': 2},
        'requires_explicit_runtime_opponent_protocol': True,
    }
    data.update(num_timesteps=0, _n_updates=0, _total_timesteps=0, _num_timesteps_at_start=0,
                _optimizer_step_offset=3200, _episode_num=0, _current_progress_remaining=1.0,
                _last_obs=None, _last_original_obs=None, _last_episode_starts=None,
                ep_info_buffer=None, ep_success_buffer=None, start_time=0,
                n_envs=1, n_steps=2, batch_size=2, n_epochs=1, micro_batch_size=1024,
                rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={},
                tensorboard_log=None, verbose=0, seed=37026001,
                last_rollout_seconds=0.0, last_train_seconds=0.0)
    data.pop('lr_schedule', None)
    return data, removed


def migrate_version(root, evidence, output_root, version, current_data, current_parameters,
                    fresh_sha, source_code_audit):
    filename, expected_sha = SOURCES[version]
    source = root / 'runs/imported/eric-local-20261108' / filename
    helpers.require(helpers.sha(source) == expected_sha, 'Original ZIP SHA mismatch')
    with zipfile.ZipFile(source) as archive:
        helpers.require(archive.testzip() is None, 'Original ZIP CRC mismatch')
    data, parameters, variables = load_from_zip_file(source, device='cpu')
    helpers.require(data['policy_kwargs']['card_data_sha256'] == helpers.OLD_CARDS_SHA,
                    'Original card provenance differs')
    helpers.require(data['policy_kwargs']['model_config']
                    == current_data['policy_kwargs']['model_config'], 'Model architecture differs')
    network = EntityActorCritic(ModelConfig(**data['policy_kwargs']['model_config']))
    parameter_names = ['network.' + name for name, _ in network.named_parameters()]
    helpers.require(len(parameter_names) == 91, 'Expected all91 learned parameters')
    original_policy = parameters['policy']
    original_adam = parameters['policy.optimizer']
    mapping = helpers.audit_optimizer(original_policy, original_adam, parameter_names, 3200)
    learned = {name: original_policy[name] for name in parameter_names}
    original_learned_sha = helpers.fingerprint(learned)
    original_adam_sha = helpers.fingerprint(original_adam)
    original_adam_moments_sha = helpers.fingerprint(original_adam['state'])
    adapted_policy = helpers.refresh_policy(
        original_policy, current_parameters['policy'], parameter_names)
    helpers.require(torch.equal(network.static_features, adapted_policy['network.static_features']),
                    'Refreshed static buffer differs from current native features')
    source_history = {
        'source_version': version, 'source_checkpoint': str(source),
        'source_sha256': expected_sha, 'source_ppo_iteration': data['_n_updates'],
        'source_samples': data['num_timesteps'], 'source_adam_total': 3200,
        'source_stage_adam_offset': data.get('_optimizer_step_offset', 0),
        'source_card_data_sha256': helpers.OLD_CARDS_SHA,
    }
    # Rejection of original sources remains mandatory; no constructor guard is bypassed.
    try:
        EntityPPO.load(source, device='cpu', n_envs=1, n_steps=2, batch_size=2,
                       rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    except ValueError as error:
        helpers.require('Checkpoint card vocabulary differs' in str(error),
                        'Unexpected original source rejection')
    else:
        raise ValueError('Original source unexpectedly bypassed the vocabulary guard')
    migrated_data, removed = prepared_metadata(
        data, version, source_history, current_data['rule_text_metadata'])
    source_lr = float(original_adam['param_groups'][0]['lr'])
    migrated_data['learning_rate'] = source_lr
    migrated_data['experiment_config'].update(
        source_optimizer_lr_preserved_not_runtime_lr_choice=source_lr,
        fresh_asset_checkpoint_sha256=fresh_sha)
    staged = evidence / f'{version}-prepared.zip'
    target = output_root / f'{version}-3702-adapted.zip'
    helpers.require(not staged.exists() and not target.exists(), 'Refusing to overwrite an artifact')
    save_to_zip_file(staged, data=migrated_data,
                     params={**parameters, 'policy': adapted_policy}, pytorch_variables=variables)
    model = EntityPPO.load(staged, device='cpu')
    helpers.require(model.n_envs == 1 and model.n_steps == 2, 'Historical rollout was allocated')
    helpers.require(helpers.fingerprint(model.policy.optimizer.state_dict()) == original_adam_sha,
                    'Standard load changed complete Adam state')
    model.start_new_stage(preserve_adam=True)
    model._last_episode_starts = None
    model.save(target)
    restored = EntityPPO.load(target, device='cpu')
    state = restored.policy.state_dict()
    helpers.require(all(torch.equal(state[name], original_policy[name]) for name in parameter_names),
                    'Standard save/reload changed a learned parameter')
    helpers.require(all(torch.equal(state[name], adapted_policy[name]) for name in adapted_policy),
                    'Standard save/reload changed a rule/action buffer')
    helpers.require(helpers.fingerprint(restored.policy.optimizer.state_dict()) == original_adam_sha,
                    'Standard save/reload changed Adam state/parameter mapping')
    helpers.require(restored.num_timesteps == 0 and restored._n_updates == 0
                    and restored.optimizer_step_count() == 0
                    and restored.optimizer_total_step_count() == 3200,
                    'Local/cumulative stage counters differ')
    helpers.require(restored.policy.card_data_sha256 == helpers.CURRENT_CARDS_SHA,
                    'Current card metadata differs')
    forward = helpers.cpu_forward(restored)
    helpers.require(helpers.fingerprint(restored.policy.optimizer.state_dict()) == original_adam_sha,
                    'CPU legal forward changed Adam')
    helpers.require(helpers.sha(source) == expected_sha, 'Original source changed')
    fingerprints = {
        'source_learned': original_learned_sha,
        'adapted_learned': helpers.fingerprint({name: state[name] for name in parameter_names}),
        'source_adam_full': original_adam_sha,
        'adapted_adam_full': helpers.fingerprint(restored.policy.optimizer.state_dict()),
        'source_adam_moments': original_adam_moments_sha,
        'adapted_adam_moments': helpers.fingerprint(restored.policy.optimizer.state_dict()['state']),
        'source_policy_all': helpers.fingerprint(original_policy),
        'adapted_policy_all': helpers.fingerprint(state),
        'adapted_buffers': {name: helpers.fingerprint(state[name]) for name in state
                            if name not in parameter_names},
    }
    report = {
        'status': 'cpu_adapted_frozen_opponent_candidate', 'version': version,
        'model_version': f'eric-3702-{version}-adapted', 'reward_shaping': None,
        'source': source_history, 'adapted_checkpoint': str(target),
        'adapted_sha256': helpers.sha(target), 'stage_samples': 0, 'stage_ppo': 0,
        'stage_adam': 0, 'cumulative_adam': 3200, 'adam_offset': 3200,
        'parameter_count': 91, 'learned_parameters_exact': True, 'full_adam_exact': True,
        'refreshed_buffers_only': sorted(helpers.REFRESHED_BUFFERS),
        'fresh_asset_checkpoint_sha256': fresh_sha, 'source_code_audit': source_code_audit,
        'optimizer_registration_order_and_slot_mapping_verified': True,
        'old_experiment_protocol_keys_removed': removed,
        'cpu_forward_actions': len(forward), 'cpu_forward_seeds': [37025011, 37025012],
        'original_card_guard_rejection_verified': True,
        'fingerprints': fingerprints, 'source_forward_precision':
        data['policy_kwargs'].get('forward_precision', 'fp32'),
        'gpu_computation': False, 'optimizer_updates_executed': 0,
        'training_or_evaluation_started': False,
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    }
    (evidence / f'{version}-optimizer-parameter-map.json').write_text(
        json.dumps(mapping, indent=2) + '\n')
    (evidence / f'{version}-report.json').write_text(json.dumps(report, indent=2) + '\n')
    (evidence / f'{version}-cpu-forward.json').write_text(json.dumps(forward, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=Path(
        'experiments/20261108-eric-v7-continuous/historical-adaptation'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs/imported/eric-local-20261108-adapted'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    status = evidence / 'status.json'
    status.write_text(json.dumps({'status': 'cpu_adapting_v1_v2', 'gpu_computation': False,
                                  'optimizer_updates_executed': 0}, indent=2) + '\n')
    try:
        source_audit = helpers.verify_source_contract(
            root, root / 'experiments/20261108-eric-3702-weight-migration')
        fresh = root / 'runs/20261108-eric-3702-fresh-v1-init/initial_model.zip'
        initialization = root / 'experiments/20261108-eric-3702-fresh-v1/initialization/initialization.json'
        fresh_sha = json.loads(initialization.read_text())['sha256']
        helpers.require(helpers.sha(fresh) == fresh_sha, 'Current asset initialization changed')
        current_data, current_parameters, _ = load_from_zip_file(fresh, device='cpu')
        helpers.require(current_data['policy_kwargs']['card_data_sha256'] == helpers.CURRENT_CARDS_SHA,
                        'Current asset card SHA differs')
        reports = [migrate_version(root, evidence, args.output_dir.resolve(), version,
                                   current_data, current_parameters, fresh_sha, source_audit)
                   for version in SOURCES]
    except BaseException as error:
        status.write_text(json.dumps({'status': 'failed_preserved',
                                      'error': f'{type(error).__name__}: {error}',
                                      'gpu_computation': False}, indent=2) + '\n')
        raise
    summary = {'status': 'v1_v2_cpu_adaptation_verified', 'candidates': reports,
               'gpu_computation': False, 'optimizer_updates_executed': 0,
               'training_or_evaluation_started': False}
    (evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    status.write_text(json.dumps({'status': summary['status'], 'gpu_computation': False,
                                  'next': 'orchestrator may bind frozen source protocol explicitly',
                                  'artifacts': [{key: report[key] for key in
                                                 ('version', 'adapted_checkpoint', 'adapted_sha256')}
                                                for report in reports]}, indent=2) + '\n')
    print(json.dumps(status.read_text()))


if __name__ == '__main__':
    main()
