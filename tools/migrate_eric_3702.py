"""Strict, CPU-only V7 weight migration; this tool cannot start training."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import zipfile

import torch
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

import taverngym as tg
from taverngym.learning.model import EntityActorCritic, ModelConfig
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.text import card_manifest

SOURCE_SHA = 'cdb10c8bb08587b52dd2100ba71302920c549561747f47a7eb1e72fa7333f354'
OLD_CARDS_SHA = '429305fb9dd53e6c5240bf5219a747602e96d7a0e4bbd19a0d65b8b044ed6168'
CURRENT_CARDS_SHA = '7a02d10794cf80c56fa77b1dd235d49b892c8d3e31c2cf56887f696e2cf922a6'
SOURCE_COMMIT = '6825a87aec76e4bad9964c0387ba6352335ad352'
REFRESHED_BUFFERS = {'network.static_features', 'network.text_features'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    result = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            result.update(f'{item.dtype}:{tuple(item.shape)}:'.encode())
            result.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for entry in item:
                add(entry)
        else:
            result.update(repr(item).encode())

    add(value)
    return result.hexdigest()


def audit_optimizer(policy, optimizer, parameter_names, expected_step):
    """Bind source registration order to saved Adam slots, rejecting ambiguity."""
    require([key for key in policy if key in parameter_names] == parameter_names,
            'Source policy parameter registration order changed')
    groups = optimizer['param_groups']
    require(len(groups) == 1, 'Expected one source Adam parameter group')
    indices = groups[0]['params']
    require(indices == list(range(len(parameter_names))), 'Unexpected Adam index order')
    require(set(optimizer['state']) == set(indices), 'Incomplete Adam parameter state')
    mapping = []
    for index, name in zip(indices, parameter_names, strict=True):
        parameter = policy[name]
        require(parameter.dtype == torch.float32 and torch.isfinite(parameter).all(),
                f'Invalid FP32 learned parameter: {name}')
        state = optimizer['state'][index]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected Adam state fields')
        step = state['step'].item() if torch.is_tensor(state['step']) else state['step']
        require(step == expected_step, f'Unexpected Adam clock for {name}')
        moments = {}
        for field in ('exp_avg', 'exp_avg_sq'):
            value = state[field]
            require(value.shape == parameter.shape and value.dtype == torch.float32,
                    f'Adam shape/dtype mismatch: {name}/{field}')
            require(torch.isfinite(value).all(), f'Nonfinite Adam moment: {name}/{field}')
            moments[field + '_sha256'] = fingerprint(value)
        mapping.append({'index': index, 'parameter': name, 'shape': list(parameter.shape),
                        'dtype': str(parameter.dtype), 'adam_step': expected_step, **moments})
    return mapping


def refresh_policy(source, current, parameter_names):
    require(list(source) == list(current), 'Policy state names or ordering differ')
    require(all(source[k].shape == current[k].shape for k in source), 'Policy shapes differ')
    differences = {key for key in source if not torch.equal(source[key], current[key])
                   and key not in parameter_names}
    require(differences == REFRESHED_BUFFERS, 'Unexpected non-learned buffer differences')
    adapted = copy.deepcopy(source)
    for key in REFRESHED_BUFFERS:
        adapted[key] = current[key].clone()
    require(all(torch.equal(adapted[k], source[k]) for k in parameter_names),
            'Migration changed a learned parameter')
    require(all(torch.equal(adapted[k], source[k]) for k in source
                if k not in REFRESHED_BUFFERS), 'Migration changed an unrelated buffer')
    return adapted


def verify_source_contract(root, evidence):
    rows = json.loads((evidence / 'source-code-audit.json').read_text())
    required = {'python/taverngym/learning/model.py', 'python/taverngym/learning/policy.py',
                'crates/taverngym/src/actions.rs', 'crates/taverngym/src/obs.rs'}
    require({row['path'] for row in rows} == required, 'Source-code audit is incomplete')
    for row in rows:
        require(SOURCE_COMMIT in row['source_url'] and row['exact_match'],
                'Source-code audit provenance differs')
        require(sha(root / row['path']) == row['source_sha256'],
                f"Current source changed: {row['path']}")
        require(sha(evidence / 'source-evidence' / Path(row['path']).name)
                == row['source_sha256'], 'Source evidence bytes changed')
    old_raw = subprocess.check_output([
        'git', 'show',
        '719a3491eb0b95c16f5d028d549d944cbc7cd286:crates/taverngym/data/cards.json'], cwd=root)
    require(hashlib.sha256(old_raw).hexdigest() == OLD_CARDS_SHA, 'Old card data differs')
    old_ids = [card['id'] for card in json.loads(old_raw)['cards']]
    ids, _, digest = card_manifest()
    require(digest == CURRENT_CARDS_SHA and old_ids == ids[1:] and len(old_ids) == 3983,
            'Card ID order/current rules differ')
    require((tg.N_CARDS, tg.N_ACTIONS, tg.OBS_FLOATS, tg.OBS_CARDS)
            == (3984, 245, 3227, 124), 'Native observation/action contract differs')
    require(tg.build_is_current() and tg.server_rules_version() == '37.0.2',
            'Native engine is stale or wrong rules version')
    return rows


def cpu_forward(model):
    """Small visible-observation check, not a full game/development evaluation."""
    cases = []
    with torch.no_grad():
        for seed in (37025011, 37025012):
            env = tg.BattlegroundsEnv(seed=seed, config={'hero_choices': 2,
                                                        'turn_action_caps': [60]})
            observation, _ = env.reset(seed=seed)
            for index in range(8):
                tensors, _ = model.policy.obs_to_tensor(observation)
                output = model.policy._network_output(tensors)
                require(all(torch.isfinite(value).all() for value in output),
                        'Nonfinite CPU forward')
                action, _ = model.predict(observation, deterministic=bool(seed % 2))
                action = int(action)
                require(bool(observation['action_mask'][action]), 'CPU action is illegal')
                cases.append({'seed': seed, 'step': index, 'action': action,
                              'value': float(output.values.reshape(-1)[0])})
                observation, _, terminated, truncated, _ = env.step(action)
                require(not terminated and not truncated, 'Unexpected early terminal in smoke')
    return cases


def migrate(root, evidence, output):
    torch.set_num_threads(2)
    source = root / 'runs/imported/eric-local-20261108/20261101-115350-best_model.zip'
    fresh = root / 'runs/20261108-eric-3702-fresh-v1-init/initial_model.zip'
    require(sha(source) == SOURCE_SHA, 'Source V7 ZIP SHA mismatch')
    with zipfile.ZipFile(source) as archive:
        require(archive.testzip() is None, 'Source ZIP CRC mismatch')
    source_rows = verify_source_contract(root, evidence)
    assets = json.loads((root / 'experiments/20261108-eric-3702-fresh-v1/assets-summary.json')
                        .read_text())
    fresh_sha = json.loads((root / 'experiments/20261108-eric-3702-fresh-v1/initialization/'
                           'initialization.json').read_text())['sha256']
    require(sha(fresh) == fresh_sha, 'Current asset initialization changed')
    data, parameters, variables = load_from_zip_file(source, device='cpu')
    current_data, current_parameters, _ = load_from_zip_file(fresh, device='cpu')
    require(data['policy_kwargs']['card_data_sha256'] == OLD_CARDS_SHA, 'Source card SHA differs')
    require(current_data['policy_kwargs']['card_data_sha256'] == CURRENT_CARDS_SHA,
            'Current asset card SHA differs')
    require(data['policy_kwargs']['model_config'] == current_data['policy_kwargs']['model_config'],
            'Model architectures differ')
    network = EntityActorCritic(ModelConfig(**data['policy_kwargs']['model_config']))
    parameter_names = ['network.' + name for name, _ in network.named_parameters()]
    require(len(parameter_names) == 91, 'Unexpected number of learned parameters')
    source_policy, source_adam = parameters['policy'], parameters['policy.optimizer']
    mapping = audit_optimizer(source_policy, source_adam, parameter_names, 8800)
    learned_source = {key: source_policy[key] for key in parameter_names}
    fingerprints = {'source_learned': fingerprint(learned_source),
                    'source_optimizer_full': fingerprint(source_adam),
                    'source_optimizer_moments': fingerprint(source_adam['state']),
                    'source_policy_all': fingerprint(source_policy)}
    # The original archive must still be rejected by the normal vocabulary guard.
    try:
        EntityPPO.load(source, device='cpu', n_envs=1, n_steps=2, batch_size=2,
                       rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    except ValueError as error:
        require('Checkpoint card vocabulary differs' in str(error), 'Unexpected source rejection')
    else:
        raise ValueError('The original incompatible archive bypassed the vocabulary guard')
    adapted_policy = refresh_policy(source_policy, current_parameters['policy'], parameter_names)
    require(fingerprint(network.static_features)
            == fingerprint(adapted_policy['network.static_features']), 'Current native static differs')
    text_asset = Path(assets['text_asset'])
    require(sha(text_asset) == assets['text_asset_sha256'], 'Current text asset SHA differs')
    text = torch.load(text_asset, map_location='cpu', weights_only=True)
    require(torch.equal(text['embeddings'], adapted_policy['network.text_features']),
            'Current text vectors differ from the migrated buffers')
    source_history = {'source_checkpoint': str(source), 'source_sha256': SOURCE_SHA,
                      'source_commit': SOURCE_COMMIT, 'source_ppo_iteration': data['_n_updates'],
                      'source_samples': data['num_timesteps'], 'source_adam_total': 8800,
                      'source_stage_adam_offset': data['_optimizer_step_offset'],
                      'source_stage_adam_updates': 8800 - data['_optimizer_step_offset'],
                      'source_card_data_sha256': OLD_CARDS_SHA}
    removed_config_keys = sorted(data['experiment_config'])
    source_lr = float(source_adam['param_groups'][0]['lr'])
    data['policy_kwargs'] = {**data['policy_kwargs'], 'card_data_sha256': CURRENT_CARDS_SHA}
    data['rule_text_metadata'] = copy.deepcopy(current_data['rule_text_metadata'])
    data['experiment_config'] = {
        'name': 'eric-3702-v7-adapted-inspection', 'migration_only': True,
        'training_start_authorized': False, 'requires_explicit_new_stage_config': True,
        'source_history': source_history, 'current_card_data_sha256': CURRENT_CARDS_SHA,
        'asset_initialization_sha256': fresh_sha, 'forward_precision': data['policy_kwargs']
        ['forward_precision'], 'source_optimizer_lr_preserved_not_stage_lr_choice': source_lr,
        'inspection_rollout_dimensions': {'n_envs': 1, 'n_steps': 2, 'batch_size': 2},
    }
    data.update(num_timesteps=0, _n_updates=0, _total_timesteps=0, _num_timesteps_at_start=0,
                _optimizer_step_offset=8800, _episode_num=0, _current_progress_remaining=1.0,
                _last_obs=None, _last_original_obs=None, _last_episode_starts=None,
                ep_info_buffer=None, ep_success_buffer=None, start_time=0,
                n_envs=1, n_steps=2, batch_size=2, n_epochs=1, micro_batch_size=1024,
                rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={},
                tensorboard_log=None, verbose=0, seed=37025001, learning_rate=source_lr,
                last_rollout_seconds=0.0, last_train_seconds=0.0)
    data.pop('lr_schedule', None)  # Remove source schedule objects, retaining archived group LR only.
    require(not output.exists(), 'Refusing to overwrite an existing adaptation directory')
    output.mkdir(parents=True)
    staged = output / 'v7-3702-staged.zip'
    save_to_zip_file(staged, data=data,
                     params={**parameters, 'policy': adapted_policy}, pytorch_variables=variables)
    model = EntityPPO.load(staged, device='cpu')  # Tiny dimensions are already sealed in metadata.
    require(model.rollout_buffer.buffer_size == 2 and model.n_envs == 1,
            'Standard load allocated historical rollout dimensions')
    require(fingerprint(model.policy.optimizer.state_dict()) == fingerprints['source_optimizer_full'],
            'Standard load changed Adam state/group mapping')
    model.start_new_stage(preserve_adam=True)
    model._last_episode_starts = None
    require(model.optimizer_step_count() == 0 and model.optimizer_total_step_count() == 8800,
            'Stage-local and cumulative Adam counters were mixed')
    adapted = output / 'eric-v7-3702-adapted.zip'
    model.save(adapted)
    restored = EntityPPO.load(adapted, device='cpu')
    state = restored.policy.state_dict()
    require(all(torch.equal(state[name], source_policy[name]) for name in parameter_names),
            'Standard save/reload changed a learned parameter')
    require(all(torch.equal(state[name], adapted_policy[name]) for name in adapted_policy),
            'Standard save/reload changed a migrated buffer')
    require(fingerprint(restored.policy.optimizer.state_dict()) == fingerprints['source_optimizer_full'],
            'Standard save/reload changed complete Adam state')
    require(restored.num_timesteps == 0 and restored._n_updates == 0
            and restored.optimizer_step_count() == 0 and restored.optimizer_total_step_count() == 8800,
            'Adapted checkpoint is not a fresh local stage with inherited Adam')
    require(restored.policy.card_data_sha256 == CURRENT_CARDS_SHA, 'Vocabulary metadata differs')
    forward = cpu_forward(restored)
    require(fingerprint(restored.policy.optimizer.state_dict()) == fingerprints['source_optimizer_full'],
            'CPU forward unexpectedly changed Adam')
    require(sha(source) == SOURCE_SHA, 'Original archive changed during migration')
    fingerprints.update(adapted_learned=fingerprint({k: state[k] for k in parameter_names}),
                        adapted_optimizer_full=fingerprint(restored.policy.optimizer.state_dict()),
                        adapted_policy_all=fingerprint(state),
                        adapted_buffers={k: fingerprint(state[k]) for k in state
                                         if k not in parameter_names})
    report = {'status': 'cpu_migration_verified_not_selected_for_training',
              'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'source': source_history, 'adapted_checkpoint': str(adapted),
              'adapted_sha256': sha(adapted), 'staged_checkpoint': str(staged),
              'current_card_data_sha256': CURRENT_CARDS_SHA,
              'fresh_asset_checkpoint_sha256': fresh_sha, 'text_asset_sha256': sha(text_asset),
              'source_code_exact_matches': source_rows,
              'refreshed_buffers_only': sorted(REFRESHED_BUFFERS),
              'learned_parameters_exact': True, 'adam_full_exact': True,
              'original_vocab_guard_rejection_verified': True,
              'saved_local_samples': 0, 'saved_local_adam_updates': 0,
              'cumulative_adam_steps': 8800, 'parameter_mapping_count': len(mapping),
              'fingerprints': fingerprints, 'cpu_forward_actions': forward,
              'old_experiment_config_keys_removed': removed_config_keys,
              'source_optimizer_lr_preserved_not_stage_lr_choice': source_lr,
              'metadata_changes': ['current card SHA and rule-text provenance',
                  'new-stage counters0 and cumulative Adam offset8800',
                  'inspection-only1x2 rollout metadata; never an authorized training budget',
                  'old reward/opponent/evaluation/tracking/stage-budget config fully removed',
                  'source RNG/observations/episode history/schedule object removed'],
              'gpu_computation': False, 'optimizer_updates_executed': 0,
              'development_or_final_games_executed': 0, 'fresh_training_touched': False,
              'training_authorized': False}
    (evidence / 'optimizer-parameter-map.json').write_text(json.dumps(mapping, indent=2) + '\n')
    (evidence / 'migration-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path,
                        default=Path('experiments/20261108-eric-3702-weight-migration'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs/imported/eric-local-20261108-adapted'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = args.evidence_dir.resolve()
    status = evidence / 'status.json'
    try:
        report = migrate(root, evidence, args.output_dir.resolve())
    except BaseException as error:
        status.write_text(json.dumps({'status': 'failed_preserved_no_training_change',
                                      'error': f'{type(error).__name__}: {error}',
                                      'gpu_computation': False, 'fresh_training_touched': False},
                                     indent=2) + '\n')
        raise
    status.write_text(json.dumps({'status': report['status'],
                                  'adapted_checkpoint': report['adapted_checkpoint'],
                                  'report': str(evidence / 'migration-report.json'),
                                  'gpu_computation': False, 'fresh_training_touched': False,
                                  'next': 'await source choice; no training switch authorized'},
                                 indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'status', 'adapted_checkpoint', 'adapted_sha256', 'learned_parameters_exact',
        'adam_full_exact', 'saved_local_samples', 'saved_local_adam_updates',
        'cumulative_adam_steps', 'parameter_mapping_count', 'gpu_computation')}, indent=2))


if __name__ == '__main__':
    main()
