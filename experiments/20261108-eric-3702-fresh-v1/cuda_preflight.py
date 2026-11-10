"""One isolated 8192-sample CUDA update, excluded from the formal stage budget."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file

import taverngym as tg
from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
STATUS = Path('/root/autodl-tmp/bootstrap/eric-3702-status')


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = json.loads((EXP / 'config.json').read_text())
    source = ROOT / config['initial_model']
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == config['initial_model_sha256']
    assert (tg.N_CARDS, tg.N_ACTIONS, tg.OBS_FLOATS, tg.OBS_CARDS) == (3984, 245, 3227, 124)
    assert tg.game_version() == '37.0.2.256589'
    assert tg.server_rules_version() == '37.0.2'
    assert tg.build_is_current()
    env = NativeVectorEnv(32, seed=37024001, threads=20,
                          config=config['simulator_config'], reward_shaping=config['reward_shaping'])
    model = EntityPPO.load(source, env=env, device='cuda', n_steps=256, batch_size=8192,
                           micro_batch_size=1024, n_epochs=1, seed=37024001,
                           rollout_buffer_class=CompactPlacementBuffer, rollout_buffer_kwargs={})
    model.set_forward_precision('bf16')
    model.set_logger(configure(format_strings=[]))
    assert model.num_timesteps == model.optimizer_step_count() == 0
    observations = env.reset()
    masks = observations['action_mask']
    assert np.all(masks.sum(axis=1) == 2), 'Expected exactly two native hero choices'
    chosen, _ = model.predict(observations, deterministic=False)
    assert np.all(masks[np.arange(32), chosen.astype(int)])
    updates = []

    def updated(current, metrics, step):
        assert all(np.isfinite(v) for v in metrics.values())
        updates.append({'step': step, **metrics})
        assert all(p.dtype == torch.float32 for p in current.policy.parameters())
        assert all(p.grad is None or p.grad.dtype == torch.float32
                   for p in current.policy.parameters())
        for state in current.policy.optimizer.state.values():
            for name in ('exp_avg', 'exp_avg_sq'):
                assert state[name].dtype == torch.float32 and torch.isfinite(state[name]).all()

    model._after_optimizer_step = updated
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.learn(8192, log_interval=None)
    assert model.num_timesteps == 8192
    assert model.optimizer_step_count() == 1 and len(updates) == 1
    assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
    output = ROOT / 'runs/20261108-eric-3702-fresh-v1-preflight'
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'preflight-complete.zip'
    model.save(path)
    data, parameters, _ = load_from_zip_file(path, device='cpu')
    assert data['num_timesteps'] == 8192 and parameters['policy.optimizer']['state']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    result = {'status': 'passed', 'seed': 37024001, 'formal_budget_samples': 0,
              'preflight_samples':8192, 'preflight_adam_updates':1, 'micro_batch_size':1024,
              'parameter_dtype':'float32', 'gradient_dtype':'float32', 'adam_dtype':'float32',
              'forward_precision':'bf16', 'elapsed_seconds':time.monotonic()-started,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(), 'updates':updates,
              'initialization_unchanged_sha256':source_sha, 'checkpoint':str(path),
              'build_info':tg.build_info(), 'hero_choices':2}
    (STATUS / 'cuda-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    model.rollout_buffer.close()
    env.close()


if __name__ == '__main__':
    main()
