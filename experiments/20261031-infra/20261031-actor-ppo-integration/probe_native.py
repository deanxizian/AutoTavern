"""Full rollout budget, inference only, preserving native failure diagnostics."""
import json
from pathlib import Path
import time
import traceback
import torch
from taverngym.learning.actor_sampling import PolicyActor, UnifiedActorSampler
from taverngym.learning.experiment import load_frozen_policy, write_json, now

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / 'pyproject.toml').exists())
SOURCE = ROOT / 'runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip'

def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    learner, fixed = [PolicyActor(load_frozen_policy(SOURCE, 'cuda', forward_precision='bf16')) for _ in range(2)]
    sampler = UnifiedActorSampler(256, learner, fixed, frozen_opponents=2, seed=43, threads=4,
        config={'turn_action_caps': [60]}, reward_shaping={'operation_cost': .0001, 'cycle_cost': .0002})
    seen, last = [0], [0.0]
    def record(t):
        seen[0] += 1
        if time.monotonic() - last[0] > 5:
            last[0] = time.monotonic()
            print('learner_samples', seen[0], flush=True)
    torch.manual_seed(1043)
    try:
        result = sampler.collect(1024, record)
        write_json(FOLDER / 'native-rollout.json', {'status': 'passed', 'at': now(), **result})
        print(json.dumps(result), flush=True)
    except Exception:
        failure = sampler.last_failure
        if failure:
            failure['retry_errors'] = []
            for slot, action in zip(failure['rejected_slots'], failure['actions']):
                game = sampler.batch.game(slot // sampler.roles)
                seat = int(sampler.batch.slot_seats[slot])
                try:
                    game.step(seat, action)
                    error = None
                except Exception as exc:
                    error = str(exc)
                failure['retry_errors'].append({'slot': slot, 'seat': seat, 'error': error})
        write_json(FOLDER / 'native-failure.json', {'status': 'failed', 'at': now(),
            'samples': seen[0], 'traceback': traceback.format_exc(), 'failure': failure})
        raise

if __name__ == '__main__':
    main()
