"""Resume the completed W&B run to log evaluation performance only."""
import json
from pathlib import Path
import wandb

folder = Path(__file__).resolve().parent
result = json.loads((folder / 'results.json').read_text())
run = wandb.init(entity='eric_eric_', project='AutoTavern', id=json.loads((folder / 'wandb.json').read_text())['run_id'], resume='must', dir=str(folder))
run.define_metric('evaluated_lobbies')
run.define_metric('test/*', step_metric='evaluated_lobbies')
keys = ('elapsed_seconds', 'games_per_second', 'inference_and_observe_seconds',
        'native_step_seconds', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes',
        'actual_initial_concurrency', 'global_rounds', 'deferred_actions', 'rejected_actions')
payload = {'evaluated_lobbies': result['completed_lobbies']}
payload.update({f'test/performance_{key}': result[key] for key in keys})
run.log(payload)
run.finish()
(folder / 'performance-sync.json').write_text(json.dumps({'status':'synced', 'run_url':run.url, 'metrics':payload}, indent=2)+'\n')
