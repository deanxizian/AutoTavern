"""Detach one bounded local training stage; never automatically restart it."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
STATUS = Path('/root/autodl-tmp/bootstrap/eric-3702-status')
RUN = ROOT / 'runs/20261108-eric-3702-fresh-v1'


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_progress():
    for name in ('training.json', 'progress.json'):
        path = EXP / name
        if path.exists():
            return json.loads(path.read_text())
    return {}


def terminate(process, grace=120):
    """Never leave a live training child behind when the guardian exits."""
    if process.poll() is None:
        process.terminate()
        try:
            return process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.wait(timeout=10)


def main():
    STATUS.mkdir(exist_ok=True)
    with (STATUS / 'training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gates = json.loads((STATUS / 'validation-gates.json').read_text())
        if gates.get('status') != 'passed':
            raise RuntimeError('Training requires completed validation gates')
        config_bytes = (EXP / 'config.json').read_bytes()
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        if gates.get('config_sha256') != config_sha:
            raise RuntimeError('Validation gates do not match the frozen formal configuration')
        git_head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if gates.get('code_commit') != git_head:
            raise RuntimeError('Validation gates do not match the committed checkout')
        for relative, digest in gates.get('runtime_files', {}).items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Runtime changed after validation: {relative}')
        config = json.loads(config_bytes)
        source = ROOT / config['initial_model']
        if hashlib.sha256(source.read_bytes()).hexdigest() != config['initial_model_sha256']:
            raise RuntimeError('Initialization hash mismatch')
        if RUN.exists():
            raise FileExistsError(
                'Refuse to replace or automatically resume an existing training run')
        frozen = STATUS / f'formal-config-{config_sha}.json'
        if frozen.exists():
            if frozen.read_bytes() != config_bytes:
                raise RuntimeError('Frozen configuration contents changed')
        else:
            with frozen.open('xb') as stream:
                stream.write(config_bytes)
            frozen.chmod(0o400)
        started = time.monotonic()
        started_at = stamp()
        deadline = started + config['max_wall_seconds']
        environment = os.environ.copy()
        environment.update(WANDB_MODE='offline', WANDB_DISABLED='false',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           RAYON_NUM_THREADS='20', PYTHONUNBUFFERED='1',
                           TOKENIZERS_PARALLELISM='false')
        # Offline training must not depend on the SSH proxy or inherited service credentials.
        for key in list(environment):
            if key.lower().endswith('_proxy') or key in ('WANDB_API_KEY', 'WANDB_ENTITY',
                                                        'WEAVE_PROJECT', 'WANDB_PROJECT'):
                environment.pop(key)
        command = [sys.executable, '-m', 'taverngym.learning.experiment', '--config',
                   str(frozen), '--model', str(source), '--experiment-dir',
                   str(EXP), '--run-dir', str(RUN), '--device', 'cuda']
        stop_reason = None
        request_at = None

        def stop(signum, frame):
            nonlocal stop_reason
            if stop_reason is None:
                stop_reason = f'user_signal_{signal.Signals(signum).name}'

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        process = None
        resources = None
        console = None
        state = {'instance_id': config['instance_id'], 'training_started': False,
                 'used_samples': 0, 'completed_ppo': 0, 'adam_updates': 0,
                 'config': str(frozen), 'config_sha256': config_sha,
                 'experiment_dir': str(EXP), 'log': str(EXP / 'training-console.log')}
        guardian_error = None
        try:
            if stop_reason is not None or (STATUS / 'STOP').exists():
                raise InterruptedError('Stop requested before training launch')
            console = (EXP / 'training-console.log').open('ab', buffering=0)
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=console,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            # start_new_session makes the child its own process-group leader. Using
            # its PID also avoids getpgid racing an immediately failed child.
            launch = {'instance_id': config['instance_id'], 'started_at': started_at,
                      'guardian_pid': os.getpid(), 'training_pid': process.pid,
                      'checkout': str(ROOT), 'git_head': git_head,
                      'training_pgid': process.pid, 'guardian_pgid': os.getpgrp(),
                      'stdin': '/dev/null', 'max_wall_seconds': config['max_wall_seconds'],
                      'command': command, 'config': str(frozen), 'config_sha256': config_sha,
                      'automatic_restarts': 0, 'max_diagnosed_manual_environment_resumes': 1}
            state.update(launch, training_started=True)
            write(STATUS / 'training-launch.json', launch)
            resources = (STATUS / 'resources.jsonl').open('a', buffering=1)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                free = shutil.disk_usage(ROOT).free
                if not stop_reason and time.monotonic() >= deadline:
                    stop_reason = 'six_hour_wall_budget'
                if not stop_reason and (STATUS / 'STOP').exists():
                    stop_reason = 'user_stop_file'
                if not stop_reason and free < 2 * 1024 ** 3:
                    stop_reason = 'disk_free_below_2GiB'
                if stop_reason and request_at is None:
                    request_at = time.monotonic()
                    process.send_signal(signal.SIGTERM)
                if request_at is not None and time.monotonic() - request_at > 120:
                    process.kill()
                    stop_reason += '_forced_after_120s_grace'
                    break
                progress = read_progress()
                state.update(stage=progress.get('phase', 'startup'), updated_at=stamp(),
                             training_alive=True, elapsed_seconds=elapsed,
                             remaining_wall_seconds=max(0, deadline - time.monotonic()),
                             used_samples=progress.get('sampled_decisions', 0),
                             completed_ppo=progress.get('completed_iterations', 0),
                             adam_updates=progress.get('optimizer_steps', 0),
                             latest_complete_checkpoint=str(RUN / 'latest_model.zip')
                             if (RUN / 'latest_model.zip').exists() else None,
                             stop_reason=stop_reason)
                write(STATUS / 'controller-status.json', state)
                try:
                    gpu = subprocess.run([
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                        '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=10)
                    resource = {
                        'at': stamp(), 'disk_free_bytes': free, 'loadavg': os.getloadavg(),
                        'gpu_csv': gpu.stdout.strip(), 'gpu_returncode': gpu.returncode,
                        'memory_cgroup_bytes':
                            Path('/sys/fs/cgroup/memory.current').read_text().strip(),
                        'cpu_cgroup_stat': Path('/sys/fs/cgroup/cpu.stat').read_text(),
                        'cpu_cgroup_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                        'training_proc_stat': Path(f'/proc/{process.pid}/stat').read_text()}
                    resources.write(json.dumps(resource) + '\n')
                except (OSError, subprocess.TimeoutExpired) as error:
                    resources.write(
                        json.dumps({'at': stamp(), 'telemetry_error': str(error)}) + '\n')
                # Bound the wait by the budget deadline; telemetry may delay a
                # stop request by at most its ten-second subprocess timeout.
                process_wait = min(15, max(0.05, deadline - time.monotonic()))
                time.sleep(process_wait)
            code = process.wait(timeout=10)
        except BaseException as error:
            guardian_error = f'{type(error).__name__}: {error}'
            stop_reason = stop_reason or 'guardian_exception'
            raise
        finally:
            # Includes telemetry/status I/O exceptions, signal interruptions and
            # failures immediately after Popen. Cleanup precedes further disk I/O.
            if process is not None:
                code = terminate(process)
            else:
                code = None
            if resources is not None:
                resources.close()
            if console is not None:
                console.close()
            try:
                progress = read_progress()
                completed = (process is not None and code == 0 and not stop_reason
                             and progress.get('status') == 'completed'
                             and progress.get('completed_iterations') == config['iterations']
                             and progress.get('sampled_decisions') == config['max_samples']
                             and (EXP / 'evaluation-best-heldout.json').exists())
                final = {**state, 'updated_at': stamp(), 'stage': 'ended',
                         'training_alive': False,
                         'used_samples': progress.get('sampled_decisions', state['used_samples']),
                         'completed_ppo': progress.get(
                             'completed_iterations', state['completed_ppo']),
                         'adam_updates': progress.get('optimizer_steps', state['adam_updates']),
                         'latest_complete_checkpoint': str(RUN / 'latest_model.zip')
                         if (RUN / 'latest_model.zip').exists() else None,
                         'training_status': progress.get('status'),
                         'exit_code': code, 'stop_reason': stop_reason,
                         'status': 'completed' if completed else 'paused',
                         'guardian_error': guardian_error, 'automatic_restarts': 0,
                         'elapsed_seconds': time.monotonic() - started}
                write(STATUS / 'controller-status.json', final)
                resolved = EXP / 'resolved-config.json'
                if resolved.exists():
                    from taverngym.learning.overview import update_stage_overview
                    update_stage_overview(
                        EXP, json.loads(resolved.read_text()), completed=completed)
            except BaseException:
                # The training child is already stopped even if evidence cannot
                # be written (for example, an exhausted or failed disk).
                if guardian_error is None:
                    raise
        return code


if __name__ == '__main__':
    raise SystemExit(main())
