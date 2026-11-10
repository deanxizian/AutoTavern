"""Run existing, targeted Rust tests for the paper's interface claims."""
from pathlib import Path
import datetime
import json
import os
import re
import shutil
import subprocess

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
SOURCE_DIR = OUT
if (OUT / 'interface-results.json').exists():
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    OUT = OUT / 'reruns' / ('interfaces-' + stamp)
    OUT.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
cargo = shutil.which('cargo')
if cargo is None:
    cargo = '/tmp/autotavern-cargo/bin/cargo'
    if not Path(cargo).exists():
        raise SystemExit('Install the Rust toolchain pinned in rust-toolchain.toml first.')
    env.update(RUSTUP_HOME='/tmp/autotavern-rustup', CARGO_HOME='/tmp/autotavern-cargo',
               RUSTUP_TOOLCHAIN='1.98.1', PATH='/tmp/autotavern-cargo/bin:' + env['PATH'])

groups = [
    ('lib', ['--lib'], 'Actions, RNG, batched stepping, thread-independent results and exact batch save/load'),
    ('games', ['--test', 'games'], 'Completed bot games, deterministic replay, snapshot roundtrip and random legal play'),
    ('legal-mask', ['--test', 'mechanics', 'every_masked_legal_action_applies'], 'Sampled masked-legal actions are accepted by the engine'),
    ('observation', ['--test', 'mechanics', 'observation_'], 'Unseen opponent boards are hidden and previously fought boards are remembered'),
    ('scoreboard', ['--test', 'scoreboard'], 'Public scoreboard does not reveal live opponent cards'),
]
records = []
for name, args, purpose in groups:
    command = [cargo, '+1.98.1', 'test', '--release', '--locked', '--offline', '-p', 'taverngym',
               *args, '--', '--test-threads=1']
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    filename = f'tests-{name}.txt'
    (OUT / filename).write_text(completed.stdout)
    found = re.search(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;', completed.stdout)
    record = {'group': name, 'purpose': purpose, 'command': command, 'output': filename,
              'returncode': completed.returncode,
              'passed': int(found[2]) if found else None,
              'failed': int(found[3]) if found else None,
              'ignored': int(found[4]) if found else None}
    records.append(record)
    summary = {'created_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               'groups': records,
               'total_passed': sum(x['passed'] or 0 for x in records),
               'total_failed': sum(x['failed'] or 0 for x in records),
               'unit_tests_passed': len(records) == len(groups) and all(x['returncode'] == 0 and x['passed'] for x in records),
               'all_passed': False}
    (OUT / 'interface-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f"{name}: {record['passed']} passed; {record['failed']} failed (exit {completed.returncode})", flush=True)
    if completed.returncode or not found:
        raise SystemExit(f'Verification failed; inspect {OUT / filename}')

print('Total:', summary['total_passed'], 'passed')

# Existing snapshot tests are supplemented with an actively advanced mid-game,
# subsequent episode transitions, and validation of every benchmark seed.
rustc = shutil.which('rustc', path=env['PATH'])
assert rustc, 'rustc was not found'
deps = ROOT / 'target/release/deps'
engine = list(deps.glob('libtaverngym-*.rlib'))
serde = list(deps.glob('libserde_json-*.rlib'))
assert len(engine) == len(serde) == 1, 'Ambiguous compiled libraries; inspect target/release/deps'
binary = ROOT / 'target/paper-verification'
build = [rustc, '+1.98.1', '--edition=2024', '-O',
         '-L', 'dependency=' + str(deps), '--extern', 'taverngym=' + str(engine[0]),
         '--extern', 'serde_json=' + str(serde[0]), str(SOURCE_DIR / 'verify_trajectories.rs'),
         '-o', str(binary)]
compiled = subprocess.run(build, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(OUT / 'trajectory-build.txt').write_text(compiled.stdout)
assert compiled.returncode == 0, compiled.stdout
checked = subprocess.run([str(binary)], cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(OUT / 'trajectory-verification.json').write_text(checked.stdout)
(OUT / 'trajectory-stderr.txt').write_text(checked.stderr)
summary['trajectory_build_command'] = build
summary['trajectory_command'] = [str(binary)]
summary['trajectory_returncode'] = checked.returncode
summary['trajectory_result'] = 'trajectory-verification.json'
summary['all_passed'] = summary['unit_tests_passed'] and checked.returncode == 0
(OUT / 'interface-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
assert checked.returncode == 0, checked.stderr
print(checked.stdout)
print('Results saved to:', OUT)
