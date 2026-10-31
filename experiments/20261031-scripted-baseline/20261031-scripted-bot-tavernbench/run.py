"""Generate sealed baseline answers without labels, then invoke official scoring.

Run from anywhere: python experiments/20261031-scripted-baseline/20261031-scripted-bot-tavernbench/run.py
Existing results are retained; repeated runs go into reruns/.
"""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT = SOURCE
if (OUT / 'results.json').exists() or (OUT / 'inference.jsonl').exists():
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    OUT = SOURCE / 'reruns' / stamp
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'tavernbench/data/eval'
env = os.environ.copy()
cargo = shutil.which('cargo')
if cargo is None:
    cargo = '/tmp/autotavern-cargo/bin/cargo'
    if not Path(cargo).exists():
        raise SystemExit('Install the Rust toolchain pinned in rust-toolchain.toml first.')
    env.update(RUSTUP_HOME='/tmp/autotavern-rustup', CARGO_HOME='/tmp/autotavern-cargo',
               RUSTUP_TOOLCHAIN='1.98.1', PATH='/tmp/autotavern-cargo/bin:' + env['PATH'])
rustc = shutil.which('rustc', path=env['PATH'])
assert rustc

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(command, filename):
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = OUT / filename
    attempt = 1
    while output.exists():
        output = OUT / f'{Path(filename).stem}.attempt-{attempt}{Path(filename).suffix}'
        attempt += 1
    output.write_text(result.stdout)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(f'Command failed: {command}; inspect {output}')
    return result

execute([cargo, '+1.98.1', 'build', '--release', '--locked', '--offline', '-p', 'taverngym', '--lib'], 'engine-build.txt')
deps = ROOT / 'target/release/deps'
libraries = {}
for name in ['taverngym', 'serde_json']:
    found = list(deps.glob(f'lib{name}-*.rlib'))
    assert len(found) == 1, f'Ambiguous release library for {name}'
    libraries[name] = found[0]
common = [rustc, '+1.98.1', '--edition=2024', '-O', '-L', 'dependency=' + str(deps)]
for name, library in libraries.items():
    common += ['--extern', name + '=' + str(library)]
binary = ROOT / 'target/scripted-tavernbench'
test_binary = ROOT / 'target/scripted-tavernbench-tests'
execute([*common, str(SOURCE / 'adapter.rs'), '-o', str(binary)], 'adapter-build.txt')
execute([*common, '--test', str(SOURCE / 'adapter.rs'), '-o', str(test_binary)], 'test-build.txt')
tested = execute([str(test_binary), '--test-threads=1'], 'adapter-tests.txt')
print(tested.stdout, flush=True)

# Only prompts and saved input states are opened before all predictions are sealed.
prompts = [json.loads(line) for line in (DATA / 'prompts.jsonl').read_text().splitlines()]
with gzip.open(DATA / 'states.jsonl.gz', 'rt') as source:
    specs = {row['id']: row for row in map(json.loads, source)}
allowed_spec = {'id', 'state', 'states', 'options', 'perms', 'answers', 'map', 'seat', 'health', 'cap'}
for spec in specs.values():
    assert set(spec) <= allowed_spec, f'Unexpected input fields: {set(spec)-allowed_spec}'
    for option in spec.get('options', []):
        assert set(option) == {'answer', 'ops'}, 'Candidate metadata contains unexpected fields'
requests = []
for prompt in prompts:
    task_id = prompt['id']
    seed = int.from_bytes(hashlib.blake2b(('scripted-v1:' + task_id).encode(), digest_size=8).digest(), 'little')
    requests.append({'id': task_id, 'scenario': prompt['scenario'], 'seed': seed, 'spec': specs.get(task_id, {})})
assert len(requests) == 368
started = time.perf_counter()
generated = subprocess.run([str(binary)], cwd=ROOT, env=env, text=True,
                           input=''.join(json.dumps(row) + '\n' for row in requests),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(OUT / 'inference.jsonl').write_text(generated.stdout)
(OUT / 'inference-stderr.txt').write_text(generated.stderr)
if generated.returncode:
    raise SystemExit(generated.stderr)
inference = [json.loads(line) for line in generated.stdout.splitlines()]
assert len(inference) == len(prompts)
assert [r['id'] for r in inference] == [p['id'] for p in prompts]
assert all(not r.get('panic') for r in inference), 'Native adapter panicked; inspect inference.jsonl'
files = {}
for system, key in [('scripted-policy', 'policy'), ('scripted-combat256', 'combat256')]:
    folder = OUT / 'answers' / system
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in inference:
        answer = row[key]
        ok = answer['status'] == 'ok'
        rows.append({'id': row['id'], 'model': system,
                     'response': json.dumps(answer['response']) if ok else None,
                     'error': None if ok else answer.get('reason', answer['status']),
                     'status': answer['status'], 'finish_reason': 'stop' if ok else None,
                     'latency_s': answer['latency_s'], 'input_tokens': 0, 'output_tokens': 0,
                     'adapter_details': answer.get('details', {})})
    path = folder / 'answers.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    files[system] = path
manifest = {'created_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dataset': str(DATA.relative_to(ROOT)), 'tasks': len(prompts),
            'input_hashes': {name: digest(DATA / name) for name in ['prompts.jsonl', 'states.jsonl.gz']},
            'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [SOURCE/'adapter.rs', SOURCE/'run.py', ROOT/'crates/taverngym/src/bot.rs']},
            'answers_sha256': {name: digest(path) for name, path in files.items()},
            'inference_wall_seconds': time.perf_counter() - started,
            'labels_read_by_inference': False, 'candidate_score_values_used': False,
            'adapter_test_output': 'adapter-tests.txt'}
(OUT / 'prediction-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('Predictions sealed. Scoring starts now.', flush=True)

# The label file is opened for the first time here, after answer hashes are recorded.
sys.path.insert(0, str(ROOT / 'tavernbench'))
from tavernbench.report import load, evaluate, summarize, RULE, scenario_acc
from tavernbench.score import load_answers, score_answer
labels, prompt_map = load(DATA)
per_item = []
summary = {}
for system, path in files.items():
    acc, cost = evaluate(labels, prompt_map, path)
    answers = load_answers(path)
    coverage = collections.Counter()
    by_scenario = collections.defaultdict(collections.Counter)
    covered_accuracy = collections.defaultdict(list)
    for iid, prompt in prompt_map.items():
        row = answers[iid]
        status = row['status']
        coverage[status] += 1
        if prompt['scenario'] not in RULE:
            coverage['play_total'] += 1
            coverage['play_ok'] += status == 'ok'
        by_scenario[prompt['scenario']][status] += 1
        correct, graded = score_answer(labels[iid], row)
        if status == 'ok':
            covered_accuracy[prompt['scenario']].append((labels[iid]['answer'], correct))
        per_item.append({'system': system, 'id': iid, 'scenario': prompt['scenario'], 'status': status,
                         'correct': correct, 'graded_score': graded})
    summary[system] = {**summarize(acc), **cost, 'by_scenario': acc,
                       'coverage': dict(coverage),
                       'coverage_by_scenario': {k: dict(v) for k,v in by_scenario.items()},
                       'rules_applicable': False,
                       'answered_scenario_mean': sum(scenario_acc(k,v) for k,v in covered_accuracy.items()) / max(1,len(covered_accuracy))}
    assert digest(path) == manifest['answers_sha256'][system]
    print(system, 'Play', f"{summary[system]['play']:.4%}", 'coverage', dict(coverage), flush=True)
references = {}
for system in ['tavernagent', 'gpt-6-astra', 'claude-opus-5.5', 'jev']:
    path = ROOT / 'tavernbench/results/eval' / system / 'answers.jsonl'
    acc, cost = evaluate(labels, prompt_map, path)
    references[system] = {**summarize(acc), 'answers_sha256': digest(path)}
(OUT / 'item-scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in per_item))
results = {'dataset': 'eval', 'tasks': len(prompts), 'play_tasks': 288, 'play_scenarios': 30,
           'systems': summary, 'released_reference_results': references,
           'rules_interpretation': 'Not applicable for these bots; official numeric rule_score is zero because all 80 rule-QA tasks are intentionally unsupported.',
           'labels_sha256': digest(DATA / 'labels.jsonl'), 'prediction_manifest': 'prediction-manifest.json'}
(OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print('Saved:', OUT)
