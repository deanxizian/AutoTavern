"""Run the repository's original benchmark on one logical CPU; retain raw results."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
if (OUT / 'benchmark-results.json').exists():
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    OUT = OUT / 'reruns' / ('benchmark-' + stamp)
    OUT.mkdir(parents=True, exist_ok=False)
BINARY = ROOT / 'target/release/examples/bench'
CPU = min(os.sched_getaffinity(0))
GAMES = 2000
RUNS = 3

def run_bench(n, name):
    command = ['taskset', '-c', str(CPU), str(BINARY), str(n)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    (OUT / name).write_text(result.stdout + result.stderr)
    return command, result.stdout

command, warmup = run_bench(200, 'warmup.txt')
records = []
for i in range(1, RUNS + 1):
    command, output = run_bench(GAMES, f'run-{i}.txt')
    full = re.search(r'full games: (\d+) in ([\d.]+)s -> (\d+) games/s, avg ([\d.]+) turns', output)
    combat = re.search(r'combats: (\d+) in ([\d.]+)s -> (\d+) combats/s \((\d+) decided\)', output)
    assert full and combat, output
    assert int(full[1]) == GAMES and int(combat[1]) == 20000
    record = {'run': i, 'games': GAMES, 'seeds': [0, GAMES-1], 'games_per_second': int(full[3]),
              'milliseconds_per_game': 1000 / int(full[3]), 'mean_turns': float(full[4]),
              'combats': int(combat[1]), 'combats_per_second': int(combat[3]),
              'microseconds_per_combat': 1e6 / int(combat[3]), 'decided_combats': int(combat[4]),
              'command': command}
    records.append(record)
    print(f"Run {i}: {record['games_per_second']} games/s ({record['milliseconds_per_game']:.3f} ms/game); "
          f"{record['combats_per_second']} combats/s ({record['microseconds_per_combat']:.3f} us/combat)", flush=True)

cpu_info = subprocess.check_output(['lscpu'], text=True)
(OUT / 'lscpu.txt').write_text(cpu_info)
source_hash = hashlib.sha256()
for path in sorted([*ROOT.glob('crates/taverngym/src/**/*.rs'), ROOT/'crates/taverngym/data/cards.json', ROOT/'crates/taverngym/examples/bench.rs', ROOT/'Cargo.lock', ROOT/'Cargo.toml']):
    source_hash.update(path.relative_to(ROOT).as_posix().encode() + b'\0' + path.read_bytes())
summary = {
    'created_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_sha256': source_hash.hexdigest(), 'system': platform.platform(),
    'cpu_model': next(line.split(':',1)[1].strip() for line in cpu_info.splitlines() if line.startswith('Model name:')),
    'cpu_affinity': [CPU], 'threads_in_benchmark': 1,
    'build': 'release; workspace defaults; thin LTO; one codegen unit; no target-cpu=native override',
    'rules': 'RulesConfig::default(); 36.6.3; eight built-in heuristic bots; record_log=false',
    'warmup': {'games': 200, 'combats': 20000, 'included_in_measurement': False},
    'timing': 'Original benchmark internal Instant timing; excludes compilation and console output; per-operation latency derived from printed throughput.',
    'combat_fixture': 'Exactly as examples/bench.rs: seed-0 terminal snapshot; revive all seats and copy last_board into board; repeat 20,000 pairwise fights.',
    'combat_pairing_calls': sum(1 for i in range(20000) if i%8 != (i+1+(i//8)%7)%8),
    'runs': records,
    'aggregate': {
        'median_games_per_second': statistics.median(x['games_per_second'] for x in records),
        'median_milliseconds_per_game': statistics.median(x['milliseconds_per_game'] for x in records),
        'median_combats_per_second': statistics.median(x['combats_per_second'] for x in records),
        'median_microseconds_per_combat': statistics.median(x['microseconds_per_combat'] for x in records),
    },
    'paper_ranges': {'games_per_second': [190,280], 'milliseconds_per_game': [3.6,5.3],
                     'combats_per_second': [52000,76000], 'microseconds_per_combat': [13,19]},
}
(OUT/'benchmark-results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary['aggregate'], ensure_ascii=False, indent=2))
print('Results saved to:', OUT)
