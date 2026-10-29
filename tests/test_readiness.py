"""Audit corruption, statistical uncertainty and the final-evaluation lifecycle."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from training import readiness as audit


def summary(values):
    return dict(games=len(values), mean_placement=sum(values) / len(values),
                top4_rate=sum(x <= 4 for x in values) / len(values),
                win_rate=values.count(1) / len(values))


def set_arena_placements(report, values):
    report['candidate'] = summary(values)
    report['lobbies'] = []
    for lane, placement in enumerate(values):
        others = iter(p for p in range(1, 9) if p != placement)
        report['lobbies'].append(dict(lane=lane,
            role_by_seat=[(seat-lane) % 8 for seat in range(8)],
            placement_by_seat=[placement if seat == lane % 8 else next(others) for seat in range(8)]))


@pytest.fixture
def assessment():
    m = dict(format_version=1, purpose='final_simulator_audit',
             created_at='2026-10-29T01:00:00+00:00', engine_hash='e' * 16,
             evaluation_code_hash='d' * 64, rules={'turn_action_caps':[128], 'hotfix_37_0_1':True},
             decoding={'kind':'greedy', 'cancel_retry_guard':True, 'guard_version':1},
             seeds=[910000001, 910000101, 910000201], excluded_seeds=[1, 100],
             reference_evidence=[{'path':'reference-evidence.json', 'sha256':'e' * 64}],
             reference_selection_note='Previous stage champion and its strongest reference')
    for key, char in [('candidate','a'), ('champion','b'), ('reference','c')]:
        m[key] = dict(path=f'{key}.pt', sha256=char * 64, engine_hash=m['engine_hash'],
                      action_cap=128, train_seed=1, validation_seed=100)
    reports = {}
    for suite, seed, count, name in audit.jobs(m):
        report = dict(seed=seed, games=count, rules=dict(m['rules']), decoding=dict(m['decoding']),
                      evaluation_code_hash=m['evaluation_code_hash'], engine={'source_hash':m['engine_hash']},
                      evaluation_started_at='2026-10-29T01:01:00+00:00', guards_checked=True,
                      truncated_combats=0, turn_limit_ends=0, rejected_actions=0, incomplete_lobbies=0)
        if suite == 'bots':
            values = [1, 1, 1, 2] * 128
            report.update(driver=audit.BOT_DRIVER, kind='policy', checkpoint=dict(m['candidate']),
                          placements_by_seat={str(s):values[s*64:(s+1)*64] for s in range(8)},
                          **summary(values))
        else:
            report.update(driver=audit.ARENA_DRIVER,
                          role_models=[{'kind':'bundled_heuristic'} if x == 'bot' else {'sha256':x}
                                       for x in audit.role_ids(m, suite)])
            set_arena_placements(report, [1, 2, 3, 4] * 64)
        reports[name] = report
    return m, reports


def test_complete_audit_passes_only_simulator_gate(assessment):
    m, reports = assessment
    result = audit.verify_reports(m, reports)
    assert result['performance_gates_passed'] and not result['live_ready']
    assert result['total_games'] == 3072
    assert result['suites']['bots']['aggregate']['mean_placement'] == 1.25
    assert result['suites']['champion']['aggregate']['games'] == 768


@pytest.mark.parametrize('case', [
    'missing_report', 'reused_seed', 'training_seed', 'different_candidate', 'different_opponent',
    'missing_guards', 'guard_end', 'duplicate_lane', 'duplicate_placement', 'wrong_rotation',
    'wrong_decoder', 'wrong_code', 'wrong_engine', 'predates_freeze', 'edited_score',
    'unbalanced_bots', 'wrong_driver', 'nonfinite_score', 'missing_counter',
])
def test_corrupt_assessments_cannot_pass(assessment, case):
    m, reports = assessment
    r = reports['champion-1.json']
    if case == 'missing_report': reports.pop('mixed-3.json')
    elif case == 'reused_seed': m['seeds'][1] = m['seeds'][0] + 1
    elif case == 'training_seed': m['candidate']['train_seed'] = m['seeds'][0]
    elif case == 'different_candidate': reports['bots-1.json']['checkpoint']['sha256'] = '0' * 64
    elif case == 'different_opponent': r['role_models'][2]['sha256'] = m['reference']['sha256']
    elif case == 'missing_guards': r.pop('guards_checked')
    elif case == 'guard_end': r['turn_limit_ends'] = 1
    elif case == 'duplicate_lane': r['lobbies'][5]['lane'] = 4
    elif case == 'duplicate_placement': r['lobbies'][0]['placement_by_seat'] = [1] * 8
    elif case == 'wrong_rotation': r['lobbies'][1]['role_by_seat'] = list(range(8))
    elif case == 'wrong_decoder': r['decoding']['cancel_retry_guard'] = False
    elif case == 'wrong_code': r['evaluation_code_hash'] = '0' * 64
    elif case == 'wrong_engine': r['engine']['source_hash'] = '0' * 16
    elif case == 'predates_freeze': r['evaluation_started_at'] = '2026-10-29T00:59:59+00:00'
    elif case == 'edited_score': r['candidate']['mean_placement'] = 1.
    elif case == 'unbalanced_bots': reports['bots-1.json']['placements_by_seat']['0'].pop()
    elif case == 'wrong_driver': r['driver'] = 'batch_snapshot'
    elif case == 'nonfinite_score': r['candidate']['top4_rate'] = float('nan')
    elif case == 'missing_counter': r.pop('incomplete_lobbies')
    with pytest.raises((ValueError, KeyError)):
        audit.verify_reports(m, reports)


@pytest.mark.parametrize('values,failed_check', [
    ([1]*160 + [8]*96, 'top4_lower_at_least_60'),
    ([3]*210 + [8]*46, 'mean_upper_below_4'),
])
def test_point_estimate_is_not_a_confidence_gate(assessment, values, failed_check):
    m, reports = assessment
    for i in range(1, 4): set_arena_placements(reports[f'champion-{i}.json'], values)
    result = audit.verify_reports(m, reports)
    suite = result['suites']['champion']
    assert suite['aggregate']['mean_placement'] < 4
    assert suite['aggregate']['top4_rate'] > .60
    assert not suite['checks'][failed_check] and not result['performance_gates_passed']


def test_good_aggregate_does_not_hide_a_bad_seed(assessment):
    m, reports = assessment
    for i in range(1, 4): set_arena_placements(reports[f'champion-{i}.json'], [5 if i == 1 else 1]*256)
    result = audit.verify_reports(m, reports)
    suite = result['suites']['champion']
    assert suite['checks']['mean_upper_below_4'] and suite['checks']['top4_lower_at_least_60']
    assert not suite['checks']['every_seed_mean_below_4_5'] and not result['performance_gates_passed']


def test_changed_artifacts_invalidate_a_frozen_manifest(assessment, tmp_path):
    m, _ = assessment
    m['evaluation_code_hash'] = audit.source_hash()
    for key in ('candidate', 'champion', 'reference'):
        file = tmp_path / f'{key}.pt'
        file.write_bytes(key.encode())
        m[key].update(path=str(file), sha256=audit.checksum(file))
    evidence = tmp_path / 'reference.json'
    evidence.write_text('{}')
    m['reference_evidence'] = [dict(path=str(evidence), sha256=audit.checksum(evidence))]
    manifest = tmp_path / 'manifest.json'
    audit.write_new(manifest, m)
    assert audit.load_audit(manifest) == m
    (tmp_path / 'candidate.pt').write_bytes(b'new checkpoint')
    with pytest.raises(ValueError, match='Frozen candidate changed'):
        audit.load_audit(manifest)


@pytest.mark.parametrize('fail', [False, True])
def test_one_shot_runner_retains_results_or_failure(assessment, tmp_path, monkeypatch, fail):
    m, reports = assessment
    manifest = tmp_path / 'manifest.json'
    audit.write_new(manifest, m)
    monkeypatch.setattr(audit, 'load_audit', lambda path: copy.deepcopy(m))
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        out = Path(command[command.index('--out')+1])
        if fail and len(commands) == 4:
            raise subprocess.CalledProcessError(1, command)
        audit.write_new(out, reports[out.name])

    monkeypatch.setattr(audit.subprocess, 'run', execute)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            audit.run_audit(manifest, device='cuda', threads=4)
        assert not (tmp_path / 'readiness.json').exists()
        assert json.loads((tmp_path / 'execution.failed.json').read_text())['performance_gates_passed'] is False
        assert len(commands) == 4
    else:
        audit.run_audit(manifest, device='cuda', threads=4)
        assert len(commands) == 9
        assert json.loads((tmp_path / 'readiness.json').read_text())['performance_gates_passed']
        champ, ref = (str(tmp_path / f'{k}.pt') for k in ('champion', 'reference'))
        assert commands[3][commands[3].index('--opponents')+1:] == [champ]*7
        assert commands[6][commands[6].index('--opponents')+1:] == [champ]*2 + [ref]*2 + ['bot']*3
        audit.verify_audit(manifest)  # Existing identical result can be checked again.
    with pytest.raises((ValueError, FileExistsError)):
        audit.run_audit(manifest, device='cuda', threads=4)


def test_freeze_copies_models_before_drawing_seeds(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import taverngym as tg
    from training.common import Policy

    source, identities = {}, []
    for index, key in enumerate(('candidate', 'champion', 'reference')):
        torch.manual_seed(index)
        policy = Policy(hidden=8, embedding=2)
        checkpoint = dict(model=policy.state_dict(), architecture=policy.architecture,
                          engine_hash=tg.source_hash(), training_code_hash='test', steps=0, update=0,
                          args=dict(seed=index+1, eval_seed=100+index*20, action_cap=128, hotfix_37_0_1=True))
        source[key] = tmp_path / f'source-{key}.pt'
        torch.save(checkpoint, source[key])
        identities.append(audit.checksum(source[key]))
    evidence = tmp_path / 'reference.json'
    audit.write_new(evidence, dict(guards_checked=True, driver=audit.ARENA_DRIVER,
        seed=9001, rules={'turn_action_caps':[128], 'hotfix_37_0_1':True},
        role_models=[{'sha256':x} for x in identities[1:]]))
    root = tmp_path / 'audit'

    def seeds(used):
        assert 9001 in used
        for index, key in enumerate(source):
            assert audit.checksum(root / f'{key}.pt') == identities[index]
        return [910000001, 910000101, 910000201]

    monkeypatch.setattr(audit, 'draw_seeds', seeds)
    audit.freeze(Namespace(**source, directory=root, reference_evidence=[evidence],
                           reference_selection_note='Prior stage comparison', history_root=tmp_path / 'history'))
    manifest = audit.load_audit(root / 'manifest.json')
    assert manifest['seeds'] == [910000001, 910000101, 910000201]
    assert all(manifest[k]['path'] == f'{k}.pt' and (root / manifest[k]['path']).exists() for k in source)
    assert (root / manifest['reference_evidence'][0]['path']).exists()
    moved = tmp_path / 'downloaded-audit'
    shutil.copytree(root, moved)
    root.rename(tmp_path / 'original-audit')
    assert audit.load_audit(moved / 'manifest.json') == manifest


@pytest.mark.parametrize('module_name,driver', [
    ('training.evaluate', audit.BOT_DRIVER), ('training.arena', audit.ARENA_DRIVER),
])
def test_real_cli_reports_evaluation_provenance(module_name, driver, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import importlib
    import sys
    import taverngym as tg

    class FirstLegal(torch.nn.Module):
        def forward(self, features, cards, masks):
            return torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9), torch.zeros(len(masks))

    module = importlib.import_module(module_name)
    meta = dict(trained_hotfix_37_0_1=True, action_cap=128, steps=0, train_seed=1,
                validation_seed=100, sha256='a'*64, engine_hash=tg.source_hash())
    monkeypatch.setattr(module, 'frozen_checkpoint', lambda *args: (FirstLegal(), dict(meta)))
    output = tmp_path / 'score.json'
    monkeypatch.setattr(sys, 'argv', ['assessment', 'test.pt', '--device', 'cpu', '--games', '8',
                                    '--seed', '6001', '--threads', '1', '--cancel-guard', '--out', str(output)])
    module.main()
    result = json.loads(output.read_text())
    assert result['driver'] == driver and result['guards_checked']
    assert result['evaluation_code_hash'] == audit.source_hash()
    assert result['engine']['source_hash'] == tg.source_hash()
    assert result['evaluation_started_at'].endswith('+00:00')
    assert result['rejected_actions'] == result['incomplete_lobbies'] == 0
