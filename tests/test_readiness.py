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


@pytest.fixture
def position_assessment(assessment):
    from training import audit_bundles as bundles
    m, reports = assessment
    m.update(format_version=2, decoding=dict(bundles.ROLE_DECODING),
             seed_history={'path':'seed-history.json','sha256':'f'*64},
             role_decoding={k:dict(bundles.GREEDY) for k in bundles.POLICY_KEYS})
    for key in ('candidate','champion'):
        model = m[key]
        model['trained_hotfix_37_0_1'] = True
        config = dict(format_version=1, kind='public_position_v1', engine_hash=m['engine_hash'],
                      rules=dict(m['rules']), threshold=.025,
                      actor={'path':'actor.pt','sha256':model['sha256']},
                      encoder={'path':'encoder.pt','sha256':'9'*64},
                      ridge={'path':'ridge.pt','sha256':'8'*64})
        encoder = dict(m['reference'], path=f'{key}-position/encoder.pt', sha256='9'*64,
                       trained_hotfix_37_0_1=True, train_seed=201)
        model['position_bundle'] = dict(path=f'{key}-position/position.json', sha256='7'*64,
                                        config=config, encoder_metadata=encoder)
        m['role_decoding'][key] = bundles.expected_position_decoder(model['position_bundle'])
    for suite, _, _, name in audit.jobs(m):
        report = reports[name]
        event = dict(identity=0, turn=5, action=197, predicted_gain=.1)
        report['position_decoder'] = dict(counted_episode_moves=1, events=[dict(event)])
        if suite == 'bots':
            report['decoding'] = dict(m['role_decoding']['candidate'])
        else:
            roles = bundles.suite_roles(suite)
            report['role_decoding'] = [dict(m['role_decoding'][k]) if k != 'bot'
                                       else {'kind':'bundled_heuristic'} for k in roles]
            report['decoding'] = dict(bundles.GREEDY, kind='role_specific_public_position_v1')
            report['position_decoders'] = [dict(roles=[0], statistics={'moves':1}, events=[dict(event)]),
                dict(roles=[i for i,k in enumerate(roles) if k == 'champion'],
                     statistics={'moves':1}, events=[dict(event, identity=1)])]
    return m, reports


def test_position_audit_keeps_all_nine_gates(position_assessment):
    m, reports = position_assessment
    result = audit.verify_reports(m, reports)
    assert result['performance_gates_passed'] and not result['live_ready']
    assert result['total_games'] == 3072
    # Human-readable scope is excluded from the semantic identity only.
    reports['champion-1.json']['role_decoding'][0]['scope'] = 'display text'
    audit.verify_reports(m, reports)


@pytest.mark.parametrize('case', [
    'weak_reference', 'wrong_encoder', 'wrong_ridge', 'threshold', 'lineage_seed',
    'path_escape', 'missing_role', 'wrong_event_role', 'duplicate_move', 'ineligible_turn',
    'nonfinite_gain', 'threshold_boundary', 'missing_candidate_events', 'counter_mismatch',
    'bot_autoreset_event', 'manifest_changed_decoder', 'missing_seed_history',
])
def test_position_corruption_cannot_pass(position_assessment, case):
    m, reports = position_assessment
    r = reports['champion-1.json']
    bundle = m['candidate']['position_bundle']
    if case == 'weak_reference': r['role_decoding'][1] = dict(audit.audit_bundles.GREEDY)
    elif case == 'wrong_encoder': r['role_decoding'][0]['encoder_sha256'] = '0'*64
    elif case == 'wrong_ridge': r['role_decoding'][1]['ridge_sha256'] = '0'*64
    elif case == 'threshold': r['role_decoding'][0]['threshold'] = .5
    elif case == 'lineage_seed': bundle['encoder_metadata']['train_seed'] = m['seeds'][0]
    elif case == 'path_escape': bundle['config']['encoder']['path'] = '../encoder.pt'
    elif case == 'missing_role': r['position_decoders'][1]['roles'].pop()
    elif case == 'wrong_event_role': r['position_decoders'][1]['events'][0]['identity'] = 9
    elif case == 'duplicate_move':
        r['position_decoders'][0]['events'] *= 2
        r['position_decoders'][0]['statistics']['moves'] = 2
    elif case == 'ineligible_turn': r['position_decoders'][0]['events'][0]['turn'] = 3
    elif case == 'nonfinite_gain': r['position_decoders'][0]['events'][0]['predicted_gain'] = float('nan')
    elif case == 'threshold_boundary': r['position_decoders'][0]['events'][0]['predicted_gain'] = .025
    elif case == 'missing_candidate_events': r['position_decoder']['events'] = []
    elif case == 'counter_mismatch': r['position_decoders'][1]['statistics']['moves'] = 2
    elif case == 'bot_autoreset_event': reports['bots-1.json']['position_decoder']['events'][0]['identity'] = 512
    elif case == 'manifest_changed_decoder': m['role_decoding']['candidate']['ridge_sha256'] = '0'*64
    elif case == 'missing_seed_history': m.pop('seed_history')
    with pytest.raises((ValueError, KeyError)):
        audit.verify_reports(m, reports)


def test_position_runner_assigns_bundles_to_exact_roles(position_assessment, tmp_path, monkeypatch):
    m, reports = position_assessment
    path = tmp_path/'manifest.json'
    audit.write_new(path, m)
    monkeypatch.setattr(audit, 'load_audit', lambda _:copy.deepcopy(m))
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        target = Path(command[command.index('--out')+1])
        audit.write_new(target, reports[target.name])
    monkeypatch.setattr(audit.subprocess, 'run', execute)
    audit.run_audit(path, device='cuda', threads=2)
    assert len(commands) == 9
    for command in commands:
        assert command[command.index('--position-bundle')+1] == str(tmp_path/'candidate-position/position.json')
    champion = str(tmp_path/'champion-position/position.json')
    assert commands[3][commands[3].index('--opponent-position-bundles')+1:][:7] == [champion]*7
    assert commands[6][commands[6].index('--opponent-position-bundles')+1:][:7] == [champion]*2+['none']*5


def test_seed_history_includes_research_jsonl_and_extra_files(tmp_path):
    history, root = tmp_path/'history', tmp_path/'audit'
    history.mkdir(); root.mkdir()
    research = history/'research'; research.mkdir()
    audit.write_new(research/'plan.json', {'confirmation_seeds':[17, 29], 'dataset_seed':37})
    (history/'metrics.jsonl').write_text('{"validation_seed":43}\n{"seed":59}\n')
    extra = tmp_path/'extra.json'; audit.write_new(extra, {'reused_control_seeds':[61]})
    used, record = audit.freeze_seed_history(root, history, [extra], {71}, {73}, required=True)
    assert used == {17,29,37,43,59,61,71,73}
    manifest = dict(format_version=2, seed_history=record, excluded_seeds=sorted(used),
                    candidate={'train_seed':71}, champion={}, reference={})
    audit.verify_seed_history(root, manifest)
    (root/'seed-history.json').write_text('{}')
    with pytest.raises(ValueError, match='history changed'):
        audit.verify_seed_history(root, manifest)
    (history/'metrics.jsonl').write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        audit.freeze_seed_history(root, history, [], set(), set(), required=True)


def test_reference_identity_includes_position_components(position_assessment):
    from training import audit_bundles as bundles
    m, reports = position_assessment
    report = reports['mixed-1.json']
    expected = {bundles.policy_identity(m[k]['sha256'], m['role_decoding'][k]) for k in bundles.POLICY_KEYS}
    assert bundles.evidence_identities(report) == expected
    legacy = copy.deepcopy(report)
    legacy['role_policies'] = legacy.pop('role_decoding')
    for decoder in legacy['role_policies']:
        if decoder['kind'] == bundles.POSITION_KIND:
            decoder['kind'] = 'public_position_v1'
            decoder.pop('bundle_sha256')
            decoder.pop('position_version')
    assert bundles.evidence_identities(legacy) == expected
    report['role_decoding'][1]['ridge_sha256'] = '0'*64
    assert bundles.evidence_identities(report) != expected


def test_position_freeze_is_portable_binds_encoder_and_draws_seeds_last(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import taverngym as tg
    from training.entity import EntityPolicy
    from training.position import freeze_bundle
    from training.arena import frozen_checkpoint
    from training import audit_bundles as bundles

    sources, metadata = {}, {}
    for i,key in enumerate((*bundles.POLICY_KEYS, 'encoder')):
        torch.manual_seed(i+11)
        model = EntityPolicy(width=8, layers=1, heads=2, text_dim=8)
        sources[key] = tmp_path/f'{key}.pt'
        torch.save(dict(engine_hash=tg.source_hash(), training_code_hash='test-only',
                        architecture=model.architecture, model=model.state_dict(), update=0, steps=0,
                        args=dict(seed=i+1, eval_seed=101+i*20, action_cap=128, hotfix_37_0_1=True)), sources[key])
        _, metadata[key] = frozen_checkpoint(sources[key], torch.device('cpu'))
    size = 6*model.architecture['width']+51+7*model.layout['MINION_F']
    ridge = tmp_path/'ridge.pt'
    torch.save(dict(family='frozen_tokens_plus_public_numeric_position_delta',
                    encoder_sha256=metadata['encoder']['sha256'], intercept=.1,
                    weight=torch.zeros(size,dtype=torch.float64), feature_mean=torch.zeros(size,dtype=torch.float64),
                    feature_scale=torch.ones(size,dtype=torch.float64)), ridge)
    bundle_paths = {}
    for key in ('candidate','champion'):
        directory = tmp_path/f'{key}-bundle'
        freeze_bundle(Namespace(actor=sources[key], encoder=sources['encoder'], ridge=ridge, threshold=.025, directory=directory))
        bundle_paths[key] = directory/'position.json'
    evidence = tmp_path/'reference.json'
    champion_config = json.loads(bundle_paths['champion'].read_text())
    champion_decoder = bundles.expected_position_decoder(dict(config=champion_config, sha256=audit.checksum(bundle_paths['champion'])))
    audit.write_new(evidence, dict(guards_checked=True, driver=audit.ARENA_DRIVER, seed=9001,
        engine={'source_hash':tg.source_hash()}, rules={'turn_action_caps':[128],'hotfix_37_0_1':True},
        truncated_combats=0, turn_limit_ends=0, rejected_actions=0, incomplete_lobbies=0,
        role_models=[metadata['champion']]*4+[metadata['reference']]*4,
        role_decoding=[champion_decoder]*4+[dict(bundles.GREEDY)]*4))
    history = tmp_path/'history'; history.mkdir()
    audit.write_new(history/'research.json', {'confirmation_seeds':[7001,8001]})
    root = tmp_path/'audit'
    def seeds(used):
        assert {7001,8001,9001,4,161} <= used
        assert (root/'seed-history.json').exists()
        for key in bundle_paths:
            assert audit.checksum(root/f'{key}-position/position.json') == audit.checksum(bundle_paths[key])
            assert audit.checksum(root/f'{key}-position/encoder.pt') == metadata['encoder']['sha256']
        return [910000001,910000101,910000201]
    monkeypatch.setattr(audit, 'draw_seeds', seeds)
    audit.freeze(Namespace(**{k:sources[k] for k in bundles.POLICY_KEYS},
        candidate_position_bundle=bundle_paths['candidate'], champion_position_bundle=bundle_paths['champion'],
        directory=root, history_root=history, reference_evidence=[evidence], reference_selection_note='Fixture only'))
    manifest = audit.load_audit(root/'manifest.json')
    assert manifest['format_version'] == 2
    moved = tmp_path/'downloaded'; shutil.copytree(root, moved)
    root.rename(tmp_path/'original')
    assert audit.load_audit(moved/'manifest.json') == manifest
    (moved/'candidate-position/encoder.pt').write_bytes(b'changed encoder')
    with pytest.raises(ValueError, match='position encoder changed'):
        audit.load_audit(moved/'manifest.json')
