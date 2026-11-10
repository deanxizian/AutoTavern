"""Migration rejects optimizer remapping and copies only the two rule buffers."""

from collections import OrderedDict
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('stable_baselines3')

spec = importlib.util.spec_from_file_location(
    'eric_weight_migration', Path(__file__).resolve().parents[1] / 'tools/migrate_eric_3702.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def fixture_state():
    policy = OrderedDict((name, torch.tensor([float(index)]))
                         for index, name in enumerate(['weight_a', 'weight_b']))
    optimizer = {'param_groups': [{'params': [0, 1]}], 'state': {
        index: {'step': torch.tensor(8800.), 'exp_avg': torch.tensor([index + .1]),
                'exp_avg_sq': torch.tensor([index + .2])} for index in range(2)}}
    return policy, optimizer


def test_adam_mapping_requires_source_registration_order_even_when_shapes_match():
    policy, optimizer = fixture_state()
    mapping = migration.audit_optimizer(policy, optimizer, ['weight_a', 'weight_b'], 8800)
    assert [(row['index'], row['parameter']) for row in mapping] == [(0, 'weight_a'),
                                                                   (1, 'weight_b')]
    swapped = OrderedDict(reversed(list(policy.items())))
    with pytest.raises(ValueError, match='registration order'):
        migration.audit_optimizer(swapped, optimizer, ['weight_a', 'weight_b'], 8800)


@pytest.mark.parametrize('damage', ['missing', 'wrong_clock', 'float64', 'nan'])
def test_invalid_or_incomplete_adam_is_never_accepted(damage):
    policy, optimizer = fixture_state()
    if damage == 'missing':
        del optimizer['state'][1]
    elif damage == 'wrong_clock':
        optimizer['state'][1]['step'] = torch.tensor(8799.)
    elif damage == 'float64':
        optimizer['state'][1]['exp_avg'] = optimizer['state'][1]['exp_avg'].double()
    else:
        optimizer['state'][1]['exp_avg_sq'][0] = float('nan')
    with pytest.raises(ValueError):
        migration.audit_optimizer(policy, optimizer, ['weight_a', 'weight_b'], 8800)


def test_rule_refresh_keeps_learned_weights_and_unrelated_action_buffers_exact():
    original = OrderedDict([
        ('weight', torch.tensor([7.])), ('network.static_features', torch.tensor([1.])),
        ('network.text_features', torch.tensor([2.])), ('network.action_kinds', torch.tensor([3])),
    ])
    current = copy.deepcopy(original)
    current['weight'] = torch.tensor([-99.])  # Fresh random learned weights must not be copied.
    current['network.static_features'] = torch.tensor([11.])
    current['network.text_features'] = torch.tensor([12.])
    adapted = migration.refresh_policy(original, current, ['weight'])
    assert adapted['weight'].item() == 7.
    assert adapted['network.static_features'].item() == 11.
    assert adapted['network.text_features'].item() == 12.
    assert torch.equal(adapted['network.action_kinds'], original['network.action_kinds'])
    assert original['network.static_features'].item() == 1.
    current['network.action_kinds'][0] = 4
    with pytest.raises(ValueError, match='Unexpected non-learned buffer'):
        migration.refresh_policy(original, current, ['weight'])
