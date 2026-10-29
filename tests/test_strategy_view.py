"""Information boundary required by human-guided strategy teaching."""
from copy import deepcopy
import json
import struct

import pytest
import taverngym as tg


def fixture():
    game = tg.Game(541, json.dumps({'num_players': 8, 'record_log': False,
                                   'hotfix_37_0_1': True}))
    for seat in range(8):
        game.step(seat, 30)
    state = json.loads(game.state_json())
    for player in state['players']:
        for key in ('powers', 'pending', 'board', 'hand', 'shop', 'trinkets', 'effects', 'scheduled'):
            player[key] = []
        player.update(skipping=False, ended_turn=False, gold=10, tier=2)
    return tg.Game.from_json(json.dumps(state))


def add(game, seat, zone):
    game.debug(json.dumps({'cmd': 'add_minion', 'p': seat, 'zone': zone, 'card': 'BG36_345'}))


def test_strategy_view_is_public_and_does_not_change_game_or_legacy_inputs():
    game = fixture()
    for seat in (0, 1):
        for zone in ('board', 'hand', 'shop'):
            add(game, seat, zone)
    state = game.state_json()
    inputs = game.observe(0), game.legal_mask(0), game.attachment_observation(0)
    view = json.loads(game.strategy_view(0))
    assert view['schema_version'] == 1
    assert not {'seed', 'rng', 'config', 'combats'} & view.keys()
    assert not {'board', 'last_board', 'hand', 'shop'} & view['opponents'][0].keys()
    assert view['me'] == json.loads(game.view(0))['me']  # No powers in this fixture.
    assert view['mask'] == list(game.legal_mask(0))
    assert game.state_json() == state
    assert (game.observe(0), game.legal_mask(0), game.attachment_observation(0)) == inputs

    private = json.loads(state)
    private['rng'] = json.loads(tg.Game(999).state_json())['rng']
    private['seed'] = 999
    opponent = private['players'][1]
    opponent['board'][0]['atk'] += 100
    opponent['last_board'] = deepcopy(opponent['board'])
    opponent['hand'][0]['item']['Minion']['atk'] += 200
    opponent['shop'][0]['item']['Minion']['hp'] += 300
    altered = tg.Game.from_json(json.dumps(private))
    assert json.loads(altered.strategy_view(0)) == view


def test_observed_opponent_memory_is_preserved_without_current_board_access():
    game = fixture()
    add(game, 1, 'board')
    state = json.loads(game.state_json())
    remembered = deepcopy(state['players'][1]['board'])
    state['players'][0]['scouted'] = [{'board': [], 'turn': 0} for _ in range(8)]
    state['players'][0]['scouted'][1] = {'board': remembered, 'turn': 1}
    state['players'][1]['board'][0]['atk'] += 100
    view = json.loads(tg.Game.from_json(json.dumps(state)).strategy_view(0))
    opponent = next(o for o in view['opponents'] if o['id'] == 1)
    assert opponent['seen_turn'] == 1
    assert opponent['seen_board'][0]['atk'] == remembered[0]['atk']
    assert 'board' not in opponent and 'last_board' not in opponent


def test_visible_hero_progress_distinguishes_equal_legacy_inputs_and_reward():
    game = fixture()
    game.debug(json.dumps({'cmd': 'set_hero', 'p': 0, 'hero': 'BG20_HERO_242'}))
    add(game, 0, 'shop')
    state = json.loads(game.state_json())
    arms = []
    for progress in (0, 19):
        modified = deepcopy(state)
        modified['players'][0]['powers'][0]['counter'] = progress
        arms.append(tg.Game.from_json(json.dumps(modified)))
    before = [json.loads(g.strategy_view(0)) for g in arms]
    assert arms[0].observe(0) == arms[1].observe(0)
    assert arms[0].legal_mask(0) == arms[1].legal_mask(0)
    assert arms[0].attachment_observation(0) == arms[1].attachment_observation(0)
    assert before[0]['me']['hero_powers'][0]['counter'] == 0
    assert before[1]['me']['hero_powers'][0]['counter'] == 19
    assert before[0]['me']['hero_powers'][0]['text'] != before[1]['me']['hero_powers'][0]['text']
    for g in arms:
        assert g.legal_mask(0)[8]
        g.step(0, 8)
    after = [json.loads(g.strategy_view(0))['me'] for g in arms]
    assert len(after[0]['hand']) == 1
    assert len(after[1]['hand']) == 2  # The same buy now also earns Guff's Triple Reward.


def test_privileged_view_and_invalid_seat_are_refused():
    game = fixture()
    state = json.loads(game.state_json())
    state['players'][0]['sees_all_boards'] = True
    game = tg.Game.from_json(json.dumps(state))
    with pytest.raises(ValueError, match='privileged'):
        game.strategy_view(0)
    with pytest.raises(ValueError):
        game.strategy_view(8)


def test_pending_source_matches_the_legacy_public_card_slot():
    game = fixture()
    game.debug(json.dumps({'cmd': 'set_hero', 'p': 0, 'hero': 'TB_BaconShop_HERO_22'}))
    add(game, 0, 'board')
    game.step(0, 4)
    index = struct.unpack_from('<i', game.observe(0)[1], 4 * 4)[0]
    actual = json.loads(game.strategy_view(0))['pending_source']
    assert actual['id'] == tg.card_info(index)[0]
    assert actual['id'] == 'TB_BaconShop_HP_024'


def test_hidden_guess_answer_is_not_a_visible_power_memory():
    game=fixture()
    game.debug(json.dumps({'cmd':'set_hero','p':0,'hero':'BG23_HERO_303'}))
    add(game,0,'board')
    state=json.loads(game.state_json())
    before=json.loads(game.strategy_view(0))
    state['players'][0]['powers'][0]['memory']=[state['players'][0]['board'][0]['card']]
    altered=tg.Game.from_json(json.dumps(state))
    assert json.loads(altered.strategy_view(0))==before


def test_visible_purchase_progress_follows_a_real_kurtrus_purchase():
    game=fixture()
    game.debug(json.dumps({'cmd':'set_hero','p':0,'hero':'BG20_HERO_280'}))
    add(game,0,'shop')
    before=json.loads(game.strategy_view(0))['me']['hero_powers'][0]
    assert before['visible_progress']['purchases_remaining']==3
    game.step(0,8)
    after=json.loads(game.strategy_view(0))['me']['hero_powers'][0]
    assert after['visible_progress']['purchases_remaining']==2
