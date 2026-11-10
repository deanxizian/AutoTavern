"""37.0.2 native contracts required by the fresh training lineage."""

import hashlib
import json

import pytest

import taverngym as tg


def test_patch_keeps_the_complete_card_vocabulary_and_observation_action_contract():
    cards = json.loads((tg.REPO_ROOT / 'crates/taverngym/data/cards.json').read_text())['cards']
    ids = [row['id'] for row in cards]
    assert len(ids) == 3983 and len(set(ids)) == 3983
    assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == (
        '3b6fa2aeebe0955cb57438a9d8d5b487d61d8beb614f0c94c360dea3e9232270')
    assert [tg.card_info(i)[0] for i in range(1, tg.N_CARDS)] == ids
    assert (tg.N_CARDS, tg.N_ACTIONS, tg.OBS_FLOATS, tg.OBS_CARDS) == (3984, 245, 3227, 124)
    assert tg.game_version() == '37.0.2.256589' and tg.server_rules_version() == '37.0.2'


def test_initialized_vector_env_explicitly_honors_two_hero_choices():
    pytest.importorskip('gymnasium')
    pytest.importorskip('sb3_contrib')
    from taverngym.learning.env import make_vector_env

    # The configured field must reach the native constructor, rather than relying on a default.
    for choices in (1, 2):
        env = make_vector_env(n_envs=2, seed=37024002,
                              config={'hero_choices': choices, 'turn_action_caps': [60]})
        try:
            obs = env.reset()
            assert (obs['action_mask'].sum(axis=1) == choices).all()
        finally:
            env.close()
