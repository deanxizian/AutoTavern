"""Tests for costly failure modes in the independent PPO baseline."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.common import (Policy, bootstrap_time_limits, check_step,
                             generalized_advantage, tensors)
from training.entity import EntityPolicy


def test_terminal_reward_does_not_bootstrap_from_auto_reset():
    rewards = torch.tensor([[0.0], [1.0], [2.0]])
    dones = torch.tensor([[False], [True], [False]])
    values = torch.tensor([[0.2], [0.3], [0.4]])
    adv, returns = generalized_advantage(rewards, dones, values, torch.tensor([5.0]), 1., 1.)
    torch.testing.assert_close(returns, torch.tensor([[1.], [1.], [7.]]))
    torch.testing.assert_close(adv, returns - values)


def test_masking_excludes_invalid_actions_and_backpropagates():
    torch.manual_seed(7)
    model = Policy(hidden=32, embedding=4)
    env = tg.BatchEnv(8, seed=3, threads=2)
    f, c, m = tensors(env.observe(), torch.device("cpu"))
    dist, value = model.distribution(f, c, m)
    samples = dist.sample((100,))
    assert m.gather(1, samples.T).all()
    loss = -dist.log_prob(samples[0]).mean() + value.square().mean() - .01 * dist.entropy().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.actor.weight.grad.abs().sum() > 0


def test_incomplete_simulations_are_rejected():
    with pytest.raises(RuntimeError, match="valid placement"):
        check_step(np.array([True]), {"rejected": np.array([False]), "placement": np.array([0])})


def test_time_limit_uses_final_value_and_cuts_the_reset_trajectory():
    class ValueFromObservation(torch.nn.Module):
        def forward(self, features, cards, masks):
            return torch.zeros_like(masks, dtype=torch.float32), features[:, 0]

    done = np.array([True])
    info = {"rejected": np.array([False]), "placement": np.array([0]),
            "truncated": np.array([True]), "truncated_lanes": np.array([0]),
            "final_observation": {"features": np.array([[7.]], np.float32),
                                  "cards": np.zeros((1, 1), np.int32),
                                  "action_mask": np.ones((1, 1), bool),
                                  "ready": np.ones(1, bool)}}
    with pytest.raises(RuntimeError, match="not a completed evaluation"):
        check_step(done, info)
    check_step(done, info, allow_truncation=True)
    original = np.array([0.], np.float32)
    reward = bootstrap_time_limits(ValueFromObservation(), original, info, .9, torch.device("cpu"))
    np.testing.assert_array_equal(original, [0.])
    np.testing.assert_allclose(reward, [6.3])
    rewards = torch.tensor([[0.], [float(reward[0])], [100.]])
    dones = torch.tensor([[False], [True], [False]])
    _, returns = generalized_advantage(rewards, dones, torch.zeros_like(rewards),
                                      torch.tensor([999.]), .9, 1.)
    torch.testing.assert_close(returns[:2], torch.tensor([[5.67], [6.3]]))
    assert returns[2, 0] > 900  # This unrelated reset value must not cross the cut.
    for invalid in (dict(info, placement=np.array([1])), dict(info, truncated=np.array([1]))):
        with pytest.raises(RuntimeError):
            check_step(done, invalid, allow_truncation=True)
    with pytest.raises(RuntimeError, match="final observation"):
        bootstrap_time_limits(ValueFromObservation(), original,
                              {k: v for k, v in info.items() if k != "final_observation"},
                              .9, torch.device("cpu"))


def test_single_seat_always_ready_and_legal_through_resets():
    env = tg.BatchEnv(4, seed=41, threads=2, config={"turn_action_caps": [128]})
    completed = 0
    for _ in range(300):
        obs = env.observe()
        tensors(obs, torch.device("cpu"))
        actions = [env.game(i).bot_action(0) for i in range(4)]
        _, done, info = env.step(actions)
        check_step(done, info)
        completed += done.sum()
    assert completed > 0


def test_entity_action_layout_and_gradients_after_recruit_actions():
    torch.set_num_threads(2)
    env = tg.BatchEnv(4, seed=19, threads=2)
    model = EntityPolicy(width=32, layers=2, heads=4)
    for _ in range(30):
        env.step([env.game(i).bot_action(0) for i in range(4)])
    f, c, m = tensors(env.observe(), torch.device("cpu"))
    dist, value, place = model.distribution(f, c, m, auxiliary=True)
    assert dist.logits.shape == (4, tg.N_ACTIONS)
    assert place.shape == (4, 8)
    actions = dist.sample()
    assert m.gather(1, actions[:, None]).all()
    loss = -dist.log_prob(actions).mean() + value.square().mean() + place.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.transformer[0].qkv.weight.grad.abs().sum() > 0
    _, done, info = env.step(actions.numpy())
    check_step(done, info)
