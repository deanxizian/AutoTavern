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


def test_skipped_kl_minibatch_releases_activations_before_next_rollout(tmp_path, monkeypatch):
    """A skipped backward must not retain a full training graph across updates."""
    import gc
    import json
    import sys
    import weakref
    from training import train

    torch.set_num_threads(2)
    torch.manual_seed(741)
    model = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1)
    initial = tmp_path / "initial.pt"
    torch.save(dict(engine_hash=tg.source_hash(), training_code_hash="test-only", weights_only=True,
                    architecture=model.architecture, model=model.state_dict(), update=0, steps=0,
                    args=dict(seed=741, eval_seed=1741, action_cap=128, hotfix_37_0_1=True)), initial)
    tracked, transitions = [], []
    learner_was_grad = False
    original = EntityPolicy.forward_aux

    def pack(tensor):
        saved = tensor.detach()
        if saved.ndim == 3 and saved.shape[1] == 49 and saved.is_floating_point():
            tracked.append(weakref.ref(saved))
        return saved

    def audited(self, *args, **kwargs):
        nonlocal learner_was_grad
        learner = any(p.requires_grad for p in self.parameters())
        grad = torch.is_grad_enabled()
        if learner and learner_was_grad and not grad:
            gc.collect()
            transitions.append(sum(ref() is not None for ref in tracked))
        if learner:
            learner_was_grad = grad
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EntityPolicy, "forward_aux", audited)
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["test-train", "--run-dir", str(tmp_path / "run"),
        "--init-from", str(initial), "--device", "cpu", "--envs", "8", "--threads", "2",
        "--torch-threads", "2", "--rollout", "64", "--minibatch", "64", "--epochs", "1",
        "--width", "16", "--layers", "1", "--heads", "2", "--seed", "741", "--eval-seed", "1741",
        "--eval-games", "8", "--eval-every", "100", "--warmup-samples", "0", "--opponents", "bot",
        "--cancel-guard", "--hotfix-37-0-1", "--public-attachments", "--attachment-lr", ".001",
        "--lr", ".001", "--target-kl", ".000001", "--league-turn-limit", "100", "--skip-baselines",
        "--checkpoint-seconds", "999999", "--max-hours", "0", "--max-updates", "2"])
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        train.main()
    metrics = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    updates = [row for row in metrics if row["event"] == "update"]
    assert len(updates) == 2 and all(row["kl_early_stop"] for row in updates)
    assert transitions == [0], "Skipped minibatch activations survived into the next rollout"
    args = json.loads((tmp_path / "run/manifest.json").read_text())["args"]
    for row in updates:
        parts = [row[name + "_loss_mean"] for name in ("policy", "value", "placement", "anchor")]
        assert all(np.isfinite(part) for part in parts)
        policy, value, placement, anchor = parts
        reconstructed = (policy + args["value_coef"] * value
                         + args["aux_coef"] * placement + args["anchor_coef"] * anchor
                         - args["entropy"] * row["entropy"])
        assert row["loss"] == pytest.approx(reconstructed, abs=1e-6)
        assert np.isfinite(row["gradient_norm_max"])
        assert 0 <= row["gradient_norm_mean"] <= row["gradient_norm_max"]
        assert 0 <= row["gradient_over_limit_fraction"] <= 1
