"""Mixed seats, warm starts and evaluation must preserve the one-learner boundary."""

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning.experiment import evaluate_suite, load_frozen_policy, log_evaluation
from taverngym.learning.model import ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import MixedOpponentVectorEnv, NativeVectorEnv


class FirstLegalPolicy:
    def predict(self, obs, deterministic=False):
        assert obs["action_mask"].any(axis=1).all()
        return obs["action_mask"].argmax(axis=1), None


def small_model(env):
    model = EntityPPO(EntityPolicy, env, n_steps=8, batch_size=8, n_epochs=1, seed=4, device="cpu",
                      policy_kwargs={"model_config": asdict(ModelConfig(
                          d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    text = torch.randn(tg.N_CARDS, 16)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text)
    return model


def test_mixed_driver_holds_ready_games_and_drains_dead_learners():
    class SlowPolicy(FirstLegalPolicy):
        def predict(self, obs, deterministic=False):
            actions, state = super().predict(obs, deterministic)
            actions[obs["action_mask"][:, 2]] = 2
            return actions, state

    class Inspected(MixedOpponentVectorEnv):
        def _advance(self, full, active, learner_actions=None):
            if learner_actions is None:
                assert not full["ready"][::self.roles][active].any()
            held = {i: self.batch.save(i) for i in np.flatnonzero(~active)}
            out = super()._advance(full, active, learner_actions)
            for i, state in held.items():
                assert self.batch.save(i) == state
            return out

    env = Inspected(3, seed=12, threads=3, opponent_policy=SlowPolicy(),
                    config={"turn_action_caps": [6]}, reward_shaping={})
    obs = env.reset()
    finished, seats, steps = 0, set(), np.zeros(3, dtype=np.int64)
    try:
        for _ in range(180):
            indices = [env.batch.episode_index(i) for i in range(3)]
            steps += 1
            actions = [2 if mask[2] and i % 2 == 0 else np.flatnonzero(mask)[0]
                       for i, mask in enumerate(obs["action_mask"])]
            obs, rewards, dones, infos = env.step(actions)
            assert obs["action_mask"].shape == (3, 245) and obs["action_mask"].any(axis=1).all()
            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    assert 1 <= info["placement"] <= 8
                    assert info["episode"]["l"] == steps[i]
                    assert info["reward_components"]["actions_per_game"] == steps[i]
                    assert env.batch.episode_index(i) == indices[i] + 1
                    assert info["reward_components"]["placement_reward"] == pytest.approx(
                        (9 - 2 * info["placement"]) / 7)
                    steps[i] = 0
                    seats.add(info["learner_seat"])
                    finished += 1
        assert finished >= 6 and len(seats) >= 4
        assert env.opponent_decisions > 0 and env.waiting_rounds > 0
    finally:
        env.close()


def test_warm_start_keeps_weights_and_restarts_adam_and_sample_counts(tmp_path):
    torch.set_num_threads(2)
    env = NativeVectorEnv(1, seed=4)
    model = small_model(env)
    model.learn(16)
    assert model.optimizer_step_count() == 2
    saved = tmp_path / "old.zip"
    model.save(saved)
    frozen = load_frozen_policy(saved, "cpu")
    assert not frozen.optimizer.state and not any(p.requires_grad for p in frozen.parameters())
    inherited = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model.start_new_stage()
    assert model.num_timesteps == model.optimizer_step_count() == model._n_updates == 0
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, inherited[name])
    model.learn(8, reset_num_timesteps=False)
    assert model.num_timesteps == 8 and model.optimizer_step_count() == 1
    for name, value in frozen.state_dict().items():
        assert torch.equal(value, inherited[name])
    env.close()


@pytest.mark.parametrize("include_scripted", [None, False, True])
def test_evaluation_matrix_respects_scripted_scope_and_flat_names(include_scripted):
    from stable_baselines3.common.logger import configure

    model = small_model(NativeVectorEnv(1, seed=4))
    config = {"evaluation_matrix": True, "frozen_opponents": 4,
              "simulator_config": {"turn_action_caps": [3]}, "reward_shaping": {}}
    if include_scripted is not None:
        config["evaluation_include_scripted"] = include_scripted
    rng = torch.random.get_rng_state().clone()
    result = evaluate_suite(model, 2, 101, config, FirstLegalPolicy())
    assert torch.equal(rng, torch.random.get_rng_state())
    expected = {"mixed_sampled", "mixed_greedy"}
    if include_scripted is not False:
        expected.update(("scripted_sampled", "scripted_greedy"))
    assert set(result["conditions"]) == expected
    assert all("4 frozen policies" in row["opponents"]
               for name, row in result["conditions"].items() if name.startswith("mixed_"))
    assert result["mean_placement"] == result["conditions"]["mixed_sampled"]["mean_placement"]
    assert result["selection_condition"] == "mixed_sampled"
    assert all(row["games"] == 2 and "cycle_returns" in row for row in result["conditions"].values())
    logger = configure(format_strings=[])
    log_evaluation(logger, "val", result)
    prefixes = ("", "greedy_")
    if include_scripted is not False:
        prefixes += ("scripted_", "scripted_greedy_")
    for prefix in prefixes:
        assert f"val/{prefix}cycle_penalty" in logger.name_to_value
        assert f"val/{prefix}mean_placement" in logger.name_to_value
    assert all(name.count("/") == 1 for name in logger.name_to_value)
    if include_scripted is False:
        assert not any(name.startswith("val/scripted_") for name in logger.name_to_value)
    model.env.close()


def test_new_stage_retains_adam_moments_and_resets_only_local_clock(tmp_path):
    import copy

    torch.set_num_threads(2)
    model = small_model(NativeVectorEnv(1, seed=4))
    model.learn(16)
    before = copy.deepcopy(model.policy.optimizer.state_dict())
    weights = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model.start_new_stage(preserve_adam=True)
    assert model.num_timesteps == model._n_updates == model.optimizer_step_count() == 0
    assert model.optimizer_total_step_count() == 2
    for key, state in before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, model.policy.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    assert all(torch.equal(value, weights[name]) for name, value in model.policy.state_dict().items())
    model.learn(8, reset_num_timesteps=False)
    path = tmp_path / "continued.zip"
    model.save(path)
    restored = EntityPPO.load(path, device="cpu")
    assert restored.optimizer_step_count() == 1 and restored.optimizer_total_step_count() == 3
    restored.start_new_stage(preserve_adam=True)
    assert restored.optimizer_step_count() == 0 and restored.optimizer_total_step_count() == 3
    restored.start_new_stage()
    assert restored.optimizer_step_count() == restored.optimizer_total_step_count() == 0
    model.env.close()
