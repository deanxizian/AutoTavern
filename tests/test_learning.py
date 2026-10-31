"""Learning integration and information-boundary checks, optional for simulator users."""

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning.env import TavernGymEnv
from taverngym.learning.model import EntityActorCritic, ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO, PlacementBuffer


@pytest.fixture
def network():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = EntityActorCritic(ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))
    # Synthetic vectors are only used for unit tests, never for the saved initialized model.
    text = torch.randn(tg.N_CARDS, 16)
    text[0].zero_()
    model.set_text_embeddings(text)
    return model.eval()


def tensor_observation():
    obs, _ = TavernGymEnv(seed=1).reset(seed=1)
    return {key: torch.from_numpy(value.copy()).unsqueeze(0) for key, value in obs.items()}


def test_small_network_parameter_budget_and_independent_layers():
    model = EntityActorCritic()
    assert model.config.d_model == 192 and len(model.transformer.layers) == 4
    assert 3_400_000 < model.parameter_counts()["trainable"] < 3_600_000
    assert not torch.equal(model.transformer.layers[0].linear1.weight,
                           model.transformer.layers[1].linear1.weight)
    with pytest.raises(RuntimeError, match="rule-text"):
        model(tensor_observation())


def test_dynamic_features_and_cardless_choices_create_distinct_tokens(network):
    obs = tensor_observation()
    layout = network.layout
    board = layout.floats["board"][0]
    obs["features"][:, board].zero_()
    items = obs["features"][:, board].reshape(1, 7, -1)
    items[:, :2, 0] = 1
    items[:, 0, 1:3] = torch.tensor([0.2, 0.4])
    items[:, 1, 1:3] = torch.tensor([2.0, 3.0])
    obs["cards"][:, layout.cards["board"]] = 0
    obs["cards"][:, layout.cards["board"].start:layout.cards["board"].start + 2] = 10
    choice = layout.floats["choices"][0].start
    obs["features"][:, choice] = 1
    obs["cards"][:, layout.cards["choices"].start] = 0
    tokens, present = network.tokenize(obs)
    start = layout.tokens["board"].start
    assert present[0, start:start + 2].all()
    assert not torch.allclose(tokens[0, start], tokens[0, start + 1])
    assert present[0, layout.tokens["choices"].start]


def test_padding_cannot_change_visible_predictions_and_invalid_actions_have_zero_probability(network):
    obs = tensor_observation()
    with torch.no_grad():
        before = network(obs)
        # The empty hand slots are ignored as attention keys, even with arbitrary IDs.
        obs["cards"][:, network.layout.cards["hand"]] = 10
        after = network(obs)
    torch.testing.assert_close(before.values, after.values)
    torch.testing.assert_close(before.logits, after.logits)
    probabilities = after.logits.softmax(-1)
    assert torch.count_nonzero(probabilities[~obs["action_mask"]]) == 0
    assert torch.isfinite(after.values).all() and torch.isfinite(after.placement_logits).all()
    obs["action_mask"].zero_()
    with pytest.raises(ValueError, match="legal action"):
        network(obs)


def test_native_observations_drive_all_three_heads_and_gradients(network):
    env = TavernGymEnv(seed=3)
    obs, _ = env.reset(seed=3)
    for _ in range(12):
        tensors = {key: torch.from_numpy(value.copy()).unsqueeze(0) for key, value in obs.items()}
        output = network(tensors)
        assert output.logits.shape == (1, 245)
        assert output.values.shape == (1, 1) and output.placement_logits.shape == (1, 8)
        action = int(torch.distributions.Categorical(logits=output.logits).sample().item())
        obs, _, done, _, info = env.step(action)
        assert info["illegal"] is None
        if done:
            obs, _ = env.reset()
    legal = torch.nonzero(tensors["action_mask"][0], as_tuple=False)[0, 0]
    loss = (-output.logits.log_softmax(-1)[0, legal] + output.values.square().mean()
            + torch.nn.functional.cross_entropy(output.placement_logits, torch.tensor([3])))
    loss.backward()
    for module in (network.identity, network.static_projection, network.text_projection,
                   network.transformer, network.pointer, network.value_head, network.placement_head):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters())
    assert network.text_features.grad is None
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


class ShortEpisodes(gym.Env):
    """Three-step synthetic episodes to test completed labels crossing PPO minibatches."""

    def __init__(self):
        native = TavernGymEnv(seed=5)
        self.observation_space, self.action_space = native.observation_space, native.action_space
        self.obs, _ = native.reset(seed=5)
        self.episode = -1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.episode += 1
        return {key: value.copy() for key, value in self.obs.items()}, {}

    def action_masks(self):
        return self.obs["action_mask"].copy()

    def step(self, action):
        assert self.action_masks()[int(action)]
        self.step_index += 1
        done = self.step_index == 3
        place = 3 if self.episode % 2 == 0 else 5
        reward = (9 - 2 * place) / 7 if done else 0.0
        return self.obs.copy(), reward, done, False, {"placement": place if done else 0}


def test_ppo_updates_auxiliary_head_and_round_trips_without_text_cache(network, tmp_path):
    model = EntityPPO(
        EntityPolicy, ShortEpisodes(), policy_kwargs={"model_config": asdict(network.config)},
        n_steps=8, batch_size=4, n_epochs=1, device="cpu", seed=2,
    )
    model.policy.network.set_text_embeddings(network.text_features)
    before = model.policy.network.placement_head[-1].weight.detach().clone()
    model.learn(total_timesteps=8)
    assert not torch.equal(before, model.policy.network.placement_head[-1].weight)
    assert np.array_equal(model.rollout_buffer.placement_targets, [2, 2, 2, 4, 4, 4, -1, -1])
    assert model.num_timesteps == 8
    assert model.logger.name_to_value["train/placement_loss"] > 0
    obs, _ = ShortEpisodes().reset()
    expected, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "test.zip")
    restored = EntityPPO.load(tmp_path / "test.zip", device="cpu")
    actual, _ = restored.predict(obs, deterministic=True)
    assert expected == actual and restored.num_timesteps == 8
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[name]), name
    restored.set_env(ShortEpisodes())
    restored.learn(total_timesteps=8, reset_num_timesteps=False)
    assert restored.num_timesteps == 16


def test_truncation_never_assigns_placement_or_crosses_episode_boundary():
    env = TavernGymEnv()
    buffer = PlacementBuffer(5, env.observation_space, env.action_space, n_envs=2)
    buffer.pos = 1
    buffer.record_endings([True, False], [{"placement": 4, "TimeLimit.truncated": True}, {}])
    assert (buffer.placement_targets == -1).all()
    buffer.pos = 3
    buffer.record_endings([True, True], [{"placement": 2}, {"placement": 7}])
    assert np.array_equal(buffer.placement_targets[:, 0], [-1, -1, 1, 1, -1])
    assert np.array_equal(buffer.placement_targets[:, 1], [6, 6, 6, 6, -1])
