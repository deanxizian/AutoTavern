"""SB3 policy adapter retaining entity pointer logits instead of an extra flat MLP."""

from dataclasses import asdict

import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor

from .model import EntityActorCritic, ModelConfig, ModelOutput
from .precision import validate_forward_precision


class EntityPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 model_config=None, card_data_sha256=None, forward_precision="fp32", **kwargs):
        from .text import card_manifest

        self.model_config = ModelConfig(**(model_config or {}))
        self.set_forward_precision(forward_precision)
        current_digest = card_manifest()[2]
        if card_data_sha256 is not None and card_data_sha256 != current_digest:
            raise ValueError("Checkpoint card vocabulary differs from this simulator")
        self.card_data_sha256 = current_digest
        if not isinstance(action_space, spaces.Discrete) or action_space.n != 245:
            raise ValueError("EntityPolicy requires TavernGym's 245 discrete actions")
        kwargs.update(features_extractor_class=CombinedExtractor, normalize_images=False,
                      net_arch=[], ortho_init=False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        self.network = EntityActorCritic(self.model_config)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(model_config=asdict(self.model_config), card_data_sha256=self.card_data_sha256,
                    forward_precision=self.forward_precision)
        return data

    def set_forward_precision(self, precision):
        self.forward_precision = validate_forward_precision(precision)

    def _network_output(self, obs):
        # Every policy entry point uses this path, including bootstrap values.
        # Explicitly disable ambient autocast for the legacy FP32 setting.
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.forward_precision == "bf16"):
            output = self.network(obs, mask_actions=False)
        return ModelOutput(*(value.float() for value in output))

    def _distribution(self, output, obs, action_masks=None):
        distribution = self.action_dist.proba_distribution(action_logits=output.logits.float())
        # Never silently sample illegal actions when predict() is called without a mask.
        masks = obs["action_mask"] if action_masks is None else action_masks
        if not torch.as_tensor(masks).bool().any(dim=-1).all():
            raise ValueError("Cannot select an action without legal candidates")
        distribution.apply_masking(masks)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        output = self._network_output(obs)
        distribution = self._distribution(output, obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape(-1), output.values, distribution.log_prob(actions)

    def get_distribution(self, obs, action_masks=None):
        return self._distribution(self._network_output(obs), obs, action_masks)

    def evaluate_actions(self, obs, actions, action_masks=None):
        output = self._network_output(obs)
        distribution = self._distribution(output, obs, action_masks)
        # Exposed for the auxiliary loss in EntityPPO's matching minibatch.
        self.placement_logits = output.placement_logits
        return output.values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs):
        return self._network_output(obs).values
