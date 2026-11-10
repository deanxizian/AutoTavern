"""Controlled inference-only comparison using the unchanged loop-audit counter."""

from collections import Counter
import importlib.util
import json
from pathlib import Path

import torch

from taverngym.learning.action_filter import ReversibleActionFilter
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


FOLDER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("loop_audit", FOLDER.parent / "20261031-action-loop-diagnostic/run.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(obs, i):
    return {key: value[i] for key, value in obs.items()}


class FilteredVectorEnv(NativeVectorEnv):
    """Experimental adapter only; normal training/evaluation defaults are unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = [ReversibleActionFilter(max_reversible_actions=8) for _ in range(self.num_envs)]
        self.blocked = Counter()
        self._inside_step = False

    def _filter(self, obs):
        for i, guard in enumerate(self.filters):
            mask = guard.mask(row(obs, i))
            self.blocked.update(guard.last_blocked.values())
            obs["action_mask"][i] = mask
        self._obs = obs
        return obs

    def _observe(self):
        obs = super()._observe()
        return obs if self._inside_step else self._filter(obs)

    def reset(self):
        for guard in self.filters:
            guard.reset()
        return super().reset()

    def step_wait(self):
        actions, before = self._actions.copy(), self._obs
        self._inside_step = True
        try:
            after, rewards, dones, infos = super().step_wait()
        finally:
            self._inside_step = False
        for i, guard in enumerate(self.filters):
            guard.commit(int(actions[i]), row(before, i), row(after, i), done=bool(dones[i]))
        return self._filter(after), rewards, dones, infos


def main():
    if (FOLDER / "results.json").exists():
        raise FileExistsError("Choose a new directory to repeat this comparison")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    baseline = json.loads((FOLDER.parent / "20261031-action-loop-diagnostic/results.json").read_text())
    before = audit.digest(audit.MODEL.read_bytes())
    assert before == baseline["model_sha256"]
    model = EntityPPO.load(audit.MODEL, device="cuda", n_envs=1, n_steps=2, batch_size=2)
    steps = model.optimizer_step_count()
    audit.FOLDER = FOLDER
    audit.NativeVectorEnv = FilteredVectorEnv
    cases = []
    modes = {"greedy": audit.run_mode(model, True, cases), "sampled": audit.run_mode(model, False, cases)}
    assert before == audit.digest(audit.MODEL.read_bytes()) and model.optimizer_step_count() == steps
    assert not cases
    for result in modes.values():
        assert result["games_with_move_returns"] == 0 and result["games_with_freeze_returns"] == 0
    audit.write(FOLDER / "results.json", {"model": baseline["model"], "model_sha256": before,
                                         "filter": {"max_reversible_actions": 8},
                                         "seed": baseline["seed"], "n_envs": baseline["n_envs"],
                                         "simulator_config": audit.CONFIG,
                                         "model_weights_unchanged": True, "training_started": False,
                                         "native_game_legality_changed": False,
                                         "baseline": "../20261031-action-loop-diagnostic/results.json",
                                         "modes": modes})


if __name__ == "__main__":
    main()
