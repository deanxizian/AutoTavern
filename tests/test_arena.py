"""Arena regressions: seat bias, auto-reset leakage and waiting-seat inference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.arena import play_arena, rotations, wilson


class EndTurn(torch.nn.Module):
    def forward(self, features, cards, masks):
        assert masks.any(axis=1).all(), "waiting seats reached inference"
        logits = torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9)
        return logits, torch.zeros(len(masks))


class UnequalDurationEnv:
    """Lane zero resets before lane seven ends; reset results must be ignored."""
    def __init__(self, n_games, **kwargs):
        self.n_slots = n_games * 8
        self.n_games = n_games
        self.tick = 0

    def observe(self):
        ready = np.tile(np.arange(8) % 2 == self.tick % 2, self.n_games)
        return {"ready": ready, "features": np.zeros((self.n_slots, 1), np.float32),
                "cards": np.zeros((self.n_slots, 1), np.int32),
                "action_mask": ready[:, None]}

    def step(self, actions):
        self.tick += 1
        done = np.repeat(self.tick % (np.arange(self.n_games) + 1) == 0, 8)
        # Different placements in subsequent episodes reveal accidental overwrites.
        places = np.concatenate([np.roll(np.arange(1, 9), self.tick // (lane + 1) - 1)
                                 for lane in range(self.n_games)]).astype(np.uint8)
        return np.zeros(self.n_slots), done, {"placement": places,
                                             "rejected": np.zeros(self.n_slots, bool)}


def test_every_role_occupies_every_seat_equally():
    roles = rotations(24)
    for role in range(8):
        np.testing.assert_array_equal((roles == role).sum(axis=0), np.full(8, 3))
    with pytest.raises(ValueError):
        rotations(9)


def test_first_lobby_only_and_waiting_slots_never_reach_policy():
    model = EndTurn().train()
    result = play_arena([model] * 8, games=8, seed=17, threads=1, config={},
                        device=torch.device("cpu"), env_factory=UnequalDurationEnv,
                        audit_guards=False)
    assert model.training, "evaluation must restore the caller's training mode"
    assert result["driver_steps"] == 8
    assert result["candidate"]["placement_counts"] == [1] * 8
    assert result["candidate"]["mean_placement"] == 4.5
    assert all(row["placement_by_seat"] == list(range(1, 9)) for row in result["lobbies"])


def test_real_arena_finishes_whole_lobbies_reproducibly():
    kwargs = dict(games=8, seed=42, config={"turn_action_caps": [32]},
                  device=torch.device("cpu"))
    a = play_arena([None] * 8, threads=1, **kwargs)
    b = play_arena([None] * 8, threads=2, **kwargs)
    assert a["lobbies"] == b["lobbies"]
    assert a["candidate"]["games"] == 8
    for row in a["lobbies"]:
        assert sorted(row["placement_by_seat"]) == list(range(1, 9))


def test_rate_intervals_remain_conservative_at_endpoints():
    assert wilson(0, 100)[0] == 0
    assert 0 < wilson(0, 100)[1] < .05
    assert .95 < wilson(100, 100)[0] < 1
    assert wilson(100, 100)[1] == 1


def test_stateful_candidate_receives_actual_ready_slot_ids_and_terminal_resets():
    class IdentifiedEnv(UnequalDurationEnv):
        def observe(self):
            obs = super().observe()
            obs["features"][:,0] = np.arange(self.n_slots)
            return obs
    class WithContext(EndTurn):
        def __init__(self):
            super().__init__()
            self.ids = None
            self.resets = []
            self.seen = set()
        def set_context(self, ids):
            assert self.ids is None
            self.ids = list(map(int,ids))
        def reset_context(self, identity=None):
            self.resets.append(identity)
        def forward(self, features,cards,masks):
            assert self.ids == features[:,0].long().tolist()
            self.seen.update(self.ids)
            self.ids = None
            return super().forward(features,cards,masks)
    model = WithContext()
    play_arena([model]+[EndTurn()]*7,games=8,seed=17,threads=1,config={},
               device=torch.device("cpu"),env_factory=IdentifiedEnv,audit_guards=False)
    assert model.resets[0] is None
    assert model.seen == {lane*8+lane for lane in range(8)}
    assert model.seen <= set(model.resets[1:])


def test_shared_position_opponents_keep_slot_history_and_report_role_ownership():
    class IdentifiedEnv(UnequalDurationEnv):
        def observe(self):
            obs = super().observe()
            obs["features"][:, 0] = np.arange(self.n_slots)
            return obs

    class Recorded(EndTurn):
        decoder_metadata = {"kind": "greedy_plus_public_position_v1", "ridge_sha256": "ridge"}

        def __init__(self):
            super().__init__()
            self.reset_context()

        def reset_context(self, identity=None):
            if identity is None:
                self.seen, self.resets, self.move_events = set(), [], []
                self.counts = {"moves": 0}
            else:
                self.resets.append(identity)

        def set_context(self, identities):
            self.identities = list(map(int, identities))

        def forward(self, features, cards, masks):
            assert self.identities == features[:, 0].long().tolist()
            for identity in self.identities:
                if identity not in self.seen:
                    self.move_events.append({"identity": identity})
                    self.counts["moves"] += 1
                    self.seen.add(identity)
            return super().forward(features, cards, masks)

    enhanced, plain = Recorded(), EndTurn()
    policies = [enhanced] * 4 + [plain] * 4
    kwargs = dict(games=8, seed=17, threads=1, config={}, device=torch.device("cpu"),
                  env_factory=IdentifiedEnv, audit_guards=False)
    with pytest.raises(ValueError, match="cancellation guard"):
        play_arena(policies, **kwargs)
    report = play_arena(policies, cancel_guard=True, **kwargs)
    roles = rotations(8).reshape(-1)
    # The fake environment exposes only alternate ready seats before early lanes
    # end. Every inferred slot must nevertheless belong to this shared decoder.
    assert enhanced.seen and all(roles[slot] < 4 for slot in enhanced.seen)
    assert enhanced.seen <= set(enhanced.resets)
    assert report["position_decoders"][0]["roles"] == [0, 1, 2, 3]
    assert report["position_decoders"][0]["statistics"]["moves"] == len(enhanced.seen)
    candidate = report["position_decoder"]
    assert "statistics" not in candidate
    assert {event["identity"] for event in candidate["events"]} == {
        slot for slot in enhanced.seen if roles[slot] == 0}
    assert candidate["counted_episode_moves"] == len(candidate["events"])
    assert report["decoding"]["kind"] == "role_specific_public_position_v1"
    assert "ridge_sha256" not in report["decoding"]
    assert [row["kind"] for row in report["role_decoding"]] == [
        "greedy_plus_public_position_v1"] * 4 + ["greedy"] * 4


@pytest.mark.parametrize("candidate_position", [False, True])
def test_cli_keeps_candidate_separate_from_shared_position_roles(monkeypatch, tmp_path, candidate_position):
    import json
    import sys
    import training.arena as arena
    import training.position as position

    actor, wrappers = EndTurn(), []
    meta = dict(trained_hotfix_37_0_1=True, train_seed=7, validation_seed=1000,
                action_cap=128, sha256="actor")
    loaded, wrapped_from = [], []

    def load(path, device):
        loaded.append(path)
        return actor, meta.copy()

    def wrap(model, metadata, bundle, device):
        wrapped_from.append((model, metadata, bundle))
        wrapped = EndTurn()
        wrapped.decoder_metadata = {"kind": "greedy_plus_public_position_v1"}
        wrappers.append(wrapped)
        return wrapped

    def play(policies, **kwargs):
        assert policies == [wrappers[0] if candidate_position else actor,
                            wrappers[-1], wrappers[-1]] + [actor] * 5
        assert policies[0] is not policies[1]
        assert kwargs["cancel_guard"] is True
        assert torch.get_float32_matmul_precision() == "highest"
        return {"candidate": {"games": 8}, "elapsed_seconds": 0.}

    monkeypatch.setattr(arena, "frozen_checkpoint", load)
    monkeypatch.setattr(position, "load_position_policy", wrap)
    monkeypatch.setattr(arena, "play_arena", play)
    output = tmp_path / "roles.json"
    extra = ["--position-bundle", "position.json"] if candidate_position else []
    monkeypatch.setattr(sys, "argv", ["arena", "actor.pt", "--opponents", *["actor.pt"] * 7,
        "--opponent-position-bundles", "position.json", "position.json", *["none"] * 5,
        "--cancel-guard", "--games", "8", "--seed", "2001", "--device", "cpu", "--out", str(output), *extra])
    arena.main()
    assert len(loaded) == 1 and len(wrapped_from) == 1 + int(candidate_position)
    assert wrapped_from[0][0] is actor
    assert json.loads(output.read_text())["format_version"] == 4


@pytest.mark.parametrize("failure", ["missing_guard", "bot_role", "rules"])
def test_opponent_bundle_cli_rejects_invalid_assignment(monkeypatch, tmp_path, failure):
    import sys
    import training.arena as arena

    actor = EndTurn()
    meta = dict(trained_hotfix_37_0_1=True, train_seed=7, validation_seed=1000, action_cap=128)
    monkeypatch.setattr(arena, "frozen_checkpoint", lambda *args: (actor, meta.copy()))
    output = tmp_path / "must-not-run.json"
    opponents = ["bot" if failure == "bot_role" else "actor.pt"] + ["bot"] * 6
    extra = [] if failure == "missing_guard" else ["--cancel-guard"]
    if failure == "rules":
        extra.append("--no-hotfix-37-0-1")
    monkeypatch.setattr(sys, "argv", ["arena", "actor.pt", "--opponents", *opponents,
        "--opponent-position-bundles", "position.json", *["none"] * 6,
        "--seed", "2001", "--device", "cpu", "--out", str(output), *extra])
    with pytest.raises(SystemExit) as error:
        arena.main()
    assert error.value.code == 2 and not output.exists()
