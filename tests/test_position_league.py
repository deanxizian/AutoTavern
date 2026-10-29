"""Stateful frozen opponents must survive validation and exact PPO recovery."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.arena import play_arena
from training.entity import EntityPolicy
from training.league import LeagueEnv, load_opponents, opponent_identity
from training.position import file_hash, freeze_bundle, preserve_position_context
from test_position import checkpoint, ridge_for


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(127)
    model = EntityPolicy(width=8, layers=1, heads=2, text_dim=8)
    actor = tmp_path / "actor.pt"
    checkpoint(actor, model)
    ridge = ridge_for(model)
    ridge["encoder_sha256"] = file_hash(actor)
    torch.save(ridge, tmp_path / "ridge.pt")
    freeze_bundle(argparse.Namespace(actor=actor, encoder=actor, ridge=tmp_path/"ridge.pt",
                                    threshold=.025, directory=tmp_path/"bundle"))
    return actor, tmp_path/"bundle/position.json"


def test_opponent_identity_includes_bundle_and_retains_encoder_lineage(bundle):
    actor, path = bundle
    models, metadata = load_opponents([str(actor)]*3 + ["bot"], "cpu", [path, path, None, None])
    assert models[0] is models[1] and models[0].actor is models[2]
    assert opponent_identity(metadata[0]) == opponent_identity(metadata[1])
    assert opponent_identity(metadata[0]) != opponent_identity(metadata[2])
    assert opponent_identity(metadata[2]) == file_hash(actor)
    assert metadata[0]["position_encoder"]["train_seed"] == 7
    config = json.loads(path.read_text()); config["threshold"] = .03
    other = path.with_name("other.json"); other.write_text(json.dumps(config))
    _, changed = load_opponents([str(actor)], "cpu", [other])
    assert opponent_identity(changed[0]) != opponent_identity(metadata[0])
    with pytest.raises(ValueError, match="align"):
        load_opponents([str(actor)], "cpu", [])
    with pytest.raises(ValueError, match="heuristic"):
        load_opponents(["bot"], "cpu", [path])
    for config, guard, match in [({"turn_action_caps": [32]}, True, "rules"),
                                  ({"turn_action_caps": [128], "hotfix_37_0_1": True}, False, "guard")]:
        with pytest.raises(ValueError, match=match):
            LeagueEnv(1, seed=17, opponents=models[:1], opponent_ids=[opponent_identity(metadata[0])],
                      device="cpu", config=config, cancel_guard=guard)


def test_real_league_position_moves_restore_across_validation_and_lane_reset(bundle, monkeypatch):
    actor, path = bundle
    config = {"turn_action_caps": [128], "hotfix_37_0_1": True}
    def make():
        models, meta = load_opponents([str(actor)], "cpu", [path])
        def end_first(f, c, m):
            return torch.zeros_like(m, dtype=torch.float32).masked_fill(~m, -1e9), torch.zeros(len(f))
        monkeypatch.setattr(models[0].actor, "forward", end_first)
        env = LeagueEnv(2, seed=17, opponents=models, opponent_ids=[opponent_identity(meta[0])],
                        device="cpu", config=config, cancel_guard=True)
        return env, models[0]
    a, pa = make(); b, pb = make()
    try:
        game = tg.Game(197, json.dumps(config))
        while game.turn < 5:
            for seat in range(8):
                if game.needs_action(seat): game.step(seat, game.bot_action(seat))
        for seat in range(8):
            game.debug(json.dumps(dict(cmd="clear_zone", p=seat, zone="board")))
            for _ in range(4):
                game.debug(json.dumps(dict(cmd="add_minion", p=seat, zone="board", card="BG28_300")))
        pa.reset_context()
        for lane in range(2):
            a.games[lane] = game.copy(); a.cursors[lane] = a.learners[lane]
            b.load(lane, a.save(lane))
        for _ in range(40):
            obs = a.observe()
            actions = [a.game(i).bot_action(seat) for i, seat in enumerate(a.learners)]
            actions = [action if obs["action_mask"][i, action] else int(np.flatnonzero(obs["action_mask"][i])[0])
                       for i, action in enumerate(actions)]
            ra, da, ia = a.step(actions); rb, db, ib = b.step(actions)
            np.testing.assert_array_equal(ra, rb); np.testing.assert_array_equal(da, db)
            assert [a.save(i) for i in range(2)] == [b.save(i) for i in range(2)]
            if pa.counts["moves"]: break
        assert pa.counts["moves"] > 0 and pa.move_events == []
        before = [a.save(i) for i in range(2)]
        with preserve_position_context([pa]):
            result = play_arena([None] + [pa]*7, games=8, seed=2371, threads=1,
                                config=config, device="cpu", cancel_guard=True)
            assert result["guards_checked"]
        assert [a.save(i) for i in range(2)] == before
        other_lane = pa.save_context(identities=range(8, 16))
        a._fresh(0)
        assert pa.save_context(identities=range(8))["states"] == []
        assert pa.save_context(identities=range(8, 16)) == other_lane
        tampered = json.loads(b.save(0)); tampered["position_history"][0]["decoder"]["threshold"] = 99
        with pytest.raises(ValueError, match="position decoding"):
            b.load(0, json.dumps(tampered))
    finally:
        a.close(); b.close()


def test_ppo_with_position_opponents_and_each_update_validation_resumes_exactly(bundle, tmp_path):
    actor, path = bundle
    model = EntityPolicy(width=8, layers=1, heads=2, text_dim=8, attachment_version=1)
    initial = tmp_path/"initial.pt"; checkpoint(initial, model)
    base = [sys.executable, "-m", "training.train", "--device", "cpu", "--envs", "4",
            "--rollout", "64", "--minibatch", "64", "--threads", "1", "--torch-threads", "2",
            "--width", "8", "--layers", "1", "--heads", "2", "--seed", "317",
            "--eval-seed", "2371", "--eval-games", "8", "--eval-every", "1",
            "--warmup-samples", "0", "--opponents", str(actor), "bot",
            "--opponent-position-bundles", str(path), "none", "--cancel-guard", "--hotfix-37-0-1",
            "--public-attachments", "--attachment-lr", "0.0001", "--league-turn-limit", "100",
            "--skip-baselines", "--checkpoint-seconds", "999999", "--max-hours", "0"]
    for run, updates, start in [("whole", 2, ["--init-from", str(initial)]),
                                 ("split", 1, ["--init-from", str(initial)]),
                                 ("split", 2, ["--resume", str(tmp_path/"split/latest.pt")])]:
        result = subprocess.run(base + ["--run-dir", str(tmp_path/run), "--max-updates", str(updates)] + start,
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=240)
        (tmp_path/f"{run}-{updates}.log").write_text(result.stdout+result.stderr)
        assert result.returncode == 0, (result.stdout+result.stderr)[-6000:]
    a, b = [torch.load(tmp_path/f"{name}/latest.pt", map_location="cpu", weights_only=False)
            for name in ("whole", "split")]
    assert a["update"] == b["update"] == 2 and a["steps"] == b["steps"] == 512
    for group in ("model", "anchor"):
        assert all(torch.equal(v, b[group][k]) for k, v in a[group].items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for key, state in a["optimizer"]["state"].items():
        assert all(torch.equal(v, b["optimizer"]["state"][key][k]) for k, v in state.items())
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert np.array_equal(a["numpy_rng"][1], b["numpy_rng"][1])
    assert json.loads(gzip.decompress(a["environment"])) == json.loads(gzip.decompress(b["environment"]))
    # Budget stops precede endpoint validation: the whole run alone validates
    # update 1, yet its subsequent weights and environment still match exactly.
    for name, update in (("whole", 1), ("split", 0)):
        report = json.loads((tmp_path/name/f"league_eval_{update:06d}.json").read_text())
        assert report["guards_checked"] and report["position_decoders"]
    changed = json.loads(path.read_text()); changed["threshold"] = .03
    other = path.with_name("changed.json"); other.write_text(json.dumps(changed))
    forbidden = list(base); forbidden[forbidden.index(str(path))] = str(other)
    result = subprocess.run(forbidden + ["--run-dir", str(tmp_path/"rejected"), "--max-updates", "3",
                                         "--resume", str(tmp_path/"split/latest.pt")],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and "pool or position decoder" in result.stderr
