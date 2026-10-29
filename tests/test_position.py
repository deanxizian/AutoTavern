"""State, masking and frozen encoder contracts for public position inference."""
import argparse
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.arena import frozen_checkpoint
from training.entity import EntityPolicy
from training.position import PublicPositionPolicy, file_hash, freeze_bundle, load_position_policy


def ridge_for(model, gain=.1):
    size = 6*model.architecture["width"]+51+7*model.layout["MINION_F"]
    return dict(family="frozen_tokens_plus_public_numeric_position_delta",
                weight=torch.zeros(size, dtype=torch.float64),
                feature_mean=torch.zeros(size, dtype=torch.float64),
                feature_scale=torch.ones(size, dtype=torch.float64), intercept=gain,
                encoder_sha256="set-by-the-bundle-test")


@pytest.fixture
def fixture(monkeypatch):
    torch.set_num_threads(2)
    actor = EntityPolicy(width=8, layers=1, heads=2, text_dim=8).eval()
    encoder = EntityPolicy(width=8, layers=1, heads=2, text_dim=8).eval()
    def choose_first(features, cards, masks):
        return torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9), torch.zeros(len(features))
    monkeypatch.setattr(actor, "forward", choose_first)
    policy = PublicPositionPolicy(actor, encoder, ridge_for(encoder)).eval()
    f = torch.zeros(1, tg.OBS_FLOATS)
    f[0, 0], f[0, 33] = 5/20, 4/7
    f[0, policy.opponent_offset+1] = f[0, policy.opponent_offset+8] = 1
    c = torch.zeros((1, tg.OBS_CARDS), dtype=torch.int32)
    m = torch.zeros((1, tg.N_ACTIONS), dtype=torch.bool)
    m[0, [0, 197, 198, 199, 200]] = True
    return policy, (f, c, m)


def choose(policy, inputs, identity=7):
    policy.set_context([identity])
    logits, _ = policy(*inputs)
    action = int(logits.argmax(-1).item())
    assert inputs[2][0, action]
    return action


def test_one_move_then_end_survives_json_restore_and_pending_choices(fixture):
    policy, (f,c,m) = fixture
    precision = torch.get_float32_matmul_precision()
    assert choose(policy, (f,c,m)) == 197
    saved = json.loads(json.dumps(policy.save_context()))
    policy.reset_context()
    policy.load_context(saved)
    pending = torch.zeros_like(m)
    pending[0, 30] = True
    assert choose(policy, (f,c,pending)) == 30
    assert choose(policy, (f,c,m)) == 0
    assert choose(policy, (f,c,m)) == 0
    assert choose(policy, (f,c,m), identity=8) == 197
    f[0,0] = 6/20
    assert choose(policy, (f,c,m)) == 197
    policy.reset_context(7)
    assert choose(policy, (f,c,m)) == 197
    assert torch.get_float32_matmul_precision() == precision
    assert not policy.encoder.norm._forward_hooks


def test_lane_history_restore_and_validation_do_not_consume_other_slots(fixture):
    from training.position import preserve_position_context
    policy, inputs = fixture
    assert choose(policy, inputs, 1) == choose(policy, inputs, 9) == 197
    lane = policy.save_context(identities=range(8))
    policy.reset_context(1)
    policy.load_context(lane, identities=range(8))
    assert choose(policy, inputs, 1) == choose(policy, inputs, 9) == 0
    with pytest.raises(ValueError, match="escapes"):
        policy.load_context(lane, identities=range(8, 16))
    history, counts, events = policy.save_context(), dict(policy.counts), list(policy.move_events)
    policy.record_events = False
    with pytest.raises(RuntimeError, match="test validation error"):
        with preserve_position_context([policy, policy]):
            policy.reset_context()
            assert choose(policy, inputs, 1) == 197
            policy.set_context([99])  # Failure before this request consumes context.
            raise RuntimeError("test validation error")
    assert policy.save_context() == history and policy.counts == counts
    assert policy.move_events == events and not policy.record_events


@pytest.mark.parametrize("change", ["early", "late", "ghost", "unknown", "dead", "small", "few_moves"])
def test_public_eligibility_preserves_the_original_end(fixture, change):
    policy, (f,c,m) = fixture
    if change == "early": f[0,0] = 3/20
    if change == "late": f[0,0] = 21/20
    if change == "ghost": f[0,37] = 1
    if change == "unknown": f[0,policy.opponent_offset+8] = 0
    if change == "dead": f[0,policy.opponent_offset+1] = 0
    if change == "small": f[0,33] = 3/7
    if change == "few_moves": m[0,200] = False
    assert choose(policy,(f,c,m)) == 0
    assert policy.counts["eligible_ends"] == policy.counts["moves"] == 0


def test_context_cannot_be_reused_or_cross_slots(fixture):
    policy, inputs = fixture
    with pytest.raises(ValueError, match="identity"):
        policy(*inputs)
    with pytest.raises(ValueError, match="distinct"):
        policy.set_context([7,7])
    policy.set_context([7])
    with pytest.raises(ValueError, match="consumed"):
        policy.set_context([8])
    with pytest.raises(ValueError, match="completing"):
        policy.save_context()
    policy(*inputs)
    with pytest.raises(ValueError, match="identity"):
        policy(*inputs)
    original = policy.save_context()
    damaged = json.loads(json.dumps(original))
    damaged["states"].append(damaged["states"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        policy.load_context(damaged)
    assert policy.save_context() == original


def test_threshold_and_invalid_coefficients_fail_closed(fixture):
    policy, inputs = fixture
    policy.threshold = .1
    assert choose(policy, inputs) == 0  # Strictly greater, including the boundary.
    for key, bad in [("weight", float("nan")), ("feature_scale", 0.)]:
        ridge = ridge_for(policy.encoder)
        ridge[key][0] = bad
        with pytest.raises(ValueError):
            PublicPositionPolicy(policy.actor, policy.encoder, ridge)
    with pytest.raises(ValueError, match="threshold"):
        PublicPositionPolicy(policy.actor, policy.encoder, ridge_for(policy.encoder), float("nan"))


def checkpoint(path, model):
    torch.save(dict(engine_hash=tg.source_hash(), training_code_hash="unit-test-only",
                    architecture=model.architecture, model=model.state_dict(), update=0, steps=0,
                    args=dict(seed=7, eval_seed=1000, action_cap=128, hotfix_37_0_1=True)), path)


def test_bundle_keeps_actor_and_encoder_separate_and_rejects_changed_binding(tmp_path):
    torch.manual_seed(11)
    actor = EntityPolicy(width=8,layers=1,heads=2,text_dim=8)
    encoder = EntityPolicy(width=8,layers=1,heads=2,text_dim=8)
    for name, model in [("actor",actor),("encoder",encoder)]:
        checkpoint(tmp_path/f"{name}.pt", model)
    ridge = ridge_for(encoder)
    ridge["encoder_sha256"] = file_hash(tmp_path/"encoder.pt")
    torch.save(ridge,tmp_path/"ridge.pt")
    directory = tmp_path/"bundle"
    freeze_bundle(argparse.Namespace(actor=tmp_path/"actor.pt",encoder=tmp_path/"encoder.pt",
        ridge=tmp_path/"ridge.pt",threshold=.025,directory=directory))
    loaded, metadata = frozen_checkpoint(tmp_path/"actor.pt",torch.device("cpu"))
    wrapped = load_position_policy(loaded,metadata,directory/"position.json",torch.device("cpu"))
    assert wrapped.actor is loaded and wrapped.encoder is not loaded
    assert not torch.equal(wrapped.actor.positions,wrapped.encoder.positions)
    assert wrapped.decoder_metadata["encoder_sha256"] == ridge["encoder_sha256"]
    config_path = directory/"position.json"
    config = json.loads(config_path.read_text())
    with pytest.raises(ValueError, match="actor SHA"):
        load_position_policy(loaded,dict(metadata,sha256="wrong"),config_path,torch.device("cpu"))
    ridge["encoder_sha256"] = metadata["sha256"]
    torch.save(ridge,directory/"ridge.pt")
    with pytest.raises(ValueError, match="ridge SHA"):
        load_position_policy(loaded,metadata,config_path,torch.device("cpu"))
    config["ridge"]["sha256"] = file_hash(directory/"ridge.pt")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="another encoder"):
        load_position_policy(loaded,metadata,config_path,torch.device("cpu"))
    config["rules"]["unknown_rule"] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported rules"):
        load_position_policy(loaded,metadata,config_path,torch.device("cpu"))


@pytest.mark.parametrize("module_name", ["training.arena", "training.evaluate"])
def test_position_cli_cannot_override_its_frozen_rules(monkeypatch, tmp_path, module_name):
    import importlib
    import sys
    import training.position as position
    module = importlib.import_module(module_name)
    actor = EntityPolicy(width=8,layers=1,heads=2,text_dim=8)
    meta = dict(trained_hotfix_37_0_1=True,train_seed=7,validation_seed=1000,action_cap=128)
    monkeypatch.setattr(module,"frozen_checkpoint",lambda *args:(actor,meta.copy()))
    monkeypatch.setattr(position,"load_position_policy",lambda model,*args:model)
    output = tmp_path/"must-not-run.json"
    monkeypatch.setattr(sys,"argv",[module_name,"candidate.pt","--position-bundle","position.json",
        "--cancel-guard","--no-hotfix-37-0-1","--seed","2001","--device","cpu","--out",str(output)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not output.exists()


def test_bot_evaluation_assigns_fresh_ids_and_excludes_auto_reset_moves(monkeypatch):
    from training.common import evaluate
    class ResettingEnv:
        def __init__(self,n_games,**kwargs):
            assert n_games == 2
            self.n_games = n_games
            self.tick = 0
        def observe(self):
            masks = np.zeros((2,tg.N_ACTIONS),bool)
            masks[:,[0,197]] = True
            return dict(features=np.ones((2,1),np.float32),cards=np.zeros((2,1),np.int32),
                        action_mask=masks,ready=np.ones(2,bool))
        def step(self,actions):
            self.tick += 1
            return np.zeros(2),np.array([True,self.tick==3]),dict(
                placement=np.array([1,2]),rejected=np.zeros(2,bool))
    class RecordedPolicy(torch.nn.Module):
        decoder_metadata = {"kind":"greedy_plus_public_position_v1"}
        def __init__(self):
            super().__init__()
            self.seen,self.move_events,self.calls=set(),[],[]
        def reset_context(self,identity=None):
            if identity is None:self.seen.clear();self.move_events.clear()
            else:self.seen.discard(identity)
        def set_context(self,ids):
            self.ids=list(map(int,ids));self.calls.append(self.ids)
        def forward(self,features,cards,masks):
            logits=torch.full(masks.shape,-1e9)
            for i,identity in enumerate(self.ids):
                action=0 if identity in self.seen else 197
                logits[i,action]=0
                if action==197:
                    self.move_events.append(dict(identity=identity,turn=5,action=action))
                    self.seen.add(identity)
            return logits,torch.zeros(len(features))
    monkeypatch.setattr(tg,"BatchEnv",ResettingEnv)
    model=RecordedPolicy()
    report=evaluate(model,games=16,seed=2001,threads=1,config={},device=torch.device("cpu"))
    for seat in range(8):
        assert model.calls[seat*3:seat*3+3] == [
            [seat*2,seat*2+1],[seat*2+16,seat*2+1],[seat*2+32,seat*2+1]]
    assert len(model.move_events)==32
    assert report["position_decoder"]["counted_episode_moves"]==16
    assert {e["identity"] for e in report["position_decoder"]["events"]}==set(range(16))
