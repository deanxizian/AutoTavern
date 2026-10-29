"""Public-only stationing after an entity actor has chosen End.

This inference decoder uses a separately frozen encoder and a learned ridge score.
It accepts no native Game or hidden opponent state. A bundle is an inference
artifact, not a readiness certificate or a claim about human playing strength.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
from pathlib import Path
import shutil

import numpy as np
import torch

from .entity import EntityPolicy


MOVE_START = 196
ACTIONS = 245
FAMILY = "frozen_tokens_plus_public_numeric_position_delta"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PublicPositionPolicy(torch.nn.Module):
    """At most one predicted Move per slot/turn, followed by End when legal.

    Call set_context with stable public scheduler IDs before every forward, and
    reset_context for each new game. Context is JSON-serializable. In CUDA
    inference the process must select float32 matmul precision 'highest' before
    serving requests; forward never changes global precision or installs hooks.
    Forward advances history for the returned action. Speculative previews must
    restore their saved history if the action is not executed.
    """
    version = 1

    def __init__(self, actor, encoder, ridge, threshold=.025):
        super().__init__()
        require(isinstance(actor, EntityPolicy) and isinstance(encoder, EntityPolicy),
                "Position decoding requires entity actor and encoder policies")
        require(ridge.get("family") == FAMILY, "Unsupported position feature family")
        require(type(threshold) in (float, int) and math.isfinite(threshold) and threshold >= 0,
                "Position threshold must be finite and nonnegative")
        self.actor, self.encoder = actor, encoder
        self.threshold = float(threshold)
        self.layout, self.header = dict(encoder.layout), encoder.header
        self.opponent_offset = self.header + 7*self.layout["MINION_F"] + 34*self.layout["ITEM_F"]
        width = encoder.architecture["width"]
        expected = 6*width + 51 + 7*self.layout["MINION_F"]
        for name, key in (("weight", "weight"), ("mean", "feature_mean"), ("scale", "feature_scale")):
            tensor = ridge.get(key)
            require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (expected,),
                    f"Position {key} has the wrong feature dimension")
            require(tensor.is_floating_point() and torch.isfinite(tensor).all().item(),
                    f"Position {key} contains invalid numbers")
            setattr(self, name, tensor.detach().cpu().numpy().copy())
        require(np.all(self.scale > 0), "Position feature scales must be positive")
        self.intercept = float(ridge["intercept"])
        require(math.isfinite(self.intercept), "Position intercept must be finite")
        self.context = None
        self.states = {}
        self.decoder_metadata = {}
        self.reset_context()

    def reset_context(self, identity=None):
        """Reset one finished slot, or start a fresh match with empty statistics."""
        self.context = None
        if identity is None:
            self.states.clear()
            self.counts = dict(forward_calls=0, eligible_ends=0, moves=0, forced_ends=0, base_decisions=0)
            self.move_events = []
        else:
            self.states.pop(operator.index(identity), None)

    def set_context(self, identities):
        require(self.context is None, "The previous position context was not consumed")
        values = [operator.index(x) for x in identities]
        require(len(set(values)) == len(values) and all(x >= 0 for x in values),
                "Position identities must be distinct nonnegative integers")
        self.context = values

    def save_context(self):
        require(self.context is None, "Save position history after completing a decision")
        return dict(version=self.version, states=[dict(identity=k, **v) for k, v in sorted(self.states.items())])

    def load_context(self, data):
        require(data.get("version") == self.version, "Unsupported position history version")
        states = {}
        for row in data["states"]:
            identity, turn, moved = row["identity"], row["turn"], row["moved"]
            require(type(identity) is int and identity >= 0 and identity not in states,
                    "Invalid or duplicate position history identity")
            require(type(turn) is int and turn >= 0 and type(moved) is bool,
                    "Invalid position history turn or moved flag")
            states[identity] = dict(turn=turn, moved=moved)
        self.states, self.context = states, None

    def eligible(self, features, masks):
        f, m = features.detach().cpu().numpy(), masks.detach().cpu().numpy()
        turn = np.rint(f[:, 0]*20).astype(int)
        opponents = f[:, self.opponent_offset:self.opponent_offset+7*self.layout["OPP_F"]]
        opponents = opponents.reshape(-1, 7, self.layout["OPP_F"])
        next_alive = np.any((opponents[:, :, 8] == 1) & (opponents[:, :, 1] == 1), axis=1)
        return ((turn >= 4) & (turn <= 20) & (np.rint(f[:, 33]*7) >= 4) & (f[:, 37] == 0)
                & next_alive & m[:, 0] & (m[:, MOVE_START:].sum(1) >= 4))

    @torch.inference_mode()
    def score_moves(self, features, cards, masks):
        require(masks.shape[1] == ACTIONS, "Position action layout differs")
        if features.is_cuda and torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("Position inference requires process-wide float32 matmul precision 'highest'")
        with torch.autocast(features.device.type, enabled=False):
            tokens = self.encoder.encode_tokens(features, cards, masks)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Position encoder produced non-finite tokens")
        tokens = tokens.detach().cpu().numpy()
        f = features.detach().cpu().numpy()
        numeric = f[:, self.header:self.header+7*self.layout["MINION_F"]].reshape(len(f), 7, -1)
        numeric = np.sign(numeric)*np.log1p(np.abs(numeric))
        legal = masks.detach().cpu().numpy()
        scores = np.full((len(f), ACTIONS), np.nan, np.float64)
        scores[:, 0] = 0.
        for i in range(len(f)):
            actions = np.flatnonzero(legal[i, MOVE_START:])+MOVE_START
            vectors = []
            for action in actions:
                local = int(action)-MOVE_START
                a, b = divmod(local, 7)
                require(a != b, "A no-op Move is marked legal")
                g, board = tokens[i, 0], tokens[i, 1:8]
                src, tgt = board[a], board[b]
                shifted = (board[min(a,b):max(a,b)+1].sum(0)-src)/abs(b-a)
                onehot = np.zeros(49, np.float32)
                onehot[local] = 1
                order = list(range(7))
                order.insert(b, order.pop(a))
                vectors.append(np.concatenate([g, src, tgt, src*tgt, g*(src-tgt),
                    np.sign(b-a)*(src-shifted), onehot,
                    np.array([(b-a)/6, abs(b-a)/6], np.float32),
                    (numeric[i, order]-numeric[i]).reshape(-1)]))
            if len(actions):
                x = np.stack(vectors)
                scores[i, actions] = ((x-self.mean)/self.scale)@self.weight+self.intercept
                if not np.isfinite(scores[i, actions]).all():
                    raise RuntimeError("Position prediction is non-finite")
        return scores

    @torch.inference_mode()
    def forward(self, features, cards, masks):
        require(self.context is not None and len(self.context) == len(features),
                "One stable position identity is required per observation")
        ids, self.context = self.context, None
        logits, value = self.actor(features, cards, masks)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise RuntimeError("Position actor produced non-finite outputs")
        self.counts["forward_calls"] += 1
        self.counts["base_decisions"] += len(features)
        turns = torch.round(features[:, 0]*20).long().cpu().numpy()
        chosen, mask = logits.argmax(-1).cpu().numpy(), masks.detach().cpu().numpy()
        possible, eligible, override = self.eligible(features, masks), [], {}
        for i, (identity, turn) in enumerate(zip(ids, turns)):
            state = self.states.get(identity)
            if state is None or state["turn"] != int(turn):
                state = self.states[identity] = dict(turn=int(turn), moved=False)
            if state["moved"] and mask[i, 0]:
                override[i] = 0
                self.counts["forced_ends"] += 1
            elif not state["moved"] and chosen[i] == 0 and possible[i]:
                eligible.append(i)
        self.counts["eligible_ends"] += len(eligible)
        if eligible:
            index = torch.as_tensor(eligible, device=features.device)
            scores = self.score_moves(features[index], cards[index], masks[index])
            for j, i in enumerate(eligible):
                actions = np.flatnonzero(mask[i, MOVE_START:])+MOVE_START
                action = int(actions[np.argmax(scores[j, actions])])
                gain = float(scores[j, action])
                if gain > self.threshold:
                    override[i] = action
                    self.states[ids[i]]["moved"] = True
                    self.counts["moves"] += 1
                    self.move_events.append(dict(identity=ids[i], turn=int(turns[i]),
                                                 action=action, predicted_gain=gain))
        if override:
            logits = logits.clone()
            for i, action in override.items():
                require(mask[i, action], "Position decoder selected an illegal action")
                logits[i].fill_(-1e9)
                logits[i, action] = 0.
        return logits, value


def load_position_policy(actor, actor_metadata, bundle, device):
    """Load a trusted bundle; enforce actor/encoder/ridge identities independently."""
    from .arena import frozen_checkpoint
    bundle = Path(bundle)
    raw = bundle.read_bytes()
    config = json.loads(raw)
    require(config.get("format_version") == 1 and config.get("kind") == "public_position_v1",
            "Unsupported public position bundle")
    require(set(config["rules"]) <= {"turn_action_caps", "hotfix_37_0_1"},
            "Position bundle contains unsupported rules")
    require(actor_metadata["sha256"] == config["actor"]["sha256"], "Position actor SHA differs")
    require(actor_metadata["engine_hash"] == config["engine_hash"], "Position engine differs")

    def component(name):
        info = config[name]
        path = (bundle.parent/info["path"]).resolve()
        require(path.is_relative_to(bundle.parent.resolve()), "Position component escapes its bundle")
        require(file_hash(path) == info["sha256"], f"Position {name} SHA differs")
        return path

    component("actor")
    encoder_path = component("encoder")
    if config["encoder"]["sha256"] == actor_metadata["sha256"]:
        encoder, encoder_metadata = actor, actor_metadata
    else:
        encoder, encoder_metadata = frozen_checkpoint(encoder_path, device)
        require(encoder_metadata["sha256"] == config["encoder"]["sha256"], "Position encoder changed while loading")
    ridge_path = component("ridge")
    # Hash and deserialize the same bytes; a concurrent file replacement cannot
    # silently change the coefficient/encoder binding after its identity check.
    import io
    ridge_bytes = ridge_path.read_bytes()
    require(hashlib.sha256(ridge_bytes).hexdigest() == config["ridge"]["sha256"], "Position ridge changed while loading")
    ridge = torch.load(io.BytesIO(ridge_bytes), map_location="cpu", weights_only=True)
    require(ridge["encoder_sha256"] == config["encoder"]["sha256"], "Ridge was trained with another encoder")
    for metadata in (actor_metadata, encoder_metadata):
        require(metadata["engine_hash"] == config["engine_hash"] and metadata["action_cap"] == 128,
                "Position model engine or action cap differs")
        require(metadata["trained_hotfix_37_0_1"] == bool(config["rules"].get("hotfix_37_0_1")),
                "Position model hotfix rules differ")
    require(config["rules"].get("turn_action_caps") == [128], "Position rules require the 128-action cap")
    model = PublicPositionPolicy(actor, encoder, ridge, config["threshold"]).eval().requires_grad_(False)
    model.decoder_metadata = dict(kind="greedy_plus_public_position_v1", position_version=1,
        actor_sha256=actor_metadata["sha256"], encoder_sha256=encoder_metadata["sha256"],
        ridge_sha256=config["ridge"]["sha256"], threshold=model.threshold,
        encoder_precision="float32_highest", bundle_sha256=hashlib.sha256(raw).hexdigest(),
        scope="public position for explicitly assigned policy roles")
    return model


def freeze_bundle(args):
    """Package trusted inputs without claiming that the combination was evaluated."""
    from .arena import frozen_checkpoint
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    components = {}
    for name in ("actor", "encoder", "ridge"):
        if name == "encoder" and Path(args.encoder).resolve() == Path(args.actor).resolve():
            components[name] = dict(components["actor"])
            continue
        path = directory/f"{name}.pt"
        with Path(getattr(args, name)).open("rb") as source, path.open("xb") as target:
            shutil.copyfileobj(source, target, 1 << 20)
        components[name] = dict(path=path.name, sha256=file_hash(path))
    actor, metadata = frozen_checkpoint(directory/"actor.pt", torch.device("cpu"))
    config = dict(format_version=1, kind="public_position_v1", **components,
                  created_at=datetime.now(timezone.utc).isoformat(), engine_hash=metadata["engine_hash"],
                  rules=dict(turn_action_caps=[128], hotfix_37_0_1=metadata["trained_hotfix_37_0_1"]),
                  threshold=args.threshold, readiness_certified=False)
    path = directory/"position.json"
    path.write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    load_position_policy(actor, metadata, path, torch.device("cpu"))
    print(json.dumps(dict(bundle=str(path), sha256=file_hash(path), readiness_certified=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.025)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    freeze_bundle(args)


if __name__ == "__main__":
    main()
