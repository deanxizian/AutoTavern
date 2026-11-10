"""Paper-inspired entity actor-critic. Unspecified paper details are explicit choices.

The published observation omits attachment identities and some per-effect/history
fields. This model consumes only the existing player-visible arrays, never state_json.
"""

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

import taverngym as tg


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int = 1024
    text_dim: int = 384
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.d_model, self.n_layers, self.n_heads, self.ff_dim, self.text_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dropout != 0:
            raise ValueError("Use zero dropout to keep rollout and PPO action probabilities aligned")


class ObservationLayout:
    """The block ordering in crates/taverngym/src/obs.rs, checked against native sizes."""

    def __init__(self):
        self.native = dict(tg.layout())
        n = self.native
        self.floats, self.cards, self.tokens = {}, {}, {}
        f = c = t = 0

        def floats(name, count, width):
            nonlocal f
            self.floats[name] = (slice(f, f + count * width), count, width)
            f += count * width

        def cards(name, count):
            nonlocal c
            self.cards[name] = slice(c, c + count)
            c += count

        def tokens(name, count):
            nonlocal t
            self.tokens[name] = slice(t, t + count)
            t += count

        floats("global", 1, n["GLOBAL_F"] + n["N_QUESTS"] * n["QUEST_F"] + n["PENDING_KINDS"])
        cards("metadata", 1 + n["N_POWERS"] + 2 + 3 * n["N_QUESTS"] + n["N_SECRETS"])
        tokens("global", 1)
        for name, count, width in (
            ("board", n["N_BOARD"], n["MINION_F"]),
            ("hand", n["N_HAND"], n["ITEM_F"]),
            ("shop", n["N_SHOP"], n["ITEM_F"]),
            ("choices", n["N_CHOICE"], n["ITEM_F"]),
        ):
            floats(name, count, width)
            cards(name, count)
            tokens(name, count)
        cards("trinkets", 8)
        floats("opponents", n["N_OPP"], n["OPP_F"])
        cards("opponents", n["N_OPP"])
        tokens("opponents", n["N_OPP"])
        floats("opponent_boards", n["N_OPP"] * n["N_BOARD"], n["MINION_F"])
        cards("opponent_boards", n["N_OPP"] * n["N_BOARD"])
        if f != tg.OBS_FLOATS or c != tg.OBS_CARDS:
            raise ValueError("Native observation layout changed; update the entity encoder")
        self.n_tokens = t

    def features(self, values, name):
        span, count, width = self.floats[name]
        return values[:, span].reshape(-1, count, width)


def symlog(x: Tensor) -> Tensor:
    return x.sign() * torch.log1p(x.abs())


def masked_mean(x: Tensor, present: Tensor, dim: int) -> Tensor:
    weights = present.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim) / weights.sum(dim).clamp_min(1)


class ModelOutput(NamedTuple):
    logits: Tensor
    values: Tensor
    placement_logits: Tensor


class EntityActorCritic(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.layout = ObservationLayout()
        n, d = self.layout.native, self.config.d_model
        width, raw = tg.card_static()
        static = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(tg.N_CARDS, width))
        # Kind/type/mechanic bits remain categorical; compress static numeric columns.
        static[:, 10:15] = symlog(static[:, 10:15])
        static[:, -1] = symlog(static[:, -1])
        self.register_buffer("static_features", static)
        self.register_buffer("text_features", torch.zeros(tg.N_CARDS, self.config.text_dim))
        self.register_buffer("text_ready", torch.tensor(False))
        self.identity = nn.Embedding(tg.N_CARDS, d, padding_idx=0)
        self.static_projection = nn.Linear(width, d, bias=False)
        self.text_projection = nn.Linear(self.config.text_dim, d, bias=False)
        self.minion_projection = nn.Linear(n["MINION_F"], d)
        self.item_projection = nn.Linear(n["ITEM_F"], d)
        self.opponent_projection = nn.Linear(n["OPP_F"], d)
        global_width = self.layout.floats["global"][2]
        self.global_projection = nn.Sequential(nn.Linear(global_width, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        metadata_count = self.layout.cards["metadata"].stop + 8
        self.metadata_roles = nn.Embedding(metadata_count, d)
        self.metadata_encoder = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.opponent_minion_encoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.zone = nn.Embedding(7, d)
        self.position = nn.Embedding(max(n["N_CHOICE"], n["N_HAND"], n["N_SHOP"]), d)
        self.global_token = nn.Parameter(torch.zeros(d))
        self.input_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, self.config.n_heads, self.config.ff_dim, dropout=0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, self.config.n_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False,
        )
        # TransformerEncoder clones its initial layer; initialize each clone independently.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.action_type = nn.Embedding(14, d)
        self.insertion_position = nn.Embedding(n["N_POS"] + 1, d, padding_idx=0)
        self.power_slot = nn.Embedding(n["N_POWERS"] + 1, d, padding_idx=0)
        self.null_entity = nn.Parameter(torch.zeros(d))
        self.pointer = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.placement_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 8))
        for embedding in (self.identity, self.zone, self.position, self.metadata_roles,
                          self.action_type, self.insertion_position, self.power_slot):
            nn.init.normal_(embedding.weight, std=0.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()
        nn.init.normal_(self.pointer[-1].weight, std=0.01)
        nn.init.zeros_(self.pointer[-1].bias)
        self._build_actions()

    def _build_actions(self):
        n, tokens = self.layout.native, self.layout.tokens
        null = self.layout.n_tokens
        kinds, sources, targets, positions, powers = [], [], [], [], []
        for a in range(tg.N_ACTIONS):
            src = dst = null
            pos = power = 0
            if a < n["A_HERO_POWER"]:
                kind = a
            elif a < n["A_DARK_GIFT"]:
                kind, power = 4, a - n["A_HERO_POWER"] + 1
            elif a == n["A_DARK_GIFT"]:
                kind = 5
            elif a == n["A_CANCEL"]:
                kind = 6
            elif a < n["A_SELL"]:
                kind, src = 7, tokens["shop"].start + a - n["A_BUY"]
            elif a < n["A_ACTIVATE"]:
                kind, src = 8, tokens["board"].start + a - n["A_SELL"]
            elif a < n["A_CHOOSE"]:
                kind, src = 9, tokens["board"].start + a - n["A_ACTIVATE"]
            elif a < n["A_PLAY"]:
                kind, src = 10, tokens["choices"].start + a - n["A_CHOOSE"]
            elif a < n["A_MAGNETIZE"]:
                hand, position = divmod(a - n["A_PLAY"], n["N_POS"])
                kind, src, pos = 11, tokens["hand"].start + hand, position + 1
            elif a < n["A_MOVE"]:
                hand, board = divmod(a - n["A_MAGNETIZE"], n["N_BOARD"])
                kind, src, dst = 12, tokens["hand"].start + hand, tokens["board"].start + board
            else:
                before, after = divmod(a - n["A_MOVE"], n["N_BOARD"])
                kind, src, dst = 13, tokens["board"].start + before, tokens["board"].start + after
            kinds.append(kind)
            sources.append(src)
            targets.append(dst)
            positions.append(pos)
            powers.append(power)
        for name, values in (("action_kinds", kinds), ("action_sources", sources),
                             ("action_targets", targets), ("action_positions", positions),
                             ("action_powers", powers)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.long))

    def set_text_embeddings(self, values: Tensor):
        if values.shape != self.text_features.shape or not torch.isfinite(values).all():
            raise ValueError(f"Expected finite text embeddings shaped {tuple(self.text_features.shape)}")
        if torch.count_nonzero(values[0]):
            raise ValueError("Text embedding row 0 must be zero padding")
        with torch.no_grad():
            self.text_features.copy_(values)
            self.text_ready.fill_(True)

    def encode_cards(self, indices: Tensor) -> Tensor:
        if not self.text_ready.item():
            raise RuntimeError("Load rule-text embeddings before using the model; see learning.text")
        return (self.identity(indices) + self.static_projection(self.static_features[indices])
                + self.text_projection(self.text_features[indices]))

    def tokenize(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features, indices = obs["features"].float(), obs["cards"].long()
        if features.ndim != 2 or features.shape[1] != tg.OBS_FLOATS:
            raise ValueError(f"features must have shape [batch, {tg.OBS_FLOATS}]")
        if indices.shape != (features.shape[0], tg.OBS_CARDS):
            raise ValueError(f"cards must have shape [batch, {tg.OBS_CARDS}]")
        cards = self.encode_cards(indices)
        layout, n = self.layout, self.layout.native
        device, batch = features.device, features.shape[0]
        blocks, validity = [], []
        metadata_span, trinket_span = layout.cards["metadata"], layout.cards["trinkets"]
        metadata = torch.cat((cards[:, metadata_span], cards[:, trinket_span]), dim=1)
        metadata_present = torch.cat((indices[:, metadata_span], indices[:, trinket_span]), dim=1) != 0
        metadata = self.metadata_encoder(metadata + self.metadata_roles.weight)
        global_state = self.global_projection(symlog(layout.features(features, "global").squeeze(1)))
        global_state = global_state + masked_mean(metadata, metadata_present, dim=1) + self.global_token
        blocks.append(global_state.unsqueeze(1) + self.zone.weight[0])
        validity.append(torch.ones(batch, 1, dtype=torch.bool, device=device))
        for zone, name in enumerate(("board", "hand", "shop", "choices"), start=1):
            dynamic = layout.features(features, name)
            projection = self.minion_projection if name == "board" else self.item_projection
            x = cards[:, layout.cards[name]] + projection(symlog(dynamic))
            x = x + self.zone.weight[zone] + self.position.weight[:dynamic.shape[1]]
            blocks.append(x)
            # A present choice may have card ID 0 (e.g. leaving Timewarp).
            validity.append(dynamic[..., 0] > 0)
        opp_dynamic = layout.features(features, "opponent_boards")
        opp_minions = cards[:, layout.cards["opponent_boards"]] + self.minion_projection(symlog(opp_dynamic))
        opp_minions = opp_minions.reshape(batch, n["N_OPP"], n["N_BOARD"], -1)
        opp_minions = self.opponent_minion_encoder(
            opp_minions + self.position.weight[:n["N_BOARD"]] + self.zone.weight[6]
        )
        seen = opp_dynamic[..., 0].reshape(batch, n["N_OPP"], n["N_BOARD"]) > 0
        opponents = cards[:, layout.cards["opponents"]] + masked_mean(opp_minions, seen, dim=2)
        public = layout.features(features, "opponents")
        opponents = opponents + self.opponent_projection(symlog(public))
        opponents = opponents + self.zone.weight[5] + self.position.weight[:n["N_OPP"]]
        blocks.append(opponents)
        validity.append(public[..., 0] > 0)
        present = torch.cat(validity, dim=1)
        x = self.input_norm(torch.cat(blocks, dim=1))
        return x.masked_fill(~present.unsqueeze(-1), 0), present

    def forward(self, obs: dict[str, Tensor], mask_actions: bool = True) -> ModelOutput:
        x, present = self.tokenize(obs)
        hidden = self.transformer(x, src_key_padding_mask=~present)
        global_state = hidden[:, 0]
        null = self.null_entity.expand(hidden.shape[0], 1, -1)
        entities = torch.cat((hidden, null), dim=1)
        source = entities[:, self.action_sources]
        target = entities[:, self.action_targets] + self.insertion_position(self.action_positions)
        kind = self.action_type(self.action_kinds) + self.power_slot(self.action_powers)
        inputs = torch.cat((global_state[:, None].expand(-1, tg.N_ACTIONS, -1), source,
                            target, kind.unsqueeze(0).expand(hidden.shape[0], -1, -1)), dim=-1)
        logits = self.pointer(inputs).squeeze(-1)
        if mask_actions:
            legal = obs["action_mask"].bool()
            if legal.shape != logits.shape or not legal.any(dim=-1).all():
                raise ValueError("Every policy observation needs at least one legal action")
            logits = logits.masked_fill(~legal, -torch.inf)
        return ModelOutput(logits, self.value_head(global_state), self.placement_head(global_state))

    def parameter_counts(self):
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "buffers": sum(b.numel() for b in self.buffers()),
            "by_module": {name: sum(p.numel() for p in module.parameters())
                          for name, module in self.named_children()},
            "config": asdict(self.config),
        }
