"""Paper-informed entity transformer over the public observation contract.

This implementation has 49 tokens, including global, and cannot represent attachment
identities or hidden counters absent from the released observation. It is not a replica.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                nn.Linear(4 * width, width))

    def forward(self, x, present):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        attended = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=present[:, None, None, :], dropout_p=0.)
        x = x + self.out(attended.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.norm2(x))


class EntityPolicy(nn.Module):
    def __init__(self, width=384, layers=8, heads=8, text_dim=384):
        super().__init__()
        self.architecture = dict(kind="entity", width=width, layers=layers, heads=heads, text_dim=text_dim)
        self.layout = dict(tg.layout())
        l = self.layout
        self.header = l["GLOBAL_F"] + l["N_QUESTS"] * l["QUEST_F"] + l["PENDING_KINDS"]
        self.prefix = 1 + l["N_POWERS"] + 2 + 3 * l["N_QUESTS"] + l["N_SECRETS"]
        size, data = tg.card_static()
        static = torch.from_numpy(np.frombuffer(data, dtype="<f4").copy().reshape(tg.N_CARDS, size))
        self.register_buffer("static", static.sign() * torch.log1p(static.abs()))
        self.register_buffer("text_vectors", torch.zeros(tg.N_CARDS, text_dim))
        self.identity = nn.Embedding(tg.N_CARDS, width, padding_idx=0)
        self.static_proj = nn.Linear(size, width, bias=False)
        self.text_proj = nn.Linear(text_dim, width, bias=False)
        self.global_proj = nn.Linear(self.header, width)
        self.minion_proj = nn.Linear(l["MINION_F"], width)
        self.item_proj = nn.Linear(l["ITEM_F"], width)
        self.opponent_proj = nn.Linear(l["OPP_F"], width)
        self.positions = nn.Parameter(torch.randn(49, width) * .02)
        self.transformer = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.global_actions = nn.Linear(width, 8)
        self.queries = nn.ModuleList([nn.Linear(width, width) for _ in range(7)])
        self.keys = nn.ModuleList([nn.Linear(width, width) for _ in range(7)])
        self.end_position = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.value = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.placement = nn.Linear(width, 8)
        self.scale = math.sqrt(width)

    def card_repr(self, indices):
        indices = indices.long()
        return (self.identity(indices) + self.static_proj(self.static[indices])
                + self.text_proj(self.text_vectors[indices]))

    @staticmethod
    def pool(vectors, present):
        weights = present.to(vectors.dtype).unsqueeze(-1)
        return (vectors * weights).sum(-2) / weights.sum(-2).clamp_min(1.)

    def forward_aux(self, features, cards, masks):
        with torch.autocast(features.device.type, dtype=torch.bfloat16, enabled=features.is_cuda):
            logits, value, placement = self._forward_aux(features, cards, masks)
        # Ratios, categorical probabilities and losses must remain in float32.
        return logits.float(), value.float(), placement.float()

    def encode_tokens(self, features, cards, masks):
        """Encode only the public observation, using the caller's precision context."""
        l, n = self.layout, len(features)
        f = features.sign() * torch.log1p(features.abs())
        c = self.card_repr(cards)
        fi, ci = self.header, self.prefix
        numeric, card_groups = [], []
        for count, size in ((7, l["MINION_F"]), (10, l["ITEM_F"]),
                            (8, l["ITEM_F"]), (16, l["ITEM_F"])):
            numeric.append(f[:, fi:fi + count * size].reshape(n, count, size))
            card_groups.append(c[:, ci:ci + count])
            fi += count * size
            ci += count
        global_cards = torch.cat((c[:, :self.prefix], c[:, ci:ci + 8]), 1)
        global_ids = torch.cat((cards[:, :self.prefix], cards[:, ci:ci + 8]), 1)
        global_token = self.global_proj(f[:, :self.header]) + self.pool(global_cards, global_ids != 0)
        ci += 8
        opp_public = f[:, fi:fi + 7 * l["OPP_F"]].reshape(n, 7, l["OPP_F"])
        fi += 7 * l["OPP_F"]
        opp_board = f[:, fi:].reshape(n, 7, 7, l["MINION_F"])
        seen_cards = c[:, ci + 7:].reshape(n, 7, 7, -1)
        opponents = (self.opponent_proj(opp_public) + c[:, ci:ci + 7]
                     + self.pool(seen_cards + self.minion_proj(opp_board), opp_board[..., 0] != 0))
        groups = [card_groups[0] + self.minion_proj(numeric[0])]
        groups += [card_groups[i] + self.item_proj(numeric[i]) for i in range(1, 4)]
        x = torch.cat((global_token[:, None], *groups, opponents), 1) + self.positions
        # Pending hero/target choices can be present without a minion body.
        entity_ids = cards[:, self.prefix:self.prefix + 41]
        present = torch.cat((torch.ones((n, 1), dtype=torch.bool, device=f.device),
                             entity_ids != 0,
                             torch.ones((n, 7), dtype=torch.bool, device=f.device)), 1)
        # The released action mask can describe non-card choices (e.g. leave Timewarp).
        present[:, 26:42] |= masks[:, l["A_CHOOSE"]:l["A_CHOOSE"] + 16]
        for block in self.transformer:
            x = block(x, present)
        return self.norm(x)

    def _forward_aux(self, features, cards, masks):
        n = len(features)
        x = self.encode_tokens(features, cards, masks)
        g, board, hand, shop, choices = x[:, 0], x[:, 1:8], x[:, 8:18], x[:, 18:26], x[:, 26:42]
        def pointer(i, objects):
            return (self.keys[i](objects) * self.queries[i](g)[:, None]).sum(-1) / self.scale
        def pair(i, source, target):
            return torch.bmm(self.keys[i](source) + self.queries[i](g)[:, None],
                             target.transpose(1, 2)).flatten(1) / self.scale
        positions = torch.cat((board, self.end_position.expand(n, -1, -1) + g[:, None]), 1)
        logits = torch.cat((self.global_actions(g), pointer(0, shop), pointer(1, board),
                            pointer(2, board), pointer(3, choices), pair(4, hand, positions),
                            pair(5, hand, board), pair(6, board, board)), 1)
        return logits.masked_fill(~masks, -1e9), self.value(g).squeeze(-1), self.placement(g)

    def forward(self, features, cards, masks):
        logits, value, _ = self.forward_aux(features, cards, masks)
        return logits, value

    def distribution(self, features, cards, masks, auxiliary=False):
        logits, value, placement = self.forward_aux(features, cards, masks)
        result = (Categorical(logits=logits, validate_args=False), value)
        return (*result, placement) if auxiliary else result
