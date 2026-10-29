"""Versioned, player-visible expert advice for policy decisions and PPO.

Knowledge is an explicit private JSON input, not inferred from hidden simulator
state. Advice is a bounded logit prior, never a reward or a claimed win probability.
PPO must retain the prior collected at each step and reuse it in every minibatch.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def card_id(card):
    return (card or {}).get("id", "")


class StrategyTeacher:
    """Stateless conditional advice. Every decision can return its firing reasons."""
    version = 1

    def __init__(self, bundle):
        self.bundle = deepcopy(bundle)
        bundle = self.bundle
        if tg.N_ACTIONS != 245:
            raise ValueError("Expert action layout needs revalidation")
        if not hasattr(tg.Game, "strategy_view"):
            raise ValueError("The native build has no player-visible strategy API")
        if bundle.get("format_version") != 1 or bundle.get("view_version") != 1:
            raise ValueError("Unsupported expert strategy contract")
        if bundle.get("carddb_sha256") != hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "crates/taverngym/data/cards.json").read_bytes()).hexdigest():
            raise ValueError("Expert strategy card table differs")
        self.cards = bundle["cards"]
        self.heroes = bundle["heroes"]
        self.powers = {h["power_id"]: h for h in self.heroes.values()}
        self.comps = bundle["compositions"]
        self.entry_preference = bundle.get("composition_entry_preference")
        if self.entry_preference is not None:
            pref = self.entry_preference
            if (not isinstance(pref, dict) or pref.get("format_version") != 1
                    or pref.get("scope") != "last_missing_entry_group_buy"
                    or type(pref.get("bonus")) not in (int, float)
                    or not math.isfinite(pref["bonus"]) or not 0 <= pref["bonus"] <= 6):
                raise ValueError("Invalid composition entry preference")
        self.fresh_preference = bundle.get("fresh_perspective_preference")
        if self.fresh_preference is not None:
            pref = self.fresh_preference
            if (not isinstance(pref, dict) or pref.get("format_version") != 1
                    or pref.get("scope") != "dark_gift_discover"
                    or not isinstance(pref.get("bonus"), (int, float))
                    or not math.isfinite(pref["bonus"]) or not 0 <= pref["bonus"] <= 6
                    or not isinstance(pref.get("strength_ratio"), (int, float))
                    or not 0 < pref["strength_ratio"] <= 1
                    or type(pref.get("min_turn")) is not int or type(pref.get("max_turn")) is not int
                    or not 0 <= pref["min_turn"] <= pref["max_turn"]):
                raise ValueError("Invalid conditional Fresh Perspective preference")
        self.sha256 = digest(bundle)
        self.counts = Counter()
        self.normal = {cid: c.get("normal_id") or cid for cid, c in self.cards.items()}
        for comp in self.comps:
            if not comp.get("entry_groups") and not comp.get("emergency"):
                raise ValueError("A composition must have an explicit entry condition")
            if not all(cid in self.cards for cid in comp["core_ids"]):
                raise ValueError("Composition contains an unknown card")

    def cid(self, card):
        cid = card_id(card)
        return self.normal.get(cid, cid)

    def tags(self, card):
        result = set(self.cards.get(self.cid(card), {}).get("tags", []))
        for attached in (card.get("attached") or []):
            result.update(self.cards.get(card_id(attached), {}).get("tags", []))
        return result

    def has(self, card, tag):
        return tag in self.tags(card)

    def matches(self, card, selector):
        if not card:
            return False
        if selector.get("ids") and self.cid(card) not in selector["ids"]:
            return False
        if selector.get("tags") and not self.tags(card).intersection(selector["tags"]):
            return False
        if selector.get("races") and not set(card.get("races", [])).intersection(selector["races"] + ["ALL"]):
            return False
        if selector.get("not_keywords") and set(card.get("keywords", [])).intersection(selector["not_keywords"]):
            return False
        return True

    def active_compositions(self, view):
        me = view["me"]
        owned = me["board"] + me["hand"]
        ids = {self.cid(c) for c in owned}
        health = me["health"] + me["armor"]
        economy = sum(self.has(c, "economy_engine") for c in owned)
        result = []
        for comp in self.comps:
            if comp.get("emergency"):
                eligible = view["turn"] >= 8 and health <= 15
            else:
                eligible = all(ids.intersection(group) for group in comp["entry_groups"])
                eligible &= economy >= comp.get("economy_engines", 0)
                if comp.get("stat_min"):
                    eligible &= all(me["stats"].get(k, 0) >= v for k, v in comp["stat_min"].items())
                # Late acquisition does not undo accumulated growth, but does not
                # start a new slow engine after its documented entry deadline.
                if view["turn"] > comp.get("latest_entry_turn", 100):
                    eligible &= any(self.cid(c) in comp.get("growth_ids", []) and
                                    (c.get("counter", 0) > 0 or c.get("atk", 0) > c.get("base_atk", c.get("atk", 0)) + 4)
                                    for c in owned)
            if not eligible:
                continue
            overlap = len(ids.intersection(comp["core_ids"]))
            # The observed tribe mean is only a small tie breaker after entry.
            rank_prior = comp.get("observed_mean")
            score = overlap + (0.1 * (4.5 - rank_prior) if rank_prior is not None else 0.)
            result.append((score, comp))
        result.sort(key=lambda x: (-x[0], x[1]["name"]))
        return [c for _, c in result[:2]]

    def purchase_entry_compositions(self, view, card, active):
        """Directions enabled by owning this last missing entry component.

        This public-state preview grants no ownership and mutates nothing. It
        reuses existing timing, growth, economy, statistics and ranking checks;
        it does not simulate purchase effects or estimate the cost of a pivot.
        """
        me = view["me"]
        if (self.entry_preference is None or not self.entry_preference["bonus"]
                or me.get("pending") or card.get("type") != "minion"
                or len(me["hand"]) >= me["hand_limit"]):
            return []
        owned = {self.cid(c) for c in me["board"] + me["hand"]}
        cid = self.cid(card)
        current = {c["name"] for c in active}
        candidates = set()
        for comp in self.comps:
            if comp.get("emergency") or comp["name"] in current:
                continue
            missing = [group for group in comp["entry_groups"] if not owned.intersection(group)]
            if len(missing) == 1 and cid in missing[0]:
                candidates.add(comp["name"])
        if not candidates:
            return []
        # A missing ID is not already owned, so acquiring it cannot triple
        # existing copies of itself and remove another entry component.
        preview = dict(view, me=dict(me, hand=me["hand"] + [card]))
        return [c for c in self.active_compositions(preview) if c["name"] in candidates]

    def effective_profile(self, view):
        me = view["me"]
        profile = self.heroes.get(card_id(me.get("hero")), {})
        if profile.get("inherit_power"):
            for power in me.get("hero_powers", []):
                if card_id(power) != profile.get("power_id") and card_id(power) in self.powers:
                    return self.powers[card_id(power)]
        return profile

    def fresh_choice_bonus(self, card, options, turn):
        """A bounded preference among comparable offers, never a forced action.

        Across tiers, visible attack/health is only a conservative comparison
        proxy: require identical combat keywords and positive finite body stats.
        This is not a combat simulation or a valuation of other card effects.
        """
        pref = self.fresh_preference
        if (pref is None or not pref["min_turn"] <= turn <= pref["max_turn"]
                or not any(card_id(a) == self.bundle["fresh_perspective_id"]
                           for a in (card.get("attached") or [])) or len(options) < 2):
            return 0.
        tier = card.get("tier", 0)
        if tier <= 0 or any(c.get("tier", 0) <= 0 for c in options):
            return 0.
        if tier >= max(c["tier"] for c in options):
            return float(pref["bonus"])
        keywords = set(card.get("keywords") or [])
        strengths = []
        for option in [card, *options]:
            if set(option.get("keywords") or []) != keywords:
                return 0.
            attack, health = option.get("atk"), option.get("hp")
            if (not isinstance(attack, (int, float)) or not isinstance(health, (int, float))
                    or not math.isfinite(attack) or not math.isfinite(health)
                    or attack <= 0 or health <= 0):
                return 0.
            strengths.append(math.sqrt(attack) * math.sqrt(health))
        return float(pref["bonus"]) if strengths[0] >= pref["strength_ratio"] * max(strengths[1:]) else 0.

    def advise(self, view, mask=None, *, explain=False):
        if view.get("schema_version") != 1:
            raise ValueError("Missing player-visible strategy schema")
        native = np.asarray(view["mask"], dtype=bool)
        mask = native if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (tg.N_ACTIONS,) or np.any(mask & ~native) or not mask.any():
            raise ValueError("Strategy mask must be a nonempty subset of native legal actions")
        bias = np.zeros(tg.N_ACTIONS, np.float32)
        reasons = []

        def add(action, score, reason):
            if 0 <= action < len(mask) and mask[action] and score:
                bias[action] += score
                if explain:
                    reasons.append({"action": action, "score": float(score), "rule": reason})
                self.counts[reason] += 1

        me, turn = view["me"], view["turn"]
        profile = self.effective_profile(view)
        board, hand, shop = me["board"], me["hand"], me["shop"]
        owned = board + hand
        pending = me.get("pending") or {}
        if pending.get("kind") == "hero_power":
            profile = self.heroes.get(card_id(me.get("hero")), profile)
        comps = self.active_compositions(view)
        core = set().union(*(set(c["core_ids"]) for c in comps)) if comps else set()
        targets = set().union(*(set(c.get("spell_targets", [])) for c in comps)) if comps else set()
        buy_tags = set(profile.get("buy_tags", []))
        priorities = profile.get("buy_ids", [])
        source = self.cid(view.get("pending_source") or pending.get("source"))
        actual_powers = {card_id(p): p for p in me.get("hero_powers", [])}

        def worth(card):
            cid = self.cid(card)
            tags = self.tags(card)
            score = (2.6 if cid in core else 0.)
            score += 1.0 * bool(tags & buy_tags) + 1.2 * (cid in priorities)
            if turn <= 6 and set(card.get("races", [])).intersection(profile.get("early_races", [])):
                score += 1.1
            if comps and "economy" in tags and any(c.get("cycle") for c in comps):
                score += 1.1
            if comps and "spell_generator" in tags and any(c.get("spells") for c in comps):
                score += .9
            if comps and "battlecry" in tags and any(c.get("battlecries") for c in comps):
                score += .6
            # Preserve old bundles exactly. The clarified preference is scoped
            # to choosing a Dark Gift, so it must not also leak into generic worth.
            if self.fresh_preference is None and "fresh_perspective" in tags and turn <= 6:
                score += 1.5
            return score

        def hero_score(cid, *, skill=False):
            h = self.heroes.get(cid) or self.powers.get(cid)
            if not h:
                return 0.
            means = h.get("observed_means", {})
            # No missing Top1 values are filled from another cohort.
            values = [(means.get("top50"), .5), (means.get("top1"), .5)]
            available = [(m, w) for m, w in values if m is not None]
            score = sum((4.5 - m) * w for m, w in available) / sum(w for _, w in available) if available else 0.
            score = max(-1.5, min(1.5, score))
            races = set(view["lobby_races"])
            score += .25 * bool(races.intersection(h.get("favorite_races", [])))
            if skill and profile.get("prefer_low_armor_power"):
                score += .02 * max(0, 15 - h.get("armor", 15))
            if skill and profile.get("avoid_early_power") and h.get("power", {}).get("start", 99) <= 3:
                score -= .6
            return score

        if pending.get("kind") in ("hero", "hero_power"):
            for i, card in enumerate(pending.get("options") or []):
                score = 2 * hero_score(card_id(card), skill=pending["kind"] == "hero_power")
                if pending["kind"] == "hero_power":
                    for rule in profile.get("choice_rules", []):
                        if (self.matches(card, rule) and self.condition(rule.get("when", {}), view)
                                and (not rule.get("source_ids") or source in rule["source_ids"])):
                            score += rule["score"]
                add(30 + i, score, "hero_selection")
            return self._result(bias, mask, reasons, comps, explain)

        if pending:
            legal_options = [card for i, card in enumerate(pending.get("options") or [])
                             if 30 + i < len(mask) and mask[30 + i]]
            for i, card in enumerate(pending.get("options") or []):
                score = worth(card)
                tags = self.tags(card)
                if pending["kind"] in ("discover", "trinket", "timewarp"):
                    score += .9 * bool(tags.intersection(profile.get("discover_tags", [])))
                    score += 1.3 * (self.cid(card) in profile.get("discover_ids", []))
                    if not comps and profile.get("discover_direction"):
                        score += 1.0 * ("direction" in tags)
                if (self.fresh_preference is None and
                        source in self.bundle.get("dark_gift_ids", []) and turn <= 5):
                    if any(card_id(a) == self.bundle["fresh_perspective_id"] for a in (card.get("attached") or [])):
                        score += 6.
                for rule in profile.get("choice_rules", []):
                    if (not rule.get("source_ids") or source in rule["source_ids"]) and self.matches(card, rule):
                        if self.condition(rule.get("when", {}), view):
                            score += rule["score"]
                if pending["kind"] == "quest":
                    # Compare only known progress workload, not hypothetical future rewards.
                    goal = card.get("goal", 0)
                    if goal > 0:
                        score += min(1., 3. / goal)
                        if "summon_quest" in tags and any(self.has(c, "summon") for c in board):
                            score += 1.
                if profile.get("public_guess") and source == profile.get("power_id"):
                    observed = next((o for o in view["opponents"] if o["is_next_opponent"]), None)
                    if observed and self.cid(card) in {self.cid(c) for c in observed["seen_board"]}:
                        score += 2.
                if profile.get("public_wager") and source == profile.get("power_id"):
                    opponent = next((o for o in view["opponents"] if card_id(o.get("hero")) == card_id(card)), None)
                    if opponent:
                        score += .3 * opponent["tier"] + .01 * (opponent["health"] + opponent["armor"])
                add(30 + i, score, "choice_value")
                if pending["kind"] == "discover" and source in self.bundle.get("dark_gift_ids", []):
                    add(30 + i, self.fresh_choice_bonus(card, legal_options, turn),
                        "fresh_perspective_preference")
            if pending["kind"] == "target":
                source_profile = self.powers.get(source, {})
                for i, target in enumerate(pending.get("targets") or []):
                    card, zone = self.resolve_target(target, me)
                    if not card:
                        continue
                    target_score = 0.
                    if zone == "board":
                        if source in self.bundle.get("buff_spells", []):
                            target_score += 1.7 * (self.cid(card) in targets)
                        for rule in source_profile.get("targets", []):
                            if self.matches(card, rule) and self.condition(rule.get("when", {}), view):
                                target_score += rule["score"]
                        if source in self.bundle.get("buff_spells", []):
                            target_score += .5 * self.has(card, "carrier")
                        for comp in comps:
                            for rule in comp.get("target_rules", []):
                                if source in rule["source_ids"] and self.matches(card, rule):
                                    target_score += rule["score"]
                    if zone == "shop" and source_profile.get("shop_target"):
                        target_score += worth(card)
                        if source_profile["shop_target"] == "low_stats":
                            target_score += 1. / (1 + card.get("atk", 0) + card.get("hp", 0))
                    if source_profile.get("same_tribe_cooking"):
                        pot = actual_powers.get(source, {}).get("visible_progress", {}).get("pot_cards", [])
                        pot_races = set(r for c in pot for r in c.get("races", []) if r != "ALL")
                        target_score += (1.5 if zone == "shop" else -1.)
                        if pot_races.intersection(card.get("races", [])):
                            target_score += 2.
                    if zone == "hand":
                        for rule in source_profile.get("discard_targets", []):
                            if self.matches(card, rule):
                                target_score += rule["score"]
                    add(30 + i, target_score, "public_target")
            return self._result(bias, mask, reasons, comps, explain)

        # User opening: Dark Gift (action 6) costs 3; Refresh and the Aberration
        # tribe are different concepts. Hero-specific curves override this default.
        curve = profile.get("curve", self.bundle["default_curve"])
        for rule in curve:
            if turn == rule["turn"] and me["tier"] < rule["tier"] and self.condition(rule.get("when", {}), view):
                add(3, 6., "hero_curve" if "curve" in profile else "user_curve")
                if me["gold"] < me["upgrade_price"]:
                    for i, card in enumerate(hand):
                        if self.has(card, "gold_spell"):
                            for pos in range(8):
                                add(46 + i * 8 + pos, 3., "curve_resource")
        if turn in profile.get("gift_turns", [3, 4]) and me["tier"] >= 2:
            add(6, 5., "user_dark_gift")
        for power in me.get("hero_powers", []):
            if power.get("action") not in (4, 5) or not mask[power["action"]]:
                continue
            p = self.powers.get(card_id(power), profile)
            rule = p.get("power", {})
            if not rule or not self.condition(rule.get("when", {}), view):
                continue
            price = power.get("price", 99)
            desired = rule.get("start", 1) <= turn <= rule.get("end", 99)
            desired &= turn not in rule.get("exclude_turns", [])
            desired &= me["tier"] >= rule.get("min_tier", 1)
            desired &= price <= rule.get("max_price", 99)
            if rule.get("spare"):
                desired &= price == 0 or (me["gold"] <= max(2, price) and turn != 2)
            if rule.get("requires_target"):
                desired &= any(self.matches(c, rule["requires_target"]) for c in board)
            if p.get("public_copy"):
                known = next((o for o in view["opponents"] if o["was_last_opponent"]), None)
                desired &= bool(known and any(self.cid(c) in core for c in known["seen_board"]))
            if desired:
                add(power["action"], rule.get("score", 2.), "hero_power_timing")
            elif rule.get("discourage_outside"):
                add(power["action"], -2., "hero_power_limit")
        for i, card in enumerate(shop):
            value = worth(card)
            if mask[8 + i]:
                entries = self.purchase_entry_compositions(view, card, comps)
                if entries:
                    # One bounded acquisition preference, even if several
                    # directions share this enabling card. No forced purchase.
                    add(8 + i, self.entry_preference["bonus"], "composition_entry")
                    if explain:
                        reasons[-1]["compositions"] = [c["name"] for c in entries]
            if profile.get("timewarp_triple_type") and turn in (4, 5):
                counts = Counter(r for c in board for r in c.get("races", []) if r != "ALL")
                if any(counts[r] == 2 for r in card.get("races", [])):
                    value += 1.2
            progress = profile.get("progress_purchase")
            hp = actual_powers.get(profile.get("power_id"), {})
            if progress and hp:
                if progress["metric"] == "tier" and hp.get("counter", 0) + card.get("tier", 0) >= progress["threshold"]:
                    value += 2.
                if (progress["metric"] == "minions_bought_turn" and card.get("type") == "minion"
                        and hp.get("visible_progress", {}).get("purchases_remaining") == 1):
                    value += 1.5
            if card.get("ticket") and card.get("price") is not None and card["price"] <= 1:
                value += .8
            add(8 + i, value, "buy_components")
        for i, card in enumerate(board):
            if self.cid(card) in core or self.cid(card) in profile.get("protect_ids", []):
                add(16 + i, -3., "retain_components")
            if len(board) == 7 and self.has(card, "economy") and self.cid(card) not in core:
                add(16 + i, .5, "cycle_space")
            if any(self.cid(card) in c.get("activate_ids", []) for c in comps):
                add(23 + i, 1.5, "composition_activation")
        for i, card in enumerate(hand):
            value = min(2., worth(card))
            if (profile.get("hold_generated_triples") and turn <= 3 and card.get("link")
                    and not card.get("golden") and len(hand) < me["hand_limit"] - 2
                    and len(board) >= 2):
                value -= 2.
            for pos in range(8):
                position = 0.
                if pos <= len(board):
                    if self.has(card, "engine") and pos == len(board):
                        position += .3
                    if self.has(card, "attack_trigger") and profile.get("edge_attacks") and pos in (0, len(board)):
                        position += 1.5
                    if self.has(card, "attack_trigger") and profile.get("first_windfury") and pos == 0:
                        position += 1.5
                add(46 + i * 8 + pos, value + position, "play_and_position")
            if self.has(card, "magnetic") and any(c.get("magnetics") for c in comps):
                for j, target in enumerate(board):
                    add(126 + i * 7 + j, 1.5 + .5 * self.has(target, "carrier"), "magnetic_value")
        for rule in profile.get("rules", []):
            if not self.condition(rule.get("when", {}), view):
                continue
            kind = rule["action"]
            if kind in ("upgrade", "refresh", "freeze", "gift"):
                add({"upgrade": 3, "refresh": 1, "freeze": 2, "gift": 6}[kind], rule["score"], rule["id"])
            elif kind in ("buy", "sell", "activate", "play"):
                zone, base = {"buy": (shop, 8), "sell": (board, 16), "activate": (board, 23), "play": (hand, 46)}[kind]
                for i, card in enumerate(zone):
                    if self.matches(card, rule.get("select", {})):
                        actions = range(base + i * 8, base + i * 8 + 8) if kind == "play" else [base + i]
                        for action in actions:
                            add(action, rule["score"], rule["id"])
        front_dr = set().union(*(set(c.get("leftmost_deathrattle_ids", [])) for c in comps)) if comps else set()
        right_undead = set().union(*(set(c.get("rightmost_undead_ids", [])) for c in comps)) if comps else set()
        if profile.get("edge_attacks") or profile.get("first_windfury") or front_dr or right_undead:
            # Only reward moves that strictly improve the same stateless potential;
            # inverse moves receive no advice. The cancellation guard remains active.
            def placement_value(order):
                if not order:
                    return 0.
                edge = float(self.has(order[0], "attack_trigger")) if (profile.get("edge_attacks") or profile.get("first_windfury")) else 0.
                if profile.get("edge_attacks") and len(order) > 1:
                    edge += self.has(order[-1], "attack_trigger")
                if front_dr:
                    first = next((c for c in order if self.has(c, "deathrattle")), None)
                    edge += self.cid(first) in front_dr
                if right_undead:
                    last = next((c for c in reversed(order) if "UNDEAD" in c.get("races", []) or "ALL" in c.get("races", [])), None)
                    edge += self.cid(last) in right_undead
                return float(edge)
            original = placement_value(board)
            for src in range(len(board)):
                for dest in range(len(board)):
                    moved = list(board)
                    moved.insert(dest, moved.pop(src))
                    improvement = placement_value(moved) - original
                    if improvement > 0:
                        add(196 + 7 * src + dest, improvement, "hero_position")
        return self._result(bias, mask, reasons, comps, explain)

    def condition(self, rule, view):
        me, turn = view["me"], view["turn"]
        if not rule.get("turn_min", 0) <= turn <= rule.get("turn_max", 1000):
            return False
        if not rule.get("tier_min", 0) <= me["tier"] <= rule.get("tier_max", 7):
            return False
        if not rule.get("gold_min", 0) <= me["gold"] <= rule.get("gold_max", 10000):
            return False
        if me["health"] + me["armor"] < rule.get("health_min", 0):
            return False
        if me["upgrade_price"] > rule.get("upgrade_max", 100):
            return False
        owned = me["board"] + me["hand"]
        ids = {self.cid(c) for c in owned}
        if rule.get("own_any") and not ids.intersection(rule["own_any"]):
            return False
        if rule.get("own_none") and ids.intersection(rule["own_none"]):
            return False
        if rule.get("board_tags") and not any(self.tags(c).intersection(rule["board_tags"]) for c in me["board"]):
            return False
        if rule.get("shop") and not any(self.matches(c, rule["shop"]) for c in me["shop"]):
            return False
        if rule.get("unfrozen") and me["frozen"]:
            return False
        if rule.get("power_counter_max") is not None and any(p.get("counter", 0) > rule["power_counter_max"] for p in me.get("hero_powers", [])):
            return False
        if rule.get("power_ids") and not {card_id(p) for p in me.get("hero_powers", [])}.intersection(rule["power_ids"]):
            return False
        return all(me["stats"].get(k, 0) >= v for k, v in rule.get("stats_min", {}).items())

    @staticmethod
    def resolve_target(target, me):
        # Native target options carry globally unique EntityIds, never slot indices.
        if type(target) is not int:
            return None, None
        for zone, field in (("board", "eid"), ("shop", "sid"), ("hand", "hid")):
            for card in me[zone]:
                if card.get(field) == target:
                    return card, zone
        return None, None

    @staticmethod
    def _result(bias, mask, reasons, comps, explain):
        np.clip(bias, -6., 6., out=bias)
        bias[~mask] = 0.
        if not np.isfinite(bias).all():
            raise ValueError("Non-finite strategy prior")
        return (bias, {"rules": reasons, "compositions": [c["name"] for c in comps]}) if explain else bias

    def from_env(self, env, rows, masks):
        rows = list(rows)
        if len(rows) != len(masks):
            raise ValueError("Strategy game rows and masks differ in length")
        games = getattr(env, "games", None)
        if games is None:
            games = {lane: env.game(lane) for lane, _ in rows}
        return np.stack([self.advise(json.loads(games[lane].strategy_view(seat)), mask)
                         for (lane, seat), mask in zip(rows, masks)])


class ExpertStrategyPolicy(nn.Module):
    """Actor plus frozen advice; the critic remains the learned state value."""
    requires_strategy = True

    def __init__(self, actor, bundle):
        super().__init__()
        self.actor = actor
        self.teacher = StrategyTeacher(bundle)
        self.architecture = dict(actor.architecture)

    @property
    def requires_attachments(self):
        return self.actor.requires_attachments

    @property
    def attachment_proj(self):
        return self.actor.attachment_proj

    def forward_aux(self, features, cards, masks, attachments=None, strategy_bias=None):
        if (strategy_bias is None or strategy_bias.shape != masks.shape or
                strategy_bias.device != features.device or strategy_bias.dtype != torch.float32):
            raise ValueError("Expert policy requires aligned float32 strategy priors")
        kwargs = {"attachments": attachments} if self.requires_attachments else {}
        logits, value, placement = self.actor.forward_aux(features, cards, masks, **kwargs)
        return (logits + strategy_bias).masked_fill(~masks, -1e9), value, placement

    def forward(self, features, cards, masks, attachments=None, strategy_bias=None):
        logits, value, _ = self.forward_aux(features, cards, masks, attachments, strategy_bias)
        return logits, value

    def distribution(self, features, cards, masks, auxiliary=False, *, attachments=None, strategy_bias=None):
        logits, value, placement = self.forward_aux(features, cards, masks, attachments, strategy_bias)
        result = Categorical(logits=logits, validate_args=False), value
        return (*result, placement) if auxiliary else result


def add_strategy_observation(policy, observation, env, rows):
    if getattr(policy, "requires_strategy", False):
        observation["strategy_bias"] = policy.teacher.from_env(env, rows, observation["action_mask"])


def teaching_loss(advised_logits, prior, masks, *, mode="priority-margin-v1"):
    """Teach partial preferences without assigning probabilities to unadvised actions.

    The priority margin compares the best actor choice in the highest-priority set
    with its strongest lower-priority competitor. Satisfied advice has zero loss;
    tied recommendations need not become equally likely. The old soft target is
    retained explicitly for reproducible comparisons and checkpoint resumption.
    """
    if mode not in ("priority-margin-v1", "soft-ce-v1"):
        raise ValueError("Unknown strategy teaching loss")
    base = (advised_logits - prior).masked_fill(~masks, -1e9)
    positive = prior.masked_fill(~masks, -1e9).amax(-1)
    negative = prior.masked_fill(~masks, 1e9).amin(-1)
    informative = (positive - negative > .25) & (masks.sum(-1) > 1)
    if not informative.any():
        return base.sum() * 0
    base, prior, masks = base[informative], prior[informative], masks[informative]
    if mode == "soft-ce-v1":
        target = prior.masked_fill(~masks, -1e9).softmax(-1).detach()
        return -(target * base.log_softmax(-1)).sum(-1).mean()
    best_score = positive[informative, None]
    preferred = masks & (prior >= best_score - .25)
    best = base.masked_fill(~preferred, -1e9).amax(-1)
    rival = (base + best_score - prior).masked_fill(~(masks & ~preferred), -1e9).amax(-1)
    return (rival - best).clamp_min(0).mean()
