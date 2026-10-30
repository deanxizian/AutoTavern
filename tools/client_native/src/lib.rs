//! Public-state bridge only. Never exposes synthetic opponents to the actor and
//! never edits the training engine. A projected continuation is a hypothesis,
//! not a server acknowledgement or an observation from the client.
use std::ffi::{CStr, CString, c_char};
use serde_json::{Value, json};
use taverngym::{Game, RulesConfig, cards::{db, CardId}, engine::{Phase, apply_bonuses_to},
    state::{Player, Minion, Item, HandCard, ShopSlot, HeroPowerState},
    script::{script, flag}, types::Races};

type Result<T> = std::result::Result<T, String>;
fn card(v: &Value) -> Result<CardId> {
    let id = v["id"].as_str().ok_or("missing public card id")?;
    db().find(id).ok_or_else(|| format!("unknown card {id}"))
}
fn rows(v: &Value) -> impl Iterator<Item = &Value> { v.as_array().into_iter().flatten() }
fn copy_fields(to: &mut Value, from: &Value, keys: &[&str]) {
    for k in keys { if !from[k].is_null() { to[k] = from[k].clone(); } }
}
fn minion(v: &Value, p: &Player, cfg: &RulesConfig, tavern: bool) -> Result<Minion> {
    let cid = card(v)?;
    let eid = v["eid"].as_u64().ok_or("missing minion entity")? as u32;
    let mut m = Minion::new(eid, cid);
    if let Some(f) = script(cid).and_then(|s| s.on_create) { f(&mut m); }
    let mut raw = serde_json::to_value(&m).unwrap();
    copy_fields(&mut raw, v, &["atk", "hp", "max_hp", "tier", "golden", "shield_hits", "shield_max",
        "counter", "triple_reward", "activations", "temp_atk", "temp_hp", "gift_data", "attachment_data",
        "first_play_reward_used", "gems", "gem_atk", "gem_hp", "cycle", "cycle_turn", "doomed_turn"]);
    raw["races"] = json!(rows(&v["races"]).fold(0, |r, n| r | Races::from_name(n.as_str().unwrap_or("")).0));
    let names = ["TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS", "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE", "IMMUNE", "IMMUNE_ATTACKING"];
    raw["kw"] = json!(rows(&v["keywords"]).filter_map(|k| names.iter().position(|n| Some(*n) == k.as_str())).fold(0u32, |b, i| b | 1 << i));
    raw["attached"] = json!(rows(&v["attached"]).map(card).collect::<Result<Vec<_>>>()?);
    if !v["gained_dr"].is_null() { raw["gained_dr"] = json!(rows(&v["gained_dr"]).map(card).collect::<Result<Vec<_>>>()?); }
    m = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    // Public stats already include these auras. Initialise their bookkeeping,
    // preserving the observed values, so summon cannot apply them twice.
    let observed = (m.atk, m.hp, m.max_hp);
    apply_bonuses_to(cfg.volumizer_attached_grow, &p.stats, &p.trinkets, p.health, tavern, &mut m);
    (m.atk, m.hp, m.max_hp) = observed;
    Ok(m)
}
fn item(v: &Value, p: &Player, cfg: &RulesConfig, tavern: bool) -> Result<Item> {
    if v["type"] == "minion" || !v["max_hp"].is_null() { Ok(Item::Minion(minion(v, p, cfg, tavern)?)) }
    else { Ok(Item::Spell { card: card(v)?, data: v["data"].as_i64().unwrap_or(0) as i32 }) }
}
fn hydrate(v: &Value, seed: u64) -> Result<Game> {
    let seat = v["seat"].as_u64().ok_or("missing seat")? as usize;
    if seat >= 8 { return Err("invalid seat".into()); }
    let cfg: RulesConfig = serde_json::from_value(json!({"hotfix_37_0_1": true, "aberration_forced": false, "lobby_types": v["lobby_races"]})).map_err(|e| e.to_string())?;
    cfg.validate()?;
    let mut g = Game::new(seed, cfg);
    // Delete every randomly dealt hero and every generated opponent state.
    g.players = (0..8).map(Player::new).collect();
    g.phase = Phase::Recruit;
    g.turn = v["turn"].as_u64().unwrap_or(1) as u32;
    g.next_eid = 1_000_000;
    g.deity = if v["deity"].is_null() { None } else { Some(card(&v["deity"])?) };
    g.paradox = if v["paradox"].is_null() { None } else { Some(card(&v["paradox"])?) };
    let me = &v["me"];
    let mut raw = serde_json::to_value(&g.players[seat]).unwrap();
    copy_fields(&mut raw, me, &["health", "armor", "alive", "tier", "gold", "max_gold_bonus", "gold_next_turn", "free_refreshes", "chronum_saved", "chronum_balance", "next_is_ghost", "turn_data", "counters"]);
    raw["upgrade_cost"] = me["upgrade_price"].clone();
    if let Some(stats) = me["stats"].as_object() {
        for (k, x) in stats { if raw["stats"].get(k).is_some() { raw["stats"][k] = x.clone(); } }
    }
    let mut p: Player = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    p.hero = if me["hero"].is_null() { None } else { Some(card(&me["hero"])?) };
    for h in rows(&me["hero_powers"]) {
        let mut x = serde_json::to_value(HeroPowerState::new(card(h)?)).unwrap();
        copy_fields(&mut x, h, &["uses_game", "uses_turn", "counter", "cooldown", "level", "values"]);
        x["cost_mod"] = json!(h["price"].as_i64().unwrap_or(0) - db().get(card(h)?).cost as i64);
        x["memory"] = json!(rows(&h["memory"]).map(card).collect::<Result<Vec<_>>>()?);
        p.powers.push(serde_json::from_value(x).map_err(|e| e.to_string())?);
    }
    p.trinkets = rows(&me["trinkets"]).map(card).collect::<Result<Vec<_>>>()?;
    p.rewards = rows(&me["rewards"]).map(card).collect::<Result<Vec<_>>>()?;
    p.secrets = rows(&me["secrets"]).map(card).collect::<Result<Vec<_>>>()?;
    for q in rows(&me["quests"]) {
        p.quests.push(serde_json::from_value(json!({"card": card(&q["quest"])?, "reward": card(&q["reward"])?,
            "progress": q["progress"], "goal": q["goal"], "data": 0, "reward_data": 0})).map_err(|e| e.to_string())?);
    }
    for e in rows(&me["effects"]) { p.effects.push(taverngym::state::PlayerEffect { card: card(&e["card"])?, stacks: e["stacks"].as_i64().unwrap_or(1) as i32 }); }
    if !me["deity"].is_null() {
        let d = &me["deity"];
        p.deity = Some(taverngym::state::DeityState { card: card(&d["card"])?, atk: d["atk"].as_i64().unwrap_or(0) as i32, hp: d["hp"].as_i64().unwrap_or(0) as i32, golden: d["golden"].as_bool().unwrap_or(false), reborn: d["reborn"].as_bool().unwrap_or(false) });
    }
    p.board = rows(&me["board"]).map(|m| minion(m, &p, &g.cfg, false)).collect::<Result<Vec<_>>>()?;
    for h in rows(&me["hand"]) {
        p.hand.push(HandCard { hid: h["hid"].as_u64().ok_or("missing hand identity")? as u32,
            item: item(h, &p, &g.cfg, false)?, locked: h["locked"].as_u64().unwrap_or(0) as u8,
            link: h["link"].as_u64().unwrap_or(0) as u32, group: h["group"].as_u64().unwrap_or(0) as u32,
            group_left: h["group_left"].as_u64().unwrap_or(0) as u8, temporary: h["temporary"].as_bool().unwrap_or(false),
            combined: h["combined"].as_bool().unwrap_or(false), src: h["src"].as_u64().unwrap_or(0) as u32,
            held: h["held"].as_bool().unwrap_or(false), held_gold: h["held_gold"].as_i64().unwrap_or(0) as i32 });
    }
    for s in rows(&me["shop"]) {
        let mut slot = ShopSlot::new(s["sid"].as_u64().ok_or("missing shop identity")? as u32, item(s, &p, &g.cfg, true)?);
        slot.price = s["price"].as_i64().map(|n| n as i32);
        slot.frozen = s["frozen"].as_bool().unwrap_or(false);
        slot.kept = s["kept"].as_bool().unwrap_or(false);
        slot.filled = s["filled"].as_bool().unwrap_or(true);
        p.shop.push(slot);
    }
    p.hand_queue = rows(&me["hand_queue"]).map(|h| item(h, &p, &g.cfg, false)).collect::<Result<Vec<_>>>()?;
    g.players[seat] = p;
    Ok(g)
}
fn metadata() -> Value {
    let mut out = serde_json::Map::new();
    for c in db().iter() {
        if let Some(s) = script(c.cid) {
            out.insert(c.id.clone(), json!({"choose_one": s.choose_one,
                "battlecry": s.battlecry.is_some(), "battlecry_target": format!("{:?}", s.battlecry_target),
                "hero_power": s.hero_power.is_some() && !s.has_flag(flag::PASSIVE), "hero_power_uses": s.hero_power_uses,
                "persistent": s.persistent_event.is_some(), "activate": s.activate.map(|x| x.0),
                "next_combat": s.has_flag(flag::NEXT_COMBAT), "one_pending": s.has_flag(flag::ONE_PENDING),
                "spell_target": format!("{:?}", s.spell_target)}));
        }
    }
    Value::Object(out)
}
fn run(req: Value) -> Result<Value> {
    if req["op"] == "contract" { return Ok(json!({"engine_hash": env!("CLIENT_ENGINE_HASH")})); }
    if req["op"] == "metadata" { return Ok(metadata()); }
    let v = &req["view"];
    let seat = v["seat"].as_u64().ok_or("missing seat")? as usize;
    let mut g = hydrate(v, req["seed"].as_u64().unwrap_or(0))?;
    let before = g.rng.clone();
    if req["op"] == "play" {
        let b = &req["binding"];
        let hid = b["source_entity_id"].as_u64().ok_or("missing source")? as u32;
        let index = g.players[seat].hand.iter().position(|h| h.hid == hid).ok_or("source not in hand")?;
        let position = b["position"].as_u64().ok_or("missing position")? as usize;
        g.play(seat, index, position.saturating_sub(1)).map_err(str::to_string)?;
        if let Some(mode) = b["resolved_mode"]["id"].as_u64() {
            if matches!(g.players[seat].pending.last(), Some(taverngym::state::Pending::ChooseOne { .. })) {
                g.choose(seat, mode as usize).map_err(str::to_string)?;
            }
        }
    }
    let mut output = taverngym::debug::public_view_without_combats(&g, seat);
    for (row, h) in output["me"]["hand"].as_array_mut().unwrap().iter_mut().zip(&g.players[seat].hand) {
        row["doomed"] = json!(h.item.as_minion().is_some_and(|m| m.doomed_turn == g.turn));
        row["held_gold"] = json!(h.held_gold);
    }
    output["me"]["chronum_saved"] = json!(g.players[seat].chronum_saved);
    // Return own fields only: generated opponents must never enter live features.
    let pending_card = taverngym::obs::observe(&g, seat).cards[4];
    let pending_source = if pending_card > 0 { taverngym::debug::card_json(CardId((pending_card-1) as u16)) } else { Value::Null };
    Ok(json!({"me": output["me"], "pending_source": pending_source, "random_transition": g.rng != before,
        "cancellable": g.players[seat].pending.last().is_some_and(|p| p.cancellable())}))
}

/// JSON boundary keeps Rust types and allocators out of Python. Panic is contained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn client_project(input: *const c_char) -> *mut c_char {
    let result = std::panic::catch_unwind(|| {
        let s = unsafe { CStr::from_ptr(input) }.to_str().map_err(|e| e.to_string())?;
        run(serde_json::from_str(s).map_err(|e| e.to_string())?)
    });
    let value = match result { Ok(Ok(v)) => json!({"ok": v}), Ok(Err(e)) => json!({"error": e}), Err(_) => json!({"error": "native projection panic"}) };
    CString::new(value.to_string()).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn client_free(p: *mut c_char) { if !p.is_null() { drop(unsafe { CString::from_raw(p) }); } }
