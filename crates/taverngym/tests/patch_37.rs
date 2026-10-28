//! Published 37.0 / 37.0.1 regressions. Live-client compatibility needs additional
//! validation; these tests cover the concrete behaviors listed in PATCH_COMPATIBILITY.md.

mod common;
use common::*;
use taverngym::content::gift_fits;
use taverngym::debug::{DebugCmd, run_debug};
use taverngym::script::{Shown, script};
use taverngym::state::DeityState;

fn trinket(g: &mut Game, id: &str) {
    run_debug(
        g,
        DebugCmd::GiveTrinket {
            p: 0,
            card: id.into(),
        },
    )
    .unwrap();
}

#[test]
fn hotfix_is_explicit_and_survives_save_load() {
    let legacy = RulesConfig::from_json("{}").unwrap();
    assert!(!legacy.hotfix_37_0_1);
    assert!(
        serde_json::to_value(legacy)
            .unwrap()
            .get("hotfix_37_0_1")
            .is_none()
    );
    let cfg = RulesConfig::from_json(r#"{"hotfix_37_0_1":true}"#).unwrap();
    let g = Game::new(37, cfg);
    let saved = serde_json::to_string(&g).unwrap();
    let restored: Game = serde_json::from_str(&saved).unwrap();
    assert!(restored.cfg.hotfix_37_0_1);
}

#[test]
fn gift_exclusions_cover_normal_and_golden_without_changing_other_forms() {
    let mut g = game(37);
    for enabled in [false, true] {
        g.cfg.hotfix_37_0_1 = enabled;
        let mut fits = |gift: &str, minion: &str| {
            g.with_recruit(0, |c| gift_fits(c, 0, cid(gift), db().get(cid(minion)), 1))
        };
        for id in ["BG23_318", "BG23_318_G"] {
            assert_eq!(fits("BG36_MidGameEffect_000t16", id), !enabled, "{id}");
        }
        for id in ["BG36_849", "BG36_849_G"] {
            assert_eq!(fits("BG36_MidGameEffect_000t60", id), !enabled, "{id}");
        }
        // The published Tier 4 form has no Divine Shield even in the old table.
        for id in ["BG36_360t5", "BG36_360_Gt5"] {
            assert_eq!(db().get(cid(id)).tier, 4);
            assert!(!fits("BG36_MidGameEffect_000t15", id));
        }
        // Do not silently reinterpret "Tier 4" to mean the differently named t4 form.
        for id in ["BG36_360t4", "BG36_360_Gt4"] {
            assert_eq!(db().get(cid(id)).tier, 5);
            assert!(fits("BG36_MidGameEffect_000t15", id));
        }
        assert!(fits("BG36_MidGameEffect_000t16", "BG19_010"));
        assert!(fits("BG36_MidGameEffect_000t60", "BG36_210"));
        assert!(fits("BG36_MidGameEffect_000t15", "BGS_034"));
    }
}

#[test]
fn ship_crew_discovery_triggers_hooktusk_once_per_pick_without_a_phantom_hand_card() {
    for enabled in [false, true] {
        for (ship, picks) in [("BG34_Giant_074t", 1), ("BG34_Giant_074t_G", 2)] {
            for (hook, multiplier) in [("BG36_344", 1), ("BG36_344_G", 2)] {
                let mut g = (37..237)
                    .map(game)
                    .find(|g| g.lobby_races.contains(Races::PIRATE))
                    .unwrap();
                g.cfg.hotfix_37_0_1 = enabled;
                g.players[0].tier = 3;
                g.players[0].stats.golden_played = 3;
                set_board(&mut g, 0, &[ship, hook, "BGS_055"]);
                trinket(&mut g, "BG30_MagicItem_801");
                let before = stats(&g, 0);
                g.with_recruit(0, |c| {
                    c.run_board_hook(0, |s| s.start_of_turn, 1, "start of turn");
                });
                assert_eq!(g.players[0].pending.len(), picks);
                for pick in 1..=picks {
                    g.apply(0, Action::Choose(0)).unwrap();
                    let delta = if enabled {
                        pick as i32 * multiplier * 4
                    } else {
                        0
                    };
                    let after = stats(&g, 0);
                    assert_eq!(after[1], before[1], "Hooktusk only buffs other Pirates");
                    assert_eq!(after[2], (before[2].0 + delta, before[2].1 + delta));
                    assert!(g.players[0].hand.is_empty(), "crew is stored on the ship");
                }
                assert_ne!(g.players[0].board[0].counter, 0);
                // Preserve the existing behavior of unrelated hand-copy listeners. The
                // confirmed Hooktusk fix must not consume Sinstone's first-two allowance.
                for _ in 0..2 {
                    g.with_recruit(0, |c| {
                        let item = Item::Minion(c.new_minion(0, cid("BG31_826")));
                        c.offer_discover(0, cid("TB_BaconShop_Triples_01"), vec![item]);
                    });
                    g.apply(0, Action::Choose(0)).unwrap();
                }
                assert_eq!(g.players[0].hand.len(), 4);
            }
        }
    }
}

#[test]
fn recruit_reborn_stays_consumed_after_combat_and_next_recruit() {
    for enabled in [false, true] {
        let mut g = game(38);
        g.cfg.hotfix_37_0_1 = enabled;
        set_board(&mut g, 0, &["BGS_034"]);
        let old = g.players[0].board[0].eid;
        g.with_recruit(0, |c| {
            c.destroy(old);
            c.resolve_deaths();
        });
        let reborn = &g.players[0].board[0];
        assert_ne!(reborn.eid, old);
        assert!(!reborn.has(Kw::REBORN));
        let eid = reborn.eid;
        end_turn_all(&mut g);
        assert_eq!(g.turn, 2);
        let restored = g.players[0].board.iter().find(|m| m.eid == eid).unwrap();
        assert!(!restored.has(Kw::REBORN));
    }
}

#[test]
fn mask_keeps_deity_growth_when_the_golden_deity_is_summoned() {
    let mut g = game(39);
    g.players[0].deity = Some(DeityState {
        card: cid("BGFYM_000"),
        atk: 105,
        hp: 207,
        golden: false,
        reborn: false,
    });
    trinket(&mut g, "BG36_MagicItem_602");
    let expected = g.players[0].deity.clone().unwrap();
    assert_eq!((expected.atk, expected.hp), (106, 208));
    set_board(&mut g, 0, &["BG36_098"; 4]);
    let mut c = combat_ctx(&mut g);
    let ids: Vec<_> = c.boards[0].iter().map(|m| m.eid).collect();
    for eid in ids {
        c.destroy(eid);
    }
    c.resolve_deaths();
    let deity = c.boards[0]
        .iter()
        .find(|m| m.card == cid("BGFYM_000_G"))
        .unwrap();
    assert_eq!((deity.atk, deity.hp), (expected.atk, expected.hp));
    assert!(deity.golden);
}

#[test]
fn full_hyena_warbands_finish_without_attack_limit() {
    for id in ["BG36_210", "BG36_210_G"] {
        for seed in 0..32 {
            let mut g = game(seed);
            set_board(&mut g, 0, &[id; 7]);
            set_board(&mut g, 1, &[id; 7]);
            let outcome = fight(&mut g);
            assert!(!outcome.truncated, "{id}, seed {seed}");
        }
    }
}

#[test]
fn copy_highest_hero_powers_summon_once_with_immediate_or_delayed_space() {
    for power in ["BG22_HERO_002p", "BG22_HERO_003p"] {
        for turn in [6, 7] {
            for count in [2, 7] {
                let mut g = game(40);
                g.turn = turn;
                g.set_hero_power(0, 0, cid(power));
                set_board(&mut g, 0, &vec!["BGS_055"; count]);
                set_board(&mut g, 1, &["BGS_055"]);
                g.players[1].board[0].set_stats(100, 10000);
                let outcome = fight(&mut g);
                assert!(!outcome.truncated);
                assert_eq!(
                    outcome.stats[0].summons,
                    u16::from(turn >= 7),
                    "{power}, {count}"
                );
                assert_eq!(g.players[0].board.len(), count, "copy is combat-only");
            }
        }
    }
}

#[test]
fn courier_displays_remaining_gold_from_the_five_gold_counter() {
    for id in ["BG26_810", "BG26_810_G"] {
        let mut g = game(41);
        set_board(&mut g, 0, &[id]);
        for (spent, left) in [(0, 5), (1, 4), (3, 1), (1, 5)] {
            g.with_recruit(0, |c| {
                c.gain_gold(0, spent);
                c.spend_gold(0, spent);
                let m = &c.boards[0][0];
                let sh = Shown {
                    side: 0,
                    card: m.card,
                    golden: m.golden,
                    minion: Some(m),
                    data: 0,
                };
                let shown = script(m.card).unwrap().shown.unwrap()(c, &sh);
                assert_eq!(*shown.last().unwrap(), left);
            });
        }
    }
}
