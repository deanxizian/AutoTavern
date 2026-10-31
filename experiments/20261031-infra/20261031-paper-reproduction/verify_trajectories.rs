//! Additional end-to-end checks; link against the existing release engine.
use taverngym::actions::{Action, N_ACTIONS};
use taverngym::batch::{BatchEnv, SlotResult};
use taverngym::bot::run_bot_game;
use taverngym::obs::{OBS_CARDS, OBS_FLOATS, observe};
use taverngym::rng::Rng;
use taverngym::RulesConfig;

fn actions(env: &BatchEnv, rng: &mut Rng) -> Vec<i64> {
    (0..env.n_games()).map(|i| {
        let g = env.game(i);
        if !g.needs_action(0) { return 0; }
        let legal = g.legal_actions(0);
        assert!(!legal.is_empty());
        if legal.contains(&Action::EndTurn.encode()) && rng.chance(1, 3) {
            Action::EndTurn.encode() as i64
        } else {
            legal[rng.below(legal.len())] as i64
        }
    }).collect()
}

fn main() {
    assert_eq!(N_ACTIONS, 245);
    let mut stopped = Vec::new();
    let mut invalid_placements = Vec::new();
    let mut capped = Vec::new();
    let mut truncated_combats = 0_u64;
    let mut truncated_recruit_contexts = 0_u64;
    let mut turn_sum = 0_u64;
    for seed in 0..2000 {
        let g = run_bot_game(seed, RulesConfig::default());
        if !g.is_over() || g.stopped_reason.is_some() { stopped.push(seed); }
        if g.turn_limit_end { capped.push(seed); }
        let mut placements: Vec<_> = g.players.iter().map(|p| p.placement).collect();
        placements.sort();
        if placements != (1..=8).collect::<Vec<_>>() { invalid_placements.push(seed); }
        turn_sum += g.turn as u64;
        truncated_combats += g.truncated_combats as u64;
        truncated_recruit_contexts += g.truncated_recruit_contexts as u64;
    }

    let mut a = BatchEnv::new(2, 9, RulesConfig::default(), vec![0], 1).unwrap();
    let mut rng = Rng::new(731);
    let mut result = vec![SlotResult::default(); 2];
    let mut preparation_steps = 0;
    while a.game(0).turn < 8 {
        assert!(preparation_steps < 5000, "failed to reach mid-game");
        let selected = actions(&a, &mut rng);
        a.step(&selected, &mut result).unwrap();
        assert!(result.iter().all(|r| !r.rejected));
        preparation_steps += 1;
    }
    assert!(!a.game(0).is_over());
    let saved_turn = a.game(0).turn;
    let obs = observe(a.game(0), 0);
    assert_eq!(obs.floats.len(), OBS_FLOATS);
    assert_eq!(obs.cards.len(), OBS_CARDS);
    let mut b = BatchEnv::new(2, 9, RulesConfig::default(), vec![0], 3).unwrap();
    for i in 0..2 {
        b.load(i, &a.save(i).unwrap()).unwrap();
        assert_eq!(a.save(i).unwrap(), b.save(i).unwrap());
    }
    let mut finished_episodes = 0;
    let mut continuation_actions = 0;
    for _ in 0..200 {
        let selected = actions(&a, &mut rng);
        let mut ra = vec![SlotResult::default(); 2];
        let mut rb = vec![SlotResult::default(); 2];
        a.step(&selected, &mut ra).unwrap();
        b.step(&selected, &mut rb).unwrap();
        assert_eq!(ra, rb, "restored trajectory diverged");
        assert!(ra.iter().all(|r| !r.rejected));
        finished_episodes += ra.iter().filter(|r| r.done).count();
        continuation_actions += selected.len();
        for i in 0..2 {
            assert_eq!(a.save(i).unwrap(), b.save(i).unwrap(), "state diverged");
            let oa = observe(a.game(i), 0);
            let ob = observe(b.game(i), 0);
            assert_eq!(oa.floats, ob.floats);
            assert_eq!(oa.cards, ob.cards);
        }
    }
    assert!(finished_episodes > 0, "continuation never reached an episode end");
    let all_completed = stopped.is_empty() && invalid_placements.is_empty() && capped.is_empty();
    let summary = serde_json::json!({
        "games_checked": 2000, "seeds": [0, 1999], "all_completed_naturally": all_completed,
        "stopped_seeds": stopped, "invalid_placement_seeds": invalid_placements,
        "turn_cap_seeds": capped, "mean_turns": turn_sum as f64 / 2000.0,
        "truncated_combats": truncated_combats,
        "truncated_recruit_contexts": truncated_recruit_contexts,
        "n_actions": N_ACTIONS, "observation_floats": OBS_FLOATS, "observation_cards": OBS_CARDS,
        "snapshot": {"games": 2, "saved_turn": saved_turn, "preparation_steps": preparation_steps,
            "continuation_steps": 200, "continuation_actions": continuation_actions,
            "finished_episodes": finished_episodes, "threads_original": 1, "threads_restored": 3,
            "rejected_actions": 0, "states_observations_rewards_and_placements_equal": true}
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    assert!(all_completed, "some benchmark seeds did not complete naturally");
}
