//! Task adapter for the unchanged built-in heuristic bot. No labels are accepted.
use std::io::{self, BufRead};
use std::time::Instant;
use serde_json::{Value, json};
use taverngym::Game;
use taverngym::actions::Action;
use taverngym::bot::{choose_action, minion_value};
use taverngym::rng::Rng;
use taverngym::state::{Item, Pending};

type Answer = Result<(Value, Value), String>;
fn number(v: &Value, key: &str) -> Result<usize, String> {
    v[key].as_u64().map(|x| x as usize).ok_or_else(|| format!("missing integer {key}"))
}
fn load(s: &Value, seed: u64) -> Result<Game, String> {
    let mut g: Game = serde_json::from_str(s.as_str().ok_or("state is not a string")?).map_err(|e| e.to_string())?;
    g.check_loaded()?;
    g.cfg.record_log = false;
    g.rng = Rng::new(seed);
    g.seed = seed;
    Ok(g)
}
fn eid(item: &Item) -> Option<u32> { if let Item::Minion(m) = item { Some(m.eid) } else { None } }
fn hand(g: &Game, p: usize, id: u32) -> Result<usize, String> {
    g.players[p].hand.iter().position(|h| eid(&h.item) == Some(id)).ok_or_else(|| format!("hand eid {id} absent"))
}
fn board(g: &Game, p: usize, id: u32) -> Result<usize, String> {
    g.players[p].board.iter().position(|m| m.eid == id).ok_or_else(|| format!("board eid {id} absent"))
}
fn action(g: &Game, p: usize, op: &Value) -> Result<Action, String> {
    let id = op["eid"].as_u64().unwrap_or(0) as u32;
    match op["kind"].as_str().unwrap_or("") {
        "play" => { let h = hand(g, p, id)?; Ok(Action::Play {hand:h, pos:if matches!(g.players[p].hand[h].item, Item::Minion(_)) {g.players[p].board.len()} else {0}}) },
        "spell" => Ok(Action::Play {hand:g.players[p].hand.iter().position(|h| Some(h.hid as u64)==op["hid"].as_u64()).ok_or("spell hid absent")?, pos:0}),
        "sell" => Ok(Action::Sell(board(g,p,id)?)),
        "buy" => Ok(Action::Buy(g.players[p].shop.iter().position(|s| eid(&s.item)==Some(id)).ok_or("shop eid absent")?)),
        "choose" => Ok(Action::Choose(number(op,"k")?)),
        "spell_target" => Ok(Action::Play {hand:number(op,"hand")?,pos:0}),
        "activate_target" => Ok(Action::Activate(board(g,p,id)?)),
        "power_target" => Ok(Action::HeroPower(op["slot"].as_u64().unwrap_or(0) as usize)),
        "magnetize" => Ok(Action::Magnetize {hand:hand(g,p,id)?,board:board(g,p,number(op,"target")? as u32)?}),
        _ => Err(format!("unsupported operation: {}",op["kind"]))
    }
}
fn apply(g: &mut Game, p: usize, a: Action) -> Result<(), String> {
    g.apply(p,a).map(|_|()).map_err(|e|e.to_string())
}
fn choices(spec: &Value) -> Result<&Vec<Value>, String> { spec["options"].as_array().ok_or("options absent".into()) }
fn map_answer(spec: &Value, a: Action) -> Answer {
    let index = a.encode().to_string();
    let value = spec["map"].get(&index).ok_or_else(||format!("unmapped_policy_action:{a:?}"))?;
    Ok((json!({"answer":value}),json!({"native_action":format!("{a:?}"),"index":a.encode()})))
}

fn ordering(g: &mut Game, p: usize, spec: &Value) -> Answer {
    let first = choices(spec)?.first().ok_or("no ordering candidates")?;
    let ops = first["ops"].as_array().ok_or("ops absent")?;
    let letters: Vec<char> = first["answer"].as_str().ok_or("order answer is not text")?.chars().collect();
    if letters.len()!=ops.len() { return Err("ordering candidate length mismatch".into()); }
    let mut remaining:Vec<usize>=(0..ops.len()).collect();
    let mut selected=String::new();
    let mut trace=Vec::new();
    while !remaining.is_empty() {
        let mask=g.legal_mask(p);
        let mut eligible=Vec::new();
        for &i in &remaining {
            if let Ok(a)=action(g,p,&ops[i]) {
                if mask[a.encode()] {
                    let key=match a { Action::Play{hand,..}=>(0,hand as i64,i),
                        Action::Sell(b)=>(1,minion_value(&g.players[p].board[b]) as i64,i),
                        _=>return Err("unexpected ordering action".into())};
                    eligible.push((key,i,a));
                }
            }
        }
        let (_,i,a)=eligible.into_iter().min_by_key(|x|x.0).ok_or("no legal required operation remains")?;
        trace.push(json!({"letter":letters[i].to_string(),"action":format!("{a:?}")}));
        apply(g,p,a)?;
        if !g.players[p].pending.is_empty() { return Err("required operation opened an unspecified choice".into()); }
        selected.push(letters[i]);
        remaining.retain(|x|*x!=i);
    }
    if !choices(spec)?.iter().any(|o| o["answer"].as_str()==Some(&selected)) { return Err("order is not a candidate".into()); }
    Ok((json!({"answer":selected}),json!({"operations":trace})))
}
fn purchase(g: &mut Game, p: usize, spec: &Value) -> Answer {
    let candidates=choices(spec)?;
    let originals:Vec<(u32,Value)>=candidates.iter().filter_map(|o| {
        let ops=o["ops"].as_array()?;
        if ops.len()!=1 {return None;}
        Some((ops[0]["eid"].as_u64()? as u32,o["answer"].as_array()?.first()?.clone()))
    }).collect();
    let mut selected=Vec::new();
    let mut trace=Vec::new();
    for _ in 0..16 {
        let mask=g.legal_mask(p);
        let pl=&g.players[p];
        let owned=pl.board.len()+pl.hand.iter().filter(|h|matches!(h.item,Item::Minion(_))).count();
        let best=pl.shop.iter().enumerate().filter_map(|(i,s)| {
            let Item::Minion(m)=&s.item else {return None;};
            if !mask[Action::Buy(i).encode()] || !originals.iter().any(|(e,_)|*e==m.eid) {return None;}
            let pair=!m.golden && pl.board.iter().filter(|x|x.card==m.card).count()+pl.hand.iter().filter(|h|h.item.card()==m.card).count()>=1;
            if owned>=g.cfg.board_size && !pair {return None;}
            Some((i,m.eid,minion_value(m)+if pair {15} else {0}))
        }).max_by_key(|x|x.2);
        let Some((index,id,value))=best else {break;};
        apply(g,p,Action::Buy(index))?;
        if !g.players[p].pending.is_empty() {return Err("purchase opened an unspecified choice".into());}
        let h=hand(g,p,id).or_else(|_| g.players[p].hand.len().checked_sub(1).ok_or_else(||String::from("empty hand after purchase")))?;
        let pos=g.players[p].board.len();
        apply(g,p,Action::Play{hand:h,pos})?;
        if !g.players[p].pending.is_empty() {return Err("purchase play opened an unspecified choice".into());}
        selected.push(originals.iter().find(|(e,_)|*e==id).unwrap().1.clone());
        trace.push(json!({"eid":id,"heuristic_value_with_pair_bonus":value}));
    }
    let answer=json!(selected);
    if !candidates.iter().any(|o|o["answer"]==answer) {return Err("purchase sequence not listed".into());}
    Ok((json!({"answer":answer}),json!({"purchases":trace})))
}
fn target(g: &mut Game,p:usize,sid:&str,spec:&Value) -> Answer {
    let options=choices(spec)?;
    if sid=="T5" {
        let selected=options.iter().filter_map(|o| {
            let b=board(g,p,o["ops"][0]["eid"].as_u64()? as u32).ok()?;
            Some((minion_value(&g.players[p].board[b]),o))
        }).min_by_key(|x|x.0).ok_or("no sale candidates")?;
        return Ok((json!({"answer":selected.1["answer"]}),json!({"minion_value":selected.0})));
    }
    if sid=="T4" {
        let mask=g.legal_mask(p);
        let mut eligible:Vec<_>=options.iter().filter_map(|o| {
            let a=action(g,p,&o["ops"][0]).ok()?;
            if !mask[a.encode()] {return None;}
            if let Action::Magnetize{board,..}=a {Some((board,o))} else {None}
        }).collect();
        eligible.sort_by_key(|x|x.0);
        let (b,o)=eligible.first().ok_or("no magnetic host")?;
        return Ok((json!({"answer":o["answer"]}),json!({"first_legal_host":b})));
    }
    let op=&options.first().ok_or("no target candidates")?["ops"][0];
    let a=action(g,p,op)?;
    apply(g,p,a)?;
    let chosen=choose_action(g,p);
    let Action::Choose(k)=chosen else {return Err("specified action did not open a choice".into());};
    let Some(Pending::Target{options:targets,..})=g.players[p].pending.last() else {return Err("not a target choice".into());};
    let target=*targets.get(k).ok_or("native target out of range")?;
    let found=options.iter().find(|o| {
        let op=&o["ops"][0];
        op.get("target").or_else(||op.get("eid")).and_then(Value::as_u64)==Some(target as u64)
    }).ok_or("native target outside task options")?;
    Ok((json!({"answer":found["answer"]}),json!({"native_action":format!("{chosen:?}"),"target_eid":target})))
}
fn policy(sid:&str,spec:&Value,seed:u64) -> Answer {
    if sid=="D1" {
        let states=spec["states"].as_array().ok_or("hero states absent")?;
        let mut g=load(states.first().ok_or("no hero states")?,seed)?;
        let options=states.iter().map(|s|load(s,seed).and_then(|g|g.players[0].hero.ok_or("hero absent".into()))).collect::<Result<Vec<_>,_>>()?;
        g.players[0].pending=vec![Pending::HeroSelect{options}];
        let Action::Choose(k)=choose_action(&mut g,0) else {return Err("native hero choice missing".into());};
        return Ok((json!({"answer":spec["answers"][k]}),json!({"random_seed":seed,"native_choice":k})));
    }
    let mut g=load(&spec["state"],seed)?;
    let p=spec["seat"].as_u64().unwrap_or(0) as usize;
    if p>=g.players.len() {return Err("invalid seat".into());}
    if sid.starts_with('P') {
        let identity:Vec<usize>=(0..g.players[p].board.len()).collect();
        let i=spec["perms"].as_array().ok_or("perms absent")?.iter().position(|v|*v==json!(identity)).ok_or("identity order not listed")?;
        return Ok((json!({"answer":spec["answers"][i]}),json!({"kept_original_order":true})));
    }
    if sid.starts_with('O') {return ordering(&mut g,p,spec);}
    if sid.starts_with('T') {return target(&mut g,p,sid,spec);}
    if sid=="E3" {return purchase(&mut g,p,spec);}
    if sid=="D2" || sid=="D4" || sid.starts_with('N') {
        let a=choose_action(&mut g,p);
        return map_answer(spec,a);
    }
    if sid=="D3" {
        let a=choose_action(&mut g,p);
        let Action::Choose(k)=a else {return Err("discover did not produce a native choice".into());};
        let o=choices(spec)?.iter().find(|o|o["ops"][0]["k"].as_u64()==Some(k as u64)).ok_or("discover option not listed")?;
        return Ok((json!({"answer":o["answer"]}),json!({"native_choice":k})));
    }
    if sid=="D5" {
        let mask=g.legal_mask(p);
        let mut candidates:Vec<_>=choices(spec)?.iter().filter_map(|o| {
            let a=action(&g,p,&o["ops"][0]).ok()?;
            if !mask[a.encode()] {return None;}
            if let Action::Play{hand,..}=a {Some((hand,o))} else {None}
        }).collect();
        candidates.sort_by_key(|x|x.0);
        let (h,o)=candidates.first().ok_or("no playable hand choice")?;
        return Ok((json!({"answer":o["answer"]}),json!({"first_playable_hand_position":h})));
    }
    Err(format!("unsupported_scenario:{sid}"))
}

fn combat(sid:&str,spec:&Value,seed:u64) -> Answer {
    let g=load(&spec["state"],seed)?;
    if sid=="B1" {
        let a=&g.players[0];let b=&g.players[1];
        let answer=if a.board.len()!=b.board.len() {if a.board.len()>b.board.len(){"you"}else{"opponent"}}
            else if a.tier!=b.tier {if a.tier>b.tier{"you"}else{"opponent"}} else {"random"};
        return Ok((json!({"answer":answer}),json!({"engine_rule":"first_attacker","simulations":0})));
    }
    let mut counts=[0_u32;3];let mut damage=0.0;let mut lethal=0_u32;let mut truncated=0_u32;
    for i in 0..256_u64 {
        let mut sample=g.clone();
        let child=seed^0x9E3779B97F4A7C15_u64.wrapping_mul(i+1);
        sample.seed=child; sample.rng=Rng::new(child);
        if sid=="B5" {sample.players[0].health=number(spec,"health")? as i32;}
        let before=sample.players[0].health+sample.players[0].armor;
        let out=sample.simulate_combat(0,Some(1),None);
        if out.truncated {truncated+=1;}
        counts[out.winner as usize]+=1;
        sample.apply_combat_result(0,Some(1),out);
        if out.winner==1 {
            damage+=(before-sample.players[0].health-sample.players[0].armor).max(0) as f64;
            lethal+=u32::from(sample.players[0].health<=0);
        }
    }
    let answer=match sid {
        "R2"|"B2"=>{let i=(0..3).max_by_key(|&i|counts[i]).unwrap();json!({"answer":(["win","loss","tie"][i])})},
        "B3"|"B6"=>json!({"win":counts[0] as f64/256.0,"tie":counts[2] as f64/256.0,"loss":counts[1] as f64/256.0}),
        "B4"=>json!({"answer":damage/256.0}),
        "B5"=>json!({"answer":lethal as f64/256.0}),
        _=>return Err("not a combat scenario".into())
    };
    Ok((answer,json!({"simulations":256,"counts_win_loss_tie":counts,"truncated":truncated,"independent_reseeded_copies":true})))
}

fn record(result:Answer,started:Instant) -> Value {
    match result {
        Ok((response,details))=>json!({"status":"ok","response":response,"details":details,"latency_s":started.elapsed().as_secs_f64()}),
        Err(reason)=>json!({"status":if reason.starts_with("unmapped_policy_action:"){"unmapped_policy_action"}else{"adapter_error"},"reason":reason,"latency_s":started.elapsed().as_secs_f64()})
    }
}
fn process(request:&Value) -> Value {
    let id=request["id"].as_str().unwrap();let sid=request["scenario"].as_str().unwrap();
    let seed=request["seed"].as_u64().unwrap();let spec=&request["spec"];
    let predicts=sid=="R2"||sid.starts_with('B');
    let rules=["R1","R3","R4","R5","R6","R7","E1","E2","E4","E5"].contains(&sid);
    let policy_result=if rules||predicts {json!({"status":"unsupported_question","reason":if rules{"no rule-QA interface"}else{"no combat-prediction interface"},"latency_s":0.0})}
        else {let t=Instant::now();record(policy(sid,spec,seed),t)};
    // The two systems share exactly the same policy output on every non-combat task.
    let augmented=if predicts {let t=Instant::now();record(combat(sid,spec,seed),t)}else{policy_result.clone()};
    json!({"id":id,"policy":policy_result,"combat256":augmented})
}
fn main() {
    for line in io::stdin().lock().lines() {
        let line=line.unwrap();if line.trim().is_empty(){continue;}
        let request:Value=serde_json::from_str(&line).unwrap();
        let allowed=["id","scenario","seed","spec"];
        assert!(request.as_object().unwrap().keys().all(|k|allowed.contains(&k.as_str())),"unexpected inference field");
        let id=request["id"].clone();
        let result=std::panic::catch_unwind(||process(&request)).unwrap_or_else(|_|json!({"id":id,"panic":true}));
        println!("{}",serde_json::to_string(&result).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use taverngym::RulesConfig;
    use taverngym::cards::cid;
    fn fixture()->Game {
        let cfg=RulesConfig{hero_pool:vec!["TB_BaconShop_HERO_34".into()],hero_choices:1,..RulesConfig::default()};
        let mut g=Game::new(7,cfg);
        for p in 0..8 {apply(&mut g,p,Action::Choose(0)).unwrap();}
        g.players[0].hand.clear();g.players[0].board.clear();g.players[0].pending.clear();g
    }
    #[test]
    fn hand_operations_follow_entity_ids_after_cards_shift() {
        let mut g=fixture();
        for card in ["BG25_001","BG25_001"] {g.with_recruit(0,|c|c.give_card(0,cid(card)).unwrap());}
        let ids:Vec<_>=g.players[0].hand.iter().map(|h|eid(&h.item).unwrap()).collect();
        let second=json!({"kind":"play","eid":ids[1]});
        assert_eq!(action(&g,0,&second).unwrap(),Action::Play{hand:1,pos:0});
        let first=action(&g,0,&json!({"kind":"play","eid":ids[0]})).unwrap();apply(&mut g,0,first).unwrap();
        assert_eq!(action(&g,0,&second).unwrap(),Action::Play{hand:0,pos:1});
    }
    #[test]
    fn native_target_chooses_the_strongest_not_the_first() {
        let mut g=fixture();
        let small=g.with_recruit(0,|c|c.new_minion(0,cid("BG25_001")));
        let mut big=small.clone();big.eid=g.alloc_eid();big.atk=40;big.hp=40;big.max_hp=40;
        g.players[0].board=vec![small.clone(),big.clone()];
        g.players[0].pending=vec![Pending::Target{options:vec![small.eid,big.eid],cont:taverngym::state::Continuation::Spell{card:cid("BG28_810"),data:0,opt:0}}];
        assert_eq!(choose_action(&mut g,0),Action::Choose(1));
    }
    #[test]
    fn native_map_does_not_invent_an_answer_for_unlisted_actions() {
        let result=map_answer(&json!({"map":{"3":"A"}}),Action::Refresh);
        assert!(result.unwrap_err().starts_with("unmapped_policy_action:"));
    }
    #[test]
    fn position_adapter_keeps_identity_even_when_not_the_first_option() {
        let mut g=fixture();
        for _ in 0..2 {let m=g.with_recruit(0,|c|c.new_minion(0,cid("BG25_001")));g.players[0].board.push(m);}
        let spec=json!({"state":serde_json::to_string(&g).unwrap(),"perms":[[1,0],[0,1]],"answers":["A","B"]});
        assert_eq!(policy("P1",&spec,17).unwrap().0,json!({"answer":"B"}));
    }
    #[test]
    fn combat_wrapper_is_repeatable_and_does_not_mutate_the_input() {
        let mut g=fixture();
        let mut m=g.with_recruit(0,|c|c.new_minion(0,cid("BG25_001")));
        m.atk=100;m.hp=100;m.max_hp=100;g.players[0].board=vec![m];g.players[1].board.clear();
        let spec=json!({"state":serde_json::to_string(&g).unwrap()});let before=spec.clone();
        let a=combat("B3",&spec,19).unwrap();let b=combat("B3",&spec,19).unwrap();
        assert_eq!(a,b);assert_eq!(a.0,json!({"win":1.0,"tie":0.0,"loss":0.0}));assert_eq!(spec,before);
    }
}
