"""Executable information, action, PPO-likelihood and knowledge identity contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import taverngym as tg

from training.attachments import policy_kwargs
from training.common import load_checkpoint
from training.entity import EntityPolicy
from training.strategy import StrategyTeacher, ExpertStrategyPolicy, digest, teaching_loss


def book():
    # Small synthetic fixture; subscription content is not needed by the test suite.
    db = Path(__file__).resolve().parents[1] / 'crates/taverngym/data/cards.json'
    return dict(format_version=1, view_version=1, carddb_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),
                cards={'BG36_345': {'tags': ['economy']}, 'BG_LOE_077': {'tags': ['engine']},
                       'BG36_700': {'tags': ['direction']}, 'BG36_MidGameEffect_000t52': {'tags': ['deathrattle']}},
                heroes={'TB_BaconShop_HERO_45': dict(power_id='TB_BaconShop_HP_053',
                         curve=[{'turn': 3, 'tier': 2}], power={'start': 2, 'end': 4}),
                        'BG20_HERO_242': dict(power_id='BG20_HERO_242p',
                         progress_purchase={'threshold': 20, 'metric': 'tier'})},
                compositions=[dict(name='Synthetic two-card engine', core_ids=['BG_LOE_077','BG36_700'],
                                   entry_groups=[['BG_LOE_077'],['BG36_700']])],
                default_curve=[{'turn': 2, 'tier': 2},{'turn':5,'tier':3}],
                fresh_perspective_id='BG36_MidGameEffect_000t52', dark_gift_ids=['BG36_MidGameEffect_000t52'])


def view(turn=2, tier=1):
    mask = np.zeros(tg.N_ACTIONS, np.uint8)
    mask[[0,1,3,6,8,9,16,17,30,31,46,47]] = 1
    return dict(schema_version=1, turn=turn, lobby_races=['MURLOC'], opponents=[],
                mask=mask.tolist(), me=dict(hero=None, hero_powers=[], health=30, armor=10,
                tier=tier, gold=10, upgrade_price=4, frozen=False, stats={}, pending=None,
                board=[], hand=[], shop=[], hand_limit=10))


@pytest.mark.parametrize('turn,tier,action', [(2,1,3),(3,2,6),(4,2,6),(5,2,3)])
def test_user_curve_names_exact_action_not_refresh_or_tribe(turn,tier,action):
    teacher=StrategyTeacher(book()); v=view(turn,tier)
    scores=teacher.advise(v)
    assert scores.argmax()==action and scores[action]>0 and scores[1]==0
    v['mask'][action]=0
    assert teacher.advise(v)[action]==0


def test_hero_exception_and_progress_are_actual_inputs():
    teacher=StrategyTeacher(book()); v=view()
    v['me']['hero']={'id':'TB_BaconShop_HERO_45'}
    assert teacher.advise(v)[3]==0
    v['turn']=3
    assert teacher.advise(v)[3]==6
    v=view(8,4);v['me']['hero']={'id':'BG20_HERO_242'}
    v['me']['hero_powers']=[dict(id='BG20_HERO_242p',counter=18)]
    v['me']['shop']=[dict(id='BG36_345',tier=1),dict(id='BG36_700',tier=2)]
    score=teacher.advise(v)
    assert score[9]>score[8]
    v['me']['hero_powers'][0]['counter']=0
    assert teacher.advise(v)[9]==0


def test_composition_requires_owned_entry_and_does_not_chase_shop_offer():
    teacher=StrategyTeacher(book()); v=view(8,4)
    v['me']['shop']=[{'id':'BG_LOE_077'},{'id':'BG36_700'}]
    assert teacher.active_compositions(v)==[]
    v['me']['board']=[{'id':'BG_LOE_077'}]
    assert teacher.active_compositions(v)==[]
    v['me']['hand']=[{'id':'BG36_700'}]
    assert len(teacher.active_compositions(v))==1
    scores=teacher.advise(v)
    assert scores[8]>0 and scores[16]<0 and scores[46]>0


def test_fresh_choice_reads_attached_effect_only_in_gift_discover():
    teacher=StrategyTeacher(book());v=view(3,2)
    v['me']['pending']=dict(kind='discover',source={'id':'BG36_MidGameEffect_000t52'},
        options=[{'id':'BG36_345'}, {'id':'BG36_345','attached':[{'id':'BG36_MidGameEffect_000t52'}]}])
    assert teacher.advise(v)[31]>teacher.advise(v)[30]
    v['me']['pending']['source']={'id':'unrelated'}
    assert teacher.advise(v)[31]==teacher.advise(v)[30]


def test_target_uses_entity_id_not_slot_and_respects_spell_source():
    v=view(8,4);b=book();b['heroes']['TB_BaconShop_HERO_45']['targets']=[dict(ids=['BG36_700'],score=2)]
    v['me']['hero']={'id':'TB_BaconShop_HERO_45'}
    v['me']['board']=[{'id':'BG36_700','eid':892},{'id':'BG36_345','eid':410}]
    v['pending_source']={'id':'TB_BaconShop_HP_053'}
    v['me']['pending']={'kind':'target','targets':[410,892]}
    teacher=StrategyTeacher(b)
    assert teacher.advise(v)[31]>teacher.advise(v)[30]
    v['pending_source']={'id':'unrelated_destruction_spell'}
    assert teacher.advise(v)[31]==teacher.advise(v)[30]


def test_prior_cannot_unmask_or_accept_bad_contract():
    teacher=StrategyTeacher(book());v=view()
    mask=np.array(v['mask'],bool);mask[1]=False
    assert teacher.advise(v,mask)[1]==0
    mask[244]=True
    with pytest.raises(ValueError,match='subset'):teacher.advise(v,mask)
    b=book();b['carddb_sha256']='wrong'
    with pytest.raises(ValueError,match='table'):StrategyTeacher(b)


def test_teacher_owns_its_knowledge_and_handles_nullable_ui_fields():
    b=book();teacher=StrategyTeacher(b);v=view()
    expected=teacher.advise(v)
    b['default_curve'].clear()
    np.testing.assert_array_equal(expected,teacher.advise(v))
    v['me']['shop']=[dict(id='BG36_345',attached=None)]
    v['me']['pending']=dict(kind='target',options=None,targets=[])
    assert np.isfinite(teacher.advise(v)).all()


def test_strategy_forward_inside_position_decoder_preserves_action_ownership():
    from training.position import PublicPositionPolicy, FAMILY
    actor=EntityPolicy(width=32,layers=1,heads=4,text_dim=8)
    model=ExpertStrategyPolicy(actor,book())
    size=6*32+51+7*actor.layout['MINION_F']
    ridge=dict(family=FAMILY,weight=torch.zeros(size),feature_mean=torch.zeros(size),
               feature_scale=torch.ones(size),intercept=0.)
    wrapped=PublicPositionPolicy(model,actor,ridge)
    f,c,m,p=inputs()
    wrapped.set_context(range(len(f)))
    actual=wrapped(f,c,m,strategy_bias=p)
    expected=model(f,c,m,strategy_bias=p)
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    assert wrapped.requires_strategy and wrapped.teacher.sha256==model.teacher.sha256


def inputs(n=4):
    torch.manual_seed(41)
    f=torch.zeros(n,tg.OBS_FLOATS);c=torch.zeros(n,tg.OBS_CARDS,dtype=torch.long)
    m=torch.zeros(n,tg.N_ACTIONS,dtype=torch.bool);m[:,[0,1,3,6]]=True
    prior=torch.zeros(n,tg.N_ACTIONS);prior[:,3]=torch.arange(n,dtype=torch.float32)
    return f,c,m,prior


def test_ppo_reuses_exact_prior_for_shuffled_likelihood_and_teaches_base_actor():
    torch.set_num_threads(1)
    base=EntityPolicy(width=32,layers=1,heads=4,text_dim=8)
    model=ExpertStrategyPolicy(base,book());f,c,m,p=inputs()
    dist,value=model.distribution(f,c,m,strategy_bias=p)
    actions=dist.sample();old=dist.log_prob(actions).detach()
    idx=torch.tensor([2,0,3,1,2])
    shuffled,_=model.distribution(f[idx],c[idx],m[idx],strategy_bias=p[idx])
    torch.testing.assert_close((shuffled.log_prob(actions[idx])-old[idx]).exp(),torch.ones(len(idx)))
    no_prior,_=model.distribution(f,c,m,strategy_bias=torch.zeros_like(p))
    original,v=base(f,c,m)
    base_dist,_=base.distribution(f,c,m)
    torch.testing.assert_close(no_prior.logits,base_dist.logits,rtol=0,atol=0)
    torch.testing.assert_close(value,v,rtol=0,atol=0)
    loss=teaching_loss(dist.logits,p,m);loss.backward()
    assert loss.isfinite() and base.global_actions.weight.grad.abs().sum()>0
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    with pytest.raises(ValueError,match='aligned'):model(f,c,m)


def test_checkpoint_embeds_exact_knowledge_and_rejects_corruption(tmp_path):
    model=ExpertStrategyPolicy(EntityPolicy(width=32,layers=1,heads=4,text_dim=8),book())
    checkpoint=dict(architecture=model.architecture,engine_hash=tg.source_hash(),model=model.state_dict(),
                    strategy_bundle=book(),strategy_sha256=digest(book()))
    path=tmp_path/'policy.pt';torch.save(checkpoint,path)
    loaded,_=load_checkpoint(path)
    f,c,m,p=inputs();torch.testing.assert_close(model(f,c,m,strategy_bias=p)[0],loaded(f,c,m,strategy_bias=p)[0])
    checkpoint['strategy_bundle']['default_curve']=[];torch.save(checkpoint,path)
    with pytest.raises(ValueError,match='checksum'):load_checkpoint(path)


def test_teacher_loss_no_advice_no_signal():
    f,c,m,p=inputs()
    logits=torch.zeros_like(p,requires_grad=True)
    loss=teaching_loss(logits,torch.zeros_like(p),m)
    assert loss.item()==0
    loss.backward();assert logits.grad.abs().sum()==0


def test_real_strategy_ppo_checkpoint_resume_is_exact(tmp_path):
    import gzip
    import subprocess
    import sys

    torch.manual_seed(741)
    actor=EntityPolicy(width=16,layers=1,heads=2,text_dim=8,attachment_version=1)
    initial=tmp_path/'initial.pt'
    torch.save(dict(engine_hash=tg.source_hash(),training_code_hash='test-only',weights_only=True,
        architecture=actor.architecture,model=actor.state_dict(),update=0,steps=0,
        args=dict(seed=741,eval_seed=1741,action_cap=128,hotfix_37_0_1=True)),initial)
    knowledge=tmp_path/'knowledge.json';knowledge.write_text(json.dumps(book()))
    base=[sys.executable,'-m','training.train','--device','cpu','--envs','8','--rollout','128',
        '--minibatch','128','--threads','2','--torch-threads','1','--width','16','--layers','1',
        '--heads','2','--seed','741','--eval-seed','1741','--eval-games','8','--eval-every','100',
        '--warmup-samples','0','--opponents','bot','--cancel-guard','--hotfix-37-0-1',
        '--public-attachments','--league-turn-limit','100','--attachment-lr','0.0001',
        '--strategy-bundle',str(knowledge),'--strategy-coef','0.02','--skip-baselines',
        '--checkpoint-seconds','999999','--max-hours','0']
    for name,count,initialization in [('whole',2,['--init-from',str(initial)]),
            ('split',1,['--init-from',str(initial)]),
            ('split',2,['--resume',str(tmp_path/'split/latest.pt')])]:
        result=subprocess.run(base+['--run-dir',str(tmp_path/name),'--max-updates',str(count)]+initialization,
            cwd=Path(__file__).resolve().parents[1],text=True,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,timeout=180)
        (tmp_path/f'{name}-{count}.log').write_text(result.stdout)
        assert result.returncode==0,result.stdout[-6000:]
    _,whole=load_checkpoint(tmp_path/'whole/latest.pt');_,split=load_checkpoint(tmp_path/'split/latest.pt')
    assert whole['strategy_sha256']==split['strategy_sha256']==digest(book())
    assert whole['steps']==split['steps']==2048
    for key in whole['model']:
        assert torch.equal(whole['model'][key],split['model'][key]),key
    assert whole['optimizer']['param_groups']==split['optimizer']['param_groups']
    for key,state in whole['optimizer']['state'].items():
        for name,tensor in state.items():
            assert torch.equal(tensor,split['optimizer']['state'][key][name]),(key,name)
    assert torch.equal(whole['torch_rng'],split['torch_rng'])
    assert json.loads(gzip.decompress(whole['environment']))==json.loads(gzip.decompress(split['environment']))
    assert any(not torch.equal(actor.state_dict()[k],whole['model']['actor.'+k]) for k in actor.state_dict())
    metrics=[json.loads(line) for line in (tmp_path/'whole/metrics.jsonl').read_text().splitlines()]
    assert any(row.get('strategy_rule_applications',{}).get('user_curve',0)>0 for row in metrics)
