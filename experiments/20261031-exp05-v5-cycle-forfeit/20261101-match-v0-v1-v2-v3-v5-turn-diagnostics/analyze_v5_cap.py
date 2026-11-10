from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,statistics,hashlib
f=Path(__file__).resolve().parent

def kind(a):
    for start,name in reversed(((0,'End'),(1,'Refresh'),(2,'Freeze'),(3,'Upgrade'),(4,'HeroPower'),(6,'DarkGift'),(7,'Cancel'),(8,'Buy'),(16,'Sell'),(23,'Activate'),(30,'Choose'),(46,'Play'),(126,'Magnetize'),(196,'Move'))):
        if a>=start:return name

rows=[];byturn=defaultdict(list);cap_prefix=Counter();all_categories=Counter();tail=Counter();tail_cycles=0;cap_rows=[];capped_details=[]
with gzip.open(f/'turns.jsonl.gz','rt') as stream:
    for line in stream:
        t=json.loads(line)
        if t['model']!='v5' or t['turn']<=0:continue
        tr=t.pop('trace');cats=Counter(kind(a[1]) for a in tr)
        all_categories.update(cats)
        lastc=max((i for i,a in enumerate(tr) if kind(a[1]) not in ('Move','Freeze','End','Cancel')),default=-1)
        suffix=tr[lastc+1:]
        suffix_mf=sum(kind(a[1]) in ('Move','Freeze','Cancel') for a in suffix)
        # Unique full-state transitions do not establish useful tactical progress.
        d={k:t[k] for k in ('lineup_id','lane','seat','turn','actions','cycle_returns','forced_ends')}
        d.update(action_categories=dict(cats),move_freeze_cancel_tail_after_last_other=suffix_mf)
        rows.append(d);byturn[t['turn']].append(d)
        if t['forced_ends']:
            cap_rows.append(d);cap_prefix.update(kind(a[1]) for a in tr[:60])
            end_i=next(i for i,a in enumerate(tr) if a[7])
            last10=tr[max(0,end_i-10):end_i]
            tail.update(kind(a[1]) for a in last10);tail_cycles+=sum(a[5] for a in last10)
            d['tail10_categories']=dict(Counter(kind(a[1]) for a in last10))
            d['tail10_cycle_returns']=sum(a[5] for a in last10)
            capped_details.append(d)
count=len(rows);nc=len(cap_rows)
output={'source':'turns.jsonl.gz','source_sha256':json.loads((f/'verification.json').read_text())['file_sha256']['turns.jsonl.gz'],'model':'v5','protocol':'320 eight-player games; ten balanced V0/V1/V2/V3/V5 1-2-seat lineups,32 games each;512 V5 entries;stochastic BF16 frozen models','player_turns':count,'capped_turns':nc,'cap_fraction':nc/count,'zero_return_cap_turns':sum(t['cycle_returns']==0 for t in cap_rows),'all_action_categories':dict(all_categories),'cap_first60_action_categories':dict(cap_prefix),'cap_first60_mean_categories':{k:v/nc for k,v in cap_prefix.items()},'cap_last10_action_categories':dict(tail),'cap_last10_return_fraction':tail_cycles/(nc*10),'cap_move_freeze_cancel_suffix_after_last_other':{'mean':statistics.mean(t['move_freeze_cancel_tail_after_last_other'] for t in cap_rows),'at_least_10':sum(t['move_freeze_cancel_tail_after_last_other']>=10 for t in cap_rows),'at_least_20':sum(t['move_freeze_cancel_tail_after_last_other']>=20 for t in cap_rows),'at_least_30':sum(t['move_freeze_cancel_tail_after_last_other']>=30 for t in cap_rows)},'by_turn':{},'cautions':['Move/Freeze/Cancel category does not prove an action lacks tactical value.','No raw state or uncapped legal mask; cannot estimate the count/value of useful blocked operations.','Zero whole-observation return does not prove meaningful progress; observation changes from others may break continuity.']}
for turn,ts in sorted(byturn.items()):
    cr=[t for t in ts if t['forced_ends']]
    output['by_turn'][str(turn)]={'n':len(ts),'mean_actions':statistics.mean(t['actions'] for t in ts),'capped':len(cr),'cap_fraction':len(cr)/len(ts),'mean_moves':statistics.mean(t['action_categories'].get('Move',0) for t in ts),'mean_returns':statistics.mean(t['cycle_returns'] for t in ts)}
output['examples']=[{k:v for k,v in d.items()} for d in sorted(capped_details,key=lambda d:d['move_freeze_cancel_tail_after_last_other'],reverse=True)[:3]]
(f/'v5-cap-focus.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
