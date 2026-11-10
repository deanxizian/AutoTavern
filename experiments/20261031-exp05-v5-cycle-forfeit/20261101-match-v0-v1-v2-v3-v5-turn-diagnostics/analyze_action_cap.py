from collections import Counter,defaultdict
import gzip,json,statistics
from pathlib import Path
folder=Path(__file__).resolve().parent

def kind(a):
    if a==0:return 'End'
    if a==1:return 'Refresh'
    if a==2:return 'Freeze'
    if a==3:return 'Upgrade'
    if 4<=a<6:return 'HeroPower'
    if a==6:return 'DarkGift'
    if a==7:return 'Cancel'
    if 8<=a<16:return 'Buy'
    if 16<=a<23:return 'Sell'
    if 23<=a<30:return 'Activate'
    if 30<=a<46:return 'Choose'
    if 46<=a<126:return 'Play'
    if 126<=a<196:return 'Magnetize'
    if 196<=a<245:return 'Move'
    raise ValueError(a)

stats={n:dict(turns=0,length_counts=Counter(),forced=0,forced_without_cycle=0,forced_cycles=[],tail_counts=Counter(),tail_cycles=0,tail_actions=0,excess_categories=Counter(),excess_after_end=0,forced_prefix_move_freeze_cancel=0,forced_prefix_actions=0) for n in ('v1','v2','v3','v5')}
examples={}
with gzip.open(folder/'turns.jsonl.gz','rt') as stream:
    for line in stream:
        t=json.loads(line)
        if t['turn']<=0:continue
        n=t['model'];s=stats[n];trace=t['trace']
        s['turns']+=1;s['length_counts'][t['actions']]+=1
        for i,a in enumerate(trace):
            if i>=60:
                s['excess_categories'][kind(a[1])]+=1
                s['excess_after_end']+=int(any(v[1]==0 for v in trace[:i]))
        if not t['forced_ends']:continue
        s['forced']+=1;s['forced_without_cycle']+=int(t['cycle_returns']==0)
        s['forced_cycles'].append(t['cycle_returns'])
        prefix=trace[:60]
        s['forced_prefix_actions']+=len(prefix)
        s['forced_prefix_move_freeze_cancel']+=sum(kind(a[1]) in ('Move','Freeze','Cancel') for a in prefix)
        # Last ten executed actions before the first forced End, including any pending choices.
        cap_i=next(i for i,a in enumerate(trace) if a[7])
        tail=trace[max(0,cap_i-10):cap_i]
        s['tail_counts'].update(kind(a[1]) for a in tail)
        s['tail_actions']+=len(tail);s['tail_cycles']+=sum(a[5] for a in tail)
        if n=='v5':
            tag='zero_return' if t['cycle_returns']==0 else 'high_return' if t['cycle_returns']>=15 else None
            if tag and tag not in examples:
                examples[tag]={k:t[k] for k in ('lineup_id','lane','seat','model','turn','actions','cycle_returns','forced_ends')}
                examples[tag].update(action_categories=dict(Counter(kind(a[1]) for a in trace)),tail_categories=dict(Counter(kind(a[1]) for a in tail)),tail_return_count=sum(a[5] for a in tail),tail_action_ids=[a[1] for a in tail])
for n,s in stats.items():
    s['forced_fraction']=s['forced']/s['turns']
    s['above60']=sum(v for k,v in s['length_counts'].items() if k>60)
    s['above60_fraction']=s['above60']/s['turns']
    s['max_actions']=max(s['length_counts'])
    s['forced_mean_cycle_returns']=statistics.mean(s.pop('forced_cycles'))
    s['forced_prefix_move_freeze_cancel_fraction']=s['forced_prefix_move_freeze_cancel']/s['forced_prefix_actions']
    s['tail_return_fraction']=s['tail_cycles']/s['tail_actions']
    s['length_counts']={str(k):s['length_counts'][k] for k in (60,61,62,63)}
report={'status':'completed','source':'turns.jsonl.gz','groups':stats,'v5_examples':examples,'limitations':['No raw state, uncapped flag or pre-cap unrestricted action mask in hashes: cannot infer how many useful actions were prevented or optimal cap.','Visible-state non-return does not establish useful progress; novel permutations consume budget too.','This is existing-trace analysis; no new match, training or cap change.']}
(folder/'action-cap-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
