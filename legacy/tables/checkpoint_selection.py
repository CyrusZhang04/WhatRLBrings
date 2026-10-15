# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, glob, collections, pyarrow.parquet as pq, re
R='${WRB_ROOT}/'
V=R+'runs/qwen3_4b_base_grpo_v1/validation_data/'
D=R+'token_analysis/data/'
PFX='user\nPlease reason step by step, and put your final answer within \\boxed{}.\n\n'
def key(s): return re.sub(r'\s+',' ',s).strip()
src={}
for r in pq.read_table(D+'math500-raw.parquet').to_pylist(): src[key(r['problem'])]='math500'
for f,lab in (('aime-2024-msg.parquet','aime24'),('aime-2025-msg.parquet','aime25'),('math500-msg.parquet','math500msg')):
    for r in pq.read_table(D+f).to_pylist():
        c=r['prompt'][-1]['content']; q=c.split('\n\n',1)[1] if c.startswith('Please reason') else c
        src.setdefault(key(q),lab)
# also map gate_mcmc 120 by year
flags=json.load(open('${WRB_WORK}/eval_flags.json'))
yr={key(o['q']):o['set'] for o in flags if o['set']!='math500'}
steps=sorted(int(f.split('/')[-1][:-6]) for f in glob.glob(V+'*.jsonl'))
res={}
comp=None
for s in steps:
    rows=[json.loads(l) for l in open(V+'%d.jsonl'%s)]
    seg=collections.defaultdict(lambda:[0,0]); labs=[]
    for k,r in enumerate(rows):
        inp=r['input']; assert inp.startswith(PFX) and inp.endswith('\nassistant\n'),inp[:80]
        q=key(inp[len(PFX):-len('\nassistant\n')])
        lab=src.get(q,'?')
        if lab=='aime24': lab2=yr.get(q,'?')  # confirm year via gate_mcmc
        labs.append(lab)
        seg[lab][0]+=1; seg[lab][1]+=float(r['acc'])
    if comp is None:
        comp=collections.Counter(labs); order=[labs[0],labs[499],labs[500],labs[529],labs[530],labs[559]]
        print('val composition',dict(comp),'labels at rows 0,499,500,529,530,559:',order)
        # check aime24 val questions are AIME2024 per gate_mcmc labels
        a24=[key(json.loads(l)['input'][len(PFX):-11]) for l in list(open(V+'0.jsonl'))[500:530]]
        print('val rows 500-529 gate_mcmc years:',collections.Counter(yr.get(q,'?') for q in a24))
        a25=[key(json.loads(l)['input'][len(PFX):-11]) for l in list(open(V+'0.jsonl'))[530:560]]
        print('val rows 530-559 gate_mcmc years:',collections.Counter(yr.get(q,'?') for q in a25))
    res[s]={lab:(v[1],v[0]) for lab,v in seg.items()}
json.dump(res,open('${WRB_WORK}/val_by_step.json','w'))
def fmt(s):
    d=res[s]; m=d.get('math500',(0,1)); a=d.get('aime24',(0,1)); b=d.get('aime25',(0,1))
    tot=sum(v[0] for v in d.values()); n=sum(v[1] for v in d.values())
    return 'step %3d  MATH500 %3d/500=%.3f  AIME24 %2d/30  AIME25 %2d/30  AIME24+25 %2d/60=%.3f  total %3d/560=%.4f'%(s,m[0],m[0]/m[1],a[0],b[0],a[0]+b[0],(a[0]+b[0])/60,tot,tot/n)
print('--- saved ckpts (and step160/0) ---')
for s in [0,160]+list(range(400,581,20)): print(fmt(s))
tots={s:sum(v[0] for v in res[s].values()) for s in steps}
best=max(tots.values()); print('global best total',best,'at steps',[s for s in steps if tots[s]==best])
ck=list(range(400,561,20))
print('among saved complete ckpts 400-560: totals',{s:tots[s] for s in ck})
a2425={s:res[s]['aime24'][0]+res[s]['aime25'][0] for s in steps}
print('AIME24+25 among saved ckpts 400-560:',{s:a2425[s] for s in ck},'mean %.2f'%(sum(a2425[s] for s in ck)/len(ck)))
print('AIME24+25 max over all val steps',max(a2425.values()),'at',[s for s in steps if a2425[s]==max(a2425.values())])
m5={s:res[s]['math500'][0] for s in steps}
print('MATH500 among saved ckpts 400-560:',{s:m5[s] for s in ck})
