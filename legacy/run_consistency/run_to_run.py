# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,statistics,random
sys.path.insert(0,'${WRB_WORK}')
AO='${WRB_RUNS}'
pred=json.load(open('${WRB_WORK}/pred_flags.json'))
def load_run(pattern_fmt, dss=('aime24','aime25','aime2023','aime2022')):
    out={}
    for ds in dss:
        for fn in glob.glob(pattern_fmt.format(ds=ds)):
            for line in open(fn):
                d=json.loads(line)
                out[d['question']]={m:[bool(a) for a in d[m]['accs']] for m in ('base','rl_replace','full_rl') if m in d}
    return out
v2=load_run(AO+'/eval_rl_replace_v2/{ds}/{ds}-n16-shard*of8.jsonl')
opd=load_run(AO+'/eval_opd/opd_step200/{ds}/{ds}-n16-shard*of8.jsonl')
print('v2',len(v2),'opd',len(opd))
B=[(0,4),(4,8),(8,12),(12,15),(0,8),(0,15)]
for a,b in B:
    R=[r for r in pred if r['m']=='full_rl' and a<=r['i']<b]
    p=[sum(r['accs'])/16 for r in R]; v=[sum(v2[r['q']]['full_rl'])/16 for r in R]; o=[sum(opd[r['q']]['full_rl'])/16 for r in R]
    d=[x-y for x,y in zip(p,v)]; d2=[x-y for x,y in zip(p,o)]
    se=statistics.stdev(d)/len(d)**0.5; se2=statistics.stdev(d2)/len(d2)**0.5
    p16=100*sum(any(r['accs']) for r in R)/len(R); v16=100*sum(any(v2[r['q']]['full_rl']) for r in R)/len(R); o16=100*sum(any(opd[r['q']]['full_rl']) for r in R)/len(R)
    print(f'i{a}-{b-1} nq={len(R)}: full_rl pass1 pred={100*statistics.mean(p):.2f} v2={100*statistics.mean(v):.2f} opd200={100*statistics.mean(o):.2f} | pred-v2={100*statistics.mean(d):+.2f}±{100*se:.2f}(SE) pred-opd={100*statistics.mean(d2):+.2f}±{100*se2:.2f} | pass16 pred={p16:.2f} v2={v16:.2f} opd={o16:.2f}')
# base col sanity
for a,b in [(0,15)]:
    R=[r for r in pred if r['m']=='base' and a<=r['i']<b]
    p=[sum(r['accs'])/16 for r in R]; v=[sum(v2[r['q']]['base'])/16 for r in R]
    print('base pred=%.2f v2=%.2f'%(100*statistics.mean(p),100*statistics.mean(v)))
