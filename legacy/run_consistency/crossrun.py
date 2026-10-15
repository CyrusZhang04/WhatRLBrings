# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,os,sys,random,statistics
from math import comb
def pak(n,c,k):
    if n-c<k: return 1.0
    return 1-comb(n-c,k)/comb(n,k)
def load(d):
    out={}
    for fn in sorted(glob.glob(os.path.join(d,'*.jsonl'))):
        with open(fn) as f:
            for line in f:
                x=json.loads(line)
                out[x['question']]={c:(len(x[c]['accs']),sum(bool(a) for a in x[c]['accs'])) for c in ('base','rl_replace','full_rl')}
    return out
R='${WRB_RUNS}/'
random.seed(0)
def boot(vals_fn, qs, B=2000):
    res=[]
    for _ in range(B):
        s=[random.choice(qs) for _ in qs]
        v=vals_fn(s)
        if v is not None: res.append(v)
    res.sort()
    return res[int(0.025*len(res))], res[int(0.975*len(res))]
print('### v2 vs v4 per-question consistency (same questions)')
for ds in ['aime24','aime25','aime2023','aime2022']:
    a=load(R+'eval_rl_replace_v2/'+ds); b=load(R+'eval_rl_replace_v4/'+ds)
    qs=[q for q in b if q in a]
    def m(dct,c,k=1): return statistics.mean(pak(*dct[q][c],k) for q in qs)
    rows=[]
    for c in ('base','rl_replace','full_rl'):
        rows.append(f"{c}: v2 p@1={m(a,c):.4f} v4 p@1={m(b,c):.4f} | v2 p@16={m(a,c,16):.4f} v4 p@16={m(b,c,16):.4f}")
    # cross: v4 rl_replace vs v2 full_rl
    x1=m(b,'rl_replace'); y1=m(a,'full_rl'); z1=m(a,'base')
    x16=m(b,'rl_replace',16); y16=m(a,'full_rl',16); z16=m(a,'base',16)
    print(f"{ds} nq_common={len(qs)}"); [print('   ',r) for r in rows]
    print(f"    CROSS v4-rl_replace vs v2-full_rl: p@1 {x1:.4f} vs {y1:.4f}; p@16 {x16:.4f} vs {y16:.4f}; v2 base p@1 {z1:.4f} p@16 {z16:.4f}")
print()
print('### pooled recovery with bootstrap CI (questions resampled)')
def pooled(dirs, label, k=1):
    data=[]
    for d in dirs:
        L=load(R+d); data+= list(L.values())
    def stat(s, c, k=k): return statistics.mean(pak(*x[c],k) for x in s)
    b=stat(data,'base'); r=stat(data,'rl_replace'); f=stat(data,'full_rl')
    def rec(s):
        bb=stat(s,'base'); rr=stat(s,'rl_replace'); ff=stat(s,'full_rl')
        return (rr-bb)/(ff-bb) if abs(ff-bb)>1e-9 else None
    def drf(s): return stat(s,'rl_replace')-stat(s,'full_rl')
    def drb(s): return stat(s,'rl_replace')-stat(s,'base')
    ci_rec=boot(rec,data); ci_drf=boot(drf,data); ci_drb=boot(drb,data)
    print(f"{label} k={k} nq={len(data)} base={b:.4f} rl_replace={r:.4f} full_rl={f:.4f} | recovery={(r-b)/(f-b) if f!=b else float('nan'):.3f} CI95=[{ci_rec[0]:.3f},{ci_rec[1]:.3f}] | rlrep-full={r-f:+.4f} CI95=[{ci_drf[0]:+.4f},{ci_drf[1]:+.4f}] | rlrep-base={r-b:+.4f} CI95=[{ci_drb[0]:+.4f},{ci_drb[1]:+.4f}]")
for k in (1,16):
    pooled(['eval_rl_replace/aime24','eval_rl_replace/aime25'],'v1 (aime24+25)',k)
    pooled(['eval_rl_replace_v2/aime24','eval_rl_replace_v2/aime25','eval_rl_replace_v2/aime2023','eval_rl_replace_v2/aime2022'],'v2 (4 AIME sets)',k)
    pooled(['eval_rl_replace_v2/aime24','eval_rl_replace_v2/aime25'],'v2 (aime24+25)',k)
    pooled(['eval_rl_replace_llama3/aime24','eval_rl_replace_llama3/aime25','eval_rl_replace_llama3/aime2023','eval_rl_replace_llama3/aime2022'],'llama3 (4 AIME sets)',k)
    pooled(['eval_rl_replace_math500/math500_500'],'math500 (full_rl CORRUPTED)',k)
    pooled(['eval_rl_replace_v4/aime24','eval_rl_replace_v4/aime25','eval_rl_replace_v4/aime2023','eval_rl_replace_v4/aime2022'],'v4 (base/full_rl CORRUPTED)',k)
