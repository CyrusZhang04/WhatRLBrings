# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,os,statistics,random
from math import comb
def pak(n,c,k):
    if n-c<k: return 1.0
    return 1-comb(n-c,k)/comb(n,k)
R='${WRB_RUNS}/'
def load(d):
    out={}
    for fn in sorted(glob.glob(os.path.join(R+d,'*.jsonl'))):
        with open(fn) as f:
            for line in f:
                x=json.loads(line)
                out[x['question']]={c:(len(x[c]['accs']),sum(bool(a) for a in x[c]['accs'])) for c in ('base','rl_replace','full_rl')}
    return out
v2={};v4={}
for ds in ['aime24','aime25','aime2023','aime2022']:
    v2.update(load('eval_rl_replace_v2/'+ds)); v4.update(load('eval_rl_replace_v4/'+ds))
qs=[q for q in v4 if q in v2]
print('common questions',len(qs))
for k in (1,4,16):
    a=statistics.mean(pak(*v4[q]['rl_replace'],k) for q in qs)
    b=statistics.mean(pak(*v2[q]['full_rl'],k) for q in qs)
    c=statistics.mean(pak(*v2[q]['base'],k) for q in qs)
    d=statistics.mean(pak(*v2[q]['rl_replace'],k) for q in qs)
    random.seed(1)
    diffs=[]
    for _ in range(2000):
        s=[random.choice(qs) for _ in qs]
        diffs.append(statistics.mean(pak(*v4[q]['rl_replace'],k) for q in s)-statistics.mean(pak(*v2[q]['full_rl'],k) for q in s))
    diffs.sort()
    print(f"k={k}: v4 rl_replace={a:.4f} | v2 rl_replace={d:.4f} | v2 full_rl={b:.4f} | v2 base={c:.4f} | diff(v4rlrep - v2full)={a-b:+.4f} CI95=[{diffs[50]:+.4f},{diffs[1950]:+.4f}]")
for k in (1,4,16,64,128,256):
    a=statistics.mean(pak(*v4[q]['rl_replace'],k) for q in qs)
    bb=statistics.mean(pak(*v4[q]['base'],k) for q in qs)
    print(f"v4 pooled(116q) k={k}: rl_replace={a:.4f} base(v4, partially corrupted)={bb:.4f}")
