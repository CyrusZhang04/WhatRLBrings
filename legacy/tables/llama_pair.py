# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, glob, numpy as np
from lib import key, group, contaminated, p1, boot_idx, ci
AO='${WRB_RUNS}/'
R=[]
for f in sorted(glob.glob(AO+'eval_rl_replace_llama3/*/*-n16-shard*of8.jsonl')):
    ds=f.split('/')[-2]
    for l in open(f):
        r=json.loads(l); R.append(dict(ds=ds,q=r['question'],b=[int(x) for x in r['base']['accs']],l=[int(x) for x in r['rl_replace']['accs']],f=[int(x) for x in r['full_rl']['accs']]))
print('n',len(R),{ds:sum(1 for r in R if r['ds']==ds) for ds in ('aime24','aime25','aime2022','aime2023')},'label check',sum(1 for r in R if (group(r['q'])=='clean')==(r['ds'] in ('aime24','aime25'))))
for g,sel in (('all120',lambda r:True),('clean60',lambda r:group(r['q'])=='clean'),('contam60',lambda r:group(r['q'])=='contam')):
    S=[r for r in R if sel(r)]; n=len(S)
    b=np.array([p1(r['b']) for r in S]); l=np.array([p1(r['l']) for r in S]); f=np.array([p1(r['f']) for r in S])
    bk=np.mean([any(r['b']) for r in S]); lk=np.mean([any(r['l']) for r in S]); fk=np.mean([any(r['f']) for r in S])
    I=boot_idx(n,10000,51)
    with np.errstate(divide='ignore',invalid='ignore'): rr=(l[I].mean(1)-b[I].mean(1))/(f[I].mean(1)-b[I].mean(1))
    print('llama3 %-8s n=%d base %.2f/%.2f rl %.2f/%.2f full %.2f/%.2f | full-base %+.2f [%+.2f,%+.2f] | recov %.1f%% CI [%.1f, %.1f] P(denom<=0)=%.3f'%(g,n,100*b.mean(),100*bk,100*l.mean(),100*lk,100*f.mean(),100*fk,100*(f.mean()-b.mean()),*ci(100*(f[I].mean(1)-b[I].mean(1))),100*(l.mean()-b.mean())/(f.mean()-b.mean()),*[100*x for x in ci(rr)],np.mean(f[I].mean(1)-b[I].mean(1)<=0)))
