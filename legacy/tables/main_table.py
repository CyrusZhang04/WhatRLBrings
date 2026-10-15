# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import lib, numpy as np, json
from lib import *
R=recs('v2')
def arr(rs,c): return np.array([p1(r['conds'][c]) for r in rs])
def arrk(rs,c,k=16): return np.array([float(any(r['conds'][c])) for r in rs])
groups={
 'all120':R,
 'clean(AIME24+25,60)':[r for r in R if group(r['question'])=='clean'],
 'contam(AIME22+23,60)':[r for r in R if group(r['question'])=='contam'],
 'contam_strict(55 in DAPO)':[r for r in R if group(r['question'])=='contam' and contaminated(r['question'])],
 'AIME22+23 not in DAPO (5)':[r for r in R if group(r['question'])=='contam' and not contaminated(r['question'])],
 'aime24':[r for r in R if r['ds']=='aime24'],'aime25':[r for r in R if r['ds']=='aime25'],
 'aime2022':[r for r in R if r['ds']=='aime2022'],'aime2023':[r for r in R if r['ds']=='aime2023'],
}
out={}
for g,rs in groups.items():
    b,l,f=arr(rs,'base'),arr(rs,'rl_replace'),arr(rs,'full_rl')
    bk,lk,fk=arrk(rs,'base'),arrk(rs,'rl_replace'),arrk(rs,'full_rl')
    n=len(rs)
    rec=(l.mean()-b.mean())/(f.mean()-b.mean()) if f.mean()!=b.mean() else float('nan')
    line='%-28s n=%3d | p@1 base %.2f rl %.2f full %.2f | gain full-base %+.2f rl-base %+.2f rl-full %+.2f | recov %.1f%% | p@16 base %.2f rl %.2f full %.2f'%(g,n,100*b.mean(),100*l.mean(),100*f.mean(),100*(f.mean()-b.mean()),100*(l.mean()-b.mean()),100*(l.mean()-f.mean()),100*rec,100*bk.mean(),100*lk.mean(),100*fk.mean())
    if n>=30:
        res={}
        for B,seed in ((2000,0),(10000,12345)):
            I=boot_idx(n,B,seed)
            bm,lm,fm=b[I].mean(1),l[I].mean(1),f[I].mean(1)
            with np.errstate(divide='ignore',invalid='ignore'): rr=(lm-bm)/(fm-bm)
            res[B]=dict(recov=ci(rr),rl_full=ci(100*(lm-fm)),rl_base=ci(100*(lm-bm)),full_base=ci(100*(fm-bm)),
                       k16_rl_full=ci(100*(lk[I].mean(1)-fk[I].mean(1))),k16_rl_base=ci(100*(lk[I].mean(1)-bk[I].mean(1))),
                       frac_denom_le0=float(np.mean(fm-bm<=0)))
        r10=res[10000]; r2=res[2000]
        line+='\n      B=10000: recov [%.3f, %.3f] | rl-full [%+.2f, %+.2f] | rl-base [%+.2f, %+.2f] | full-base [%+.2f, %+.2f] | p@16 rl-full [%+.2f, %+.2f] rl-base [%+.2f, %+.2f] | P(denom<=0)=%.4f'%(*r10['recov'],*r10['rl_full'],*r10['rl_base'],*r10['full_base'],*r10['k16_rl_full'],*r10['k16_rl_base'],r10['frac_denom_le0'])
        line+='\n      B=2000 (seed0): recov [%.3f, %.3f] | rl-full [%+.2f, %+.2f] | rl-base [%+.2f, %+.2f]'%(*r2['recov'],*r2['rl_full'],*r2['rl_base'])
        out[g]=dict(n=n,base=b.mean(),rl=l.mean(),full=f.mean(),recov=rec,ci10000=r10)
    print(line)
# difference in RL gain between clean and contam groups (bootstrap, independent resampling)
C=groups['clean(AIME24+25,60)']; X=groups['contam(AIME22+23,60)']
def stats(rs):
    return arr(rs,'base'),arr(rs,'rl_replace'),arr(rs,'full_rl')
cb,cl,cf=stats(C); xb,xl,xf=stats(X)
I=boot_idx(60,10000,7); J=boot_idx(60,10000,8)
gc=cf[I].mean(1)-cb[I].mean(1); gx=xf[J].mean(1)-xb[J].mean(1)
with np.errstate(divide='ignore',invalid='ignore'):
    rc=(cl[I].mean(1)-cb[I].mean(1))/gc; rx=(xl[J].mean(1)-xb[J].mean(1))/gx
print('RL gain (full-base) clean - contam: %+.2f pp, 95%% CI [%+.2f, %+.2f]'%(100*(gc.mean()-gx.mean()),*ci(100*(gc-gx))))
print('recovery clean - contam: point %+.3f, 95%% CI [%+.3f, %+.3f]'%(((cl.mean()-cb.mean())/(cf.mean()-cb.mean()))-((xl.mean()-xb.mean())/(xf.mean()-xb.mean())),*ci(rc-rx)))
print('base clean - contam: %+.2f pp CI [%+.2f, %+.2f]'%(100*(cb.mean()-xb.mean()),*ci(100*(cb[I].mean(1)-xb[J].mean(1)))))
json.dump(out,open('${WRB_WORK}/part2a_v2.json','w'),indent=1,default=float)
# v1 clean only
R1=recs('v1'); b,l,f=arr(R1,'base'),arr(R1,'rl_replace'),arr(R1,'full_rl')
print('v1 (aime24+25, 60): base %.2f rl %.2f full %.2f recov %.1f%%'%(100*b.mean(),100*l.mean(),100*f.mean(),100*(l.mean()-b.mean())/(f.mean()-b.mean())))
