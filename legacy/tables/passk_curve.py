# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
from lib import *
import numpy as np
R4=recs('v4'); R2={key(r['question']):r for r in recs('v2')}
ks=[1,4,16,64,128,256]
def curve(rs,c,ks):
    return [100*np.mean([pk(r['conds'][c],k) for r in rs]) for k in ks]
for gname,sel in (('all116',lambda q:True),('clean60',lambda q:group(q)=='clean'),('contam56',lambda q:group(q)=='contam'),('contam_strict',lambda q:group(q)=='contam' and contaminated(q))):
    rs=[r for r in R4 if sel(r['question'])]
    n=len(rs); ns=set(len(r['conds']['rl_replace']) for r in rs)
    v4=curve(rs,'rl_replace',ks)
    same=[R2[key(r['question'])] for r in rs]
    v2r=curve(same,'rl_replace',[1,4,16]); v2f=curve(same,'full_rl',[1,4,16]); v2b=curve(same,'base',[1,4,16])
    print('%-14s n=%d (samples per q %s) v4 rl_replace k=%s: %s'%(gname,n,ns,ks,' / '.join('%.2f'%x for x in v4)))
    print('               v2 same-q k=1/4/16: rl %s | full %s | base %s'%(' / '.join('%.2f'%x for x in v2r),' / '.join('%.2f'%x for x in v2f),' / '.join('%.2f'%x for x in v2b)))
    # v4 rl_replace - v2 full_rl, pass@1 and pass@16, bootstrap
    a1=np.array([pk(r['conds']['rl_replace'],1) for r in rs]); b1=np.array([pk(s['conds']['full_rl'],1) for s in same])
    a16=np.array([pk(r['conds']['rl_replace'],16) for r in rs]); b16=np.array([pk(s['conds']['full_rl'],16) for s in same])
    I=boot_idx(n,10000,3)
    print('               v4rl - v2full: p@1 %+.2f [%+.2f, %+.2f]  p@16 %+.2f [%+.2f, %+.2f]'%(100*(a1.mean()-b1.mean()),*ci(100*(a1[I].mean(1)-b1[I].mean(1))),100*(a16.mean()-b16.mean()),*ci(100*(a16[I].mean(1)-b16[I].mean(1)))))
# per-set pass@256 as in aggregate (fraction of questions with any correct among 256)
for ds in ['aime24','aime25','aime2022','aime2023']:
    rs=[r for r in R4 if r['ds']==ds]
    print(ds,'n=%d rl_replace pass@256(any) %.2f  unbiased k=256 %.2f'%(len(rs),100*np.mean([any(r['conds']['rl_replace']) for r in rs]),100*np.mean([pk(r['conds']['rl_replace'],256) for r in rs])))
