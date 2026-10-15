# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
from lib import *
import numpy as np
runs={'alright120':['alright','alright_i'],'rule_injection':['soft2','soft4','pronoun05','pronoun1'],'wait_topk':['topk','alright_topk','alright'],
      'greedy_prefix':['greedy','soft','al_greedy'],'early_stop':['early_low','early_both']}
v2={key(r['question']):r for r in recs('v2')}
G={'all120':lambda q:True,'clean60':lambda q:group(q)=='clean','contam60':lambda q:group(q)=='contam'}
print('reused base check: base_accs identical to v2 base?')
for run in runs:
    rs=recs(run); same=sum(1 for r in rs if r['conds']['base']==v2[key(r['question'])]['conds']['base'])
    print('  ',run,'%d/%d'%(same,len(rs)))
for run,conds in runs.items():
    rs=recs(run)
    for g,sel in G.items():
        S=[r for r in rs if sel(r['question'])]; n=len(S)
        b=np.array([p1(r['conds']['base']) for r in S]); bk=np.array([float(any(r['conds']['base'])) for r in S])
        I=boot_idx(n,5000,11)
        parts=['base %.2f/%.2f'%(100*b.mean(),100*bk.mean())]
        for c in conds:
            x=np.array([p1(r['conds'][c]) for r in S]); xk=np.array([float(any(r['conds'][c])) for r in S])
            d=x-b; lo,hi=ci(100*d[I].mean(1))
            parts.append('%s %.2f/%.2f Δ%+.2f[%+.2f,%+.2f] Δk16 %+.2f'%(c,100*x.mean(),100*xk.mean(),100*d.mean(),lo,hi,100*(xk.mean()-bk.mean())))
        print('%-15s %-8s n=%d | '%(run,g,n)+' | '.join(parts))
# v2 rl_replace gain per group for "share of gain" reference
for g,sel in G.items():
    S=[r for r in v2.values() if sel(r['question'])]
    b=np.mean([p1(r['conds']['base']) for r in S]); l=np.mean([p1(r['conds']['rl_replace']) for r in S])
    bk=np.mean([any(r['conds']['base']) for r in S]); lk=np.mean([any(r['conds']['rl_replace']) for r in S])
    print('v2 rl_replace gain %-8s p@1 %+.2f  p@16 %+.2f'%(g,100*(l-b),100*(lk-bk)))
# same-run comparisons
def samerun(run,c1,c2,g):
    S=[r for r in recs(run) if G[g](r['question'])]; n=len(S)
    d=np.array([p1(r['conds'][c1])-p1(r['conds'][c2]) for r in S]); I=boot_idx(n,5000,13)
    return '%s-%s %s %+.2f [%+.2f,%+.2f]'%(c1,c2,g,100*d.mean(),*ci(100*d[I].mean(1)))
for g in G:
    print('same-run:',samerun('rule_injection','pronoun05','soft2',g),'|',samerun('rule_injection','pronoun1','soft2',g),'|',samerun('early_stop','early_both','early_low',g))
# 9-question alright
A9=recs('alright9')
for g in ('clean','contam'):
    S=[r for r in A9 if group(r['question'])==g]
    if not S: continue
    print('alright9',g,'n=%d'%len(S),' '.join('%s %.2f/%.2f'%(c,100*np.mean([p1(r['conds'][c]) for r in S]),100*np.mean([any(r['conds'][c]) for r in S])) for c in ('base','base_with_alright','base_with_alright_i')),[(r['ds'],r['qi'],r['gt']) for r in S])
