# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
from lib import *
import numpy as np
A=[80,101,110,175,276,285,292,412,239,150,475]; A2=[138,497,126]; B=[46,320,123,177,189,360,63,238,155,115,428,118,474]
STRICT=set(A+A2); BROAD=set(A+A2+B)
def idx_of(q):
    o=M500.get(key(q)); return int(o['src'].split(':')[1]) if o else None
R=recs('m500_rlr'); O=recs('opd:opd_step200','math500_500')
for rs in (R,O):
    ok=sum(1 for r in rs if idx_of(r['question'])==63*r['shard']+r['i']); print('order check 63*shard+i == math500-raw idx: %d/%d'%(ok,len(rs)))
rl={idx_of(r['question']):r for r in R}; op={idx_of(r['question']):r for r in O}
sub=[k for k in range(500) if rl[k]['i']<30]
print('240-subset size',len(sub),' strict overlaps in subset:',sorted(k for k in sub if k in STRICT),' broad overlaps in subset:',sorted(k for k in sub if k in BROAD))
print('levels of strict-in-subset:',[(k,M500[key(rl[k]['question'])]['level']) for k in sub if k in STRICT])
def stats(ks,label,Bn=4000,seed=41):
    b=np.array([p1(rl[k]['conds']['base']) for k in ks]); l=np.array([p1(rl[k]['conds']['rl_replace']) for k in ks]); f=np.array([p1(rl[k]['conds']['full_rl']) for k in ks])
    s=np.array([p1(op[k]['conds']['base']) for k in ks]); fo=np.array([p1(op[k]['conds']['full_rl']) for k in ks]); lo=np.array([p1(op[k]['conds']['rl_replace']) for k in ks])
    n=len(ks); I=boot_idx(n,Bn,seed)
    with np.errstate(divide='ignore',invalid='ignore'):
        rec=(l[I].mean(1)-b[I].mean(1))/(f[I].mean(1)-b[I].mean(1))
        clo=(s[I].mean(1)-b[I].mean(1))/(fo[I].mean(1)-b[I].mean(1))
    print('%-34s n=%3d | rlr: base %.2f rl %.2f full %.2f -> recovery %.1f%% CI [%.1f, %.1f] | opd: student %.2f rl %.2f full %.2f -> D1 closure %.1f%% CI [%.1f, %.1f] (closure w/ rlr full %.1f%%)'%(
        label,n,100*b.mean(),100*l.mean(),100*f.mean(),100*(l.mean()-b.mean())/(f.mean()-b.mean()),*[100*x for x in ci(rec)],
        100*s.mean(),100*lo.mean(),100*fo.mean(),100*(s.mean()-b.mean())/(fo.mean()-b.mean()),*[100*x for x in ci(clo)],100*(s.mean()-b.mean())/(f.mean()-b.mean())))
    return b,l,f
stats(sub,'240 subset (all)')
stats([k for k in sub if k not in STRICT],'240 minus strict overlaps')
stats([k for k in sub if k not in BROAD],'240 minus broad overlaps (cont>=.5)')
stats([k for k in sub if k in BROAD],'overlaps only (broad) in 240')
stats([k for k in sub if k in STRICT],'overlaps only (strict) in 240')
# full 500 (base/rl only, full_rl damaged) for reference: rl-base gain with/without overlaps
for lab,ks in (('500 all',list(range(500))),('500 minus strict',[k for k in range(500) if k not in STRICT]),('500 minus broad',[k for k in range(500) if k not in BROAD])):
    b=np.mean([p1(rl[k]['conds']['base']) for k in ks]); l=np.mean([p1(rl[k]['conds']['rl_replace']) for k in ks]); s=np.mean([p1(op[k]['conds']['base']) for k in ks])
    print('%-18s n=%d base %.2f rl_replace %.2f (rl-base %+.2f) v0 student %.2f'%(lab,len(ks),100*b,100*l,100*(l-b),100*s))
# per-problem detail for overlaps
print('overlap detail (idx, class, level, in240, base/rl/full of 16, student/16):')
for k in sorted(BROAD):
    cls='A' if k in A else 'A2' if k in A2 else 'B'
    print('  %3d %-2s L%d in240=%-5s base %2d rl %2d full %2d | student %2d | opd full %2d'%(k,cls,M500[key(rl[k]['question'])]['level'],k in sub,sum(rl[k]['conds']['base']),sum(rl[k]['conds']['rl_replace']),sum(rl[k]['conds']['full_rl']),sum(op[k]['conds']['base']),sum(op[k]['conds']['full_rl'])))
