# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,random
rlr=json.load(open('${WRB_WORK}/rlr_math500_flags.json')); opd=json.load(open('${WRB_WORK}/opd_math500_flags.json'))
def per_q(R,m): return {(r['shard'],r['i']):sum(r['accs'])/16 for r in R if r['m']==m and r['i']<30}
B=per_q(rlr,'base'); RR=per_q(rlr,'rl_replace'); FA=per_q(rlr,'full_rl'); ST=per_q(opd,'base'); FB=per_q(opd,'full_rl')
keys=sorted(B)
random.seed(0)
rec=[];clo=[];clo2=[];gapS=[]
for _ in range(4000):
    ks=[random.choice(keys) for _ in keys]
    b=sum(B[k] for k in ks); rr=sum(RR[k] for k in ks); fa=sum(FA[k] for k in ks); fb=sum(FB[k] for k in ks); st=sum(ST[k] for k in ks)
    fp=(fa+fb)/2
    rec.append((rr-b)/(fa-b)); clo.append((st-b)/(fb-b)); clo2.append((st-b)/(fp-b))
def ci(x):
    x=sorted(x); return f'{100*x[int(0.025*len(x))]:.1f}–{100*x[int(0.975*len(x))]:.1f}%'
print('n_q',len(keys))
print('recovery (rl_replace-base)/(full_rl_rlr-base) 95% CI:',ci(rec))
print('D1 v0 closure (student-base)/(full_rl_opd-base) 95% CI:',ci(clo))
print('D1 v0 closure pooled full_rl 95% CI:',ci(clo2))
