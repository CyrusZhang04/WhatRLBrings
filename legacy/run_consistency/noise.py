# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,random
a=json.load(open('${WRB_WORK}/rlr_math500_flags.json')); b=json.load(open('${WRB_WORK}/opd_math500_flags.json'))
A={(r['shard'],r['i']):sum(r['accs']) for r in a if r['m']=='full_rl' and r['i']<30}
B={(r['shard'],r['i']):sum(r['accs']) for r in b if r['m']=='full_rl' and r['i']<30}
random.seed(0); sims=[]
for _ in range(300):
    tot=0
    for k in A:
        p=(A[k]+B[k])/32
        x=sum(random.random()<p for _ in range(16)); y=sum(random.random()<p for _ in range(16))
        tot+=abs(x-y)/16
    sims.append(100*tot/len(A))
sims.sort()
obs=100*sum(abs(A[k]-B[k])/16 for k in A)/len(A)
print('observed mean|diff| i<30 = %.2fpp; binomial-null mean=%.2f, 95%% range %.2f–%.2f'%(obs,sum(sims)/len(sims),sims[7],sims[292]))
