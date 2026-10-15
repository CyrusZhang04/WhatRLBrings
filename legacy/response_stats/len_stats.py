# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,statistics
# usage: len_stats.py MAXTOK files...
MAX=int(sys.argv[1]); files=sys.argv[2:]
agg={}
nq=0
for fn in files:
    with open(fn) as f:
        for line in f:
            d=json.loads(line); nq+=1
            for cond in ('base','rl_replace','full_rl'):
                c=d.get(cond)
                if c is None: continue
                a=agg.setdefault(cond,{'L':[], 'A':[], 'rr':[]})
                a['L']+=c['lengths']; a['A']+=[bool(x) for x in c['accs']]
                if 'replace_rates' in c: a['rr']+=c['replace_rates']
print('questions',nq)
for cond,a in agg.items():
    L=a['L']; A=a['A']
    s=f"{cond:10s} samples={len(L)} acc={sum(A)/len(A):.4f} meanlen={statistics.mean(L):.0f} medlen={statistics.median(L):.0f} frac_at_max={sum(1 for x in L if x>=MAX)/len(L):.3f}"
    if a['rr']:
        s+=f" mean_rr(sample-avg)={statistics.mean(a['rr']):.4f}"
    print(s)
