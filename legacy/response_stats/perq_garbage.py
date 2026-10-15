# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,os,re,statistics
CYR=re.compile(r'[Ѐ-ӿ]')
RUN2=re.compile(r'(?:\b(\S{1,3})\s+)(?:\1\s+){25,}')
def bad(t,L):
    if L<10: return True
    if CYR.search(t[:200]): return True
    if re.match(r'^\s*[\d\.\s\$]{1,60}$', t[:60]) and L<200: return True
    if RUN2.search(t[-3000:]): return True
    return False
for d in sys.argv[1:]:
    out={'base':[], 'full_rl':[], 'rl_replace':[]}
    accs={'base':[], 'full_rl':[], 'rl_replace':[]}
    for fn in sorted(glob.glob(os.path.join(d,'*.jsonl'))):
        with open(fn) as f:
            for line in f:
                x=json.loads(line)
                for c in out:
                    T=x[c]['texts']; L=x[c]['lengths']; A=x[c]['accs']
                    b=[bad(t,l) for t,l in zip(T,L)]
                    out[c].append(sum(b)/len(b))
                    # accuracy among non-bad samples
                    good=[a for a,bb in zip(A,b) if not bb]
                    accs[c].append((sum(A)/len(A), (sum(good)/len(good)) if good else None, len(good)))
    print('==',d,'nq',len(out['base']))
    for c,v in out.items():
        hi=sum(1 for x in v if x>0.5); mid=sum(1 for x in v if 0.1<x<=0.5); z=sum(1 for x in v if x==0)
        acc_all=statistics.mean(a for a,_,_ in accs[c]); acc_good=statistics.mean(g for _,g,n in accs[c] if g is not None)
        print(f"  {c:10s} mean_bad={statistics.mean(v):.3f} q_bad>50%={hi} q_bad10-50%={mid} q_bad=0:{z} | acc_all={acc_all:.4f} acc_among_nonflagged(per-q mean)={acc_good:.4f}")
