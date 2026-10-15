# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, collections
recs=json.load(open(sys.argv[1]))
methods=['base','rl_replace','full_rl']
buckets=[(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,63)]
def bk(i):
    for a,b in buckets:
        if a<=i<b: return f'{a}-{b-1}'
print('### per question-index bucket: anomaly% (any) | acc% (pass@1) | nq')
for m in methods:
    rows=[]
    for a,b in buckets:
        R=[r for r in recs if r['m']==m and a<=r['i']<b]
        ns=sum(len(r['fl']) for r in R)
        an=sum(any(f) for r in R for f in r['fl'])
        ac=sum(sum(r['accs']) for r in R)
        rows.append(f'{a}-{b-1}: anom={100*an/ns:5.1f} acc={100*ac/ns:5.2f} nq={len(R)}')
    print(m.ljust(10),' | '.join(rows))
print('### per criterion (full_rl), per bucket: len / nobox / rep (%)')
for m in methods:
    rows=[]
    for a,b in buckets:
        R=[r for r in recs if r['m']==m and a<=r['i']<b]
        ns=sum(len(r['fl']) for r in R)
        c=[sum(f[k] for r in R for f in r['fl']) for k in range(3)]
        rows.append(f'{a}-{b-1}: '+'/'.join(f'{100*x/ns:.1f}' for x in c))
    print(m.ljust(10),' | '.join(rows))
print('### per sample index j: anomaly%')
for m in methods:
    v=[]
    for j in range(16):
        R=[r for r in recs if r['m']==m]
        v.append(100*sum(any(r['fl'][j]) for r in R)/len(R))
    print(m.ljust(10),' '.join(f'{x:4.1f}' for x in v))
print('### per sample index j (only i<30): anomaly%')
for m in methods:
    v=[]
    R=[r for r in recs if r['m']==m and r['i']<30]
    for j in range(16):
        v.append(100*sum(any(r['fl'][j]) for r in R)/len(R))
    print(m.ljust(10),' '.join(f'{x:4.1f}' for x in v))
def stats(R):
    ns=sum(len(r['fl']) for r in R)
    an=sum(any(f) for r in R for f in r['fl'])
    ac=sum(sum(r['accs']) for r in R)
    p16=sum(any(r['accs']) for r in R)/len(R)
    # pass@1 as mean per-question acc (all n=16 so same as sample mean)
    return dict(nq=len(R), ns=ns, anom=100*an/ns, pass1=100*ac/ns, pass16=100*p16)
for K in [20,25,30,35,40]:
    print(f'### subset i<{K}')
    for m in methods:
        s=stats([r for r in recs if r['m']==m and r['i']<K])
        print(m.ljust(10), {k:(round(v,2) if isinstance(v,float) else v) for k,v in s.items()})
    b=stats([r for r in recs if r['m']=='base' and r['i']<K])['pass1']
    rr=stats([r for r in recs if r['m']=='rl_replace' and r['i']<K])['pass1']
    fr=stats([r for r in recs if r['m']=='full_rl' and r['i']<K])['pass1']
    print(f'   recovery=(rl_replace-base)/(full_rl-base)={100*(rr-b)/(fr-b):.1f}%')
print('### complement i>=30')
for m in methods:
    s=stats([r for r in recs if r['m']==m and r['i']>=30])
    print(m.ljust(10), {k:(round(v,2) if isinstance(v,float) else v) for k,v in s.items()})
print('### all 500')
for m in methods:
    s=stats([r for r in recs if r['m']==m])
    print(m.ljust(10), {k:(round(v,2) if isinstance(v,float) else v) for k,v in s.items()})
