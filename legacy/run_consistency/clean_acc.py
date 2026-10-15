# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys
buckets=[(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,63)]
for name in ['rlr','opd']:
    recs=json.load(open(f'${WRB_WORK}/{name}_math500_flags.json'))
    print('=====',name)
    for m in ['base','rl_replace','full_rl']:
        rows=[]
        for a,b in buckets:
            R=[r for r in recs if r['m']==m and a<=r['i']<b]
            cl=[(acc) for r in R for acc,f in zip(r['accs'],r['fl']) if not any(f)]
            an=[(acc) for r in R for acc,f in zip(r['accs'],r['fl']) if any(f)]
            rows.append(f'{a}-{b-1}: cleanAcc={100*sum(cl)/max(1,len(cl)):5.1f}(n={len(cl)}) anomAcc={100*sum(an)/max(1,len(an)):4.1f}(n={len(an)})')
        print(m.ljust(10),' | '.join(rows))
# cross-run per-question full_rl agreement
a=json.load(open('${WRB_WORK}/rlr_math500_flags.json')); b=json.load(open('${WRB_WORK}/opd_math500_flags.json'))
A={(r['shard'],r['i']):r for r in a if r['m']=='full_rl'}; B={(r['shard'],r['i']):r for r in b if r['m']=='full_rl'}
import statistics
for lo,hi in buckets+[(0,30),(30,63)]:
    ks=[k for k in A if lo<=k[1]<hi]
    xa=[sum(A[k]['accs'])/16 for k in ks]; xb=[sum(B[k]['accs'])/16 for k in ks]
    d=[x-y for x,y in zip(xa,xb)]
    try: c=statistics.correlation(xa,xb)
    except Exception as e: c=float('nan')
    print(f'full_rl cross-run {lo}-{hi-1}: meanA={100*statistics.mean(xa):.2f} meanB={100*statistics.mean(xb):.2f} mean|diff|={100*statistics.mean(abs(x) for x in d):.2f} corr={c:.3f} nq={len(ks)}')
