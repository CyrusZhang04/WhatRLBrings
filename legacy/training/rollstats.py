# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, os, statistics, collections, glob
def stats_file(path):
    rows=[]
    with open(path) as fh:
        for l in fh:
            l=l.strip()
            if not l: continue
            try: rows.append(json.loads(l))
            except Exception: pass
    if not rows: return None
    groups=collections.defaultdict(list)
    for r in rows: groups[r.get('input')].append(r)
    def acc(r):
        a=r.get('acc')
        if a is None:
            s=r.get('score',0); return 1.0 if s>0 else 0.0
        return float(a)
    accs=[acc(r) for r in rows]
    scores=[float(r.get('score',0)) for r in rows]
    L=[len(r.get('output','')) for r in rows]
    gsz=[len(v) for v in groups.values()]
    allc=sum(1 for v in groups.values() if all(acc(x)>0 for x in v))
    allw=sum(1 for v in groups.values() if all(acc(x)<=0 for x in v))
    # response chars by correctness
    Lc=[len(r.get('output','')) for r in rows if acc(r)>0]
    Lw=[len(r.get('output','')) for r in rows if acc(r)<=0]
    return dict(rows=len(rows), prompts=len(groups), gmin=min(gsz), gmax=max(gsz), acc=sum(accs)/len(accs), score=sum(scores)/len(scores),
        len_mean=sum(L)/len(L), len_med=statistics.median(L), len_max=max(L), len_c=(sum(Lc)/len(Lc) if Lc else float('nan')), len_w=(sum(Lw)/len(Lw) if Lw else float('nan')),
        g_allcorrect=allc, g_allwrong=allw, step=rows[0].get('step'))
if __name__=='__main__':
    d=sys.argv[1]; out=sys.argv[2]; every=int(sys.argv[3]) if len(sys.argv)>3 else 1
    files=sorted(glob.glob(os.path.join(d,'*.jsonl')), key=lambda p:int(os.path.basename(p).split('.')[0]))
    with open(out,'w') as fo:
        fo.write('file_step,mtime,rows,prompts,gmin,gmax,acc,score,len_mean,len_med,len_max,len_correct,len_wrong,g_allcorrect,g_allwrong,step_field\n')
        for p in files:
            s=int(os.path.basename(p).split('.')[0])
            if every>1 and s%every!=0 and s!=1 and p!=files[-1]: continue
            st=stats_file(p)
            if st is None: continue
            import time
            mt=time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))
            fo.write(f"{s},{mt},{st['rows']},{st['prompts']},{st['gmin']},{st['gmax']},{st['acc']:.4f},{st['score']:.4f},{st['len_mean']:.0f},{st['len_med']:.0f},{st['len_max']},{st['len_c']:.0f},{st['len_w']:.0f},{st['g_allcorrect']},{st['g_allwrong']},{st['step']}\n")
            fo.flush()
