# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, os, glob, time
import pandas as pd
D='${WRB_DATA}'
def prob_from_msg(p):
    c=p[0]['content']
    return c.split('\\boxed{}.\n\n',1)[-1]
sets={}
sets['math500']=[prob_from_msg(p) for p in pd.read_parquet(f'{D}/math500-msg.parquet')['prompt']]
sets['aime24']=[prob_from_msg(p) for p in pd.read_parquet(f'{D}/aime-2024-msg.parquet')['prompt']]
sets['aime25']=[prob_from_msg(p) for p in pd.read_parquet(f'{D}/aime-2025-msg.parquet')['prompt']]
a90=list(pd.read_parquet(f'{D}/aime-2024.parquet')['question'])
sets['aime22_23']=[q for q in a90 if not any(q[:60] in x for x in sets['aime24'])]
snips=[]
for name,lst in sets.items():
    for q in lst:
        q=q.strip()
        s=q[:60] if len(q)>=60 else q
        snips.append((name,s))
def ident(inp):
    hits=set(n for n,s in snips if s in inp)
    if len(hits)==1: return hits.pop()
    if not hits: return 'unknown'
    return '+'.join(sorted(hits))
def main(d,out):
    files=sorted(glob.glob(os.path.join(d,'*.jsonl')), key=lambda p:int(os.path.basename(p).split('.')[0]))
    cache={}
    with open(out,'w') as fo:
        fo.write('step,mtime,n,ds,count,acc,len_mean\n')
        for p in files:
            s=int(os.path.basename(p).split('.')[0])
            mt=time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))
            agg={}
            with open(p) as fh:
                for l in fh:
                    try: r=json.loads(l)
                    except Exception: continue
                    inp=r['input']
                    if inp not in cache: cache[inp]=ident(inp)
                    ds=cache[inp]
                    a=float(r.get('acc', 1.0 if r.get('score',0)>0 else 0.0))
                    L=len(r.get('output',''))
                    x=agg.setdefault(ds,[0,0.0,0])
                    x[0]+=1; x[1]+=a; x[2]+=L
            n=sum(v[0] for v in agg.values())
            for ds,(c,a,L) in sorted(agg.items()):
                fo.write(f'{s},{mt},{n},{ds},{c},{a/c:.4f},{L/c:.0f}\n')
            fo.flush()
if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
