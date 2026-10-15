# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,os,statistics
# For each jsonl file: per question, count unique prefixes (first P chars) per condition
P=int(sys.argv[1]); files=sys.argv[2:]
agg={}
for fn in files:
    with open(fn) as f:
        for line in f:
            d=json.loads(line)
            for cond in ('base','rl_replace','full_rl'):
                c=d.get(cond)
                if c is None: continue
                texts=c.get('texts')
                if texts is None:
                    continue
                u=len(set(t[:P] for t in texts))
                agg.setdefault(cond,[]).append((u,len(texts)))
for cond,v in agg.items():
    us=[u for u,n in v]; ns=[n for u,n in v]
    print(f"{cond:10s} questions={len(v)} n={ns[0]} unique_prefix{P}: mean={statistics.mean(us):.1f} min={min(us)} max={max(us)} frac_q_with_<=2_unique={sum(1 for u in us if u<=2)/len(us):.2f}")
