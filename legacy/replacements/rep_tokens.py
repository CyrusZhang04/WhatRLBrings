# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,os,collections
from tokenizers import Tokenizer
tokpath=sys.argv[1]; files=sys.argv[2:]
tok=Tokenizer.from_file(tokpath)
def dec(i):
    s=tok.decode([i], skip_special_tokens=False)
    return s
pairs=collections.Counter(); rl_tok=collections.Counter(); base_tok=collections.Counter(); first=collections.Counter()
n_ev=0
for fn in files:
    with open(fn) as f:
        for line in f:
            d=json.loads(line)
            reps=d['rl_replace'].get('replacements')
            if reps is None: continue
            for ev in reps:
                for e in ev:
                    n_ev+=1
                    pairs[(e['base_id'],e['rl_id'])]+=1; rl_tok[e['rl_id']]+=1; base_tok[e['base_id']]+=1
                    if e['pos']==0: first[(e['base_id'],e['rl_id'])]+=1
print('total events',n_ev)
print('--- top 40 (base -> rl) pairs')
for (b,r),c in pairs.most_common(40):
    print(f"{c:7d} {c/n_ev:.4f}  {dec(b)!r:>20s} -> {dec(r)!r}")
print('--- top 30 inserted RL tokens')
for r,c in rl_tok.most_common(30):
    print(f"{c:7d} {c/n_ev:.4f}  {dec(r)!r}")
print('--- top 20 replaced base tokens')
for b,c in base_tok.most_common(20):
    print(f"{c:7d} {c/n_ev:.4f}  {dec(b)!r}")
print('--- top 10 pos0 pairs')
tot0=sum(first.values())
for (b,r),c in first.most_common(10):
    print(f"{c:7d} {c/tot0:.4f}  {dec(b)!r} -> {dec(r)!r}")
# share of top-k pairs
cum=0
for k,(p,c) in enumerate(pairs.most_common(),1):
    cum+=c
    if k in (10,50,100,500,1000): print(f'top{k} pairs cover {cum/n_ev:.3f}')
print('distinct pairs',len(pairs),'distinct rl tokens',len(rl_tok))
