# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, os, glob, re, collections
sys.path.insert(0,'${WRB_WORK}')
from anom_lib import flags
D = sys.argv[1]
methods = ['base','rl_replace','full_rl']
buckets = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,63)]
def bk(i):
    for a,b in buckets:
        if a<=i<b: return f'{a}-{b-1}'
recs=[]  # (shard, i, question, method, accs list, anom list, flags)
for s in range(8):
    fn=os.path.join(D,f'math500_500-n16-shard{s}of8.jsonl')
    with open(fn) as fh:
        for i,line in enumerate(fh):
            d=json.loads(line)
            for m in methods:
                o=d[m]
                fl=[flags(t,l) for t,l in zip(o['texts'],o['lengths'])]
                recs.append(dict(shard=s,i=i,q=d['question'],m=m,accs=[bool(x) for x in o['accs']],fl=fl,lengths=o['lengths']))
json.dump(recs, open(sys.argv[2],'w'))
print('saved',len(recs))
