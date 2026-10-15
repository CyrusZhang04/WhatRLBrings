# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, zlib, hashlib, collections
path, tag = sys.argv[1], sys.argv[2]
n_rec=0; n_seq=0; dup_seqs=0; recs_with_dup=0; all_same=0
tail_ratios=[]; deg=0; deg_tok=0; tot_tok=0
first_tok_starts=collections.Counter()
with open(path) as f:
    for line in f:
        r=json.loads(line); n_rec+=1
        T=r['generated_texts']; L=r['lengths']
        hs=[hashlib.md5(t.encode('utf-8','ignore')).hexdigest() for t in T]
        c=collections.Counter(hs)
        d=sum(v-1 for v in c.values() if v>1)
        dup_seqs+=d; recs_with_dup+= (d>0); all_same += (len(c)==1)
        for t,l in zip(T,L):
            n_seq+=1; tot_tok+=l
            tail=t[-4000:].encode('utf-8','ignore')
            ratio=len(zlib.compress(tail))/max(1,len(tail))
            tail_ratios.append(ratio)
            if ratio<0.12 and not (t.endswith('<|endoftext|>') or t.endswith('<|im_end|>')):
                deg+=1; deg_tok+=l
            first_tok_starts[t[:9]]+=1
        del r
import numpy as np
tr=np.array(tail_ratios)
out=dict(tag=tag, n_rec=n_rec, n_seq=n_seq, exact_duplicate_seqs=dup_seqs, records_with_dup=recs_with_dup, records_all_identical=all_same,
         degenerate_seqs_heuristic=deg, degenerate_token_frac=deg_tok/tot_tok, tail_zlib_ratio_p10=float(np.percentile(tr,10)), tail_zlib_ratio_p50=float(np.percentile(tr,50)),
         top_starts=first_tok_starts.most_common(8))
print(json.dumps(out, ensure_ascii=False))
json.dump(out, open(f'${WRB_WORK}/{tag}.dup.json','w'), ensure_ascii=False)
