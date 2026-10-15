# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
# Read-only streaming scanner for stage1 jsonl files. Writes npz + json stats to ${WRB_WORK}/<tag>.*
import json, sys, re, os, time
import numpy as np
path, tag = sys.argv[1], sys.argv[2]
out = '${WRB_WORK}/' + tag
FEATS = ['entropy','margin','top1_prob','token_rank','chosen_logp','rel_pos']
ORAC = ['neg_logp','kl_fwd','kl_rev','js']
EXTRA = ['rl_top_logprob','chosen_rank_in_rl','rl_top_id']
def last_boxed(s):
    i = s.rfind('\\boxed')
    if i < 0: return None
    j = s.find('{', i)
    if j < 0: return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == '{': d += 1
        elif s[k] == '}':
            d -= 1
            if d == 0: return s[j+1:k]
    return None
def norm(a):
    if a is None: return None
    a = a.strip().replace(' ', '').replace('\\left','').replace('\\right','').replace('\\!','').replace('dfrac','frac').replace('tfrac','frac').replace('$','')
    a = a.rstrip('.')
    return a
cols = {k: [] for k in FEATS + ORAC + EXTRA}
seq_id, rec_id, seq_len_tok = [], [], []
recs = []
t0 = time.time()
n_rec = 0; seq_counter = 0; has_extra = None
with open(path) as f:
    for line in f:
        r = json.loads(line)
        q = r['question']; gt = str(r['gt'])
        gt_ans = last_boxed(gt) if '\\boxed' in gt else gt
        L = r['lengths']; texts = r['generated_texts']
        tf = r['token_features']; os_ = r['oracle_signals']
        rec = {'q_head': q[:100], 'gt_ans': (gt_ans or '')[:60], 'lengths': L, 'n': len(L), 'correct': [], 'ends': [], 'tf_len': [len(x) for x in tf], 'os_len': [len(x) for x in os_]}
        for si in range(len(L)):
            t = texts[si]
            pred = last_boxed(t)
            rec['correct'].append(bool(pred is not None and norm(pred) == norm(gt_ans)))
            rec['ends'].append('eot' if t.endswith('<|endoftext|>') or t.endswith('<|im_end|>') else 'other')
            feats = tf[si]; ors = os_[si]
            m = len(feats)
            for k in FEATS:
                cols[k].append(np.fromiter((d[k] for d in feats), dtype=np.float32, count=m))
            for k in ORAC:
                cols[k].append(np.fromiter((d[k] for d in ors), dtype=np.float32, count=len(ors)))
            if has_extra is None: has_extra = ('rl_top_logprob' in ors[0]) if ors else False
            if has_extra:
                for k in EXTRA:
                    cols[k].append(np.fromiter((d[k] for d in ors), dtype=np.float64 if k=='rl_top_id' else np.float32, count=len(ors)))
            seq_id.append(np.full(m, seq_counter, dtype=np.int32)); rec_id.append(np.full(m, n_rec, dtype=np.int32))
            seq_len_tok.append(np.full(m, L[si], dtype=np.int32))
            seq_counter += 1
        recs.append(rec); n_rec += 1
        del r, tf, os_, texts
        if n_rec % 50 == 0: print(tag, 'records', n_rec, 'elapsed', round(time.time()-t0), flush=True)
arrs = {k: np.concatenate(v) for k, v in cols.items() if v}
arrs['seq_id'] = np.concatenate(seq_id); arrs['rec_id'] = np.concatenate(rec_id); arrs['seq_len'] = np.concatenate(seq_len_tok)
np.savez(out + '.npz', **arrs)
json.dump({'path': path, 'n_records': n_rec, 'n_seqs': seq_counter, 'has_extra': has_extra, 'records': recs, 'elapsed_s': time.time()-t0}, open(out + '.json', 'w'), ensure_ascii=False)
print(tag, 'DONE', n_rec, seq_counter, sum(len(a) for a in cols['entropy']), round(time.time()-t0), flush=True)
