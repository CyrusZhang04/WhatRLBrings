# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import numpy as np, json, sys
from sklearn.metrics import roc_auc_score, precision_recall_curve
tag = sys.argv[1]
d = np.load(f'${WRB_WORK}/{tag}.npz')
meta = json.load(open(f'${WRB_WORK}/{tag}.json'))
FEATS = ['entropy','margin','top1_prob','token_rank','chosen_logp','rel_pos']
out = {'tag': tag, 'n_records': meta['n_records'], 'n_seqs': meta['n_seqs'], 'n_tokens': int(len(d['entropy']))}
recs = meta['records']
L = np.array([l for r in recs for l in r['lengths']])
out['len_stats'] = dict(mean=float(L.mean()), median=float(np.median(L)), p90=float(np.percentile(L,90)), max=int(L.max()), min=int(L.min()), n_ge_16000=int((L>=16000).sum()), n_ge_30000=int((L>=30000).sum()))
out['n_per_record'] = sorted(set(r['n'] for r in recs))
out['tf_len_match'] = all(r['tf_len']==r['lengths'] and r['os_len']==r['lengths'] for r in recs)
corr = np.array([c for r in recs for c in r['correct']])
ends = [e for r in recs for e in r['ends']]
out['approx_boxed_acc'] = float(corr.mean())
out['frac_end_eot'] = float(np.mean([e=='eot' for e in ends]))
# per-question any-correct
out['approx_pass_at_n'] = float(np.mean([any(r['correct']) for r in recs]))
# field summaries
summ = {}
for k in d.files:
    if k in ('seq_id','rec_id','seq_len','rl_top_id'): continue
    a = d[k].astype(np.float64)
    summ[k] = dict(mean=float(a.mean()), p10=float(np.percentile(a,10)), p50=float(np.percentile(a,50)), p90=float(np.percentile(a,90)), min=float(a.min()), max=float(a.max()))
out['field_summary'] = summ
# relations
kf, kr, js = d['kl_fwd'].astype(np.float64), d['kl_rev'].astype(np.float64), d['js'].astype(np.float64)
out['js_minus_avgkl_maxabs'] = float(np.max(np.abs(js - 0.5*(kf+kr))))
nl = d['neg_logp'].astype(np.float64); cl = d['chosen_logp'].astype(np.float64)
out['neg_logp_frac_pos'] = float((nl>1e-6).mean())
rank = d['token_rank']
out['frac_token_rank_eq1'] = float((rank==1).mean())
r1 = rank==1
out['top1prob_vs_exp_chosenlogp_when_rank1_mean_absdiff'] = float(np.mean(np.abs(d['top1_prob'][r1] - np.exp(cl[r1]))))
out['top1prob_vs_exp_chosenlogp_when_rank1_median_absdiff'] = float(np.median(np.abs(d['top1_prob'][r1] - np.exp(cl[r1]))))
if 'chosen_rank_in_rl' in d.files:
    cr = d['chosen_rank_in_rl']; rtl = d['rl_top_logprob'].astype(np.float64)
    out['frac_chosen_rank_in_rl_eq1'] = float((cr==1).mean())
    m = cr==1
    out['neg_logp_vs_(rltop-chosen)_when_rlrank1_mean_absdiff'] = float(np.mean(np.abs(nl[m] - (rtl[m]-cl[m]))))
    out['neg_logp_vs_(chosen-rltop)_when_rlrank1_mean_absdiff'] = float(np.mean(np.abs(nl[m] - (cl[m]-rtl[m]))))
    out['chosen_rank_in_rl_values'] = {str(int(v)): int(c) for v, c in zip(*np.unique(np.clip(cr, -1, 50), return_counts=True))}
# critical labels
def lab_top(a, pct=10):
    thr = np.percentile(a, 100-pct); return a >= thr, float(thr)
def lab_bot(a, pct=10):
    thr = np.percentile(a, pct); return a <= thr, float(thr)
labels = {}
y, thr = lab_top(kf); labels['kl_fwd_top10'] = (y, thr)
y, thr = lab_bot(nl); labels['neg_logp_bottom10'] = (y, thr)
y, thr = lab_top(nl); labels['neg_logp_top10'] = (y, thr)
y, thr = lab_top(kr); labels['kl_rev_top10'] = (y, thr)
y, thr = lab_top(js); labels['js_top10'] = (y, thr)
res = {}
for name, (y, thr) in labels.items():
    rr = {'threshold': thr, 'pos_rate': float(y.mean()), 'auroc_raw': {}}
    for f in FEATS:
        a = d[f].astype(np.float64)
        auc = roc_auc_score(y, a)
        rr['auroc_raw'][f] = round(float(auc), 4)
    res[name] = rr
out['labels'] = res
# overlap with argmax disagreement
if 'chosen_rank_in_rl' in d.files:
    y = labels['kl_fwd_top10'][0]; dis = d['chosen_rank_in_rl'] != 1
    out['disagree_rate_overall'] = float(dis.mean())
    out['P(disagree|crit_klfwd)'] = float(dis[y].mean())
    out['P(crit_klfwd|disagree)'] = float(y[dis].mean())
    # also base greedy != rl greedy cannot be computed directly (no base argmax id), but token_rank==1 & chosen_rank_in_rl!=1 means base-top1 != rl-top1
    bt = d['token_rank']==1
    both = bt & dis
    out['frac_(base_top1_chosen & rl_top1_differs)'] = float(both.mean())
    out['P(rl_top1!=chosen | chosen is base top1)'] = float(dis[bt].mean())
# degenerate sequences contribution
sl = d['seq_len']
deg = sl >= 16000
out['frac_tokens_in_len>=16000_seqs'] = float(deg.mean())
y = labels['kl_fwd_top10'][0]
out['frac_crit_klfwd_tokens_in_len>=16000_seqs'] = float(deg[y].mean())
if deg.any() and (~deg).any():
    yy = y[~deg]
    out['auroc_raw_excluding_len>=16000'] = {f: round(float(roc_auc_score(yy, d[f][~deg].astype(np.float64))),4) for f in FEATS}
# rel_pos check
out['rel_pos_first_values'] = [float(x) for x in d['rel_pos'][:3]]
json.dump(out, open(f'${WRB_WORK}/{tag}.analysis.json','w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
