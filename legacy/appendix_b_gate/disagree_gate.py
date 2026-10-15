# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import numpy as np, json
from sklearn.metrics import roc_auc_score, precision_recall_curve
out={}
for tag in ['a24n16','a25n16']:
    d=np.load(f'${WRB_WORK}/{tag}.npz')
    dis = d['chosen_rank_in_rl'] != 1
    kf = d['kl_fwd']; crit = kf >= np.percentile(kf, 90)
    r={}
    for lab_name, y in [('disagree', dis), ('crit_klfwd_top10', crit)]:
        rr={'pos_rate': float(y.mean())}
        for f, sign in [('entropy',1),('top1_prob',-1),('margin',-1),('chosen_logp',-1)]:
            s = sign*d[f].astype(np.float64)
            auc = roc_auc_score(y, s)
            p, rc, t = precision_recall_curve(y, s)
            idx = np.where(rc[:-1] >= 0.9)[0]; i = idx[-1]
            flagged = float((s >= t[i]).mean())
            rr[f] = dict(auroc=round(float(auc),4), prec_at_r90=round(float(p[i]),4), thr=float(sign*t[i]), flagged_frac_at_r90=round(flagged,4))
        r[lab_name]=rr
    # clean subset (seq_len < 16000)
    clean = d['seq_len'] < 16000
    s = d['entropy'][clean].astype(np.float64); y = crit[clean]
    p, rc, t = precision_recall_curve(y, s); idx = np.where(rc[:-1] >= 0.9)[0]; i = idx[-1]
    r['clean_len_lt_16000'] = dict(n_tokens=int(clean.sum()), crit_rate=float(y.mean()), entropy_auroc=round(float(roc_auc_score(y, s)),4), prec_at_r90=round(float(p[i]),4), flagged_frac=round(float((s>=t[i]).mean()),4), disagree_rate=float(dis[clean].mean()))
    out[tag]=r
print(json.dumps(out, indent=1))
json.dump(out, open('${WRB_WORK}/disagree_gate.json','w'), indent=1)
