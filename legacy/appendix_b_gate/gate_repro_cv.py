# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import numpy as np, json, sys, time
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
tag = sys.argv[1]
d = np.load(f'${WRB_WORK}/{tag}.npz')
FEATS = ['entropy','margin','top1_prob','token_rank','chosen_logp','rel_pos']
X = np.stack([d[f].astype(np.float64) for f in FEATS], 1)
kf = d['kl_fwd'].astype(np.float64)
y = kf >= np.percentile(kf, 90)
def prec_at_recall(y, s, target=0.9):
    p, r, t = precision_recall_curve(y, s)
    # choose highest threshold with recall >= target
    idx = np.where(r[:-1] >= target)[0]
    i = idx[-1]
    return float(p[i]), float(t[i])
res = {}
t0=time.time()
skf = StratifiedKFold(5, shuffle=True, random_state=0)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
s = cross_val_predict(lr, X, y, cv=skf, method='predict_proba')[:,1]
res['lr_full_cv5_auroc'] = roc_auc_score(y, s); res['lr_full_cv5_prec@r0.9,thr'] = prec_at_recall(y, s)
lrb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
s = cross_val_predict(lrb, X, y, cv=skf, method='predict_proba')[:,1]
res['lr_bal_cv5_auroc'] = roc_auc_score(y, s); res['lr_bal_cv5_prec@r0.9,thr'] = prec_at_recall(y, s)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for name, cw in [('hgb', None), ('hgb_bal', 'balanced')]:
    m = HistGradientBoostingClassifier(class_weight=cw, random_state=42)
    m.fit(Xtr, ytr); s = m.predict_proba(Xte)[:,1]
    res[name+'_holdout_auroc'] = roc_auc_score(yte, s); res[name+'_holdout_prec@r0.9,thr'] = prec_at_recall(yte, s)
    s2 = m.predict_proba(X)[:,1]
    res[name+'_all_auroc'] = roc_auc_score(y, s2); res[name+'_all_prec@r0.9,thr'] = prec_at_recall(y, s2)
for name, cw in [('lr', None), ('lr_bal', 'balanced')]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw)); m.fit(Xtr, ytr); s = m.predict_proba(Xte)[:,1]
    res[name+'_holdout_auroc'] = roc_auc_score(yte, s); res[name+'_holdout_prec@r0.9,thr'] = prec_at_recall(yte, s)
res['elapsed'] = time.time()-t0
print(tag, json.dumps(res, indent=1))
json.dump(res, open(f'${WRB_WORK}/{tag}.gate_repro.json','w'), indent=1)
