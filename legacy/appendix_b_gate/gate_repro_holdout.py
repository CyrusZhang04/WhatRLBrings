# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import numpy as np, json, sys, time
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
tag = sys.argv[1]
d = np.load(f'${WRB_WORK}/{tag}.npz')
FEATS = ['entropy','margin','top1_prob','token_rank','chosen_logp','rel_pos']
X = np.stack([d[f].astype(np.float32) for f in FEATS], 1)
kf = d['kl_fwd']; y = kf >= np.percentile(kf, 90)
def par(y, s, target=0.9):
    p, r, t = precision_recall_curve(y, s); idx = np.where(r[:-1] >= target)[0]; i = idx[-1]; return round(float(p[i]),4), round(float(t[i]),4)
res = {}; t0 = time.time()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for name, cw in [('hgb', None), ('hgb_bal', 'balanced')]:
    m = HistGradientBoostingClassifier(class_weight=cw, random_state=42); m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:,1]; res[name+'_test'] = (round(roc_auc_score(yte, s),4),) + par(yte, s)
    print(name, res[name+'_test'], round(time.time()-t0), flush=True)
sc = StandardScaler().fit(Xtr)
for name, cw in [('lr', None), ('lr_bal', 'balanced')]:
    m = LogisticRegression(max_iter=200, class_weight=cw); m.fit(sc.transform(Xtr), ytr)
    s = m.predict_proba(sc.transform(Xte))[:,1]; res[name+'_test'] = (round(roc_auc_score(yte, s),4),) + par(yte, s)
    print(name, res[name+'_test'], round(time.time()-t0), flush=True)
json.dump(res, open(f'${WRB_WORK}/{tag}.gate2.json','w'))
print('DONE', json.dumps(res))
