# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, re, collections, numpy as np
def key(s): return re.sub(r'\s+',' ',str(s)).strip()
FL=json.load(open('${WRB_WORK}/eval_flags.json'))
YEAR={key(o['q']):o['set'] for o in FL if o['set']!='math500'}
CONT={key(o['q']):o.get('cont',0.0) for o in FL}
M500={}
for o in FL:
    if o['set']=='math500': M500[key(o['q'])]=o
def year(q): return YEAR.get(key(q))
def contaminated(q): return CONT.get(key(q),0.0)>=0.5
def group(q):
    y=year(q)
    if y in ('aime24','aime25'): return 'clean'
    if y in ('aime2022','aime2023'): return 'contam'
    return None
PERQ=json.load(open('${WRB_WORK}/perq.json'))
def recs(run,ds=None):
    return [r for r in PERQ if r['run']==run and (ds is None or r['ds']==ds)]
def p1(a): return float(np.mean(a))
def pk(a,k):
    # unbiased pass@k for one question
    n=len(a); c=int(sum(a))
    if n-c<k: return 1.0
    return 1.0-float(np.prod(1.0-k/np.arange(n-c+1,n+1)))
def boot_idx(n,B,seed):
    rng=np.random.default_rng(seed); return rng.integers(0,n,size=(B,n))
def ci(x,lo=2.5,hi=97.5):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return float(np.percentile(x,lo)),float(np.percentile(x,hi))
