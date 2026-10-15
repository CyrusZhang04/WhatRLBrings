# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
from lib import *
import numpy as np, ast, collections
R=recs('gate_mcmc')
v2={key(r['question']):r for r in recs('v2')}
def eq(p,a):
    if p is None: return False
    try: return int(str(p).strip())==int(str(a).strip())
    except: 
        try: return abs(float(str(p))-float(str(a)))<1e-9
        except: return str(p).strip()==str(a).strip()
rows=[]
for r in R:
    preds=r['gate_preds']
    if isinstance(preds,str): preds=ast.literal_eval(preds)
    acc=[eq(p,r['answer']) for p in preds]
    cnt=collections.Counter(str(int(p)) if p is not None and str(p).strip().lstrip('-').isdigit() else p for p in preds if p is not None)
    plur=False
    if cnt:
        top=cnt.most_common(1)[0][0]; plur=eq(top,r['answer'])
    rows.append(dict(q=r['question'],ds=r['dataset'],p16=int(r['gate_pass16']),maj=int(r['gate_maj16']),mean=np.mean(acc),plur=plur,none=sum(p is None for p in preds),anyacc=any(acc)))
print('consistency: gate_pass16 == any(preds==answer):',sum(1 for x in rows if bool(x['p16'])==x['anyacc']),'/',len(rows),' None preds:',sum(x['none'] for x in rows))
for g,sel in (('all120',lambda x:True),('clean60',lambda x:group(x['q'])=='clean'),('contam60',lambda x:group(x['q'])=='contam'),('aime24',lambda x:x['ds']=='aime24'),('aime25',lambda x:x['ds']=='aime25'),('aime2022',lambda x:x['ds']=='aime2022'),('aime2023',lambda x:x['ds']=='aime2023')):
    S=[x for x in rows if sel(x)]; n=len(S)
    vb=[v2[key(x['q'])] for x in S]
    b1=np.mean([p1(r['conds']['base']) for r in vb]); bk=np.mean([any(r['conds']['base']) for r in vb])
    print('%-9s n=%3d gate pass@16 %.2f | maj16(strict) %.2f | plurality maj %.2f | mean@16 %.2f || v2 base p@1 %.2f p@16 %.2f'%(g,n,100*np.mean([x['p16'] for x in S]),100*np.mean([x['maj'] for x in S]),100*np.mean([x['plur'] for x in S]),100*np.mean([x['mean'] for x in S]),100*b1,100*bk))
