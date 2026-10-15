# -*- coding: utf-8 -*-
# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, re, glob, collections
import numpy as np
AO='${WRB_RUNS}'
G1='${WRB_WORK}'
def nk(s): return re.sub(r'\s+',' ',str(s)).strip()
FL=json.load(open(G1+'/eval_flags.json'))
YEAR={nk(o['q']):o['set'] for o in FL if o['set']!='math500'}
def grp(q):
    y=YEAR.get(nk(q)); return 'clean' if y in ('aime24','aime25') else ('contam' if y in ('aime2022','aime2023') else 'unknown')
BOX=re.compile(r'\\boxed')
def last_boxed(t):
    i=t.rfind('\\boxed')
    if i<0: return None
    j=t.find('{',i)
    if j<0: return None
    d=0
    for k in range(j,len(t)):
        if t[k]=='{': d+=1
        elif t[k]=='}':
            d-=1
            if d==0: return t[j+1:k]
    return None
def norm_ans(a):
    if a is None: return None
    for s in ('\\left','\\right','\\!','\\,','\;','$','{','}',' '): a=a.replace(s,'')
    a=re.sub(r'^\\text(bf|rm|it)?','',a).rstrip('.').strip()
    return a or None

rows=collections.defaultdict(list)   # cond -> list of records
for ds in ['aime24','aime25','aime2022','aime2023']:
    files=sorted(glob.glob(f'{AO}/eval_rl_replace_v2/{ds}/{ds}-n16-shard*of8.jsonl'),
                 key=lambda x:int(x.split('shard')[1].split('of')[0]))
    qi=0
    for f in files:
        for line in open(f):
            r=json.loads(line); g=grp(r['question']); pid=f'{ds}#{qi+1}'
            for c in ('base','rl_replace','full_rl'):
                if c not in r: continue
                b=r[c]; lens=b.get('lengths') or [None]*len(b['texts']); rrs=b.get('replace_rates') or [None]*len(b['texts'])
                for j,(t,a,L,rr) in enumerate(zip(b['texts'],b['accs'],lens,rrs)):
                    rows[c].append(dict(problem=pid,group=g,sample=j,correct=bool(a),n_tokens=L,
                                        has_boxed=bool(BOX.search(t)),answer=norm_ans(last_boxed(t)),
                                        opening=re.sub(r'\s+',' ',t.strip())[:48],replace_rate=rr))
            qi+=1
for line in open(f'{AO}/ablation_alright/ablation_alright.jsonl'):
    r=json.loads(line); g=grp(r['question']); pid=f"{r['ds']}#{int(r['qi'])+1}"
    for c,name in (('base','alright9_base'),('base_with_alright','alright'),('base_with_alright_i','alright_i')):
        b=r[c]; lens=b.get('lengths') or [None]*len(b['texts'])
        for j,(t,a,L) in enumerate(zip(b['texts'],b['accs'],lens)):
            rows[name].append(dict(problem=pid,group=g,sample=j,correct=bool(a),n_tokens=L,
                                   has_boxed=bool(BOX.search(t)),answer=norm_ans(last_boxed(t)),
                                   opening=re.sub(r'\s+',' ',t.strip())[:48],replace_rate=None))
for c,rs in rows.items():
    with open(f'cond_{c}.jsonl','w') as fh:
        for x in rs: fh.write(json.dumps(x,ensure_ascii=False)+'\n')
print('导出：'+', '.join(f'cond_{c}.jsonl({len(rs)} 行)' for c,rs in rows.items()))

def perq(c,sel=None,answered_only=False):
    d=collections.defaultdict(list)
    for x in rows[c]:
        if sel and x['group']!=sel: continue
        if answered_only and not x['has_boxed']: continue
        d[x['problem']].append(x['correct'])
    return {k:np.mean(v) for k,v in d.items() if v}
def boot_d(A,B,seed=11,n=5000):
    ks=sorted(set(A)&set(B)); d=np.array([A[k]-B[k] for k in ks])
    rng=np.random.default_rng(seed); I=rng.integers(0,len(d),size=(n,len(d)))
    m=d[I].mean(1)
    return 100*d.mean(),100*np.percentile(m,2.5),100*np.percentile(m,97.5),len(ks)
print()
print('='*100)
print('F. "增益不是修好了输出格式" —— 只看有 \\boxed 答案的回答，pass@1 的差还在不在')
print('   口径：题级宏平均（每题先在它自己"有答案"的样本里算正确率，再按题平均），配对 bootstrap 5000 次 seed 11')
for sel,lab in ((None,'全部 120 题'),('clean','干净组 60 题'),('contam','污染组 60 题')):
    print('-'*100); print(f'【{lab}】')
    for ao in (False,True):
        tag='只看有答案' if ao else '全样本  '
        vals={c:perq(c,sel,ao) for c in ('base','rl_replace','full_rl')}
        line=' '.join(f'{c} {100*np.mean(list(vals[c].values())):.2f}%' for c in vals)
        print(f'  {tag}  {line}')
        for a,b in (('rl_replace','base'),('rl_replace','full_rl'),('full_rl','base')):
            m,lo,hi,n=boot_d(vals[a],vals[b])
            print(f'      {a} − {b}: {m:+.2f}pp  95%CI [{lo:+.2f},{hi:+.2f}]  (n={n} 题)')
print()
print('='*100)
print('G. 替换率与难度的关系（解释"池化看答对替换率更低、题内配对却更高"）')
q=collections.defaultdict(lambda: dict(rr=[],ok=[]))
for x in rows['rl_replace']:
    q[x['problem']]['rr'].append(x['replace_rate']); q[x['problem']]['ok'].append(x['correct'])
R=np.array([np.mean(v['rr']) for v in q.values()]); A=np.array([np.mean(v['ok']) for v in q.values()])
from scipy.stats import pearsonr, spearmanr
print(f'  题级：平均替换率 vs 题级 pass@1  Pearson r={pearsonr(R,A)[0]:+.3f} (p={pearsonr(R,A)[1]:.3g})，'
      f'Spearman ρ={spearmanr(R,A)[0]:+.3f} (p={spearmanr(R,A)[1]:.3g})，n={len(R)} 题')
qa=[k for k,v in q.items() if np.mean(v['ok'])>0]
print(f'  有解出的题 {len(qa)} 道：平均替换率 {np.mean([np.mean(q[k]["rr"]) for k in qa])*100:.2f}%；'
      f'全错的题 {len(q)-len(qa)} 道：{np.mean([np.mean(q[k]["rr"]) for k in q if k not in qa])*100:.2f}%')
bins=[(0,0.001),(0.001,0.0625),(0.0625,0.25),(0.25,1.01)]
print('  按题级 pass@1 分箱的平均替换率：')
for lo,hi in bins:
    sel=[k for k,v in q.items() if lo<=np.mean(v['ok'])<hi]
    if sel: print(f'    pass@1 ∈ [{lo*100:.1f}%,{hi*100:.1f}%)  {len(sel):>3} 题  平均替换率 {np.mean([np.mean(q[k]["rr"]) for k in sel])*100:.2f}%')
