# -*- coding: utf-8 -*-
# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
"""Response Statistics for the rl-replace appendix.
Reads raw jsonl (texts) and reports: no-boxed rate, answered-only pass@1,
length distribution + cap hits, per-question diversity, replace-rate distribution."""
import json, re, glob, statistics as st, collections
import numpy as np

AO='${WRB_RUNS}'
G1='${WRB_WORK}'
def nk(s): return re.sub(r'\s+',' ',str(s)).strip()
FL=json.load(open(G1+'/eval_flags.json'))
YEAR={nk(o['q']):o['set'] for o in FL if o['set']!='math500'}
def grp(q):
    y=YEAR.get(nk(q))
    return 'clean' if y in ('aime24','aime25') else ('contam' if y in ('aime2022','aime2023') else 'unknown')

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
    return None          # 截断在 \boxed{ 之后
def norm_ans(a):
    if a is None: return None
    a=a.replace('\\left','').replace('\\right','').replace('\\!','').replace('\\,','').replace('\;','')
    a=a.replace('$','').replace('{','').replace('}','').replace(' ','').rstrip('.').strip()
    a=re.sub(r'^\\text(bf|rm|it)?','',a)
    return a if a else None
def opening(t,n=48):
    return re.sub(r'\s+',' ',t.strip())[:n]

class Cond:
    def __init__(s,name): s.name=name; s.q=collections.defaultdict(lambda: dict(acc=[],box=[],ans=[],op=[],ln=[],rr=[]))
    def add(s,qid,g,**kw):
        d=s.q[(qid,g)]
        for k,v in kw.items(): d[k].append(v)

def flat(cond,sel=None,f='acc'):
    return [x for (qid,g),d in cond.q.items() if sel in (None,g) for x in d[f]]

def report(cond,sel,cap_list=(2048,4096,8192)):
    acc=np.array(flat(cond,sel,'acc'),dtype=float)
    box=np.array(flat(cond,sel,'box'),dtype=bool)
    ln=np.array([x for x in flat(cond,sel,'ln') if x is not None],dtype=float) if flat(cond,sel,'ln') else np.array([])
    n=len(acc); nb=int((~box).sum())
    out=[]
    out.append(f"  样本数 {n}；无 \\boxed 答案 {nb} 条（{nb/n*100:.2f}%）")
    out.append(f"  pass@1 全样本 {acc.mean()*100:.2f}%；只看有答案的 {acc[box].mean()*100:.2f}%（分母 {int(box.sum())}）"
               + ("" if box.all() else f"；无答案样本里答对 {int(acc[~box].sum())} 条"))
    if len(ln):
        out.append(f"  长度(token) 均值 {ln.mean():.0f} 中位数 {np.median(ln):.0f} p90 {np.percentile(ln,90):.0f} 最大 {ln.max():.0f}")
        parts=[]
        for c in cap_list:
            parts.append(f">= {c}: {(ln>=c).mean()*100:.2f}%  == {c}: {(ln==c).mean()*100:.2f}%")
        out.append("  撞上限比例  "+" | ".join(parts))
    # diversity per question
    dop=[];dans=[];dans_all=[]
    for (qid,g),d in cond.q.items():
        if sel not in (None,g): continue
        dop.append(len(set(d['op'])))
        a=[x for x in d['ans'] if x]
        dans.append(len(set(a))); dans_all.append(len(set(a))+ (1 if len(a)<len(d['ans']) else 0))
    if dop:
        out.append(f"  每题不同开头数 均值 {np.mean(dop):.2f}（中位 {np.median(dop):.1f}，范围 {min(dop)}–{max(dop)}）")
        out.append(f"  每题不同答案数 均值 {np.mean(dans):.2f}（只数非空答案；中位 {np.median(dans):.1f}，范围 {min(dans)}–{max(dans)}）")
    rr=[x for x in flat(cond,sel,'rr') if x is not None]
    if rr:
        rr=np.array(rr); q=np.percentile(rr,[0,25,50,75,100])
        out.append(f"  替换率 均值 {rr.mean()*100:.2f}%  min {q[0]*100:.2f} / Q1 {q[1]*100:.2f} / 中位 {q[2]*100:.2f} / Q3 {q[3]*100:.2f} / max {q[4]*100:.2f}（%）")
        ok=np.array(flat(cond,sel,'acc'),dtype=bool)
        if ok.any() and (~ok).any():
            a,b=rr[ok],rr[~ok]
            out.append(f"  替换率 答对样本 均值 {a.mean()*100:.2f}%（中位 {np.median(a)*100:.2f}，n={len(a)}）vs 答错 {b.mean()*100:.2f}%（中位 {np.median(b)*100:.2f}，n={len(b)}）；差 {(a.mean()-b.mean())*100:+.2f}pp")
            # 题内配对：同题内答对均值 − 答错均值
            dif=[]
            for (qid,g),d in cond.q.items():
                if sel not in (None,g): continue
                r=np.array(d['rr'],dtype=float); o=np.array(d['acc'],dtype=bool)
                if o.any() and (~o).any(): dif.append(r[o].mean()-r[~o].mean())
            if dif:
                rng=np.random.default_rng(11); D=np.array(dif); I=rng.integers(0,len(D),size=(5000,len(D)))
                m=D[I].mean(1)
                out.append(f"  替换率 题内配对（答对−答错）{D.mean()*100:+.3f}pp  95%CI [{np.percentile(m,2.5)*100:+.3f},{np.percentile(m,97.5)*100:+.3f}]（{len(D)} 道题有两类样本）")
    return out

def top_openings(cond,sel,k=6):
    c=collections.Counter(x for (qid,g),d in cond.q.items() if sel in (None,g) for x in d['op'])
    return c.most_common(k)

# ---------------- v2: base / rl_replace / full_rl ----------------
CONDS={c:Cond(c) for c in ('base','rl_replace','full_rl')}
for ds in ['aime24','aime25','aime2022','aime2023']:
    files=sorted(glob.glob(f'{AO}/eval_rl_replace_v2/{ds}/{ds}-n16-shard*of8.jsonl'),
                 key=lambda x:int(x.split('shard')[1].split('of')[0]))
    qi=0
    for f in files:
        for line in open(f):
            r=json.loads(line); g=grp(r['question']); qid=f'{ds}|{qi}'
            for c in CONDS:
                if c not in r: continue
                blk=r[c]; texts=blk['texts']; accs=blk['accs']; lens=blk.get('lengths') or [None]*len(texts)
                rrs=blk.get('replace_rates') or [None]*len(texts)
                for t,a,L,rr in zip(texts,accs,lens,rrs):
                    CONDS[c].add(qid,g,acc=bool(a),box=bool(BOX.search(t)),ans=norm_ans(last_boxed(t)),
                                 op=opening(t),ln=L,rr=rr)
            qi+=1
print('='*100)
print('C. Response Statistics（v2 run，4 套 AIME 120 题 × n=16，T=0.6，V1 引擎）')
for sel,label in ((None,'全部 120 题'),('clean','干净组 AIME24+25（60 题）'),('contam','污染组 AIME22+23（60 题）')):
    print('-'*100); print(f'【{label}】')
    for c in ('base','rl_replace','full_rl'):
        print(f' {c}:')
        for l in report(CONDS[c],sel): print(l)
print('-'*100)
print('最常见开头（全部 120 题，前 6）：')
for c in ('base','rl_replace','full_rl'):
    print(f' {c}:')
    for s,n in top_openings(CONDS[c],None): print(f'    {n:>5}  {s}')

# ---------------- ablation_alright（9 题版，有文本） ----------------
print()
print('='*100)
print('D. +Alright / +Alright, I 的文本统计 —— 只有 9 题版（ablation_alright）存了文本')
A9={c:Cond(c) for c in ('base','base_with_alright','base_with_alright_i')}
for line in open(f'{AO}/ablation_alright/ablation_alright.jsonl'):
    r=json.loads(line); g=grp(r['question']); qid=f"{r['ds']}|{r['qi']}"
    for c in A9:
        blk=r[c]; lens=blk.get('lengths') or [None]*len(blk['texts'])
        for t,a,L in zip(blk['texts'],blk['accs'],lens):
            A9[c].add(qid,g,acc=bool(a),box=bool(BOX.search(t)),ans=norm_ans(last_boxed(t)),op=opening(t),ln=L,rr=None)
for c in A9:
    print(f' {c}:')
    for l in report(A9[c],None,cap_list=(2048,4096,8192)): print(l)
    for s,n in top_openings(A9[c],None,4): print(f'    {n:>5}  {s}')

# ---------------- greedy 对照：只有前缀 ----------------
print()
print('='*100)
print('E. greedy 对照（ablation_greedy_prefix，120 题）—— 文件只存了 100-token 前缀和正确性，没有完整回答')
rows=[json.loads(l) for f in sorted(glob.glob(f'{AO}/ablation_greedy_prefix/greedy_prefix_shard*of8.jsonl')) for l in open(f)]
for fld,name in (('greedy_prefixes','[B] 贪心前缀 100 token'),('al_greedy_pfx','[D] 强制 Alright + 贪心前缀')):
    dop=[len(set(re.sub(r'\s+',' ',x.strip())[:48] for x in r[fld])) for r in rows if fld in r]
    dfull=[len(set(r[fld])) for r in rows if fld in r]
    print(f'  {name}: 每题不同开头数（前 48 字符）均值 {np.mean(dop):.2f}，整条前缀去重均值 {np.mean(dfull):.2f}（共 {len(dop)} 题，每题 16 条）')
print('  → 该条件无法给出"无 \\boxed 比例 / 只看有答案的 pass@1 / 长度 / 不同答案数"；要补这些必须重跑并存文本。')
