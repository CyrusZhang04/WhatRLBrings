# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, re, collections, itertools
import numpy as np
G1='${WRB_WORK}'
def key(s): return re.sub(r'\s+',' ',str(s)).strip()
FL=json.load(open(G1+'/eval_flags.json'))
YEAR={key(o['q']):o['set'] for o in FL if o['set']!='math500'}
def group(q):
    y=YEAR.get(key(q))
    return 'clean' if y in ('aime24','aime25') else ('contam' if y in ('aime2022','aime2023') else None)
P=json.load(open(G1+'/perq.json'))
def recs(run): return [r for r in P if r['run']==run]

# ---------- part A: E.1 per-problem counts ----------
TARGET=[('aime24',11),('aime24',28),('aime25',2),('aime25',13),('aime25',23),('aime25',29),('aime2022',15),('aime2023',13)]
EXTRA=[('aime2023',24)]   # 9th unique-solve problem, not in user's list
A120={(r['ds'],int(r['qi'])):r for r in recs('alright120')}
A9={(r['ds'],int(r['qi'])):r for r in recs('alright9')}
V2={key(r['question']):r for r in recs('v2')}
def v2of(r): return V2[key(r['question'])]
print('='*100)
print('A. E.1  固定开头消融在"rl-replace 独解题"上的逐题结果（ablation_alright_120，06-26 18:14 那一次）')
print('   题号 = 每套内 1-based（#11 → 文件里的 qi=10）；每格 = 16 条里答对几条')
print('-'*100)
hdr=f"{'题目':<14}{'gt':>5} | {'base':>6} {'+Alright':>9} {'+Alright, I':>12} || {'v2 base':>8} {'rl_rep':>7} {'full_rl':>8} | 组别"
print(hdr)
def rowfmt(ds,num,r,extra=''):
    v=v2of(r)
    c=r['conds']
    return (f"{ds+' #'+str(num):<14}{r['gt']:>5} | {sum(c['base']):>6} {sum(c['alright']):>9} {sum(c['alright_i']):>12} || "
            f"{sum(v['conds']['base']):>8} {sum(v['conds']['rl_replace']):>7} {sum(v['conds']['full_rl']):>8} | {group(r['question'])}{extra}")
tot={'base':0,'alright':0,'alright_i':0}; solved={'base':0,'alright':0,'alright_i':0}
rows=[]
for ds,num in TARGET:
    r=A120[(ds,num-1)]; rows.append((ds,num,r)); print(rowfmt(ds,num,r))
    for c in tot:
        tot[c]+=sum(r['conds'][c]); solved[c]+= (1 if any(r['conds'][c]) else 0)
print('-'*100)
print(f"{'8 题合计':<14}{'':>5} | {tot['base']:>6} {tot['alright']:>9} {tot['alright_i']:>12}   (分母 8×16 = 128 条)")
print(f"{'解出题数':<14}{'':>5} | {str(solved['base'])+'/8':>6} {str(solved['alright'])+'/8':>9} {str(solved['alright_i'])+'/8':>12}")
print()
print('>>> 直接回答 E.1 需要的两个数：')
for c,name in (('alright','+Alright'),('alright_i','+Alright, I')):
    print(f'    {name:<12}: 128 条里答对 {tot[c]} 条（{tot[c]/128*100:.2f}%）；8 道题里解出 {solved[c]} 道')
print(f"    {'base（对照）':<12}: 128 条里答对 {tot['base']} 条（{tot['base']/128*100:.2f}%）；8 道题里解出 {solved['base']} 道")
print('    注：alright_120 的 base 列逐样本复用 v2 base，不是新生成的对照。')
print()
print('第 9 道独解题（你的清单里没有，单列备用）：')
for ds,num in EXTRA:
    r=A120[(ds,num-1)]; print('   ',rowfmt(ds,num,r))
print()
print('同日更早的 9 题版（ablation_alright，存了文本）在同样 8 道题上：')
t9={'base':0,'base_with_alright':0,'base_with_alright_i':0}; s9=dict(t9)
for ds,num in TARGET:
    r=A9.get((ds,num-1))
    if r is None: print('    缺',ds,num); continue
    for c in t9: t9[c]+=sum(r['conds'][c]); s9[c]+=(1 if any(r['conds'][c]) else 0)
for c in t9: print(f"    {c:<22} 128 条答对 {t9[c]:>3} 条；解出 {s9[c]}/8 道")

# ---------- part B: CI 口径排查 ----------
def boot(d,B=5000,seed=11):
    rng=np.random.default_rng(seed); n=len(d)
    I=rng.integers(0,n,size=(B,n)); m=np.asarray(d)[I].mean(1)
    return float(np.percentile(m,2.5)), float(np.percentile(m,97.5))
CONDS=[]
for run in ['v2','v1','v4','predictor','alright120','rule_injection','wait_topk','greedy_prefix','early_stop']:
    rs=recs(run)
    if not rs: continue
    for c in sorted(rs[0]['conds']):
        CONDS.append((run,c))
RR={key(r['question']):r['conds']['rl_replace'] for r in recs('v2')}
GSEL={'all120':lambda q:True,'clean60':lambda q:group(q)=='clean','contam60':lambda q:group(q)=='contam'}
print()
print('='*100)
print('B. "rl-replace − X" 的配对 bootstrap 区间（找 [+1.44, +11.25] 是哪一个口径）')
print('   rl-replace 固定取 v2 的 rl_replace 列；配对单位 = 题；5000 次重抽，seed 11；pass@16 = 16 条里任一对')
print('-'*100)
print(f"{'对照 X':<28}{'组':<9}{'n':>4}  {'Δpass@1':>9} {'95% CI':>20}   {'Δpass@16':>9} {'95% CI':>20}  命中")
hits=[]
for run,c in CONDS:
    for g,sel in GSEL.items():
        S=[r for r in recs(run) if sel(r['question']) and key(r['question']) in RR]
        if len(S)<10: continue
        d1=[];d16=[]
        for r in S:
            a=r['conds'][c]; b=RR[key(r['question'])]
            d1.append(np.mean(b)-np.mean(a)); d16.append(float(any(b))-float(any(a)))
        m1,m16=100*np.mean(d1),100*np.mean(d16)
        lo1,hi1=[100*x for x in boot(d1)]; lo16,hi16=[100*x for x in boot(d16)]
        hit=''
        for tag,(lo,hi) in (('pass@1',(lo1,hi1)),('pass@16',(lo16,hi16))):
            if abs(lo-1.44)<0.06 and abs(hi-11.25)<0.06: hit='<<< 命中 '+tag; hits.append((run,c,g,tag,lo,hi))
        print(f"{run+':'+c:<28}{g:<9}{len(S):>4}  {m1:>+9.2f} [{lo1:>+7.2f},{hi1:>+7.2f}]   {m16:>+9.2f} [{lo16:>+7.2f},{hi16:>+7.2f}]  {hit}")
print()
print('命中结果：', hits if hits else '没有任何口径给出 [+1.44, +11.25]')
