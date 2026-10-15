# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob
B='${WRB_RUNS}/'
v2=json.load(open('${WRB_WORK}/perq_ablation.json'))
def load(pat):
    rows=[]
    for f in sorted(glob.glob(B+pat)): rows+=[json.loads(l) for l in open(f)]
    return rows
DS=['aime24','aime25','aime2022','aime2023']
def stat(rows,key):
    out={}
    for ds in DS+['overall']:
        rr=[r for r in rows if ds=='overall' or r['ds']==ds]
        p1=sum(sum(r[key]) for r in rr)/sum(len(r[key]) for r in rr)
        p16=sum(1 for r in rr if any(r[key]))/len(rr)
        out[ds]=(p1*100,p16*100)
    return out
def v2stat(c):
    out={}
    for ds in DS+['overall']:
        vv=[v for k,v in v2.items() if ds=='overall' or k.split('|')[0]==ds]
        out[ds]=(sum(v[c] for v in vv)/(16*len(vv))*100, sum(1 for v in vv if v[c]>0)/len(vv)*100)
    return out
groups=[
 ('ablation_alright_120',load('ablation_alright_120/ablation_alright_shard*of8.jsonl'),[('A_base (v2复用)','base_accs'),('B +Alright','alright_accs'),('C +Alright, I','alright_i_accs')]),
 ('ablation_rule_injection',load('ablation_rule_injection/rule_injection_shard*of8.jsonl'),[('A_base (v2复用)','base_accs'),('B Re +2','soft2_accs'),('C Re +4','soft4_accs'),('D pron +0.5','pronoun05_accs'),('E pron +1.0','pronoun1_accs')]),
 ('ablation_wait_topk',load('ablation_wait_topk/wait_topk_shard*of8.jsonl'),[('A_base (v2复用)','base_accs'),('B_alright (复用)','alright_accs'),('E Wait top-5','topk_accs'),('F Alright+Wait top-5','alright_topk_accs')]),
 ('ablation_greedy_prefix',load('ablation_greedy_prefix/greedy_prefix_shard*of8.jsonl'),[('A_base (v2复用)','base_accs'),('B greedy100 (T=0)','greedy_accs'),('C soft100 (T=0.3)','soft_accs'),('D Alright+greedy100','al_greedy_accs')]),
 ('ablation_early_stop',load('ablation_early_stop/early_stop_shard*of8.jsonl'),[('A_base (v2复用)','base_accs'),('B 低熵终止 (<0.05)','early_low_accs'),('C 双侧终止 (<0.05 或 >0.3)','early_both_accs')]),
]
hdr='| 条件 | aime24 p@1 / p@16 | aime25 p@1 / p@16 | aime2022 p@1 / p@16 | aime2023 p@1 / p@16 | **overall p@1 / p@16** |\n|---|---|---|---|---|---|'
def row(name,s):
    return '| '+name+' | '+' | '.join(f'{s[d][0]:.2f} / {s[d][1]:.2f}' for d in DS)+f" | **{s['overall'][0]:.2f} / {s['overall'][1]:.2f}** |"
print('### v2 reference'); print(hdr)
for c,n in [('base','base (v2)'),('rl_replace','rl_replace (v2)'),('full_rl','full_rl (v2)')]:
    print(row(n,v2stat(c)))
for g,rows,conds in groups:
    print('\n### '+g); print(hdr)
    for n,k in conds: print(row(n,stat(rows,k)))
