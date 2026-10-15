# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,math,random
B='${WRB_RUNS}/'
def load(pat):
    rows=[]
    for f in sorted(glob.glob(B+pat)):
        rows+=[json.loads(l) for l in open(f)]
    rows.sort(key=lambda r:(r['ds'],r['qi']))
    return rows
v2=json.load(open('${WRB_WORK}/perq_ablation.json'))
sets={
 'alright120':('ablation_alright_120/ablation_alright_shard*of8.jsonl',['alright_accs','alright_i_accs']),
 'rule':('ablation_rule_injection/rule_injection_shard*of8.jsonl',['soft2_accs','soft4_accs','pronoun05_accs','pronoun1_accs']),
 'waittopk':('ablation_wait_topk/wait_topk_shard*of8.jsonl',['alright_accs','topk_accs','alright_topk_accs']),
 'greedy':('ablation_greedy_prefix/greedy_prefix_shard*of8.jsonl',['greedy_accs','soft_accs','al_greedy_accs']),
 'early':('ablation_early_stop/early_stop_shard*of8.jsonl',['early_low_accs','early_both_accs']),
}
random.seed(0)
def boot(diffs,n=5000):
    m=len(diffs); res=[]
    for _ in range(n):
        s=[diffs[random.randrange(m)] for _ in range(m)]
        res.append(sum(s)/m)
    res.sort(); return res[int(0.025*n)], res[int(0.975*n)]
for name,(pat,conds) in sets.items():
    rows=load(pat)
    print('=====',name,len(rows))
    base=[sum(r['base_accs'])/16 for r in rows]
    rr=[v2[f"{r['ds']}|{r['qi']}"]['rl_replace']/16 for r in rows]
    for c in ['base_accs']+conds:
        acc=[sum(r[c])/len(r[c]) for r in rows]
        tot=sum(sum(r[c]) for r in rows); n=sum(len(r[c]) for r in rows)
        solved=sum(1 for r in rows if any(r[c]))
        d=[a-b for a,b in zip(acc,base)]
        md=sum(d)/len(d); sd=math.sqrt(sum((x-md)**2 for x in d)/(len(d)-1)); se=sd/math.sqrt(len(d))
        lo,hi=boot(d) if c!='base_accs' else (0,0)
        # pass16 gained/lost
        gain=sum(1 for r in rows if any(r[c]) and not any(r['base_accs']))
        lost=sum(1 for r in rows if (not any(r[c])) and any(r['base_accs']))
        print(f"{c:18s} correct={tot:4d}/{n} pass1={tot/n*100:6.2f}% pass16={solved/len(rows)*100:6.2f}% ({solved}/120)  d_pass1_vs_base={md*100:+.2f}pp se={se*100:.2f} boot95=[{lo*100:+.2f},{hi*100:+.2f}]  pass16 gained={gain} lost={lost}")
