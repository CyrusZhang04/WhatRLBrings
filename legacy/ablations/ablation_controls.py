# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,random,math
B='${WRB_RUNS}/'
def load(pat):
    rows=[]
    for f in sorted(glob.glob(B+pat)): rows+=[json.loads(l) for l in open(f)]
    return {(r['ds'],r['qi']):r for r in rows}
R=load('ablation_rule_injection/rule_injection_shard*of8.jsonl')
E=load('ablation_early_stop/early_stop_shard*of8.jsonl')
A=load('ablation_alright_120/ablation_alright_shard*of8.jsonl')
G=load('ablation_greedy_prefix/greedy_prefix_shard*of8.jsonl')
W=load('ablation_wait_topk/wait_topk_shard*of8.jsonl')
random.seed(1)
def cmp(D1,k1,D2,k2,label):
    keys=sorted(D1)
    d=[sum(D1[k][k1])/16-sum(D2[k][k2])/16 for k in keys]
    d16=[int(any(D1[k][k1]))-int(any(D2[k][k2])) for k in keys]
    m=sum(d)/len(d); m16=sum(d16)/len(d16)
    bs=[];bs16=[]
    for _ in range(5000):
        idx=[random.randrange(len(d)) for _ in d]
        bs.append(sum(d[i] for i in idx)/len(d)); bs16.append(sum(d16[i] for i in idx)/len(d))
    bs.sort(); bs16.sort()
    print(f'{label:45s} d_pass1={m*100:+.2f}pp [{bs[125]*100:+.2f},{bs[4875]*100:+.2f}]  d_pass16={m16*100:+.2f}pp [{bs16[125]*100:+.2f},{bs16[4875]*100:+.2f}]')
cmp(R,'pronoun05_accs',R,'soft2_accs','pron+0.5 vs Re+2 (in-run control)')
cmp(R,'pronoun1_accs',R,'soft2_accs','pron+1.0 vs Re+2 (in-run control)')
cmp(E,'early_both_accs',E,'early_low_accs','early_both vs early_low (in-run control)')
cmp(A,'alright_accs',A,'alright_i_accs','Alright vs Alright,I')
cmp(G,'al_greedy_accs',G,'greedy_accs','Alright+greedy vs greedy')
cmp(A,'alright_accs',G,'al_greedy_accs','Alright(120run) vs Alright+greedy')
cmp(W,'topk_accs',R,'soft2_accs','Wait top-5 vs Re+2(fresh-base proxy)')
cmp(A,'alright_accs',A,'base_accs','Alright vs v2 base')
cmp(A,'alright_i_accs',A,'base_accs','Alright,I vs v2 base')
cmp(R,'soft2_accs',E,'early_low_accs','Re+2 vs early_low (two fresh base proxies)')
