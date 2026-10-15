# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, glob, os
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
M='${WRB_MODELS}/Qwen3-4B-Base'
tok=AutoTokenizer.from_pretrained(M, local_files_only=True)
print('tokenizer ok:', type(tok).__name__, 'fast=',tok.is_fast)
for i in (71703,71486,11,220,358,14190,13824):
    print('  id',i,'->',repr(tok.decode([i])))
V2='${WRB_RUNS}/eval_rl_replace_v2'
TARGETS={('aime24',14):'aime24/aime24-n16-shard3of8.jsonl',('aime25',3):'aime25/aime25-n16-shard0of8.jsonl'}
def getrec(ds,qi):
    files=sorted(glob.glob(f'{V2}/{ds}/{ds}-n16-shard*of8.jsonl'),key=lambda x:int(x.split('shard')[1].split('of')[0]))
    k=0
    for f in files:
        for line in open(f):
            if k==qi: return json.loads(line)
            k+=1
for (ds,qi) in TARGETS:
    r=getrec(ds,qi); rr=r['rl_replace']
    print(f'\n=== {ds} #{qi+1} gt={r["gt"]}  rl_replace 对 {sum(rr["accs"])}/16')
    for j,(t,a,L,reps) in enumerate(zip(rr['texts'],rr['accs'],rr['lengths'],rr['replacements'])):
        ids=tok(t,add_special_tokens=False)['input_ids']
        ok=sum(1 for x in reps if x['pos']<len(ids) and ids[x['pos']]==x['rl_id'])
        waits=[x for x in reps if tok.decode([x['rl_id']]).strip()=='Wait']
        print(f'  样本{j:>2} 对={int(a)} len={L} tok={len(ids)} 替换={len(reps)} 对齐={ok}/{len(reps)}'
              f' 插入Wait={len(waits)} 文本里Wait={t.count("Wait")}')
