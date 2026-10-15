# -*- coding: utf-8 -*-
# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
"""导出 rl-replace 回答原文 + 替换位置标注（**粗体** = RL 换入的 token）。"""
import json, glob, os, sys, re
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
M='${WRB_MODELS}/Qwen3-4B-Base'
tok=AutoTokenizer.from_pretrained(M, local_files_only=True)
V2='${WRB_RUNS}/eval_rl_replace_v2'
def getrec(ds,qi):
    files=sorted(glob.glob(f'{V2}/{ds}/{ds}-n16-shard*of8.jsonl'),key=lambda x:int(x.split('shard')[1].split('of')[0]))
    k=0
    for f in files:
        for line in open(f):
            if k==qi: return json.loads(line)
            k+=1
def annotate(text,reps):
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids']; off=enc['offset_mapping']
    pos={x['pos']:x for x in reps if x['pos']<len(ids)}
    good=[p for p in pos if ids[p]==pos[p]['rl_id']]
    spans=sorted((off[p][0],off[p][1]) for p in good)
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],b)
        else: merged.append([a,b])
    out=[];prev=0
    for a,b in merged:
        seg=text[a:b]
        lead=len(seg)-len(seg.lstrip()); trail=len(seg)-len(seg.rstrip())
        out.append(text[prev:a]); out.append(seg[:lead]); out.append('**'+seg[lead:len(seg)-trail]+'**' if seg.strip() else seg); out.append(seg[len(seg)-trail:] if trail else '')
        prev=b
    out.append(text[prev:])
    rows=[]
    for p in sorted(good):
        a,b=off[p]
        rows.append(dict(pos=p,char=a,base=tok.decode([pos[p]['base_id']]),rl=tok.decode([pos[p]['rl_id']]),
                         ctx=re.sub(r'\s+',' ',text[max(0,a-60):b+60])))
    return ''.join(out), rows, len(good), len(pos)
CAND=[('aime24',14,[6,9,0,14]),('aime25',3,[9,8,1,11])]
for ds,qi,samples in CAND:
    r=getrec(ds,qi); rr=r['rl_replace']; bs=r['base']
    tag=f'{ds}_q{qi+1}'
    with open(f'raw_{tag}_base.md','w') as fh:
        fh.write(f"# {ds} #{qi+1}  gt={r['gt']}  base 原文（对 {sum(bs['accs'])}/16）\n\n问题：{r['question']}\n\n")
        for j,(t,a,L) in enumerate(zip(bs['texts'],bs['accs'],bs['lengths'])):
            fh.write(f"\n## base 样本 {j}（{'对' if a else '错'}，{L} token）\n\n{t}\n")
    for j in samples:
        t=rr['texts'][j]; reps=rr['replacements'][j]
        ann,rows,ok,tot=annotate(t,reps)
        with open(f'annot_{tag}_s{j}.md','w') as fh:
            fh.write(f"# {ds} #{qi+1} gt={r['gt']} · rl-replace 样本 {j}（{'对' if rr['accs'][j] else '错'}，"
                     f"{rr['lengths'][j]} token，替换 {len(reps)} 处 = {rr['replace_rates'][j]*100:.2f}%，对齐 {ok}/{tot}）\n\n")
            fh.write(f"问题：{r['question']}\n\n**粗体 = RL 换入的 token**（其余由 base 自己采样）\n\n---\n\n{ann}\n")
        with open(f'reps_{tag}_s{j}.tsv','w') as fh:
            fh.write('pos\tchar\tbase_token\trl_token\tcontext\n')
            for x in rows: fh.write(f"{x['pos']}\t{x['char']}\t{x['base']!r}\t{x['rl']!r}\t{x['ctx']}\n")
        waits=[x for x in rows if x['rl'].strip()=='Wait']
        print(f'{tag} s{j}: 对={int(rr["accs"][j])} {rr["lengths"][j]}tok 替换{len(reps)} 对齐{ok}/{tot} 插入Wait {len(waits)} 处')
        for x in waits: print(f"    pos={x['pos']} char={x['char']}  base 原本要写 {x['base']!r} → {x['rl']!r}  …{x['ctx']}…")
print()
print('文件：annot_*.md（标注原文）、reps_*.tsv（逐处替换）、raw_*_base.md（base 原文）')
