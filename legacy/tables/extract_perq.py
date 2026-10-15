# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
# Extract per-question accs from eval jsonl files into a compact json. Read-only on ROOT.
import json, glob, os, re
from multiprocessing import Pool
AO='${WRB_RUNS}/'
def shard_of(f):
    m=re.search(r'shard(\d+)of(\d+)',f); return (int(m.group(1)),int(m.group(2))) if m else (0,1)
def job(args):
    run,f,ds_dir=args
    recs=[]
    with open(f) as fh:
        for i,l in enumerate(fh):
            r=json.loads(l)
            rec={'run':run,'file':f.replace(AO,''),'i':i,'shard':shard_of(f)[0],'question':r.get('question'),'gt':r.get('gt'),'ds':r.get('ds',ds_dir),'qi':r.get('qi')}
            conds={}
            for k,v in r.items():
                if isinstance(v,dict) and 'accs' in v:
                    conds[k]=[int(bool(a)) for a in v['accs']]
                    if 'replace_rate' in v: rec.setdefault('rr',{})[k]=v['replace_rate']
                    if 'lengths' in v: rec.setdefault('len',{})[k]=v['lengths']
                elif k.endswith('_accs') and isinstance(v,list):
                    conds[k[:-5]]=[int(bool(a)) for a in v]
                elif k=='accs' and isinstance(v,list):
                    conds[r.get('condition','accs')]=[int(bool(a)) for a in v]
            for k in ('gate_preds','answer','dataset','gate_pass16','gate_maj16','condition','replace_rate'):
                if k in r: rec[k]=r[k]
            rec['conds']=conds
            recs.append(rec)
    return recs
tasks=[]
def add(run,pattern,dsfrom='dir'):
    for f in sorted(glob.glob(AO+pattern)):
        ds=f.split('/')[-2]
        tasks.append((run,f,ds))
add('v1','eval_rl_replace/aime2*/*-n16-shard*.jsonl')
add('v2','eval_rl_replace_v2/*/*-n16-shard*of8.jsonl')
add('v4','eval_rl_replace_v4/*/*-n256-shard*of8.jsonl')
add('predictor','eval_predictor/*/*-predictor-n16-shard*of8.jsonl')
add('alright9','ablation_alright/ablation_alright.jsonl')
add('alright120','ablation_alright_120/ablation_alright_shard*of8.jsonl')
add('rule_injection','ablation_rule_injection/rule_injection_shard*of8.jsonl')
add('wait_topk','ablation_wait_topk/wait_topk_shard*of8.jsonl')
add('greedy_prefix','ablation_greedy_prefix/greedy_prefix_shard*of8.jsonl')
add('early_stop','ablation_early_stop/early_stop_shard*of8.jsonl')
add('gate_mcmc','gate_mcmc_aime/gate_mcmc_shard*of8.jsonl')
add('a2','eval_a2/*/*-a2-*-n*-shard*of8.jsonl')
add('cf','eval_counterfactual/*/*-cf-*-n32-shard*of8.jsonl')
add('m500_rlr','eval_rl_replace_math500/math500_500/*-n16-shard*of8.jsonl')
for tag in ['opd_step200','opd_full_kl_step200','opd_context_kl_step200','opd_revkl_step200']:
    add('opd:'+tag,'eval_opd/%s/*/*-n16-shard*of8.jsonl'%tag)
if __name__=='__main__':
    print('files',len(tasks),flush=True)
    out=[]
    with Pool(10) as p:
        for recs in p.imap_unordered(job,tasks): out.extend(recs)
    # v4 lengths are large; drop lengths except v2/m500/opd (keep for anomaly checks if needed)
    for r in out:
        if r['run'] in ('v4','v1','predictor'): r.pop('len',None)
    json.dump(out,open('${WRB_WORK}/perq.json','w'))
    import collections
    c=collections.Counter((r['run'],r['ds']) for r in out)
    for k in sorted(c): print(k,c[k])
