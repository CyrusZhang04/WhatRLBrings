# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,sys
# stream: only keep accs & lengths, and question idx by order within shard
out={}
for ds in ['aime24','aime25','aime2022','aime2023']:
    files=sorted(glob.glob(f'{ds}/{ds}-n16-shard*of8.jsonl'), key=lambda x:int(x.split('shard')[1].split('of')[0]))
    qi=0
    for f in files:
        for line in open(f):
            r=json.loads(line)
            rec={'gt':r['gt'],'q':r['question'][:60]}
            for c in ['base','rl_replace','full_rl']:
                if c in r:
                    rec[c]=sum(r[c]['accs'])
                    rec[c+'_n']=len(r[c]['accs'])
            if 'rl_replace' in r:
                rec['rr']=r['rl_replace'].get('replace_rate')
                reps=r['rl_replace'].get('replacements')
                rec['first_reps']=[ (x[:3] if x else []) for x in reps[:2]]
            out[(ds,qi)]=rec
            qi+=1
json.dump({f'{k[0]}|{k[1]}':v for k,v in out.items()}, open('${WRB_WORK}/perq_ablation.json','w'))
print(len(out))
