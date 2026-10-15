# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,collections,re
B='${WRB_RUNS}/eval_rl_replace_v2/'
c_rr=collections.Counter(); c_full=collections.Counter(); c_base=collections.Counter()
nw=collections.Counter()
for f in glob.glob(B+'*/*-n16-shard*of8.jsonl'):
    for line in open(f):
        r=json.loads(line)
        for cond,cnt in [('rl_replace',c_rr),('full_rl',c_full),('base',c_base)]:
            for txt in r[cond]['texts']:
                for m in re.finditer(r'(?<=[ \n])Re[a-z\-]*', txt):
                    cnt[m.group(0)]+=1
                nw[(cond,'Wait')]+=len(re.findall(r'\bWait\b',txt))
                nw[(cond,'n')]+=1
for name,c in [('rl_replace',c_rr),('full_rl',c_full),('base',c_base)]:
    print(name, sum(c.values()), c.most_common(12))
for cond in ['base','rl_replace','full_rl']:
    print(cond,'mean Wait per text %.2f'%(nw[(cond,'Wait')]/nw[(cond,'n')]))
