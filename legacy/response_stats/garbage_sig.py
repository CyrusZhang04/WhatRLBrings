# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,re,glob,os,collections
sys.path.insert(0,'${WRB_WORK}')
from anom_lib import flags
FOREIGN=re.compile(r'[֐-׿؀-ۿ฀-๿ᄀ-ᇿ㄰-㆏가-힯ሀ-፿ऀ-ॿ]')
ECHO=re.compile(r'(^|\n)\s*(user|assistant|system)\s*\n')
def scan(files, cols, maxtok=16384):
    res=collections.defaultdict(lambda: collections.Counter())
    for fn in files:
        for i,line in enumerate(open(fn)):
            d=json.loads(line)
            for c in cols:
                if c not in d or 'texts' not in d[c]: continue
                o=d[c]
                for t,l,a in zip(o['texts'],o['lengths'],o['accs']):
                    R=res[c]; R['n']+=1
                    f=flags(t,l,maxtok)
                    R['anom']+=any(f); R['foreign']+=bool(FOREIGN.search(t)); R['echo']+=bool(ECHO.search(t)); R['trunc']+=(l==maxtok)
                    R['acc']+=bool(a)
    return res
AO='${WRB_RUNS}'
runs={
 'v2(V1) 4xAIME': sorted(glob.glob(AO+'/eval_rl_replace_v2/*/*.jsonl')),
 'math500 rlr(V0)': sorted(glob.glob(AO+'/eval_rl_replace_math500/math500_500/*.jsonl')),
 'opd_step200 math500(V0)': sorted(glob.glob(AO+'/eval_opd/opd_step200/math500_500/*.jsonl')),
 'opd_step200 4xAIME(V0)': sorted(glob.glob(AO+'/eval_opd/opd_step200/aime*/*.jsonl')),
}
for k,files in runs.items():
    res=scan(files,['base','rl_replace','full_rl'])
    for c,R in res.items():
        n=R['n']
        print(f'{k:26s} {c:10s} n={n:6d} anom={100*R["anom"]/n:5.2f} foreign={100*R["foreign"]/n:5.2f} echo={100*R["echo"]/n:5.2f} trunc={100*R["trunc"]/n:5.2f} acc={100*R["acc"]/n:5.2f}')
