# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,os
for d in sys.argv[1:]:
    res={}
    for fn in sorted(glob.glob(os.path.join(d,'*.jsonl'))):
        with open(fn) as f:
            for line in f:
                x=json.loads(line)
                for c in ('base','rl_replace','full_rl'):
                    r=res.setdefault(c,[0,0,0,0])
                    for t,L,a in zip(x[c]['texts'],x[c]['lengths'],x[c]['accs']):
                        r[0]+=1
                        if L<300: r[1]+=1
                        if L<300 and '\\boxed' not in t: r[2]+=1
                        if L>=300 and L<16384 and '\\boxed' not in t: r[3]+=1
    print(d, {c:f"short<300={v[1]/v[0]:.3f} short_noboxed={v[2]/v[0]:.3f} midlen_noboxed={v[3]/v[0]:.3f}" for c,v in res.items()})
