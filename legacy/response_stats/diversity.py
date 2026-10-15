# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,statistics
def wrapped(text):
    if text and text[0]=='{':
        st,ans=1,''
        for c in text[1:]:
            if c=='{': st+=1
            elif c=='}':
                st-=1
                if st==0: return ans
            ans+=c
    return ''
def lastbox(t): return wrapped(t.split('\\boxed')[-1]) if '\\boxed' in t else ''
res={c:{'uniq_ans':[], 'uniq_pref':[]} for c in ('base','rl_replace','full_rl')}
for fn in sys.argv[1:]:
    for l in open(fn):
        x=json.loads(l)
        for c in res:
            T=x[c]['texts']
            res[c]['uniq_ans'].append(len(set(lastbox(t).strip() for t in T)))
            res[c]['uniq_pref'].append(len(set(t[:300] for t in T)))
for c,v in res.items():
    print(c,'questions',len(v['uniq_ans']),'mean distinct final answers per q (of 16): %.2f'%statistics.mean(v['uniq_ans']),'| mean distinct 300-char prefixes: %.2f'%statistics.mean(v['uniq_pref']))
