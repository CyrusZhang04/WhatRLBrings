# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,sys,glob,collections,statistics,re
from tokenizers import Tokenizer
tok=Tokenizer.from_file(sys.argv[1]); files=sys.argv[2:]
dec_cache={}
def dec(i):
    if i not in dec_cache: dec_cache[i]=tok.decode([i], skip_special_tokens=False)
    return dec_cache[i]
REASON=set(w.lower() for w in ['Wait','But','Maybe','maybe','Perhaps','Alright','Hmm','think','consider','check','verify','Actually','actually','So','Let','let','Now','First','Next','Therefore','Thus','However','Hmm','recall','try','seems','means','Okay','OK','Oh','Ah','double','confirm','re','Re','again','approach','realize','notice','assume','suppose'])
FUNC=set(['the','a','an','we','i','of','and','to','in','is','that','this','it','for','be','can','are','as','with','by','on','at','or','so','if','then','which','from','our','my','me','us','there','these','those','will','would','should','have','has','was','were'])
def cat(s):
    st=s.strip()
    if st=='' : return 'whitespace/newline'
    if re.fullmatch(r'[\.\,\:\;\!\?\)\(\]\[\}\{]+[\n ]*',s.strip(' ')) or re.fullmatch(r'[\s\.\,\:\;\!\?\)\(\]\[]+',s): return 'punct/format'
    if '#' in st or '**' in st: return 'markdown'
    if re.fullmatch(r'\d+',st): return 'digit'
    if re.search(r'[\\\$=\+\-\*/\^_<>|]',st): return 'math/latex'
    w=st.lower().strip('.,:;!?')
    if w in REASON: return 'discourse/reasoning'
    if w in FUNC: return 'function word'
    if re.fullmatch(r'[A-Za-z]+',st): return 'other word'
    return 'other'
dens_ev=[0]*10; dens_tok=[0]*10
cats=collections.Counter(); bcats=collections.Counter(); pair_cats=collections.Counter()
corr=[];inc=[]; corr_n=[]; inc_n=[]
nev=0
for fn in files:
    with open(fn) as f:
        for line in f:
            d=json.loads(line); r=d['rl_replace']
            for L,a,rr,ev in zip(r['lengths'],r['accs'],r['replace_rates'],r['replacements']):
                (corr if a else inc).append(rr); (corr_n if a else inc_n).append(len(ev))
                if L<=0: continue
                for k in range(10):
                    lo=int(L*k/10); hi=int(L*(k+1)/10)
                    dens_tok[k]+=hi-lo
                for e in ev:
                    p=e['pos']; k=min(9,int(10*p/L)); dens_ev[k]+=1
                    nev+=1
                    c=cat(dec(e['rl_id'])); cats[c]+=1; bcats[cat(dec(e['base_id']))]+=1
print('events',nev)
print('replacement density by relative-position decile (events/token):',[round(e/t,4) for e,t in zip(dens_ev,dens_tok)])
print('replace_rate correct vs incorrect: mean %.4f (n=%d) vs %.4f (n=%d)'%(statistics.mean(corr),len(corr),statistics.mean(inc),len(inc)))
print('n_replacements correct vs incorrect: mean %.1f vs %.1f; median %.1f vs %.1f'%(statistics.mean(corr_n),statistics.mean(inc_n),statistics.median(corr_n),statistics.median(inc_n)))
print('RL-token category share:',{k:round(v/nev,3) for k,v in cats.most_common()})
print('base-token category share:',{k:round(v/nev,3) for k,v in bcats.most_common()})
