# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json,glob,collections
from tokenizers import Tokenizer
t=Tokenizer.from_file('${WRB_MODELS}/Qwen3-4B-Base/tokenizer.json')
B='${WRB_RUNS}/eval_rl_replace_v2/'
rl=collections.Counter(); pair=collections.Counter(); base=collections.Counter()
pos0=collections.Counter(); pos2=collections.Counter()
nrep=0; nseq=0; first_text=collections.Counter(); rr_text=collections.Counter()
for f in glob.glob(B+'*/*-n16-shard*of8.jsonl'):
    for line in open(f):
        r=json.loads(line)
        x=r['rl_replace']
        for reps,txt in zip(x['replacements'],x['texts']):
            nseq+=1
            first_text[txt[:12]]+=1
            for d in reps:
                nrep+=1
                rl[d['rl_id']]+=1; base[d['base_id']]+=1; pair[(d['base_id'],d['rl_id'])]+=1
                if d['pos']==0: pos0[(d['base_id'],d['rl_id'])]+=1
                if d['pos']==2: pos2[(d['base_id'],d['rl_id'])]+=1
        for txt in r['full_rl']['texts']:
            rr_text[txt[:12]]+=1
dec=lambda i: repr(t.decode([i]))
print('n sequences',nseq,'n replacements',nrep, 'avg per seq %.1f'%(nrep/nseq))
print('top rl_id (replacement token):'); print(', '.join(f'{dec(i)}:{c}' for i,c in rl.most_common(30)))
print('top base_id (replaced token):'); print(', '.join(f'{dec(i)}:{c}' for i,c in base.most_common(30)))
print('top pairs base->rl:'); print(', '.join(f'{dec(a)}->{dec(b)}:{c}' for (a,b),c in pair.most_common(30)))
print('pos0 pairs:', [(dec(a),dec(b),c) for (a,b),c in pos0.most_common(5)], 'total seq with pos0 replaced', sum(pos0.values()))
print('pos2 pairs:', [(dec(a),dec(b),c) for (a,b),c in pos2.most_common(5)])
print('rl_replace text starts:', first_text.most_common(6))
print('full_rl text starts:', rr_text.most_common(6))
for tok in [' Re','Wait',' I',' we',' our',' We','Alright']:
    i=t.encode(tok,add_special_tokens=False).ids[0]
    print(f'{tok!r} id={i} as rl_id count={rl[i]} ({rl[i]/nrep*100:.2f}% of reps); as base_id count={base[i]}')
