# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, collections
path, tag, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
dis_rl = collections.Counter(); crit_rl = collections.Counter(); dis_n = 0; crit_n = 0
dis_in_deg = 0
with open(path) as f:
    for line in f:
        r = json.loads(line)
        for si, ors in enumerate(r['oracle_signals']):
            L = r['lengths'][si]
            for o in ors:
                if o['chosen_rank_in_rl'] != 1:
                    dis_rl[o['rl_top_text']] += 1; dis_n += 1
                    if L >= 16000: dis_in_deg += 1
                if o['kl_fwd'] >= thr:
                    crit_rl[o['rl_top_text']] += 1; crit_n += 1
out = dict(tag=tag, n_disagree=dis_n, disagree_in_len_ge_16000=dis_in_deg, n_crit=crit_n,
           top_rl_tokens_at_disagree=dis_rl.most_common(30), top_rl_tokens_at_crit=crit_rl.most_common(30))
json.dump(out, open(f'${WRB_WORK}/{tag}.crittok.json','w'), ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False))
