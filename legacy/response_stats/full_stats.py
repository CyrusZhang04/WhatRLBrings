# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import json, sys, glob, os, re, zlib, statistics, math
from math import comb

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

CYR = re.compile(r'[Ѐ-ӿ]')
RUN2 = re.compile(r'(?:\b(\S{1,3})\s+)(?:\1\s+){25,}')  # same short token repeated >=26 times
def garbage_flags(t, L, maxtok):
    f = {}
    f['empty_or_lt10tok'] = L < 10
    tail = t[-2000:]
    f['short_token_run'] = bool(RUN2.search(t[-3000:])) if t else False
    f['cyrillic'] = bool(CYR.search(t[:200]))
    f['digits_only_start'] = bool(re.match(r'^\s*[\d\.\s\$]{1,60}$', t[:60])) and L < 200
    ratio = len(zlib.compress(tail.encode())) / max(1, len(tail.encode())) if len(tail) >= 500 else 1.0
    f['tail_low_entropy_at_max'] = (L >= maxtok) and ratio < 0.12
    f['has_boxed'] = '\\boxed' in t
    f['ends_eos'] = t.endswith('<|endoftext|>') or t.endswith('<|im_end|>') or t.endswith('<|eot_id|>')
    return f

def main(d, maxtok=16384, ks=(1,4,16,64,256)):
    files = sorted(glob.glob(os.path.join(d, '*.jsonl')))
    res = {}
    nq = 0
    per_q = []  # per question accs per cond
    rep = {'n_events': [], 'rr': [], 'len': [], 'pos0': 0, 'relpos': [], 'abspos': [], 'n_resp': 0, 'first_pos': []}
    for fn in files:
        with open(fn) as f:
            for line in f:
                dd = json.loads(line); nq += 1
                q = {'gt': dd['gt']}
                for cond in ('base', 'rl_replace', 'full_rl'):
                    c = dd.get(cond)
                    if c is None: continue
                    r = res.setdefault(cond, {'L': [], 'A': [], 'passk': {k: [] for k in ks}, 'flags': {}, 'n_text': 0})
                    L = c['lengths']; A = [bool(x) for x in c['accs']]
                    n = len(A); cc = sum(A)
                    r['L'] += L; r['A'] += A
                    for k in ks:
                        if k <= n: r['passk'][k].append(pass_at_k(n, cc, k))
                    q[cond] = (n, cc)
                    texts = c.get('texts')
                    if texts:
                        for t, l in zip(texts, L):
                            r['n_text'] += 1
                            for kk, v in garbage_flags(t, l, maxtok).items():
                                r['flags'][kk] = r['flags'].get(kk, 0) + int(v)
                    if cond == 'rl_replace':
                        reps = c.get('replacements')
                        rrs = c.get('replace_rates')
                        for i in range(n):
                            rep['n_resp'] += 1
                            rep['rr'].append(rrs[i]); rep['len'].append(L[i])
                            if reps is not None:
                                ev = reps[i]
                                rep['n_events'].append(len(ev))
                                if ev:
                                    ps = [e['pos'] for e in ev]
                                    rep['first_pos'].append(min(ps))
                                    if 0 in ps: rep['pos0'] += 1
                                    rep['abspos'] += ps
                                    if L[i] > 0:
                                        rep['relpos'] += [p / L[i] for p in ps]
                per_q.append(q)
    out = {'dir': d, 'n_questions': nq}
    for cond, r in res.items():
        o = {'samples': len(r['A']), 'acc_mean(pass@1 sample-avg)': sum(r['A']) / len(r['A']),
             'meanlen': statistics.mean(r['L']), 'medlen': statistics.median(r['L']),
             'frac_at_max': sum(1 for x in r['L'] if x >= maxtok) / len(r['L'])}
        for k, v in r['passk'].items():
            if v: o[f'pass@{k}'] = sum(v) / len(v)
        if r['n_text']:
            o['n_text'] = r['n_text']
            o['flags_frac'] = {k: round(v / r['n_text'], 4) for k, v in r['flags'].items()}
        out[cond] = o
    if rep['n_resp']:
        o = {'n_resp': rep['n_resp'], 'mean_replace_rate(sample-avg)': statistics.mean(rep['rr']),
             'token_weighted_replace_rate': (sum(r*l for r, l in zip(rep['rr'], rep['len'])) / sum(rep['len'])) if sum(rep['len']) else None}
        if rep['n_events']:
            o['mean_n_replacements_per_resp'] = statistics.mean(rep['n_events'])
            o['median_n_replacements_per_resp'] = statistics.median(rep['n_events'])
            o['frac_resp_with_pos0_replaced'] = rep['pos0'] / rep['n_resp']
            o['median_first_pos'] = statistics.median(rep['first_pos']) if rep['first_pos'] else None
            rp = sorted(rep['relpos'])
            if rp:
                o['relpos_quantiles(10/25/50/75/90)'] = [round(rp[int(len(rp) * q)], 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                o['frac_events_in_first_10pct'] = sum(1 for x in rp if x < 0.1) / len(rp)
            ap = rep['abspos']
            o['frac_events_pos_lt_100'] = sum(1 for x in ap if x < 100) / len(ap)
        out['replace_stats'] = o
    # paired per-question comparisons
    if per_q and all('base' in q and 'rl_replace' in q and 'full_rl' in q for q in per_q):
        def p1(t): return t[1] / t[0]
        b = [p1(q['base']) for q in per_q]; r_ = [p1(q['rl_replace']) for q in per_q]; f_ = [p1(q['full_rl']) for q in per_q]
        out['per_question'] = {
            'n_q_rlrep_gt_base': sum(1 for x, y in zip(r_, b) if x > y + 1e-12),
            'n_q_rlrep_lt_base': sum(1 for x, y in zip(r_, b) if x < y - 1e-12),
            'n_q_rlrep_gt_full': sum(1 for x, y in zip(r_, f_) if x > y + 1e-12),
            'n_q_rlrep_lt_full': sum(1 for x, y in zip(r_, f_) if x < y - 1e-12),
            'n_q_solved_any_base': sum(1 for q in per_q if q['base'][1] > 0),
            'n_q_solved_any_rlrep': sum(1 for q in per_q if q['rl_replace'][1] > 0),
            'n_q_solved_any_full': sum(1 for q in per_q if q['full_rl'][1] > 0),
        }
        out['per_question_pass1'] = [{'gt': q['gt'], 'base': round(p1(q['base']), 4), 'rl_replace': round(p1(q['rl_replace']), 4), 'full_rl': round(p1(q['full_rl']), 4)} for q in per_q]
    return out

if __name__ == '__main__':
    d = sys.argv[1]; maxtok = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    out = main(d, maxtok)
    outfn = sys.argv[3] if len(sys.argv) > 3 else None
    pq = out.pop('per_question_pass1', None)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if outfn:
        out['per_question_pass1'] = pq
        json.dump(out, open(outfn, 'w'), indent=1, ensure_ascii=False)
