#!/usr/bin/env python
"""Build the paper's tables from one or more run directories.

Every run directory is read into the per-problem aggregate, the runs are merged
by problem, and the requested tables are printed.

Examples
--------
::

    python scripts/make_tables.py --runs outputs/v1 outputs/alright outputs/rules \\
        --tables 3 4 6 8

    python scripts/make_tables.py --runs outputs/large_k --tables 11
    python scripts/make_tables.py --runs outputs/v1 --tables stats --max-tokens 8192
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrb.analysis import aggregate, mixture, response_stats
from wrb.constants import MAX_NEW_TOKENS
from wrb.analysis.tables import delta_table, passk_curve, render, table3, table8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument(
        "--tables",
        nargs="+",
        default=["3"],
        help="any of: 3, 4, 6, 8, 11, mix, stats",
    )
    p.add_argument("--k", type=int, default=16, help="k for the pass@k column")
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS, help="generation limit of the run")
    p.add_argument("--base", default="base")
    p.add_argument("--method", default="rl_replace")
    p.add_argument("--rl", default="full_rl")
    p.add_argument("--save-aggregate", default=None, help="write the merged aggregate here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sets = [aggregate.build(r) for r in args.runs]
    rs = aggregate.merge(*sets) if len(sets) > 1 else sets[0]
    print("{} problems from {} run(s)".format(len(rs), len(args.runs)))

    counts = aggregate.sample_counts(rs)
    ragged = {c: dict(h) for c, h in counts.items() if len(h) > 1}
    if ragged:
        print("\nWARNING: conditions with an uneven sample count per problem:")
        for c, h in ragged.items():
            print("  {}: {}".format(c, h))
        print("  pass@k is computed per problem, so these rows mix sample budgets.")

    if args.save_aggregate:
        aggregate.save(rs, args.save_aggregate)
        print("aggregate written to {}".format(args.save_aggregate))

    conds = sorted({c for r in rs for c in r.conds})
    print("\nconditions present: {}".format(", ".join(conds)))

    for t in args.tables:
        print()
        if t == "3":
            print(render(table3(rs, args.base, args.method, args.rl, k=args.k),
                         "Table 3 | rl-replace versus base and RL"))
        elif t == "4":
            prefill = [c for c in ("base_with_alright", "base_with_alright_i",
                                   "base_greedy_at_disagree") if c in conds]
            print(render(delta_table(rs, prefill, args.base, k=args.k),
                         "Table 4 | forcing the RL model's opening (Test 2)"))
        elif t == "6":
            rules = [c for c in ("force_wait", "force_re", "force_wait_re",
                                 "re_logit_plus2", "re_logit_plus4", "wait_topk") if c in conds]
            print(render(delta_table(rs, rules, args.base, k=args.k),
                         "Table 6 | rule-based insertion of reflective tokens (Test 3)"))
        elif t == "8":
            print(render(table8(rs, args.base, args.method, args.rl, k=args.k),
                         "Table 8 | results by competition year"))
        elif t == "11":
            print(render(passk_curve(rs, [c for c in (args.base, args.rl, args.method) if c in conds]),
                         "Table 11 | pass@k (Figure 1)"))
        elif t == "mix":
            print(mixture.render(
                mixture.mixture_curve(rs, args.base, args.rl, match=args.method)
            ))
        elif t == "stats":
            for run in args.runs:
                print(response_stats.render(
                    response_stats.collect(run, max_tokens=args.max_tokens)
                ))
        else:
            print("unknown table {!r}; known: 3, 4, 6, 8, 11, mix, stats".format(t))


if __name__ == "__main__":
    main()
