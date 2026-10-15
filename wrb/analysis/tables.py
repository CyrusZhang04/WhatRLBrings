"""Reproduce the paper's tables from a per-problem aggregate.

    Table 3   rl-replace vs base and RL, with recovery and bootstrap CIs
    Table 4   forcing the RL model's opening (Test 2)
    Table 6   rule-based insertion (Test 3)
    Table 8   results by competition year
    Table 11  exact pass@k values (Figure 1)

Each builder takes the aggregate and the condition names to read, so a rerun
with different condition names does not need the table code edited.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..metrics import pass_at_k_mean, recovery_ci
from ..metrics import bootstrap_ci, paired_diff_ci
from .aggregate import ProblemRecord, groups_of, per_problem_pass1, per_problem_passk, select

BASE = "base"
RL_REPLACE = "rl_replace"
FULL_RL = "full_rl"


def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else "{:.2f}".format(100 * x)


def table3(
    rs: Sequence[ProblemRecord],
    base: str = BASE,
    method: str = RL_REPLACE,
    rl: str = FULL_RL,
    k: int = 16,
    groups: Sequence[str] = ("aime24+25", "aime22+23"),
) -> List[Dict[str, object]]:
    """Table 3: pass@1, pass@k and recovery for each problem group and overall."""
    rows = []
    conds = [base, method, rl]
    for label in list(groups) + ["All"]:
        sel = select(rs, None if label == "All" else label, conds)
        if not sel:
            continue
        p1 = {c: per_problem_pass1(sel, c) for c in conds}
        rec = recovery_ci(p1[method], p1[base], p1[rl], groups=groups_of(sel))
        rows.append(
            {
                "problems": label,
                "n": len(sel),
                "pass1_base": _fmt_pct(sum(p1[base]) / len(sel)),
                "pass1_method": _fmt_pct(sum(p1[method]) / len(sel)),
                "pass1_rl": _fmt_pct(sum(p1[rl]) / len(sel)),
                "recovery": "{:.1f} [{:.1f}, {:.1f}]".format(
                    100 * rec["recovery"], 100 * rec["lo"], 100 * rec["hi"]
                ),
                "passk_base": _fmt_pct(pass_at_k_mean([r.conds[base] for r in sel], k)),
                "passk_method": _fmt_pct(pass_at_k_mean([r.conds[method] for r in sel], k)),
                "passk_rl": _fmt_pct(pass_at_k_mean([r.conds[rl] for r in sel], k)),
                "frac_denom_nonpositive": round(rec["frac_denom_nonpositive"], 4),
            }
        )
    return rows


def delta_table(
    rs: Sequence[ProblemRecord],
    conditions: Sequence[str],
    base: str = BASE,
    k: int = 16,
    reference: Optional[Sequence[str]] = (RL_REPLACE, FULL_RL),
) -> List[Dict[str, object]]:
    """Tables 4 and 6: each condition's pass@1 / pass@k and delta vs the base.

    Conditions are compared against the base model on the problems where both
    are present, so each delta is paired.  ``reference`` rows (rl-replace, RL)
    are appended for context, as both tables do.
    """
    rows = []
    wanted = list(conditions) + [c for c in (reference or ()) if c not in conditions]
    for cond in [base] + wanted:
        sel = select(rs, None, [base, cond])
        if not sel:
            continue
        p1_c = per_problem_pass1(sel, cond)
        p1_b = per_problem_pass1(sel, base)
        pk_c = per_problem_passk(sel, cond, k)
        pk_b = per_problem_passk(sel, base, k)
        g = groups_of(sel)
        d1, d1lo, d1hi = paired_diff_ci(p1_c, p1_b, groups=g)
        dk, dklo, dkhi = paired_diff_ci(pk_c, pk_b, groups=g)
        rows.append(
            {
                "condition": cond,
                "n": len(sel),
                "pass1": _fmt_pct(sum(p1_c) / len(sel)),
                "delta_pass1": "-" if cond == base else "{:+.2f} [{:+.2f}, {:+.2f}]".format(
                    100 * d1, 100 * d1lo, 100 * d1hi
                ),
                "passk": _fmt_pct(sum(pk_c) / len(sel)),
                "delta_passk": "-" if cond == base else "{:+.2f} [{:+.2f}, {:+.2f}]".format(
                    100 * dk, 100 * dklo, 100 * dkhi
                ),
            }
        )
    return rows


def table8(
    rs: Sequence[ProblemRecord],
    base: str = BASE,
    method: str = RL_REPLACE,
    rl: str = FULL_RL,
    k: int = 16,
) -> List[Dict[str, object]]:
    """Table 8: one row per competition year, with recovery and replace rate."""
    rows = []
    for ds in sorted({r.ds for r in rs}):
        sel = [r for r in select(rs, None, [base, method, rl]) if r.ds == ds]
        if not sel:
            continue
        p1 = {c: per_problem_pass1(sel, c) for c in (base, method, rl)}
        means = {c: sum(v) / len(sel) for c, v in p1.items()}
        rec = recovery_ci(p1[method], p1[base], p1[rl])
        rr = [x for r in sel for x in r.replace_rates.get(method, [])]
        rows.append(
            {
                "year": ds,
                "n": len(sel),
                "pass1": "{} / {} / {}".format(
                    _fmt_pct(means[base]), _fmt_pct(means[method]), _fmt_pct(means[rl])
                ),
                "passk": "{} / {} / {}".format(
                    _fmt_pct(pass_at_k_mean([r.conds[base] for r in sel], k)),
                    _fmt_pct(pass_at_k_mean([r.conds[method] for r in sel], k)),
                    _fmt_pct(pass_at_k_mean([r.conds[rl] for r in sel], k)),
                ),
                "recovery": "{:.1f}".format(100 * rec["recovery"]),
                "replace_rate": "{:.1f}".format(100 * sum(rr) / len(rr)) if rr else "-",
            }
        )
    return rows


def passk_curve(
    rs: Sequence[ProblemRecord],
    conditions: Sequence[str] = (BASE, FULL_RL, RL_REPLACE),
    ks: Sequence[int] = (1, 4, 16, 64, 128, 256),
    group: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Table 11 / Figure 1: pass@k for each condition.

    A condition is skipped at a given k when it has fewer than k samples: the
    paper draws 512 samples for the base and RL models and 256 for rl-replace,
    so the curves end at different k.
    """
    rows = []
    for cond in conditions:
        sel = select(rs, group, [cond])
        if not sel:
            continue
        row: Dict[str, object] = {"condition": cond, "n_problems": len(sel)}
        n_min = min(len(r.conds[cond]) for r in sel)
        for k in ks:
            row["k={}".format(k)] = _fmt_pct(
                pass_at_k_mean([r.conds[cond] for r in sel], k)
            ) if k <= n_min else "-"
        rows.append(row)
    return rows


def diversity_table(
    rs: Sequence[ProblemRecord],
    conditions: Sequence[str] = (BASE, RL_REPLACE, FULL_RL),
    texts_by_cond: Optional[Dict[str, Dict[str, Sequence[str]]]] = None,
    answers_by_cond: Optional[Dict[str, Dict[str, Sequence[Optional[str]]]]] = None,
) -> List[Dict[str, object]]:
    """§3.6: mean distinct openings and distinct final answers per problem.

    Needs the response texts, which the aggregate does not carry, so the two
    maps are ``condition -> problem -> values``.  ``wrb.analysis.response_stats``
    builds them from a run directory.
    """
    rows = []
    from ..metrics import distinct_answers, distinct_openings

    for cond in conditions:
        row: Dict[str, object] = {"condition": cond}
        if texts_by_cond and cond in texts_by_cond:
            per_q = [distinct_openings(v) for v in texts_by_cond[cond].values()]
            row["distinct_openings"] = round(sum(per_q) / len(per_q), 2) if per_q else "-"
        if answers_by_cond and cond in answers_by_cond:
            per_q = [distinct_answers(v) for v in answers_by_cond[cond].values()]
            row["distinct_answers"] = round(sum(per_q) / len(per_q), 2) if per_q else "-"
        rows.append(row)
    return rows


def render(rows: Sequence[Dict[str, object]], title: str = "") -> str:
    """Render a list of row dicts as a fixed-width text table."""
    if not rows:
        return "{}\n(no rows)".format(title)
    cols = list(rows[0].keys())
    widths = {
        c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols
    }
    out = []
    if title:
        out.append(title)
    out.append("  ".join(str(c).ljust(widths[c]) for c in cols))
    out.append("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(out)
