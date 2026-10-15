"""Turn raw generations into the per-problem aggregate every table reads.

One ``ProblemRecord`` per problem per run, holding the per-sample correctness
of each condition.  Every number in the paper is computed from these: pass@1
and pass@k are means over problems, the bootstrap resamples problems, and the
recovery rate is a ratio of problem-level means.  Keeping the per-sample
booleans (rather than a precomputed accuracy) is what makes the paired
bootstraps and the pass@k estimator possible after the fact.

This replaces the ad-hoc ``perq.json`` of the original analysis, which held the
same thing.
"""

from __future__ import annotations

import collections
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .. import records as rec_io
from ..metrics import pass_at_1, pass_at_k


@dataclass
class ProblemRecord:
    problem: str
    ds: str
    qi: int
    group: str
    question: str
    gt: str
    #: condition -> per-sample correctness
    conds: Dict[str, List[bool]] = field(default_factory=dict)
    #: condition -> per-sample generated length
    lengths: Dict[str, List[int]] = field(default_factory=dict)
    #: condition -> per-sample replace rate (rl-replace only)
    replace_rates: Dict[str, List[float]] = field(default_factory=dict)

    def pass1(self, cond: str) -> float:
        return pass_at_1(self.conds[cond])

    def passk(self, cond: str, k: int) -> float:
        return pass_at_k(self.conds[cond], k)


def build(
    run_dir: str,
    splits: Optional[Sequence[str]] = None,
) -> List[ProblemRecord]:
    """Read a run directory into per-problem records."""
    out: List[ProblemRecord] = []
    seen: Dict[str, ProblemRecord] = {}
    for raw in rec_io.read_run(run_dir):
        if splits and raw["ds"] not in splits:
            continue
        pid = rec_io.problem_id(raw)
        pr = seen.get(pid)
        if pr is None:
            pr = ProblemRecord(
                problem=pid,
                ds=raw["ds"],
                qi=int(raw["qi"]),
                group=rec_io.group_of(raw["ds"]),
                question=raw["question"],
                gt=str(raw["gt"]),
            )
            seen[pid] = pr
            out.append(pr)
        for cond in rec_io.condition_names(raw):
            blk = raw[cond]
            pr.conds.setdefault(cond, []).extend(bool(a) for a in blk["accs"])
            pr.lengths.setdefault(cond, []).extend(int(x) for x in blk["lengths"])
            if blk.get("replace_rates"):
                pr.replace_rates.setdefault(cond, []).extend(
                    float(x) for x in blk["replace_rates"]
                )
    return out


def merge(*record_sets: Sequence[ProblemRecord]) -> List[ProblemRecord]:
    """Merge records for the same problems across runs.

    Conditions from different runs are merged by name.  A condition present in
    two runs is a collision -- the same name for two different generations --
    and raises, because silently concatenating them would mix sample budgets.
    """
    merged: Dict[str, ProblemRecord] = {}
    order: List[str] = []
    for rs in record_sets:
        for r in rs:
            cur = merged.get(r.problem)
            if cur is None:
                merged[r.problem] = r
                order.append(r.problem)
                continue
            for cond, accs in r.conds.items():
                if cond in cur.conds:
                    raise ValueError(
                        "condition {!r} appears in two runs for problem {}; "
                        "rename one before merging".format(cond, r.problem)
                    )
                cur.conds[cond] = accs
                cur.lengths[cond] = r.lengths.get(cond, [])
                if cond in r.replace_rates:
                    cur.replace_rates[cond] = r.replace_rates[cond]
    return [merged[p] for p in order]


def select(
    rs: Sequence[ProblemRecord],
    group: Optional[str] = None,
    conds: Optional[Sequence[str]] = None,
) -> List[ProblemRecord]:
    """Problems in ``group`` that have every condition in ``conds``.

    Restricting to problems that have *all* requested conditions is what makes
    the comparisons paired; an unpaired comparison across different problem
    subsets is the easiest way to get a wrong confidence interval.
    """
    out = []
    for r in rs:
        if group and r.group != group:
            continue
        if conds and not all(c in r.conds for c in conds):
            continue
        out.append(r)
    return out


def per_problem_pass1(rs: Sequence[ProblemRecord], cond: str) -> List[float]:
    return [r.pass1(cond) for r in rs]


def per_problem_passk(rs: Sequence[ProblemRecord], cond: str, k: int) -> List[float]:
    return [r.passk(cond, k) for r in rs]


def groups_of(rs: Sequence[ProblemRecord]) -> List[str]:
    """Group label per problem, for the stratified bootstrap."""
    return [r.group for r in rs]


def sample_counts(rs: Sequence[ProblemRecord]) -> Dict[str, collections.Counter]:
    """Per-condition histogram of samples-per-problem.

    Worth printing before a table: a condition with a ragged sample count
    across problems means a shard did not finish, and pass@k would be computed
    at different n per problem.
    """
    out: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rs:
        for cond, accs in r.conds.items():
            out[cond][len(accs)] += 1
    return dict(out)


def save(rs: Sequence[ProblemRecord], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(
                json.dumps(
                    {
                        "problem": r.problem,
                        "ds": r.ds,
                        "qi": r.qi,
                        "group": r.group,
                        "question": r.question,
                        "gt": r.gt,
                        "conds": {k: [bool(x) for x in v] for k, v in r.conds.items()},
                        "lengths": r.lengths,
                        "replace_rates": r.replace_rates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load(path: str) -> List[ProblemRecord]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                ProblemRecord(
                    problem=d["problem"],
                    ds=d["ds"],
                    qi=d["qi"],
                    group=d["group"],
                    question=d["question"],
                    gt=d["gt"],
                    conds={k: [bool(x) for x in v] for k, v in d["conds"].items()},
                    lengths=d.get("lengths", {}),
                    replace_rates=d.get("replace_rates", {}),
                )
            )
    return out
