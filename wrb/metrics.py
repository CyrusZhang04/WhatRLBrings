"""Metrics (paper §4.1, "Metrics").

    pass@1    mean accuracy over all samples
    pass@k    fraction of problems solved by at least one of k samples, with the
              unbiased estimator of Chen et al. (2021) when k < n
    maj@k     accuracy of the most frequent final answer among k samples,
              grouping answers judged equivalent
    recovery  (method - base) / (RL - base), on pass@1

Confidence intervals are problem-level bootstrap percentile intervals with
10,000 resamples drawn *within each problem group*, so that a resample of the
120 AIME problems keeps 60 from each group.

"""

from __future__ import annotations

import collections
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 11


def pass_at_1(correct: Sequence[bool]) -> float:
    """Mean accuracy over samples of one problem."""
    a = np.asarray(correct, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(a.mean())


def pass_at_k(correct: Sequence[bool], k: int) -> float:
    """Unbiased pass@k for a single problem with n >= k samples.

    1 - C(n-c, k) / C(n, k), evaluated as a product to avoid overflow.
    """
    n = len(correct)
    if n == 0:
        return float("nan")
    if k > n:
        raise ValueError("pass@{} needs at least {} samples, got {}".format(k, k, n))
    c = int(np.asarray(correct, dtype=bool).sum())
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_mean(per_problem: Sequence[Sequence[bool]], k: int) -> float:
    """pass@k averaged over problems."""
    vals = [pass_at_k(c, k) for c in per_problem]
    return float(np.mean(vals)) if vals else float("nan")


def maj_at_k(
    answers: Sequence[Optional[str]],
    reference: str,
    k: Optional[int] = None,
    equivalent: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """1.0 if the most frequent answer among the first ``k`` samples is correct.

    ``answers`` are normalized answer strings (``None`` for a response with no
    final answer, which cannot win a vote).  Answers are grouped by
    ``equivalent`` when given; otherwise by exact string, which is what
    ``normalize_answer`` is for.  Ties go to the answer that appeared first.
    """
    pool = list(answers if k is None else answers[:k])
    groups: List[Tuple[str, int]] = []
    for a in pool:
        if a is None:
            continue
        for i, (rep, n) in enumerate(groups):
            same = equivalent(rep, a) if equivalent else rep == a
            if same:
                groups[i] = (rep, n + 1)
                break
        else:
            groups.append((a, 1))
    if not groups:
        return 0.0
    winner = max(groups, key=lambda t: t[1])[0]
    if equivalent:
        return float(equivalent(winner, reference))
    return float(winner == reference)


def recovery_rate(method_p1: float, base_p1: float, rl_p1: float) -> float:
    """R(method) = (method - base) / (RL - base), paper Eq. (1).

    Returns NaN when the RL model shows no gain over the base model, in which
    case the ratio is not interpretable.
    """
    denom = rl_p1 - base_p1
    if abs(denom) < 1e-12:
        return float("nan")
    return (method_p1 - base_p1) / denom


def _bootstrap_indices(
    n: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def _group_bootstrap_indices(
    groups: Sequence[str], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample within each group, preserving group sizes."""
    idx = np.arange(len(groups))
    by_group: Dict[str, np.ndarray] = {}
    for g in dict.fromkeys(groups):
        by_group[g] = idx[np.asarray(groups) == g]
    cols = [
        rows[rng.integers(0, len(rows), size=(n_boot, len(rows)))]
        for rows in by_group.values()
    ]
    return np.concatenate(cols, axis=1)


def bootstrap_ci(
    values: Sequence[float],
    groups: Optional[Sequence[str]] = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of a per-problem quantity.

    ``values`` has one entry per problem, so a paired difference is passed as
    the per-problem difference.  With ``groups``, resampling happens within
    each group (the paper's "drawn within each problem group").

    Returns ``(mean, lo, hi)``.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    I = (
        _group_bootstrap_indices(groups, n_boot, rng)
        if groups is not None
        else _bootstrap_indices(v.size, n_boot, rng)
    )
    means = v[I].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(v.mean()), lo, hi


def recovery_ci(
    method: Sequence[float],
    base: Sequence[float],
    rl: Sequence[float],
    groups: Optional[Sequence[str]] = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI for the recovery *ratio*, resampling problems.

    The three sequences are per-problem pass@1 values for the intervention, the
    base model and the RL model, aligned by problem.  The ratio is recomputed
    on each resample from the resampled means, which is why the interval can be
    wide and asymmetric (the paper's [77.0, 108.6] for 90.5%).

    ``frac_denom_nonpositive`` reports how often the resampled RL gain was
    <= 0, where the ratio is meaningless; a large value means the interval
    should not be read as a percentage.
    """
    m = np.asarray(method, dtype=float)
    b = np.asarray(base, dtype=float)
    r = np.asarray(rl, dtype=float)
    if not (m.shape == b.shape == r.shape):
        raise ValueError("method/base/rl must be aligned per problem")
    rng = np.random.default_rng(seed)
    I = (
        _group_bootstrap_indices(groups, n_boot, rng)
        if groups is not None
        else _bootstrap_indices(m.size, n_boot, rng)
    )
    mm, bm, rm = m[I].mean(1), b[I].mean(1), r[I].mean(1)
    denom = rm - bm
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (mm - bm) / denom
    bad = ~np.isfinite(ratios) | (denom <= 0)
    good = ratios[~bad]
    point = recovery_rate(float(m.mean()), float(b.mean()), float(r.mean()))
    return {
        "recovery": point,
        "lo": float(np.percentile(good, 100 * alpha / 2)) if good.size else float("nan"),
        "hi": float(np.percentile(good, 100 * (1 - alpha / 2))) if good.size else float("nan"),
        "frac_denom_nonpositive": float(bad.mean()),
        "n_boot": int(n_boot),
    }


def paired_diff_ci(
    a_per_problem: Sequence[float],
    b_per_problem: Sequence[float],
    groups: Optional[Sequence[str]] = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float]:
    """CI for the paired per-problem difference ``a - b`` (in the same units)."""
    a = np.asarray(a_per_problem, dtype=float)
    b = np.asarray(b_per_problem, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired inputs must be aligned per problem")
    return bootstrap_ci(a - b, groups=groups, n_boot=n_boot, seed=seed)


def distinct_openings(texts: Sequence[str], n_chars: int = 300) -> int:
    """Number of distinct response openings (§3.6: the first 300 characters)."""
    return len({t[:n_chars] for t in texts})


def distinct_answers(answers: Sequence[Optional[str]]) -> int:
    """Number of distinct non-empty final answers (§3.6)."""
    return len({a for a in answers if a})


def answer_histogram(answers: Sequence[Optional[str]]) -> "collections.Counter":
    return collections.Counter(a for a in answers if a)
