"""§3.6: mixing whole responses from the two models, as a control.

`rl-replace` mixes the two models *within* a response, at about 4% of token
positions.  The obvious cheaper control is to mix them *between* responses:
draw each attempt entirely from one model or the other.  §3.6 reports that this
does not reproduce the result -- drawing 92% of the responses from the RL model
and 8% from the base model matches rl-replace at k = 1 (19.00%) but solves only
61.45% of the problems at k = 256, against 67.50% for rl-replace.

The mixing fraction is not a free parameter: it is chosen so that the mixture
matches the method it is being compared against at k = 1, which is what makes
the divergence at large k the whole of the finding.  ``matching_fraction``
solves for it in closed form.

Two estimators are provided.  The analytic one treats each attempt as an
independent draw and is exact under that model; the empirical one resamples
actual responses, which is slower but makes no distributional assumption.  They
agree to within Monte-Carlo error, and the analytic one is the default.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .aggregate import ProblemRecord


def matching_fraction(
    base_pass1: float, rl_pass1: float, target_pass1: float, clamp: bool = True
) -> float:
    """The RL fraction p for which the mixture's pass@1 equals ``target_pass1``.

    pass@1 of the mixture is linear in p::

        pass@1(p) = p * rl_pass1 + (1 - p) * base_pass1

    so p = (target - base) / (rl - base).  With the paper's 512-sample values
    (base 10.26, RL 19.77, rl-replace 18.98) this gives 0.917, the "92% of the
    responses from the RL model" of §3.6.

    Returns NaN when the two models have the same pass@1, where no mixture can
    be pinned down.  Outside [0, 1] the target is unreachable by any mixture;
    ``clamp`` caps it at the nearer endpoint rather than returning a fraction
    that cannot be realized.
    """
    denom = rl_pass1 - base_pass1
    if abs(denom) < 1e-12:
        return float("nan")
    p = (target_pass1 - base_pass1) / denom
    if clamp:
        p = min(1.0, max(0.0, p))
    return float(p)


def mixture_pass_at_k(
    base_correct: Sequence[bool],
    rl_correct: Sequence[bool],
    p_rl: float,
    k: int,
) -> float:
    """Analytic pass@k for one problem under the response mixture.

    Each of the k attempts is drawn from the RL model with probability
    ``p_rl``, and is then correct with that model's accuracy on this problem::

        P(all k fail) = (p * (1 - acc_rl) + (1 - p) * (1 - acc_base)) ** k
    """
    if not 0.0 <= p_rl <= 1.0:
        raise ValueError("p_rl must be in [0, 1], got {}".format(p_rl))
    acc_base = float(np.mean(np.asarray(base_correct, dtype=float))) if len(base_correct) else 0.0
    acc_rl = float(np.mean(np.asarray(rl_correct, dtype=float))) if len(rl_correct) else 0.0
    p_fail = p_rl * (1.0 - acc_rl) + (1.0 - p_rl) * (1.0 - acc_base)
    return 1.0 - p_fail ** k


def mixture_pass_at_k_empirical(
    base_correct: Sequence[bool],
    rl_correct: Sequence[bool],
    p_rl: float,
    k: int,
    n_trials: int = 2000,
    seed: int = 11,
) -> float:
    """Monte-Carlo pass@k for one problem, resampling actual responses.

    Draws k attempts with replacement: each one picks a model, then a response
    from that model's pool.  Makes no assumption beyond the recorded samples.
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(base_correct, dtype=bool)
    r = np.asarray(rl_correct, dtype=bool)
    if b.size == 0 or r.size == 0:
        return float("nan")
    from_rl = rng.random((n_trials, k)) < p_rl
    picks_b = b[rng.integers(0, b.size, size=(n_trials, k))]
    picks_r = r[rng.integers(0, r.size, size=(n_trials, k))]
    correct = np.where(from_rl, picks_r, picks_b)
    return float(correct.any(axis=1).mean())


def mixture_curve(
    records: Sequence[ProblemRecord],
    base: str = "base",
    rl: str = "full_rl",
    match: Optional[str] = "rl_replace",
    p_rl: Optional[float] = None,
    ks: Sequence[int] = (1, 4, 16, 64, 128, 256),
    empirical: bool = False,
) -> Dict[str, object]:
    """The §3.6 mixture row: pass@k of the response mixture across k.

    By default the mixing fraction is chosen to match ``match`` at k = 1, which
    is how the paper sets it.  Pass ``p_rl`` to fix it explicitly instead.
    """
    usable = [r for r in records if base in r.conds and rl in r.conds]
    if not usable:
        return {"error": "no problems have both {!r} and {!r}".format(base, rl)}

    base_p1 = float(np.mean([r.pass1(base) for r in usable]))
    rl_p1 = float(np.mean([r.pass1(rl) for r in usable]))

    if p_rl is None:
        if match is None:
            raise ValueError("pass either p_rl or a condition to match at k=1")
        matched = [r for r in usable if match in r.conds]
        if not matched:
            return {"error": "no problems have {!r} to match against".format(match)}
        target = float(np.mean([r.pass1(match) for r in matched]))
        p_rl = matching_fraction(base_p1, rl_p1, target)
        matched_to = {"condition": match, "pass1_pct": 100 * target}
    else:
        matched_to = None

    estimator = mixture_pass_at_k_empirical if empirical else mixture_pass_at_k
    curve = {}
    for k in ks:
        vals = [estimator(r.conds[base], r.conds[rl], p_rl, k) for r in usable]
        curve["k={}".format(k)] = 100 * float(np.mean(vals))

    return {
        "n_problems": len(usable),
        "p_rl": p_rl,
        "pct_from_rl": 100 * p_rl,
        "base_pass1_pct": 100 * base_p1,
        "rl_pass1_pct": 100 * rl_p1,
        "matched_to": matched_to,
        "estimator": "empirical" if empirical else "analytic",
        "curve": curve,
    }


def render(result: Dict[str, object]) -> str:
    """Render the mixture row as text."""
    if "error" in result:
        return "response mixture: {}".format(result["error"])
    lines = [
        "Response mixture (§3.6): {:.1f}% of responses from the RL model, "
        "{:.1f}% from the base model".format(
            result["pct_from_rl"], 100 - float(result["pct_from_rl"])
        )
    ]
    m = result.get("matched_to")
    if m:
        lines.append(
            "  fraction chosen to match {} at k=1 ({:.2f}%)".format(
                m["condition"], m["pass1_pct"]
            )
        )
    lines.append(
        "  {} problems, {} estimator".format(result["n_problems"], result["estimator"])
    )
    curve = result["curve"]
    lines.append("  " + "  ".join("{:>10}".format(k) for k in curve))
    lines.append("  " + "  ".join("{:>10.2f}".format(v) for v in curve.values()))
    return "\n".join(lines)
