"""Appendix B: base-only features, and the labels they are scored against.

The question of Appendix B is whether the positions where the two models
diverge can be located *using only the base model*, without running the RL
model.  For each token the base model generates, six features are recorded,
all computed from the base model's own next-token distribution:

    entropy       entropy of the next-token distribution
    top1_prob     probability of the top candidate
    margin        gap between the top two candidates
    chosen_logp   log-probability of the token actually sampled
    token_rank    rank of that token in the base distribution
    rel_pos       relative position in the response

Two labels are evaluated, and they are not interchangeable:

    kl_top10   the 10% of positions with the largest KL(RL || base) between the
               two next-token distributions.  Depends only on the two
               distributions, not on what was sampled.  **This is the label the
               paper's conclusions rest on.**
    mismatch   positions where the RL model's top-1 token differs from the
               token the base model sampled.  Reaches AUROC 0.993 from
               chosen_logp alone, but almost by construction: a low chosen
               log-prob means the base model sampled something other than its
               own top-1, which makes the label positive. It also is *not* the
               criterion rl-replace uses, which compares the two models' top-1
               tokens to each other.

``rank`` is 1-based: the sampled token being the base model's own top choice
gives ``token_rank == 1``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

#: The six base-only features, in the canonical order used by the gate.
FEATURES = ("entropy", "margin", "top1_prob", "token_rank", "chosen_logp", "rel_pos")

#: Sign applied before computing a single-feature AUROC, so that "higher means
#: more likely to diverge" holds for each one.  The base model diverges where it
#: is *unsure*: high entropy, low top-1 probability, small margin, low chosen
#: log-probability.
AUROC_SIGNS = {
    "entropy": 1,
    "top1_prob": -1,
    "margin": -1,
    "chosen_logp": -1,
    "token_rank": 1,
    "rel_pos": 1,
}


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def base_features(
    base_logits: np.ndarray,
    chosen_id: int,
    position: int,
    total_length: int,
) -> Dict[str, float]:
    """The six features for one position.

    ``base_logits`` is the base model's raw next-token logits at this position.
    """
    logp = _log_softmax(np.asarray(base_logits, dtype=np.float64))
    p = np.exp(logp)
    order = np.argsort(-p)
    top1, top2 = p[order[0]], p[order[1]] if p.size > 1 else 0.0
    rank = int(np.nonzero(order == chosen_id)[0][0]) + 1
    return {
        "entropy": float(-(p * logp).sum()),
        "margin": float(top1 - top2),
        "top1_prob": float(top1),
        "token_rank": float(rank),
        "chosen_logp": float(logp[chosen_id]),
        "rel_pos": float(position / max(1, total_length)),
    }


def kl_rl_base(rl_logits: np.ndarray, base_logits: np.ndarray) -> float:
    """KL(RL || base) between the two next-token distributions.

    This is the quantity the ``kl_top10`` label thresholds, and the sense in
    which Appendix B's "high-KL positions" are defined.
    """
    lp_rl = _log_softmax(np.asarray(rl_logits, dtype=np.float64))
    lp_base = _log_softmax(np.asarray(base_logits, dtype=np.float64))
    p_rl = np.exp(lp_rl)
    return float((p_rl * (lp_rl - lp_base)).sum())


def kl_top_label(kl: Sequence[float], pct: float = 10.0):
    """Mark the top ``pct``% of positions by KL.  Returns ``(labels, threshold)``."""
    a = np.asarray(kl, dtype=np.float64)
    thr = float(np.percentile(a, 100.0 - pct))
    return a >= thr, thr


def mismatch_label(chosen_rank_in_rl: Sequence[int]) -> np.ndarray:
    """Positions where the RL model's top-1 differs from the sampled token."""
    return np.asarray(chosen_rank_in_rl, dtype=np.int64) != 1


def feature_matrix(records: Sequence[Dict[str, float]]) -> np.ndarray:
    """Stack per-position feature dicts into an ``(n, 6)`` matrix."""
    return np.stack(
        [np.asarray([r[f] for f in FEATURES], dtype=np.float32) for r in records], axis=0
    )


def disagreement_rate(base_top1: Sequence[int], rl_top1: Sequence[int]) -> float:
    """Fraction of positions where the two models' greedy tokens differ.

    This is the rl-replace criterion, reported as about 4% in §3.3.  Note it is
    neither of the Appendix B labels.
    """
    b = np.asarray(base_top1)
    r = np.asarray(rl_top1)
    if b.shape != r.shape:
        raise ValueError("aligned top-1 sequences required")
    return float((b != r).mean()) if b.size else float("nan")
