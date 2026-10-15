"""Appendix B: train the gate and reproduce the AUROC table (Table 10).

Two models are fit on the six base-only features:

* a logistic regression, as a linear reference (0.946 / 0.958 under the KL
  label), and
* the **gate**, a gradient-boosted classifier (0.969 / 0.980), whose
  probability threshold at 90% recall is what ``gate-MCMC`` uses in §3.2.

The threshold is not a round number chosen by hand: it is the decision
threshold that comes out at 90% recall, which for the paper's run is 0.8567.
``precision_at_recall`` returns it alongside the precision, so a rerun on new
data reports its own threshold rather than inheriting that one.

Table 10 reports the gate per competition year (AIME24 / AIME25), i.e. fit on
one year's responses and scored on the other, so ``train_gate`` expects a
holdout and that is the configuration the paper's numbers come from. A 5-fold
cross-validated fit is available through ``cv_folds`` for the case where no year
is held out; it gives a different threshold, and the two should not be mixed
within one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .features import AUROC_SIGNS, FEATURES

RANDOM_STATE = 42
RECALL_TARGET = 0.9


@dataclass
class GateResult:
    auroc: float
    precision_at_recall: float
    threshold: float
    recall_target: float = RECALL_TARGET
    flagged_fraction: float = float("nan")


def precision_at_recall(
    y: Sequence[bool], scores: Sequence[float], target: float = RECALL_TARGET
) -> Tuple[float, float]:
    """Precision and decision threshold at the highest-precision point with
    recall >= ``target``.

    Returns ``(precision, threshold)``.
    """
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(np.asarray(y, dtype=int), np.asarray(scores, dtype=float))
    # r is decreasing; the last index meeting the target is the most precise one.
    idx = np.nonzero(r[:-1] >= target)[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    i = int(idx[-1])
    return float(p[i]), float(t[i])


def single_feature_auroc(X: np.ndarray, y: Sequence[bool]) -> Dict[str, float]:
    """AUROC of each feature on its own, with the sign convention applied."""
    from sklearn.metrics import roc_auc_score

    yy = np.asarray(y, dtype=int)
    out: Dict[str, float] = {}
    for j, name in enumerate(FEATURES):
        s = AUROC_SIGNS[name] * X[:, j].astype(np.float64)
        out[name] = float(roc_auc_score(yy, s))
    return out


def train_gate(
    X: np.ndarray,
    y: Sequence[bool],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[Sequence[bool]] = None,
    cv_folds: Optional[int] = None,
) -> Dict[str, GateResult]:
    """Fit the logistic baseline and the gradient-boosted gate.

    With ``X_test``/``y_test``, scores come from that holdout.  With
    ``cv_folds``, they come from cross-validated out-of-fold predictions on
    ``X`` instead, which is how the original analysis evaluated the gate when no
    separate year was held out.  Passing neither scores on the training data and
    will read optimistically -- it warns rather than silently reporting it.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=int)
    models = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "gate": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    results: Dict[str, GateResult] = {}
    for name, model in models.items():
        if cv_folds:
            cv = StratifiedKFold(cv_folds, shuffle=True, random_state=0)
            scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            y_eval = y
        else:
            model.fit(X, y)
            if X_test is None or y_test is None:
                import warnings

                warnings.warn(
                    "no holdout and no cv_folds: {} is being scored on its own "
                    "training data".format(name),
                    stacklevel=2,
                )
                scores, y_eval = model.predict_proba(X)[:, 1], y
            else:
                scores = model.predict_proba(X_test)[:, 1]
                y_eval = np.asarray(y_test, dtype=int)

        prec, thr = precision_at_recall(y_eval, scores)
        results[name] = GateResult(
            auroc=float(roc_auc_score(y_eval, scores)),
            precision_at_recall=prec,
            threshold=thr,
            flagged_fraction=float((scores >= thr).mean()) if np.isfinite(thr) else float("nan"),
        )
    return results


def auroc_table(
    X: np.ndarray,
    y_kl: Sequence[bool],
    y_mismatch: Optional[Sequence[bool]] = None,
    **kwargs,
) -> str:
    """Render Table 10: per-feature AUROC plus the two fitted models."""
    rows = []
    per_feat_kl = single_feature_auroc(X, y_kl)
    per_feat_mm = single_feature_auroc(X, y_mismatch) if y_mismatch is not None else {}
    fitted_kl = train_gate(X, y_kl, **kwargs)

    header = "{:<28}{:>16}{:>18}".format("", "Mismatch label", "KL label (top 10%)")
    rows.append(header)
    rows.append("-" * len(header))
    pretty = {
        "chosen_logp": "Chosen log-prob",
        "entropy": "Entropy",
        "top1_prob": "Top-1 probability",
        "margin": "Margin",
        "rel_pos": "Relative position",
        "token_rank": "Token rank",
    }
    for name in ("chosen_logp", "entropy", "top1_prob", "margin", "rel_pos", "token_rank"):
        mm = "{:.3f}".format(per_feat_mm[name]) if per_feat_mm else "-"
        rows.append(
            "{:<28}{:>16}{:>18.3f}".format(pretty[name], mm, per_feat_kl[name])
        )
    rows.append(
        "{:<28}{:>16}{:>18.3f}".format(
            "Logistic regression (all)", "-", fitted_kl["logistic"].auroc
        )
    )
    rows.append(
        "{:<28}{:>16}{:>18.3f}".format(
            "Gate (gradient boosting)", "-", fitted_kl["gate"].auroc
        )
    )
    g = fitted_kl["gate"]
    rows.append("")
    rows.append(
        "gate: precision {:.2f} at recall {:.0%}, decision threshold {:.4f}, "
        "flags {:.1%} of positions".format(
            g.precision_at_recall, g.recall_target, g.threshold, g.flagged_fraction
        )
    )
    return "\n".join(rows)
