"""Appendix B: locating high-KL positions from the base model alone."""

from .features import (
    AUROC_SIGNS,
    FEATURES,
    base_features,
    disagreement_rate,
    feature_matrix,
    kl_rl_base,
    kl_top_label,
    mismatch_label,
)
from .train import GateResult, auroc_table, precision_at_recall, single_feature_auroc, train_gate

__all__ = [
    "AUROC_SIGNS",
    "FEATURES",
    "base_features",
    "disagreement_rate",
    "feature_matrix",
    "kl_rl_base",
    "kl_top_label",
    "mismatch_label",
    "GateResult",
    "auroc_table",
    "precision_at_recall",
    "single_feature_auroc",
    "train_gate",
]
