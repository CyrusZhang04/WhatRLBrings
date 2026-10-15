"""What RL Brings to Reasoning: diagnostics for what RL adds to a base model.

Four interventions on the base model, each motivated by one explanation of the
RL gain, all measured by the same recovery rate on the same problems:

    Test 0  does sharpening the base distribution reach the gain?  (power sampling)
    Test 1  is the gain localized?                                 (rl-replace)
    Test 2  is the localized gain an opening cue?                  (force "Alright")
    Test 3  can the later choices be scripted?                     (fixed rules)

See ``docs/paper_map.md`` for the modules associated with each experiment,
table, and metric.
"""

__version__ = "0.1.0"

from . import analysis, constants, grading, metrics, prompts, records
from .constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_NEW_TOKENS

__all__ = [
    "analysis",
    "constants",
    "grading",
    "metrics",
    "prompts",
    "records",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "MAX_NEW_TOKENS",
    "__version__",
]
