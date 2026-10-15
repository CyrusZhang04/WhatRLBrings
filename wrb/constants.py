"""Decoding and evaluation constants shared by every condition (paper §4.1).

Kept in a module with no third-party imports, so that the analysis layer can
use them without pulling in torch.
"""

#: Temperature for standard sampling, used by every condition that samples.
DEFAULT_TEMPERATURE = 0.6
#: Nucleus cutoff.
DEFAULT_TOP_P = 0.95
#: The generation limit used throughout the paper: every model and every
#: intervention generates at most this many tokens, power sampling included.
#: Single-sourced here because the no-final-answer rate and the
#: fraction-at-the-limit statistic are both defined relative to it.
MAX_NEW_TOKENS = 8192
#: Samples per problem on AIME; MATH-500 uses 64 and reports pass@16 with the
#: unbiased estimator.
DEFAULT_N_SAMPLES = 16
