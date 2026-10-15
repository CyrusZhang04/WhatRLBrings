"""Decoders for the four diagnostic tests.

    Test 0  power_sampling   sharper base distribution (greedy / sampling / MCMC)
    Test 1  rl_replace       borrow the RL token at disagreement positions
    Test 2  prefill          force the RL model's opening word
    Test 3  rules            insert reflective tokens by fixed rule

All of them generate with the *base* model.  Only Test 1 runs the RL model at
decoding time; Tests 2 and 3 reuse tokens taken from its outputs.
"""

from .prefill import ALRIGHT, ALRIGHT_I, PrefillConfig, generate_with_prefill
from .rl_replace import ReplaceConfig, ReplaceOutput, RLReplaceDecoder, replacement_stats
from .rules import RuleConfig, RuleGuidedDecoder, table6_configs
from .sampling import greedy_from_logits, sample_from_logits

__all__ = [
    "ALRIGHT",
    "ALRIGHT_I",
    "PrefillConfig",
    "generate_with_prefill",
    "ReplaceConfig",
    "ReplaceOutput",
    "RLReplaceDecoder",
    "replacement_stats",
    "RuleConfig",
    "RuleGuidedDecoder",
    "table6_configs",
    "greedy_from_logits",
    "sample_from_logits",
]
