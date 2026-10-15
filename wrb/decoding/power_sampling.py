"""Test 0: is the gain a sharper base distribution? (paper 搂3.2, Appendix A)

Test 0 borrows nothing from the RL model.  It compares three ways of decoding
from the base model alone -- greedy, standard sampling, and power sampling with
MCMC -- plus a gated variant that restricts the sampler to positions the
Appendix B gate marks as likely high-KL positions.

**The MCMC kernel is not vendored here.**  The paper uses the power-sampling
MCMC of Karan and Du (arXiv:2510.14901) with the settings of the original
paper: target distribution p_base^alpha with alpha = 4, 16 blocks, 10 MCMC
steps per block, and the same 8,192-token budget as the RL model.  This module
defines the backend interface and keeps the paper's settings in one place.
Supply the upstream MCMC implementation as ``backend`` to run power sampling
for Test 0.

What *is* implemented here: the greedy and standard-sampling baselines, and the
gate restriction of ``gate_mcmc``, which is this paper's own contribution to
Test 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import torch

from .sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_NEW_TOKENS

#: Settings of the original power-sampling paper, used unchanged (搂3.2).
ALPHA = 4
N_BLOCKS = 16
MCMC_STEPS_PER_BLOCK = 10
#: Gate score at or above which gate-MCMC resamples (Appendix A, "Gate-MCMC").
#: This is the probability threshold at 90% recall from the Appendix B gate.
GATE_THRESHOLD = 0.8567
#: Power sampling and standard sampling are averaged over these seeds; greedy
#: decoding is run once (Appendix A).
SEEDS = tuple(range(8))


@dataclass
class PowerSamplingConfig:
    alpha: int = ALPHA
    n_blocks: int = N_BLOCKS
    mcmc_steps_per_block: int = MCMC_STEPS_PER_BLOCK
    max_new_tokens: int = MAX_NEW_TOKENS
    #: When set, resample only where the gate scores at least this much.
    gate_threshold: Optional[float] = None
    seed: int = 0


class PowerSamplerBackend:
    """Interface a power-sampling implementation must provide.

    Implement ``sample`` by wrapping the upstream MCMC sampler.  ``gate_scores``
    is supplied for the gated variant: one score per candidate position, from
    :mod:`wrb.gate`, and the backend should restrict its resampling to
    positions scoring at least ``config.gate_threshold``.
    """

    def sample(
        self,
        prompt: str,
        n_samples: int,
        config: PowerSamplingConfig,
        gate_scores: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> List[str]:
        raise NotImplementedError(
            "plug in the power-sampling MCMC of Karan and Du (arXiv:2510.14901); "
            "provide a backend implementing PowerSamplerBackend.sample"
        )


@torch.no_grad()
def generate_greedy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy decoding from the base model: a single run, no seeds (Table 2)."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def generate_standard(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_samples: int = 16,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: Optional[int] = None,
) -> List[str]:
    """Standard sampling from the base model (temperature 0.6, top-p 0.95)."""
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(int(seed))
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(
        out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
    )
