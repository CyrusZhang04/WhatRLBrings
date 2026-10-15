"""Test 2: forcing the RL model's opening (paper §3.4).

Every one of the 1,920 rl-replace responses starts with "Alright", because the
RL model always opens with "Alright, I have this problem..." and its first
token never matches the base model's.  Test 2 asks how much of the localized
gain that opening accounts for, and it is measured by ablation rather than by
frequency: the base model's response is forced to begin with the prefix, and it
then generates the rest **without any help from the RL model**.

The RL model is never loaded here.  That is the point of the test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from .sampling import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_NEW_TOKENS

#: The two prefixes in Table 4.  "Alright, I" extends the forced span by one
#: word and adds nothing further on pass@1.
ALRIGHT = "Alright"
ALRIGHT_I = "Alright, I"


@dataclass
class PrefillConfig:
    prefill: str = ALRIGHT
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: Optional[int] = None


@torch.no_grad()
def generate_with_prefill(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_samples: int = 16,
    config: Optional[PrefillConfig] = None,
) -> List[str]:
    """Sample ``n_samples`` continuations of ``prompt + prefill``.

    The returned texts **include** the forced prefix, so that length and
    opening statistics are comparable with the other conditions: a reader
    counting distinct openings should see the forced prefix collapse them.
    """
    cfg = config or PrefillConfig()
    device = next(model.parameters()).device
    full_prompt = prompt + cfg.prefill
    enc = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = enc["input_ids"].shape[1]

    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))

    sampling = cfg.temperature > 0
    if not sampling and n_samples != 1:
        raise ValueError(
            "greedy decoding is deterministic, so n_samples must be 1; got {}".format(n_samples)
        )
    kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "num_return_sequences": n_samples,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "do_sample": sampling,
    }
    if sampling:
        kwargs["temperature"] = cfg.temperature
        kwargs["top_p"] = cfg.top_p

    out = model.generate(**enc, **kwargs)
    continuations = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return [cfg.prefill + c for c in continuations]
