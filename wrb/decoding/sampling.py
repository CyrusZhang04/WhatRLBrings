"""Token sampling shared by the interventions.

Temperature 0.6 with top-p 0.95 are the paper's defaults (§4.1); they are
parameters here so that the greedy and low-temperature controls of §3.5 use the
same code path.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from ..constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, MAX_NEW_TOKENS  # noqa: F401

#: End-of-text markers to stop on, across the model families in the paper.
#: Which of these a checkpoint actually has is checked, not assumed.
STOP_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")


def stop_token_ids(tokenizer: Any) -> List[int]:
    """Token ids that end a response, for this tokenizer only.

    ``convert_tokens_to_ids`` returns the *unknown* token's id for a string the
    tokenizer does not have, rather than ``None``, so a candidate is accepted
    only when decoding its id gives the string back.  Without that check, a
    Llama tokenizer asked for ``<|im_end|>`` would contribute the unk id as a
    stop token and every response would end at its first unknown token.
    """
    ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.add(int(eos))
    for text in STOP_TOKEN_STRINGS:
        tid = tokenizer.convert_tokens_to_ids(text)
        if tid is None or tid < 0:
            continue
        if tokenizer.convert_ids_to_tokens(int(tid)) == text:
            ids.add(int(tid))
    return sorted(ids)


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the tail beyond nucleus ``top_p`` and renormalize.

    ``probs`` is ``(batch, vocab)``.  The token that crosses the threshold is
    kept, so the retained mass is always >= ``top_p``.
    """
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Drop everything strictly after the first index where the cumulative mass
    # reaches top_p.
    drop = cumulative - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token per row from ``(batch, vocab)`` logits.

    ``temperature <= 0`` means greedy decoding.
    """
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    probs = apply_top_p(probs, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def greedy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Argmax token per row, i.e. each model's preferred next token."""
    return logits.argmax(dim=-1)
