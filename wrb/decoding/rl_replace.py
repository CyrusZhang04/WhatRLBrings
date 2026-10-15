"""Test 1: ``rl-replace``, the token-level hybrid decoder (paper §3.1, §3.3).

The decoding rule, verbatim from the paper:

    Let p_B and p_R denote the base and RL models.  At each step t with prefix
    y_<t, we compute the two greedy tokens b_t = argmax p_B(. | y_<t) and
    r_t = argmax p_R(. | y_<t).  If b_t != r_t we emit r_t; otherwise we sample
    y_t ~ p_B(. | y_<t) at temperature 0.6.

So the RL model only intervenes at *disagreement positions* -- about 4% of
tokens -- and the base model writes everything else.  Both models score the
same shared prefix at every step, which is why this costs two forward passes
per token.

The same loop serves the Test 2 control in Table 4 ("base greedy at
disagreements"): with ``at_disagreement="base_greedy"`` the RL model is used
only to *locate* the positions, and the base model's own greedy token is
emitted there.  That control separates "committing to a token here" from
"using the RL model's choice here" -- 1.5 of the 5.2 points, per §3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_NEW_TOKENS,
    greedy_from_logits,
    sample_from_logits,
    stop_token_ids,
)


@dataclass
class ReplaceConfig:
    """Decoding configuration for one rl-replace run."""

    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    #: "rl" emits the RL model's greedy token at disagreement positions
    #: (Test 1); "base_greedy" emits the base model's own (Table 4 control).
    at_disagreement: str = "rl"
    #: Text prefilled into the assistant turn before decoding starts; used when
    #: combining rl-replace with the forced opening of Test 2.
    prefill: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.at_disagreement not in ("rl", "base_greedy"):
            raise ValueError(
                "at_disagreement must be 'rl' or 'base_greedy', got {!r}".format(
                    self.at_disagreement
                )
            )


@dataclass
class ReplaceOutput:
    """One generated response and its replacement record."""

    text: str
    token_ids: List[int] = field(default_factory=list)
    #: One entry per position where the two models' greedy tokens differed.
    replacements: List[Dict[str, int]] = field(default_factory=list)
    finished: bool = False

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def replace_rate(self) -> float:
        """Emitted RL tokens divided by response length (§4.1)."""
        if not self.token_ids:
            return 0.0
        return len(self.replacements) / len(self.token_ids)


class RLReplaceDecoder:
    """Two-model hybrid decoder.

    Both models must share a tokenizer: the loop compares token ids from the
    two vocabularies directly, so a base/RL pair with different tokenizers
    cannot be combined this way.  The paper's three pairs are each an RL model
    trained from its own base, so they always share one.

    Example
    -------
    >>> decoder = RLReplaceDecoder(base_model, rl_model, tokenizer)  # doctest: +SKIP
    >>> outs = decoder.generate(prompt, n_samples=16)                # doctest: +SKIP
    """

    def __init__(
        self,
        base_model: Any,
        rl_model: Any,
        tokenizer: Any,
        config: Optional[ReplaceConfig] = None,
    ) -> None:
        self.base = base_model
        self.rl = rl_model
        self.tok = tokenizer
        self.cfg = config or ReplaceConfig()
        self.base.eval()
        self.rl.eval()
        self._check_shared_vocab()

    def _check_shared_vocab(self) -> None:
        bv = getattr(self.base.config, "vocab_size", None)
        rv = getattr(self.rl.config, "vocab_size", None)
        if bv is not None and rv is not None and bv != rv:
            raise ValueError(
                "base and RL models must share a tokenizer; vocab sizes differ "
                "({} vs {})".format(bv, rv)
            )

    def _stop_ids(self) -> List[int]:
        return stop_token_ids(self.tok)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_samples: int = 16,
        config: Optional[ReplaceConfig] = None,
    ) -> List[ReplaceOutput]:
        """Generate ``n_samples`` responses to one prompt.

        The samples are decoded as a batch: the prompt is identical across the
        batch, and they diverge as soon as the base model's sampling differs.
        """
        cfg = config or self.cfg
        device = next(self.base.parameters()).device
        if cfg.prefill:
            prompt = prompt + cfg.prefill

        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device).repeat(n_samples, 1)
        attn = torch.ones_like(input_ids)

        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(cfg.seed))

        stop_ids = torch.tensor(self._stop_ids(), device=device)
        outs = [ReplaceOutput(text="") for _ in range(n_samples)]
        alive = torch.ones(n_samples, dtype=torch.bool, device=device)

        base_past = rl_past = None
        cur = input_ids
        for step in range(cfg.max_new_tokens):
            base_logits, base_past = self._forward(self.base, cur, attn, base_past)
            rl_logits, rl_past = self._forward(self.rl, cur, attn, rl_past)

            b_tok = greedy_from_logits(base_logits)
            r_tok = greedy_from_logits(rl_logits)
            disagree = b_tok != r_tok

            sampled = sample_from_logits(
                base_logits, cfg.temperature, cfg.top_p, generator=generator
            )
            forced = r_tok if cfg.at_disagreement == "rl" else b_tok
            next_tok = torch.where(disagree, forced, sampled)

            # Finished sequences keep emitting a pad-ish token that is never
            # recorded, so the batch stays rectangular for the KV cache.
            next_tok = torch.where(alive, next_tok, b_tok)

            for i in range(n_samples):
                if not bool(alive[i]):
                    continue
                tid = int(next_tok[i])
                outs[i].token_ids.append(tid)
                if bool(disagree[i]) and cfg.at_disagreement == "rl":
                    outs[i].replacements.append(
                        {
                            "pos": len(outs[i].token_ids) - 1,
                            "base_id": int(b_tok[i]),
                            "rl_id": int(r_tok[i]),
                        }
                    )

            hit_stop = (next_tok.unsqueeze(-1) == stop_ids).any(dim=-1) if stop_ids.numel() else torch.zeros_like(alive)
            newly_done = alive & hit_stop
            for i in torch.nonzero(newly_done).flatten().tolist():
                outs[i].finished = True
            alive = alive & ~hit_stop
            if not bool(alive.any()):
                break

            cur = next_tok.unsqueeze(-1)
            attn = torch.cat([attn, alive.long().unsqueeze(-1)], dim=-1)

        for out in outs:
            out.text = self.tok.decode(out.token_ids, skip_special_tokens=True)
        return outs

    @staticmethod
    def _forward(model: Any, tokens: torch.Tensor, attn: torch.Tensor, past: Any):
        """One incremental forward pass; returns last-position logits and cache."""
        out = model(
            input_ids=tokens,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
        )
        return out.logits[:, -1, :], out.past_key_values


def replacement_stats(outputs: Sequence[ReplaceOutput]) -> Dict[str, float]:
    """Summary of how sparse the intervention was.

    ``mean_replace_rate_per_response`` is the per-response average the paper
    quotes as 4.5%; ``token_weighted_replace_rate`` pools all tokens first and
    gives the 4.0% figure.  They differ because longer responses have slightly
    lower rates, so which one is reported matters.
    """
    if not outputs:
        return {}
    n_tok = sum(o.length for o in outputs)
    n_rep = sum(len(o.replacements) for o in outputs)
    per_resp = [o.replace_rate for o in outputs if o.length]
    return {
        "n_responses": float(len(outputs)),
        "n_tokens": float(n_tok),
        "n_replacements": float(n_rep),
        "token_weighted_replace_rate": (n_rep / n_tok) if n_tok else float("nan"),
        "mean_replace_rate_per_response": (
            sum(per_resp) / len(per_resp) if per_resp else float("nan")
        ),
    }
