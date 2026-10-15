"""Test 3: inserting the RL model's reflective tokens by fixed rule (§3.5).

Test 3 asks whether the later replacements of Test 1 can be *scripted*.  The
rules are built from the tokens the RL model uses -- "Wait" and the prefix "Re"
-- and are applied to the base model **without consulting the RL model**.

Two families, both in Table 6:

* **Hard rules** force a token at a trigger position.  The triggers come from
  statistics of where the two models disagree, via the base-only predictor of
  Appendix B: the "Wait" rule fires when the base model emits a section header
  right after a formula block, and the "Re" rule uses an analogous trigger for
  the reflective prefix.
* **Soft rules** add a constant (+2 or +4) to the logit of "Re" instead of
  forcing it.

Every hard rule falls 5.5-7.7 points *below* the base model, and the soft rules
bring no improvement.  The damage is largest on pass@16, because a rule applies
the same edit at the same kind of position in all 16 samples and removes much
of their diversity.  This module exists to reproduce that negative result, so
the triggers are written to be inspectable rather than clever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_NEW_TOKENS,
    sample_from_logits,
    stop_token_ids,
)

#: A markdown section header the base model is about to start.
HEADER_RE = re.compile(r"(?:^|\n)\s*#{1,6}\s*$")
#: The tail of a display-formula block, allowing trailing whitespace.
FORMULA_END_RE = re.compile(r"(?:\$\$|\\\]|\\end\{(?:equation|align|aligned|gather)\*?\})\s*$")


@dataclass
class HardRule:
    """Force ``token`` when the decoded tail matches the trigger.

    ``after_formula`` requires a formula block to have just closed, which is
    what makes this the "section header right after a formula block" trigger of
    §3.5.  ``min_gap`` keeps a rule from firing twice in immediate succession.
    """

    token: str
    name: str
    after_formula: bool = True
    require_header: bool = True
    min_gap: int = 64
    max_fires: Optional[int] = None


@dataclass
class SoftRule:
    """Add ``bias`` to the logit of every token in ``token_strings``."""

    token_strings: Sequence[str]
    bias: float
    name: str = "soft"


@dataclass
class RuleConfig:
    hard: List[HardRule] = field(default_factory=list)
    soft: List[SoftRule] = field(default_factory=list)
    #: Force a token whenever it appears among the base model's top-k
    #: candidates -- the "Wait among top-5" variant of §3.5.
    force_if_in_topk: Optional[str] = None
    topk: int = 5
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    prefill: Optional[str] = None
    seed: Optional[int] = None


def wait_rule() -> HardRule:
    """The "Wait" hard rule of Table 6."""
    return HardRule(token="Wait", name="force_wait")


def re_rule() -> HardRule:
    """The "Re" hard rule: the same trigger for the reflective prefix."""
    return HardRule(token="Re", name="force_re")


def re_soft_rule(bias: float) -> SoftRule:
    """Soft rule: a constant added to the logit of "Re" (+2 or +4 in Table 6)."""
    return SoftRule(token_strings=("Re", " Re"), bias=bias, name="re_logit_plus{:g}".format(bias))


class RuleGuidedDecoder:
    """Single-model decoder with hard and soft rules over reflective tokens."""

    def __init__(self, model: Any, tokenizer: Any, config: Optional[RuleConfig] = None) -> None:
        self.model = model
        self.tok = tokenizer
        self.cfg = config or RuleConfig()
        self.model.eval()

    def _first_id(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("token string {!r} encodes to nothing".format(text))
        return int(ids[0])

    def _stop_ids(self) -> List[int]:
        return stop_token_ids(self.tok)

    @staticmethod
    def _trigger_fires(rule: HardRule, tail: str, next_piece: str) -> bool:
        """Whether ``rule`` fires given the decoded tail and the next token."""
        if rule.require_header:
            starts_header = HEADER_RE.search(tail + next_piece) is not None
            if not starts_header:
                return False
        if rule.after_formula:
            # Look at the tail *before* the header characters began.
            stripped = HEADER_RE.sub("", tail + next_piece)
            if FORMULA_END_RE.search(stripped) is None:
                return False
        return True

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_samples: int = 16,
        config: Optional[RuleConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses under the configured rules.

        Returns one dict per sample with ``text``, ``token_ids`` and
        ``fires`` (the positions at which each rule fired), so that the
        "all 282 responses that receive six or more insertions are wrong"
        statistic of §3.5 can be recomputed.
        """
        cfg = config or self.cfg
        device = next(self.model.parameters()).device
        if cfg.prefill:
            prompt = prompt + cfg.prefill

        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device).repeat(n_samples, 1)
        attn = torch.ones_like(input_ids)

        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(cfg.seed))

        soft_ids = [
            ([self._first_id(s) for s in r.token_strings], r.bias) for r in cfg.soft
        ]
        hard_ids = [(r, self._first_id(r.token)) for r in cfg.hard]
        topk_id = self._first_id(cfg.force_if_in_topk) if cfg.force_if_in_topk else None
        stop_ids = torch.tensor(self._stop_ids(), device=device)

        results = [
            {"text": "", "token_ids": [], "fires": []} for _ in range(n_samples)
        ]
        alive = torch.ones(n_samples, dtype=torch.bool, device=device)
        last_fire = [-10**9] * n_samples
        n_fires = [0] * n_samples

        past = None
        cur = input_ids
        for _ in range(cfg.max_new_tokens):
            out = self.model(
                input_ids=cur, attention_mask=attn, past_key_values=past, use_cache=True
            )
            logits = out.logits[:, -1, :].clone()
            past = out.past_key_values

            for ids, bias in soft_ids:
                logits[:, ids] += bias

            nxt = sample_from_logits(logits, cfg.temperature, cfg.top_p, generator=generator)

            if topk_id is not None:
                topk = logits.topk(cfg.topk, dim=-1).indices
                in_topk = (topk == topk_id).any(dim=-1)
                nxt = torch.where(in_topk & alive, torch.full_like(nxt, topk_id), nxt)

            if hard_ids:
                for i in range(n_samples):
                    if not bool(alive[i]):
                        continue
                    pos = len(results[i]["token_ids"])
                    tail = self.tok.decode(results[i]["token_ids"][-48:], skip_special_tokens=True)
                    piece = self.tok.decode([int(nxt[i])], skip_special_tokens=True)
                    for rule, tid in hard_ids:
                        if pos - last_fire[i] < rule.min_gap:
                            continue
                        if rule.max_fires is not None and n_fires[i] >= rule.max_fires:
                            continue
                        if self._trigger_fires(rule, tail, piece):
                            nxt[i] = tid
                            last_fire[i] = pos
                            n_fires[i] += 1
                            results[i]["fires"].append({"pos": pos, "rule": rule.name})
                            break

            for i in range(n_samples):
                if bool(alive[i]):
                    results[i]["token_ids"].append(int(nxt[i]))

            hit = (nxt.unsqueeze(-1) == stop_ids).any(dim=-1) if stop_ids.numel() else torch.zeros_like(alive)
            alive = alive & ~hit
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(-1)
            attn = torch.cat([attn, alive.long().unsqueeze(-1)], dim=-1)

        for r in results:
            r["text"] = self.tok.decode(r["token_ids"], skip_special_tokens=True)
            if cfg.prefill:
                r["text"] = cfg.prefill + r["text"]
        return results


def table6_configs(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, RuleConfig]:
    """The five rule conditions of Table 6, by condition name."""
    return {
        "force_wait": RuleConfig(hard=[wait_rule()], max_new_tokens=max_new_tokens),
        "force_re": RuleConfig(hard=[re_rule()], max_new_tokens=max_new_tokens),
        "force_wait_re": RuleConfig(hard=[wait_rule(), re_rule()], max_new_tokens=max_new_tokens),
        "re_logit_plus2": RuleConfig(soft=[re_soft_rule(2.0)], max_new_tokens=max_new_tokens),
        "re_logit_plus4": RuleConfig(soft=[re_soft_rule(4.0)], max_new_tokens=max_new_tokens),
    }
