"""Answer extraction and grading (paper §4.1, "Prompting and decoding").

A response is correct when the content of its last ``\\boxed{}`` is equivalent
to the reference answer according to Math-Verify.  Two other properties of a
response are also measured here because §3.2 reports them for power sampling:
whether it produced a final answer at all, and whether it ended in repetition.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

BOXED = "\\boxed"


def last_boxed(text: str) -> Optional[str]:
    """Content of the last complete ``\\boxed{...}`` in ``text``.

    Returns ``None`` when there is no ``\\boxed`` at all, and also when the
    response was cut off inside one (an unbalanced brace), which is the common
    case at the generation limit.
    """
    i = text.rfind(BOXED)
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def has_final_answer(text: str) -> bool:
    """Whether the response produced a usable final answer.

    §3.2 reports the complement of this ("the fraction of AIME responses with
    no final answer rises from 12.2% to 22.4%" under power sampling).
    """
    return last_boxed(text) is not None


_WS = re.compile(r"\s+")


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Light surface normalization, for *grouping* answers, not for grading.

    Used by ``maj@k`` and by the distinct-answer diversity counts, where the
    cost of an exact-string comparison is over-counting cosmetic variants.
    Grading itself goes through Math-Verify (``is_correct``), which is far more
    permissive than this.
    """
    if ans is None:
        return None
    out = ans
    for s in ("\\left", "\\right", "\\!", "\\,", "\\;", "$", "{", "}", " "):
        out = out.replace(s, "")
    out = re.sub(r"^\\text(bf|rm|it)?", "", out)
    out = out.rstrip(".").strip()
    return out or None


class Grader:
    """Math-Verify grading with a cached parse of the reference answer.

    Math-Verify is imported lazily so that the analysis modules, which only
    read already-graded records, stay importable without it.
    """

    def __init__(self) -> None:
        from math_verify import parse, verify  # noqa: F401  (import check)

        self._parse = parse
        self._verify = verify
        self._gold_cache: dict = {}

    def _gold(self, reference: str):
        key = str(reference)
        if key not in self._gold_cache:
            self._gold_cache[key] = self._parse("\\boxed{%s}" % key)
        return self._gold_cache[key]

    def is_correct(self, text: str, reference: str) -> bool:
        """Whether ``text``'s last boxed answer matches ``reference``."""
        boxed = last_boxed(text)
        if boxed is None:
            return False
        try:
            return bool(self._verify(self._gold(reference), self._parse("\\boxed{%s}" % boxed)))
        except Exception:
            # Math-Verify raises on a few malformed expressions; a response we
            # cannot parse is not a correct response.
            return False

    def grade_many(self, texts: Sequence[str], reference: str) -> List[bool]:
        return [self.is_correct(t, reference) for t in texts]


def ends_in_repetition(
    text: str,
    tail_chars: int = 400,
    min_period: int = 8,
    max_period: int = 120,
    min_repeats: int = 4,
) -> bool:
    """Whether the response ends in a repeating loop.

    §3.2 reports this for power sampling ("the fraction ending in repetition
    rises from 6.7% to 13.1%").  The rule: some substring of length
    ``min_period..max_period`` tiles the last ``tail_chars`` characters at
    least ``min_repeats`` times.  Degenerate loops at the generation limit are
    what this is meant to catch, not genuine repeated structure earlier in a
    solution.
    """
    tail = _WS.sub(" ", text[-tail_chars:])
    n = len(tail)
    if n < min_period * min_repeats:
        return False
    for period in range(min_period, min(max_period, n // min_repeats) + 1):
        unit = tail[-period:]
        repeats = 1
        pos = n - period
        while pos - period >= 0 and tail[pos - period : pos] == unit:
            repeats += 1
            pos -= period
            if repeats >= min_repeats:
                return True
    return False
