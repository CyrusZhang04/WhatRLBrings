"""Response-level statistics: answer rate, length, repetition, diversity.

These back three kinds of claim in the paper:

* 搂3.2, that power sampling *degrades* long generations -- the fraction of AIME
  responses with no final answer rises from 12.2% to 22.4%, and the fraction
  ending in repetition from 6.7% to 13.1%;
* 搂3.6, the diversity comparison (distinct openings and distinct final answers
  per problem);
* 搂4.1, the replace rate of rl-replace.

The degradation flags follow the original definition: a response is flagged on
length when it is shorter than 50 tokens or exactly at the generation limit, on
answer when it contains no ``\\boxed``, and on repetition when a short string
repeats at the end.  ``max_tokens`` must be passed in, because "exactly at the
limit" depends on the budget the run used.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .. import records as rec_io
from ..constants import MAX_NEW_TOKENS
from ..grading import ends_in_repetition, last_boxed, normalize_answer

#: A string of 1-3 characters repeated 26 or more times: the degenerate tail.
SHORT_REPEAT_RE = re.compile(r"(.{1,3}?)\1{25,}", re.S)
MIN_USEFUL_LENGTH = 50
REPEAT_TAIL_CHARS = 3000


def degradation_flags(
    text: str, length: Optional[int], max_tokens: int
) -> Dict[str, bool]:
    """The three flags, per response.

    ``length`` may be ``None`` when a run did not record token counts, in which
    case the length flag is simply ``False``: an unknown length is not evidence
    of a degenerate response.
    """
    return {
        "flag_length": (
            False
            if length is None
            else bool(length < MIN_USEFUL_LENGTH or length == max_tokens)
        ),
        "flag_no_answer": last_boxed(text) is None,
        "flag_repetition": SHORT_REPEAT_RE.search(text[-REPEAT_TAIL_CHARS:]) is not None
        or ends_in_repetition(text),
    }


@dataclass
class ConditionStats:
    """Accumulated per-sample statistics for one condition."""

    name: str
    max_tokens: int = MAX_NEW_TOKENS
    n: int = 0
    n_correct: int = 0
    n_no_answer: int = 0
    n_repetition: int = 0
    n_at_limit: int = 0
    lengths: List[int] = field(default_factory=list)
    replace_rates: List[float] = field(default_factory=list)
    #: problem -> response texts / answers, for the diversity metrics
    texts: Dict[str, List[str]] = field(default_factory=lambda: collections.defaultdict(list))
    answers: Dict[str, List[Optional[str]]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )
    #: pass@1 among responses that produced an answer at all
    n_answered: int = 0
    n_answered_correct: int = 0

    def add(
        self,
        problem: str,
        text: str,
        correct: bool,
        length: Optional[int] = None,
        replace_rate: Optional[float] = None,
    ) -> None:
        self.n += 1
        self.n_correct += int(bool(correct))
        flags = degradation_flags(text, length, self.max_tokens)
        answered = not flags["flag_no_answer"]
        self.n_no_answer += int(not answered)
        self.n_repetition += int(flags["flag_repetition"])
        if answered:
            self.n_answered += 1
            self.n_answered_correct += int(bool(correct))
        if length is not None:
            self.lengths.append(int(length))
            self.n_at_limit += int(length == self.max_tokens)
        if replace_rate is not None:
            self.replace_rates.append(float(replace_rate))
        self.texts[problem].append(text)
        self.answers[problem].append(normalize_answer(last_boxed(text)))

    def summary(self) -> Dict[str, Any]:
        import numpy as np

        from ..metrics import distinct_answers, distinct_openings

        L = np.asarray(self.lengths, dtype=float) if self.lengths else np.asarray([])
        rr = np.asarray(self.replace_rates, dtype=float) if self.replace_rates else np.asarray([])
        openings = [distinct_openings(v) for v in self.texts.values()]
        answers = [distinct_answers(v) for v in self.answers.values()]
        out: Dict[str, Any] = {
            "condition": self.name,
            "n_samples": self.n,
            "n_problems": len(self.texts),
            "pass1_pct": 100.0 * self.n_correct / self.n if self.n else float("nan"),
            "no_answer_pct": 100.0 * self.n_no_answer / self.n if self.n else float("nan"),
            "repetition_pct": 100.0 * self.n_repetition / self.n if self.n else float("nan"),
            "pass1_answered_only_pct": (
                100.0 * self.n_answered_correct / self.n_answered
                if self.n_answered
                else float("nan")
            ),
        }
        if L.size:
            out.update(
                {
                    "len_mean": float(L.mean()),
                    "len_median": float(np.median(L)),
                    "len_p90": float(np.percentile(L, 90)),
                    "at_limit_pct": 100.0 * self.n_at_limit / self.n,
                }
            )
        if rr.size:
            out.update(
                {
                    "replace_rate_mean_pct": 100.0 * float(rr.mean()),
                    "replace_rate_median_pct": 100.0 * float(np.median(rr)),
                }
            )
        if openings:
            out["distinct_openings_mean"] = float(sum(openings) / len(openings))
        if answers:
            out["distinct_answers_mean"] = float(sum(answers) / len(answers))
        return out


def collect(
    run_dir: str,
    max_tokens: int = MAX_NEW_TOKENS,
    conditions: Optional[Sequence[str]] = None,
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, ConditionStats]:
    """Walk a run directory and accumulate statistics per condition."""
    stats: Dict[str, ConditionStats] = {}
    for raw in rec_io.read_run(run_dir):
        if splits and raw["ds"] not in splits:
            continue
        pid = rec_io.problem_id(raw)
        for cond in rec_io.condition_names(raw):
            if conditions and cond not in conditions:
                continue
            blk = raw[cond]
            st = stats.setdefault(cond, ConditionStats(cond, max_tokens=max_tokens))
            n = len(blk["texts"])
            rates = blk.get("replace_rates") or [None] * n
            lengths = blk.get("lengths") or [None] * n
            for text, acc, length, rr in zip(blk["texts"], blk["accs"], lengths, rates):
                st.add(pid, text, bool(acc), length, rr)
    return stats


def render(stats: Dict[str, ConditionStats], keys: Optional[Sequence[str]] = None) -> str:
    """Render the summaries as a text table."""
    from .tables import render as render_rows

    rows = [st.summary() for st in stats.values()]
    if keys:
        rows = [{k: r.get(k, "") for k in keys} for r in rows]
    else:
        rows = [
            {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in r.items()
            }
            for r in rows
        ]
    return render_rows(rows)
