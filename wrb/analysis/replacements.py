"""Table 5: what the replacements actually are, and where they fall (§3.5).

Scope, as in the paper: the 38 successful rl-replace responses on the 19
problems where the base model fails all 16 samples.  Of their replacements,
only those that **change the text** are counted (9,099 of them); the rest
differ only in tokenization -- the same string split into different tokens --
and are not edits in any meaningful sense.

Categories are assigned by a precedence ladder, in the order the table lists
them from most to least specific.  Two of them are *bidirectional*: a
replacement counts as LaTeX formatting or as a first-person pronoun if either
side is one, because the interesting event is the category boundary being
crossed, in either direction.  Precedence is why "Correct -> Re" lands in the
"Re-" prefix row and not in "Other content words".
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

#: Table 5 category labels, in report order.
CATEGORIES = (
    "other_content_words",
    "latex_formatting",
    "first_person_pronouns",
    "numbers",
    "opening_and_pause_words",
    "markdown_headers",
    "punctuation",
    "re_prefix",
    "others",
)

PRETTY = {
    "other_content_words": "Other content words (paraphrase, wording)",
    "latex_formatting": "LaTeX formatting (both directions)",
    "first_person_pronouns": "First-person pronouns (both directions)",
    "numbers": "Numbers",
    "opening_and_pause_words": "Opening and pause words (-> Alright, So -> Wait)",
    "markdown_headers": "Markdown headers (Let -> ###)",
    "punctuation": "Punctuation",
    "re_prefix": '"Re-" prefix (Correct -> Re)',
    "others": "Others",
}

#: Words that open a response or mark a pause in it.  "Wait" and "Alright" are
#: the two the paper singles out; the rest are the same move in other words.
OPENING_AND_PAUSE = frozenset(
    {
        "alright",
        "wait",
        "hmm",
        "hold",
        "actually",
        "okay",
        "ok",
        "well",
        "hmmm",
        "oh",
    }
)

FIRST_PERSON = frozenset({"i", "we", "my", "our", "me", "us", "i'm", "we're", "i've", "we've"})

#: The reflective prefix: "Re" as in re-checking a step.  Matched as a token,
#: not as a substring, so "Real" and "Return" do not count.
RE_PREFIX_RE = re.compile(r"^\s*Re$")

MARKDOWN_RE = re.compile(r"(#{1,6}|\*\*|^\s*[-*]\s+$)")
LATEX_RE = re.compile(r"[\\$^_{}]|\\(?:frac|boxed|begin|end|left|right|times|cdot)")
NUMBER_RE = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")
PUNCT_RE = re.compile(r"^\s*[\.\,\:\;\!\?\)\(\]\[\}\{\-—\"'`]+\s*$")
WORD_RE = re.compile(r"^\s*[A-Za-z][A-Za-z'\-]*\s*$")
WHITESPACE_RE = re.compile(r"^\s*$")


@dataclass
class Replacement:
    """One replacement, decoded to text on both sides."""

    pos: int
    base_text: str
    rl_text: str
    rel_pos: float = float("nan")

    @property
    def changes_text(self) -> bool:
        """Whether this is an edit rather than a re-tokenization."""
        return self.base_text != self.rl_text


def _word(s: str) -> str:
    return s.strip().strip(".,:;!?()[]{}\"'`").lower()


def _specific_category(token: str) -> Optional[str]:
    """The category a single token belongs to, if it is a distinctive one."""
    w = _word(token)
    if RE_PREFIX_RE.match(token):
        return "re_prefix"
    if MARKDOWN_RE.search(token.strip()):
        return "markdown_headers"
    if w in OPENING_AND_PAUSE:
        return "opening_and_pause_words"
    if w in FIRST_PERSON:
        return "first_person_pronouns"
    if NUMBER_RE.match(token):
        return "numbers"
    return None


def categorize(base_text: str, rl_text: str) -> str:
    """Assign one Table 5 category to a replacement.

    The category follows **the token the RL model put in**, which is what every
    example in the table reads from: "-> Alright", "So -> Wait", "Let -> ###",
    "Correct -> Re".  Only when the RL token is unremarkable does the token it
    displaced decide, so that "### -> the" still counts as a markdown header
    while "### -> Wait" counts as a pause word -- the latter being exactly the
    insertion §3.5 describes, the RL model pausing where the base model was
    about to start a new section.

    LaTeX formatting and first-person pronouns are bidirectional, as the table
    notes: either side puts the replacement in that row.
    """
    for token in (rl_text, base_text):
        cat = _specific_category(token)
        if cat is not None:
            return cat
    if LATEX_RE.search(rl_text) or LATEX_RE.search(base_text):
        return "latex_formatting"
    if PUNCT_RE.match(rl_text) or PUNCT_RE.match(base_text):
        return "punctuation"
    if WORD_RE.match(rl_text) and WORD_RE.match(base_text):
        return "other_content_words"
    return "others"


def decode_replacements(
    raw: Iterable[Dict[str, int]],
    tokenizer: Any,
    response_length: Optional[int] = None,
) -> List[Replacement]:
    """Decode ``{pos, base_id, rl_id}`` records into text pairs."""
    out: List[Replacement] = []
    for rec in raw:
        pos = int(rec["pos"])
        out.append(
            Replacement(
                pos=pos,
                base_text=tokenizer.decode([int(rec["base_id"])]),
                rl_text=tokenizer.decode([int(rec["rl_id"])]),
                rel_pos=(pos / response_length) if response_length else float("nan"),
            )
        )
    return out


def table5(replacements: Sequence[Replacement]) -> Dict[str, Any]:
    """Counts and shares per category, over text-changing replacements only."""
    edits = [r for r in replacements if r.changes_text]
    counts = collections.Counter(categorize(r.base_text, r.rl_text) for r in edits)
    total = sum(counts.values())
    rows = [
        {
            "category": c,
            "label": PRETTY[c],
            "count": counts.get(c, 0),
            "share_pct": (100.0 * counts.get(c, 0) / total) if total else float("nan"),
        }
        for c in CATEGORIES
    ]
    return {
        "rows": rows,
        "total": total,
        "n_replacements_all": len(replacements),
        "n_tokenization_only": len(replacements) - len(edits),
    }


def positional_density(
    replacements: Sequence[Replacement], n_bins: int = 10
) -> List[Dict[str, float]]:
    """Share of replacements per relative-position bin.

    §3.5: densest in the first 20% of the response (26.1% of all
    replacements) and sparsest around 60-70% (7.8%) -- they cluster where the
    solution direction is set and thin out during routine computation.
    """
    edits = [r for r in replacements if r.changes_text and r.rel_pos == r.rel_pos]
    counts = [0] * n_bins
    for r in edits:
        b = min(n_bins - 1, int(n_bins * r.rel_pos))
        counts[b] += 1
    total = sum(counts) or 1
    return [
        {
            "bin_lo": i / n_bins,
            "bin_hi": (i + 1) / n_bins,
            "count": counts[i],
            "share_pct": 100.0 * counts[i] / total,
        }
        for i in range(n_bins)
    ]


def recurring_pairs(
    replacements: Sequence[Replacement], top: int = 40
) -> List[Tuple[str, str, int]]:
    """Most frequent ``(base -> rl)`` pairs.

    The patterns §3.5 calls out: So -> Wait, But -> Wait, ### -> Wait,
    Let -> ###, But -> ###, So -> But.
    """
    c = collections.Counter(
        (r.base_text.strip(), r.rl_text.strip()) for r in replacements if r.changes_text
    )
    return [(b, r, n) for (b, r), n in c.most_common(top)]


def render_table5(summary: Dict[str, Any]) -> str:
    """Render Table 5 as text."""
    lines = ["{:<46}{:>8}{:>12}".format("Category", "Count", "Share (%)")]
    lines.append("-" * 66)
    for row in summary["rows"]:
        lines.append(
            "{:<46}{:>8}{:>12.1f}".format(row["label"], row["count"], row["share_pct"])
        )
    lines.append("-" * 66)
    lines.append("{:<46}{:>8}{:>12.1f}".format("Total", summary["total"], 100.0))
    if summary["n_tokenization_only"]:
        lines.append(
            "\n({} further replacements change only tokenization and are "
            "excluded)".format(summary["n_tokenization_only"])
        )
    return "\n".join(lines)
