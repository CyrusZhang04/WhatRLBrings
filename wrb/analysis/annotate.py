"""Appendix E.1: export a response with its replacement positions marked.

Writes a markdown file in which every token the RL model supplied is bold, so
a reader can see what 4% of the tokens looks like in place, and a TSV with one
row per replacement and its surrounding context.

The alignment check matters.  ``replacements`` records token ids at generated
positions, but the exported text is re-tokenized to find character offsets, and
re-tokenization is not guaranteed to reproduce the original split.  Only
positions whose re-tokenized id still matches the recorded ``rl_id`` are
marked, and the counts of matched and total positions are reported so a silent
misalignment shows up as a low match rate instead of as bold text in the wrong
places.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

WS = re.compile(r"\s+")


def align_replacements(
    text: str,
    replacements: Sequence[Dict[str, int]],
    tokenizer: Any,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]], int, int]:
    """Map recorded replacement positions onto character spans of ``text``.

    Returns ``(spans, rows, n_matched, n_total)``.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    by_pos = {int(r["pos"]): r for r in replacements if int(r["pos"]) < len(ids)}
    matched = [p for p, r in by_pos.items() if ids[p] == int(r["rl_id"])]

    spans = sorted(tuple(offsets[p]) for p in matched)
    merged: List[List[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    rows = []
    for p in sorted(matched):
        a, b = offsets[p]
        rows.append(
            {
                "pos": p,
                "char": a,
                "base_token": tokenizer.decode([int(by_pos[p]["base_id"])]),
                "rl_token": tokenizer.decode([int(by_pos[p]["rl_id"])]),
                "context": WS.sub(" ", text[max(0, a - 60) : b + 60]),
            }
        )
    return [tuple(m) for m in merged], rows, len(matched), len(by_pos)


def bold_spans(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    """Wrap each span in ``**``, keeping surrounding whitespace outside."""
    out: List[str] = []
    prev = 0
    for a, b in spans:
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        out.append(text[prev:a])
        out.append(seg[:lead])
        core = seg[lead : len(seg) - trail]
        out.append("**{}**".format(core) if core.strip() else seg)
        if trail:
            out.append(seg[len(seg) - trail :])
        prev = b
    out.append(text[prev:])
    return "".join(out)


def export_response(
    out_dir: str,
    tag: str,
    sample_index: int,
    question: str,
    gt: str,
    text: str,
    replacements: Sequence[Dict[str, int]],
    tokenizer: Any,
    correct: Optional[bool] = None,
    length: Optional[int] = None,
    replace_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Write ``annot_<tag>_s<i>.md`` and ``reps_<tag>_s<i>.tsv``."""
    os.makedirs(out_dir, exist_ok=True)
    spans, rows, matched, total = align_replacements(text, replacements, tokenizer)
    annotated = bold_spans(text, spans)

    md_path = os.path.join(out_dir, "annot_{}_s{}.md".format(tag, sample_index))
    with open(md_path, "w", encoding="utf-8") as fh:
        head = "# {} · sample {}".format(tag, sample_index)
        bits = []
        if correct is not None:
            bits.append("correct" if correct else "incorrect")
        if length is not None:
            bits.append("{} tokens".format(length))
        bits.append("{} replacements".format(len(replacements)))
        if replace_rate is not None:
            bits.append("{:.2f}% replaced".format(100 * replace_rate))
        bits.append("{}/{} aligned".format(matched, total))
        fh.write("{} ({})\n\n".format(head, ", ".join(bits)))
        fh.write("Reference answer: {}\n\n".format(gt))
        fh.write("## Problem\n\n{}\n\n".format(question))
        fh.write("**Bold = token supplied by the RL model**; the base model "
                 "sampled everything else.\n\n---\n\n")
        fh.write(annotated + "\n")

    tsv_path = os.path.join(out_dir, "reps_{}_s{}.tsv".format(tag, sample_index))
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("pos\tchar\tbase_token\trl_token\tcontext\n")
        for r in rows:
            fh.write(
                "{pos}\t{char}\t{base_token!r}\t{rl_token!r}\t{context}\n".format(**r)
            )

    return {
        "markdown": md_path,
        "tsv": tsv_path,
        "n_aligned": matched,
        "n_total": total,
        "align_rate": (matched / total) if total else float("nan"),
    }


def inserted_token_positions(
    rows: Sequence[Dict[str, Any]], token: str = "Wait"
) -> List[Dict[str, Any]]:
    """Replacements that inserted a particular token, with context.

    §3.5 reads these directly: the RL model inserts "Wait" where the base model
    was about to accept a conclusion or start a new section.
    """
    return [r for r in rows if r["rl_token"].strip() == token]
