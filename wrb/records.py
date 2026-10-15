"""On-disk format for generations, and the per-problem aggregate built from it.

Every intervention writes the same record layout, which is what makes the
conditions comparable and the analysis code condition-agnostic.  One JSON
object per line, one line per problem:

    {
      "ds": "aime24",            # benchmark split
      "qi": 0,                   # 0-based problem index within the split
      "question": "...",         # problem statement, as prompted
      "gt": "480",               # reference answer
      "base": {                  # one block per condition present in this run
        "texts":   ["...", ...],           # n responses
        "accs":    [true, false, ...],     # n booleans from Grader
        "lengths": [8192, 3104, ...],      # n generated-token counts
        "replace_rates": null,             # rl-replace only
        "replacements":  null              # rl-replace only
      },
      "rl_replace": {
        ...,
        "replace_rates": [0.035, ...],     # emitted RL tokens / response length
        "replacements":  [[{"pos": 0, "base_id": 1427, "rl_id": 83438}, ...], ...]
      }
    }

Condition names used by the paper's runs:

    base                    base model, standard sampling (T=0.6, top-p 0.95)
    base_greedy             base model, greedy decoding
    rl_replace              Test 1
    full_rl                 the RL model
    base_with_alright       Test 2, prefill "Alright"
    base_with_alright_i     Test 2, prefill "Alright, I"
    base_greedy_at_disagree Test 2 control: base's own greedy token there
    mcmc                    Test 0, power sampling
    gate_mcmc               Test 0, power sampling restricted by the gate
    force_wait              Test 3, hard rule
    force_re                Test 3, hard rule
    force_wait_re           Test 3, both hard rules
    re_logit_plus2          Test 3, soft rule
    re_logit_plus4          Test 3, soft rule

``replacements`` holds one list per sample, with the token position in the
*generated* sequence and the two token ids, which is what the Appendix E.1
annotation and the Table 5 taxonomy are built from.  ``pos`` is an index into
the generated tokens, not into the prompt.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

#: Fields every condition block carries.
REQUIRED_BLOCK_FIELDS = ("texts", "accs", "lengths")
#: Fields only rl-replace carries.
REPLACE_BLOCK_FIELDS = ("replace_rates", "replacements")


def make_block(
    texts: Sequence[str],
    accs: Sequence[bool],
    lengths: Sequence[int],
    replace_rates: Optional[Sequence[float]] = None,
    replacements: Optional[Sequence[Sequence[Dict[str, int]]]] = None,
) -> Dict[str, Any]:
    n = len(texts)
    if not (len(accs) == len(lengths) == n):
        raise ValueError(
            "texts/accs/lengths must have equal length, got {}/{}/{}".format(
                n, len(accs), len(lengths)
            )
        )
    for name, seq in (("replace_rates", replace_rates), ("replacements", replacements)):
        if seq is not None and len(seq) != n:
            raise ValueError("{} must have {} entries, got {}".format(name, n, len(seq)))
    return {
        "texts": list(texts),
        "accs": [bool(a) for a in accs],
        "lengths": [int(x) for x in lengths],
        "replace_rates": None if replace_rates is None else [float(x) for x in replace_rates],
        "replacements": None if replacements is None else [list(x) for x in replacements],
    }


def make_record(
    ds: str,
    qi: int,
    question: str,
    gt: str,
    blocks: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "ds": ds,
        "qi": int(qi),
        "question": question,
        "gt": str(gt),
    }
    for name, blk in blocks.items():
        validate_block(blk, where="{}:{} [{}]".format(ds, qi, name))
        rec[name] = blk
    return rec


def validate_block(blk: Dict[str, Any], where: str = "") -> None:
    missing = [f for f in REQUIRED_BLOCK_FIELDS if f not in blk]
    if missing:
        raise ValueError("condition block {} is missing {}".format(where, missing))
    n = len(blk["texts"])
    if not (len(blk["accs"]) == len(blk["lengths"]) == n):
        raise ValueError("condition block {} has ragged arrays".format(where))


def condition_names(rec: Dict[str, Any]) -> List[str]:
    """Condition blocks present in a record, in insertion order."""
    meta = {"ds", "qi", "question", "gt"}
    return [k for k, v in rec.items() if k not in meta and isinstance(v, dict)]


def write_shard(
    path: str,
    records: Iterable[Dict[str, Any]],
    *,
    sync_each: bool = False,
    overwrite: bool = True,
) -> int:
    """Write records as jsonl, creating parent directories.

    With ``sync_each``, flush and sync each record before requesting the next
    one from the iterable. Set ``overwrite=False`` to preserve existing shards.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 0
    with open(path, "w" if overwrite else "x", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if sync_each:
                fh.flush()
                os.fsync(fh.fileno())
            n += 1
    return n


def shard_paths(run_dir: str, ds: str, n_samples: int, n_shards: int) -> List[str]:
    """Canonical shard filenames: ``<ds>-n<N>-shard<i>of<S>.jsonl``."""
    return [
        os.path.join(run_dir, ds, "{}-n{}-shard{}of{}.jsonl".format(ds, n_samples, i, n_shards))
        for i in range(n_shards)
    ]


def read_run(run_dir: str, ds: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield every record under a run directory, shards in numeric order.

    Accepts both the sharded layout (``<run>/<ds>/<ds>-n16-shard0of8.jsonl``)
    and a flat one (``<run>/*.jsonl``).
    """
    import glob
    import re

    pattern = (
        os.path.join(run_dir, ds, "*.jsonl") if ds else os.path.join(run_dir, "**", "*.jsonl")
    )
    files = glob.glob(pattern, recursive=True) or glob.glob(os.path.join(run_dir, "*.jsonl"))

    def shard_key(p: str):
        m = re.search(r"shard(\d+)of(\d+)", os.path.basename(p))
        return (os.path.dirname(p), int(m.group(1)) if m else 0, p)

    for path in sorted(files, key=shard_key):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def problem_id(rec: Dict[str, Any]) -> str:
    """Stable per-problem key: ``aime24#1`` (1-based, as the paper numbers them)."""
    return "{}#{}".format(rec["ds"], int(rec["qi"]) + 1)


#: AIME splits, and the two groups the paper reports (§3.1).
AIME_SPLITS = ("aime22", "aime23", "aime24", "aime25")
GROUPS = {
    "aime24": "aime24+25",
    "aime25": "aime24+25",
    "aime22": "aime22+23",
    "aime23": "aime22+23",
}


def group_of(ds: str) -> str:
    """Problem group for the per-group bootstrap and the paper's two AIME rows."""
    return GROUPS.get(ds, ds)
