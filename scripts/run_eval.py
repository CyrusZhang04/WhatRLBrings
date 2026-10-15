#!/usr/bin/env python
"""Generate and grade one condition, writing the shard format in wrb/records.py.

Examples
--------
Test 1, rl-replace on AIME 2024 with 16 samples per problem::

    python scripts/run_eval.py --condition rl_replace \\
        --base-model /models/Qwen3-4B-Base \\
        --rl-model   /models/qwen3-4b-grpo-step460 \\
        --problems data/aime24.jsonl --split aime24 \\
        --n-samples 16 --out outputs/rl_replace_v1

Test 2, forcing the opening word::

    python scripts/run_eval.py --condition base_with_alright \\
        --base-model /models/Qwen3-4B-Base \\
        --problems data/aime24.jsonl --split aime24 --out outputs/alright

Test 3, a hard rule::

    python scripts/run_eval.py --condition force_wait \\
        --base-model /models/Qwen3-4B-Base \\
        --problems data/aime24.jsonl --split aime24 --out outputs/rules

The problems file is jsonl with ``question`` and ``answer`` on each line.

Each completed question is written and synced to disk before starting the next.
Existing output files are never overwritten; use a new --out directory to rerun.

Note on cost: rl-replace runs a forward pass of *both* models for every
generated token.  One AIME year (30 problems, 16 samples each) took 7.3-7.9
GPU-hours in the paper's runs, and MATH-500 about 55.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrb import records as rec_io
from wrb.constants import DEFAULT_N_SAMPLES, MAX_NEW_TOKENS
from wrb.grading import Grader
from wrb.prompts import build_eval_prompt

BASE_ONLY = {
    "base",
    "base_greedy",
    "base_with_alright",
    "base_with_alright_i",
    "force_wait",
    "force_re",
    "force_wait_re",
    "re_logit_plus2",
    "re_logit_plus4",
    "wait_topk",
}
NEEDS_RL = {"rl_replace", "base_greedy_at_disagree", "full_rl"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, help="condition name (see wrb/records.py)")
    p.add_argument("--base-model", required=True)
    p.add_argument("--rl-model", default=None, help="required for rl_replace / full_rl")
    p.add_argument("--problems", required=True, help="jsonl with question/answer")
    p.add_argument("--split", required=True, help="benchmark split name, e.g. aime24")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True, help="run directory")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--family", default="qwen3", help="chat-template family")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def load_problems(path, shard, n_shards):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i % n_shards != shard:
                continue
            d = json.loads(line)
            out.append((i, d["question"], str(d["answer"])))
    return out


def load_model(path, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), device_map=device
    )


def main(argv=None):
    args = parse_args(argv)
    if args.condition in NEEDS_RL and not args.rl_model:
        raise SystemExit("--rl-model is required for condition {!r}".format(args.condition))
    if args.condition not in BASE_ONLY | NEEDS_RL:
        raise SystemExit(
            "unknown condition {!r}; known: {}".format(
                args.condition, ", ".join(sorted(BASE_ONLY | NEEDS_RL))
            )
        )

    path = os.path.join(
        args.out,
        args.split,
        "{}-n{}-shard{}of{}.jsonl".format(args.split, args.n_samples, args.shard, args.n_shards),
    )
    n = rec_io.write_shard(path, evaluate(args), sync_each=True, overwrite=False)
    print("wrote {} records to {}".format(n, path))


def evaluate(args):
    """Yield one completed question at a time, without retaining earlier results."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = load_model(args.base_model, args.dtype, args.device)
    rl = load_model(args.rl_model, args.dtype, args.device) if args.rl_model else None

    grader = Grader()
    problems = load_problems(args.problems, args.shard, args.n_shards)
    print("{}: {} problems, {} samples each".format(args.condition, len(problems), args.n_samples))

    for qi, question, answer in problems:
        prompt = build_eval_prompt(question, tokenizer=tokenizer, family=args.family)
        texts, lengths, rates, reps = generate(args, prompt, tokenizer, base, rl)
        accs = grader.grade_many(texts, answer)
        block = rec_io.make_block(
            texts=texts,
            accs=accs,
            lengths=lengths,
            replace_rates=rates,
            replacements=reps,
        )
        yield rec_io.make_record(args.split, qi, question, answer, {args.condition: block})
        print("  {}#{}: {}/{} correct (saved)".format(args.split, qi + 1, sum(accs), len(accs)))


def generate(args, prompt, tokenizer, base, rl):
    """Dispatch to the decoder for this condition.

    Returns ``(texts, lengths, replace_rates, replacements)``, with the last two
    ``None`` for every condition except rl-replace.
    """
    cond = args.condition

    if cond in ("rl_replace", "base_greedy_at_disagree"):
        from wrb.decoding.rl_replace import ReplaceConfig, RLReplaceDecoder

        cfg = ReplaceConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            at_disagreement="rl" if cond == "rl_replace" else "base_greedy",
            seed=args.seed,
        )
        outs = RLReplaceDecoder(base, rl, tokenizer, cfg).generate(prompt, args.n_samples)
        return (
            [o.text for o in outs],
            [o.length for o in outs],
            [o.replace_rate for o in outs],
            [o.replacements for o in outs],
        )

    if cond == "full_rl":
        from wrb.decoding.power_sampling import generate_standard

        texts = generate_standard(
            rl, tokenizer, prompt, args.n_samples, args.temperature, args.top_p,
            args.max_new_tokens, args.seed,
        )
        return texts, [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts], None, None

    if cond in ("base_with_alright", "base_with_alright_i"):
        from wrb.decoding.prefill import ALRIGHT, ALRIGHT_I, PrefillConfig, generate_with_prefill

        cfg = PrefillConfig(
            prefill=ALRIGHT if cond == "base_with_alright" else ALRIGHT_I,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        texts = generate_with_prefill(base, tokenizer, prompt, args.n_samples, cfg)
        return texts, [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts], None, None

    if cond in ("force_wait", "force_re", "force_wait_re", "re_logit_plus2", "re_logit_plus4", "wait_topk"):
        from wrb.decoding.rules import RuleConfig, RuleGuidedDecoder, table6_configs

        if cond == "wait_topk":
            cfg = RuleConfig(force_if_in_topk="Wait", topk=5, max_new_tokens=args.max_new_tokens)
        else:
            cfg = table6_configs(args.max_new_tokens)[cond]
        cfg.temperature = args.temperature
        cfg.top_p = args.top_p
        cfg.seed = args.seed
        results = RuleGuidedDecoder(base, tokenizer, cfg).generate(prompt, args.n_samples)
        return (
            [r["text"] for r in results],
            [len(r["token_ids"]) for r in results],
            None,
            None,
        )

    if cond == "base_greedy":
        from wrb.decoding.power_sampling import generate_greedy

        text = generate_greedy(base, tokenizer, prompt, args.max_new_tokens)
        return [text], [len(tokenizer.encode(text, add_special_tokens=False))], None, None

    if cond == "base":
        from wrb.decoding.power_sampling import generate_standard

        texts = generate_standard(
            base, tokenizer, prompt, args.n_samples, args.temperature, args.top_p,
            args.max_new_tokens, args.seed,
        )
        return texts, [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts], None, None

    raise SystemExit("no decoder wired for condition {!r}".format(cond))


if __name__ == "__main__":
    main()
