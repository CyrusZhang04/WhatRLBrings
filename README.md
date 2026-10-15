# What RL Brings to Reasoning: A Few Decisions the Base Model Would Not Make

Code for the paper. RL substantially improves base language models on hard math
problems, but what it *adds* is unclear. We test common explanations against
each other on one base/RL pair (Qwen3-4B-Base and the model trained from it with
GRPO) and one set of hard problems (the 120 problems of AIME 2022-2025), where
RL nearly doubles pass@1 from 10.00% to 19.32%.

Each test is an intervention on the **base** model motivated by one explanation.
If the explanation accounts for the gain, the intervention should recover it:

| Test | Explanation tested | Intervention | Result (pass@1 / pass@16) |
|---|---|---|---|
| 0 | Sharper base distribution | Power sampling (MCMC) | no gain over greedy on AIME |
| 1 | Gain localized at disagreement positions | Replace at ~4% of positions | **18.44 / 43.33** |
| 2 | Opening cue | Force the first token | 13.23 / 35.00 |
| 3 | Fixed reflective behavior | Insert the same tokens by rule | at or below base |

Base model: 10.00 / 30.00. RL model: 19.32 / 40.00.

Borrowing the RL model's token at only the positions where the two models'
greedy tokens differ (`rl-replace`) recovers **90.5%** of the gain while leaving
96% of the tokens to the base model. Forcing the opening word "Alright" recovers
35%. Inserting the RL model's reflective tokens ("Wait", the prefix "Re") by
fixed rules does not help. And because the base model still writes 96% of each
response, the gain largely separates from RL's loss of diversity: at k = 256,
rl-replace solves 67.50% of the problems against 61.40% for the RL model and
61.64% for the base model.

## Install

```bash
pip install -e .            # analysis only (CPU)
pip install -e ".[gen]"     # + transformers/torch, for generation
```

## The decoding rule

`rl-replace` is the whole of Test 1. At each step, with base model `p_B` and RL
model `p_R` scoring the same prefix:

```
b_t = argmax p_B(. | y_<t)
r_t = argmax p_R(. | y_<t)
y_t = r_t                      if b_t != r_t      # a disagreement position
      sample from p_B at T=0.6 otherwise
```

The RL model intervenes only at disagreement positions — about 4.0% of tokens
(4.5% averaged per response) — and the base model writes everything else.

## Quickstart

Generate one condition, then build the tables:

```bash
# Test 1: rl-replace
python scripts/run_eval.py --condition rl_replace \
    --base-model /models/Qwen3-4B-Base \
    --rl-model   /models/qwen3-4b-grpo-step460 \
    --problems data/aime24.jsonl --split aime24 \
    --n-samples 16 --out outputs/v1

# Test 2: force the opening word
python scripts/run_eval.py --condition base_with_alright \
    --base-model /models/Qwen3-4B-Base \
    --problems data/aime24.jsonl --split aime24 --out outputs/alright

# Tables 3, 4, 6, 8 from whatever conditions are present
python scripts/make_tables.py --runs outputs/v1 outputs/alright --tables 3 4 8

# pass@k curve (Figure 1) and the §3.6 whole-response mixture control
python scripts/make_tables.py --runs outputs/large_k --tables 11 mix
```

`--problems` is jsonl with `question` and `answer` per line.

Available conditions: `base`, `base_greedy`, `full_rl`, `rl_replace`,
`base_greedy_at_disagree`, `base_with_alright`, `base_with_alright_i`,
`force_wait`, `force_re`, `force_wait_re`, `re_logit_plus2`, `re_logit_plus4`,
`wait_topk`.

**Cost.** rl-replace runs a forward pass of both models for every generated
token. One AIME year (30 problems, 16 samples each) took 7.3-7.9 GPU-hours in
the paper's runs, and MATH-500 about 55. It is a diagnostic, not a practical
decoding method.

## Layout

```
wrb/
  prompts.py            Appendix D: the prompts, identical across conditions
  grading.py            last \boxed{}, Math-Verify, repetition detection
  metrics.py            pass@1, unbiased pass@k, maj@k, recovery, bootstrap CIs
  records.py            the on-disk generation format
  decoding/
    rl_replace.py       Test 1 (and the base-greedy control of Table 4)
    prefill.py          Test 2
    rules.py            Test 3, hard and soft rules
    power_sampling.py   Test 0: greedy, standard sampling, the MCMC interface
  gate/                 Appendix B: six base-only features, the gate, Table 10
  analysis/
    aggregate.py        raw shards -> per-problem records
    tables.py           Tables 3, 4, 6, 8, 11
    mixture.py          the whole-response mixture control of §3.6
    replacements.py     Table 5 taxonomy and positional density
    response_stats.py   answer rate, length, repetition, diversity
    annotate.py         Appendix E.1 annotated export
configs/rl_training/    verl GRPO configs for the three pairs (§4.1)
scripts/                run_eval.py, make_tables.py
legacy/                 analysis scripts for reference
docs/                   paper_map.md
```

[docs/paper_map.md](docs/paper_map.md) maps every table and claim to the code
that produces it.

## Training

The three RL models are trained directly from their base models, with no SFT,
using GRPO on DAPO-Math-17k in [verl](https://github.com/volcengine/verl):
constant learning rate 1e-6, 128 prompts per step with 8 rollouts each, reward
+1 for a correct final answer and -1 otherwise. Checkpoints are selected on a
validation set drawn from DAPO-Math-17k, so no evaluation benchmark is used for
selection: step 460 (Qwen3-4B), 320 (Qwen3-8B), 500 (Llama-3.2-3B).

```bash
python -m verl.trainer.main_ppo \
    --config-path configs/rl_training --config-name qwen3_4b_base_grpo
```
