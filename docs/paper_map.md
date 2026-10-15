# Paper to code

This page maps the paper's experiments, tables, and metrics to the relevant
modules in this repository.

## The four tests

| Paper | Claim | Code |
|---|---|---|
| Test 0, §3.2 | Power sampling gives no gain over greedy decoding on AIME | `wrb/decoding/power_sampling.py` (settings + gate; MCMC kernel is upstream) |
| Test 0, §3.2 | Sharpening degrades long generations (no-answer 12.2% -> 22.4%, repetition 6.7% -> 13.1%) | `wrb/analysis/response_stats.py::degradation_flags` |
| Test 0, Appendix A | Gate-MCMC: 9.84 / 24.17 against 10.00 / 30.00 | `power_sampling.GATE_THRESHOLD` + `wrb/gate/` |
| Test 1, §3.3 | rl-replace recovers 90.5% of the gain at ~4% of positions | `wrb/decoding/rl_replace.py`, `wrb/analysis/tables.py::table3` |
| Test 1, §4.1 | Replace rate = emitted RL tokens / response length | `rl_replace.ReplaceOutput.replace_rate`, `replacement_stats` |
| Test 2, §3.4 | Forcing "Alright" recovers 35%; "Alright, I" adds nothing | `wrb/decoding/prefill.py`, `tables.py::delta_table` |
| Test 2, Table 4 | Base's own greedy token at the same positions gives 14.74% | `ReplaceConfig(at_disagreement="base_greedy")` |
| Test 3, §3.5 | Hard rules fall 5.5-7.7 points below base; soft rules do nothing | `wrb/decoding/rules.py`, `tables.py::delta_table` |
| Test 3, §3.5 | "Wait" among the base model's top-5 lowers accuracy to 6.25% | `RuleConfig(force_if_in_topk="Wait", topk=5)` |

## Tables and figures

| Paper | Code |
|---|---|
| Table 1 (summary of the four tests) | — (narrative) |
| Table 2 (greedy / sampling / MCMC / rl-replace / RL) | `power_sampling.py` + `tables.py::delta_table` |
| Table 3 (rl-replace vs base and RL, recovery + CI) | `tables.py::table3` |
| Table 4 (forced opening) | `tables.py::delta_table` |
| Table 5 (taxonomy of 9,099 replacements) | `wrb/analysis/replacements.py::table5` |
| Table 6 (rule-based insertion) | `tables.py::delta_table` |
| Table 7 (three model pairs on MATH-500) | `tables.py::table3` per pair |
| Table 8 (per competition year) | `tables.py::table8` |
| Table 10 (Appendix B AUROC) | `wrb/gate/train.py::auroc_table` |
| Table 11 / Figure 1 (pass@k to k=256) | `tables.py::passk_curve` |
| §3.5 positional density (26.1% in the first 20%) | `replacements.py::positional_density` |
| §3.5 recurring patterns (So->Wait, Let->###) | `replacements.py::recurring_pairs` |
| §3.6 diversity (15.6 / 3.2 / 9.7 distinct openings) | `metrics.py::distinct_openings`, `distinct_answers` |
| §3.6 response mixture (92% RL / 8% base; 61.45% at k=256) | `wrb/analysis/mixture.py::mixture_curve` |
| Appendix D (prompts) | `wrb/prompts.py` |
| Appendix E.1 (worked example, annotated) | `wrb/analysis/annotate.py` |

## Metrics

| Paper | Code |
|---|---|
| pass@1 = mean accuracy over samples | `metrics.pass_at_1` |
| pass@k, unbiased estimator (Chen et al., 2021) | `metrics.pass_at_k` |
| maj@16, grouping equivalent answers | `metrics.maj_at_k` |
| Recovery rate, Eq. (1) | `metrics.recovery_rate`, `metrics.recovery_ci` |
| 95% CI, problem-level bootstrap, 10,000 resamples within each group | `metrics.bootstrap_ci(groups=...)` |
| Correctness via Math-Verify on the last `\boxed{}` | `grading.Grader.is_correct` |

## Setup (§4.1)

| Paper | Code |
|---|---|
| GRPO in verl, DAPO-Math-17k, lr 1e-6 constant, 128 prompts x 8 rollouts, reward ±1 | `configs/rl_training/*.yaml` |
| Checkpoints: step 460 / 320 / 500 | the three config files |
| Temperature 0.6, top-p 0.95, 8,192 tokens | `wrb/decoding/sampling.py`, `--max-new-tokens` |
| 16 samples per AIME problem; 64 on MATH-500 | `scripts/run_eval.py --n-samples` |
