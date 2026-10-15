# legacy/

Analysis scripts for inspecting per-problem results, generation statistics,
and training logs.

**These scripts do not run as-is.** Each one reads intermediate artifacts that
are not in this repository (per-problem aggregates, feature archives, raw
generation shards, parquet datasets). Provide those inputs and configure the
paths before running them. For the package-based workflow, see the
[main README](../README.md).

Conventions used by these scripts:

* The generation-length budget is 8,192 tokens throughout (§4.1), matching
  `wrb/` and the paper.
* The two problem groups are named `clean` and `contam` here, and some scripts
  carry an 8-gram overlap check against the training set. That is left over
  from a contamination audit the paper does not report. The paper's groups are
  simply **AIME24+25** and **AIME22+23**, which is what `clean` and `contam`
  mean wherever they appear below.

Internal absolute paths were replaced by placeholders:

| Placeholder | What it was |
|---|---|
| `${WRB_RUNS}` | the generation output tree, one directory per run |
| `${WRB_WORK}` | scratch directory holding per-problem aggregates and feature archives |
| `${WRB_DATA}` | parquet datasets (DAPO-Math-17k, MATH-500, AIME) |
| `${WRB_MODELS}` | model and tokenizer directories |
| `${WRB_ROOT}`, `${WRB_HANDOFF}` | project roots |

## What is here

| Directory | Contents |
|---|---|
| `stats_lib/` | `lib.py` (unbiased pass@k, bootstrap, percentile CI) and `anom_lib.py` (the degradation flags). Related package modules: `wrb/metrics.py` and `wrb/analysis/response_stats.py`. |
| `appendix_b_gate/` | Appendix B: feature extraction, per-feature AUROC, and the gate. `gate_repro_holdout.py` and `gate_repro_cv.py` are two fits of the same model — a holdout split and a 5-fold cross-validated one — and they give different decision thresholds. |
| `tables/` | The per-problem aggregate builder (`extract_perq.py`) and the table scripts: main recovery table, pass@k curve, ablations, gate-MCMC, the Llama pair, MATH-500, and checkpoint selection. |
| `replacements/` | Replacement token/pair counts and density-and-category analysis. Its categories differ from Table 5's; use [`wrb/analysis/replacements.py`](../wrb/analysis/replacements.py) for the Table 5 taxonomy. |
| `response_stats/` | Length, answer rate, repetition, diversity. |
| `ablations/` | Test 2 and Test 3 condition tables and paired controls. |
| `run_consistency/` | Cross-run agreement and the run-to-run variation behind the ±1.5-point noise band of §4.3. |
| `training/` | Training-log parsing: hyperparameters, rollout statistics, validation curves. |
| `appendix_stats/` | The October 2026 appendix statistics: per-problem E.1 counts, response statistics, answered-only confidence intervals, run alignment, and the annotated-response export. |

## What was removed

Roughly a hundred further scripts from the same handoff were dropped: forensics
of a corrupted inference engine, a training-data contamination audit, an
online-distillation side experiment, and reverse-engineering of another run's
sampling parameters, prompt template and RNG seed. None of them supports a
number in the paper. Also removed: a packaging script containing a literal
password, and a script that queried an internal job server through a
proprietary client.
