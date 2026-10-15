# What RL Brings to Reasoning: A Few Decisions the Base Model Would Not Make

Code for the paper, studying what reinforcement learning adds to base language
models on mathematical reasoning tasks.

The repository includes decoding interventions, evaluation and grading tools,
and analysis utilities. The central intervention, **rl-replace**, uses the RL
model's token when the two models' greedy predictions disagree, and samples
from the base model otherwise.

See [the paper-to-code map](docs/paper_map.md) for experiment and table references.

## Installation

Requires Python 3.9+. Run these commands from the repository root:

```bash
pip install -e .          # CPU analysis tools
pip install -e ".[gen]"   # Also install generation dependencies
```

The evaluation example below uses a CUDA GPU. Model checkpoints and datasets
must be supplied separately.

## Quickstart

Prepare a JSONL file with one problem per line, using `question` and `answer`:

```json
{"question": "What is 1 + 1?", "answer": "2"}
```

Run rl-replace, replacing the example model and dataset paths with your own.
The multiline command uses Bash syntax:

```bash
python scripts/run_eval.py --condition rl_replace \
    --base-model /path/to/base-model \
    --rl-model /path/to/rl-model \
    --problems data/aime24.jsonl --split aime24 \
    --n-samples 16 --out outputs/rl_replace
```

Each completed question is saved and synced to disk. Existing output files
are not overwritten; use a separate `--out` directory for each condition or
new run. Automatic resume is not supported.

Summarize the saved responses:

```bash
python scripts/make_tables.py --runs outputs/rl_replace --tables stats
```

Other conditions include `base`, `base_greedy`, `full_rl`, opening-word prefills,
and rule-based interventions. See [the evaluation script](scripts/run_eval.py)
for the supported conditions, and use `--help` for command-line options.
rl-replace evaluates both models at every generated token.

Power sampling requires an external MCMC backend; see
[`wrb/decoding/power_sampling.py`](wrb/decoding/power_sampling.py).

## Repository structure

```text
wrb/                   Prompts, grading, metrics, and record formats
  decoding/            Baselines and decoding interventions
  analysis/            Aggregation, tables, and response statistics
  gate/                Feature extraction and gate training
scripts/               Evaluation and analysis entry points
configs/rl_training/   GRPO training configurations
tests/                 Unit and regression tests
docs/                  Paper-to-code map
legacy/                Reference analysis scripts
```

## Training

GRPO configurations for Qwen3-4B, Qwen3-8B, and Llama-3.2-3B are in
[`configs/rl_training/`](configs/rl_training/). They require a separate
[verl](https://github.com/volcengine/verl) installation, model paths, and prepared
DAPO-Math-17k training and validation data. See the configuration files for
environment variables and training settings.

## Citation and license

Citation metadata is available in [CITATION.cff](CITATION.cff).
This repository is released under the [MIT License](LICENSE).
