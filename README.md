# MFC-Transport-REINFORCE

Code for mean-field control experiments comparing:

- REINFORCE
- MF-REINFORCE
- Transport REINFORCE
- tabular mean-field Q-learning on cybersecurity

## Setup

```bash
uv sync
```

Run commands from the repository root.

## Repository layout

- `src/mfc/environments/`: benchmark environments.
- `src/mfc/algorithms/`: training and gradient estimators.
- `src/mfc/visualization/`: result loading, plots, tables, and diagnostics.
- `scripts/`: training, plotting, and report-generation scripts.
- `results/`: saved experiment outputs.

## Run one experiment

```bash
uv run python scripts/train.py \
  --env lq \
  --algorithm transport \
  --horizon 20 \
  --perturbation 0.1 \
  --n-train 10000 \
  --device cpu
```

Common environments are `twostate`, `cybersecurity`, `distribution`, `advertising`, `lq`, and `portfolio`.

## Run the full suite

```bash
scripts/run_suite.sh
```

Useful overrides:

```bash
WORKERS=4 scripts/run_suite.sh
RESULTS_ROOT=results_new scripts/run_suite.sh
scripts/run_suite.sh --no-resume
```

## Build plots and tables

```bash
uv run python scripts/plot.py \
  --env all \
  --results-root results \
  --output-root results/figures
```

Report-specific figures and tables:

```bash
uv run python scripts/report_figures.py --results-root results --output-root results/figures
uv run python scripts/report_tables.py --results-root results --output-root results/tables
```

Theory-verification figure:

```bash
uv run python scripts/verify_theory.py --part all
```
