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
- `scripts/`: training, diagnostics, and final output generation.
- `results/`: saved experiment outputs.
- `outputs/`: generated figures and tables.

## Run one experiment

```bash
uv run python scripts/train.py \
  --env lq \
  --algorithm transport \
  --horizon 20 \
  --flow particle \
  --perturbation 0.308084 \
  --eta 0.429187 \
  --n-components 1 \
  --n-train 10000 \
  --device cpu
```

Common environments are `twostate`, `cybersecurity`, `distribution`, `advertising`, `lq`, and `portfolio`.

## Run the full suite

The suite runs the current benchmark set on CPU using the bound-derived transport scales from
`scripts/run.py`. It does not include Kuramoto, adaptive transport, an eta sweep, or LQ `K=3`
experiments.

```bash
scripts/run_suite.sh
```

Useful overrides:

```bash
WORKERS=4 scripts/run_suite.sh
CORES=18 WORKERS=6 scripts/run_suite.sh
scripts/run_suite.sh --no-resume
```

On a fresh pod, the typical setup is:

```bash
apt update
apt install -y tmux git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync
tmux new -s mfc
CORES=18 WORKERS=6 scripts/run_suite.sh
```

Detach from tmux with `Ctrl-b d`, reattach with:

```bash
tmux attach -t mfc
```

To inspect progress across running jobs:

```bash
for f in $(find results/logs -name '*.log' | sort); do
  echo "== $f =="
  tail -n 1 "$f"
done
```

## Build figures and tables

The full suite calls this automatically after training finishes:

```bash
uv run python scripts/make_outputs.py --results-root results
```

It writes the final artifacts to:

```text
outputs/figures/learning_curves.pdf
outputs/figures/learned_policies.pdf
outputs/figures/theory_verification.pdf
outputs/tables/objective_summary.tex
outputs/tables/budget_runtime.tex
```

Use `scripts/plot.py` only for exploratory diagnostics from saved runs.

## Theory diagnostics

The final theory figure expects the CSVs created by:

```bash
uv run python scripts/verify_theory.py --part all
```
