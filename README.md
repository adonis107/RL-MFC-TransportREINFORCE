# Mean-field control with transport-randomized policy gradients

Code for the mean-field control experiments comparing:

- REINFORCE
- MF-REINFORCE
- Transport REINFORCE, with a zero-order population-flow sensitivity
- Transport-Proba REINFORCE, with a likelihood-ratio population-flow sensitivity
  (continuous-state benchmarks only)
- tabular mean-field Q-learning, on cybersecurity

## Setup

```bash
uv sync
```

Run every command from the repository root.

## Layout

- `src/mfc/environments/` benchmark environments
- `src/mfc/algorithms/` training and gradient estimators
- `src/mfc/visualization/` result loading, plots and tables
- `scripts/` training, diagnostics and final outputs
- `results/` saved runs
- `outputs/` generated figures and tables

## One experiment

```bash
uv run python scripts/train.py \
  --env lq --algorithm transport --horizon 20 \
  --perturbation 0.308084 --eta 0.95 --n-components 1 \
  --n-train 10000 --device cpu
```

Environments are `twostate`, `cybersecurity`, `distribution`, `advertising`, `lq`
and `portfolio`. Algorithms are `reinforce`, `mfreinforce`, `transport`,
`gaussian` and `mfqlearning`.

`gaussian` is Transport-Proba. It represents the population by the pair
`(m_t, sigma_t)` of a single Gaussian, randomizes it as
`m_t^lambda = (1-lambda) m_t + lambda A_t` and
`Sigma_t^lambda = ((1-lambda) + lambda B_t)^2 sigma_t^2`, and estimates
`grad m_t` and `grad log sigma_t` by a likelihood ratio rather than by centered
policy differences. It takes no `--eta`.

## The full suite

```bash
scripts/run_suite.sh
```

Useful overrides: `WORKERS=4`, `CORES=18`, `ENVS="lq portfolio"`, `--no-resume`.
Runs already carrying a `summary.json` are skipped, so the suite resumes.

To follow progress:

```bash
for f in $(find results/logs -name '*.log' | sort); do echo "== $f"; tail -n 1 "$f"; done
```

## Figures and tables

```bash
uv run python scripts/make_outputs.py --results-root results
```

The decomposition table recomputes a perturbed optimum per scale, so it has its
own entry point:

```bash
uv run python scripts/decomposition.py lq portfolio --root results \
  --tex outputs/tables/continuous_decomposition.tex
```

## Diagnostics

```bash
uv run python scripts/verify_randomizers.py   # closed-form J^lambda against simulation
uv run python scripts/verify_gaussian.py      # Transport-Proba score, sensitivities, gradient
uv run python scripts/verify_bounds.py        # error rates in eta, n, M, lambda and B
uv run python scripts/verify_theory.py        # perturbation estimate and consistency
uv run python scripts/tune_perturbation.py    # auxiliary scales against the gradient oracle
uv run python scripts/correction_alignment.py # alignment of the transport correction
uv run python scripts/lambda_tradeoff.py      # gradient bias and dispersion against lambda
```

`scripts/plot.py` is for exploratory inspection of saved runs only.
