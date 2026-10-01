# Randomized transport maps for model-free policy-gradient mean-field control

Code for the mean-field control experiments comparing:

- REINFORCE
- MF-REINFORCE
- Transport REINFORCE, with a zero-order population-flow sensitivity
- Transport-Proba REINFORCE, with a likelihood-ratio population-flow sensitivity
  (continuous-state benchmarks only)
- tabular mean-field Q-learning, on cybersecurity
- centered finite differences of the objective, on the continuous-state benchmarks,
  using the shifted systems of the transport auxiliary stage

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

Environments are `twostate`, `cybersecurity`, `distribution`, `advertising`, `lq`,
`portfolio` and `bimodal`. Algorithms are `reinforce`, `mfreinforce`, `transport`,
`gaussian`, `finitediff` and `mfqlearning`. `finitediff` takes its step as
`--perturbation`; `--common-random-numbers` drives both systems of a coordinate with
the same draws.

`bimodal` is the bimodal population allocation benchmark (`T=1`,
`Theta=[0.75, 0.95]`, initialized at `theta=0.8`). Its terminal law has mean zero
and variance one for every `theta`, so `--n-components 1` sees no gradient and
`--n-components 2` represents the law exactly. Updates are projected onto `Theta`,
and the auxiliary radius is `--eta 0.04`, the largest that keeps every shifted
policy valid. Validation is the exact `J(theta)`, whose maximum is zero. It has no
`gaussian` arm, since that estimator hands the environment the population mean.

```bash
uv run python scripts/train.py \
  --env bimodal --algorithm transport --horizon 1 \
  --perturbation 0.0281171 --eta 0.04 --n-components 2 --device cpu
```

`gaussian` is Transport-Proba. It represents the population by the pair
`(m_t, sigma_t)` of a single Gaussian, randomizes it as
`m_t^lambda = (1-lambda) m_t + lambda A_t` and
`Sigma_t^lambda = ((1-lambda) + lambda B_t)^2 sigma_t^2`, and estimates
`grad m_t` and `grad log sigma_t` by a likelihood ratio rather than by centered
policy differences. It takes no `--eta`.

Every ten updates the frozen policy is validated by Monte Carlo with
`validation_particles` interacting particles (default `10^5`, override with
`--validation-particles`), from the benchmark's validation law and with a seed fixed
within the run. `make_outputs.py` evaluates the optimal policy with the same particles
and seed, so reported gaps are free of the finite-particle offset.

Every headline run reads the population off `M` interacting particles (`--flow particle`),
finite benchmarks included; `--flow exact` remains available as an oracle. In finite state
space the validation particles are simulated through their counts, which has the same law.

In finite state space the sensitivities `D_1, ..., D_T` are estimated from one batch
of `n` auxiliary trajectories of length `T`. The auxiliary radius is selected before
training, per benchmark, in `scripts/run.py` (`MEASURED_AUXILIARY_ETA`); two-state
control also sweeps the radii `n^(-1/4) / 2` and `n^(-1/4)` against the large grid.

## The full suite

Every configuration of the plan, with its settings, budget and command:

```bash
uv run python scripts/run.py --env all --manifest outputs/run_manifest.csv
```

```bash
scripts/run_suite.sh
```

Useful overrides: `WORKERS=4`, `CORES=18`, `ENVS="lq portfolio"`, `--no-resume`.
Leave algorithms out with `--exclude-algorithms`, e.g. `scripts/run_suite.sh --exclude-algorithms finitediff`.

The figures also read diagnostic CSVs under `results/figures/`, which do not depend on the
training runs. Regenerate them, for instance beside the suite, with
`THREADS=1 scripts/run_diagnostics.sh`.
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
own entry point, which `run_suite.sh` also calls at the end:

```bash
uv run python scripts/decomposition.py --root results --tex outputs/tables/decomposition.tex
```

## Diagnostics

```bash
uv run python scripts/verify_randomizers.py   # closed-form J^lambda against simulation
uv run python scripts/verify_gaussian.py      # Transport-Proba score, sensitivities, gradient
uv run python scripts/verify_bounds.py        # error rates in eta, n, M, lambda and B
uv run python scripts/verify_discrete_eta.py --compare-batches  # finite-state eta, shared vs fresh batches
uv run python scripts/verify_discrete_bounds.py --env all       # finite-state rates in n, M, lambda and B
uv run python scripts/tune_finite_difference.py --independent  # finite-difference step against the oracle
uv run python scripts/verify_theory.py        # perturbation estimate and consistency
uv run python scripts/tune_perturbation.py    # auxiliary scales against the gradient oracle
uv run python scripts/correction_alignment.py # alignment of the transport correction
uv run python scripts/lambda_tradeoff.py      # gradient bias and dispersion against lambda
```

`scripts/plot.py` is for exploratory inspection of saved runs only.
