"""Empirical rates of the finite-state transport estimator, one block size or scale at a time.

The bias and mean-square-error bound of the finite-state estimator reads

    E||G_hat - grad J||^2 <= C (lambda^2 + (1 + lambda^-2)/B + eta^2 + 1/(n eta^2) + 1/M).

Each sweep below varies one quantity at the benchmark's allocation, from its validation law and
initial policy, with every other error source removed or held fixed:

  n       E sum_t ||D_hat_t - D_t||^2 against the exact sensitivities, at the selected eta and
          exact flow. Falls as 1/n above the floor set by the eta-perturbed target.
  M       E sum_t ||z_hat_t - z_t||^2 of the particle flow against the exact flow, as 1/M.
  lambda  ||grad J^lambda - grad J||, the bias of the estimator once flow and sensitivities are
          exact, computed exactly by differentiating the perturbed population recursion over a
          fixed bank of simplex draws (scripts/decomposition.py), as order lambda.
  B       E||G_hat - E G_hat||^2 at lambda*, with exact flow and sensitivities, as 1/B.

The output has the layout of scripts/verify_bounds.py, so both state spaces share the figures.

    uv run python scripts/verify_discrete_bounds.py --env all
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run as run_plan
from decomposition import MixtureRandomizer, discrete_value
from verify_discrete_eta import ENVIRONMENTS, build, oracles

MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
M_GRID = (25, 50, 100, 200, 400, 800, 1600)
LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.4)
B_GRID = (32, 64, 128, 256, 512, 1024, 2048)


def squared(estimates, exact, rows=slice(None)):
    """sum_t ||estimate_t[rows] - exact_t||^2, over t >= 1."""
    return float(sum(((estimate[rows] - target) ** 2).sum() for estimate, target in zip(estimates[1:], exact[1:])))


def row(name, sweep, value, **fields):
    return {"env": name, "sweep": sweep, "knob": sweep, "value": value, **fields}


def sweep_n(name, replications):
    rows = []
    eta, base = run_plan.auxiliary_eta(name), run_plan.TRANSPORT_ALLOCATIONS[name]["n"]
    for multiplier in MULTIPLIERS:
        n = max(2, int(round(base * multiplier)))
        algorithm = build(name, eta)
        algorithm.config = replace(algorithm.config, n_logit_gradient=n)
        initial = algorithm.env.initial_distribution
        laws, _ = algorithm.mean_field_law_flow(initial_distribution=initial)
        exact, _ = oracles(algorithm, initial)
        errors = [squared(algorithm.estimate_state_sensitivities(laws, 1000 + r, initial, eta=eta), exact,
                          slice(None, -1)) for r in range(replications)]
        rows.append(row(name, "n", n, error=float(np.mean(errors)),
                        error_se=float(np.std(errors, ddof=1) / np.sqrt(len(errors)))))
        print(f"{name:13s} n={n:<6d} error={rows[-1]['error']:.4g}", flush=True)
    return rows


def sweep_m(name, replications):
    rows = []
    algorithm = build(name, run_plan.auxiliary_eta(name))
    initial = algorithm.env.initial_distribution
    exact, _ = algorithm.mean_field_law_flow(initial_distribution=initial)
    for m in M_GRID:
        algorithm.config = replace(algorithm.config, flow="particle", n_flow_particles=m)
        errors = [squared(algorithm.particle_law_flow(algorithm.horizon, 2000 + r, initial)[0], exact)
                  for r in range(replications)]
        algorithm.config = replace(algorithm.config, flow="exact")
        rows.append(row(name, "M", m, error=float(np.mean(errors)),
                        error_se=float(np.std(errors, ddof=1) / np.sqrt(len(errors)))))
        print(f"{name:13s} M={m:<6d} error={rows[-1]['error']:.4g}", flush=True)
    return rows


def gradients(algorithm, laws, sensitivities, initial, lambda_, n_particles, replications):
    algorithm.config = replace(algorithm.config, n_particles=n_particles, lambda_=lambda_)
    return torch.stack([algorithm.batched_trajectory_gradient(laws, sensitivities, 3000 + r, initial)[0].detach()
                        for r in range(replications)])


def exact_perturbed_gradient(algorithm, initial, lambda_):
    """grad J^lambda at the current policy, exactly, over a fixed bank of simplex draws."""
    env = algorithm.env
    randomizer = MixtureRandomizer(env.n_states, algorithm.config.simplex_sigma, dtype=env.dtype)
    policy = algorithm.policy
    if isinstance(policy, torch.nn.Module):
        module = policy
        policy = lambda t, law: module(algorithm.policy_time(t), law)
    value = discrete_value(env, policy, algorithm.horizon, algorithm.discount, initial, lambda_, randomizer)
    return algorithm.flat_grad(value).detach()


def sweep_lambda_and_b(name, replications):
    rows = []
    algorithm = build(name, run_plan.auxiliary_eta(name))
    initial = algorithm.env.initial_distribution
    laws, _ = algorithm.mean_field_law_flow(initial_distribution=initial)
    exact, gradient = oracles(algorithm, initial)
    sensitivities = [torch.cat([d, -d.sum(dim=0, keepdim=True)]) for d in exact]
    norm = float(gradient.norm())
    unperturbed = exact_perturbed_gradient(algorithm, initial, 0.0)
    for lambda_ in LAMBDA_GRID:
        bias = float((exact_perturbed_gradient(algorithm, initial, lambda_) - unperturbed).norm())
        rows.append(row(name, "lambda", lambda_, bias=bias, reference_norm=norm, lambda_=lambda_))
        print(f"{name:13s} lambda={lambda_:<6g} bias={bias:.4g}", flush=True)
    lambda_ = run_plan.asymptotic_main_lambda(name)
    for b in B_GRID:
        estimates = gradients(algorithm, laws, sensitivities, initial, lambda_, b, replications)
        variance = float((estimates - estimates.mean(dim=0)).pow(2).sum(dim=1).mean())
        mse = float((estimates - gradient).pow(2).sum(dim=1).mean())
        rows.append(row(name, "B", b, variance=variance, mse=mse, reference_norm=norm, lambda_=lambda_))
        print(f"{name:13s} B={b:<6d} variance={variance:.4g}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="all", choices=list(ENVIRONMENTS) + ["all"])
    parser.add_argument("--replications", type=int, default=64)
    parser.add_argument("--output", default="results/figures/bounds/bounds_discrete.csv")
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for name in list(ENVIRONMENTS) if args.env == "all" else [args.env]:
        rows += sweep_n(name, args.replications)
        rows += sweep_m(name, args.replications)
        rows += sweep_lambda_and_b(name, args.replications)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
