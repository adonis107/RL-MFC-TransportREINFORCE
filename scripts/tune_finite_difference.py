"""Select the step of the finite-difference comparator against each benchmark's exact gradient.

The finite-difference estimator simulates, for every coordinate, the systems at theta +- h e_l and
differences their particle objectives. Its error trades the curvature bias of order h^2 against
the particle noise of order 1/(h sqrt(n0)), so h is selected the way the transport radius eta is:
by the mean-square error of the gradient against the exact gradient at a reference policy, at the
benchmark's budget. The reference is half the optimal parameter on linear-quadratic control and
the portfolio, as in scripts/verify_bounds.py, and the initialization on the bimodal benchmark.

    uv run python scripts/tune_finite_difference.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run as run_plan
from mfc.algorithms import FiniteDifference, FiniteDifferenceConfig
from mfc.environments import Bimodal, BimodalConfig, LQ, LQConfig, Portfolio, PortfolioConfig

# Environment, sign turning objective() into a reward, reference policy, candidate steps.
BENCHMARKS = {
    "lq": (lambda: LQ(LQConfig(device="cpu")), -1.0, lambda env: 0.5 * env.optimal_theta(),
           (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)),
    "portfolio": (lambda: Portfolio(PortfolioConfig(device="cpu")), 1.0, lambda env: 0.5 * env.optimal_theta(),
                  (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)),
    # Steps beyond 0.04 leave the admissible policies of the bimodal benchmark.
    "bimodal": (lambda: Bimodal(BimodalConfig(device="cpu")), 1.0, lambda env: env.zero_policy(),
                (0.005, 0.01, 0.02, 0.04)),
}


def screen(name, replications, common_random_numbers):
    factory, sign, reference, steps = BENCHMARKS[name]
    env = factory()
    theta = torch.nn.Parameter(reference(env).clone())
    exact = torch.autograd.grad(sign * env.objective(theta, lambda_=0.0), theta)[0].reshape(-1)
    rows = []
    for step in steps:
        estimator = FiniteDifference(env, policy=theta, config=FiniteDifferenceConfig(
            n_particles=run_plan.transport_per_step_budget(name), step=step,
            common_random_numbers=common_random_numbers))
        estimates = torch.stack([estimator.estimate_gradient(1000 + r)[0].reshape(-1) for r in range(replications)])
        mean = estimates.mean(dim=0)
        rows.append({
            "benchmark": name, "common_random_numbers": common_random_numbers, "step": step, "particles_per_system": estimator.particles_per_system,
            "grad_norm": float(exact.norm()),
            "bias": float((mean - exact).norm() / exact.norm()),
            "rmse": float((estimates - exact).norm(dim=1).pow(2).mean().sqrt() / exact.norm()),
        })
        print(f"{name:10s} crn={int(common_random_numbers)} h={step:<7g} n0={estimator.particles_per_system:<6d} "
              f"bias={rows[-1]['bias']:.4f} rmse={rows[-1]['rmse']:.4f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="all", choices=list(BENCHMARKS) + ["all"])
    parser.add_argument("--replications", type=int, default=64)
    parser.add_argument("--independent", action="store_true", help="independent systems, no common random numbers")
    parser.add_argument("--output", default="results/figures/finite_difference_steps.csv")
    args = parser.parse_args()
    names = list(BENCHMARKS) if args.env == "all" else [args.env]
    table = pd.DataFrame([row for name in names for row in screen(name, args.replications, not args.independent)])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"wrote {output}")
    for name, group in table.groupby("benchmark", sort=False):
        best = group.loc[group["rmse"].idxmin()]
        print(f"selected {name}: h = {best['step']:g} (relative RMSE {best['rmse']:.4f})")


if __name__ == "__main__":
    main()
