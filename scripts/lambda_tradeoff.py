"""Bias and dispersion of the transport gradient against the perturbation scale.

The correction carries a factor lambda^{-1}, so its dispersion falls as lambda grows while the
perturbation bias rises with it. Their sum has an interior minimum, and that minimum is what the
choice of lambda is trading. Measured against the exact gradient oracle at the initial policy,
which is the point at which the scale has to be chosen.

    uv run python scripts/lambda_tradeoff.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfc.environments import LQ, LQConfig, Portfolio, PortfolioConfig
from mfc.algorithms.transport import ContinuousTransport, ContinuousTransportConfig
from mfc.visualization.diagnostics import reward_gradient
from mfc.visualization.tables import save_table

SPECS = {
    "lq": (LQ, LQConfig, dict(horizon=20, n_particles=111, n_law_gradient=160,
                              n_flow_particles=150, n_components=3),
           (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)),
    "portfolio": (Portfolio, PortfolioConfig, dict(horizon=10, n_particles=211, n_law_gradient=700,
                                                   n_flow_particles=100, n_components=1),
                  (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/figures/theory/lambda_tradeoff.csv")
    args = parser.parse_args()

    rows = []
    for name, (env_class, config_class, common, lambdas) in SPECS.items():
        env = env_class(config_class(T=common["horizon"]))
        policy = nn.Parameter(env.zero_policy())
        exact = reward_gradient(env, policy, lambda_=0.0).detach().reshape(-1)
        norm = exact.norm()
        print(f"{name}: ||grad J|| = {float(norm):.4f}", flush=True)
        for lambda_ in lambdas:
            estimator = ContinuousTransport(env, policy=policy, config=ContinuousTransportConfig(
                lambda_=lambda_, eta=0.85, flow="particle", seed=args.seed, **common))
            estimates = torch.stack([
                estimator.estimate_gradient(index * 7919)[0].detach().reshape(-1)
                for index in range(args.replications)])
            mean = estimates.mean(dim=0)
            rows.append({
                "benchmark": name, "lambda": lambda_,
                "bias": float((mean - exact).norm() / norm),
                "dispersion": float((estimates - mean).norm(dim=1).pow(2).mean().sqrt() / norm),
                "rmse": float((estimates - exact).norm(dim=1).pow(2).mean().sqrt() / norm),
                "cosine": float(torch.dot(mean, exact) / (mean.norm() * norm)),
            })
            print(f"  lambda={lambda_:<7} bias={rows[-1]['bias']:.3f} "
                  f"disp={rows[-1]['dispersion']:.3f} rmse={rows[-1]['rmse']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    save_table(table, ROOT / args.output)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
