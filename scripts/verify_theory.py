"""Numerical verification of the perturbation results, on every benchmark.

Two claims are checked, in both state spaces.

V1, perturbation estimate. In finite state space the perturbed law is
(1-lambda)*mu + lambda*psi, so d_TV(M, mu) = lambda * d_TV(psi, mu) <= lambda pathwise. We draw the
perturbations the estimator actually uses and print the largest realized ratio d_TV(M, mu)/lambda,
which says whether lambda is the true total-variation radius or a loose parametrization. In
continuous state space the reference gives a squared-Wasserstein perturbation estimate whose
root scales as sqrt(lambda), up to mixture projection error. We compute the one-dimensional W2
distance from the decoded projected law to its transport perturbation, through quantile functions,
and the W1 distance beside it.

V2, perturbation consistency. |J^lambda - J| and ||grad J^lambda - grad J|| are bounded by C*lambda.
Finite-state benchmarks carry an exact population recursion, so for a fixed draw of the perturbation
path the objective and its gradient are exact; common random numbers across lambda make the
difference smooth. The continuous mixture-transport objective has no closed-form oracle in these
benchmark environments in general. Linear-quadratic control and the portfolio read the population
only through its mean, and their J^lambda under the transport randomizer is then exact (see
mfc.environments.randomizers), so their V2 curves are exact; the bimodal benchmark has none.

    uv run python scripts/verify_theory.py --part all
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as run_plan
from correction_alignment import build_env, reference_policy
from mfc.algorithms.transport import DiscreteTransport, DiscreteTransportConfig
from mfc.visualization.tables import save_table

DISCRETE = {"twostate": 5, "cybersecurity": 3, "distribution": 5, "advertising": 5}
CONTINUOUS = {"lq": 20, "portfolio": 10, "bimodal": 1}
# Continuous benchmarks with a closed-form perturbed objective, for V2.
CONTINUOUS_EXACT = {"lq": -1.0, "portfolio": 1.0}
LAMBDAS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)
REFERENCE_STEPS = 200


def discrete_flow(env, algorithm, horizon, laws=None):
    """Exact population recursion, optionally along a given perturbed law path."""
    law = env.initial_distribution.clone()
    objective = torch.zeros((), dtype=env.dtype, device=env.device)
    discount = 1.0
    for t in range(horizon):
        argument = law if laws is None else (1.0 - laws[1]) * law + laws[1] * laws[0][t]
        nxt = torch.zeros_like(law)
        for x in range(env.n_states):
            state = torch.tensor(x, dtype=torch.long, device=env.device)
            probabilities = env.policy(algorithm.policy, algorithm.policy_time(t), state, argument)
            for a in range(env.n_actions):
                action = torch.tensor(a, dtype=torch.long, device=env.device)
                objective = objective + discount * law[x] * probabilities[a] * env.reward(state, argument, action)
                nxt = nxt + law[x] * probabilities[a] * env.transition(state, argument, action)
        law = nxt / nxt.sum()
        discount = discount * algorithm.discount
    argument = law if laws is None else (1.0 - laws[1]) * law + laws[1] * laws[0][horizon]
    for x in range(env.n_states):
        state = torch.tensor(x, dtype=torch.long, device=env.device)
        objective = objective + discount * law[x] * env.terminal_reward(state, argument)
    return objective


def flat_grad(value, algorithm):
    parameters = (
        list(algorithm.policy.parameters())
        if hasattr(algorithm.policy, "parameters")
        else [algorithm.policy]
    )
    grads = torch.autograd.grad(value, parameters, allow_unused=True, materialize_grads=True)
    return torch.cat([g.flatten() for g in grads])


def v1_discrete(name, horizon, device, seed, draws):
    env = build_env(name, device)
    algorithm = reference_policy(env, horizon, REFERENCE_STEPS, seed)
    estimator = DiscreteTransport(env, policy=algorithm.policy, config=DiscreteTransportConfig(
        horizon=horizon, simplex_sigma=run_plan.TRANSPORT_ALLOCATIONS[name]["simplex_sigma"]))
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)

    law = env.initial_distribution.clone()
    flow = [law.clone()]
    with torch.no_grad():
        for t in range(horizon):
            law = estimator.mean_field_next_law(t, law)
            flow.append(law.clone())

    rows = []
    for lambda_ in LAMBDAS:
        worst = 0.0
        for t, mu in enumerate(flow):
            q = estimator.sample_simplex_batch(draws, generator)
            # d_TV((1-l)mu + l q, mu) = l * d_TV(q, mu)
            ratio = 0.5 * (q - mu.unsqueeze(0)).abs().sum(dim=-1)
            worst = max(worst, float(ratio.max()))
        rows.append({"benchmark": name, "space": "finite", "lambda": lambda_,
                     "max_tv": lambda_ * worst, "max_ratio": worst, "holds": worst <= 1.0 + 1e-12})
    return rows


def v1_continuous(name, horizon, device, seed, draws):
    from mfc.algorithms.transport import ContinuousTransport, ContinuousTransportConfig

    env = build_env(name, device)
    algorithm = reference_policy(env, horizon, REFERENCE_STEPS, seed)
    estimator = ContinuousTransport(env, policy=algorithm.policy, config=ContinuousTransportConfig(
        horizon=horizon, flow="particle", seed=seed,
        n_components=max(run_plan.CONTINUOUS_COMPONENTS.get(name, (1,)))))
    coordinates, _ = estimator.population_coordinates(seed=seed, jacobians=False)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    grid = torch.linspace(-12.0, 12.0, 4001, dtype=env.dtype, device=env.device)
    width = float(grid[1] - grid[0])
    # Midpoint levels of (0, 1); in one dimension W2^2 is the L2 distance between quantile functions.
    levels = (torch.arange(4000, dtype=env.dtype, device=env.device) + 0.5) / 4000

    def quantile(distribution):
        # Linear interpolation of the inverse of a grid CDF.
        distribution = torch.cummax(distribution, dim=0).values
        upper = torch.searchsorted(distribution, levels).clamp(1, grid.numel() - 1)
        left, right = distribution[upper - 1], distribution[upper]
        share = ((levels - left) / (right - left).clamp_min(1e-15)).clamp(0.0, 1.0)
        return grid[upper - 1] + share * width

    def cdf(z):
        # decode, not unpack: for K = 1 the free weight block is empty and the scales are
        # the unconstrained parameters rather than positive factors.
        weights, means, scale_tril = estimator.mixture.decode(z)
        normal = torch.distributions.Normal(means.reshape(-1, 1), scale_tril.reshape(-1, 1).clamp_min(1e-9))
        return (weights.reshape(-1, 1) * normal.cdf(grid.unsqueeze(0))).sum(dim=0)

    rows = []
    for lambda_ in LAMBDAS:
        squared_w1, squared_w2 = [], []
        for z in coordinates:
            base = cdf(z)
            base_quantile = quantile(base)
            randomizers = estimator.sample_transport_randomizers((draws,), generator)
            for randomizer in randomizers:
                perturbed = cdf(estimator.transport_coordinate(z, randomizer, lambda_))
                squared_w1.append(float(((perturbed - base).abs().sum() * width) ** 2))
                squared_w2.append(float(((quantile(perturbed) - base_quantile) ** 2).mean()))
        w1 = (sum(squared_w1) / len(squared_w1)) ** 0.5
        w2 = (sum(squared_w2) / len(squared_w2)) ** 0.5
        rows.append({"benchmark": name, "space": "continuous", "lambda": lambda_,
                     "w1_rms": w1, "w2_rms": w2, "max_ratio": w2 / lambda_ ** 0.5, "holds": True})
    return rows


def v2_discrete(name, horizon, device, seed, paths):
    env = build_env(name, device)
    algorithm = reference_policy(env, horizon, REFERENCE_STEPS, seed)
    estimator = DiscreteTransport(env, policy=algorithm.policy, config=DiscreteTransportConfig(
        horizon=horizon, simplex_sigma=run_plan.TRANSPORT_ALLOCATIONS[name]["simplex_sigma"]))
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed + 5000)
    # Common random numbers: one bank of perturbation paths reused at every lambda.
    bank = [[estimator.sample_simplex(generator) for _ in range(horizon + 1)] for _ in range(paths)]

    base_value = discrete_flow(env, algorithm, horizon)
    base_grad = flat_grad(base_value, algorithm)
    base_value = float(base_value)

    rows = []
    for lambda_ in LAMBDAS:
        total = torch.zeros((), dtype=env.dtype, device=env.device)
        for psi in bank:
            total = total + discrete_flow(env, algorithm, horizon, laws=(psi, lambda_))
        total = total / paths
        gradient = flat_grad(total, algorithm)
        objective_gap = abs(float(total) - base_value)
        gradient_gap = float((gradient - base_grad).norm())
        rows.append({"benchmark": name, "space": "finite", "lambda": lambda_,
                     "objective_gap": objective_gap, "gradient_gap": gradient_gap,
                     "objective_over_lambda": objective_gap / lambda_,
                     "gradient_over_lambda": gradient_gap / lambda_,
                     "objective_over_lambda2": objective_gap / lambda_ ** 2})
    return rows


def v2_continuous(name, horizon, device, seed):
    """Exact |J^lambda - J| and ||grad J^lambda - grad J|| under the transport mean randomizer."""
    from mfc.environments.randomizers import LawRandomizer

    if name not in CONTINUOUS_EXACT:
        return []
    env = build_env(name, device)
    algorithm = reference_policy(env, horizon, REFERENCE_STEPS, seed)
    theta = algorithm.policy.detach().clone().requires_grad_(True)
    randomizer = LawRandomizer(kind="mixture", sigma=1.0)
    base = env.objective(theta, lambda_=0.0)
    base_grad = torch.autograd.grad(base, theta)[0]
    rows = []
    for lambda_ in LAMBDAS:
        value = env.objective(theta, lambda_=lambda_, randomizer=randomizer)
        gradient = torch.autograd.grad(value, theta)[0]
        objective_gap = abs(float(value - base))
        gradient_gap = float((gradient - base_grad).norm())
        rows.append({"benchmark": name, "space": "continuous", "lambda": lambda_,
                     "objective_gap": objective_gap, "gradient_gap": gradient_gap,
                     "objective_over_lambda": objective_gap / lambda_,
                     "gradient_over_lambda": gradient_gap / lambda_,
                     "objective_over_lambda2": objective_gap / lambda_ ** 2})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", default="all", choices=["v1", "v2", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--continuous-draws", type=int, default=200,
                        help="transport randomizers per time step in the continuous V1 estimate")
    parser.add_argument("--paths", type=int, default=200)
    parser.add_argument("--output-root", default="results/figures/theory")
    args = parser.parse_args()
    output = Path(args.output_root)

    if args.part in {"v1", "all"}:
        rows = []
        for name, horizon in DISCRETE.items():
            print(f"V1 {name} ...", flush=True)
            rows += v1_discrete(name, horizon, args.device, args.seed, args.draws)
        for name, horizon in CONTINUOUS.items():
            print(f"V1 {name} ...", flush=True)
            rows += v1_continuous(name, horizon, args.device, args.seed, args.continuous_draws)
        table = pd.DataFrame(rows)
        save_table(table, output / "perturbation_estimate.csv")
        print("\n=== V1 perturbation estimate ===")
        print(table.to_string(index=False))

    if args.part in {"v2", "all"}:
        rows = []
        for name, horizon in DISCRETE.items():
            print(f"V2 {name} ...", flush=True)
            rows += v2_discrete(name, horizon, args.device, args.seed, args.paths)
        for name, horizon in CONTINUOUS.items():
            if name not in CONTINUOUS_EXACT:
                print(f"V2 {name} skipped: no closed-form perturbed objective.", flush=True)
                continue
            print(f"V2 {name} ...", flush=True)
            rows += v2_continuous(name, horizon, args.device, args.seed)
        table = pd.DataFrame(rows)
        save_table(table, output / "perturbation_consistency.csv")
        print("\n=== V2 perturbation consistency ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
