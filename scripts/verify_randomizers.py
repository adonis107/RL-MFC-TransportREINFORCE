"""Check each continuous-state closed-form J^lambda against its own simulation.

The perturbed objective of a continuous benchmark is only meaningful relative
to the randomization it is written for, and the three randomizations in this
repository differ. This script simulates each one with the code the estimator
actually runs and compares the sample mean against the closed form, reporting
the gap in standard errors.

    uv run python scripts/verify_randomizers.py --n-particles 1000000

A |z| of a few is expected from Monte Carlo alone; a closed form that does not
match its randomizer shows up as tens or hundreds.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfc.algorithms import (
    ContinuousTransport,
    ContinuousTransportConfig,
    GaussianTransport,
    GaussianTransportConfig,
)
from mfc.environments import LQ, LQConfig, Portfolio, PortfolioConfig
from mfc.environments.randomizers import LawRandomizer

# sign turns each benchmark's objective() into the reward the estimators maximize.
BENCHMARKS = {
    "lq": (lambda: LQ(LQConfig(device="cpu")), -1.0, (0.154042, 0.308084, 0.616168)),
    "portfolio": (lambda: Portfolio(PortfolioConfig(device="cpu")), 1.0, (0.13119, 0.262379, 0.524758)),
}


def rollout(env, laws, policy, n, generator):
    """Total reward of n particles shown the per-time randomized laws in `laws`."""
    states = env.sample_initial(n, generator)
    total = torch.zeros(n, dtype=env.dtype, device=env.device)
    horizon = len(laws) - 1
    for t in range(horizon):
        actions = env.sample_action(policy, t, states, laws[t], generator)
        total = total + env.reward(states, laws[t], actions)
        with torch.no_grad():
            try:
                states = env.sample(states, laws[t], actions, generator, t=t)
            except TypeError:
                states = env.sample(states, laws[t], actions, generator)
    return total + env.terminal_reward(states, laws[-1])


def mixture_laws(env, policy, lambda_, n, generator):
    """The population means Transport REINFORCE's K=1 mixture chart draws."""
    algorithm = ContinuousTransport(
        env,
        policy=policy,
        config=ContinuousTransportConfig(
            n_particles=n, n_components=1, lambda_=lambda_, eta=0.95, flow="exact"
        ),
    )
    coordinates, _ = algorithm.population_coordinates(jacobians=False)
    return [
        algorithm.population_law(
            algorithm.transport_coordinate(
                coordinate, algorithm.sample_transport_randomizers((n,), generator), lambda_
            )
        )
        for coordinate in coordinates
    ], LawRandomizer(kind="mixture", sigma=algorithm.config.mean_randomizer_sigma)


def gaussian_laws(env, policy, lambda_, n, generator):
    """The population means the Gaussian-manifold chart draws.

    Its mean randomization has the same law as the mixture chart's, so this arm
    is also the check that the two really do coincide in what the benchmark
    sees, and not only on paper.
    """
    algorithm = GaussianTransport(
        env, policy=policy, config=GaussianTransportConfig(lambda_=lambda_, flow="exact")
    )
    means, _ = algorithm.population_moments(0)
    laws = []
    for t in range(algorithm.horizon + 1):
        a, _ = algorithm.sample_randomizers(n, generator)
        laws.append(algorithm.perturbed_law(a, means[t]))
    return laws, algorithm.randomizer


def additive_laws(env, policy, lambda_, n, generator):
    """The reference's additive convention: the mean plus lambda rho noise."""
    algorithm = GaussianTransport(
        env, policy=policy, config=GaussianTransportConfig(lambda_=lambda_, flow="exact")
    )
    means, _ = algorithm.population_moments(0)
    rho = env.config.rho
    laws = [
        means[t]
        + lambda_ * rho * torch.randn(n, dtype=env.dtype, device=env.device, generator=generator)
        for t in range(algorithm.horizon + 1)
    ]
    return laws, LawRandomizer(kind="additive", sigma=rho)


ARMS = {"additive": additive_laws, "mixture": mixture_laws, "gaussian": gaussian_laws}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-particles", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tolerance", type=float, default=5.0, help="largest |z| accepted")
    args = parser.parse_args()

    worst = 0.0
    print(f"{'benchmark':<10} {'arm':<9} {'lambda':>9} {'closed form':>13} {'simulated':>22} {'z':>7}")
    for name, (factory, sign, scales) in BENCHMARKS.items():
        env = factory()
        policy = torch.nn.Parameter(0.5 * env.optimal_theta())
        for arm, build in ARMS.items():
            for lambda_ in scales:
                generator = torch.Generator(device=env.device)
                generator.manual_seed(args.seed)
                laws, randomizer = build(env, policy, lambda_, args.n_particles, generator)
                total = rollout(env, laws, policy, args.n_particles, generator)
                with torch.no_grad():
                    closed = float(sign * env.objective(policy, lambda_=lambda_, randomizer=randomizer))
                simulated = float(total.mean())
                error = float(total.std()) / math.sqrt(args.n_particles)
                z = (simulated - closed) / error
                worst = max(worst, abs(z))
                print(
                    f"{name:<10} {arm:<9} {lambda_:>9.4g} {closed:>13.4f} "
                    f"{simulated:>13.4f} +- {error:<6.4f} {z:>+7.2f}"
                )

    print(f"\nlargest |z| = {worst:.2f} (tolerance {args.tolerance})")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
