"""Consistency check of the Gaussian-manifold transport estimator.

Both continuous benchmarks expose the perturbed objective J^lambda of the
Gaussian-manifold randomizer in closed form, so its gradient is available by
autograd. This script compares three things against that reference at a fixed
policy:

  1. the represented flow (m_t, sigma_t) the population block fits;
  2. the flow sensitivities grad_theta m_t and grad_theta log sigma_t, against
     autograd of the analytic moment recursion at lambda = 0;
  3. the full gradient estimate, averaged over independent replications.

Run it as
    uv run python scripts/verify_gaussian.py --env lq --n-law-gradient 4096
"""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfc.algorithms import GaussianTransport, GaussianTransportConfig
from mfc.environments import LQ, LQConfig, Portfolio, PortfolioConfig

ENVIRONMENTS = {
    "lq": (LQ, LQConfig, -1.0),
    "portfolio": (Portfolio, PortfolioConfig, 1.0),
}


def score_identity_error(env, algorithm, policy, lambda_, horizon, seed=3):
    """Largest relative gap between the Corollary score and plain autograd.

    log q_t^{lambda,theta}(m, Sigma) is written out directly from the density
    lemma at a randomized coordinate held fixed, and differentiated through the
    analytic flow (m_t^theta, sigma_t^theta). The closed form used by the
    estimator contracts the same gradient with the two sensitivities instead,
    so the two must agree to rounding.
    """
    config = algorithm.config
    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    for t in {1, horizon // 4, horizon // 2, horizon} - {0}:
        a, b = algorithm.sample_randomizers(4, generator)
        for index in range(4):
            means, variances = env.moment_flow(policy, lambda_=0.0)
            mean, deviation = means[t], variances[t].sqrt()
            randomized_mean = ((1.0 - lambda_) * mean + lambda_ * a[index]).detach()
            randomized_variance = (
                ((1.0 - lambda_) + lambda_ * b[index]) ** 2 * deviation**2
            ).detach()
            # The inverse maps of the density lemma, as functions of theta.
            target_mean = (randomized_mean - (1.0 - lambda_) * mean) / lambda_
            target_scale = (randomized_variance.sqrt() / deviation - (1.0 - lambda_)) / lambda_
            log_density = (
                -0.5 * ((target_mean - config.mean_randomizer_mean) / config.mean_randomizer_sigma) ** 2
                - 0.5 * ((target_scale - config.scale_randomizer_mean) / config.scale_randomizer_sigma) ** 2
                - math.log(config.mean_randomizer_sigma * config.scale_randomizer_sigma)
                - math.log(2.0 * lambda_**2)
                - torch.log(deviation)
                - 0.5 * torch.log(randomized_variance)
            )
            reference = torch.autograd.grad(log_density, policy, retain_graph=True)[0].reshape(-1)
            mean_gradient = torch.autograd.grad(means[t], policy, retain_graph=True)[0].reshape(-1)
            log_deviation_gradient = (
                0.5
                * torch.autograd.grad(variances[t], policy, retain_graph=True)[0].reshape(-1)
                / variances[t].detach()
            )
            c_sigma, c_mean = algorithm.score_coefficients(a[index], b[index])
            closed_form = c_sigma * log_deviation_gradient + c_mean * mean_gradient
            worst = max(worst, relative_error(closed_form, reference))
    return worst


def relative_error(estimate, reference):
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-30))


def cosine(estimate, reference):
    return float(
        (estimate * reference).sum()
        / (estimate.norm() * reference.norm()).clamp_min(1e-30)
    )


def exact_flow_sensitivities(env, theta, horizon):
    """Autograd reference for grad_theta m_t and grad_theta log sigma_t."""
    means, variances = env.moment_flow(theta, lambda_=0.0)
    mean_gradients, log_deviation_gradients = [], []
    for t in range(horizon + 1):
        mean_gradients.append(
            torch.autograd.grad(means[t], theta, retain_graph=True, allow_unused=True)[0].reshape(-1)
        )
        log_deviation_gradients.append(
            0.5
            * torch.autograd.grad(variances[t], theta, retain_graph=True, allow_unused=True)[0].reshape(-1)
            / variances[t].detach()
        )
    return mean_gradients, log_deviation_gradients, means.detach(), variances.detach().sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=ENVIRONMENTS, default="lq")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--perturbation", type=float, default=0.3)
    parser.add_argument("--n-particles", type=int, default=2048)
    parser.add_argument("--n-law-gradient", type=int, default=4096)
    parser.add_argument("--n-flow-particles", type=int, default=4096)
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--flow", choices=["exact", "particle"], default="particle")
    parser.add_argument("--policy", choices=["zero", "optimal", "half"], default="half")
    args = parser.parse_args()

    env_class, config_class, sign = ENVIRONMENTS[args.env]
    config = config_class(device="cpu")
    if args.horizon is not None:
        config = config_class(device="cpu", T=args.horizon)
    env = env_class(config)
    horizon = config.T

    if args.policy == "zero":
        theta = env.zero_policy()
    elif args.policy == "optimal":
        theta = env.optimal_theta()
    else:
        theta = 0.5 * env.optimal_theta()
    policy = torch.nn.Parameter(theta.clone())

    algorithm_config = GaussianTransportConfig(
        n_particles=args.n_particles,
        n_law_gradient=args.n_law_gradient,
        n_flow_particles=args.n_flow_particles,
        horizon=horizon,
        lambda_=args.perturbation,
        flow=args.flow,
        seed=0,
    )
    algorithm = GaussianTransport(env, policy=policy, config=algorithm_config)

    # The perturbed objective is a reward in every benchmark's sign convention
    # used by the estimator, so LQ's cost is negated here as it is in training.
    reference_objective = sign * env.objective(
        policy, lambda_=args.perturbation, randomizer=algorithm.randomizer
    )
    reference = torch.autograd.grad(reference_objective, policy)[0].reshape(-1)

    print(f"env={args.env}  T={horizon}  lambda={args.perturbation}  policy={args.policy}")
    print(f"J^lambda(theta) = {float(reference_objective):.6f}   |grad| = {float(reference.norm()):.6f}")

    print(
        "\nCorollary score against autograd: max relative gap = "
        f"{score_identity_error(env, algorithm, policy, args.perturbation, horizon):.2e}"
    )

    exact_means, exact_log_deviations, means, deviations = exact_flow_sensitivities(
        env, policy, horizon
    )

    fitted_means, fitted_deviations = algorithm.population_moments(12345)
    print(
        "\nrepresented flow: max |m_t - m_t^exact| = "
        f"{float((fitted_means - means).abs().max()):.3e}"
        f"   max |sigma_t - sigma_t^exact| = {float((fitted_deviations - deviations).abs().max()):.3e}"
    )

    mean_gradients, log_deviation_gradients = algorithm.flow_sensitivities(
        fitted_means.detach(), fitted_deviations.detach(), 999
    )
    print("\n  t   rel.err grad m_t   cos   |   rel.err grad log sigma_t   cos")
    for t in range(0, horizon + 1, max(1, horizon // 5)):
        if t == 0:
            continue
        print(
            f"{t:3d}   {relative_error(mean_gradients[t], exact_means[t]):>13.4f}"
            f"  {cosine(mean_gradients[t], exact_means[t]):>6.3f}   |"
            f"   {relative_error(log_deviation_gradients[t], exact_log_deviations[t]):>22.4f}"
            f"  {cosine(log_deviation_gradients[t], exact_log_deviations[t]):>6.3f}"
        )

    estimates = []
    for replication in range(args.replications):
        estimates.append(algorithm.estimate_gradient(7_000_000 + 31 * replication)[0])
    stacked = torch.stack(estimates)
    average = stacked.mean(dim=0)
    print(
        f"\ngradient over {args.replications} replications:"
        f"\n  relative error of the mean : {relative_error(average, reference):.4f}"
        f"\n  cosine of the mean         : {cosine(average, reference):.4f}"
        f"\n  mean per-replication cosine: {sum(cosine(e, reference) for e in estimates) / len(estimates):.4f}"
        f"\n  |mean| / |reference|       : {float(average.norm() / reference.norm()):.4f}"
    )


if __name__ == "__main__":
    main()
