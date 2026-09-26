"""Empirical check of the bias and mean-square-error bounds of the transport estimator.

The reference proves two bounds for the continuous-state estimator. For the
auxiliary block,

    E||D_hat - D||^2  <=  C (eta^4 + 1/(n eta^2)),                          (P)

and for the gradient estimate, with eps_z the population fitting error and
eps_D the square root of (P),

    E||G_hat - grad J_K||^2  <=  C (lambda^2 + eps_z^2 + eps_D^2)
                                 + C (1 + lambda^-2) / B.                   (T)

Each sweep below isolates one term by making the others negligible, which is
what makes the exponents readable rather than a sum of confounded effects:

  eta         (P), both terms.  flow='particle'.  A U-curve whose minimum the
              bound places at eta ~ n^(-1/6).
  eta-exact   (P), Taylor term alone.  flow='exact' makes the shifted flows
              analytic, so 1/(n eta^2) vanishes and the error is pure eta^4.
  n           (P), sampling term alone.  eta held fixed, so the eta^4 floor is
              a constant and the excess above it should fall as 1/n.
  lambda      (T), perturbation term alone.  flow='exact' and the oracle
              sensitivity give eps_z = eps_D = 0, so the bias is O(lambda).
  B           (T), variance term alone.  The conditional variance should fall
              as 1/B, with the prefactor growing like lambda^-2 between the two
              scales tested.
  M           (T), population term alone.  eps_z^2 should fall as 1/M.

On both continuous benchmarks the reward reads the population only through its
mean, and the K=1 chart reproduces the mean of its fitting set exactly, so the
projected objective J_K coincides with J and the oracle gradient is the
autograd gradient of the analytic objective. The coordinate is z_t =
(m_t, log sigma_t), so the oracle sensitivity is read off the analytic moment
recursion.

    uv run python scripts/verify_bounds.py --env lq --sweep all
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfc.algorithms.transport import ContinuousTransport, ContinuousTransportConfig
from mfc.environments import LQ, LQConfig, Portfolio, PortfolioConfig

# sign turns each benchmark's objective() into the reward the estimator maximizes.
BENCHMARKS = {
    "lq": (lambda horizon: LQ(LQConfig(T=horizon, device="cpu")), -1.0, 20, 0.154042, 150, 10240, 111),
    "portfolio": (lambda horizon: Portfolio(PortfolioConfig(T=horizon, device="cpu")), 1.0, 10, 0.13119, 100, 700, 211),
}

ETA_GRID = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95)
def auxiliary_grid(n, d_theta):
    """Halvings of the benchmark's own auxiliary budget.

    The estimator allocates n across 2 d_theta centered shifts, so a grid in
    absolute counts would mean different things on benchmarks with different
    parameter counts. Halving the benchmark's own n keeps every point a
    realizable allocation and stops once a shift would carry fewer than two
    particles.
    """
    grid, current = [], n
    while current >= 4 * d_theta:
        grid.append(int(current))
        current //= 2
    return tuple(reversed(grid))
LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
B_GRID = (32, 64, 128, 256, 512, 1024, 2048)
M_GRID = (25, 50, 100, 200, 400, 800, 1600)


def reference_policy(env, fraction):
    """A deliberately non-stationary policy.

    Scoring at the optimum divides by a vanishing gradient and measures noise,
    so the reference is a fixed fraction of the optimal parameter, which keeps
    the oracle gradient well away from zero.
    """
    return torch.nn.Parameter(fraction * env.optimal_theta())


def oracle_gradient(env, policy, sign):
    value = sign * env.objective(policy, lambda_=0.0)
    return torch.autograd.grad(value, policy)[0].reshape(-1).detach()


def oracle_coordinates(env, policy, horizon):
    """z_t = (m_t, log sigma_t) and D_t = grad_theta z_t, from the analytic flow."""
    means, variances = env.moment_flow(policy, lambda_=0.0)
    coordinates, sensitivities = [], []
    for t in range(horizon + 1):
        log_deviation = 0.5 * torch.log(variances[t])
        coordinates.append(torch.stack([means[t], log_deviation]).detach())
        rows = []
        for component in (means[t], log_deviation):
            grad = torch.autograd.grad(component, policy, retain_graph=True, allow_unused=True)[0]
            rows.append(torch.zeros(policy.numel(), dtype=env.dtype) if grad is None else grad.reshape(-1))
        sensitivities.append(torch.stack(rows).detach())
    return coordinates, sensitivities


def build(env, policy, lambda_, eta, flow, n_particles, n_law_gradient, n_flow_particles, horizon):
    return ContinuousTransport(
        env,
        policy=policy,
        config=ContinuousTransportConfig(
            n_particles=n_particles,
            n_law_gradient=n_law_gradient,
            n_flow_particles=n_flow_particles,
            horizon=horizon,
            lambda_=lambda_,
            eta=eta,
            n_components=1,
            flow=flow,
            seed=0,
        ),
    )


def sensitivity_error(algorithm, oracle_D, replications, seed=0):
    """E||D_hat - D||^2, summed over time, averaged over replications."""
    errors = []
    for replication in range(replications):
        offset = seed + 1_000 * replication
        coordinates, _ = algorithm.population_coordinates(seed=offset + 20_000, jacobians=False)
        estimate = algorithm.estimate_coordinate_sensitivities(coordinates, seed=offset + 10_000)
        errors.append(sum(float((estimate[t] - oracle_D[t]).square().sum()) for t in range(len(oracle_D))))
    return float(np.mean(errors)), float(np.std(errors, ddof=1) / math.sqrt(replications))


def coordinate_error(algorithm, oracle_z, replications, seed=0):
    """E||z_hat - z||^2, summed over time."""
    errors = []
    for replication in range(replications):
        coordinates, _ = algorithm.population_coordinates(
            seed=seed + 1_000 * replication + 20_000, jacobians=False
        )
        errors.append(sum(float((coordinates[t] - oracle_z[t]).square().sum()) for t in range(len(oracle_z))))
    return float(np.mean(errors)), float(np.std(errors, ddof=1) / math.sqrt(replications))


def gradient_statistics(algorithm, reference, replications, oracle_D=None, seed=0):
    """Bias, mean-square error and conditional variance of the gradient estimate.

    Passing oracle_D replaces the auxiliary block by the exact sensitivity,
    which removes eps_D from the bound and leaves the perturbation term alone.
    """
    estimates = []
    for replication in range(replications):
        offset = seed + 1_000 * replication
        coordinates, _ = algorithm.population_coordinates(seed=offset + 20_000, jacobians=False)
        sensitivities = (
            oracle_D
            if oracle_D is not None
            else algorithm.estimate_coordinate_sensitivities(coordinates, seed=offset + 10_000)
        )
        gradient, _ = algorithm.batched_trajectory_gradient(coordinates, sensitivities, offset)
        estimates.append(gradient.detach())
    stacked = torch.stack(estimates)
    mean = stacked.mean(dim=0)
    bias = float((mean - reference).norm())
    mse = float((stacked - reference).square().sum(dim=1).mean())
    variance = float((stacked - mean).square().sum(dim=1).mean())
    return bias, mse, variance, float(reference.norm())


def log_slope(x, y):
    """Least-squares exponent of y against x on a log-log scale."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def run_sweeps(name, args):
    factory, sign, horizon, lambda_default, M, n, B = BENCHMARKS[name]
    horizon = args.horizon or horizon
    env = factory(horizon)
    policy = reference_policy(env, args.policy_fraction)
    reference = oracle_gradient(env, policy, sign)
    oracle_z, oracle_D = oracle_coordinates(env, policy, horizon)
    eta_default = 0.95
    d_theta = policy.numel()
    n_grid = auxiliary_grid(n, d_theta)
    replications_b = args.replications_b or args.replications_g

    print(f"\n===== {name}  T={horizon}  |grad J| = {float(reference.norm()):.6f}")
    print(f"      reference allocation M={M} n={n} B={B}, lambda={lambda_default}, eta={eta_default}")
    rows = []
    selected = set(args.sweep) if "all" not in args.sweep else {"eta", "eta-exact", "n", "lambda", "B", "M"}

    def record(sweep, knob, value, **fields):
        rows.append({"env": name, "sweep": sweep, "knob": knob, "value": value, **fields})

    for sweep, flow, grid, label in (
        ("eta", "particle", ETA_GRID, "eta"),
        ("eta-exact", "exact", ETA_GRID, "eta"),
    ):
        if sweep not in selected:
            continue
        print(f"\n-- {sweep}: E||D_hat - D||^2 at n={n}, flow={flow}")
        errors = []
        for eta in grid:
            algorithm = build(env, policy, lambda_default, eta, flow, B, n, M, horizon)
            error, deviation = sensitivity_error(algorithm, oracle_D, args.replications_d)
            errors.append(error)
            record(sweep, label, eta, error=error, error_se=deviation)
            print(f"   eta={eta:<6} E||D_hat-D||^2 = {error:12.5e} +- {deviation:.1e}")
        if sweep == "eta-exact":
            print(f"   fitted exponent in eta: {log_slope(grid, errors):+.2f}   (bound: +4)")
        else:
            best = grid[int(np.argmin(errors))]
            print(f"   minimiser eta = {best}   (bound's rule n^(-1/6) = {n ** (-1 / 6):.3f})")

    if "n" in selected:
        print(f"\n-- n: E||D_hat - D||^2 at eta={eta_default}, flow=particle")
        errors = []
        for auxiliary in n_grid:
            algorithm = build(env, policy, lambda_default, eta_default, "particle", B, auxiliary, M, horizon)
            error, deviation = sensitivity_error(algorithm, oracle_D, args.replications_d)
            errors.append(error)
            record("n", "n", auxiliary, error=error, error_se=deviation)
            print(f"   n={auxiliary:<7} E||D_hat-D||^2 = {error:12.5e} +- {deviation:.1e}")
        floor = min(errors)
        excess = [error - floor for error in errors[:-1]]
        print(f"   fitted exponent in n, total  : {log_slope(n_grid, errors):+.2f}   (bound: -1 above the eta^4 floor)")
        print(f"   fitted exponent in n, excess : {log_slope(n_grid[:-1], excess):+.2f}")

    if "lambda" in selected:
        print(f"\n-- lambda: bias at B={args.large_b}, flow=exact, oracle sensitivity")
        biases = []
        for scale in LAMBDA_GRID:
            algorithm = build(env, policy, scale, eta_default, "exact", args.large_b, n, M, horizon)
            bias, mse, variance, norm = gradient_statistics(
                algorithm, reference, args.replications_g, oracle_D=oracle_D
            )
            biases.append(bias)
            # A bias is a norm of an R-sample mean, so it cannot be read below
            # about sqrt(var / R); print that resolution next to it.
            resolution = math.sqrt(variance / args.replications_g)
            record("lambda", "lambda", scale, bias=bias, mse=mse, variance=variance,
                   reference_norm=norm, bias_resolution=resolution)
            print(f"   lambda={scale:<6} bias = {bias:10.5f} +- {resolution:8.5f}"
                  f"   relative = {bias / norm:8.5f}   bias/resolution = {bias / resolution:6.2f}")
        # Once the relative bias approaches one a linear rate is no longer
        # meaningful, so the exponent is also fitted on the lower half.
        lower = [scale for scale in LAMBDA_GRID if scale <= LAMBDA_GRID[len(LAMBDA_GRID) // 2 - 1]]
        print(f"   fitted exponent in lambda, full grid   : {log_slope(LAMBDA_GRID, biases):+.2f}   (bound: +1)")
        print(f"   fitted exponent in lambda, small scales: {log_slope(lower, biases[:len(lower)]):+.2f}")

    if "B" in selected:
        for scale in (lambda_default, 2.0 * lambda_default):
            print(f"\n-- B: conditional variance at lambda={scale:g}, flow=exact, oracle sensitivity")
            variances = []
            for particles in B_GRID:
                algorithm = build(env, policy, scale, eta_default, "exact", particles, n, M, horizon)
                bias, mse, variance, norm = gradient_statistics(
                    algorithm, reference, replications_b, oracle_D=oracle_D
                )
                variances.append(variance)
                resolution = math.sqrt(variance / replications_b)
                record("B", "B", particles, lambda_=scale, bias=bias, mse=mse,
                       variance=variance, bias_resolution=resolution)
                print(f"   B={particles:<6} var = {variance:12.5e}   mse = {mse:12.5e}"
                      f"   bias = {bias:9.5f} +- {resolution:.5f}")
            print(f"   fitted exponent in B: {log_slope(B_GRID, variances):+.2f}   (bound: -1)")

    if "M" in selected:
        print(f"\n-- M: E||z_hat - z||^2, flow=particle")
        errors = []
        for particles in M_GRID:
            algorithm = build(env, policy, lambda_default, eta_default, "particle", B, n, particles, horizon)
            error, deviation = coordinate_error(algorithm, oracle_z, args.replications_d)
            errors.append(error)
            record("M", "M", particles, error=error, error_se=deviation)
            print(f"   M={particles:<6} E||z_hat-z||^2 = {error:12.5e} +- {deviation:.1e}")
        print(f"   fitted exponent in M: {log_slope(M_GRID, errors):+.2f}   (bound: -1)")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="lq", choices=list(BENCHMARKS) + ["all"])
    parser.add_argument(
        "--sweep",
        default=["all"],
        nargs="+",
        choices=["all", "eta", "eta-exact", "n", "lambda", "B", "M"],
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--policy-fraction", type=float, default=0.5)
    parser.add_argument("--replications-d", type=int, default=16)
    parser.add_argument("--replications-g", type=int, default=64)
    parser.add_argument(
        "--replications-b",
        type=int,
        default=None,
        help="replications for the B sweep, whose bias column needs many more than the rest",
    )
    parser.add_argument("--large-b", type=int, default=4096)
    parser.add_argument("--output", default="results/figures/bounds")
    args = parser.parse_args()

    names = list(BENCHMARKS) if args.env == "all" else [args.env]
    rows = []
    for name in names:
        rows.extend(run_sweeps(name, args))

    table = pd.DataFrame(rows)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "bounds.csv"
    table.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
