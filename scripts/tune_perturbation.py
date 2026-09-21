"""Select the auxiliary perturbation scales of the discrete transport estimator.

The auxiliary block perturbs the population law to (1-eta)*mu + eta*q with q a random
simplex draw, and reweights by a score containing 1/q terms. That score is heavy-tailed:
on a ten-state benchmark its magnitude has excess kurtosis above fifty, against zero for
the Gaussian score of the logit-space perturbation used by MF-REINFORCE. eta and
simplex_sigma govern that tail, so they govern the dispersion of the estimate, and the
useful setting depends on how many states the simplex has.

Both were nevertheless fixed suite-wide at eta=0.85 and simplex_sigma=1.0. This applies
to them the same oracle-based selection already used to size the auxiliary block, so the
estimator is tuned on the same footing as the baselines it is compared against.

    uv run python scripts/tune_perturbation.py --env all

The reference policy matters. Scoring at a near-stationary policy divides by a vanishing
gradient and estimates noise, so the reference is chosen as the warm start with the largest
oracle gradient, and that norm is printed alongside every row.
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
from correction_alignment import build_env, discrete_gradient, reference_policy
from mfc.algorithms.transport import DiscreteTransport, DiscreteTransportConfig
from mfc.visualization.tables import save_table

DISCRETE = ["twostate", "cybersecurity", "distribution", "advertising"]

SETTINGS = {
    "twostate": {"horizon": 5, "lambda_": 0.1},
    "cybersecurity": {"horizon": 3, "lambda_": 0.4},
    "distribution": {"horizon": 5, "lambda_": 0.2},
    "advertising": {"horizon": 5, "lambda_": 0.2},
}

ETAS = (0.85, 0.95, 0.98)
SIGMAS = (0.5, 0.75, 1.0)
DEFAULT_CELL = (0.85, 1.0)

WARM_STARTS = (0, 100, 200, 500)


def budget_split(env_name, horizon, lambda_):
    """Main and auxiliary block sizes of the run, so the screen keeps its budget."""
    job_spec = {
        "env": env_name,
        "algorithm": "transport",
        "horizon": horizon,
        "flow": "exact",
        "perturbation": lambda_,
    }
    parameters = run_plan.fair_run_parameters(job_spec)
    return parameters["n_particles"], parameters["n_logit_gradient"]


def pick_reference(env_name, horizon, device, seed):
    """Warm start with the largest oracle gradient, with that gradient."""
    best = None
    for steps in WARM_STARTS:
        env = build_env(env_name, device)
        algorithm = reference_policy(env, horizon, steps, seed)
        exact = discrete_gradient(env, algorithm, horizon, detach=False).detach()
        if best is None or exact.norm() > best[3].norm():
            best = (env, algorithm, steps, exact)
    return best


def score(estimator, exact, replications):
    estimates = torch.stack(
        [estimator.estimate_gradient(index * 7919)[0].detach().reshape(-1) for index in range(replications)]
    )
    mean = estimates.mean(dim=0)
    norm = exact.norm()
    return {
        "bias": float((mean - exact).norm() / norm),
        "dispersion": float((estimates - mean).norm(dim=1).pow(2).mean().sqrt() / norm),
        "rmse": float((estimates - exact).norm(dim=1).pow(2).mean().sqrt() / norm),
        "cosine": float(torch.dot(mean, exact) / (mean.norm() * norm)),
    }


def screen(env_name, device, seeds, replications):
    setting = SETTINGS[env_name]
    horizon, lambda_ = setting["horizon"], setting["lambda_"]
    particles, gradient_samples = budget_split(env_name, horizon, lambda_)

    rows = []
    for seed in seeds:
        env, algorithm, steps, exact = pick_reference(env_name, horizon, device, seed)
        # Only the estimator's own randomness should vary between cells, so the initial
        # law is held fixed rather than resampled inside every gradient call.
        law = env.initial_distribution.clone()
        for eta in ETAS:
            for sigma in SIGMAS:
                estimator = DiscreteTransport(
                    env,
                    policy=algorithm.policy,
                    config=DiscreteTransportConfig(
                        n_particles=particles,
                        n_logit_gradient=gradient_samples,
                        horizon=horizon,
                        lambda_=lambda_,
                        eta=eta,
                        simplex_sigma=sigma,
                        flow="exact",
                        seed=seed,
                    ),
                )
                estimator.sample_initial_distribution = lambda _generator, _law=law: _law
                rows.append(
                    {
                        "benchmark": env_name,
                        "horizon": horizon,
                        "lambda": lambda_,
                        "eta": eta,
                        "simplex_sigma": sigma,
                        "n_particles": particles,
                        "n_logit_gradient": gradient_samples,
                        "seed": seed,
                        "reference_steps": steps,
                        "grad_norm": float(exact.norm()),
                        **score(estimator, exact, replications),
                    }
                )
    return pd.DataFrame(rows)


def summarize(table):
    return (
        table.groupby(["benchmark", "eta", "simplex_sigma"], sort=False, as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            grad_norm=("grad_norm", "mean"),
            bias=("bias", "mean"),
            dispersion=("dispersion", "mean"),
            rmse=("rmse", "mean"),
            cosine=("cosine", "mean"),
        )
    )


def print_summary(summary):
    selected = {}
    for env_name, group in summary.groupby("benchmark", sort=False):
        best = group.loc[group["rmse"].idxmin()]
        default = group[
            (group["eta"] == DEFAULT_CELL[0]) & (group["simplex_sigma"] == DEFAULT_CELL[1])
        ]
        print(f"\n=== {env_name}  ||grad J|| = {best['grad_norm']:.4f} ===")
        print(f"{'eta':>6}{'sigma':>7}{'bias':>9}{'dispersion':>12}{'rmse':>9}{'cos':>8}")
        for _, row in group.iterrows():
            marker = "  <- best" if row["rmse"] == best["rmse"] else ""
            print(
                f"{row['eta']:>6}{row['simplex_sigma']:>7}{row['bias']:>9.3f}"
                f"{row['dispersion']:>12.3f}{row['rmse']:>9.3f}{row['cosine']:>8.3f}{marker}"
            )
        if not default.empty:
            factor = float(default["rmse"].iloc[0]) / best["rmse"]
            print(f"  gain over the suite-wide default {DEFAULT_CELL}: {factor:.2f}x in gradient RMSE")
        unreliable = best["rmse"] > 1.0
        if unreliable:
            print(
                "  warning: best RMSE exceeds the gradient norm, so this benchmark carries "
                "too little signal for the screen to be informative."
            )
        selected[env_name] = (float(best["eta"]), float(best["simplex_sigma"]), unreliable)

    print("\nTRANSPORT_PERTURBATION_SCALES = {")
    for env_name, (eta, sigma, unreliable) in selected.items():
        prefix = "    # " if unreliable else "    "
        suffix = "  # no signal, not selected" if unreliable else ""
        print(f'{prefix}"{env_name}": {{"eta": {eta}, "simplex_sigma": {sigma}}},{suffix}')
    print("}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="all", choices=DISCRETE + ["all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=lambda v: [int(p) for p in v.split(",") if p.strip()], default=[0, 1, 2])
    parser.add_argument("--replications", type=int, default=120)
    parser.add_argument("--output-root", default="results/figures")
    parser.add_argument("--force", action="store_true", help="rescreen benchmarks already present in the output")
    args = parser.parse_args()

    output = Path(args.output_root) / "transport_perturbation_scales.csv"
    existing = pd.read_csv(output) if output.exists() and not args.force else pd.DataFrame()
    done = set(existing["benchmark"]) if not existing.empty else set()

    tables = [existing] if not existing.empty else []
    for env_name in DISCRETE if args.env == "all" else [args.env]:
        if env_name in done:
            print(f"skipping {env_name}: already screened in {output}")
            continue
        print(f"screening {env_name} ...", flush=True)
        tables.append(screen(env_name, args.device, args.seeds, args.replications))

    table = pd.concat(tables, ignore_index=True)
    save_table(table, output)
    print(f"wrote {output}")
    print_summary(summarize(table))


if __name__ == "__main__":
    main()
