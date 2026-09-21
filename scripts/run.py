import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

PRIMARY_ENVS = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
DISCRETE_ENVS = {"twostate", "cybersecurity", "distribution", "advertising"}
CONTINUOUS_ENVS = {"lq", "portfolio"}

DISCRETE_REFERENCE = {
    "twostate": {"n_particles": 200, "n_gradient": 10},
    "cybersecurity": {"n_particles": 200, "n_gradient": 1},
    "distribution": {"n_particles": 500, "n_gradient": 10},
    "advertising": {"n_particles": 200, "n_gradient": 10},
}

# Auxiliary split selected for MF-REINFORCE against each benchmark's gradient oracle, by the same
# procedure used to size the transport block. Its cost model is B*T + n*T*(T+1), so these respect the
# same matched budget as the published allocation they replace. Benchmarks absent here were screened
# and carried too little gradient signal for the selection to mean anything, so they keep the
# published allocation of the reference configuration.
MF_REINFORCE_SPLITS = {
    "twostate": {"n_particles": 20, "n_logit_gradient": 40},
    "distribution": {"n_particles": 260, "n_logit_gradient": 50},
}

CONTINUOUS_REFERENCE = {
    "lq": {"n_particles": 200, "n_gradient": 1},
    "portfolio": {"n_particles": 500, "n_gradient": 1},
}

CONTINUOUS_COMPONENTS = {
    "lq": (1, 2),
    "portfolio": (1,),
}

# Transport allocation from the ICLR reference table. Here M is the population
# block, n the auxiliary sensitivity block, and B the main trajectory block.
TRANSPORT_ALLOCATIONS = {
    "twostate": {
        "horizon": 5,
        "M": 0,
        "n": 12,
        "B": 248,
        "updates": 10_000,
        "lr": 1e-3,
        "simplex_sigma": 0.75,
    },
    "cybersecurity": {
        "horizon": 3,
        "M": 0,
        "n": 51,
        "B": 153,
        "updates": 20_000,
        "lr": 1e-3,
        "simplex_sigma": 1.0,
    },
    "distribution": {
        "horizon": 5,
        "M": 0,
        "n": 280,
        "B": 280,
        "updates": 30_000,
        "lr": 1e-4,
        "simplex_sigma": 0.5,
    },
    "advertising": {
        "horizon": 5,
        "M": 0,
        "n": 65,
        "B": 195,
        "updates": 10_000,
        "lr": 1e-3,
        "simplex_sigma": 1.0,
    },
    "lq": {
        "horizon": 20,
        "M": 150,
        "n": 160,
        "B": 111,
        "updates": 10_000,
        "lr": 1e-3,
        "d_theta": 40,
    },
    "portfolio": {
        "horizon": 10,
        "M": 100,
        "n": 700,
        "B": 211,
        "updates": 50_000,
        "lr": 1e-2,
        "d_theta": 20,
    },
}

BOUND_LAMBDA_MULTIPLIERS = (0.5, 1.0, 2.0)


def round_scale(value):
    """Stable command-line representation for bound-derived scales."""
    return float(f"{value:.6g}")


def effective_auxiliary_samples(env):
    allocation = TRANSPORT_ALLOCATIONS[env]
    auxiliary = allocation["n"]
    if env in CONTINUOUS_ENVS:
        shifts = 2 * allocation["d_theta"]
        return shifts * max(1, auxiliary // shifts)
    return auxiliary


def asymptotic_auxiliary_eta(env):
    exponent = 1.0 / 6.0 if env in CONTINUOUS_ENVS else 1.0 / 4.0
    return round_scale(effective_auxiliary_samples(env) ** (-exponent))


def asymptotic_main_lambda(env, multiplier=1.0):
    return round_scale(multiplier * TRANSPORT_ALLOCATIONS[env]["B"] ** (-0.25))


def bound_lambda_grid(env):
    return tuple(asymptotic_main_lambda(env, multiplier) for multiplier in BOUND_LAMBDA_MULTIPLIERS)


def transport_per_step_budget(env):
    allocation = TRANSPORT_ALLOCATIONS[env]
    return allocation["M"] + allocation["n"] + allocation["B"]


def transport_job(env, flow="exact", lambda_=None, n_components=None):
    allocation = TRANSPORT_ALLOCATIONS[env]
    return job(
        env,
        "transport",
        allocation["horizon"],
        flow=flow,
        perturbation=asymptotic_main_lambda(env) if lambda_ is None else lambda_,
        eta=asymptotic_auxiliary_eta(env),
        n_components=n_components,
        simplex_sigma=allocation.get("simplex_sigma"),
    )


def job(
    env,
    algorithm,
    horizon,
    flow="exact",
    perturbation=None,
    eta=None,
    n_components=None,
    simplex_sigma=None,
):
    return {
        "env": env,
        "algorithm": algorithm,
        "horizon": horizon,
        "flow": flow,
        "perturbation": perturbation,
        "eta": eta,
        "n_components": n_components,
        "simplex_sigma": simplex_sigma,
    }


def continuous_transport_jobs(env, components=None):
    """Continuous transport arm: bound-scale lambda multipliers times mixture sizes."""
    components = CONTINUOUS_COMPONENTS[env] if components is None else components
    return [
        transport_job(env, flow="particle", lambda_=lambda_, n_components=k)
        for k in components
        for lambda_ in bound_lambda_grid(env)
    ]


def experiment_plan(env):
    if env not in TRANSPORT_ALLOCATIONS:
        raise ValueError(f"Unknown environment: {env}")

    allocation = TRANSPORT_ALLOCATIONS[env]
    horizon = allocation["horizon"]

    if env == "twostate":
        jobs = [job(env, "reinforce", horizon)]
        for flow in ("exact", "particle"):
            jobs.append(job(env, "mfreinforce", horizon, flow=flow, perturbation=0.2))
            jobs.extend(transport_job(env, flow=flow, lambda_=lambda_) for lambda_ in bound_lambda_grid(env))
        return jobs

    if env == "cybersecurity":
        jobs = [job(env, "reinforce", horizon)]
        jobs.append(job(env, "mfreinforce", horizon, perturbation=1.0))
        jobs.extend(transport_job(env, lambda_=lambda_) for lambda_ in bound_lambda_grid(env))
        jobs.append(job(env, "mfqlearning", horizon))
        return jobs

    if env == "distribution":
        jobs = [job(env, "reinforce", horizon)]
        jobs.append(job(env, "mfreinforce", horizon, perturbation=2.0))
        jobs.extend(transport_job(env, lambda_=lambda_) for lambda_ in bound_lambda_grid(env))
        return jobs

    if env == "advertising":
        jobs = [job(env, "reinforce", horizon)]
        jobs.append(job(env, "mfreinforce", horizon, perturbation=1.0))
        jobs.extend(transport_job(env, lambda_=lambda_) for lambda_ in bound_lambda_grid(env))
        return jobs

    if env == "lq":
        return [job(env, "reinforce", horizon)] + continuous_transport_jobs(env)

    if env == "portfolio":
        return [job(env, "reinforce", horizon)] + continuous_transport_jobs(env)

    raise ValueError(f"Unknown environment: {env}")


def continuous_budget(env, horizon):
    """Transitions per time step of one update, shared by both arms of a continuous benchmark.

    Transport spends them as M + n + B. REINFORCE has no separate population
    block, reading the law off its own particles, so it spends all of them on
    trajectories and the two arms cost the same T(M + n + B).
    """
    if env in TRANSPORT_ALLOCATIONS:
        return transport_per_step_budget(env)
    return round(mf_reference_cost(env, horizon) / horizon) + CONTINUOUS_REFERENCE[env]["n_particles"]


def reference_budget(env):
    if env in DISCRETE_REFERENCE:
        return DISCRETE_REFERENCE[env]
    return CONTINUOUS_REFERENCE[env]


def mf_reference_cost(env, horizon):
    reference = reference_budget(env)
    particles = reference["n_particles"]
    gradient_samples = reference["n_gradient"]
    trajectory_cost = particles * horizon
    state_gradient_cost = 2 * gradient_samples * horizon * (horizon + 1) / 2.0
    return trajectory_cost + state_gradient_cost


def fair_run_parameters(job_spec):
    env = job_spec["env"]
    algorithm = job_spec["algorithm"]
    horizon = job_spec["horizon"]
    allocation = TRANSPORT_ALLOCATIONS.get(env)
    reference = reference_budget(env)
    ref_particles = reference["n_particles"]
    ref_gradient = reference["n_gradient"]
    base_cost = mf_reference_cost(env, horizon)

    parameters = {}
    if allocation is not None:
        parameters["n_train"] = allocation["updates"]
        parameters["lr"] = allocation["lr"]

    if algorithm == "mfreinforce":
        selected = MF_REINFORCE_SPLITS.get(env, {})
        parameters["n_particles"] = selected.get("n_particles", ref_particles)
        parameters["n_logit_gradient"] = selected.get("n_logit_gradient", ref_gradient)
    elif algorithm == "reinforce":
        if allocation is not None:
            parameters["n_particles"] = transport_per_step_budget(env)
        else:
            parameters["n_particles"] = (
                continuous_budget(env, horizon)
                if env in CONTINUOUS_REFERENCE
                else max(1, round(base_cost / horizon))
            )
    elif algorithm == "transport" and allocation is not None:
        parameters["n_particles"] = allocation["B"]
        if env in DISCRETE_ENVS:
            parameters["n_logit_gradient"] = allocation["n"]
        else:
            parameters["n_flow_particles"] = allocation["M"]
            parameters["n_law_gradient"] = allocation["n"]
    elif algorithm == "mfqlearning" and allocation is None:
        parameters["n_train"] = round(base_cost * (reference["n_train"] if "n_train" in reference else 20_000))

    if (
        job_spec["flow"] == "particle"
        and algorithm in {"mfreinforce", "transport"}
        and "n_flow_particles" not in parameters
    ):
        parameters["n_flow_particles"] = ref_particles

    return parameters


def command_for(job_spec, seed, args):
    command = [
        sys.executable,
        str(TRAIN),
        "--env",
        job_spec["env"],
        "--algorithm",
        job_spec["algorithm"],
        "--horizon",
        str(job_spec["horizon"]),
        "--flow",
        job_spec["flow"],
        "--seed",
        str(seed),
        "--results-root",
        args.results_root,
    ]

    if job_spec["perturbation"] is not None:
        command.extend(["--perturbation", str(job_spec["perturbation"])])
    if job_spec.get("n_components") is not None:
        command.extend(["--n-components", str(job_spec["n_components"])])
    if job_spec["algorithm"] == "transport":
        eta = args.eta if args.eta is not None else job_spec.get("eta")
        if eta is not None:
            command.extend(["--eta", str(eta)])

    fair_parameters = (
        fair_run_parameters(job_spec)
        if args.budget_mode == "fair"
        else {}
    )

    optional_values = {
        "--device": args.device,
        "--n-train": args.n_train if args.n_train is not None else fair_parameters.get("n_train"),
        "--lr": args.lr if args.lr is not None else fair_parameters.get("lr"),
        "--n-particles": args.n_particles if args.n_particles is not None else fair_parameters.get("n_particles"),
        "--n-logit-gradient": args.n_logit_gradient
        if args.n_logit_gradient is not None
        else fair_parameters.get("n_logit_gradient"),
        "--n-law-gradient": args.n_law_gradient
        if args.n_law_gradient is not None
        else fair_parameters.get("n_law_gradient"),
        "--n-law-particles": args.n_law_particles,
        "--n-components": args.n_components if job_spec.get("n_components") is None else None,
        "--n-flow-particles": args.n_flow_particles
        if args.n_flow_particles is not None
        else fair_parameters.get("n_flow_particles"),
        "--validation-interval": args.validation_interval,
        "--simplex-sigma": args.simplex_sigma
        if args.simplex_sigma is not None
        else job_spec.get("simplex_sigma"),
        "--simplex-resolution": args.simplex_resolution,
        "--q-learning-lr-power": args.q_learning_lr_power,
        "--q-learning-sampling": args.q_learning_sampling,
    }
    for flag, value in optional_values.items():
        if value is not None:
            command.extend([flag, str(value)])

    if args.baseline:
        command.append("--baseline")
    if args.no_baseline:
        command.append("--no-baseline")
    if args.no_reuse_state_gradient:
        command.append("--no-reuse-state-gradient")

    return command


def parse_seed_list(value):
    return [int(seed) for seed in value.split(",") if seed.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the training grid for one environment.")
    parser.add_argument(
        "--env",
        choices=PRIMARY_ENVS + ["all"],
        required=True,
    )
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--budget-mode", choices=["fair", "manual"], default="fair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=None)
    parser.add_argument("--simplex-resolution", type=int, default=None)
    parser.add_argument("--q-learning-lr-power", type=float, default=None)
    parser.add_argument("--q-learning-sampling", choices=["sweep", "iid"], default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.baseline and args.no_baseline:
        raise ValueError("Use at most one of --baseline and --no-baseline.")

    selected_envs = PRIMARY_ENVS if args.env == "all" else [args.env]

    commands = []
    for env in selected_envs:
        for job_spec in experiment_plan(env):
            for seed in args.seeds:
                commands.append(command_for(job_spec, seed, args))

    print(f"Prepared {len(commands)} training jobs.")
    for command in commands:
        print(" ".join(command))

    if args.dry_run:
        return

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
