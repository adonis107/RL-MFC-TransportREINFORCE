import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

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
    "kuramoto": {"n_particles": 500, "n_gradient": 1},
}

CONTINUOUS_COMPONENTS = {
    "lq": (1, 2, 3),
    "portfolio": (1,),
    "kuramoto": (1, 2, 3),
}

TRANSPORT_LAMBDAS = (0.05, 0.1, 0.2, 0.4, 0.8)
TWOSTATE_TRANSPORT_ETAS = (0.4, 0.6, 0.85, 0.95)
DEFAULT_TRANSPORT_ETA = 0.85
TRANSPORT_AUXILIARY_GRADIENTS = {
    "cybersecurity": 51,
    "distribution": 280,
    "advertising": 65,
}

CONTINUOUS_SPLIT = {
    "lq": {"population": 150, "auxiliary": 160},
    "portfolio": {"population": 100, "auxiliary": 700},
    "kuramoto": {"population": 200, "auxiliary": 400},
}


def job(env, algorithm, horizon, flow="exact", perturbation=None, eta=None, n_components=None):
    return {
        "env": env,
        "algorithm": algorithm,
        "horizon": horizon,
        "flow": flow,
        "perturbation": perturbation,
        "eta": eta,
        "n_components": n_components,
    }


def continuous_transport_jobs(env, horizon, lambdas, components=None):
    """Transport arm of a continuous benchmark: the perturbation grid times the mixture sizes."""
    components = CONTINUOUS_COMPONENTS[env] if components is None else components
    return [
        job(env, "transport", horizon, flow="particle", perturbation=lambda_,
            eta=DEFAULT_TRANSPORT_ETA, n_components=k)
        for k in components
        for lambda_ in lambdas
    ]


def experiment_plan(env):
    if env == "twostate":
        jobs = []
        # T = 5 only: the shorter horizon separates the estimators too weakly to add anything.
        for horizon in (5,):
            jobs.append(job(env, "reinforce", horizon))
            for flow in ("exact", "particle"):
                jobs.append(job(env, "mfreinforce", horizon, flow=flow, perturbation=0.2))
                for lambda_ in TRANSPORT_LAMBDAS:
                    for eta in TWOSTATE_TRANSPORT_ETAS:
                        jobs.append(job(env, "transport", horizon, flow=flow, perturbation=lambda_, eta=eta))
        return jobs

    if env == "cybersecurity":
        jobs = [job(env, "reinforce", 3)]
        jobs.append(job(env, "mfreinforce", 3, perturbation=1.0))
        jobs.extend(
            job(env, "transport", 3, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        jobs.append(job(env, "mfqlearning", 3))
        return jobs

    if env == "distribution":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=2.0))
        jobs.extend(
            job(env, "transport", 5, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        return jobs

    if env == "advertising":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=1.0))
        jobs.extend(
            job(env, "transport", 5, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        return jobs

    if env == "lq":
        return [job(env, "reinforce", 20)] + continuous_transport_jobs(env, 20, TRANSPORT_LAMBDAS)

    if env == "portfolio":
        return [job(env, "reinforce", 10)] + continuous_transport_jobs(env, 10, (0.025, 0.05, 0.1, 0.2, 0.4))

    if env == "kuramoto":
        return [job(env, "reinforce", 20)] + continuous_transport_jobs(env, 20, TRANSPORT_LAMBDAS)

    raise ValueError(f"Unknown environment: {env}")


def continuous_budget(env, horizon):
    """Transitions per time step of one update, shared by both arms of a continuous benchmark.

    Transport spends them as M + n + B. REINFORCE has no separate population
    block, reading the law off its own particles, so it spends all of them on
    trajectories and the two arms cost the same T(M + n + B).
    """
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
    reference = reference_budget(env)
    ref_particles = reference["n_particles"]
    ref_gradient = reference["n_gradient"]
    base_cost = mf_reference_cost(env, horizon)

    parameters = {}
    if algorithm == "mfreinforce":
        selected = MF_REINFORCE_SPLITS.get(env, {})
        parameters["n_particles"] = selected.get("n_particles", ref_particles)
        parameters["n_logit_gradient"] = selected.get("n_logit_gradient", ref_gradient)
    elif algorithm == "reinforce":
        parameters["n_particles"] = (
            continuous_budget(env, horizon)
            if env in CONTINUOUS_REFERENCE
            else max(1, round(base_cost / horizon))
        )
    elif algorithm == "transport" and env in CONTINUOUS_SPLIT:
        split = CONTINUOUS_SPLIT[env]
        parameters["n_flow_particles"] = split["population"]
        parameters["n_law_gradient"] = split["auxiliary"]
        parameters["n_particles"] = max(
            1, continuous_budget(env, horizon) - split["population"] - split["auxiliary"]
        )
    elif algorithm == "transport":
        per_step_budget = base_cost / horizon
        auxiliary_gradient = TRANSPORT_AUXILIARY_GRADIENTS.get(env)
        if auxiliary_gradient is None:
            scale = per_step_budget / (ref_particles + ref_gradient)
            parameters["n_particles"] = max(1, round(scale * ref_particles))
            auxiliary_gradient = max(1, round(scale * ref_gradient))
        else:
            auxiliary_gradient = min(auxiliary_gradient, max(1, int(per_step_budget) - 1))
            parameters["n_particles"] = max(1, int(per_step_budget - auxiliary_gradient))

        if env in DISCRETE_REFERENCE:
            parameters["n_logit_gradient"] = auxiliary_gradient
        else:
            parameters["n_law_gradient"] = auxiliary_gradient
    elif algorithm == "mfqlearning":
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
        "--lr": args.lr,
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
        "--simplex-sigma": args.simplex_sigma,
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
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "kuramoto", "all"],
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

    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "kuramoto"]
    selected_envs = envs if args.env == "all" else [args.env]

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
