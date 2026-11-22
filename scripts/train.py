import argparse
import importlib
import json
import subprocess
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from mfc.algorithms import (
    ContinuousTransport,
    ContinuousTransportConfig,
    GaussianTransport,
    GaussianTransportConfig,
    DiscreteTransport,
    DiscreteTransportConfig,
    FiniteDifference,
    FiniteDifferenceConfig,
    IHMFCAC,
    IHMFCACConfig,
    MFReinforce,
    MFReinforceConfig,
    Reinforce,
    ReinforceConfig,
)
from mfc.environments import (
    Advertising,
    AdvertisingConfig,
    Bimodal,
    BimodalConfig,
    Cybersecurity,
    CybersecurityConfig,
    Distribution,
    DistributionConfig,
    LQ,
    LQConfig,
    Portfolio,
    PortfolioConfig,
    TwoState,
    TwoStateConfig,
    Traffic,
    TrafficConfig,
    TrafficPolicy,
)


ENVIRONMENTS = {
    "twostate": (TwoState, TwoStateConfig),
    "traffic": (Traffic, TrafficConfig),
    "cybersecurity": (Cybersecurity, CybersecurityConfig),
    "distribution": (Distribution, DistributionConfig),
    "advertising": (Advertising, AdvertisingConfig),
    "lq": (LQ, LQConfig),
    "portfolio": (Portfolio, PortfolioConfig),
    "bimodal": (Bimodal, BimodalConfig),
}

DISCRETE_ENVS = {"traffic", "twostate", "cybersecurity", "distribution", "advertising"}
CONTINUOUS_ENVS = {"lq", "portfolio", "bimodal"}
TRANSPORT_ALGORITHMS = {"transport"}
# Transport on the Gaussian manifold with a likelihood-ratio flow sensitivity: no
# auxiliary radius, so it is not part of the eta-carrying family above.
GAUSSIAN_ALGORITHMS = {"gaussian"}


def maybe_float(value):
    if value is None or value.lower() in {"none", "na", "null"}:
        return None
    return float(value)


def update_dataclass(config, **updates):
    names = {field.name for field in fields(config)}
    kept = {key: value for key, value in updates.items() if key in names and value is not None}
    return replace(config, **kept)


def asymptotic_auxiliary_eta(env_name, auxiliary_samples):
    """Reference MSE-bound auxiliary scale for a benchmark family."""
    if auxiliary_samples is None:
        return None
    exponent = 1.0 / 6.0 if env_name in CONTINUOUS_ENVS else 1.0 / 4.0
    return float(max(1, auxiliary_samples) ** (-exponent))


def asymptotic_main_lambda(main_samples):
    """Reference MSE-bound main perturbation scale, up to unknown constants."""
    if main_samples is None:
        return None
    return float(max(1, main_samples) ** (-0.25))


def configured_auxiliary_samples(args, env):
    if args.env in DISCRETE_ENVS:
        if args.n_logit_gradient is not None:
            return args.n_logit_gradient
        return getattr(env.config, "n_logit_gradient", 10)
    if args.n_law_gradient is not None:
        return args.n_law_gradient
    if hasattr(env.config, "n_law_gradient"):
        return env.config.n_law_gradient
    return getattr(env.config, "n_logit_gradient", 10)


def default_policy_parameter_count(env):
    if hasattr(env, "zero_policy"):
        return int(env.zero_policy().numel())

    module = importlib.import_module(type(env).__module__)
    policy_class = getattr(module, f"{type(env).__name__}Policy", None)
    if policy_class is None:
        return None
    policy = policy_class(env.config)
    return sum(parameter.numel() for parameter in policy.parameters())


def effective_auxiliary_samples(args, env):
    auxiliary = configured_auxiliary_samples(args, env)
    if args.env not in CONTINUOUS_ENVS:
        return auxiliary
    parameter_count = default_policy_parameter_count(env)
    if parameter_count is None:
        return auxiliary
    shifts = 2 * parameter_count
    return shifts * max(1, auxiliary // shifts)


def asymptotic_bound_scales(args, algorithm):
    if args.algorithm not in TRANSPORT_ALGORITHMS:
        return None
    auxiliary = algorithm.n_logit_gradient if isinstance(algorithm, DiscreteTransport) else algorithm.effective_n_law_gradient
    return {
        "lambda": asymptotic_main_lambda(algorithm.n_particles),
        "eta": asymptotic_auxiliary_eta(args.env, auxiliary),
        "lambda_rate": "B^(-1/4)",
        "eta_rate": "n^(-1/6)" if args.env in CONTINUOUS_ENVS else "n^(-1/4)",
    }


def perturbation_label(algorithm, perturbation, eta, common_random_numbers=False):
    if algorithm == "finitediff":
        return f"h_{perturbation:g}" + ("_crn" if common_random_numbers else "")
    if algorithm == "gaussian":
        return f"lambda_{perturbation:g}"
    if algorithm == "reinforce":
        return "none"
    if algorithm == "ihmfcac":
        return "continuing"
    if algorithm == "mfreinforce":
        return f"eps_{perturbation:g}"
    if eta is None:
        return f"lambda_{perturbation:g}"
    return f"lambda_{perturbation:g}_eta_{eta:g}"


def output_directory(args, algorithm=None):
    eta = metadata_eta(args, algorithm) if algorithm is not None else args.eta
    label = perturbation_label(args.algorithm, args.perturbation, eta, getattr(args, "common_random_numbers", False))
    if args.algorithm == "ihmfcac":
        c = algorithm.config if algorithm is not None else None
        gamma = c.discount if c else getattr(args, "ac_discount", IHMFCACConfig.discount)
        label = f"discount_{gamma:g}"
        if args.env in CONTINUOUS_ENVS:
            m = c.state_bins if c else getattr(args, "state_bins", 6)
            a = c.action_bins if c else getattr(args, "action_bins", 7)
            label += f"_bins_{m}x{a}"
    if args.algorithm == "transport" and args.env in DISCRETE_ENVS and getattr(args, "auxiliary_batches", "shared") == "fresh":
        label = f"{label}_fresh"
    components = getattr(args, "n_components", None)
    if components is not None and args.env in CONTINUOUS_ENVS and args.algorithm == "transport":
        label = f"{label}_K_{components}"
    if args.env == "traffic":
        config = algorithm.env.config if algorithm is not None else TrafficConfig()
        reward = getattr(args, "traffic_reward", None) or config.congestion_reward
        precision = getattr(args, "traffic_precision", None) or config.precision
        if reward != "local":
            label += f"_reward_{reward}"
        if precision != "float32":
            label += "_fp64"
        family = getattr(args, "traffic_initial_laws", None) or config.training_initial_law
        probability = getattr(args, "traffic_jam_probability", None)
        probability = config.training_jam_probability if probability is None else probability
        if family != "dirichlet":
            label += f"_init_{family}"
            if family == "mixed":
                label += f"_p_{probability:g}"
    name = f"{args.algorithm}_{label}_T_{args.horizon}_{args.flow}_seed_{args.seed}"
    return Path(args.results_root) / args.env / name


def build_environment(args):
    env_class, config_class = ENVIRONMENTS[args.env]
    config = config_class()

    updates = {
        "T": args.horizon,
        "device": args.device,
        "n_train": args.n_train,
        "lr": args.lr,
        "n_particles": args.n_particles,
        "validation_interval": args.validation_interval,
        "validation_particles": args.validation_particles,
    }

    if args.env in {"advertising", "distribution", "traffic"}:
        updates["T_val"] = args.horizon
    if args.env == "traffic":
        updates["cells"] = getattr(args, "traffic_cells", None)
        updates["congestion_reward"] = getattr(args, "traffic_reward", None)
        updates["precision"] = getattr(args, "traffic_precision", None)
        updates["training_initial_law"] = getattr(args, "traffic_initial_laws", None)
        updates["training_jam_probability"] = getattr(args, "traffic_jam_probability", None)

    config = update_dataclass(config, **updates)
    return env_class(config)


def build_algorithm(args, env):
    if args.baseline and args.no_baseline:
        raise ValueError("Use at most one of --baseline and --no-baseline.")

    use_baseline = not args.no_baseline
    common = {
        "n_train": args.n_train,
        "lr": args.lr,
        "n_particles": args.n_particles,
        "horizon": args.horizon,
        "validation_interval": args.validation_interval,
        "seed": args.seed,
    }

    if args.algorithm == "reinforce":
        return Reinforce(env, config=ReinforceConfig(**common, baseline=use_baseline))

    if args.algorithm == "ihmfcac":
        if args.flow != "particle":
            raise ValueError("IH-MFC-AC estimates empirical measures; use --flow particle.")
        defaults = IHMFCACConfig()
        return IHMFCAC(env, config=IHMFCACConfig(
            n_train=args.n_train if args.n_train is not None else defaults.n_train,
            horizon=args.horizon,
            validation_interval=args.validation_interval if args.validation_interval is not None else defaults.validation_interval,
            actor_lr=args.lr if args.lr is not None else defaults.actor_lr,
            **{name: getattr(args, flag, getattr(defaults, name)) for name, flag in (
                ("discount", "ac_discount"), ("critic_lr", "critic_lr"), ("population_lr", "population_lr"),
                ("actor_power", "actor_power"), ("critic_power", "critic_power"), ("population_power", "population_power"),
                ("state_bins", "state_bins"), ("action_bins", "action_bins"),
                ("state_min", "state_min"), ("state_max", "state_max"),
                ("action_min", "action_min"), ("action_max", "action_max"),
                ("max_transitions", "max_transitions"))},
            seed=args.seed))

    if args.algorithm == "mfreinforce":
        if args.env not in DISCRETE_ENVS:
            raise ValueError("MFReinforce is only configured for discrete-state environments.")
        if args.perturbation is None:
            raise ValueError("MFReinforce requires --perturbation epsilon.")
        config = MFReinforceConfig(
            **common,
            perturbation_eta=args.perturbation,
            flow=args.flow,
            n_flow_particles=args.n_flow_particles,
            n_logit_gradient=args.n_logit_gradient,
            baseline=use_baseline,
            reuse_state_gradient=not args.no_reuse_state_gradient,
        )
        return MFReinforce(env, config=config)

    if args.algorithm == "finitediff":
        if args.env not in CONTINUOUS_ENVS:
            raise ValueError("The finite-difference comparator is configured for continuous-state environments.")
        if args.perturbation is None:
            raise ValueError("Finite differences require --perturbation h, the step.")
        config = FiniteDifferenceConfig(
            **common, step=args.perturbation, common_random_numbers=args.common_random_numbers
        )
        return FiniteDifference(env, config=config)

    if args.algorithm == "gaussian":
        if args.env not in CONTINUOUS_ENVS:
            raise ValueError("Gaussian-manifold transport is only configured for continuous-state environments.")
        if args.env == "bimodal":
            # This arm hands the environment the population mean, which the bimodal reward does not read.
            raise ValueError("Gaussian-manifold transport does not apply to the bimodal benchmark.")
        if args.perturbation is None:
            raise ValueError("Gaussian transport requires --perturbation lambda.")
        config = GaussianTransportConfig(
            **common,
            lambda_=args.perturbation,
            flow=args.flow,
            n_flow_particles=args.n_flow_particles,
            n_law_gradient=args.n_law_gradient,
            baseline=use_baseline,
        )
        return GaussianTransport(env, config=config)

    if args.algorithm == "transport":
        if args.perturbation is None:
            raise ValueError("Transport requires --perturbation lambda.")
        eta = (
            asymptotic_auxiliary_eta(args.env, effective_auxiliary_samples(args, env))
            if args.eta is None
            else args.eta
        )

        if args.env in DISCRETE_ENVS:
            config = DiscreteTransportConfig(
                **common,
                lambda_=args.perturbation,
                eta=eta,
                flow=args.flow,
                n_flow_particles=args.n_flow_particles,
                n_logit_gradient=args.n_logit_gradient,
                simplex_sigma=args.simplex_sigma,
                auxiliary_batches=getattr(args, "auxiliary_batches", "shared"),
                baseline=use_baseline,
                reuse_state_gradient=not args.no_reuse_state_gradient,
            )
            return DiscreteTransport(env, config=config)

        if args.env in CONTINUOUS_ENVS:
            config = ContinuousTransportConfig(
                **common,
                lambda_=args.perturbation,
                eta=eta,
                flow=args.flow,
                n_flow_particles=args.n_flow_particles,
                n_law_gradient=args.n_law_gradient,
                n_law_particles=args.n_law_particles,
                n_components=args.n_components
                if args.n_components is not None
                else ContinuousTransportConfig.n_components,
                baseline=use_baseline,
                reuse_state_gradient=not args.no_reuse_state_gradient,
            )
            return ContinuousTransport(env, config=config)

    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def save_policy(path, policy):
    if hasattr(policy, "export_payload"):
        payload = policy.export_payload()
    elif isinstance(policy, torch.nn.Module):
        payload = {"kind": "module_state_dict", "state_dict": policy.state_dict()}
    else:
        payload = {"kind": "tensor", "tensor": policy.detach().cpu()}
    torch.save(payload, path)


def json_ready(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def simulator_budget_estimate(algorithm):
    if isinstance(algorithm, IHMFCAC):
        return algorithm.transitions_per_iteration
    horizon = algorithm.horizon
    particles = algorithm.n_particles
    flow_extra = 0

    if getattr(algorithm.config, "flow", "exact") == "particle":
        flow_particles = getattr(algorithm, "n_flow_particles", particles)
        flow_extra = flow_particles * horizon

    if isinstance(algorithm, FiniteDifference):
        return horizon * 2 * algorithm.n_parameters * algorithm.particles_per_system

    if isinstance(algorithm, Reinforce):
        return particles * horizon

    if isinstance(algorithm, DiscreteTransport):
        gradient_samples = algorithm.n_logit_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        auxiliary_steps = horizon if algorithm.config.auxiliary_batches == "shared" else horizon * (horizon + 1) // 2
        return horizon * particles + auxiliary_steps * gradient_samples * gradient_reuses + flow_extra

    if isinstance(algorithm, GaussianTransport):
        # One auxiliary batch, no shifted sub-blocks: n trajectories of length T.
        return horizon * particles + horizon * algorithm.n_law_gradient + flow_extra

    if isinstance(algorithm, ContinuousTransport):
        gradient_samples = algorithm.effective_n_law_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        return horizon * particles + horizon * gradient_samples * gradient_reuses + flow_extra

    if isinstance(algorithm, MFReinforce):
        gradient_samples = algorithm.n_logit_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        trajectory_cost = particles * horizon
        state_gradient_cost = gradient_reuses * 2 * gradient_samples * horizon * (horizon + 1) / 2.0
        return trajectory_cost + state_gradient_cost + flow_extra

    return None


def code_version():
    """Commit of the code a run was trained with, marked dirty when the tree has local changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def metadata_eta(args, algorithm):
    if args.algorithm in {"reinforce", "gaussian", "finitediff", "ihmfcac"}:
        return None
    if args.algorithm == "mfreinforce":
        return algorithm.perturbation_eta
    if args.algorithm == "transport":
        return algorithm.eta
    return args.eta


def validation_steps(algorithm, history):
    if "validation_steps" in history:
        return history["validation_steps"]
    if not algorithm.validation_interval:
        return []
    return [
        step
        for step in range(algorithm.validation_interval, algorithm.n_train + 1, algorithm.validation_interval)
    ][: len(history["validation_objective"])]


def history_time_sum(history, key):
    values = history.get(key, [])
    return float(sum(values)) if values else 0.0


def run_training(args):
    # Argument validation of torch.distributions re-checks constraints on every draw;
    # the policies here are valid by construction.
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(args.seed)
    env = build_environment(args)
    algorithm = build_algorithm(args, env)
    started_at = time.perf_counter()
    policy, history = algorithm.train()
    elapsed_seconds = time.perf_counter() - started_at
    setup_seconds = history_time_sum(history, "setup_seconds")
    train_step_seconds = history_time_sum(history, "train_step_seconds")
    validation_seconds = history_time_sum(history, "validation_seconds")
    unaccounted_seconds = max(0.0, elapsed_seconds - setup_seconds - train_step_seconds - validation_seconds)

    out_dir = output_directory(args, algorithm)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "env": args.env,
        "algorithm": args.algorithm,
        "perturbation": args.perturbation,
        "eta": metadata_eta(args, algorithm),
        "horizon": args.horizon,
        "flow": args.flow,
        "seed": args.seed,
        "code_version": code_version(),
        # asdict() only sees dataclass fields, so an environment that derives its
        # per-step discount from a rate (cybersecurity: gamma ** dt) would not record the
        # value the algorithms actually use. Record it explicitly.
        "env_config": {key: json_ready(value) for key, value in asdict(env.config).items()},
        "resolved_discount": json_ready(algorithm.discount),
        "algorithm_config": {key: json_ready(value) for key, value in asdict(algorithm.config).items()},
        "asymptotic_bound_scales": asymptotic_bound_scales(args, algorithm),
        "effective_n_law_gradient": getattr(algorithm, "effective_n_law_gradient", None),
        "simulator_budget_estimate": simulator_budget_estimate(algorithm),
        "elapsed_seconds": elapsed_seconds,
        "setup_seconds": setup_seconds,
        "train_step_seconds": train_step_seconds,
        "validation_seconds": validation_seconds,
        "unaccounted_seconds": unaccounted_seconds,
    }
    if isinstance(algorithm, IHMFCAC):
        metadata.update(training_objective="discounted_continuing_running_reward",
                        training_discount=algorithm.config.discount,
                        evaluation_objective="benchmark_finite_horizon_with_terminal_reward",
                        training_objective_log="individual_path_one_step_reward",
                        simulator_transitions=algorithm.simulator_transitions,
                        training_iterations=algorithm.iterations)
        torch.save(algorithm.checkpoint(), out_dir / "algorithm.pt")
    history["validation_steps"] = validation_steps(algorithm, history)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with (out_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    save_policy(out_dir / "policy.pt", policy)

    summary = {
        "output_dir": str(out_dir),
        "training_iterations": getattr(algorithm, "iterations", algorithm.n_train),
        "simulator_transitions": getattr(algorithm, "simulator_transitions", None),
        "last_objective": history["objective"][-1] if history["objective"] else None,
        "last_validation_objective": history["validation_objective"][-1]
        if history["validation_objective"]
        else None,
        "elapsed_seconds": elapsed_seconds,
        "setup_seconds": setup_seconds,
        "train_step_seconds": train_step_seconds,
        "validation_seconds": validation_seconds,
        "unaccounted_seconds": unaccounted_seconds,
        "seconds_per_training_step": train_step_seconds / max(algorithm.n_train, 1),
        "wall_seconds_per_training_step": elapsed_seconds / max(algorithm.n_train, 1),
        "validation_seconds_per_call": validation_seconds / max(len(history.get("validation_seconds", [])), 1),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    return out_dir


def default_flow(args):
    """Use particle population estimates for the model-free training algorithms."""
    if args.algorithm in TRANSPORT_ALGORITHMS | GAUSSIAN_ALGORITHMS | {"mfreinforce", "ihmfcac"}:
        return "particle"
    return "exact"


def parse_args():
    parser = argparse.ArgumentParser(description="Train one MFC experiment.")
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument(
        "--algorithm",
        "--alg",
        choices=["reinforce", "mfreinforce", "transport", "gaussian", "finitediff", "ihmfcac"],
        required=True,
    )
    parser.add_argument("--perturbation", type=maybe_float, default=None)
    parser.add_argument("--eta", type=maybe_float, default=None)
    parser.add_argument("--horizon", "--T", type=int, required=True)
    parser.add_argument("--flow", choices=["exact", "particle"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--traffic-cells", type=int, default=None, help="Traffic position cells per lane")
    parser.add_argument("--traffic-reward", choices=["local", "shared"], default=None)
    parser.add_argument("--traffic-precision", choices=["float32", "float64"], default=None)
    parser.add_argument("--traffic-initial-laws", choices=["dirichlet", "jam", "mixed"], default=None)
    parser.add_argument("--traffic-jam-probability", type=float, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-particles", type=int, default=None)
    parser.add_argument("--auxiliary-batches", choices=["shared", "fresh"], default="shared",
                        help="discrete Transport: shared experiment batch or independent theorem batches")
    parser.add_argument("--simplex-sigma", type=float, default=1.0)
    parser.add_argument("--ac-discount", type=float, default=IHMFCACConfig.discount,
                        help="IH-MFC-AC continuing training discount; evaluation retains benchmark discount")
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--population-lr", type=float, default=1e-3)
    parser.add_argument("--actor-power", type=float, default=0.9)
    parser.add_argument("--critic-power", type=float, default=0.75)
    parser.add_argument("--population-power", type=float, default=0.6)
    parser.add_argument("--state-bins", type=int, default=6)
    parser.add_argument("--action-bins", type=int, default=7)
    parser.add_argument("--state-min", type=float, default=-3.0)
    parser.add_argument("--state-max", type=float, default=3.0)
    parser.add_argument("--action-min", type=float, default=-3.0)
    parser.add_argument("--action-max", type=float, default=3.0)
    parser.add_argument("--max-transitions", type=int, default=None,
                        help="IH-MFC-AC hard training transition cap; incomplete iterations are unused")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    parser.add_argument(
        "--common-random-numbers",
        action="store_true",
        help="finite differences: drive the two systems of a coordinate with the same draws",
    )
    args = parser.parse_args()
    if args.flow is None:
        args.flow = default_flow(args)
    return args


if __name__ == "__main__":
    run_training(parse_args())
