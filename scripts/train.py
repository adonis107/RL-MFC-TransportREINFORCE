import argparse
import importlib
import json
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from mfc.algorithms import (
    AdaptiveContinuousTransport,
    AdaptiveContinuousTransportConfig,
    AdaptiveDiscreteTransport,
    AdaptiveDiscreteTransportConfig,
    ContinuousTransport,
    ContinuousTransportConfig,
    DiscreteTransport,
    DiscreteTransportConfig,
    MeanFieldQLearning,
    MeanFieldQLearningConfig,
    MFReinforce,
    MFReinforceConfig,
    Reinforce,
    ReinforceConfig,
)
from mfc.environments import (
    Advertising,
    AdvertisingConfig,
    Cybersecurity,
    CybersecurityConfig,
    Distribution,
    DistributionConfig,
    Kuramoto,
    KuramotoConfig,
    LQ,
    LQConfig,
    Portfolio,
    PortfolioConfig,
    TwoState,
    TwoStateConfig,
)


ENVIRONMENTS = {
    "twostate": (TwoState, TwoStateConfig),
    "cybersecurity": (Cybersecurity, CybersecurityConfig),
    "distribution": (Distribution, DistributionConfig),
    "advertising": (Advertising, AdvertisingConfig),
    "lq": (LQ, LQConfig),
    "kuramoto": (Kuramoto, KuramotoConfig),
    "portfolio": (Portfolio, PortfolioConfig),
}

DISCRETE_ENVS = {"twostate", "cybersecurity", "distribution", "advertising"}
CONTINUOUS_ENVS = {"lq", "portfolio", "kuramoto"}
TRANSPORT_ALGORITHMS = {"transport", "adaptive_transport"}


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
    auxiliary = (
        algorithm.n_logit_gradient
        if isinstance(algorithm, (DiscreteTransport, AdaptiveDiscreteTransport))
        else algorithm.effective_n_law_gradient
    )
    return {
        "lambda": asymptotic_main_lambda(algorithm.n_particles),
        "eta": asymptotic_auxiliary_eta(args.env, auxiliary),
        "lambda_rate": "B^(-1/4)",
        "eta_rate": "n^(-1/6)" if args.env in CONTINUOUS_ENVS else "n^(-1/4)",
    }


def perturbation_label(algorithm, perturbation, eta):
    if algorithm == "reinforce":
        return "none"
    if algorithm == "mfqlearning":
        return "simplex"
    if algorithm == "mfreinforce":
        return f"eps_{perturbation:g}"
    if eta is None:
        return f"lambda_{perturbation:g}"
    return f"lambda_{perturbation:g}_eta_{eta:g}"


def output_directory(args, algorithm=None):
    eta = metadata_eta(args, algorithm) if algorithm is not None else args.eta
    label = perturbation_label(args.algorithm, args.perturbation, eta)
    if args.algorithm == "mfqlearning":
        label = f"Nm_{args.simplex_resolution}"
    components = getattr(args, "n_components", None)
    if components is not None and args.env in CONTINUOUS_ENVS and args.algorithm in TRANSPORT_ALGORITHMS:
        label = f"{label}_K_{components}"
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
    }

    if args.env in {"advertising", "distribution"}:
        updates["T_val"] = args.horizon

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

    if args.algorithm == "mfqlearning":
        if args.env != "cybersecurity":
            raise ValueError("Mean-field Q-learning is currently configured only for cybersecurity.")
        config = MeanFieldQLearningConfig(
            **common,
            learning_rate_power=args.q_learning_lr_power,
            simplex_resolution=args.simplex_resolution,
            sampling=args.q_learning_sampling,
        )
        return MeanFieldQLearning(env, config=config)

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

    if args.algorithm == "adaptive_transport":
        if args.perturbation is None:
            raise ValueError("Adaptive transport requires --perturbation initial lambda.")
        eta = (
            asymptotic_auxiliary_eta(args.env, effective_auxiliary_samples(args, env))
            if args.eta is None
            else args.eta
        )

        if args.env in DISCRETE_ENVS:
            config = AdaptiveDiscreteTransportConfig(
                **common,
                lambda_=args.perturbation,
                eta=eta,
                flow=args.flow,
                n_flow_particles=args.n_flow_particles,
                n_logit_gradient=args.n_logit_gradient,
                simplex_sigma=args.simplex_sigma,
                baseline=use_baseline,
                reuse_state_gradient=not args.no_reuse_state_gradient,
                adaptive_checkpoint_interval=args.adaptive_checkpoint_interval
                if args.adaptive_checkpoint_interval is not None
                else AdaptiveDiscreteTransportConfig.adaptive_checkpoint_interval,
                adaptive_replications=args.adaptive_replications
                if args.adaptive_replications is not None
                else AdaptiveDiscreteTransportConfig.adaptive_replications,
            )
            return AdaptiveDiscreteTransport(env, config=config)

        if args.env in CONTINUOUS_ENVS:
            config = AdaptiveContinuousTransportConfig(
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
                adaptive_checkpoint_interval=args.adaptive_checkpoint_interval
                if args.adaptive_checkpoint_interval is not None
                else AdaptiveContinuousTransportConfig.adaptive_checkpoint_interval,
                adaptive_replications=args.adaptive_replications
                if args.adaptive_replications is not None
                else AdaptiveContinuousTransportConfig.adaptive_replications,
            )
            return AdaptiveContinuousTransport(env, config=config)

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
    horizon = algorithm.horizon
    particles = algorithm.n_particles
    flow_extra = 0

    if getattr(algorithm.config, "flow", "exact") == "particle":
        flow_particles = getattr(algorithm, "n_flow_particles", particles)
        flow_extra = flow_particles * horizon

    if isinstance(algorithm, Reinforce):
        return particles * horizon

    if isinstance(algorithm, MeanFieldQLearning):
        return 1

    if isinstance(algorithm, AdaptiveDiscreteTransport):
        gradient_samples = algorithm.n_logit_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        base_cost = horizon * particles + horizon * gradient_samples * gradient_reuses
        interval = algorithm.config.adaptive_checkpoint_interval
        overhead = 0.0 if not interval else 2.0 * algorithm.config.adaptive_replications / interval
        return base_cost * (1.0 + overhead) + flow_extra

    if isinstance(algorithm, DiscreteTransport):
        gradient_samples = algorithm.n_logit_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        return horizon * particles + horizon * gradient_samples * gradient_reuses + flow_extra

    if isinstance(algorithm, AdaptiveContinuousTransport):
        gradient_samples = algorithm.effective_n_law_gradient
        gradient_reuses = 1 if algorithm.config.reuse_state_gradient else particles
        base_cost = horizon * particles + horizon * gradient_samples * gradient_reuses
        interval = algorithm.config.adaptive_checkpoint_interval
        overhead = 0.0 if not interval else 2.0 * algorithm.config.adaptive_replications / interval
        return base_cost * (1.0 + overhead) + flow_extra

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


def metadata_eta(args, algorithm):
    if args.algorithm == "reinforce":
        return None
    if args.algorithm == "mfreinforce":
        return algorithm.perturbation_eta
    if args.algorithm == "transport":
        return algorithm.eta
    if args.algorithm == "adaptive_transport":
        return algorithm.eta
    return args.eta


def validation_steps(algorithm, history):
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
    history["validation_steps"] = validation_steps(algorithm, history)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with (out_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    save_policy(out_dir / "policy.pt", policy)

    summary = {
        "output_dir": str(out_dir),
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
    """Population flow used when --flow is not given.

    The continuous mixture coordinate is fitted to a population particle block,
    so 'exact' is only available there as a single-Gaussian oracle and cannot be
    the default. Everywhere else the exact law recursion remains the default.
    """
    if args.env in CONTINUOUS_ENVS and args.algorithm in TRANSPORT_ALGORITHMS:
        return "particle"
    return "exact"


def parse_args():
    parser = argparse.ArgumentParser(description="Train one MFC experiment.")
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument(
        "--algorithm",
        "--alg",
        choices=["reinforce", "mfreinforce", "transport", "adaptive_transport", "mfqlearning"],
        required=True,
    )
    parser.add_argument("--perturbation", type=maybe_float, default=None)
    parser.add_argument("--eta", type=maybe_float, default=None)
    parser.add_argument("--horizon", "--T", type=int, required=True)
    parser.add_argument("--flow", choices=["exact", "particle"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=1.0)
    parser.add_argument("--simplex-resolution", type=int, default=30)
    parser.add_argument("--q-learning-lr-power", type=float, default=0.6)
    parser.add_argument("--q-learning-sampling", choices=["sweep", "iid"], default="sweep")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--adaptive-checkpoint-interval", type=int, default=None)
    parser.add_argument("--adaptive-replications", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    args = parser.parse_args()
    if args.flow is None:
        args.flow = default_flow(args)
    return args


if __name__ == "__main__":
    run_training(parse_args())
