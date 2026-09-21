"""Reproduce the correction-alignment tables of the numerical-experiments chapter.

For every benchmark this computes two evaluation-only gradients at a common
reference policy:

    G_full  = grad_theta J(theta, mu^theta)          total derivative
    G_det   = grad_theta J(theta, mu)|_{mu=mu^theta} population channel detached

G_det is what REINFORCE targets, since it treats the population argument as
exogenous. Their alignment

    rho = <G_full, G_det> / (||G_full|| ||G_det||)

predicts whether the transport correction can change the optimization path:
rho close to +1 means the correction only rescales the step, which a
scale-invariant optimizer such as Adam cannot exploit.

Finite-state benchmarks use the exact population recursion; continuous-state
benchmarks use a pathwise rollout under common random numbers. Neither quantity
is available to the training algorithms.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

from mfc.algorithms.reinforce import Reinforce, ReinforceConfig
from mfc.visualization.constants import ENVIRONMENTS
from mfc.visualization.tables import save_table


HORIZONS = {
    "twostate": 5,
    "cybersecurity": 3,
    "distribution": 5,
    "advertising": 5,
    "lq": 20,
    "portfolio": 10,
}
KIND = {
    "twostate": "discrete",
    "cybersecurity": "discrete",
    "distribution": "discrete",
    "advertising": "discrete",
    "lq": "lq",
    "portfolio": "portfolio",
}
PATHWISE_PARTICLES = {"lq": 8192, "portfolio": 8192}


def policy_parameters(algorithm):
    if isinstance(algorithm.policy, torch.nn.Module):
        return list(algorithm.policy.parameters())
    return [algorithm.policy]


def flat_gradient(value, parameters):
    grads = torch.autograd.grad(value, parameters, allow_unused=True, materialize_grads=True)
    return torch.cat([g.flatten() for g in grads])


def reference_policy(env, horizon, steps, seed):
    """Common reference policy: REINFORCE from the benchmark initialization."""
    torch.manual_seed(seed)
    algorithm = Reinforce(
        env,
        config=ReinforceConfig(horizon=horizon, n_train=steps, validation_interval=0, seed=seed),
    )
    algorithm.train()
    return algorithm


def discrete_gradient(env, algorithm, horizon, detach):
    """Exact differentiable population recursion."""
    law = env.initial_distribution.clone()
    objective = torch.zeros((), dtype=env.dtype, device=env.device)
    discount = 1.0

    for t in range(horizon):
        argument = law.detach() if detach else law
        next_law = torch.zeros_like(law)
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            action_probabilities = env.policy(algorithm.policy, algorithm.policy_time(t), state, argument)
            for action_index in range(env.n_actions):
                action = torch.tensor(action_index, dtype=torch.long, device=env.device)
                objective = objective + discount * law[state_index] * action_probabilities[action_index] * env.reward(
                    state, argument, action
                )
                next_law = next_law + law[state_index] * action_probabilities[action_index] * env.transition(
                    state, argument, action
                )
        law = next_law / next_law.sum()
        discount = discount * algorithm.discount

    argument = law.detach() if detach else law
    for state_index in range(env.n_states):
        state = torch.tensor(state_index, dtype=torch.long, device=env.device)
        objective = objective + discount * law[state_index] * env.terminal_reward(state, argument)

    return flat_gradient(objective, policy_parameters(algorithm))


def continuous_gradient(env, algorithm, horizon, detach, kind, n_particles, seed):
    """Pathwise rollout under common random numbers."""
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    states = env.sample_initial(n_particles, generator)
    objective = torch.zeros((), dtype=env.dtype, device=env.device)

    for t in range(horizon):
        law = states.mean()
        argument = law.detach() if detach else law

        policy_noise = torch.randn(states.shape, dtype=env.dtype, device=env.device, generator=generator)
        actions = env.policy_mean(algorithm.policy, t, states, argument) + env.config.tau * policy_noise

        objective = objective + env.reward(states, argument, actions).mean()

        state_noise = torch.randn(states.shape, dtype=env.dtype, device=env.device, generator=generator)
        if kind == "lq":
            states = (
                env.config.a * states
                + env.config.b * actions
                + env.config.c * argument
                + env.config.sigma * state_noise
            )
        else:
            excess_return = env.rbar[t] + env.sigma_R[t] * state_noise
            states = env.s[t] * states + actions * excess_return

    law = states.mean()
    argument = law.detach() if detach else law
    objective = objective + env.terminal_reward(states, argument).mean()

    return flat_gradient(objective, policy_parameters(algorithm))


def alignment_row(name, env, horizon, kind, steps, seed, pathwise_seed, label=None):
    """One replication: a reference policy at `seed` and its two gradients."""
    algorithm = reference_policy(env, horizon, steps, seed)
    if kind == "discrete":
        full = discrete_gradient(env, algorithm, horizon, detach=False)
        detached = discrete_gradient(env, algorithm, horizon, detach=True)
    else:
        particles = PATHWISE_PARTICLES[kind]
        full = continuous_gradient(env, algorithm, horizon, False, kind, particles, pathwise_seed)
        detached = continuous_gradient(env, algorithm, horizon, True, kind, particles, pathwise_seed)

    full_norm = float(full.norm())
    denominator = full.norm() * detached.norm()
    cosine = float(full @ detached / denominator) if float(denominator) > 0 else float("nan")
    return {
        "benchmark": name,
        "setting": label if label is not None else "default",
        "horizon": horizon,
        "seed": seed,
        "grad_full_norm": full_norm,
        "grad_detached_norm": float(detached.norm()),
        "alignment": cosine,
        "correction_fraction": float((full - detached).norm() / max(full_norm, 1e-12)),
    }


def build_env(name, device, **overrides):
    env_class, config_class = ENVIRONMENTS[name]
    return env_class(config_class(device=device, **overrides))


def replicate(name, horizon, kind, device, seeds, steps, pathwise_seed, label=None, **overrides):
    """Repeat the diagnostic over independent reference policies."""
    rows = []
    for seed in seeds:
        env = build_env(name, device, **overrides)
        rows.append(
            alignment_row(
                name, env, horizon, kind,
                steps=steps, seed=seed, pathwise_seed=pathwise_seed + 1000 * seed, label=label,
            )
        )
    return rows


def summarize(table):
    grouped = table.groupby(["benchmark", "setting", "horizon"], sort=False, as_index=False)
    summary = grouped.agg(
        n_seeds=("seed", "count"),
        grad_full_norm=("grad_full_norm", "mean"),
        grad_detached_norm=("grad_detached_norm", "mean"),
        alignment=("alignment", "mean"),
        alignment_std=("alignment", "std"),
        alignment_min=("alignment", "min"),
        alignment_max=("alignment", "max"),
        correction_fraction=("correction_fraction", "mean"),
    )
    return summary.fillna({"alignment_std": 0.0})


def parse_seed_list(value):
    return [int(part) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-steps", type=int, default=300)
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pathwise-seed", type=int, default=4242)
    parser.add_argument("--output-root", default="results/figures")
    parser.add_argument("--latex", action="store_true", help="also print the LaTeX table bodies")
    args = parser.parse_args()

    common = dict(
        device=args.device,
        seeds=args.seeds,
        steps=args.reference_steps,
        pathwise_seed=args.pathwise_seed,
    )

    rows = []
    for name, horizon in HORIZONS.items():
        rows.extend(replicate(name, horizon, KIND[name], **common))
    main_raw = pd.DataFrame(rows)
    main_table = summarize(main_raw)

    ablation = [
        ("portfolio", "gamma=0", dict(mean_field_penalty=0.0)),
        ("portfolio", "gamma=2", dict(mean_field_penalty=2.0)),
    ]
    rows = []
    for name, label, overrides in ablation:
        rows.extend(replicate(name, HORIZONS[name], KIND[name], label=label, **common, **overrides))
    ablation_raw = pd.DataFrame(rows)
    ablation_table = summarize(ablation_raw)

    formatters = {
        "grad_full_norm": "{:.3f}".format,
        "grad_detached_norm": "{:.3f}".format,
        "alignment": "{:+.3f}".format,
        "alignment_std": "{:.3f}".format,
        "alignment_min": "{:+.3f}".format,
        "alignment_max": "{:+.3f}".format,
        "correction_fraction": "{:.3f}".format,
    }
    print("\nCorrection alignment over independent reference policies "
          f"(REINFORCE, {args.reference_steps} steps, seeds {args.seeds})\n")
    print(main_table.to_string(index=False, formatters=formatters))
    print("\nControlled pairs: explicit population dependence switched on and off\n")
    print(ablation_table.to_string(index=False, formatters=formatters))

    if args.latex:
        for title, table in (("main", main_table), ("ablation", ablation_table)):
            print(f"\n% --- {title} table body ---")
            for _, r in table.iterrows():
                setting = "" if r["setting"] == "default" else f" & {r['setting']}"
                print(f"{r['benchmark']}{setting} & {r['grad_full_norm']:.3f} & "
                      f"{r['grad_detached_norm']:.3f} & ${r['alignment']:+.3f} \\pm {r['alignment_std']:.3f}$ \\\\")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    save_table(main_table, output_root / "correction_alignment.csv")
    save_table(ablation_table, output_root / "correction_alignment_ablation.csv")
    save_table(main_raw, output_root / "correction_alignment_per_seed.csv")
    save_table(ablation_raw, output_root / "correction_alignment_ablation_per_seed.csv")
    print(f"\nwrote {output_root/'correction_alignment.csv'}")
    print(f"wrote {output_root/'correction_alignment_ablation.csv'}")


if __name__ == "__main__":
    main()
