"""Auxiliary radius eta of the finite-state transport estimator, against exact oracles.

The bias and mean-square-error bound balances an eta^2 term against 1/(n eta^2),
which suggests eta ~ n^(-1/4). The eta^2 term comes from evaluating the
prescribed-flow sensitivities at the perturbed flow, so its constant is carried
by the dependence of the kernel and policy on the population argument; the
1/(n eta^2) term comes from the population score, whose weight is
((1 - eta) / eta)^2 rather than eta^(-2). When the population coupling is weak
the minimiser therefore sits near eta = 1 rather than at n^(-1/4).

At each benchmark's transport allocation (n, T, sigma) and from its validation
initial law, this script measures over independent replications

    D-MSE   sum_t E||D_hat_t - D_t||_F^2 / sum_t ||D_t||_F^2
    G-MSE   E||G_hat - grad J||^2 / ||grad J||^2, at lambda = B^(-1/4)

for the small grid {eta*/2, eta*, 2 eta*}, eta* = n^(-1/4), and the large grid
{0.85, 0.95, 0.98}. D_t = grad_theta mu_t^theta and grad J are exact, by
automatic differentiation of the deterministic law recursion. The flow is the
exact one, so the only error left in D_hat is the auxiliary block's.

With --compare-batches, the implemented estimator, one batch of n length-T
trajectories reused for every t (cost nT), is set against the analysed one, a
fresh batch of n length-t trajectories per t (cost nT(T+1)/2).

    uv run python scripts/verify_discrete_eta.py --env all --compare-batches
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run as run_plan
from mfc.algorithms import DiscreteTransport, DiscreteTransportConfig
from mfc.environments import (
    Advertising,
    AdvertisingConfig,
    Cybersecurity,
    CybersecurityConfig,
    Distribution,
    DistributionConfig,
    TwoState,
    TwoStateConfig,
)

ENVIRONMENTS = {
    "twostate": (TwoState, TwoStateConfig),
    "cybersecurity": (Cybersecurity, CybersecurityConfig),
    "distribution": (Distribution, DistributionConfig),
    "advertising": (Advertising, AdvertisingConfig),
}
LARGE_ETAS = (0.85, 0.95, 0.98)


def eta_grid(n):
    star = n ** (-0.25)
    small = [scale * star for scale in (0.5, 1.0, 2.0) if scale * star < 1.0]
    return [("small", eta) for eta in small] + [("large", eta) for eta in LARGE_ETAS]


def build(name, eta, seed=0):
    env_class, config_class = ENVIRONMENTS[name]
    allocation = run_plan.TRANSPORT_ALLOCATIONS[name]
    env = env_class(config_class(T=allocation["horizon"], device="cpu"))
    torch.manual_seed(seed)
    config = DiscreteTransportConfig(
        n_particles=allocation["B"],
        n_logit_gradient=allocation["n"],
        horizon=allocation["horizon"],
        lambda_=run_plan.asymptotic_main_lambda(name),
        eta=eta,
        simplex_sigma=allocation["simplex_sigma"],
        flow="exact",
    )
    return DiscreteTransport(env, config=config)


def exact_laws(algorithm, initial_distribution):
    """Differentiable law recursion and objective of the frozen policy."""
    env = algorithm.env
    states = torch.arange(env.n_states, device=env.device)
    pair_states = states.repeat_interleave(env.n_actions)
    pair_actions = torch.arange(env.n_actions, device=env.device).repeat(env.n_states)
    law, laws, value, discount = initial_distribution, [initial_distribution], 0.0, 1.0
    for t in range(algorithm.horizon):
        probabilities = env.policy(algorithm.policy, algorithm.policy_time(t), states, law)
        weights = (law.unsqueeze(-1) * probabilities).reshape(-1)
        value = value + discount * (weights * env.reward(pair_states, law, pair_actions)).sum()
        law = weights @ env.transition(pair_states, law, pair_actions)
        laws.append(law)
        discount = discount * algorithm.discount
    value = value + discount * (law * env.terminal_reward(states, law)).sum()
    return laws, value


def oracles(algorithm, initial_distribution):
    laws, value = exact_laws(algorithm, initial_distribution)
    sensitivities = [torch.zeros(algorithm.env.n_states - 1, algorithm.n_parameters, dtype=algorithm.env.dtype)]
    for law in laws[1:]:
        rows = [algorithm.flat_grad(law[k], retain_graph=True) for k in range(algorithm.env.n_states - 1)]
        sensitivities.append(torch.stack(rows).detach())
    return sensitivities, algorithm.flat_grad(value).detach()


def fresh_sensitivities(algorithm, laws, seed, initial_distribution, eta):
    """The analysed estimator: batch t is new randomness, of length t, using the earlier estimates."""
    env, n = algorithm.env, algorithm.n_logit_gradient
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    sensitivities = [torch.zeros(env.n_states, algorithm.n_parameters, dtype=env.dtype)]
    factor = (1.0 - eta) / eta
    for target_t in range(1, algorithm.horizon + 1):
        states = algorithm.initial_states(n, generator, initial_distribution)
        policy_score = torch.zeros(n, algorithm.n_parameters, dtype=env.dtype)
        law_score = torch.zeros(n, algorithm.n_parameters, dtype=env.dtype)
        for s in range(target_t):
            q = algorithm.sample_simplex_batch(n, generator)
            perturbed = algorithm.perturb_law(laws[s], q, eta)
            with torch.no_grad():
                actions, _ = algorithm.sample_actions_with_log_probs(s, states, perturbed, generator)
            policy_score = policy_score + algorithm.per_sample_log_prob_gradients(s, states, perturbed, actions)
            law_score = law_score - factor * (algorithm.simplex_score_h(q) @ sensitivities[s][:-1])
            with torch.no_grad():
                states = algorithm.sample_next_state(s, states, perturbed, actions, generator)
        numerator = torch.zeros(env.n_states - 1, algorithm.n_parameters, dtype=env.dtype)
        mask = states < env.n_states - 1
        numerator.index_add_(0, states[mask], law_score[mask])
        numerator = numerator + algorithm.state_indicators(states)[:-1] @ policy_score
        current = torch.zeros(env.n_states, algorithm.n_parameters, dtype=env.dtype)
        current[:-1] = numerator / n
        current[-1] = -current[:-1].sum(dim=0)
        sensitivities.append(current)
    return sensitivities


def relative_error(estimates, oracle):
    error = sum(((estimate[:-1] - exact) ** 2).sum() for estimate, exact in zip(estimates[1:], oracle[1:]))
    return float(error / sum((exact**2).sum() for exact in oracle[1:]))


def measure(name, replications, gradient_replications, compare_batches):
    rows = []
    allocation = run_plan.TRANSPORT_ALLOCATIONS[name]
    for kind, eta in eta_grid(allocation["n"]):
        algorithm = build(name, eta)
        initial = algorithm.env.initial_distribution
        laws, _ = algorithm.mean_field_law_flow(initial_distribution=initial)
        sensitivity_oracle, gradient_oracle = oracles(algorithm, initial)

        variants = {"reused": lambda seed: algorithm.estimate_state_sensitivities(laws, seed, initial, eta=eta)}
        if compare_batches:
            variants["fresh"] = lambda seed: fresh_sensitivities(algorithm, laws, seed, initial, eta)
        for variant, estimator in variants.items():
            d_errors = [relative_error(estimator(10_000 + r), sensitivity_oracle) for r in range(replications)]
            row = {"env": name, "grid": kind, "eta": eta, "batches": variant, "n": allocation["n"],
                   "d_mse": sum(d_errors) / len(d_errors)}
            if variant == "reused":
                g_errors = []
                for r in range(gradient_replications):
                    sensitivities = estimator(50_000 + r)
                    gradient, _ = algorithm.batched_trajectory_gradient(laws, sensitivities, 90_000 + r, initial)
                    g_errors.append(float(((gradient - gradient_oracle) ** 2).sum() / (gradient_oracle**2).sum()))
                row["g_mse"] = sum(g_errors) / len(g_errors)
            rows.append(row)
            print(f"{name:13s} {kind:5s} eta={eta:.3f} {variant:6s} D-MSE={row['d_mse']:.4g}"
                  + (f" G-MSE={row['g_mse']:.4g}" if "g_mse" in row else ""), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="all", choices=list(ENVIRONMENTS) + ["all"])
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--gradient-replications", type=int, default=100)
    parser.add_argument("--compare-batches", action="store_true")
    parser.add_argument("--output", default="results/figures/discrete_eta/discrete_eta.csv")
    args = parser.parse_args()

    names = list(ENVIRONMENTS) if args.env == "all" else [args.env]
    rows = [row for name in names for row in measure(name, args.replications, args.gradient_replications,
                                                     args.compare_batches)]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
