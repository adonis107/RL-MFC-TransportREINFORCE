"""Plot one traffic run's optimality gap using a convex occupancy-measure reference.

The transition kernel is population independent. The negative reward is a
convex quadratic in state-action occupancies when the symmetric risk matrix is
positive semidefinite. Away-step Frank-Wolfe maintains feasible occupancies;
its linear MDP oracle gives a global dual gap bounding the reference error.
Logged particle-validation returns are compared with the reference policy using
the same validation particles, seed, horizon and discount, as in the paper.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfc.algorithms.validation import monte_carlo_objective
from mfc.environments import TrafficPolicy
from mfc.visualization.io import load_env_and_policy, load_runs


def optimal_reference(env, tolerance=1e-7, max_iterations=100000):
    transition = env.transitions.cpu().numpy().astype(float)
    risk = env.risk_matrix.cpu().numpy().astype(float)
    risk = (risk+risk.T)/2
    minimum_eigenvalue = float(np.linalg.eigvalsh(risk).min())
    if minimum_eigenvalue < -1e-10:
        raise ValueError('Traffic risk is not convex; a global gap cannot be certified by this solver')
    initial = env.initial_distribution.cpu().numpy().astype(float)
    initial /= initial.sum()
    horizon = env.config.T_val
    discount = env.config.gamma**np.arange(horizon)
    linear_reward = (env.speed-env.config.comfort_cost*env.effort).cpu().numpy().astype(float)
    safety = env.config.safety_cost
    states = np.arange(env.n_states)

    def linear_oracle(gradient):
        continuation = np.zeros(env.n_states)
        actions = np.empty((horizon, env.n_states), dtype=int)
        for t in reversed(range(horizon)):
            q = gradient[t] + np.einsum('san,n->sa', transition, continuation)
            actions[t] = q.argmin(-1)
            continuation = q[states, actions[t]]
        atom = np.zeros((horizon, env.n_states, env.n_actions))
        mu = initial.copy()
        for t in range(horizon):
            atom[t, states, actions[t]] = mu
            mu = np.einsum('sa,san->n', atom[t], transition)
        return atom

    x = linear_oracle(np.broadcast_to(-discount[:, None, None]*linear_reward, (horizon, env.n_states, env.n_actions)))
    atoms, weights = [x.copy()], np.ones(1)
    for iteration in range(max_iterations):
        mu = x.sum(-1)
        gradient = discount[:, None, None]*(2*safety*(mu@risk)[..., None]-linear_reward)
        vertex = linear_oracle(gradient)
        forward = vertex-x
        dual_gap = -float(np.sum(gradient*forward))
        if dual_gap <= tolerance:
            break
        costs = np.array([np.sum(gradient*atom) for atom in atoms])
        away_index = int(costs.argmax())
        away = x-atoms[away_index]
        away_gap = -float(np.sum(gradient*away))
        use_away = len(atoms)>1 and away_gap>dual_gap
        if use_away:
            direction = away
            maximum_step = weights[away_index]/(1-weights[away_index])
        else:
            direction, maximum_step = forward, 1.
        delta_mu = direction.sum(-1)
        curvature = 2*safety*np.sum(discount*np.einsum('ti,ij,tj->t', delta_mu, risk, delta_mu))
        slope = float(np.sum(gradient*direction))
        step = min(maximum_step, max(0., -slope/curvature)) if curvature>1e-20 else maximum_step
        if use_away:
            weights *= 1+step
            weights[away_index] -= step
        else:
            weights *= 1-step
            matching = next((k for k, atom in enumerate(atoms) if np.array_equal(atom, vertex)), None)
            if matching is None:
                atoms.append(vertex)
                weights = np.append(weights, step)
            else:
                weights[matching] += step
        keep = weights>1e-14
        atoms = [atom for atom, retained in zip(atoms, keep) if retained]
        weights = weights[keep]
        weights /= weights.sum()
        x = np.einsum('k,ktsa->tsa', weights, np.array(atoms))
    else:
        raise RuntimeError(f'Reference did not reach tolerance; last dual gap={dual_gap:g}')
    mu = x.sum(-1)
    value = float(np.sum(discount*(np.sum(x*linear_reward, axis=(1, 2))-safety*np.einsum('ti,ij,tj->t', mu, risk, mu))))
    policy = np.divide(x, mu[..., None], out=np.full_like(x, 1/env.n_actions), where=mu[..., None]>0)
    # Independent feasibility and objective checks before using the reference.
    residual = float(np.max(np.abs(mu[0]-initial)))
    for t in range(horizon-1):
        residual = max(residual, float(np.max(np.abs(mu[t+1]-np.einsum('sa,san->n', x[t], transition)))))
    if residual>1e-9 or np.min(x)<-1e-12:
        raise RuntimeError(f'Invalid reference occupancies: {residual=}')
    return policy, dict(exact_value_lower_bound=value, exact_value_upper_bound=value+max(0., dual_gap),
                        dual_gap=max(0., dual_gap), iterations=iteration+1,
                        minimum_risk_eigenvalue=minimum_eigenvalue, flow_residual=residual,
                        method='away-step Frank-Wolfe over feasible state-action occupancies with an exact finite-horizon MDP linear oracle',
                        scope='global finite-horizon mean-field control optimum from the fixed validation law; unrestricted time-dependent state-action policies')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/traffic/transport_lambda_0.166156_eta_0.98_T_12_particle_seed_0'))
    parser.add_argument('--output', type=Path, default=Path('outputs/traffic_seed0'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = load_runs(args.run)
    if len(runs)!=1 or runs[0]['metadata']['env']!='traffic':
        raise ValueError('Select exactly one completed traffic run')
    run = runs[0]
    env, _ = load_env_and_policy(run, device='cpu')
    h, reference = optimal_reference(env)
    table = torch.tensor(h, dtype=env.dtype)

    def policy(t, mu):
        return table[min(int(t), len(table)-1)]

    seed = run['metadata']['seed']+len(run['history']['objective'])
    value = float(monte_carlo_objective(env, policy, lambda t: t, env.config.gamma,
                  env.config.validation_particles, env.config.T_val, seed))
    reference.update(matched_particle_value=value, validation_seed=seed,
                     validation_particles=env.config.validation_particles,
                     horizon=env.config.T_val, discount=env.config.gamma)
    history = run['history']
    transitions = np.asarray(history['validation_steps'])*run['metadata']['simulator_budget_estimate']
    returns = np.asarray(history['validation_objective'])
    torch.manual_seed(run['metadata']['seed'])
    initial = TrafficPolicy(env.config)
    initial_value = float(monte_carlo_objective(env, initial, lambda t: t, env.config.gamma,
                          env.config.validation_particles, env.config.T_val, seed))
    transitions = np.r_[0, transitions]
    returns = np.r_[initial_value, returns]
    gaps = np.abs(returns-value)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'optimal_reference.npz', policy=h)
    np.savetxt(args.output/'optimality_gap.csv', np.column_stack([transitions, returns, np.full_like(returns, value), gaps]),
               delimiter=',', header='simulated_training_transitions,validation_return,reference_validation_return,absolute_gap', comments='')
    reference.update(initial_gap=float(gaps[0]), final_gap=float(gaps[-1]),
                     minimum_logged_gap=float(gaps.min()),
                     minimum_gap_training_transitions=int(transitions[gaps.argmin()]),
                     gap_convention='absolute difference of logged particle-validation returns and the optimal reference evaluated with matching particles and seed; finite-particle estimates of J')
    (args.output/'optimal_reference.json').write_text(json.dumps(reference, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(6.3, 3.7), constrained_layout=True)
    ax.plot(transitions/1e6, gaps, color='#2a78d6', lw=1.25, label=f"Transport REINFORCE, seed {run['metadata']['seed']}")
    ax.set(xlabel='Simulated training transitions (millions)', ylabel=r'$|J(\widehat\theta)-J(\theta^\star)|$', xlim=(0, transitions[-1]/1e6), ylim=(0, None))
    scale=run['metadata']['perturbation']
    ax.set_title(f'Traffic: {env.n_states} states, T={env.config.T_val}, '+rf'$\lambda={scale:.6g}$', fontsize=11)
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    for suffix in ('pdf', 'png'):
        fig.savefig(args.output/f'optimality_gap.{suffix}', dpi=220)
    print(json.dumps(reference, indent=2))


if __name__ == '__main__':
    main()
