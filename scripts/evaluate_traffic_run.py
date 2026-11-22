"""Assess one traffic checkpoint against its initialization and simple controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfc.environments import TrafficPolicy
from mfc.visualization.io import load_env_and_policy, load_runs
from mfc.algorithms.validation import monte_carlo_objective


@torch.no_grad()
def exact_rollout(env, policy, law=None):
    law = env.initial_distribution if law is None else law
    value = law.new_zeros(law.shape[:-1])
    risks, speeds = [], []
    for t in range(env.config.T):
        if isinstance(policy, int):
            h = torch.nn.functional.one_hot(torch.full((env.n_states,), policy), 9).to(env.dtype)
        elif isinstance(policy, str) and policy == 'random':
            h = torch.full((env.n_states, 9), 1/9)
        else:
            h = policy(torch.tensor(float(t)), law)
        flow = law[..., :, None] * h
        risk = (law * (law @ env.risk_matrix.T)).sum(-1)
        reward = (flow * (env.speed-env.config.comfort_cost*env.effort)).sum((-1, -2)) - env.config.safety_cost*risk
        value += env.config.gamma**t * reward
        risks.append(risk)
        speeds.append((flow * env.speed).sum((-1, -2)))
        law = torch.einsum('...sa,san->...n', flow, env.transitions)
    return value, torch.stack(risks), torch.stack(speeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/traffic_seed0'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = load_runs(args.run)
    if len(runs) != 1:
        raise ValueError('Expected exactly one completed run directory')
    run = runs[0]
    env, learned = load_env_and_policy(run, device='cpu')
    torch.manual_seed(run['metadata']['seed'])
    initial = TrafficPolicy(env.config)
    policies = {'initial': initial, 'learned': learned, 'uniform_random': 'random',
                'one_cell_no_lane_change': 4, 'max_speed_no_lane_change': 7}
    generator = torch.Generator().manual_seed(91023)
    weights = -torch.rand(128, env.n_states, generator=generator).clamp_min(1e-12).log()
    laws = weights/weights.sum(-1, keepdim=True)
    scores = {}
    traces = {}
    for name, policy in policies.items():
        value, risk, speed = exact_rollout(env, policy)
        random_values, _, _ = exact_rollout(env, policy, laws)
        scores[name] = {'jam_exact_objective': float(value),
                        'random_128_mean': float(random_values.mean()),
                        'random_128_standard_error': float(random_values.std()/128**.5),
                        'mean_speed': float(speed.mean()), 'mean_risk': float(risk.mean())}
        traces[name] = (risk.numpy(), speed.numpy())
    particle_values = [float(monte_carlo_objective(env, learned, lambda t: t, env.config.gamma,
                       env.config.validation_particles, env.config.T, 80000+seed)) for seed in range(20)]
    history = json.loads((args.run/'history.json').read_text())
    gradient = np.asarray(history['gradient_norm'])
    validation = np.asarray(history['validation_objective'])
    selection_path = Path('results/traffic_calibration/selection.json')
    report = dict(run=str(args.run), selection=json.loads(selection_path.read_text()) if selection_path.exists() else None,
                  scores=scores,
                  jam_improvement=scores['learned']['jam_exact_objective']-scores['initial']['jam_exact_objective'],
                  random_law_improvement=scores['learned']['random_128_mean']-scores['initial']['random_128_mean'],
                  random_law_gap_vs_max_speed=scores['learned']['random_128_mean']-scores['max_speed_no_lane_change']['random_128_mean'],
                  validation_particle_mean=float(np.mean(particle_values)),
                  validation_particle_standard_error=float(np.std(particle_values, ddof=1)/20**.5),
                  first_logged_validation=float(validation[0]), final_validation=float(validation[-1]),
                  best_logged_validation=float(validation.max()),
                  best_logged_validation_step=int(history['validation_steps'][int(validation.argmax())]),
                  final_100_validation_mean=float(validation[-100:].mean()),
                  training_transitions=run['metadata']['simulator_budget_estimate']*len(history['objective']),
                  finite_training_objectives=bool(np.isfinite(history['objective']).all()),
                  finite_validation_objectives=bool(np.isfinite(validation).all()),
                  finite_gradients=bool(np.isfinite(gradient).all()),
                  max_gradient_norm=float(gradient.max()), final_gradient_norm=float(gradient[-1]),
                  limitation='One training seed; repeated particle evaluations measure simulation noise, not training-seed uncertainty.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'assessment.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    axes[0, 0].plot(np.asarray(history['validation_steps'])*run['metadata']['simulator_budget_estimate']/1e6, validation)
    axes[0, 0].axhline(scores['initial']['jam_exact_objective'], color='gray', ls='--', label='initial policy')
    axes[0, 0].set(xlabel='Training transitions (millions)', ylabel='Validation return')
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(np.arange(len(gradient))+1, gradient, alpha=.7)
    axes[0, 1].set(xlabel='Update', ylabel='Gradient norm', yscale='log')
    for name in ('initial', 'learned'):
        risk, speed = traces[name]
        axes[1, 0].plot(speed, label=name)
        axes[1, 1].plot(risk, label=name)
    axes[1, 0].set(xlabel='Time step', ylabel='Mean commanded speed')
    axes[1, 1].set(xlabel='Time step', ylabel='Population proximity risk')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1, 0].legend(frameon=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'learning.{suffix}', dpi=180)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
