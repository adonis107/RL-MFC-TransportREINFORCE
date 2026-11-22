"""Compare paired MF-REINFORCE epsilon runs without selecting checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from calibrate_traffic import exact_objective
from mfc.visualization.io import load_env_and_policy, load_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epsilon-one-root', type=Path, default=Path('results/traffic_shared_seed0'))
    parser.add_argument('--epsilon-two-root', type=Path, default=Path('results/traffic_shared_eps2_seed0'))
    parser.add_argument('--reference-output', type=Path, default=Path('outputs/traffic_shared_seed0'))
    parser.add_argument('--output', type=Path, default=Path('outputs/traffic_shared_epsilon_seed0'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    selected = {}
    for epsilon, root in ((1, args.epsilon_one_root), (2, args.epsilon_two_root)):
        matches = [r for r in load_runs(root) if r['metadata']['algorithm'] == 'mfreinforce'
                   and r['metadata']['seed'] == 0 and r['metadata']['perturbation'] == epsilon]
        if len(matches) != 1:
            raise ValueError(f'Expected one completed epsilon={epsilon}, seed=0 run under {root}')
        selected[epsilon] = matches[0]
    first, second = [selected[e]['metadata'] for e in (1, 2)]
    assert first['env_config'] == second['env_config'], 'Environment mismatch'
    for key in first['algorithm_config']:
        if key != 'perturbation_eta':
            assert first['algorithm_config'][key] == second['algorithm_config'][key], key
    reference = json.loads((args.reference_output/'optimal_reference.json').read_text())
    initial = np.genfromtxt(args.reference_output/'optimality_gap.csv', delimiter=',', names=True)[0]['validation_return']
    report = {}; curves = {}; common_x = None
    for epsilon, run in selected.items():
        meta, history = run['metadata'], run['history']
        updates = len(history['objective'])
        assert updates == 10000 and meta['simulator_budget_estimate'] == 6360
        assert meta['seed'] + updates == reference['validation_seed']
        assert meta['env_config']['validation_particles'] == reference['validation_particles']
        x = np.r_[0, np.asarray(history['validation_steps'])*meta['simulator_budget_estimate']]
        if common_x is None:
            common_x = x
        else:
            np.testing.assert_array_equal(common_x, x)
        values = np.r_[initial, history['validation_objective']]
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite validation objective at epsilon={epsilon}')
        gaps = np.abs(values-reference['matched_particle_value'])
        gradients = np.asarray(history['gradient_norm'])
        env, policy = load_env_and_policy(run, device='cpu')
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        with torch.no_grad():
            exact = float(exact_objective(env, policy))
        report[epsilon] = dict(run=str(run['path']), epsilon=epsilon,
            training_transitions=int(x[-1]), final_return=float(values[-1]),
            final_gap=float(gaps[-1]), final_100_mean_gap=float(gaps[-100:].mean()),
            minimum_logged_gap=float(gaps.min()), minimum_gap_transitions=int(x[gaps.argmin()]),
            exact_final_return=exact, exact_final_gap=abs(exact-reference['exact_value_lower_bound']),
            nonfinite_gradient_norm_count=int((~np.isfinite(gradients)).sum()),
            maximum_gradient_norm=float(gradients.max()))
        curves[epsilon] = (values, gaps)
    result = dict(seed=0, M=200, n=10, B=1012, updates=10000,
        transitions_per_update=6360, training_transitions_per_run=63600000,
        reference=reference, methods=report,
        better_final_gap_epsilon=min(report, key=lambda e: report[e]['final_gap']),
        better_final_100_mean_gap_epsilon=min(report, key=lambda e: report[e]['final_100_mean_gap']),
        scope='Paired single-seed test. Only epsilon changes; no split recalibration or checkpoint selection. Does not establish a multi-seed best scale.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savetxt(args.output/'optimality_gaps.csv', np.column_stack([common_x, curves[1][1], curves[2][1], curves[1][0], curves[2][0]]),
        delimiter=',', comments='', header='training_transitions,epsilon_1_gap,epsilon_2_gap,epsilon_1_return,epsilon_2_return')
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout='constrained')
    for epsilon, color in ((1, '#eda100'), (2, '#8f55bd')):
        ax.plot(common_x/1e6, curves[epsilon][1], color=color, lw=1.2,
                label=rf'MF-REINFORCE ($\varepsilon={epsilon}$)')
    ax.set(xlabel='Simulated training transitions (millions)', ylabel=r'$|\widehat J(\widehat\theta)-\widehat J(\theta^\star)|$',
        title='Traffic: shared congestion, paired seed 0', xlim=(0, 63.6), ylim=(0, None))
    ax.legend(frameon=False); ax.grid(alpha=.22); ax.spines[['top', 'right']].set_visible(False)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output/f'optimality_gaps.{suffix}', dpi=220)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
