"""Compare the three seed-0 traffic methods at matched training budgets."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mfc.visualization.io import load_runs, load_env_and_policy


def plot_population_flows(runs, output, labels, report):
    """Exact mean-field rollout of each final checkpoint from the validation law."""
    torch.set_num_threads(1)
    laws = {}
    exact_values = {}
    for method, run in runs.items():
        env, policy = load_env_and_policy(run, device='cpu')
        law = env.initial_distribution.clone()
        trajectory = [law.numpy().copy()]
        total = 0.
        with torch.no_grad():
            for t in range(env.config.T_val):
                h = policy(torch.tensor(float(t)), law)
                states=torch.arange(env.n_states)[:,None]
                actions=torch.arange(env.n_actions)[None,:]
                total += float(env.config.gamma**t * (law[:,None]*h*env.reward(states,law,actions)).sum())
                law = torch.einsum('s,sa,san->n', law, h, env.transitions)
                trajectory.append(law.numpy().copy())
        laws[method] = np.asarray(trajectory)
        exact_values[method] = total
    reference = np.load(output/'optimal_reference.npz')['policy']
    law = env.initial_distribution.numpy().copy()
    trajectory = [law.copy()]
    for h in reference:
        law = np.einsum('s,sa,san->n', law, h, env.transitions.numpy())
        trajectory.append(law.copy())
    laws['optimal'] = np.asarray(trajectory)
    for law in laws.values():
        np.testing.assert_allclose(law.sum(-1), 1., atol=1e-10)
        assert law.min() >= -1e-12
    np.savez(output/'final_population_flows.npz', **laws)
    certificate=json.loads((output/'optimal_reference.json').read_text())
    assessment={method:dict(exact_return=total,
                exact_gap=abs(total-certificate['exact_value_lower_bound']),
                terminal_total_variation_from_reference=float(.5*np.abs(laws[method][-1]-laws['optimal'][-1]).sum()))
                for method,total in exact_values.items()}
    (output/'final_population_assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    rows = [(method,t,lane,cell,float(flow[t,lane*env.config.cells+cell]))
            for method,flow in laws.items() for t in range(len(flow))
            for lane in range(3) for cell in range(env.config.cells)]
    with (output/'final_population_flows.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['method','time','lane','cell','mass']);writer.writerows(rows)
    fig,axes=plt.subplots(1,4,figsize=(12,3.8),layout='constrained',sharey=True)
    for ax,method in zip(axes,(*runs,'optimal')):
        im=ax.imshow(laws[method].T,origin='lower',aspect='auto',vmin=0,
                     vmax=max(float(flow.max()) for flow in laws.values()),cmap='viridis')
        ax.set_title(labels.get(method,'Optimal reference'),fontsize=10)
        ax.set_xlabel('Time');ax.set_xticks(range(env.config.T_val+1))
    axes[0].set_yticks(range(env.n_states),
        [f'Lane {lane+1}, cell {cell}' for lane in range(3) for cell in range(env.config.cells)])
    fig.colorbar(im,ax=axes,label='Population mass',shrink=.85)
    fig.suptitle('Final policies: exact population flow from the validation jam (seed 0)',fontsize=12)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'final_population_flows.{suffix}',dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path('results/traffic_cells3_T5_mixed_seed0'))
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_cells3_T5_mixed_seed0'))
    parser.add_argument('--mf-results-root',type=Path,default=None,
                        help='Optional separate experiment root for the selected MF scale')
    args=parser.parse_args()
    runs=load_runs(args.results_root)
    methods=('reinforce','mfreinforce','transport')
    selected={}
    for method in methods:
        candidates=load_runs(args.mf_results_root) if method=='mfreinforce' and args.mf_results_root else runs
        matches=[run for run in candidates if run['metadata']['algorithm']==method and run['metadata']['seed']==0]
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one completed {method} seed-0 run; found {len(matches)}')
        selected[method]=matches[0]
    reference=json.loads((args.output/'optimal_reference.json').read_text())
    transport_csv=np.genfromtxt(args.output/'optimality_gap.csv',delimiter=',',names=True)
    star=reference['matched_particle_value']
    initial_value=float(transport_csv['validation_return'][0])
    physical_keys=('congestion_reward','precision','cells','p_fail','p_slow','p_fast','safety_cost','comfort_cost','forward_scale','lateral_scale','gamma',
                   'hidden_width','T','T_val','training_initial_law','training_jam_probability','validation_interval','validation_particles')
    expected=selected['transport']['metadata']
    curves={};report={};common_x=None
    for method,run in selected.items():
        meta=run['metadata'];history=run['history'];config=meta['algorithm_config']
        assert all(meta['env_config'].get(key,{'congestion_reward':'local','precision':'float32'}.get(key))==expected['env_config'].get(key,{'congestion_reward':'local','precision':'float32'}.get(key)) for key in physical_keys), 'Benchmark mismatch'
        assert not config['baseline'] and meta['env_config']['lr']==expected['env_config']['lr']
        assert len(history['objective'])==10000
        assert meta['seed']+len(history['objective'])==reference['validation_seed']
        assert meta['env_config']['validation_particles']==reference['validation_particles']
        budget=meta['simulator_budget_estimate']
        assert budget==6360 and budget*len(history['objective'])==63600000
        if method=='mfreinforce':
            assert meta['perturbation']==config['perturbation_eta'] and meta['perturbation']>0
        x=np.r_[0,np.asarray(history['validation_steps'])*budget]
        values=np.r_[initial_value,np.asarray(history['validation_objective'])]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite validation return in {method}')
        gap=np.abs(values-star)
        if common_x is None:
            common_x=x
        else:
            np.testing.assert_array_equal(common_x,x)
        curves[method]=(values,gap)
        gradients=np.asarray(history.get('gradient_norm',[]))
        report[method]=dict(run=str(run['path']),algorithm_config=config,training_transitions=int(x[-1]),
                           final_return=float(values[-1]),final_gap=float(gap[-1]),
                           minimum_logged_gap=float(gap.min()),minimum_gap_transitions=int(x[gap.argmin()]),
                           final_100_mean_gap=float(gap[-100:].mean()),
                           elapsed_seconds=run['summary'].get('elapsed_seconds'),
                           finite_recorded_gradient_norms=bool(np.isfinite(gradients).all()) if len(gradients) else None,
                           nonfinite_gradient_norm_count=int((~np.isfinite(gradients)).sum()),
                           nonfinite_gradient_norm_steps=(np.flatnonzero(~np.isfinite(gradients))+1).tolist())
    args.output.mkdir(parents=True,exist_ok=True)
    comparison=dict(reference=reference,benchmark={key:expected['env_config'].get(key,{'congestion_reward':'local','precision':'float32'}.get(key)) for key in physical_keys},
                    seed=0,training_transitions_per_method=63600000,baseline=False,methods=report,
                    scope='One training seed per algorithm. Common initial policy, benchmark, validation seed/particles and total training transitions.',
                    numerical_note='MF-REINFORCE has infinite logged gradient norms. Norm overflow does not by itself identify nonfinite gradient components; gradient vectors and Adam moments were not saved. Its numerical behavior needs separate investigation before attributing the plateau solely to the mathematical method.' if report['mfreinforce']['nonfinite_gradient_norm_count'] else None)
    (args.output/'three_algorithms_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    matrix=np.column_stack([common_x]+[curves[method][1] for method in methods]+[curves[method][0] for method in methods])
    np.savetxt(args.output/'three_algorithms_gaps.csv',matrix,delimiter=',',comments='',
               header='simulated_training_transitions,'+','.join(method+'_gap' for method in methods)+','+','.join(method+'_return' for method in methods))
    epsilon=selected['mfreinforce']['metadata']['perturbation']
    labels={'reinforce':'REINFORCE','mfreinforce':rf'MF-REINFORCE ($\varepsilon={epsilon:g}$)','transport':'Transport REINFORCE'}
    colors={'reinforce':'#eb6834','mfreinforce':'#eda100','transport':'#2a78d6'}
    fig,ax=plt.subplots(figsize=(7.1,4.1),constrained_layout=True)
    for method in methods:
        ax.plot(common_x/1e6,curves[method][1],color=colors[method],lw=1.2,label=labels[method])
    ax.set(xlabel='Simulated training transitions (millions)',ylabel=r'$|J(\widehat\theta)-J(\theta^\star)|$',
           title=f"Traffic: {expected['env_config'].get('congestion_reward','local')} congestion, mixed initial laws, seed 0",xlim=(0,63.6),ylim=(0,None))
    count=report['mfreinforce']['nonfinite_gradient_norm_count']
    if count:
        ax.set_title(f'Traffic: mixed initial laws, seed 0\nMF-REINFORCE recorded {count} infinite gradient norms',fontsize=10)
    ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.22)
    for suffix in ('pdf','png'):
        fig.savefig(args.output/f'three_algorithms_gap.{suffix}',dpi=220)
    positive=np.concatenate([gap[gap>0] for _,gap in curves.values()])
    ax.set_yscale('log');ax.set_ylim(max(1e-6,float(positive.min())*.65),float(positive.max())*1.25)
    for suffix in ('pdf','png'):
        fig.savefig(args.output/f'three_algorithms_gap_log.{suffix}',dpi=220)
    plot_population_flows(selected, args.output, labels, report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
