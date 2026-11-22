"""Reconstruct traffic initial-law coverage and compare saved-policy endpoints."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mfc.environments import TrafficPolicy
from mfc.visualization.io import load_runs,load_env_and_policy


def main():
    torch.set_num_threads(1)
    root=Path('outputs/traffic_cells3_T5_seed0')
    output=root/'learning_audit'
    audit=json.loads((output/'audit.json').read_text())
    run=load_runs(audit['run'])[0]
    env,final=load_env_and_policy(run,device='cuda')
    torch.manual_seed(run['metadata']['seed'])
    initial=TrafficPolicy(env.config)
    config=run['metadata']['algorithm_config']
    generator=torch.Generator(device=env.device)
    laws=[]
    for update in range(len(run['history']['objective'])):
        generator.manual_seed(run['metadata']['seed']+update*(config['n_particles']+1)+30000)
        laws.append(env.sample_initial_distribution(generator))
    laws=torch.stack(laws)
    mass=laws[:,::env.config.cells].sum(-1)
    distances=(laws-env.initial_distribution).abs().sum(-1)/2
    a,b=3,env.n_states-3
    threshold=.9
    probability=sum(math.comb(a+b-1,k)*threshold**k*(1-threshold)**(a+b-1-k) for k in range(a))
    coverage=dict(reconstructed_training_laws=len(laws),mean_mass_at_validation_jam_positions=float(mass.mean()),
                  largest_mass_at_validation_jam_positions=float(mass.max()),
                  count_mass_above_09=int((mass>.9).sum()),minimum_total_variation_distance_to_validation_jam=float(distances.min()),
                  theoretical_probability_mass_above_09=probability,
                  expected_count_mass_above_09=len(laws)*probability,
                  reconstruction='Same device, local-generator seeds, seed stride B+1 and sample_initial_distribution implementation as the recorded run.')
    optimal=torch.tensor(np.load(root/'optimal_reference.npz')['policy'],dtype=env.dtype,device=env.device)
    profiles={}
    with torch.no_grad():
        for name,policy in [('initial',initial),('final',final),('optimal_jam',None)]:
            mu=env.initial_distribution.clone();risk=[];speed=[];moves=[];entropy=[]
            for t in range(env.config.T):
                h=optimal[t] if policy is None else policy(t,mu)
                flow=mu[:,None]*h
                risk.append(float(mu@env.risk_matrix@mu))
                speed.append(float((flow*env.speed).sum()))
                moves.append(flow.reshape(env.n_states,3,3).sum((0,2)).cpu().tolist())
                entropy.append(float(-(mu[:,None]*h*h.clamp_min(1e-30).log()).sum()))
                mu=torch.einsum('sa,san->n',flow,env.transitions)
            profiles[name]=dict(risk=risk,speed=speed,longitudinal_action_probabilities=moves,action_entropy=entropy)
    audit.update(training_initial_law_coverage=coverage,jam_profiles=profiles)
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    jam_ref=json.loads((root/'optimal_reference.json').read_text())['exact_value_lower_bound']
    uniform_ref=audit['uniform_initial_law_optimal_reference']['exact_value_lower_bound']
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    x=np.arange(2)
    for offset,name,color in [(-.18,'initial','#8e9aaf'),(.18,'final','#2a78d6')]:
        values=[uniform_ref-audit['policies'][name]['uniform']['J'],jam_ref-audit['policies'][name]['jam']['J']]
        bars=axes[0].bar(x+offset,values,.34,label=name,color=color)
        axes[0].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    axes[0].set(xticks=x,xticklabels=['Uniform initial law','Validation jam'],ylabel='Exact population optimality gap',title='Saved-policy endpoints')
    axes[0].legend(frameon=False)
    for name,color in [('initial','#8e9aaf'),('final','#2a78d6'),('optimal_jam','#222222')]:
        axes[1].plot(profiles[name]['risk'],label=name,color=color)
    axes[1].set(xlabel='Time step',ylabel='Population proximity risk',title='Jam population flow')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'target_comparison.{suffix}',dpi=200)
    print(json.dumps(dict(coverage=coverage,profiles=profiles),indent=2))


if __name__=='__main__':
    main()
