"""Read-only audit of individual versus social traffic gradients.

No new training is performed and the benchmark reward is not changed. A
counterfactual common-population reward is evaluated only to check which
feedback channel gives REINFORCE its useful signal. The two decompositions
have exactly the same aggregate objective on the unperturbed mean-field flow.
"""
import json
from pathlib import Path
import torch

from mfc.environments import TrafficPolicy
from mfc.visualization.io import load_runs,load_env_and_policy


def value(env,policy,initial,freeze_population=False,reward='local'):
    law=initial.unsqueeze(0) if initial.ndim==1 else initial
    total=law.new_zeros(len(law))
    c=env.speed-env.config.comfort_cost*env.effort
    for t in range(env.config.T):
        argument=law.detach() if freeze_population else law
        h=policy(t,argument)
        if reward=='local':
            risk=(argument@env.risk_matrix.T)[...,None]
        elif reward=='global':
            risk=(argument*(argument@env.risk_matrix.T)).sum(-1)[...,None,None]
        elif reward=='control':
            risk=torch.zeros_like(argument[...,:1,None])
        else:
            raise ValueError(reward)
        flow=law[...,None]*h
        total+=env.config.gamma**t*(flow*(c-env.config.safety_cost*risk)).sum((-1,-2))
        law=torch.einsum('bsa,san->bn',flow,env.transitions)
    return total.mean()


def grad(value,policy):
    return torch.cat([g.flatten() for g in torch.autograd.grad(value,tuple(policy.parameters()))])


def compare(env,policy,law):
    j=value(env,policy,law,reward='local')
    full=grad(j,policy)
    global_value=value(env,policy,law,reward='global')
    global_full=grad(global_value,policy)
    torch.testing.assert_close(j.detach(),global_value.detach(),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(full,global_full,atol=1e-10,rtol=1e-9)
    frozen=grad(value(env,policy,law,True,'local'),policy)
    global_frozen=grad(value(env,policy,law,True,'global'),policy)
    control_frozen=grad(value(env,policy,law,True,'control'),policy)
    torch.testing.assert_close(global_frozen,control_frozen,atol=1e-10,rtol=1e-9)
    cosine=lambda a,b: float(a@b/(a.norm()*b.norm()))
    factor=float(full@frozen/(frozen@frozen))
    return dict(J=float(j.detach()),full_gradient_norm=float(full.norm()),individual_gradient_norm=float(frozen.norm()),
                individual_social_cosine=cosine(full,frozen),
                missing_correction_relative_norm=float((full-frozen).norm()/full.norm()),
                best_scalar_multiple_of_individual=factor,
                residual_after_scalar_rescaling=float((full-factor*frozen).norm()/full.norm()),
                global_reward_individual_gradient_norm=float(global_frozen.norm()),
                global_reward_individual_social_cosine=cosine(full,global_frozen),
                local_global_social_value_difference=float((j-global_value).detach().abs()),
                local_global_social_gradient_difference=float((full-global_full).norm()),
                global_individual_congestion_gradient_norm=float((global_frozen-control_frozen).norm()))


def main():
    torch.set_num_threads(1)
    runs=load_runs('results/traffic_cells3_T5_mixed_seed0')
    selected={run['metadata']['algorithm']:run for run in runs}
    env,_=load_env_and_policy(selected['transport'],device='cpu')
    env.dtype=torch.float64
    for key in ('initial_distribution','risk_matrix','transitions','speed','effort'):
        setattr(env,key,getattr(env,key).double())
    env.initial_distribution/=env.initial_distribution.sum()
    torch.manual_seed(0)
    initial=TrafficPolicy(env.config).double()
    generator=torch.Generator().manual_seed(61021)
    laws=torch.stack([env.sample_initial_distribution(generator) for _ in range(128)])
    policies=[('initial',initial)]
    for name in ('reinforce','mfreinforce','transport'):
        _,policy=load_env_and_policy(selected[name],device='cpu')
        policies.append((name+'_final',policy.double()))
    out=Path('outputs/traffic_cells3_T5_mixed_seed0/externality_audit');out.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Exact population gradients at initial/final saved policies; no training or reward changes.',
                definitions=dict(full='Total MFC derivative',individual='Population arguments detached: large-population individual REINFORCE target',
                                 global_reward='Counterfactual r(s,mu,a)=c(a)-kappa*mu^T R mu; identical aggregate MFC objective'),
                rows=[])
    for name,policy in policies:
        for label,law in [('jam',env.initial_distribution),('training_mixture_128',laws)]:
            row=dict(policy=name,initial_law=label,**compare(env,policy,law))
            report['rows'].append(row)
            print(json.dumps(row),flush=True)
    (out/'gradients.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
