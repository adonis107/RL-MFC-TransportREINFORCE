"""Read-only infinite-batch auxiliary-recursion bias diagnostic.

Integrates states/actions exactly and uses antithetic Monte Carlo for Q.
Propagates the mean sensitivity recursion by automatic differentiation with
unperturbed centre values and auxiliary-state tangents. This is the n->infinity
limit of the paper's recursion, not a replacement training estimator.
"""
import json
from pathlib import Path
import torch
from diagnose_traffic import objective, gradient
from mfc.visualization.io import load_runs, load_env_and_policy
from mfc.environments import TrafficPolicy


def draw(env, count, generator, batch=1, sigma=.75):
    z=torch.randn(count//2,batch,env.n_states-1,device=env.device,generator=generator)*sigma
    z=torch.cat([z,-z])
    return torch.cat([z,torch.zeros_like(z[...,:1])],-1).softmax(-1)


def limiting_gradient(env,policy,eta,lambda_,count=4096,initial_distribution=None,randomizer_sigma=.75):
    initial=env.initial_distribution if initial_distribution is None else initial_distribution
    initial=initial.unsqueeze(0) if initial.ndim==1 else initial
    batch=len(initial)
    with torch.no_grad():
        base=[initial]
        for t in range(env.config.T):
            base.append(torch.einsum('bs,bsa,san->bn',base[-1],policy(t,base[-1]),env.transitions))
    generator=torch.Generator(device=env.device).manual_seed(23872)
    auxiliary=initial
    centres=[initial]
    for t in range(env.config.T):
        if t==0:
            h=policy(t,initial)
        else:
            q=draw(env,count,generator,batch,randomizer_sigma)
            m=(1-eta)*centres[t]+eta*q
            h=policy(t,m).mean(0)
        auxiliary=torch.einsum('bs,bsa,san->bn',auxiliary,h,env.transitions)
        centres.append(base[t+1]+auxiliary-auxiliary.detach())
    generator=torch.Generator(device=env.device).manual_seed(9421)
    marginal=initial
    value=initial.new_zeros(batch)
    c=env.speed-env.config.comfort_cost*env.effort
    for t in range(env.config.T):
        if t==0:
            m=initial
            h=policy(t,m)
            h_reward=h*(c-env.config.safety_cost*(m@env.risk_matrix.T)[...,None])
        else:
            q=draw(env,count,generator,batch,randomizer_sigma)
            m=(1-lambda_)*centres[t]+lambda_*q
            kernels=policy(t,m)
            h=kernels.mean(0)
            h_reward=(kernels*(c-env.config.safety_cost*(m@env.risk_matrix.T)[...,None])).mean(0)
        value=value+env.config.gamma**t*(marginal[...,None]*h_reward).sum((-1,-2))
        marginal=torch.einsum('bs,bsa,san->bn',marginal,h,env.transitions)
    return gradient(value.mean(),policy).detach()


def main():
    torch.set_num_threads(1)
    run=load_runs('results/traffic')[0]
    env,final=load_env_and_policy(run,device='cuda')
    torch.manual_seed(0)
    initial=TrafficPolicy(env.config)
    scale=run['metadata']['perturbation']
    rows=[]
    for name,policy in [('initial',initial),('final',final)]:
        true=gradient(objective(env,policy,env.initial_distribution,scale,4096).mean(),policy).detach()
        for eta in (0.,.25,.85,.95,.98):
            estimate=limiting_gradient(env,policy,eta,scale)
            if eta==0:
                torch.testing.assert_close(estimate,true,atol=2e-5,rtol=2e-4)
            row=dict(policy=name,eta=eta,lambda_=scale,
                     relative_auxiliary_limit_bias=float((estimate-true).norm()/true.norm()),
                     cosine=float(estimate@true/(estimate.norm()*true.norm())),
                     norm_ratio=float(estimate.norm()/true.norm()),
                     randomizer_integration_samples=4096)
            rows.append(row)
            print(json.dumps(row),flush=True)
    path=Path('outputs/traffic_seed0/diagnosis/auxiliary_limit.json')
    path.write_text(json.dumps(dict(scope='mean/infinite-batch auxiliary recursion at exact population flows; Monte Carlo integration of randomizers; no training',rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
