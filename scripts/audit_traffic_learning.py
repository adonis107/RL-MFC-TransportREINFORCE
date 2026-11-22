"""Audit learning versus validation targets of a saved traffic policy.

No full training run is performed. Initial/final policies are evaluated against
exact state/action population recursions; smoothing randomizers use antithetic
Monte Carlo. Temporary exact-gradient optimizer probes are restored immediately.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from mfc.environments import TrafficPolicy
from mfc.algorithms import DiscreteTransport,DiscreteTransportConfig
from mfc.visualization.io import load_runs,load_env_and_policy
from diagnose_traffic import objective,gradient
from traffic_optimality_gap import optimal_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/traffic_cells3_T5_seed0/traffic/transport_lambda_0.415661_eta_0.98_T_5_particle_seed_0'))
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_cells3_T5_seed0/learning_audit'))
    parser.add_argument('--replications',type=int,default=240)
    parser.add_argument('--auxiliary-limit-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    run=load_runs(args.run)[0]
    env,final=load_env_and_policy(run,device='cuda')
    torch.manual_seed(run['metadata']['seed'])
    initial=TrafficPolicy(env.config)
    scale=run['metadata']['perturbation']; sigma=run['metadata']['algorithm_config']['simplex_sigma']
    gen=torch.Generator(device=env.device).manual_seed(410927)
    weights=-torch.rand(1024,env.n_states,device=env.device,generator=gen).clamp_min(1e-12).log()
    laws=weights/weights.sum(-1,keepdim=True)
    uniform=torch.full_like(env.initial_distribution,1/env.n_states)
    if args.auxiliary_limit_only:
        from audit_traffic_auxiliary import limiting_gradient
        rows=[]
        for name,policy in [('initial',initial),('final',final)]:
            reference=gradient(objective(env,policy,laws[:128],scale,q_samples=256,randomizer_sigma=sigma).mean(),policy).detach()
            for eta in (0.,.25,.85,.98):
                actual=limiting_gradient(env,policy,eta,scale,count=256,initial_distribution=laws[:128],randomizer_sigma=sigma)
                if eta==0.:
                    torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-4)
                row=dict(policy=name,eta=eta,relative_bias=float((actual-reference).norm()/reference.norm()),cosine=float(actual@reference/(actual.norm()*reference.norm())),norm_ratio=float(actual.norm()/reference.norm()))
                rows.append(row); print(json.dumps(row),flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'training_auxiliary_limit.json').write_text(json.dumps(dict(random_laws=128,randomizer_samples=256,rows=rows),indent=2)+'\n')
        return
    report=dict(run=str(args.run),lambda_=scale,randomizer_sigma=sigma,random_initial_laws=1024,
                parameter_change_norm=float(torch.cat([(a-b).flatten() for a,b in zip(final.parameters(),initial.parameters())]).norm().detach()),policies={})
    for name,policy in [('initial',initial),('final',final)]:
        row={}
        with torch.no_grad():
            for target,law in [('jam',env.initial_distribution),('uniform',uniform),('random_laws',laws)]:
                row[target]={}
                for label,lam in [('J',0.),('J_lambda',scale)]:
                    values=objective(env,policy,law,lam,q_samples=64,randomizer_sigma=sigma)
                    row[target][label]=float(values.mean())
                    if values.numel()>1:
                        row[target][label+'_law_standard_error']=float(values.std()/values.numel()**.5)
        train_target=lambda:objective(env,policy,laws[:128],scale,q_samples=128,randomizer_sigma=sigma).mean()
        jam_target=lambda:objective(env,policy,env.initial_distribution).mean()
        train_gradient=gradient(train_target(),policy).detach()
        jam_gradient=gradient(jam_target(),policy).detach()
        row['exact_training_gradient_norm']=float(train_gradient.norm())
        row['exact_jam_gradient_norm']=float(jam_gradient.norm())
        row['training_jam_gradient_cosine']=float(train_gradient@jam_gradient/(train_gradient.norm()*jam_gradient.norm()))
        estimator=DiscreteTransport(env,policy=policy,config=DiscreteTransportConfig(**run['metadata']['algorithm_config']))
        samples=[];values=[]
        for i in range(args.replications):
            sample,value=estimator.estimate_gradient(200000+i*7919)
            samples.append(sample.detach());values.append(float(value))
        samples=torch.stack(samples);mean=samples.mean(0)
        row['sampled_training_return_mean']=sum(values)/len(values)
        row['sampled_gradient_mean_norm']=float(mean.norm())
        row['sampled_mean_cosine_to_training_gradient']=float(mean@train_gradient/(mean.norm()*train_gradient.norm()))
        row['sampled_mean_cosine_to_jam_gradient']=float(mean@jam_gradient/(mean.norm()*jam_gradient.norm()))
        for label,g in [('training',train_gradient),('jam',jam_gradient)]:
            projection=samples@(g/g.norm())
            row[label+'_gradient_projection_mean']=float(projection.mean())
            row[label+'_gradient_projection_standard_error']=float(projection.std()/args.replications**.5)
        # The exact current training gradient passes through the same gradient
        # assignment and Adam sign convention used by train(). Restore weights.
        saved=copy.deepcopy(policy.state_dict())
        before=float(train_target().detach());jam_before=float(jam_target().detach())
        optimizer=estimator.optimizer();optimizer.zero_grad();estimator.set_flat_gradient(-train_gradient);optimizer.step()
        after=float(train_target().detach());jam_after=float(jam_target().detach())
        row['temporary_exact_gradient_adam_probe']=dict(before=before,after=after,change=after-before,
                                                       jam_before=jam_before,jam_after=jam_after,jam_change=jam_after-jam_before,
                                                       note='fresh Adam state, lr from run, one temporary diagnostic step; weights restored')
        policy.load_state_dict(saved)
        report['policies'][name]=row
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(policy=name,**row),indent=2),flush=True)
    cpu_env,_=load_env_and_policy(run,device='cpu')
    cpu_env.initial_distribution.fill_(1/cpu_env.n_states)
    _,bound=optimal_reference(cpu_env)
    report['uniform_initial_law_optimal_reference']=bound
    report['limits']=['Training gradients integrate states/actions exactly but approximate the random initial-law expectation and smoothing randomizers by Monte Carlo.',
                      'The optimizer probe uses fresh Adam state; intermediate training policies and historical Adam moments were not saved.',
                      'Only initial/final policies are evaluated; this does not reconstruct the full true-objective learning curve.']
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(bound,indent=2))


if __name__=='__main__':
    main()
