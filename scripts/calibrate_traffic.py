"""Select traffic allocations by raw gradient MSE over three reference policies.

Screen the split at eta=.95, sigma=.5, then eta/sigma at lambda*=B^(-1/4).
Calibration transitions are separate from the training budget.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch

from mfc.algorithms import DiscreteTransport, DiscreteTransportConfig
from mfc.environments import Traffic, TrafficConfig
from mfc.algorithms import MFReinforce, MFReinforceConfig


def exact_objective(env, policy, law=None):
    law = env.initial_distribution if law is None else law
    total = law.new_zeros(())
    states = torch.arange(env.n_states, device=env.device)[:, None]
    actions = torch.arange(env.n_actions, device=env.device)[None, :]
    for t in range(env.config.T):
        h = policy(torch.tensor(float(t), device=env.device), law)
        total = total + env.config.gamma**t * (law[:, None] * h * env.reward(states, law, actions)).sum()
        law = torch.einsum('s,sa,san->n', law, h, env.transitions)
    return total


def exact_gradient(env, policy):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(exact_objective(env, policy), tuple(policy.parameters()))])


def references(env, warm_steps=100, seeds=(0, 1, 2)):
    from correction_alignment import reference_policy

    if warm_steps < 1 or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Use three distinct reference seeds and a positive warm-start length')
    result = []
    for seed in seeds:
        algorithm = reference_policy(env, env.config.T, warm_steps, seed)
        result.append((seed, algorithm.policy, exact_gradient(env, algorithm.policy).detach()))
    return result


def score(env, refs, n, b, eta, sigma, replications, stage, population=200, baseline=False, lambda_multiplier=1.):
    rows = []
    started = time.monotonic()
    for reference_seed, policy, exact in refs:
        estimator = DiscreteTransport(env, policy=policy, config=DiscreteTransportConfig(
            n_particles=b, n_flow_particles=population, n_logit_gradient=n,
            horizon=env.config.T, lambda_=lambda_multiplier*b**(-.25), eta=eta, simplex_sigma=sigma,
            flow='particle', auxiliary_batches='shared', baseline=baseline, seed=0))
        estimator.sample_initial_distribution = lambda generator: env.initial_distribution
        squared_error = 0.
        mean = torch.zeros_like(exact)
        mean_square = 0.
        for replication in range(replications):
            estimate = estimator.estimate_gradient(100000 + replication * 7919)[0].detach()
            if not torch.isfinite(estimate).all():
                raise FloatingPointError(f'Nonfinite calibration gradient: {n=}, {b=}, {eta=}, {sigma=}')
            squared_error += float((estimate-exact).square().sum())
            mean += estimate / replications
            mean_square += float(estimate.square().sum()) / replications
        norm_squared = float(exact.square().sum())
        rows.append(dict(stage=stage, reference_seed=reference_seed, replications=replications,
                         M=population, n=n, B=b, eta=eta, simplex_sigma=sigma, lambda_anchor=b**(-.25),
                         lambda_multiplier=lambda_multiplier, lambda_actual=lambda_multiplier*b**(-.25), baseline=baseline,
                         gradient_norm=norm_squared**.5, mse=squared_error/replications,
                         relative_mse=squared_error/replications/norm_squared,
                         relative_bias=float((mean-exact).norm()/exact.norm()),
                         relative_variance=max(0., mean_square-float(mean.square().sum()))/norm_squared,
                         mean_cosine=float(mean@exact/(mean.norm()*exact.norm()))))
    print(json.dumps(dict(stage=stage, n=n, B=b, eta=eta, sigma=sigma,
                          mse=sum(row['mse'] for row in rows)/len(rows),
                          relative_mse=sum(row['relative_mse'] for row in rows)/len(rows),
                          seconds=time.monotonic()-started)), flush=True)
    return rows


def score_mf(env, refs, auxiliary, particles, population, replications, baseline):
    rows=[]
    started=time.monotonic()
    for reference_seed,policy,exact in refs:
        estimator=MFReinforce(env,policy=policy,config=MFReinforceConfig(
            n_particles=particles,n_flow_particles=population,n_logit_gradient=auxiliary,
            horizon=env.config.T,perturbation_eta=1.,flow='particle',baseline=baseline))
        estimator.sample_initial_distribution=lambda generator: env.initial_distribution
        errors=[];nonfinite=0
        for replication in range(replications):
            estimate=estimator.estimate_gradient(100000+replication*7919)[0].detach().double()
            if not torch.isfinite(estimate).all():
                nonfinite+=1
            errors.append(float((estimate-exact.double()).square().sum()))
        rows.append(dict(reference_seed=reference_seed,M=population,n=auxiliary,B=particles,epsilon=1.,
                         replications=replications,mse=sum(errors)/replications if not nonfinite else float('inf'),
                         nonfinite_gradients=nonfinite))
    print(json.dumps(dict(stage='mf_split',n=auxiliary,B=particles,mse=sum(row['mse'] for row in rows)/len(rows),
                          seconds=time.monotonic()-started)),flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications', type=int, default=120)
    parser.add_argument('--cells',type=int,default=TrafficConfig.cells)
    parser.add_argument('--reward',choices=['local','shared'],default=TrafficConfig.congestion_reward)
    parser.add_argument('--precision',choices=['float32','float64'],default=TrafficConfig.precision)
    parser.add_argument('--select-mf-split',action='store_true')
    parser.add_argument('--initial-laws',choices=['dirichlet','jam','mixed'],default=TrafficConfig.training_initial_law)
    parser.add_argument('--jam-probability',type=float,default=.5)
    parser.add_argument('--horizon',type=int,default=TrafficConfig.T)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--population',type=int,default=200)
    parser.add_argument('--budget-per-update',type=int,default=6360)
    parser.add_argument('--updates',type=int,default=10000)
    parser.add_argument('--select-lambda',action='store_true',help='Select lambda by gradient MSE within the paper grid, without additional training runs')
    parser.add_argument('--baseline',action='store_true',help='Enable the optional leave-one-out baseline; off for the literal estimator')
    parser.add_argument('--out', type=Path, default=Path('results/traffic_calibration_paper'))
    parser.add_argument('--reference-steps', type=int, default=100)
    parser.add_argument('--reference-seeds', type=lambda value: tuple(int(seed) for seed in value.split(',')), default=(0,1,2))
    args = parser.parse_args()
    if args.replications < 1:
        parser.error('--replications must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.budget_per_update % args.horizon:
        parser.error('budget-per-update must be divisible by horizon')
    main_auxiliary_budget=args.budget_per_update//args.horizon-args.population
    if main_auxiliary_budget<2 or args.population<1:
        parser.error('budget must leave positive population, main and auxiliary blocks')
    env = Traffic(TrafficConfig(device=args.device,cells=args.cells,T=args.horizon,T_val=args.horizon,
                                training_initial_law=args.initial_laws,training_jam_probability=args.jam_probability,
                                congestion_reward=args.reward,precision=args.precision))
    refs = references(env, args.reference_steps, args.reference_seeds)
    print('Reference gradient norms:', [(seed, float(gradient.norm())) for seed, _, gradient in refs], flush=True)
    for seed, policy, _ in refs:
        torch.save(policy.state_dict(), args.out/f'reference_seed_{seed}.pt')
    rows = []

    def append_score(n, b, eta, sigma, stage, lambda_multiplier=1.):
        measured = score(env, refs, n, b, eta, sigma, args.replications, stage,
                         population=args.population,baseline=args.baseline,lambda_multiplier=lambda_multiplier)
        rows.extend(measured)
        with (args.out/'gradient_mse.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return sum(row['mse'] for row in measured)/len(measured)

    splits = []
    candidates=sorted({max(1,min(main_auxiliary_budget-1,round(fraction*main_auxiliary_budget))) for fraction in (.1,.25,.5,.75,.9)})
    for n in candidates:
        b=main_auxiliary_budget-n
        error = append_score(n, b, .95, .5, 'split')
        splits.append((error, n, b))
    _, n, b = min(splits)
    settings = []
    for eta in (.85, .95, .98):
        for sigma in (.5, .75, 1.):
            error = append_score(n, b, eta, sigma, 'scales')
            settings.append((error, eta, sigma))
    error, eta, sigma = min(settings)
    multiplier=.5
    selected_stage='scales'
    if args.select_lambda:
        lambda_results=[(append_score(n,b,eta,sigma,'lambda',multiple),multiple) for multiple in (.125,.25,.5,1.,2.)]
        error,multiplier=min(lambda_results)
        selected_stage='lambda'
    selection = dict(M=args.population, n=n, B=b, cells=args.cells,T=args.horizon, updates=args.updates,
                     training_initial_law=args.initial_laws,training_jam_probability=args.jam_probability,
                     congestion_reward=args.reward,precision=args.precision,
                     budget_per_update=args.budget_per_update, total_training_transitions=args.updates*args.budget_per_update,
                     eta=eta, simplex_sigma=sigma, lambda_anchor=b**(-.25),
                     lambda_run=multiplier*b**(-.25), lambda_multiplier=multiplier, baseline=args.baseline,replications=args.replications,
                     selected_mse=error, reference_steps=args.reference_steps, reference_seeds=args.reference_seeds,
                     calibration_gradient_calls=len(rows)*args.replications,
                     calibration_simulator_transitions=len(rows)*args.replications*args.budget_per_update,
                     reference_warm_simulator_transitions=len(refs)*args.reference_steps*env.config.n_particles*args.horizon,
                     budget_excludes=['calibration', 'validation'],
                     criterion='mean unnormalized gradient MSE across three independently warm-started policies',
                     scope='sequential split, joint eta/sigma and optional lambda-grid MSE screens; not a global grid search')
    chosen=[row for row in rows if row['stage']==selected_stage and row['eta']==eta and row['simplex_sigma']==sigma
            and (selected_stage!='lambda' or row['lambda_multiplier']==multiplier)]
    selection['selected_relative_mse']=sum(row['relative_mse'] for row in chosen)/len(chosen)
    selection['screen_has_resolved_signal']=selection['selected_relative_mse']<=1
    if not selection['screen_has_resolved_signal']:
        print('WARNING: best tested setting still has gradient RMSE larger than the signal; selection is not validated for stable training.', flush=True)
    if args.select_mf_split:
        mf_rows=[]
        max_aux=(main_auxiliary_budget-1)//(args.horizon+1)
        mf_candidates=sorted({min(10,max_aux)} | {max(1,round(fraction*max_aux)) for fraction in (.15,.3,.5,.75,.9)})
        mf_errors=[]
        for auxiliary in mf_candidates:
            particles=main_auxiliary_budget-(args.horizon+1)*auxiliary
            measured=score_mf(env,refs,auxiliary,particles,args.population,args.replications,args.baseline)
            mf_rows.extend(measured)
            mse=sum(row['mse'] for row in measured)/len(measured)
            mf_errors.append((mse,auxiliary,particles))
            with (args.out/'mf_gradient_mse.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(mf_rows[0]));writer.writeheader();writer.writerows(mf_rows)
        mse,auxiliary,particles=min(mf_errors)
        if not math.isfinite(mse):
            raise FloatingPointError('No finite MF-REINFORCE calibration candidate')
        selection['mfreinforce']=dict(M=args.population,n=auxiliary,B=particles,epsilon=1.,selected_mse=mse,
                                     budget_per_update=args.horizon*(args.population+particles)+auxiliary*args.horizon*(args.horizon+1))
        selection['mf_calibration_gradient_calls']=len(mf_rows)*args.replications
        selection['mf_calibration_simulator_transitions']=len(mf_rows)*args.replications*args.budget_per_update
    (args.out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
