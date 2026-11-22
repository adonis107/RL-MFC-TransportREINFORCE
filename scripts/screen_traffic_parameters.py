"""Read-only benchmark-parameter variance screen; no training or default changes.

Uses the existing full-return Transport implementation and historical block
sizes/scales. Only the stated environment parameter changes between cases.
The fixed jam law is used for the exact-gradient comparison. Shorter horizons
have lower cost with the same blocks; this is an isolation experiment, not an
equal-cost or retuned training comparison.
"""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from mfc.environments import Traffic,TrafficConfig,TrafficPolicy
from mfc.algorithms import DiscreteTransport,DiscreteTransportConfig
from calibrate_traffic import exact_gradient


CASES = [
    ('current',{}),
    ('horizon_8',dict(T=8,T_val=8)),
    ('horizon_5',dict(T=5,T_val=5)),
    ('cells_3',dict(cells=3)),
    ('cells_2',dict(cells=2)),
    ('safety_cost_05',dict(safety_cost=.5)),
    ('discount_07',dict(gamma=.7)),
    ('cells_3_horizon_5',dict(cells=3,T=5,T_val=5)),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications',type=int,default=120)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_seed0/parameter_screen'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    base=TrafficConfig(device=args.device,cells=4,T=12,T_val=12,congestion_reward='local',precision='float32')
    rows=[]
    for name,updates in CASES:
        start=time.monotonic()
        config=replace(base,**updates)
        env=Traffic(config)
        for seed in (0,1,2):
            torch.manual_seed(seed)
            policy=TrafficPolicy(config)
            exact=exact_gradient(env,policy)
            estimator=DiscreteTransport(env,policy=policy,config=DiscreteTransportConfig(
                horizon=config.T,n_particles=82,n_logit_gradient=248,n_flow_particles=200,
                lambda_=.5*82**(-.25),eta=.98,simplex_sigma=.75,flow='particle',
                auxiliary_batches='shared',baseline=True))
            estimator.sample_initial_distribution=lambda generator: env.initial_distribution
            estimates=torch.stack([estimator.estimate_gradient(100000+r*7919)[0].detach()
                                    for r in range(args.replications)])
            mean=estimates.mean(0)
            variance=float((estimates-mean).square().sum(-1).mean())
            signal=float(exact.norm())
            rows.append(dict(case=name,seed=seed,replications=args.replications,cells=config.cells,
                        states=env.n_states,horizon=config.T,safety_cost=config.safety_cost,gamma=config.gamma,
                        transitions_per_estimate=config.T*(200+248+82),
                        gradient_signal=signal,gradient_variance=variance,rms_gradient_noise=variance**.5,
                        noise_to_signal=variance**.5/signal,
                        relative_rmse=float((estimates-exact).square().sum(-1).mean().sqrt()/exact.norm())))
        print(json.dumps(dict(case=name,mean_noise_to_signal=sum(r['noise_to_signal'] for r in rows[-3:])/3,
                              seconds=time.monotonic()-start)),flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'measurements.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    base_variance=sum(r['gradient_variance'] for r in rows if r['case']=='current')/3
    for name,updates in CASES:
        group=[r for r in rows if r['case']==name]
        var=sum(r['gradient_variance'] for r in group)/3
        summary.append(dict(case=name,changes=updates,mean_noise_to_signal=sum(r['noise_to_signal'] for r in group)/3,
                       variance_relative_to_current=var/base_variance,
                       mean_rms_noise=sum(r['rms_gradient_noise'] for r in group)/3,
                       mean_gradient_signal=sum(r['gradient_signal'] for r in group)/3,
                       transitions_per_estimate=group[0]['transitions_per_estimate']))
    result=dict(method='120 replications per case and initialization, initial seeds 0/1/2; full-return paper estimator; same historical baseline and blocks M=200 n=248 B=82, lambda=B^(-1/4)/2, eta=.98, sigma=.75',
                limitations=['Initialization-only diagnostic, not training-convergence evidence.',
                             'Benchmark changes alter the objective/dynamics.',
                             'Block counts and lambda are fixed; shorter horizons use fewer transitions per estimate.',
                             'Changing cells changes network input/output dimensions and the physical discretization; the architecture width is held fixed.'],
                summary=summary)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
