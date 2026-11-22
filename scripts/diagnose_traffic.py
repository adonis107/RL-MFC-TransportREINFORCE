"""Read-only traffic gradient ablations; does not update or retrain a policy."""
import argparse
import csv
import json
from pathlib import Path
import time

import torch
from mfc.algorithms import DiscreteTransport, DiscreteTransportConfig
from mfc.algorithms.sampling import independent_return_advantages
from mfc.environments import TrafficPolicy
from mfc.visualization.io import load_runs, load_env_and_policy


def objective(env, policy, initial, lambda_=0., q_samples=256, seed=9421, randomizer_sigma=.75):
    """Differentiate the population recursion, integrating states/actions exactly.

    Main randomizers are integrated with antithetic Gaussian Monte Carlo samples.
    Current randomizers are independent of the current main marginal, so its
    recursion uses the averaged policy kernel. The unperturbed population flow
    is propagated separately and keeps its full parameter derivative.
    """
    mu = initial.unsqueeze(0) if initial.ndim==1 else initial
    marginal = mu
    value = mu.new_zeros(len(mu))
    generator = torch.Generator(device=env.device).manual_seed(seed)
    c = env.speed-env.config.comfort_cost*env.effort
    for t in range(env.config.T):
        h = policy(t, mu)
        if t==0 or lambda_==0:
            main_h = h
            main_r = h*(c-env.config.safety_cost*(mu@env.risk_matrix.T)[..., None])
        else:
            z = torch.randn(q_samples//2, len(mu), env.n_states-1, device=env.device, generator=generator)*randomizer_sigma
            z = torch.cat([z, -z])
            z = torch.cat([z, torch.zeros_like(z[..., :1])], dim=-1)
            q = z.softmax(-1)
            m = (1-lambda_)*mu.unsqueeze(0)+lambda_*q
            kernels = policy(t, m)
            main_h = kernels.mean(0)
            main_r = (kernels*(c-env.config.safety_cost*(m@env.risk_matrix.T)[..., None])).mean(0)
        value = value+env.config.gamma**t*(marginal[..., None]*main_r).sum((-1,-2))
        mu = torch.einsum('bs,bsa,san->bn', mu, h, env.transitions)
        marginal = torch.einsum('bs,bsa,san->bn', marginal, main_h, env.transitions)
    return value


def gradient(value, policy, retain_graph=False):
    return torch.cat([g.flatten() for g in torch.autograd.grad(value, tuple(policy.parameters()), retain_graph=retain_graph)])


def true_sensitivities(env, policy):
    mu = env.initial_distribution
    d = sum(p.numel() for p in policy.parameters())
    result = [torch.zeros(env.n_states, d, device=env.device)]
    for t in range(env.config.T):
        mu = torch.einsum('s,sa,san->n', mu, policy(t, mu), env.transitions)
        result.append(torch.stack([gradient(mu[s], policy, retain_graph=True).detach() for s in range(env.n_states)]))
    return result


class DiagnosticTransport(DiscreteTransport):
    def __init__(self, *args, true_d=None, reward_to_go=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_d, self.reward_to_go = true_d, reward_to_go

    def estimate_state_sensitivities(self, *args, **kwargs):
        return self.true_d if self.true_d is not None else super().estimate_state_sensitivities(*args, **kwargs)

    def batched_trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        if not self.reward_to_go:
            return super().batched_trajectory_gradient(laws, sensitivities, seed, initial_distribution)
        generator = torch.Generator(device=self.env.device).manual_seed(seed)
        states = self.initial_states(self.n_particles, generator, initial_distribution)
        rewards, log_probabilities, scores = [], [], []
        for t in range(self.horizon):
            q = self.sample_simplex_batch(self.n_particles, generator)
            mu = self.initial_population_argument(initial_distribution).expand_as(q) if t==0 else self.perturb_law(laws[t], q, self.lambda_)
            actions, logs = self.sample_actions_with_log_probs(t, states, mu, generator)
            rewards.append(self.env.reward(states, mu, actions))
            log_probabilities.append(logs)
            scores.append(torch.zeros_like(q[..., :-1]) if t==0 else self.simplex_score_h(q))
            with torch.no_grad():
                states = self.sample_next_state(t, states, mu, actions, generator)
        q = self.sample_simplex_batch(self.n_particles, generator)
        terminal = self.env.terminal_reward(states, self.perturb_law(laws[-1],q,self.lambda_))
        tails = torch.stack(self.discounted_returns(rewards, terminal))
        weights = independent_return_advantages(tails, dim=1).detach()
        g = self.flat_grad((torch.stack(log_probabilities)*weights[:-1]).sum())
        law_g = torch.zeros_like(g)
        for t in range(self.horizon):
            law_g += (scores[t].T@weights[t])@sensitivities[t][:-1]
        g = (g-(1-self.lambda_)/self.lambda_*law_g)/self.n_particles
        return g, tails[0].detach().mean()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications',type=int,default=120)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--only',default=None)
    parser.add_argument('--output',type=Path,default=Path('outputs/traffic_seed0/diagnosis'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    run=load_runs('results/traffic')[0]
    env, final=load_env_and_policy(run, device=args.device)
    torch.manual_seed(0)
    initial=TrafficPolicy(env.config)
    generator=torch.Generator(device=env.device).manual_seed(91023)
    weights=-torch.rand(128,env.n_states,device=env.device,generator=generator).clamp_min(1e-12).log()
    test_laws=weights/weights.sum(-1,keepdim=True)
    root=args.output; root.mkdir(parents=True, exist_ok=True)
    report={}
    rows=[]
    actual=run['metadata']['perturbation']
    for name, policy in [('initial',initial),('final',final)]:
        exact=gradient(objective(env,policy,env.initial_distribution).mean(),policy).detach()
        train_g=gradient(objective(env,policy,test_laws).mean(),policy).detach()
        report[name]={'jam_gradient_norm':float(exact.norm()),'random_law_gradient_norm':float(train_g.norm()),
                      'jam_random_gradient_cosine':float(exact@train_g/(exact.norm()*train_g.norm()))}
        with torch.no_grad():
            for scale in (0.,actual/2,actual,actual*2,actual*4):
                report[name][f'J_lambda_{scale:.6g}']={'jam':float(objective(env,policy,env.initial_distribution,scale,1024).mean()),
                    'random_laws':float(objective(env,policy,test_laws,scale,128).mean())}
        true_d=true_sensitivities(env,policy)
        configurations=[('standard_half_lambda',actual/2,False,False),('standard_actual',actual,False,False),
            ('standard_double_lambda',actual*2,False,False),('standard_quadruple_lambda',actual*4,False,False),
            ('reward_to_go',actual,True,False),('true_D',actual,False,True),('true_D_reward_to_go',actual,True,True),('exact_flow_estimated_D',actual,False,False)]
        for label, scale, rtg, oracle_d in configurations:
            if args.only is not None and label!=args.only:
                continue
            start=time.monotonic()
            oracle=gradient(objective(env,policy,env.initial_distribution,scale,4096).mean(),policy).detach()
            estimator=DiagnosticTransport(env,policy=policy,true_d=true_d if oracle_d else None,reward_to_go=rtg,
                config=DiscreteTransportConfig(n_particles=82,n_logit_gradient=248,n_flow_particles=200,
                    horizon=12,lambda_=scale,eta=.98,simplex_sigma=.75,flow='exact' if oracle_d or label=='exact_flow_estimated_D' else 'particle'))
            estimator.sample_initial_distribution=lambda generator: env.initial_distribution
            estimates=[]; returns=[]
            for r in range(args.replications):
                g,j=estimator.estimate_gradient(100000+r*7919)
                estimates.append(g.detach()); returns.append(float(j))
            estimates=torch.stack(estimates)
            mean=estimates.mean(0)
            variance=float((estimates-mean).square().sum(-1).mean())
            row=dict(policy=name,configuration=label,lambda_=scale,replications=args.replications,
                     target_J_gradient_norm=float(exact.norm()),target_J_lambda_gradient_norm=float(oracle.norm()),
                     oracle_smoothing_relative_bias=float((oracle-exact).norm()/exact.norm()),
                     oracle_J_J_lambda_gradient_cosine=float(oracle@exact/(oracle.norm()*exact.norm())),
                     rms_gradient_noise=variance**.5,noise_to_true_J_gradient=variance**.5/float(exact.norm()),
                     mean_gradient_norm=float(mean.norm()),mean_cosine_to_J=float(mean@exact/(mean.norm()*exact.norm())),
                     mean_cosine_to_J_lambda=float(mean@oracle/(mean.norm()*oracle.norm())),
                     rmse_to_J=float((estimates-exact).square().sum(-1).mean().sqrt()/exact.norm()),
                     rmse_to_J_lambda=float((estimates-oracle).square().sum(-1).mean().sqrt()/oracle.norm()),
                     oracle_J_lambda=float(objective(env,policy,env.initial_distribution,scale,4096).mean().detach()),
                     sampled_J_lambda_mean=sum(returns)/len(returns),seconds=time.monotonic()-start)
            rows.append(row)
            with (root/'gradient_ablations.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            print(json.dumps(row),flush=True)
        (root/'objectives.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
