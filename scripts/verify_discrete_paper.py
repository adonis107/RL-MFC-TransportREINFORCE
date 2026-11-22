"""Independent finite-state Transport checks against the paper's equations.

No policy optimization is performed. Uses full-return weighting, the stated
simplex score and the shared/fresh auxiliary recursions without replacing them.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mfc.algorithms import DiscreteTransport, DiscreteTransportConfig
from mfc.environments import Traffic, TrafficConfig


class DiscretePaperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.env=Traffic(TrafficConfig(cells=4,T=3,T_val=3,device='cpu',congestion_reward='local',precision='float32'))
        self.env.dtype=torch.float64
        for name in ('initial_distribution','risk_matrix','transitions','speed','effort'):
            setattr(self.env,name,getattr(self.env,name).double())
        self.est=DiscreteTransport(self.env,config=DiscreteTransportConfig(
            horizon=3,n_particles=5,n_logit_gradient=7,n_flow_particles=11,
            lambda_=.31,eta=.78,simplex_sigma=.75,flow='exact'))
        self.est.policy.double()
        self.initial=self.env.initial_distribution
        self.laws,_=self.est.mean_field_law_flow(initial_distribution=self.initial)

    def score_row(self,time,state,law,action):
        probabilities=self.est.policy(torch.tensor(float(time),dtype=self.env.dtype),law)
        p=probabilities[state].clamp_min(1e-12)
        value=(p[action]/p.sum()).log()
        grads=torch.autograd.grad(value,tuple(self.est.policy.parameters()))
        return torch.cat([g.flatten() for g in grads])

    def trajectories(self,length,n,generator,scale):
        state=self.est.initial_states(n,generator,self.initial)
        states=[state]
        laws=[];actions=[];scores=[];rewards=[];hs=[]
        for t in range(length):
            q=self.est.sample_simplex_batch(n,generator)
            law=self.initial.expand_as(q) if t==0 else (1-scale)*self.laws[t]+scale*q
            with torch.no_grad():
                action,_=self.est.sample_actions_with_log_probs(t,state,law,generator)
            laws.append(law);actions.append(action)
            scores.append(torch.stack([self.score_row(t,state[r],law[r],action[r]) for r in range(n)]))
            rewards.append(self.env.reward(state,law,action))
            hs.append(torch.zeros_like(q[:,:-1]) if t==0 else self.est.simplex_score_h(q))
            state=self.est.sample_next_state(t,state,law,action,generator)
            states.append(state)
        return states,laws,actions,scores,rewards,hs

    def test_simplex_score_density_and_base_coordinate_sign(self):
        q=self.est.sample_simplex_batch(9,torch.Generator().manual_seed(7))
        less=q[:,:-1].detach().requires_grad_()
        full=torch.cat([less,1-less.sum(-1,keepdim=True)],-1)
        z=(full[:,:-1]/full[:,-1:]).log()
        log_density=-z.square().sum(-1)/(2*self.est.config.simplex_sigma**2)-full.log().sum(-1)
        expected,=torch.autograd.grad(log_density.sum(),less)
        torch.testing.assert_close(self.est.simplex_score_h(q),expected,atol=2e-11,rtol=2e-11)
        base=(torch.ones(9,self.env.n_states-1,dtype=self.env.dtype)/self.env.n_states).requires_grad_()
        lam=self.est.lambda_
        y=((1-lam)*base+lam*q[:,:-1]).detach()
        inverse=(y-(1-lam)*base)/lam
        inverse=torch.cat([inverse,1-inverse.sum(-1,keepdim=True)],-1)
        z=(inverse[:,:-1]/inverse[:,-1:]).log()
        density=-z.square().sum(-1)/(2*self.est.config.simplex_sigma**2)-inverse.log().sum(-1)
        derivative,=torch.autograd.grad(density.sum(),base)
        torch.testing.assert_close(derivative,-(1-lam)/lam*self.est.simplex_score_h(q),atol=1e-10,rtol=1e-10)

    def test_batched_policy_scores_hold_population_fixed(self):
        times=torch.tensor([0.,1.,2.,2.],dtype=self.env.dtype)
        states=torch.tensor([0,4,8,11]);actions=torch.tensor([0,3,6,8])
        laws=self.est.sample_simplex_batch(4,torch.Generator().manual_seed(6))
        actual=self.est.per_sample_log_prob_gradients(times,states,laws,actions)
        expected=torch.stack([self.score_row(times[r],states[r],laws[r],actions[r]) for r in range(4)])
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)

    def test_shared_auxiliary_literal_paper_equation(self):
        seed=217
        states,_,_,policy_scores,_,h=self.trajectories(3,7,torch.Generator().manual_seed(seed),self.est.eta)
        d=[torch.zeros(self.env.n_states,self.est.n_parameters,dtype=self.env.dtype)]
        for t in range(1,4):
            estimate=torch.zeros_like(d[0])
            for r in range(7):
                total=sum(policy_scores[s][r] for s in range(t))
                for s in range(1,t):
                    total=total-(1-self.est.eta)/self.est.eta*(d[s][:-1].T@h[s][r])
                # The paper specifies only the first N-1 indicator rows.
                if states[t][r]<self.env.n_states-1:
                    estimate[states[t][r]]+=total/7
            estimate[-1]=-estimate[:-1].sum(0)
            d.append(estimate)
        actual=self.est.estimate_state_sensitivities(self.laws,seed,self.initial)
        for a,e in zip(actual,d):
            torch.testing.assert_close(a,e,atol=1e-12,rtol=1e-10)

    def test_fresh_auxiliary_literal_paper_equation(self):
        seed=218
        generator=torch.Generator().manual_seed(seed)
        d=[torch.zeros(self.env.n_states,self.est.n_parameters,dtype=self.env.dtype)]
        # Fresh implementation does not draw the unused time-zero randomizer.
        for t in range(1,4):
            state=self.est.initial_states(7,generator,self.initial)
            cumulative=torch.zeros(7,self.est.n_parameters,dtype=self.env.dtype)
            for s in range(t):
                if s==0:
                    law=self.initial.expand(7,-1)
                else:
                    q=self.est.sample_simplex_batch(7,generator)
                    law=(1-self.est.eta)*self.laws[s]+self.est.eta*q
                    cumulative-=((1-self.est.eta)/self.est.eta)*(self.est.simplex_score_h(q)@d[s][:-1])
                with torch.no_grad():
                    action,_=self.est.sample_actions_with_log_probs(s,state,law,generator)
                cumulative+=torch.stack([self.score_row(s,state[r],law[r],action[r]) for r in range(7)])
                state=self.est.sample_next_state(s,state,law,action,generator)
            estimate=torch.zeros_like(d[0])
            for r in range(7):
                if state[r]<self.env.n_states-1:
                    estimate[state[r]]+=cumulative[r]/7
            estimate[-1]=-estimate[:-1].sum(0)
            d.append(estimate)
        actual=self.est.fresh_state_sensitivities(self.laws,seed,self.initial)
        for a,e in zip(actual,d):
            torch.testing.assert_close(a,e,atol=1e-12,rtol=1e-10)

    def test_online_full_return_literal_paper_equation(self):
        seed=219
        generator=torch.Generator().manual_seed(seed)
        states,_,_,scores,rewards,h=self.trajectories(3,5,generator,self.est.lambda_)
        q_terminal=self.est.sample_simplex_batch(5,generator)
        h.append(self.est.simplex_score_h(q_terminal))
        terminal=self.env.terminal_reward(states[-1],(1-self.est.lambda_)*self.laws[-1]+self.est.lambda_*q_terminal)
        returns=sum(self.est.discount**t*reward for t,reward in enumerate(rewards))+self.est.discount**3*terminal
        random=torch.Generator().manual_seed(5)
        d=[torch.zeros(self.env.n_states,self.est.n_parameters,dtype=self.env.dtype)]
        for _ in range(3):
            matrix=torch.randn(self.env.n_states,self.est.n_parameters,dtype=self.env.dtype,generator=random)*.01
            matrix[-1]=-matrix[:-1].sum(0)
            d.append(matrix)
        for baseline in (False,True):
            self.est.config=replace(self.est.config,baseline=baseline)
            expected=torch.zeros(self.est.n_parameters,dtype=self.env.dtype)
            for r in range(5):
                full_score=sum(scores[t][r] for t in range(3))
                for t in range(1,4):
                    full_score-=(1-self.est.lambda_)/self.est.lambda_*(d[t][:-1].T@h[t][r])
                weight=returns[r]-(returns.sum()-returns[r])/4 if baseline else returns[r]
                expected+=full_score*weight/5
            actual,value=self.est.batched_trajectory_gradient(self.laws,d,seed,self.initial)
            torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
            torch.testing.assert_close(value,returns.mean())

    def test_calibration_selects_raw_mse_not_normalized_mse(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from unittest.mock import patch
        import calibrate_traffic

        refs=[(seed,torch.nn.Linear(1,1),torch.ones(1)) for seed in (0,1,2)]

        def measured(env, refs, n, b, eta, sigma, replications, stage, **kwargs):
            # Raw and normalized errors deliberately rank candidates differently.
            mse=(1. if n==297 else 2. if n==248 else 100.) if stage=='split' else (1. if (eta,sigma)==(.98,.75) else 2.)
            relative=100. if n==297 else .1
            return [dict(stage=stage,reference_seed=seed,M=200,n=n,B=b,eta=eta,
                         simplex_sigma=sigma,mse=mse,relative_mse=relative) for seed in (0,1,2)]

        with tempfile.TemporaryDirectory() as directory:
            argv=['calibrate_traffic.py','--out',directory,'--replications','1','--horizon','12','--cells','4']
            with patch.object(sys,'argv',argv), patch.object(calibrate_traffic,'references',return_value=refs), \
                 patch.object(calibrate_traffic,'score',side_effect=measured), redirect_stdout(io.StringIO()):
                calibrate_traffic.main()
            selected=json.loads((Path(directory)/'selection.json').read_text())
            self.assertEqual((selected['n'],selected['B']),(297,33))
            self.assertEqual((selected['eta'],selected['simplex_sigma']),(.98,.75))
            self.assertFalse(selected['screen_has_resolved_signal'])

    def test_variant_cli_and_calibration_budget(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from unittest.mock import patch
        import calibrate_traffic
        import train

        argv=['train.py','--env','traffic','--algorithm','transport','--horizon','5',
              '--traffic-cells','3','--n-flow-particles','200','--n-logit-gradient','965',
              '--n-particles','107','--perturbation','0.3','--no-baseline','--device','cpu',
              '--traffic-initial-laws','mixed','--traffic-jam-probability','0.75']
        with patch.object(sys,'argv',argv):
            args=train.parse_args()
        env=train.build_environment(args)
        self.assertEqual((env.n_states,env.config.T,env.config.T_val),(9,5,5))
        self.assertEqual(env.config.training_initial_law,'mixed')
        self.assertEqual(env.config.training_jam_probability,.75)
        algorithm=train.build_algorithm(args,env)
        self.assertFalse(algorithm.config.baseline)
        self.assertEqual(train.simulator_budget_estimate(algorithm),6360)
        refs=[(seed,torch.nn.Linear(1,1),torch.ones(1)) for seed in (0,1,2)]

        def measured(env, refs, n, b, eta, sigma, replications, stage, **kwargs):
            self.assertEqual(env.config.T,5)
            self.assertEqual(env.n_states,9)
            self.assertEqual(n+b+kwargs['population'],1272)
            self.assertFalse(kwargs['baseline'])
            multiplier=kwargs['lambda_multiplier']
            mse=(1. if n==965 else 100.) if stage=='split' else ((2-multiplier)**2+1 if stage=='lambda' else (1. if (eta,sigma)==(.98,.75) else 2.))
            return [dict(stage=stage,reference_seed=seed,M=200,n=n,B=b,eta=eta,
                         simplex_sigma=sigma,mse=mse,relative_mse=10.,lambda_multiplier=multiplier) for seed in (0,1,2)]

        with tempfile.TemporaryDirectory() as directory:
            argv=['calibrate_traffic.py','--out',directory,'--replications','1','--cells','3','--horizon','5','--select-lambda']
            with patch.object(sys,'argv',argv), patch.object(calibrate_traffic,'references',return_value=refs), \
                 patch.object(calibrate_traffic,'score',side_effect=measured), redirect_stdout(io.StringIO()):
                calibrate_traffic.main()
            selected=json.loads((Path(directory)/'selection.json').read_text())
            self.assertEqual((selected['n'],selected['B']),(965,107))
            self.assertEqual(selected['lambda_multiplier'],2.)
            self.assertEqual(selected['total_training_transitions'],63600000)


if __name__=='__main__':
    unittest.main()
