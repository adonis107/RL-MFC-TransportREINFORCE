"""Traffic/source parity and interface checks: uv run python scripts/verify_traffic.py."""
from dataclasses import replace
from pathlib import Path
import sys
import json
import io
from contextlib import redirect_stdout
import math
import tempfile
from unittest.mock import patch
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "files/MARL-MFC/benchmarks/mfc_traffic"))

from mfc.environments import Traffic, TrafficConfig, TrafficPolicy
from mfc.algorithms.discrete_validation import evaluate_law
import run as run_plan


class TrafficTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TrafficConfig(device="cpu",cells=4,T=12,T_val=12,congestion_reward="local",precision="float32")

    def test_exact_source_parity(self):
        from mfc_traffic.config import TrafficConfig as SourceConfig
        from mfc_traffic.model import TrafficModel

        for cells, fail, slow, fast in ((4, 0., 0., 0.), (5, .3, .2, .1), (2, 1., .5, .5)):
            cfg = replace(self.config, cells=cells, p_fail=fail, p_slow=slow, p_fast=fast)
            env = Traffic(cfg)
            source = TrafficModel(SourceConfig(cells=cells, p_fail=fail, p_slow=slow, p_fast=fast))
            np.testing.assert_allclose(env.transitions.numpy(), source.transitions[0], atol=1e-7)
            np.testing.assert_allclose(env.risk_matrix.numpy(), source.risk_matrix, atol=1e-7)
            rng = np.random.default_rng(17)
            laws = rng.dirichlet(np.ones(env.n_states), 3)
            kernels = rng.dirichlet(np.ones(9), (3, env.n_states))
            mu = torch.tensor(laws, dtype=env.dtype)
            h = torch.tensor(kernels, dtype=env.dtype)
            states = torch.arange(env.n_states)[:, None]
            actions = torch.arange(9)[None, :]
            rewards = env.reward(states, mu[:, None, None, :], actions)
            np.testing.assert_allclose(rewards.numpy(), source.individual_rewards(laws), atol=3e-7)
            np.testing.assert_allclose((mu[..., None] * h * rewards).sum((-1, -2)).numpy(), source.reward(laws, kernels), atol=3e-7)
            flow = torch.einsum('bs,bsa,san->bn', mu, h, env.transitions)
            np.testing.assert_allclose(flow.numpy(), source.phi(laws, kernels, shock=0), atol=1e-7)

    def test_shapes_derivatives_and_boundaries(self):
        env = Traffic(self.config)
        # Wrap forward at the last cell, clip outward lateral movement at each edge.
        self.assertEqual(env.transition(torch.tensor(3), None, torch.tensor(6)).argmax().item(), 1)
        self.assertEqual(env.transition(torch.tensor(11), None, torch.tensor(8)).argmax().item(), 9)
        states = torch.tensor([0, 4, 11])
        mu = env.initial_distribution.expand(3, -1).clone().requires_grad_()
        rewards = env.reward(states, mu, torch.tensor([0, 4, 8]))
        derivative, = torch.autograd.grad(rewards.sum(), mu)
        torch.testing.assert_close(derivative, -env.config.safety_cost * env.risk_matrix[states])
        policy = TrafficPolicy(self.config)
        laws = mu.detach()
        for time in (torch.tensor(1.), torch.arange(3.)):
            probabilities = env.policy(policy, time, states, laws)
            self.assertEqual(probabilities.shape, (3, 9))
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(3))
        self.assertEqual(env.policy(policy, 0, torch.tensor(2), laws[0]).shape, (9,))
        self.assertEqual(env.reward(torch.tensor(2), laws[0], torch.tensor(4)).ndim, 0)
        torch.testing.assert_close(env.terminal_reward(states, laws), torch.zeros(3))

    def test_sampling_and_discounted_objective(self):
        env = Traffic(replace(self.config, p_fail=.3, p_slow=.2, p_fast=.1))
        generator = torch.Generator().manual_seed(29)
        states = torch.full((30000,), 4)
        actions = torch.full_like(states, 8)
        draws = env.sample(states, env.initial_distribution, actions, generator)
        frequencies = torch.bincount(draws, minlength=env.n_states) / states.numel()
        torch.testing.assert_close(frequencies, env.transitions[4, 8], atol=.012, rtol=0)
        # Always stay: the jam is fixed, so discounted running reward is a geometric sum.
        env = Traffic(self.config)
        policy = torch.zeros(env.n_states, 9)
        policy[:, 1] = 1
        value = evaluate_law(env, policy, lambda t: t, env.config.gamma, env.initial_distribution, 12)
        risk = env.initial_distribution @ env.risk_matrix @ env.initial_distribution
        expected = -env.config.safety_cost * risk * sum(env.config.gamma**t for t in range(12))
        torch.testing.assert_close(value, expected)

    def test_suite_budgets_and_config_validation(self):
        jobs = run_plan.experiment_plan("traffic")
        self.assertEqual(len(jobs), 8)
        self.assertEqual({job['algorithm'] for job in jobs}, {'reinforce', 'mfreinforce', 'transport', 'ihmfcac'})
        self.assertEqual(next(job for job in jobs if job['algorithm'] == 'mfreinforce')['perturbation'], 2.)
        params = run_plan.fair_run_parameters(next(job for job in jobs if job['algorithm'] == 'ihmfcac'))
        self.assertEqual(params['max_transitions'], 10000 * 12 * 530)
        self.assertEqual(params['n_train'], params['max_transitions'] // 82)
        with patch.object(sys, 'argv', ['run.py', '--env', 'traffic']):
            args = run_plan.parse_args()
        rows = run_plan.manifest_rows(['traffic'], [0], args)
        self.assertEqual(len(rows), 8)
        self.assertEqual(next(row for row in rows if row['algorithm'] == 'ihmfcac')['budget_per_update'], 82)
        for changes in ({'cells': 1}, {'p_fail': 1.1}, {'p_slow': .8, 'p_fast': .3}, {'gamma': 1.}, {'forward_scale': 0.}):
            with self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_calibration_oracle_and_actual_transition_cost(self):
        from calibrate_traffic import exact_objective, exact_gradient
        from correction_alignment import discrete_gradient
        from mfc.algorithms import DiscreteTransport, DiscreteTransportConfig, Reinforce

        torch.manual_seed(173)
        env = Traffic(replace(self.config, T=3, T_val=3))
        policy = TrafficPolicy(env.config)
        wrapper = Reinforce(env, policy=policy)
        oracle = exact_gradient(env, policy)
        torch.testing.assert_close(oracle, discrete_gradient(env, wrapper, 3, detach=False), atol=1e-6, rtol=1e-5)
        estimator = DiscreteTransport(env, policy=policy, config=DiscreteTransportConfig(
            horizon=3, n_particles=17, n_flow_particles=23, n_logit_gradient=11,
            lambda_=.2, eta=.95, simplex_sigma=.5, flow='particle'))
        transitions = 0
        original = env.sample

        def counted(states, mu, actions, generator):
            nonlocal transitions
            transitions += states.numel()
            return original(states, mu, actions, generator)

        with patch.object(env, 'sample', counted):
            estimator.estimate_gradient(0)
        self.assertEqual(transitions, 3*(17+23+11))

    def test_convex_optimal_reference(self):
        from traffic_optimality_gap import optimal_reference
        from mfc.algorithms.discrete_validation import evaluate_law

        env = Traffic(replace(self.config, safety_cost=0., T=3, T_val=3))
        table, reference = optimal_reference(env)
        expected = float((env.speed-env.config.comfort_cost*env.effort).max())*sum(env.config.gamma**t for t in range(3))
        self.assertAlmostEqual(reference['exact_value_lower_bound'], expected, places=7)
        self.assertLessEqual(reference['dual_gap'], 1e-7)
        env = Traffic(replace(self.config, T=3, T_val=3))
        table, reference = optimal_reference(env, tolerance=1e-6)
        probabilities = torch.tensor(table, dtype=env.dtype)
        value = evaluate_law(env, lambda t, mu: probabilities[int(t)], lambda t: t,
                             env.config.gamma, env.initial_distribution, 3)
        self.assertAlmostEqual(float(value), reference['exact_value_lower_bound'], places=5)
        self.assertLess(reference['flow_residual'], 1e-10)
        self.assertLessEqual(reference['dual_gap'], 1e-6)
        env.risk_matrix = -torch.eye(env.n_states)
        with self.assertRaises(ValueError):
            optimal_reference(env)

    def test_diagnostic_objective_gradient(self):
        from diagnose_traffic import objective
        from calibrate_traffic import exact_objective

        env = Traffic(replace(self.config, T=3, T_val=3))
        policy = TrafficPolicy(env.config).double()
        env.dtype = torch.float64
        for name in ('initial_distribution', 'risk_matrix', 'transitions', 'speed', 'effort'):
            setattr(env, name, getattr(env, name).double())
        value = objective(env, policy, env.initial_distribution, lambda_=0.).mean()
        torch.testing.assert_close(value, exact_objective(env, policy), atol=1e-12, rtol=1e-12)
        value = objective(env, policy, env.initial_distribution, lambda_=.2, q_samples=64).mean()
        derivative, = torch.autograd.grad(value, policy.net[-1].bias)
        epsilon = 1e-5
        with torch.no_grad():
            policy.net[-1].bias[7] += epsilon
            plus = objective(env, policy, env.initial_distribution, lambda_=.2, q_samples=64).mean()
            policy.net[-1].bias[7] -= 2*epsilon
            minus = objective(env, policy, env.initial_distribution, lambda_=.2, q_samples=64).mean()
            policy.net[-1].bias[7] += epsilon
        self.assertAlmostEqual(float(derivative[7]), float((plus-minus)/(2*epsilon)), places=7)

    def test_initial_law_families_and_historical_compatibility(self):
        base = replace(self.config, cells=3, training_initial_law='dirichlet')
        old = Traffic(base)
        generator = torch.Generator().manual_seed(29)
        actual = old.sample_initial_distribution(generator)
        generator = torch.Generator().manual_seed(29)
        weights = -torch.rand(old.n_states, generator=generator).clamp_min(1e-12).log()
        torch.testing.assert_close(actual, weights/weights.sum(), rtol=0, atol=0)
        jam = Traffic(replace(base, training_initial_law='jam'))
        mixed_jam = Traffic(replace(base, training_initial_law='mixed', training_jam_probability=1.))
        mixed_random = Traffic(replace(base, training_initial_law='mixed', training_jam_probability=0.))
        generator = torch.Generator().manual_seed(31)
        for _ in range(50):
            law = jam.sample_initial_distribution(generator)
            torch.testing.assert_close(law.sum(), torch.tensor(1.))
            support = law.reshape(3,3).sum(0)>0
            self.assertEqual(int(support.sum()), 1)
            self.assertEqual(int((law>0).sum()), 3)
            law = mixed_jam.sample_initial_distribution(generator)
            self.assertEqual(int((law>0).sum()), 3)
            law = mixed_random.sample_initial_distribution(generator)
            self.assertTrue(bool((law>0).all()))
        mixed = Traffic(replace(base, training_initial_law='mixed'))
        generator = torch.Generator().manual_seed(44)
        families = [int((mixed.sample_initial_distribution(generator)>0).sum()) for _ in range(100)]
        self.assertIn(3, families)
        self.assertIn(9, families)
        torch.testing.assert_close(mixed.initial_distribution, old.initial_distribution)
        with self.assertRaises(ValueError):
            replace(base, training_jam_probability=1.1)

    def test_mixed_default_legacy_metadata_and_distinct_run_paths(self):
        from dataclasses import asdict
        from unittest.mock import patch
        import train
        from mfc.visualization.io import dataclass_from_dict

        self.assertEqual(TrafficConfig().training_initial_law, 'mixed')
        self.assertEqual(TrafficConfig().training_jam_probability, .5)
        legacy = asdict(self.config)
        legacy.pop('training_initial_law')
        legacy.pop('training_jam_probability')
        self.assertEqual(dataclass_from_dict(TrafficConfig, legacy).training_initial_law, 'dirichlet')
        self.assertEqual(dataclass_from_dict(TrafficConfig, asdict(self.config)).training_initial_law, 'mixed')
        argv=['train.py','--env','traffic','--traffic-cells','4','--traffic-reward','local','--traffic-precision','float32','--algorithm','transport','--horizon','5','--perturbation','0.4','--eta','0.98','--device','cpu']
        with patch.object(sys, 'argv', argv):
            args=train.parse_args()
        env=train.build_environment(args)
        self.assertEqual(env.config.training_initial_law, 'mixed')
        mixed_path=train.output_directory(args)
        self.assertIn('_init_mixed_p_0.5_', str(mixed_path))
        args.traffic_initial_laws='dirichlet'
        historical_path=train.output_directory(args)
        self.assertNotEqual(mixed_path, historical_path)
        self.assertNotIn('_init_', str(historical_path))
        old_env=Traffic(replace(self.config, training_initial_law='dirichlet'))
        torch.testing.assert_close(env.initial_distribution, old_env.initial_distribution)

    def test_headline_outputs_separate_training_law_protocols(self):
        import make_outputs
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'traffic'
            stem='transport_lambda_0.166156_eta_0.98'
            for suffix,config,score in [('',{},100.),('_reward_shared_fp64_init_mixed_p_0.5',{'training_initial_law':'mixed','training_jam_probability':.5,'congestion_reward':'shared','precision':'float64'},.2)]:
                run=directory/f'{stem}{suffix}_T_5_particle_seed_0'
                run.mkdir(parents=True)
                (run/'metadata.json').write_text(json.dumps(dict(env='traffic',algorithm='transport',env_config=config)))
                (run/'history.json').write_text('{}')
                (run/'policy.pt').write_bytes(b'')
                (run/'summary.json').write_text(json.dumps(dict(last_validation_objective=score)))
            selected=make_outputs.load_env_runs(folder,'traffic')
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0]['metadata']['env_config']['training_initial_law'],'mixed')
            self.assertIn('_init_mixed_p_0.5_',make_outputs.transport_stem(folder,'traffic'))
            self.assertIn('_init_mixed_p_0.5_',make_outputs.run_stems(folder,'traffic')['REINFORCE'])

    def test_batched_auxiliary_limit_zero_scale(self):
        from audit_traffic_auxiliary import limiting_gradient
        from diagnose_traffic import objective, gradient
        env = Traffic(replace(self.config, cells=3, T=2, T_val=2))
        policy = TrafficPolicy(env.config)
        laws = torch.stack([env.initial_distribution, torch.full_like(env.initial_distribution, 1/env.n_states)])
        expected = gradient(objective(env, policy, laws, .4, q_samples=64, randomizer_sigma=.5).mean(), policy)
        actual = limiting_gradient(env, policy, 0., .4, count=64, initial_distribution=laws, randomizer_sigma=.5)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)

    def test_social_objective_invariant_to_congestion_reward_decomposition(self):
        from audit_traffic_externality import value, grad
        env=Traffic(replace(self.config,cells=3,T=3,T_val=3))
        env.dtype=torch.float64
        for key in ('initial_distribution','risk_matrix','transitions','speed','effort'):
            setattr(env,key,getattr(env,key).double())
        env.initial_distribution/=env.initial_distribution.sum()
        policy=TrafficPolicy(env.config).double()
        a=value(env,policy,env.initial_distribution,reward='local')
        ga=grad(a,policy)
        b=value(env,policy,env.initial_distribution,reward='global')
        gb=grad(b,policy)
        torch.testing.assert_close(a.detach(),b.detach(),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-9)
        frozen=grad(value(env,policy,env.initial_distribution,True,'global'),policy)
        direct=grad(value(env,policy,env.initial_distribution,True,'control'),policy)
        torch.testing.assert_close(frozen,direct,atol=1e-10,rtol=1e-9)

    def test_shared_population_reward_individual_score_scaling(self):
        import itertools
        for count in (1,2,3,4):
            p=.35
            estimate=0.
            for outcome in itertools.product((0.,1.),repeat=count):
                moved=sum(outcome)
                probability=p**moved*(1-p)**(count-moved)
                population_reward=-(moved/count)**2
                score=sum(action/p-(1-action)/(1-p) for action in outcome)
                estimate+=probability*population_reward*score/count
            full_derivative=-(2*p+(1-2*p)/count)
            self.assertAlmostEqual(estimate,full_derivative/count,places=12)

    def test_shared_reward_precision_and_legacy_loading(self):
        from mfc.visualization.io import dataclass_from_dict
        from calibrate_traffic import exact_objective, exact_gradient
        shared=Traffic(TrafficConfig(device='cpu',cells=3,T=3,T_val=3))
        local=Traffic(replace(shared.config,congestion_reward='local'))
        self.assertEqual(shared.dtype,torch.float64)
        policy=TrafficPolicy(shared.config)
        self.assertEqual(next(policy.parameters()).dtype,torch.float64)
        torch.testing.assert_close(exact_objective(shared,policy),exact_objective(local,policy),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(exact_gradient(shared,policy),exact_gradient(local,policy),atol=1e-11,rtol=1e-10)
        states=torch.arange(9)
        law=shared.initial_distribution.clone().requires_grad_()
        reward=shared.reward(states,law,torch.ones(9,dtype=torch.long))
        self.assertEqual(reward.shape,(9,))
        self.assertTrue(bool((reward==reward[0]).all()))
        derivative,=torch.autograd.grad(reward[0],law)
        expected=-shared.config.safety_cost*((shared.risk_matrix+shared.risk_matrix.T)@law.detach())
        torch.testing.assert_close(derivative,expected)
        legacy=dataclass_from_dict(TrafficConfig,dict(cells=4,T=12,T_val=12,device='cpu'))
        self.assertEqual(legacy.congestion_reward,'local')
        self.assertEqual(legacy.precision,'float32')
        self.assertEqual(legacy.training_initial_law,'dirichlet')

    def test_training_and_checkpoint_reload(self):
        import train
        from mfc.visualization.io import load_runs, load_env_and_policy
        from mfc.visualization.flows import learned_flow

        with tempfile.TemporaryDirectory() as folder:
            for algorithm in ('reinforce', 'mfreinforce', 'transport', 'ihmfcac'):
                argv = ['train.py', '--env', 'traffic', '--algorithm', algorithm,
                        '--horizon', '3', '--n-train', '2', '--n-particles', '16',
                        '--n-flow-particles', '200', '--n-logit-gradient', '8',
                        '--validation-interval', '1', '--validation-particles', '64',
                        '--device', 'cpu', '--results-root', folder]
                if algorithm in {'mfreinforce', 'transport'}:
                    argv += ['--perturbation', '0.2', '--eta', '0.5']
                with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                    out = train.run_training(train.parse_args())
                summary = json.loads((out / 'summary.json').read_text())
                self.assertTrue(math.isfinite(summary['last_validation_objective']))
            runs = load_runs(folder)
            self.assertEqual(len(runs), 4)
            for run in runs:
                env, policy = load_env_and_policy(run, device='cpu')
                self.assertIsInstance(env, Traffic)
                flow = learned_flow(run, horizon=3, device='cpu')
                self.assertEqual(flow.shape, (4, env.n_states))
                torch.testing.assert_close(flow.sum(-1), torch.ones(4,dtype=env.dtype))


if __name__ == '__main__':
    unittest.main()
