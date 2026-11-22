"""Regression checks against equation 13 and exact conditional initialization.

Run with: uv run python scripts/verify_transport_alignment.py
"""
import math
from dataclasses import replace
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mfc.algorithms import (
    ContinuousTransport, ContinuousTransportConfig, DiscreteTransport,
    DiscreteTransportConfig, GaussianTransport, GaussianTransportConfig,
)
from mfc.algorithms.mfreinforce import MFReinforce, MFReinforceConfig
from mfc.algorithms.reinforce import Reinforce, ReinforceConfig
from mfc.algorithms.validation import initial_population_law, particle_law
from mfc.algorithms.sampling import independent_return_advantages
from mfc.environments.randomizers import LawRandomizer
from mfc.environments import (LQ, LQConfig, Portfolio, PortfolioConfig, Bimodal, BimodalConfig,
    TwoState, TwoStateConfig, Advertising, AdvertisingConfig, Cybersecurity, CybersecurityConfig,
    Distribution, DistributionConfig)


class TransportAlignment(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = LQ(LQConfig(T=2, device="cpu"))
        self.generator = torch.Generator().manual_seed(712)

    def algorithm(self, k=1):
        return ContinuousTransport(self.env, config=ContinuousTransportConfig(
            n_components=k, n_particles=16, n_law_gradient=32, n_flow_particles=32,
            lambda_=0.3, eta=0.1, flow="exact" if k == 1 else "particle"))

    def test_equation_13(self):
        for k in (1, 2):
            algorithm = self.algorithm(k)
            mixture = algorithm.mixture
            weights = torch.arange(1, k + 1, dtype=self.env.dtype)
            weights /= weights.sum()
            means = torch.linspace(-1, 1, k, dtype=self.env.dtype).reshape(k, 1)
            std = torch.linspace(0.4, 2.0, k, dtype=self.env.dtype).reshape(k, 1, 1)
            z = mixture.encode(weights, means, std)
            r = algorithm.sample_transport_randomizers((7,), self.generator)
            beta, a, log_b = mixture.unpack(r)
            for scale in (0.0, 0.3, 1.0):
                w, m, factor = mixture.decode(algorithm.transport_coordinate(z, r, scale))
                torch.testing.assert_close(w, (1-scale)*weights + scale*mixture.weights(beta))
                torch.testing.assert_close(m, (1-scale)*means + scale*a)
                torch.testing.assert_close(factor[..., 0, 0], std[..., 0, 0]*((1-scale)+scale*log_b[..., 0].exp()))

    def test_density_change_of_variables_and_score(self):
        # Independently differentiate the forward map R -> encoded Y and use
        # log f_R(R) - log |det dY/dR|, including all mixture coordinates.
        algorithm = self.algorithm(2)
        z = torch.tensor([0.3, -1.2, 0.8, math.log(0.4), math.log(2.0)], dtype=self.env.dtype)
        r = algorithm.sample_transport_randomizers((), self.generator)
        scale = 0.3
        y = algorithm.transport_coordinate(z, r, scale)
        jac = torch.func.jacrev(lambda draw: algorithm.transport_coordinate(z, draw, scale))(r)
        expected = torch.distributions.Normal(0., 1.).log_prob(r).sum() - torch.linalg.slogdet(jac)[1]
        torch.testing.assert_close(algorithm.transport_log_density(y, z, scale), expected)
        score = algorithm.transport_scores(y, z, scale)
        delta = 1e-6 * torch.eye(z.numel(), dtype=z.dtype)
        numerical = torch.stack([
            (algorithm.transport_log_density(y, z+d, scale) - algorithm.transport_log_density(y, z-d, scale))/(2e-6)
            for d in delta
        ])
        torch.testing.assert_close(score, numerical, atol=2e-8, rtol=2e-7)
        # This analytic lognormal scale score distinguishes multiplicative
        # noise from interpolation toward an absolute standard deviation.
        b = r[-2:].exp()
        u = 1-scale+scale*b
        expected_scale_score = u/(scale*b)*(1+r[-2:]) - 1
        torch.testing.assert_close(score[-2:], expected_scale_score)
        outside = y.clone()
        outside[-1] = z[-1] + math.log(0.5)  # below 1-lambda
        self.assertTrue(torch.isneginf(algorithm.transport_log_density(outside, z, scale)))

    def assert_initial_arguments(self, env, work, initial=None):
        seen = []
        original = env.policy
        def policy(theta, t, states, law):
            if not (isinstance(t, torch.Tensor) and torch._C._are_functorch_transforms_active()) and int(t) == 0:
                seen.append(law.detach().clone())
            return original(theta, t, states, law)
        with patch.object(env, "policy", side_effect=policy):
            work()
        self.assertTrue(seen)
        initial = initial_population_law(env) if initial is None else initial
        for law in seen:
            torch.testing.assert_close(law, initial.expand_as(law))

    def test_continuous_initial_population(self):
        for env in (self.env, Portfolio(PortfolioConfig(T=2, device="cpu")), Bimodal(BimodalConfig(device="cpu"))):
            for shared in (True, False):
                algorithm = ContinuousTransport(env, config=ContinuousTransportConfig(
                    n_components=1, n_particles=8, n_law_gradient=32,
                    n_flow_particles=32, lambda_=0.3, eta=0.1,
                    reuse_state_gradient=shared, flow="particle"))
                coordinates, _ = algorithm.population_coordinates(seed=5, jacobians=False)
                sensitivities = algorithm.estimate_coordinate_sensitivities(coordinates, seed=7)
                if shared:
                    work = lambda: algorithm.batched_trajectory_gradient(coordinates, sensitivities, 5)
                else:
                    work = lambda: algorithm.trajectory_gradient(coordinates, sensitivities, 5)
                self.assert_initial_arguments(env, work)
            if hasattr(env, "moment_flow"):
                algorithm = GaussianTransport(env, config=GaussianTransportConfig(
                    n_particles=8, n_law_gradient=16, n_flow_particles=32, lambda_=0.3))
                moments = algorithm.population_moments(5)
                with patch.object(algorithm, "population_moments", return_value=moments):
                    self.assert_initial_arguments(env, lambda: algorithm.estimate_gradient(5))

    def test_discrete_initial_population(self):
        for env_class, config_class in ((TwoState, TwoStateConfig), (Advertising, AdvertisingConfig),
                                       (Cybersecurity, CybersecurityConfig), (Distribution, DistributionConfig)):
            env = env_class(config_class(device="cpu"))
            for shared in (True, False):
                algorithm = DiscreteTransport(env, config=DiscreteTransportConfig(
                    n_particles=8, n_logit_gradient=16, n_flow_particles=16,
                    horizon=2, lambda_=0.3, eta=0.2, flow="particle", reuse_state_gradient=shared))
                initial = env.sample_initial_distribution(torch.Generator().manual_seed(41))
                self.assertFalse(torch.equal(initial, env.initial_distribution))
                laws, log_laws = algorithm.mean_field_law_flow(seed=2, initial_distribution=initial)
                with patch.object(env, "sample_initial_distribution", return_value=initial) as sampler:
                    with patch.object(algorithm, "mean_field_law_flow", return_value=(laws, log_laws)) as flow:
                        self.assert_initial_arguments(env, lambda: algorithm.estimate_gradient(7), initial)
                        sampler.assert_called_once()
                        torch.testing.assert_close(flow.call_args.kwargs["initial_distribution"], initial)
                    algorithm.config = replace(algorithm.config, auxiliary_batches="fresh")
                    self.assert_initial_arguments(env, lambda: algorithm.estimate_state_sensitivities(
                        laws, seed=4, initial_distribution=initial), initial)
                    sampler.assert_called_once()
                generator = torch.Generator().manual_seed(2)
                expected_states = torch.multinomial(initial, 16, replacement=True, generator=generator)
                torch.testing.assert_close(laws[0], algorithm.empirical_law(expected_states))
            baseline = MFReinforce(env, config=MFReinforceConfig(
                n_particles=8, n_logit_gradient=16, n_flow_particles=16,
                horizon=2, perturbation_eta=0.3, flow="particle"))
            flow = baseline.mean_field_law_flow(seed=2, initial_distribution=initial)
            with patch.object(env, "sample_initial_distribution", return_value=initial) as sampler:
                with patch.object(baseline, "mean_field_law_flow", return_value=flow) as population:
                    self.assert_initial_arguments(env, lambda: baseline.estimate_gradient(7), initial)
                    sampler.assert_called_once()
                    torch.testing.assert_close(population.call_args.kwargs["initial_distribution"], initial)

    def test_offline_flow_uses_empirical_population(self):
        algorithm = ContinuousTransport(self.env, config=ContinuousTransportConfig(
            n_components=1, n_flow_particles=32, n_law_gradient=32, flow="particle"))
        # Encoding a sampled flow must not feed the projection back into it.
        with patch.object(algorithm, "population_law", side_effect=AssertionError("projected feedback")):
            coordinates, _ = algorithm.population_coordinates(seed=2, jacobians=False)
            algorithm.shifted_flows(coordinates, seed=3, eta=0.1)

    def test_reinforce_one_step_terminal_law(self):
        env = Bimodal(BimodalConfig(device="cpu"))
        algorithm = Reinforce(env, config=ReinforceConfig(n_particles=128, horizon=1))
        rollout = algorithm.rollout(seed=3)
        expected = env.empirical_law(rollout["terminal_states"])
        torch.testing.assert_close(rollout["terminal_law"], expected)
        self.assertFalse(torch.isclose(expected, initial_population_law(env)).item())
        torch.testing.assert_close(rollout["returns"], env.terminal_reward(rollout["terminal_states"], expected))

    def test_physical_secant_contraction(self):
        algorithm = self.algorithm(2)
        z = torch.tensor([0.3, -1.2, 0.8, math.log(0.4), math.log(2.0)], dtype=self.env.dtype)
        plus, minus = z + 0.2, z - 0.1
        tangent = algorithm.physical_coordinate_secant(plus, minus, z, eta=0.4)
        p, m, l = algorithm.mixture.decode(z)
        physical = torch.cat([p[:-1], m.flatten(), l[..., 0, 0].square()]).requires_grad_(True)
        # Differentiate a physical law functional, independently of the chart.
        pp = torch.cat([physical[:1], 1-physical[:1]])
        value = (pp * (physical[1:3].square() + physical[3:])).sum()
        physical_score = torch.autograd.grad(value, physical)[0]
        p_plus, m_plus, l_plus = algorithm.mixture.decode(plus)
        p_minus, m_minus, l_minus = algorithm.mixture.decode(minus)
        secant = torch.cat([(p_plus-p_minus)[:-1], (m_plus-m_minus).flatten(),
                             l_plus[..., 0, 0].square()-l_minus[..., 0, 0].square()])/0.8
        base = z.clone().requires_grad_(True)
        bp, bm, bl = algorithm.mixture.decode(base)
        chart_score = torch.autograd.grad((bp*(bm[..., 0].square()+bl[..., 0, 0].square())).sum(), base)[0]
        torch.testing.assert_close(chart_score @ tangent, physical_score @ secant)
        self.assertFalse(torch.allclose(tangent, (plus-minus)/0.8))

    def test_leave_one_out_baseline(self):
        # Enumerate independent Bernoulli trajectories. R=X and its parameter
        # score at p=1/2 give d E[R]/dp = 1, including B=1.
        import itertools
        for count in (1, 2, 3):
            estimates = []
            for outcome in itertools.product((0., 1.), repeat=count):
                returns = torch.tensor(outcome)
                score = 4*(returns-0.5)
                estimates.append((score*independent_return_advantages(returns)).mean())
            torch.testing.assert_close(torch.stack(estimates).mean(), torch.tensor(1.))

    def test_discounted_reward_to_go(self):
        import itertools
        env = Advertising(AdvertisingConfig(device="cpu", gamma=0.5))
        algorithms = [MFReinforce(env, config=MFReinforceConfig(horizon=2)),
                      DiscreteTransport(env, config=DiscreteTransportConfig(horizon=2))]
        for algorithm in algorithms:
            estimates = []
            for outcome in itertools.product((0., 1.), repeat=2):
                rewards = [torch.tensor(x) for x in outcome]
                tails = torch.stack(algorithm.discounted_returns(rewards, torch.tensor(0.)))
                scores = 4*(torch.tensor(outcome)-0.5)
                estimates.append((scores*tails[:-1]).sum())
            torch.testing.assert_close(torch.stack(estimates).mean(), torch.tensor(1+algorithm.discount))

    def test_gaussian_randomizer_matches_mixture(self):
        gaussian = GaussianTransport(self.env, config=GaussianTransportConfig(lambda_=0.3))
        mixture = self.algorithm(1)
        g1 = torch.Generator().manual_seed(2)
        g2 = torch.Generator().manual_seed(2)
        a, b = gaussian.sample_randomizers(16, g1)
        r = mixture.sample_transport_randomizers((16,), g2)
        torch.testing.assert_close(a, r[:, 0])
        torch.testing.assert_close(b, r[:, 1].exp())
        c_sigma, c_mean = gaussian.score_coefficients(a, b)
        z = torch.tensor([0.7, math.log(0.4)], dtype=self.env.dtype)
        y = mixture.transport_coordinate(z, r, 0.3)
        torch.testing.assert_close(mixture.transport_scores(y, z, 0.3), torch.stack([c_mean, c_sigma], dim=-1))

    def test_score_expectation_and_gradient(self):
        algorithm = self.algorithm(2)
        z = torch.tensor([0.3, -1.2, 0.8, math.log(0.4), math.log(2.0)], dtype=self.env.dtype,
                         requires_grad=True)
        n = 100_000
        r = algorithm.sample_transport_randomizers((n,), self.generator)
        y = algorithm.transport_coordinate(z, r, 0.3)
        scores = algorithm.transport_scores(y, z, 0.3)
        self.assertTrue(bool((scores.mean(0).abs() < 6*scores.std(0)/math.sqrt(n)).all()))
        # A bounded nonlinear law functional tests all weight, mean and scale
        # channels. Compare its likelihood-ratio gradient to reparametrization.
        weights, means, factors = algorithm.mixture.decode(y)
        value = (weights * (1+factors[..., 0, 0].square()).rsqrt()
                 * torch.exp(-means[..., 0].square()/(2*(1+factors[..., 0, 0].square())))).sum(-1)
        expected = torch.autograd.grad(value.mean(), z)[0]
        estimates = (value.detach()-value.detach().mean()).unsqueeze(-1)*scores
        error = (estimates.mean(0)-expected).abs()
        self.assertTrue(bool((error < 6*estimates.std(0)/math.sqrt(n)+1e-5).all()))

    def test_discrete_objective_oracle_preserves_population_derivative(self):
        from decomposition import MixtureRandomizer, discrete_value
        from verify_theory import discrete_flow
        env = TwoState(TwoStateConfig(device="cpu"))
        algorithm = DiscreteTransport(env, config=DiscreteTransportConfig(horizon=2))
        randomizer = MixtureRandomizer(env.n_states, sigma=1.0, n_draws=1, dtype=env.dtype)
        for scale in (0.0, 0.3):
            path = [randomizer.draws[0]]*3
            value = discrete_flow(env, algorithm, 2, laws=(path, scale))
            reference = discrete_value(env, algorithm.policy, 2, algorithm.discount,
                                       env.initial_distribution, scale, randomizer)
            torch.testing.assert_close(value, reference)
            actual_grad = algorithm.flat_grad(value)
            expected_grad = algorithm.flat_grad(reference)
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_exact_discrete_flow_diagnostic(self):
        env = TwoState(TwoStateConfig(device="cpu"))
        algorithm = DiscreteTransport(env, config=DiscreteTransportConfig(
            horizon=2, flow="exact", n_particles=16, n_logit_gradient=16, lambda_=0.3, eta=0.2))
        laws, _ = algorithm.mean_field_law_flow(seed=2)
        torch.testing.assert_close(laws[0], env.initial_distribution)
        for t in range(2):
            torch.testing.assert_close(laws[t+1], algorithm.mean_field_next_law(t, laws[t]))
        gradient, value = algorithm.estimate_gradient(seed=3)
        self.assertTrue(bool(gradient.isfinite().all() and value.isfinite()))

    def test_simulator_budget_counts(self):
        from train import simulator_budget_estimate
        discrete = TwoState(TwoStateConfig(device="cpu"))
        algorithms = [
            DiscreteTransport(discrete, config=DiscreteTransportConfig(
                horizon=2, n_particles=8, n_logit_gradient=16, n_flow_particles=32,
                lambda_=0.3, eta=0.2, auxiliary_batches=mode)) for mode in ("shared", "fresh")
        ]
        algorithms += [
            MFReinforce(discrete, config=MFReinforceConfig(
                horizon=2, n_particles=8, n_logit_gradient=16, n_flow_particles=32, perturbation_eta=0.3)),
            ContinuousTransport(self.env, config=ContinuousTransportConfig(
                n_components=1, n_particles=8, n_law_gradient=32, n_flow_particles=32, lambda_=0.3, eta=0.1)),
            GaussianTransport(self.env, config=GaussianTransportConfig(
                n_particles=8, n_law_gradient=16, n_flow_particles=32, lambda_=0.3)),
        ]
        for algorithm in algorithms:
            count = [0]
            original = algorithm.env.sample
            def sample(states, law, actions, generator):
                count[0] += states.numel()
                return original(states, law, actions, generator)
            with patch.object(algorithm.env, "sample", side_effect=sample):
                algorithm.estimate_gradient(seed=2)
            self.assertEqual(count[0], simulator_budget_estimate(algorithm))

    def test_analytic_additive_gradients(self):
        for env in (self.env, Portfolio(PortfolioConfig(T=2, device="cpu"))):
            theta = (0.5 * env.optimal_theta()).requires_grad_(True)
            for scale in (0.0, 0.3):
                expected = torch.autograd.grad(env.objective(theta, lambda_=scale), theta)[0]
                torch.testing.assert_close(env.exact_gradient(theta, lambda_=scale), expected)
                explicit = env.objective(theta, lambda_=scale, randomizer=LawRandomizer(kind="mixture", sigma=env.config.rho))
                torch.testing.assert_close(env.objective(theta, lambda_=scale), explicit)
                additive = env.objective(theta, lambda_=scale, randomizer=LawRandomizer(kind="additive", sigma=env.config.rho))
                additive_gradient = torch.autograd.grad(additive, theta)[0]
                torch.testing.assert_close(env.exact_additive_gradient(theta, lambda_=scale), additive_gradient)

    def test_reinforce_resamples_initial_law_once(self):
        env = TwoState(TwoStateConfig(device="cpu"))
        algorithm = Reinforce(env, config=ReinforceConfig(n_particles=16, horizon=2))
        initial = torch.tensor([0.75, 0.25], dtype=env.dtype)
        with patch.object(env, "sample_initial_distribution", return_value=initial) as sampler:
            with patch.object(algorithm, "initial_states", wraps=algorithm.initial_states) as states:
                rollout = algorithm.rollout(seed=7)
                sampler.assert_called_once()
                torch.testing.assert_close(states.call_args.kwargs["initial_distribution"], initial)
            torch.testing.assert_close(rollout["terminal_law"], algorithm.empirical_law(rollout["terminal_states"]))

    def test_initial_law_sampling_and_fixed_validation(self):
        for env_class, config_class in ((TwoState, TwoStateConfig), (Advertising, AdvertisingConfig),
                                       (Cybersecurity, CybersecurityConfig), (Distribution, DistributionConfig)):
            env = env_class(config_class(device="cpu"))
            fixed = env.initial_distribution.clone()
            first = env.sample_initial_distribution(torch.Generator().manual_seed(1))
            second = env.sample_initial_distribution(torch.Generator().manual_seed(2))
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(first.sum(), torch.ones((), dtype=env.dtype))
            self.assertTrue(bool((first > 0).all()))
            algorithm = DiscreteTransport(env, config=DiscreteTransportConfig(horizon=2))
            with patch.object(env, "sample_initial_distribution", side_effect=AssertionError("validation resampled mu_0")):
                self.assertTrue(math.isfinite(float(algorithm.evaluate(n_particles=32, horizon=2, seed=3))))
            torch.testing.assert_close(env.initial_distribution, fixed)


if __name__ == "__main__":
    unittest.main()
