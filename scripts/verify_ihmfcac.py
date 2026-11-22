"""Behavioral checks of FLZ Algorithm 2: uv run python scripts/verify_ihmfcac.py."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mfc.algorithms import IHMFCAC, IHMFCACConfig, StationaryActor
from mfc.environments import (TwoState, TwoStateConfig, Advertising, AdvertisingConfig,
                              Cybersecurity, CybersecurityConfig, Distribution, DistributionConfig,
                              LQ, LQConfig, Portfolio, PortfolioConfig, Bimodal, BimodalConfig)
from mfc.visualization.flows import learned_flow, final_policy_probabilities
from mfc.visualization.io import load_env_and_policy, load_runs, validation_dataframe


class AlgorithmTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = TwoState(TwoStateConfig(device="cpu", validation_particles=32))
        self.config = IHMFCACConfig(n_train=3, horizon=1, validation_interval=0,
                                    actor_width=4, critic_width=5)

    def algorithm(self, **updates):
        return IHMFCAC(self.env, replace(self.config, **updates))

    def test_forced_actions_envelope_td_and_independent_critic_updates(self):
        ac = self.algorithm()
        ac.initialize_paths()
        # Only bins (0,0) and (1,1) are in their own state cells.
        ac.states = torch.tensor([0, 1, 0, 1, 0])
        ac.population = ac.features(ac.states)
        before_laws = ac.population.clone()
        for critic, value in zip(ac.critics, [-4., -2., 3., 5.]):
            with torch.no_grad():
                for parameter in critic.parameters():
                    parameter.zero_()
                critic[-1].bias.fill_(value)
        before = [[p.clone() for p in critic.parameters()] for critic in ac.critics]
        # All next states are 1: only its action critics (values 3 and 5) are eligible.
        with patch.object(ac.policy, "sample_actions", return_value=torch.zeros(5, dtype=torch.long)), \
             patch.object(self.env, "sample", return_value=torch.ones(5, dtype=torch.long)), \
             patch.object(self.env, "reward", return_value=torch.ones(5)), \
             patch.object(self.env, "terminal_reward", side_effect=AssertionError("terminal training reward")):
            result = ac.step()
        self.assertEqual(result["actions"].tolist(), [0, 0, 0, 1, 0])
        self.assertEqual(result["active_bins"].tolist(), [True, False, False, True])
        torch.testing.assert_close(result["targets"], torch.full((5,), 1 + ac.config.discount * 5))
        torch.testing.assert_close(result["actor_td"], torch.tensor(1 + ac.config.discount * 5 + 2))
        for i in (1, 2):
            for previous, current in zip(before[i], ac.critics[i].parameters()):
                torch.testing.assert_close(previous, current)
                self.assertIsNone(current.grad)  # frozen target did not backpropagate here
        for i in (0, 3):
            expected = before[i][-1] + 2 * ac.config.critic_lr * (result["targets"][i] - before[i][-1])
            torch.testing.assert_close(ac.critics[i][-1].bias, expected)
        observed = ac.features(torch.ones(5, dtype=torch.long))
        expected = (1-ac.config.population_lr) * before_laws + ac.config.population_lr * observed
        torch.testing.assert_close(ac.population, expected)
        torch.testing.assert_close(ac.population.sum(dim=-1), torch.ones(5))
        self.assertFalse(ac.population.requires_grad)

    def test_actor_sgd_uses_detached_advantage_and_sampled_action(self):
        ac = self.algorithm()
        ac.initialize_paths()
        states = ac.states.clone()
        params = list(ac.policy.parameters())
        before = [p.clone() for p in params]
        result = ac.step()
        # Reconstruct the pre-update actor, and differentiate only its log likelihood.
        old_actor = StationaryActor(**ac.policy.spec)
        with torch.no_grad():
            for p, old in zip(old_actor.parameters(), before):
                p.copy_(old)
        expected_loss = -result["actor_td"] * old_actor.distribution(states[-1:]).log_prob(result["actions"][-1:]).sum()
        gradients = torch.autograd.grad(expected_loss, tuple(old_actor.parameters()))
        for old, gradient, actual in zip(before, gradients, params):
            torch.testing.assert_close(actual, old - ac.config.actor_lr * gradient)

    def test_budget_counts_actual_transitions_and_no_horizon_reset(self):
        ac = self.algorithm(n_train=20, max_transitions=12)
        count = 0
        sample = self.env.sample
        def counted(states, laws, actions, generator):
            nonlocal count
            count += states.numel()
            return sample(states, laws, actions, generator)
        with patch.object(self.env, "sample", side_effect=counted), \
             patch.object(ac, "initialize_paths", wraps=ac.initialize_paths) as initialize, \
             patch.object(self.env, "terminal_reward", side_effect=AssertionError("terminal training reward")):
            _, history = ac.train()
            initialize.assert_called_once()
        self.assertEqual(ac.iterations, 2)  # horizon=1 does not terminate or reset training
        self.assertEqual(count, 10)
        self.assertEqual(ac.simulator_transitions, count)
        self.assertEqual(history["simulator_transitions"], [5, 10])
        with self.assertRaises(RuntimeError):
            ac.step()

    def test_rates_and_configuration(self):
        ac = self.algorithm()
        first, later = ac.learning_rates(0), ac.learning_rates(10000)
        self.assertLess(later[0]/later[1], first[0]/first[1])
        self.assertLess(later[1]/later[2], first[1]/first[2])
        for updates in ({"discount": 1.0}, {"discount": 0.0}, {"critic_lr": 1e-6},
                        {"actor_power": 0.6}, {"max_transitions": -1}):
            with self.assertRaises(ValueError):
                self.algorithm(**updates)

    def test_all_supported_simulators_and_seeds(self):
        constructors = [(TwoState, TwoStateConfig), (Advertising, AdvertisingConfig),
                        (Cybersecurity, CybersecurityConfig), (Distribution, DistributionConfig), (LQ, LQConfig)]
        for env_class, config_class in constructors:
            env = env_class(config_class(device="cpu", validation_particles=16))
            config = replace(self.config, state_bins=2, action_bins=2)
            first, second = IHMFCAC(env, config), IHMFCAC(env, config)
            for _ in range(2):
                first.step()
                second.step()
            torch.testing.assert_close(first.states, second.states)
            torch.testing.assert_close(first.population, second.population)
            for a, b in zip(first.policy.parameters(), second.policy.parameters()):
                torch.testing.assert_close(a, b)
            self.assertTrue(math.isfinite(float(first.evaluate(n_particles=16, horizon=1))))
        for env in (Portfolio(PortfolioConfig(device="cpu")), Bimodal(BimodalConfig(device="cpu"))):
            with self.assertRaisesRegex(ValueError, "continuing problem"):
                IHMFCAC(env, self.config)

    def test_continuous_cells_and_exact_empirical_mean_update(self):
        env = LQ(LQConfig(device="cpu"))
        config = replace(self.config, state_bins=2, action_bins=2)
        ac = IHMFCAC(env, config)
        ac.initialize_paths()
        torch.testing.assert_close(ac.population, ac.states)
        self.assertEqual(ac.state_bin(torch.tensor([-100., -1., 0., 100.], dtype=env.dtype)).tolist(), [0, 0, 1, 1])
        old = ac.population.clone()
        result = ac.step()
        torch.testing.assert_close(ac.population, (1-config.population_lr) * old + config.population_lr * ac.states)
        distribution = ac.policy.distribution(ac.states)
        self.assertTrue(bool((distribution.scale > config.std_floor).all()))
        self.assertTrue(bool(result["targets"].isfinite().all()))

    def test_checkpoint_policy_loading_validation_and_cli(self):
        from train import parse_args, run_training, simulator_budget_estimate
        from mfc.visualization.tables import objective_table, runtime_table
        for name in ("cybersecurity", "lq"):
            with tempfile.TemporaryDirectory() as root:
                argv = ["train.py", "--env", name, "--algorithm", "ihmfcac", "--horizon", "2",
                        "--device", "cpu", "--n-train", "2", "--validation-interval", "1",
                        "--validation-particles", "16", "--state-bins", "2", "--action-bins", "2",
                        "--results-root", root]
                with patch.object(sys, "argv", argv):
                    path = run_training(parse_args())
                runs = load_runs(root)
                self.assertEqual(len(runs), 1)
                env, actor = load_env_and_policy(runs[0], device="cpu")
                self.assertTrue(actor.is_stationary_actor)
                flow = learned_flow(runs[0])
                self.assertEqual(flow.shape[0], 3)
                self.assertFalse(objective_table(runs).empty)
                if name == "cybersecurity":
                    self.assertEqual(final_policy_probabilities(env, actor).shape, (4, 2))
                metadata = runs[0]["metadata"]
                self.assertEqual(metadata["training_discount"], math.exp(-0.01))
                self.assertEqual(metadata["resolved_discount"], env.config.discount)
                self.assertEqual(runs[0]["history"]["validation_steps"], [1, 2])
                frame = validation_dataframe(runs)
                self.assertEqual(frame["simulator_transitions"].tolist(),
                                 [metadata["simulator_budget_estimate"], 2 * metadata["simulator_budget_estimate"]])
                checkpoint = torch.load(path / "algorithm.pt", weights_only=False)
                self.assertEqual(checkpoint["simulator_transitions"], metadata["simulator_transitions"])
                self.assertEqual(runtime_table(runs)["total_simulator_transitions_mean"].iloc[0],
                                 metadata["simulator_transitions"])
                restored = StationaryActor.from_payload(checkpoint["actor"], device=env.device, dtype=env.dtype)
                states = torch.arange(env.n_states) if hasattr(env, "n_states") else torch.zeros(3, dtype=env.dtype)
                expected = actor.distribution(states)
                actual = restored.distribution(states)
                torch.testing.assert_close(actual.probs if hasattr(env, "n_states") else actual.loc,
                                           expected.probs if hasattr(env, "n_states") else expected.loc)
                import make_outputs as plots
                with patch.dict(plots.run_plan.TRANSPORT_ALLOCATIONS[name], {"horizon": 2}):
                    figure, axes = plots.plt.subplots(1, 2)
                    plots.draw_ihmfc_learning(axes[0], root, name, gap=name == "lq")
                    plots.draw_ihmfc_flow(axes[1], root, name)
                    self.assertEqual(axes[0].lines[0].get_label(), "IH-MFC-AC")
                    torch.testing.assert_close(torch.tensor(axes[0].lines[0].get_xdata()),
                                               torch.tensor(frame["simulator_transitions"].to_numpy() / 1e6))
                    self.assertEqual(axes[1].lines[0].get_label(), "IH-MFC-AC")
                    plots.plt.close(figure)
        self.assertEqual(simulator_budget_estimate(self.algorithm()), 5)

    def test_fair_launcher_budget_and_removal_of_qlearning(self):
        import run as plan
        from train import parse_args
        for name in ("twostate", "advertising", "cybersecurity", "distribution", "lq"):
            job = next(j for j in plan.experiment_plan(name) if j["algorithm"] == "ihmfcac")
            parameters = plan.fair_run_parameters(job)
            allocation = plan.TRANSPORT_ALLOCATIONS[name]
            self.assertEqual(parameters["max_transitions"],
                             allocation["updates"] * allocation["horizon"] * plan.transport_per_step_budget(name))
            self.assertNotIn("lr", parameters)
            with patch.object(sys, "argv", ["run.py", "--env", name, "--seeds", "0", "--dry-run"]):
                args = plan.parse_args()
            command = plan.command_for(job, 0, args)
            with patch.object(sys, "argv", ["train.py"] + command[2:]):
                parsed = parse_args()
            self.assertEqual(parsed.max_transitions, parameters["max_transitions"])
            self.assertEqual(parsed.flow, "particle")
        self.assertFalse((Path(__file__).resolve().parents[1] / "src/mfc/algorithms/mfqlearning.py").exists())


if __name__ == "__main__":
    unittest.main()
