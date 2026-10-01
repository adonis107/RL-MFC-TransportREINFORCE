from dataclasses import dataclass
import importlib

import torch
from torch import nn

from .discrete_validation import mean_field_next_law
from .sampling import sample_accepts_time
from .timing import report_progress, synchronized_time
from .validation import monte_carlo_objective


@dataclass(frozen=True)
class ReinforceConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    baseline: bool = True
    seed: int = 0


class Reinforce:
    def __init__(self, env, policy=None, config=ReinforceConfig()):
        self.env = env
        self.config = config
        self.policy = self._make_policy() if policy is None else policy

    def trainable_parameters(self):
        if isinstance(self.policy, nn.Module):
            return self.policy.parameters()
        return [self.policy]

    def optimizer(self):
        return torch.optim.Adam(self.trainable_parameters(), lr=self.lr)

    @property
    def n_train(self):
        return self.env.config.n_train if self.config.n_train is None else self.config.n_train

    @property
    def lr(self):
        return self.env.config.lr if self.config.lr is None else self.config.lr

    @property
    def n_particles(self):
        return self.env.config.n_particles if self.config.n_particles is None else self.config.n_particles

    @property
    def horizon(self):
        return self.env.config.T if self.config.horizon is None else self.config.horizon

    @property
    def validation_interval(self):
        return (
            self.env.config.validation_interval
            if self.config.validation_interval is None
            else self.config.validation_interval
        )

    @property
    def discount(self):
        if hasattr(self.env.config, "discount"):
            return self.env.config.discount
        return getattr(self.env.config, "gamma", 1.0)

    def _make_policy(self):
        if hasattr(self.env, "zero_policy"):
            return nn.Parameter(self.env.zero_policy())

        module = importlib.import_module(type(self.env).__module__)
        policy_class = getattr(module, f"{type(self.env).__name__}Policy")
        return policy_class(self.env.config)

    def sample_initial_distribution(self, generator):
        if hasattr(self.env, "sample_initial_distribution"):
            return self.env.sample_initial_distribution(generator)
        return self.env.initial_distribution


    def initial_states(self, n_particles, generator, initial_distribution=None):
        if hasattr(self.env, "sample_initial"):
            return self.env.sample_initial(n_particles, generator)

        probabilities = self.sample_initial_distribution(generator) if initial_distribution is None else initial_distribution
        return torch.multinomial(probabilities, n_particles, replacement=True, generator=generator)

    def empirical_law(self, states):
        if hasattr(self.env, "empirical_law"):
            return self.env.empirical_law(states)
        if states.dtype.is_floating_point:
            return states.mean().detach()

        counts = torch.bincount(states, minlength=self.env.n_states)
        return (counts.to(self.env.dtype) / states.numel()).detach()

    def policy_time(self, t):
        if isinstance(self.policy, nn.Module):
            return torch.tensor(float(t), dtype=self.env.dtype, device=self.env.device)
        return t

    def sample_actions(self, t, states, law, generator):
        t_policy = self.policy_time(t)
        action_law = self.env.policy(self.policy, t_policy, states, law)

        if hasattr(action_law, "log_prob"):
            with torch.no_grad():
                if hasattr(self.env, "sample_action"):
                    actions = self.env.sample_action(self.policy, t, states, law, generator)
                else:
                    actions = action_law.sample()
            return actions, action_law.log_prob(actions.detach())

        probabilities = action_law.clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        flat_actions = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator
        )
        actions = flat_actions.reshape(states.shape)
        log_probs = torch.log(probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1))
        return actions, log_probs

    def sample_next_states(self, t, states, law, actions, generator):
        if sample_accepts_time(self.env.sample):
            return self.env.sample(states, law, actions, generator, t=t)
        return self.env.sample(states, law, actions, generator)

    def rollout(self, n_particles=None, horizon=None, seed=None):
        n_particles = self.n_particles if n_particles is None else n_particles
        horizon = self.horizon if horizon is None else horizon
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed if seed is None else seed)

        initial_distribution = None
        if not hasattr(self.env, "sample_initial"):
            initial_distribution = self.sample_initial_distribution(generator)
        states = self.initial_states(n_particles, generator, initial_distribution=initial_distribution)
        rewards = []
        log_probs = []

        for t in range(horizon):
            law = self.empirical_law(states)
            actions, action_log_probs = self.sample_actions(t, states, law, generator)
            rewards.append(self.env.reward(states, law, actions))
            log_probs.append(action_log_probs)

            with torch.no_grad():
                states = self.sample_next_states(t, states, law, actions.detach(), generator)

        terminal_law = self.empirical_law(states)
        terminal_rewards = self.env.terminal_reward(states, terminal_law)
        returns = self.discounted_returns(rewards, terminal_rewards)
        score = torch.stack(log_probs, dim=0).sum(dim=0)

        return {
            "returns": returns,
            "score": score,
            "objective": returns.mean(),
            "terminal_states": states,
            "terminal_law": terminal_law,
        }

    def discounted_returns(self, rewards, terminal_rewards):
        returns = torch.zeros_like(terminal_rewards)
        discount = 1.0
        for reward in rewards:
            returns = returns + discount * reward
            discount = discount * self.discount
        return returns + discount * terminal_rewards

    def mean_field_next_law(self, t, law):
        return mean_field_next_law(self.env, self.policy, self.policy_time, t, law)

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        """Monte Carlo objective of the frozen policy with M_val interacting particles."""
        horizon = getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon
        n_particles = self.env.config.validation_particles if n_particles is None else n_particles
        seed = self.config.seed if seed is None else seed
        return monte_carlo_objective(
            self.env, self.policy, self.policy_time, self.discount, n_particles, horizon, seed
        )

    def loss(self, rollout):
        advantages = rollout["returns"].detach()
        if self.config.baseline:
            advantages = advantages - advantages.mean()
        return -(rollout["score"] * advantages).mean()

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        optimizer = self.optimizer()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at
        history = {
            "objective": [],
            "loss": [],
            "validation_objective": [],
            "train_step_seconds": [],
            "validation_seconds": [],
            "setup_seconds": [setup_seconds],
        }

        for episode in range(self.n_train):
            step_started_at = synchronized_time(self.env.device)
            rollout = self.rollout(seed=self.config.seed + episode)
            loss = self.loss(rollout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hasattr(self.env, "project_policy"):
                self.env.project_policy(self.policy)
            objective_value = float(rollout["objective"].detach().cpu())
            loss_value = float(loss.detach().cpu())
            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)

            history["objective"].append(objective_value)
            history["loss"].append(loss_value)
            report_progress(episode, self.n_train, history)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                validation_value = float(validation.detach().cpu())
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(validation_value)

        return self.policy, history


def train_reinforce(env, policy=None, config=ReinforceConfig()):
    return Reinforce(env, policy=policy, config=config).train()
