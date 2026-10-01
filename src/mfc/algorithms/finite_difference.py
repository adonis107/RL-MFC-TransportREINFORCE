from dataclasses import dataclass

import torch

from .reinforce import Reinforce
from .sampling import sample_accepts_time
from .timing import report_progress, synchronized_time


@dataclass(frozen=True)
class FiniteDifferenceConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    step: float = 0.1
    common_random_numbers: bool = True
    seed: int = 0


class FiniteDifference(Reinforce):
    """Centered finite differences of the particle-system objective, one coordinate at a time.

    Every update simulates, for each parameter coordinate l, an interacting
    particle system at theta + h e_l and one at theta - h e_l. Each system reads
    the empirical law of its own particles, so the estimate targets the
    objective itself, with no population representation. These are the shifted
    systems of the transport auxiliary stage, used directly for the gradient.
    n_particles is the per-time budget M + n + B, shared across the 2 d_theta
    systems. With common random numbers, the two systems of a coordinate are
    driven by the same draws.
    """

    def __init__(self, env, policy=None, config=FiniteDifferenceConfig()):
        if not getattr(env, "per_particle_parameters", False):
            raise TypeError("FiniteDifference simulates its systems side by side and needs per_particle_parameters.")
        super().__init__(env, policy=policy, config=config)

    @property
    def n_parameters(self):
        return self.policy.numel()

    @property
    def particles_per_system(self):
        return max(1, self.n_particles // (2 * self.n_parameters))

    def system_laws(self, states, systems):
        """Population argument of every system, from its own particles."""
        grouped = states.reshape(systems, -1)
        if hasattr(self.env, "empirical_law"):
            laws = self.env.empirical_law(grouped)
        else:
            laws = grouped.mean(dim=-1)
        return laws.repeat_interleave(grouped.shape[1], dim=0)

    def system_objectives(self, thetas, seed):
        """Particle estimate of J at every row of thetas, one interacting system per row."""
        systems, n0 = thetas.shape[0], self.particles_per_system
        per_particle = thetas.repeat_interleave(n0, dim=0).T.reshape(*self.policy.shape, systems * n0)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        states = self.env.sample_initial(systems * n0, generator)
        value = torch.zeros_like(states)
        factor = 1.0
        for t in range(self.horizon):
            law = self.system_laws(states, systems)
            actions = self.env.sample_action(per_particle, t, states, law, generator)
            value = value + factor * self.env.reward(states, law, actions)
            if sample_accepts_time(self.env.sample):
                states = self.env.sample(states, law, actions, generator, t=t)
            else:
                states = self.env.sample(states, law, actions, generator)
            factor = factor * self.discount
        value = value + factor * self.env.terminal_reward(states, self.system_laws(states, systems))
        return value.reshape(systems, n0).mean(dim=1)

    @torch.no_grad()
    def estimate_gradient(self, seed):
        base = self.policy.detach().reshape(1, -1)
        offsets = self.config.step * torch.eye(self.n_parameters, dtype=base.dtype, device=base.device)
        if self.config.common_random_numbers:
            plus = self.system_objectives(base + offsets, seed)
            minus = self.system_objectives(base - offsets, seed)
        else:
            both = self.system_objectives(torch.cat([base + offsets, base - offsets]), seed)
            plus, minus = both[: self.n_parameters], both[self.n_parameters :]
        gradient = (plus - minus) / (2.0 * self.config.step)
        return gradient.reshape(self.policy.shape), 0.5 * (plus + minus).mean()

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        optimizer = self.optimizer()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at
        history = {
            "objective": [],
            "validation_objective": [],
            "gradient_norm": [],
            "train_step_seconds": [],
            "validation_seconds": [],
            "setup_seconds": [setup_seconds],
        }

        for episode in range(self.n_train):
            step_started_at = synchronized_time(self.env.device)
            gradient, objective = self.estimate_gradient(self.config.seed + episode)
            optimizer.zero_grad()
            self.policy.grad = -gradient
            optimizer.step()
            if hasattr(self.env, "project_policy"):
                self.env.project_policy(self.policy)
            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)
            history["objective"].append(float(objective))
            history["gradient_norm"].append(float(gradient.norm()))
            report_progress(episode, self.n_train, history)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(float(validation))

        return self.policy, history


def train_finite_difference(env, policy=None, config=FiniteDifferenceConfig()):
    return FiniteDifference(env, policy=policy, config=config).train()
