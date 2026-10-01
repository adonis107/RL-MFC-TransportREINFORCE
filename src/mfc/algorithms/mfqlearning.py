from dataclasses import dataclass
from itertools import product

import torch
from torch import nn

from .timing import synchronized_time
from .validation import monte_carlo_objective


def integer_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for tail in integer_compositions(total - value, parts - 1):
            yield (value, *tail)


def simplex_grid(n_states, resolution, *, dtype, device):
    points = list(integer_compositions(resolution, n_states))
    return torch.tensor(points, dtype=dtype, device=device) / float(resolution)


def pure_feedback_actions(n_states, n_actions, *, device):
    actions = list(product(range(n_actions), repeat=n_states))
    return torch.tensor(actions, dtype=torch.long, device=device)


@dataclass(frozen=True)
class MeanFieldQLearningConfig:
    n_train: int | None = None
    lr: float | None = None
    learning_rate_power: float = 0.6
    n_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    seed: int = 0
    simplex_resolution: int = 30
    initial_q: float = 0.0
    sampling: str = "sweep"


class MeanFieldQPolicy(nn.Module):
    def __init__(self, grid, action_table, action_indices):
        super().__init__()
        self.register_buffer("grid", grid.detach().clone())
        self.register_buffer("action_table", action_table.detach().clone())
        self.register_buffer("action_indices", action_indices.detach().clone())

    @property
    def n_states(self):
        return int(self.action_table.shape[1])

    @property
    def n_actions(self):
        return int(self.action_table.max().item()) + 1

    def nearest_grid_indices(self, law):
        original_shape = law.shape[:-1]
        flat = law.reshape(-1, law.shape[-1])
        distances = (flat[:, None, :] - self.grid[None, :, :]).square().sum(dim=-1)
        return distances.argmin(dim=-1).reshape(original_shape)

    def forward(self, t, mu):
        grid_indices = self.nearest_grid_indices(mu)
        action_indices = self.action_indices[grid_indices]
        actions = self.action_table[action_indices]
        probabilities = torch.nn.functional.one_hot(actions, num_classes=self.n_actions).to(self.grid.dtype)
        return probabilities

    def export_payload(self):
        return {
            "kind": "mean_field_q_policy",
            "grid": self.grid.detach().cpu(),
            "action_table": self.action_table.detach().cpu(),
            "action_indices": self.action_indices.detach().cpu(),
        }

    @classmethod
    def from_payload(cls, payload, *, device, dtype):
        return cls(
            payload["grid"].to(device=device, dtype=dtype),
            payload["action_table"].to(device=device),
            payload["action_indices"].to(device=device),
        )


class MeanFieldQLearning:
    def __init__(self, env, config=MeanFieldQLearningConfig()):
        if not hasattr(env, "n_states") or not hasattr(env, "n_actions"):
            raise ValueError("MeanFieldQLearning requires a finite-state, finite-action environment.")
        self.env = env
        self.config = config
        self.grid = simplex_grid(
            env.n_states,
            self.simplex_resolution,
            dtype=env.dtype,
            device=env.device,
        )
        self.action_table = pure_feedback_actions(env.n_states, env.n_actions, device=env.device)
        self.q = torch.full(
            (self.grid.shape[0], self.action_table.shape[0]),
            float(config.initial_q),
            dtype=env.dtype,
            device=env.device,
        )
        self.visits = torch.zeros_like(self.q, dtype=torch.long)
        self.next_indices = None
        self.rewards = None

    @property
    def n_train(self):
        if self.config.n_train is not None:
            return self.config.n_train
        return 200_000

    @property
    def lr(self):
        return 1.0 if self.config.lr is None else self.config.lr

    @property
    def n_particles(self):
        return 1 if self.config.n_particles is None else self.config.n_particles

    @property
    def horizon(self):
        return self.env.config.T if self.config.horizon is None else self.config.horizon

    @property
    def validation_interval(self):
        return 10_000 if self.config.validation_interval is None else self.config.validation_interval

    @property
    def simplex_resolution(self):
        return self.config.simplex_resolution

    @property
    def discount(self):
        if hasattr(self.env.config, "discount"):
            return self.env.config.discount
        return getattr(self.env.config, "gamma", 1.0)

    def policy_time(self, t):
        return torch.tensor(float(t), dtype=self.env.dtype, device=self.env.device)

    def initial_policy(self):
        action_indices = self.q.argmax(dim=1)
        return MeanFieldQPolicy(self.grid, self.action_table, action_indices)

    def project(self, law):
        distances = (self.grid - law).square().sum(dim=-1)
        return distances.argmin()

    def lifted_step(self, law, feedback_action):
        next_law = torch.zeros_like(law)
        reward = torch.zeros((), dtype=self.env.dtype, device=self.env.device)
        with torch.no_grad():
            for state_index in range(self.env.n_states):
                state = torch.tensor(state_index, dtype=torch.long, device=self.env.device)
                action = feedback_action[state_index]
                transition = self.env.transition(state, law, action)
                next_law = next_law + law[state_index] * transition
                reward = reward + law[state_index] * self.env.reward(state, law, action)
        return next_law / next_law.sum(), reward

    def precompute_lifted_transitions(self, chunk=4096):
        """Lifted step of every (grid law, feedback action) pair at once.

        The same quantities as lifted_step and project, for all pairs together:
        the kernel and reward of every state under its feedback action, averaged
        under the grid law, and the next law projected on the grid in chunks.
        """
        n_grid, n_feedback = self.q.shape
        n_states = self.env.n_states
        shape = (n_grid, n_feedback, n_states)
        states = torch.arange(n_states, device=self.env.device).expand(shape)
        actions = self.action_table.unsqueeze(0).expand(shape)
        population = self.grid[:, None, None, :]
        weights = self.grid.unsqueeze(1)
        with torch.no_grad():
            kernels = self.env.transition(states, population, actions)
            rewards = (weights * self.env.reward(states, population, actions)).sum(dim=-1)
            next_laws = (weights.unsqueeze(-1) * kernels).sum(dim=-2)
            next_laws = (next_laws / next_laws.sum(dim=-1, keepdim=True)).reshape(-1, n_states)
            next_indices = torch.cat([
                (block.unsqueeze(1) - self.grid.unsqueeze(0)).square().sum(dim=-1).argmin(dim=-1)
                for block in next_laws.split(chunk)
            ])
        return next_indices.reshape(n_grid, n_feedback), rewards.to(self.q.dtype)


    def evaluate(self, policy=None, horizon=None, seed=None):
        policy = self.initial_policy() if policy is None else policy
        horizon = getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon
        return monte_carlo_objective(
            self.env,
            policy,
            self.policy_time,
            self.discount,
            self.env.config.validation_particles,
            horizon,
            self.config.seed + self.n_train if seed is None else seed,
        )

    def sampled_pairs(self, generator):
        n_states, n_actions = self.q.shape
        if self.config.sampling == "iid":
            while True:
                state = torch.randint(n_states, (), generator=generator, device=self.env.device)
                action = torch.randint(n_actions, (), generator=generator, device=self.env.device)
                yield int(state.item()), int(action.item())
            return
        if self.config.sampling != "sweep":
            raise ValueError(f"Unknown MFQ sampling mode: {self.config.sampling}")

        n_pairs = n_states * n_actions
        while True:
            order = torch.randperm(n_pairs, generator=generator, device=self.env.device)
            for flat_index in order:
                index = int(flat_index.item())
                yield divmod(index, n_actions)

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed)
        if self.next_indices is None or self.rewards is None:
            self.next_indices, self.rewards = self.precompute_lifted_transitions()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at

        history = {
            "objective": [],
            "loss": [],
            "validation_objective": [],
            "train_step_seconds": [],
            "validation_seconds": [],
            "setup_seconds": [setup_seconds],
        }

        pair_source = self.sampled_pairs(generator)
        block_started_at = synchronized_time(self.env.device)
        train_seconds = 0.0
        last_target = None
        last_loss = None
        for update in range(self.n_train):
            state_index, action_index = next(pair_source)
            self.visits[state_index, action_index] += 1
            visits = float(self.visits[state_index, action_index].item())
            alpha = min(1.0, self.lr / (visits**self.config.learning_rate_power))

            next_index = self.next_indices[state_index, action_index]
            target = self.rewards[state_index, action_index] + self.discount * self.q[next_index].max()
            old_value = self.q[state_index, action_index]
            td_error = target - old_value
            self.q[state_index, action_index] = old_value + alpha * td_error

            last_target = float(target.detach().cpu())
            last_loss = float(td_error.square().detach().cpu())

            if self.validation_interval and (update + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                train_seconds += validation_started_at - block_started_at
                with torch.no_grad():
                    validation = self.evaluate()
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(float(validation.detach().cpu()))
                history["objective"].append(last_target)
                history["loss"].append(last_loss)
                block_started_at = synchronized_time(self.env.device)

        finished_at = synchronized_time(self.env.device)
        train_seconds += finished_at - block_started_at
        history["train_step_seconds"].append(train_seconds)
        if not history["objective"] and last_target is not None:
            history["objective"].append(last_target)
            history["loss"].append(last_loss)

        return self.initial_policy(), history


def train_mean_field_q_learning(env, config=MeanFieldQLearningConfig()):
    return MeanFieldQLearning(env, config=config).train()
