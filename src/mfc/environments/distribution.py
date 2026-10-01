from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class DistributionConfig:
    c_mov: float = 0.01
    hidden_width: int = 64
    T: int = 5
    T_val: int = 5
    gamma: float = 1.0
    n_train: int = 30_000
    lr: float = 1e-4
    n_particles: int = 500
    n_logit_gradient: int = 64
    validation_interval: int = 10
    validation_particles: int = 100_000
    target_distribution: tuple[float, ...] = (0.02, 0.04, 0.09, 0.16, 0.19, 0.19, 0.16, 0.09, 0.04, 0.02)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DistributionPolicy(nn.Module):
    def __init__(self, config=DistributionConfig()):
        super().__init__()
        self.device = config.device
        self.n_states = 10
        self.n_actions = 3
        self.net = nn.Sequential(
            nn.Linear(1 + self.n_states, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, self.n_states * self.n_actions),
        )
        self.to(self.device)

    def forward(self, t, mu):
        if t.ndim == 0:
            t = t.expand(mu.shape[:-1])
        t = t.unsqueeze(-1)
        logits = self.net(torch.cat([t, mu], dim=-1))
        logits = logits.reshape(*mu.shape[:-1], self.n_states, self.n_actions)
        return torch.softmax(logits, dim=-1)

class Distribution:
    def __init__(self, config=DistributionConfig()):
        self.LEFT, self.STAY, self.RIGHT = 0, 1, 2
        self.n_states, self.n_actions = 10, 3

        self.config = config
        self.dtype = torch.float32
        self.device = config.device

        self.initial_distribution = torch.full((10,), 0.1, dtype=self.dtype, device=self.device)
        self.target_distribution = torch.tensor(config.target_distribution, dtype=self.dtype, device=self.device)
        self.action_values = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)

    def sample_initial_distribution(self, generator):
        weights = -torch.rand(self.n_states, dtype=self.dtype, device=self.device, generator=generator).clamp_min(1e-12).log()
        return weights / weights.sum()

    def transition(self, states, mu, actions):
        moves = self.action_values[actions]
        next_states = (states + moves) % self.n_states
        return torch.nn.functional.one_hot(next_states, num_classes=self.n_states).to(self.dtype)

    def sample(self, states, mu, actions, generator):
        return (states + self.action_values[actions]) % self.n_states

    def reward(self, states, mu, actions):
        movement_cost = self.config.c_mov * self.action_values[actions].abs().to(self.dtype)
        distribution_cost = (mu - self.target_distribution).square().sum(dim=-1)
        return -movement_cost - distribution_cost

    def terminal_reward(self, states, mu):
        distribution_cost = (mu - self.target_distribution).square().sum(dim=-1)
        return torch.zeros_like(states, dtype=self.dtype) - distribution_cost

    def policy(self, theta, t, state, mu):
        probabilities = theta(t, mu) if callable(theta) else theta
        if state.ndim == 0:
            return probabilities[..., state, :]
        if probabilities.ndim == 2:
            return probabilities[state]

        index = state.unsqueeze(-1).unsqueeze(-1).expand(*state.shape, 1, self.n_actions)
        return torch.gather(probabilities, dim=-2, index=index).squeeze(-2)

    def population_step(self, mu, probabilities):
        """One step of the exact population recursion under a per-site action law."""
        flow = mu.unsqueeze(-1) * probabilities
        moved = (torch.arange(self.n_states, device=self.device).unsqueeze(-1) + self.action_values) % self.n_states
        return torch.zeros_like(mu).index_add(-1, moved.reshape(-1), flow.reshape(*mu.shape[:-1], -1))

    def population_objective(self, probabilities, initial_distribution=None):
        """Exact objective of a time-indexed action law, by the deterministic recursion.

        ``probabilities`` has shape (T, n_states, n_actions). Matches the reward
        functions above: a movement cost weighted by the mass that moves, plus the
        squared distance of the population to its target at every time.
        """
        mu = self.initial_distribution if initial_distribution is None else initial_distribution
        absolute_move = self.action_values.abs().to(self.dtype)
        total = torch.zeros((), dtype=self.dtype, device=self.device)
        for t in range(self.config.T):
            step = probabilities[t]
            total = total - (mu - self.target_distribution).square().sum()
            total = total - self.config.c_mov * (mu * (step * absolute_move).sum(-1)).sum()
            mu = self.population_step(mu, step)
        return total - (mu - self.target_distribution).square().sum()

    def optimal_theta(self, initial_distribution=None, steps=4000, lr=0.05):
        """Optimal action law from a given initial distribution.

        There is no closed form, but there does not need to be one: the transition
        is deterministic and every term of the objective is differentiable, so the
        optimal open-loop control is optimal outright and is recovered by gradient
        ascent on the action logits. Returns probabilities of shape
        (T, n_states, n_actions). Evaluation only; never used during training.
        """
        law = self.initial_distribution if initial_distribution is None else initial_distribution
        key = (int(self.config.T), float(self.config.c_mov), tuple(law.tolist()), steps)
        cached = getattr(self, "_optimal_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        logits = torch.zeros(
            self.config.T, self.n_states, self.n_actions,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        optimizer = torch.optim.Adam([logits], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            (-self.population_objective(torch.softmax(logits, dim=-1), law)).backward()
            optimizer.step()

        theta = torch.softmax(logits.detach(), dim=-1)
        self._optimal_cache = (key, theta)
        return theta

    def optimal_policy(self, initial_distribution=None):
        """The optimal control as a callable ``(t, mu) -> action probabilities``."""
        theta = self.optimal_theta(initial_distribution)

        def policy(t, mu):
            index = int(t.item()) if torch.is_tensor(t) else int(t)
            step = theta[min(index, theta.shape[0] - 1)]
            return step.expand(*mu.shape[:-1], *step.shape) if mu.ndim > 1 else step

        return policy

    def optimal_objective(self, initial_distribution=None):
        """Best achievable objective, the reference point for an optimality gap."""
        return self.population_objective(self.optimal_theta(initial_distribution), initial_distribution)
