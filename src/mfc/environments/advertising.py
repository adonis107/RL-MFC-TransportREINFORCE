from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class AdvertisingConfig:
    eta: float = 0.2
    ad_cost: float = 0.15
    gamma: float = 0.5
    hidden_width: int = 32
    T: int = 5
    T_val: int = 5
    validation_grid_size: int = 19
    n_train: int = 10_000
    lr: float = 1e-3
    n_particles: int = 200
    n_logit_gradient: int = 10
    validation_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AdvertisingPolicy(nn.Module):
    def __init__(self, config=AdvertisingConfig()):
        super().__init__()
        self.device = config.device
        self.n_states = 2
        self.n_actions = 2
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


class Advertising:
    def __init__(self, config=AdvertisingConfig()):
        self.NOT_CUSTOMER, self.CUSTOMER = 0, 1
        self.NO_AD, self.AD = 0, 1
        self.n_states, self.n_actions = 2, 2

        self.config = config
        self.dtype = torch.float32
        self.device = config.device

        self.initial_distribution = torch.tensor([0.8, 0.2], dtype=self.dtype, device=self.device)

    def sample_initial_distribution(self, generator):
        p = 0.05 + 0.9 * torch.rand((), dtype=self.dtype, device=self.device, generator=generator)
        return torch.stack([1.0 - p, p])

    def validation_initial_distributions(self):
        customer = torch.linspace(
            0.05,
            0.95,
            self.config.validation_grid_size,
            dtype=self.dtype,
            device=self.device,
        )
        return torch.stack([1.0 - customer, customer], dim=-1)

    def transition(self, states, mu, actions):
        prob_customer = (mu[..., self.CUSTOMER] + self.config.eta * actions).clamp(max=1.0)
        return torch.stack([1.0 - prob_customer, prob_customer], dim=-1)

    def sample(self, states, mu, actions, generator):
        probabilities = self.transition(states, mu, actions)
        flat = torch.multinomial(probabilities.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(states.shape)

    def reward(self, states, mu, actions):
        return (states == self.CUSTOMER).to(self.dtype) - self.config.ad_cost * actions.to(self.dtype)

    def terminal_reward(self, states, mu):
        return torch.zeros_like(states, dtype=self.dtype)

    def policy(self, theta, t, state, mu):
        probabilities = theta(t, mu) if callable(theta) else theta
        if state.ndim == 0:
            return probabilities[..., state, :]
        if probabilities.ndim == 2:
            return probabilities[state]

        index = state.unsqueeze(-1).unsqueeze(-1).expand(*state.shape, 1, self.n_actions)
        return torch.gather(probabilities, dim=-2, index=index).squeeze(-2)

    def optimal_ad_probability(self, mu):
        p = mu[..., self.CUSTOMER]
        beta = self.config.gamma
        eta = self.config.eta
        ratio = self.config.ad_cost / eta
        threshold = 1.0 - self.config.ad_cost * (1.0 - beta) / beta

        if ratio < beta:
            return (p < threshold).to(self.dtype)

        if ratio < beta / (1.0 - beta):
            q = torch.zeros_like(p)
            q = torch.where(p < 1.0 - 2.0 * eta, torch.ones_like(q), q)
            middle = (1.0 - eta - p) / eta
            q = torch.where((1.0 - 2.0 * eta <= p) & (p < 1.0 - (2.0 - beta) * eta), middle, q)
            q = torch.where((1.0 - (2.0 - beta) * eta <= p) & (p < threshold), torch.ones_like(q), q)
            return q.clamp(0.0, 1.0)

        return torch.zeros_like(p)

    def optimal_policy(self, mu=None):
        if mu is None:
            return lambda t, law: self.optimal_policy(law)

        q = self.optimal_ad_probability(mu)
        pi = torch.empty(self.n_states, self.n_actions, dtype=self.dtype, device=self.device)
        pi[:, self.NO_AD] = 1.0 - q
        pi[:, self.AD] = q
        return pi

    def optimal_theta(self, mu=None):
        mu = self.initial_distribution if mu is None else mu
        pi = self.optimal_policy(mu)
        pi = pi.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(pi[:, self.AD] / pi[:, self.NO_AD])
