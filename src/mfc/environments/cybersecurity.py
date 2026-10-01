from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class CybersecurityConfig:
    beta_uu: float = 0.3
    beta_ud: float = 0.4
    beta_du: float = 0.3
    beta_dd: float = 0.4
    q_rec_D: float = 0.5
    q_rec_U: float = 0.4
    q_inf_D: float = 0.4
    q_inf_U: float = 0.3
    v_H: float = 0.6
    lambda_sw: float = 0.8
    k_D: float = 0.3
    k_I: float = 0.5
    dt: float = 0.2
    gamma: float = 0.5
    hidden_width: int = 32
    T: int = 3
    T_val: int = 50
    n_train: int = 20_000
    lr: float = 1e-3
    n_particles: int = 200
    n_logit_gradient: int = 1
    validation_interval: int = 10
    validation_particles: int = 100_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def discount(self):
        # gamma is specified per unit time; the sampled chain uses gamma ** dt per step.
        return self.gamma**self.dt


class CybersecurityPolicy(nn.Module):
    def __init__(self, config=CybersecurityConfig()):
        super().__init__()
        self.device = config.device
        self.n_states = 4
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
        t = t / (1.0 + t)
        t = t.unsqueeze(-1)
        logits = self.net(torch.cat([t, mu], dim=-1))
        logits = logits.reshape(*mu.shape[:-1], self.n_states, self.n_actions)
        return torch.softmax(logits, dim=-1)

class Cybersecurity:
    def __init__(self, config=CybersecurityConfig()):
        self.DI, self.DS, self.UI, self.US = 0, 1, 2, 3
        self.KEEP, self.SWITCH = 0, 1
        self.n_states, self.n_actions = 4, 2

        self.config = config
        self.dtype = torch.float32
        self.device = config.device

        self.initial_distribution = torch.full((4,), 0.25, dtype=self.dtype, device=self.device)
        self.f = torch.tensor([config.k_D + config.k_I, config.k_D, config.k_I, 0.0], dtype=self.dtype, device=self.device)

    def sample_initial_distribution(self, generator):
        weights = -torch.rand(self.n_states, dtype=self.dtype, device=self.device, generator=generator).clamp_min(1e-12).log()
        return weights / weights.sum()

    def _generator_matrix(self, mu, actions):
        sw = self.config.lambda_sw * actions.to(self.dtype)
        iota_D = self.config.v_H * self.config.q_inf_D + self.config.beta_dd * mu[..., self.DI] + self.config.beta_ud * mu[..., self.UI]
        iota_U = self.config.v_H * self.config.q_inf_U + self.config.beta_uu * mu[..., self.UI] + self.config.beta_du * mu[..., self.DI]
        zeros = torch.zeros_like(sw)
        iota_D = iota_D + zeros
        iota_U = iota_U + zeros
        q_rec_D = zeros + self.config.q_rec_D
        q_rec_U = zeros + self.config.q_rec_U

        row_DI = torch.stack([-q_rec_D - sw, q_rec_D, sw, zeros], dim=-1)
        row_DS = torch.stack([iota_D, -iota_D - sw, zeros, sw], dim=-1)
        row_UI = torch.stack([sw, zeros, -q_rec_U - sw, q_rec_U], dim=-1)
        row_US = torch.stack([zeros, sw, iota_U, -iota_U - sw], dim=-1)
        return torch.stack([row_DI, row_DS, row_UI, row_US], dim=-2)

    def transition(self, states, mu, actions):
        actions = torch.broadcast_to(actions, states.shape)
        if mu.ndim == 1 and states.ndim > 0:
            # One population argument shared by every particle: one kernel per action covers them all.
            all_actions = torch.arange(self.n_actions, device=states.device)
            kernels = torch.matrix_exp(self.config.dt * self._generator_matrix(mu, all_actions))
            return kernels[actions, states]

        probabilities = torch.matrix_exp(self.config.dt * self._generator_matrix(mu, actions))
        if states.ndim == 0:
            return probabilities[states]

        row_index = states.unsqueeze(-1).unsqueeze(-1).expand(*states.shape, 1, self.n_states)
        return torch.gather(probabilities, dim=-2, index=row_index).squeeze(-2)

    def sample(self, states, mu, actions, generator):
        probabilities = self.transition(states, mu, actions)
        flat = torch.multinomial(probabilities.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(states.shape)

    def reward(self, states, mu, actions):
        return -self.config.dt * self.f[states]

    def terminal_reward(self, states, mu):
        return self.reward(states, mu, None)

    def policy(self, theta, t, state, mu):
        probabilities = theta(t, mu) if callable(theta) else theta
        if state.ndim == 0:
            return probabilities[..., state, :]
        if probabilities.ndim == 2:
            return probabilities[state]

        index = state.unsqueeze(-1).unsqueeze(-1).expand(*state.shape, 1, self.n_actions)
        return torch.gather(probabilities, dim=-2, index=index).squeeze(-2)

    def optimal_policy(self):
        raise NotImplementedError("No closed-form optimal policy is specified for the cybersecurity environment.")

    def optimal_theta(self):
        raise NotImplementedError("No closed-form optimal policy is specified for the cybersecurity environment.")
