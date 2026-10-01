from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TwoStateConfig:
    lambda0: float = 0.5
    lambda1: float = 0.8
    kappa: float = 10.0
    p: float = 0.6
    T: int = 5
    gamma: float = 1.0 # No discount
    n_train: int = 10_000
    lr: float = 1e-3
    n_particles: int = 200
    n_logit_gradient: int = 10
    validation_interval: int = 10
    validation_particles: int = 100_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TwoState:
    def __init__(self, config=TwoStateConfig()):
        self.ST, self.MV = 0, 1
        self.n_states, self.n_actions = 2, 2

        self.config = config
        self.dtype = torch.float32
        self.device = config.device

        self.target_law = torch.tensor([config.p, 1.0 - config.p], dtype=self.dtype, device=self.device)
        self.initial_distribution = torch.tensor([0.2, 0.8], dtype=self.dtype, device=self.device)
        self._switch_prob = torch.tensor([config.lambda0, config.lambda1], dtype=self.dtype, device=self.device)

    def sample_initial_distribution(self, generator):
        mu1 = 0.1 + 0.8 * torch.rand((), dtype=self.dtype, device=self.device, generator=generator)
        return torch.stack([1.0 - mu1, mu1])

    def transition(self, states, mu, actions):
        switch_prob = self._switch_prob[states]
        switch = torch.where(actions == self.MV, switch_prob, torch.zeros_like(switch_prob))
        prob_x0 = torch.where(states == 0, 1.0 - switch, switch)
        return torch.stack([prob_x0, 1 - prob_x0], dim=-1)

    def sample(self, states, mu, actions, generator):
        probabilities = self.transition(states, mu, actions)
        flat = torch.multinomial(probabilities.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(states.shape)

    def reward(self, states, mu, actions):
        return (states == 1).to(self.dtype) - mu[..., 1].square() - self.config.kappa * (mu[..., 1] - self.target_law[1]).abs()

    def terminal_reward(self, states, mu):
        return self.reward(states, mu, None)

    def policy(self, theta, t, state, mu):
        p_move = torch.sigmoid(theta[state])
        return torch.stack([1.0 - p_move, p_move], dim=-1)

    def optimal_policy(self):
        pi = torch.zeros(self.n_states, self.n_actions, dtype=self.dtype, device=self.device)
        pi[0, self.MV] = (1.0 - self.config.p) / self.config.lambda0
        pi[0, self.ST] = 1.0 - pi[0, self.MV]
        pi[1, self.MV] = self.config.p / self.config.lambda1
        pi[1, self.ST] = 1.0 - pi[1, self.MV]
        if torch.any((pi < 0.0) | (pi > 1.0)):
            raise ValueError("The closed-form TwoState optimum is infeasible for this configuration.")
        return pi

    def zero_policy(self):
        return torch.zeros(self.n_states, dtype=self.dtype, device=self.device)

    def optimal_theta(self):
        pi = self.optimal_policy().clamp(1e-6, 1.0 - 1e-6)
        return torch.log(pi[:, self.MV] / pi[:, self.ST])
