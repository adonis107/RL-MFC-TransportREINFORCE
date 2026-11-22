from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import nn


@dataclass(frozen=True)
class TrafficConfig:
    cells: int = 3
    p_fail: float = 0.0
    p_slow: float = 0.0
    p_fast: float = 0.0
    safety_cost: float = 2.0
    comfort_cost: float = 0.05
    forward_scale: float = 0.25
    lateral_scale: float = 1.0
    congestion_reward: str = "shared"
    precision: str = "float64"
    gamma: float = 0.9
    training_initial_law: str = "mixed"
    training_jam_probability: float = 0.5
    hidden_width: int = 64
    T: int = 5
    T_val: int = 5
    n_train: int = 10_000
    lr: float = 1e-3
    n_particles: int = 200
    n_logit_gradient: int = 10
    validation_interval: int = 10
    validation_particles: int = 100_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.congestion_reward not in {"local", "shared"}:
            raise ValueError("congestion_reward must be local or shared")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if self.training_initial_law not in {"dirichlet", "jam", "mixed"}:
            raise ValueError("training_initial_law must be dirichlet, jam or mixed")
        if not math.isfinite(self.training_jam_probability) or not 0 <= self.training_jam_probability <= 1:
            raise ValueError("training_jam_probability must be in [0, 1]")
        for name, minimum in (("cells", 2), ("T", 1), ("T_val", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("p_fail", "p_slow", "p_fast", "safety_cost", "comfort_cost",
                     "forward_scale", "lateral_scale", "gamma"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(not 0 <= getattr(self, name) <= 1 for name in ("p_fail", "p_slow", "p_fast")):
            raise ValueError("noise probabilities must be in [0, 1]")
        if self.p_slow + self.p_fast > 1:
            raise ValueError("p_slow + p_fast must not exceed 1")
        if min(self.safety_cost, self.comfort_cost) < 0:
            raise ValueError("penalties must be nonnegative")
        if min(self.forward_scale, self.lateral_scale) <= 0:
            raise ValueError("distance scales must be positive")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")


class TrafficPolicy(nn.Module):
    def __init__(self, config=TrafficConfig()):
        super().__init__()
        self.device = config.device
        self.n_states, self.n_actions = 3 * config.cells, 9
        self.net = nn.Sequential(
            nn.Linear(1 + self.n_states, config.hidden_width), nn.Tanh(),
            nn.Linear(config.hidden_width, config.hidden_width), nn.Tanh(),
            nn.Linear(config.hidden_width, self.n_states * self.n_actions),
        )
        self.to(device=self.device, dtype=getattr(torch, config.precision))

    def forward(self, t, mu):
        t = torch.as_tensor(t, dtype=mu.dtype, device=mu.device)
        t = torch.broadcast_to(t, mu.shape[:-1])
        logits = self.net(torch.cat([(t / (1 + t)).unsqueeze(-1), mu], dim=-1))
        return logits.reshape(*mu.shape[:-1], self.n_states, self.n_actions).softmax(-1)


class Traffic:
    def __init__(self, config=TrafficConfig()):
        self.config, self.device, self.dtype = config, config.device, getattr(torch, config.precision)
        self.n_states, self.n_actions = 3 * config.cells, 9
        # Same dense-kernel memory guard as the source benchmark.
        if 8 * self.n_states ** 2 * 20 > 64 * 1024 ** 2:
            raise MemoryError("physical transition tensors exceed 64 MiB budget")
        states = torch.arange(self.n_states, device=self.device)
        self.lanes = states // config.cells - 1
        self.positions = states % config.cells
        forward = ((self.positions[None, :] - self.positions[:, None]) % config.cells).to(self.dtype) / config.cells
        lateral = (self.lanes[None, :] - self.lanes[:, None]).to(self.dtype)
        self.risk_matrix = torch.exp(-torch.hypot(forward / config.forward_scale, lateral / config.lateral_scale))
        self.initial_distribution = torch.zeros(self.n_states, dtype=self.dtype, device=self.device)
        # Fixed jam for validation: all lanes at position zero (a source-supported jam).
        self.initial_distribution[::config.cells] = 1 / 3
        actions = torch.arange(9, device=self.device)
        self.speed = (actions // 3).to(self.dtype) / 2
        self.effort = self.speed.square() + (actions % 3 - 1).square()
        lane = states[:, None] // config.cells
        target_lane = (lane + actions[None, :] % 3 - 1).clamp(0, 2)
        self.transitions = torch.zeros(self.n_states, 9, self.n_states, dtype=self.dtype, device=self.device)
        for slip, chance in ((-1, config.p_slow), (0, 1 - (config.p_slow + config.p_fast)), (1, config.p_fast)):
            position = (self.positions[:, None] + (actions[None, :] // 3 + slip).clamp(0, 2)) % config.cells
            for next_lane, probability in ((target_lane, 1 - config.p_fail), (lane, config.p_fail)):
                targets = next_lane * config.cells + position
                self.transitions.scatter_add_(-1, targets.unsqueeze(-1), torch.full_like(targets, chance * probability, dtype=self.dtype).unsqueeze(-1))

    def sample_initial_distribution(self, generator):
        """Source-supported random or jam laws, sampled once per update.

        Mixed training samples source-style random arcs and lane masses as
        well as diffuse laws. Explicit dirichlet mode preserves historical
        draws exactly; validation remains the fixed jam at construction.
        """
        family = self.config.training_initial_law
        jam = family == "jam"
        if family == "mixed":
            jam = bool(torch.rand((), device=self.device, generator=generator) < self.config.training_jam_probability)
        if jam:
            cells = self.config.cells
            width = int(torch.randint(1, cells // 2 + 1, (), device=self.device, generator=generator))
            start = torch.randint(cells, (), device=self.device, generator=generator)
            arc = (start + torch.arange(width, device=self.device)) % cells
            support = (torch.arange(3, device=self.device)[:, None] * cells + arc).reshape(-1)
            weights = -torch.rand(support.numel(), dtype=self.dtype, device=self.device, generator=generator).clamp_min(1e-12).log()
            law = torch.zeros_like(self.initial_distribution)
            law[support] = weights / weights.sum()
            return law
        weights = -torch.rand(self.n_states, dtype=self.dtype, device=self.device, generator=generator).clamp_min(1e-12).log()
        return weights / weights.sum()

    def transition(self, states, mu, actions):
        return self.transitions[states, actions]

    def sample(self, states, mu, actions, generator):
        probabilities = self.transition(states, mu, actions)
        flat = torch.multinomial(probabilities.reshape(-1, self.n_states), 1, generator=generator)
        return flat.reshape(probabilities.shape[:-1])

    def reward(self, states, mu, actions):
        if self.config.congestion_reward == "shared":
            risk = (mu * (mu @ self.risk_matrix.T)).sum(-1)
        else:
            risk = (mu * self.risk_matrix[states]).sum(-1)
        reward = self.speed[actions] - self.config.safety_cost * risk - self.config.comfort_cost * self.effort[actions]
        return reward + torch.zeros_like(states, dtype=self.dtype)

    def terminal_reward(self, states, mu):
        # Source objective is discounted running reward with artificial truncation.
        return torch.zeros_like(states, dtype=self.dtype)

    def policy(self, theta, t, state, mu):
        probabilities = theta(t, mu) if callable(theta) else theta
        if state.ndim == 0:
            return probabilities[..., state, :]
        if probabilities.ndim == 2:
            return probabilities[state]
        index = state.unsqueeze(-1).unsqueeze(-1).expand(*state.shape, 1, self.n_actions)
        return torch.gather(probabilities, -2, index).squeeze(-2)
