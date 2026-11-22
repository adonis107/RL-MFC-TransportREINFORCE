"""Fouque--Lauriere--Zhang (arXiv:2511.06812v1), Algorithm 2.

Reward critics maximize value, reversing the reference's cost convention.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .timing import report_progress, synchronized_time
from .validation import monte_carlo_objective


@dataclass(frozen=True)
class IHMFCACConfig:
    n_train: int = 10_000
    horizon: int | None = None  # validation only
    validation_interval: int = 100
    discount: float = math.exp(-0.01)  # Section 7.1.3, independently of benchmark evaluation
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    population_lr: float = 1e-3
    actor_power: float = 0.9
    critic_power: float = 0.75
    population_power: float = 0.6
    state_bins: int = 6
    action_bins: int = 7
    state_min: float = -3.0
    state_max: float = 3.0
    action_min: float = -3.0
    action_max: float = 3.0
    actor_width: int = 64
    critic_width: int = 128
    std_floor: float = 1e-5
    max_transitions: int | None = None
    seed: int = 0


class StationaryActor(nn.Module):
    """Stationary state-only actor: categorical on finite states, Gaussian on R."""
    is_stationary_actor = True

    def __init__(self, n_states=None, n_actions=None, width=64, std_floor=1e-5):
        super().__init__()
        self.spec = dict(n_states=n_states, n_actions=n_actions, width=width, std_floor=std_floor)
        self.n_states, self.n_actions = n_states, n_actions
        self.std_floor = std_floor
        self.shared = nn.Sequential(nn.Linear(n_states or 1, width), nn.Tanh())
        if n_states is not None:
            self.logits = nn.Linear(width, n_actions)
        else:
            self.mean = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
            self.scale = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def distribution(self, states):
        dtype = next(self.parameters()).dtype
        features = F.one_hot(states.long(), self.n_states).to(dtype) if self.n_states else states.to(dtype).unsqueeze(-1)
        hidden = self.shared(features)
        if self.n_states:
            return torch.distributions.Categorical(logits=self.logits(hidden))
        return torch.distributions.Normal(
            self.mean(hidden).squeeze(-1), F.softplus(self.scale(hidden).squeeze(-1)) + self.std_floor)

    def sample_actions(self, states, generator):
        distribution = self.distribution(states)
        if self.n_states:
            return torch.multinomial(distribution.probs.reshape(-1, self.n_actions), 1,
                                     generator=generator).reshape(states.shape)
        noise = torch.randn(states.shape, dtype=distribution.loc.dtype,
                            device=states.device, generator=generator)
        return distribution.loc + distribution.scale * noise

    def export_payload(self):
        return {"kind": "ih_mfc_ac_actor", "spec": self.spec,
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def from_payload(cls, payload, *, device, dtype):
        actor = cls(**payload["spec"]).to(device=device, dtype=dtype)
        actor.load_state_dict(payload["state_dict"])
        actor.eval()
        return actor


class IHMFCAC:
    """Single-sample Algorithm 2 with one continuing trajectory and critic per bin."""
    def __init__(self, env, config=IHMFCACConfig()):
        self.env, self.config = env, config
        self.discrete = hasattr(env, "n_states")
        if not self.discrete and type(env).__name__ != "LQ":
            raise ValueError("IH-MFC-AC supports finite-state benchmarks and stationary LQ dynamics; "
                             "portfolio/bimodal require a separately specified continuing problem.")
        self._validate_config()
        self.generator = torch.Generator(device=env.device).manual_seed(config.seed)
        # Keep initialization reproducible without changing the caller's RNG stream.
        devices = [torch.device(env.device)] if torch.device(env.device).type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            self.policy = StationaryActor(
                env.n_states if self.discrete else None, env.n_actions if self.discrete else None,
                config.actor_width, config.std_floor).to(device=env.device, dtype=env.dtype)
            m = env.n_states if self.discrete else config.state_bins
            a = env.n_actions if self.discrete else config.action_bins
            self.critics = nn.ModuleList([
                nn.Sequential(nn.Linear(env.n_states if self.discrete else 1, config.critic_width),
                              nn.Tanh(), nn.Linear(config.critic_width, 1))
                for _ in range(m * a)
            ]).to(device=env.device, dtype=env.dtype)
        self.n_state_bins, self.n_action_bins = m, a
        self.n_bins = m * a
        self.bin_states = torch.arange(m, device=env.device).repeat_interleave(a)
        if self.discrete:
            self.bin_actions = torch.arange(a, device=env.device).repeat(m)
        else:
            self.state_edges = torch.linspace(config.state_min, config.state_max, m + 1,
                                               device=env.device, dtype=env.dtype)
            edges = torch.linspace(config.action_min, config.action_max, a + 1,
                                   device=env.device, dtype=env.dtype)
            self.bin_actions = ((edges[:-1] + edges[1:]) / 2).repeat(m)
        self.actor_optimizer = torch.optim.SGD(self.policy.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.SGD(self.critics.parameters(), lr=config.critic_lr)
        self.iterations = 0
        self.simulator_transitions = 0
        self.states = None

    def _validate_config(self):
        c = self.config
        if not 0 < c.discount < 1:
            raise ValueError("The published discounted continuing algorithm requires 0 < discount < 1.")
        if not (0 < c.actor_lr < c.critic_lr < c.population_lr <= 1):
            raise ValueError("Require 0 < actor_lr < critic_lr < population_lr <= 1.")
        if not (0.5 < c.population_power < c.critic_power < c.actor_power <= 1):
            raise ValueError("Require 1/2 < population_power < critic_power < actor_power <= 1.")
        if c.n_train < 1 or c.validation_interval < 0:
            raise ValueError("n_train must be positive and validation_interval nonnegative.")
        if min(c.state_bins, c.action_bins, c.actor_width, c.critic_width) < 1 or c.std_floor <= 0:
            raise ValueError("Bin counts, network widths and std_floor must be positive.")
        if not c.state_min < c.state_max or not c.action_min < c.action_max:
            raise ValueError("Bin bounds must be ordered.")
        if c.max_transitions is not None and c.max_transitions < 0:
            raise ValueError("max_transitions must be nonnegative.")
        scalars = (c.discount, c.actor_lr, c.critic_lr, c.population_lr, c.state_min,
                   c.state_max, c.action_min, c.action_max, c.std_floor)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Configuration values must be finite.")

    @property
    def n_train(self):
        cap = self.config.max_transitions
        return self.config.n_train if cap is None else min(self.config.n_train, cap // self.transitions_per_iteration)

    @property
    def transitions_per_iteration(self):
        return self.n_bins + 1

    @property
    def horizon(self):
        return self.env.config.T if self.config.horizon is None else self.config.horizon

    @property
    def validation_interval(self):
        return self.config.validation_interval

    @property
    def discount(self):
        """Benchmark discount for evaluation; config.discount is for training."""
        return getattr(self.env.config, "discount", getattr(self.env.config, "gamma", 1.0))

    def state_bin(self, states):
        if self.discrete:
            return states.long()
        # Outer cells include tails. States and policy actions are never clipped.
        return torch.bucketize(states.contiguous(), self.state_edges[1:-1], right=True)

    def features(self, states):
        return F.one_hot(states.long(), self.env.n_states).to(self.env.dtype) if self.discrete else states.unsqueeze(-1)

    def value(self, states):
        """Reward envelope over critics whose state cell contains x."""
        values = torch.stack([critic(self.features(states)).squeeze(-1) for critic in self.critics], dim=-1)
        eligible = self.state_bin(states).unsqueeze(-1) == self.bin_states
        return values.masked_fill(~eligible, -torch.inf).max(dim=-1).values

    def initialize_paths(self):
        if self.discrete:
            auxiliary = self.bin_states.clone()  # uniform on singleton state cells
            initial = self.env.initial_distribution
            individual = torch.multinomial(initial, 1, generator=self.generator)
            self.states = torch.cat([auxiliary, individual])
            self.population = self.features(self.states).detach()
        else:
            left, right = self.state_edges[:-1][self.bin_states], self.state_edges[1:][self.bin_states]
            auxiliary = left + (right - left) * torch.rand(self.n_bins, device=self.env.device,
                                                         dtype=self.env.dtype, generator=self.generator)
            self.states = torch.cat([auxiliary, self.env.sample_initial(1, self.generator)])
            # LQ reads a law only through its mean. This scalar recursion is
            # exactly the mean of the reference's weighted empirical measure.
            self.population = self.states.detach().clone()

    def learning_rates(self, iteration):
        c = self.config
        n = iteration + 1
        return (c.actor_lr / n**c.actor_power, c.critic_lr / n**c.critic_power,
                c.population_lr / n**c.population_power)

    def step(self):
        if self.states is None:
            self.initialize_paths()
        if self.config.max_transitions is not None and (
                self.simulator_transitions + self.transitions_per_iteration > self.config.max_transitions):
            raise RuntimeError("Insufficient remaining transition budget for a complete iteration.")
        actor_lr, critic_lr, population_lr = self.learning_rates(self.iterations)
        self.actor_optimizer.param_groups[0]["lr"] = actor_lr
        self.critic_optimizer.param_groups[0]["lr"] = critic_lr
        states, laws = self.states, self.population
        active = self.state_bin(states[:-1]) == self.bin_states
        with torch.no_grad():
            actions = self.policy.sample_actions(states, self.generator)
            actions[:-1] = torch.where(active, self.bin_actions, actions[:-1])
            rewards = self.env.reward(states, laws, actions)
            next_states = self.env.sample(states, laws, actions, self.generator)
            self.simulator_transitions += self.transitions_per_iteration
            # All targets use the iteration-n snapshot; no gradient through targets.
            targets = rewards + self.config.discount * self.value(next_states)
            actor_td = targets[-1] - self.value(states[-1:])[0]
        losses = []
        for i in active.nonzero(as_tuple=False).flatten().tolist():
            current = self.critics[i](self.features(states[i:i+1])).squeeze()
            losses.append((targets[i] - current).square())
        self.critic_optimizer.zero_grad(set_to_none=True)
        if losses:
            critic_loss = torch.stack(losses).sum()  # each bin has its own SGD update
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            critic_loss = torch.zeros((), device=self.env.device, dtype=self.env.dtype)
        self.actor_optimizer.zero_grad(set_to_none=True)
        log_probability = self.policy.distribution(states[-1:]).log_prob(actions[-1:]).sum()
        actor_loss = -actor_td.detach() * log_probability
        actor_loss.backward()
        self.actor_optimizer.step()
        with torch.no_grad():
            observed = self.features(next_states) if self.discrete else next_states
            self.population = (1 - population_lr) * laws + population_lr * observed
            self.states = next_states.detach()
        self.iterations += 1
        if not torch.isfinite(self.population).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("Nonfinite IH-MFC-AC population or TD target.")
        return {"reward": float(rewards[-1]), "actor_loss": float(actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()), "active_bins": active.detach(),
                "actions": actions.detach(), "targets": targets.detach(), "actor_td": actor_td.detach()}

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        return monte_carlo_objective(
            self.env, self.policy, lambda t: t, self.discount,
            self.env.config.validation_particles if n_particles is None else n_particles,
            getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon,
            self.config.seed if seed is None else seed)

    def train(self):
        started = synchronized_time(self.env.device)
        if self.states is None:
            self.initialize_paths()
        history = {"objective": [], "loss": [], "critic_loss": [], "validation_objective": [],
                   "train_step_seconds": [], "validation_seconds": [],
                   "setup_seconds": [synchronized_time(self.env.device) - started],
                   "validation_steps": [], "simulator_transitions": []}
        for _ in range(self.iterations, self.n_train):
            started = synchronized_time(self.env.device)
            result = self.step()
            history["train_step_seconds"].append(synchronized_time(self.env.device) - started)
            history["objective"].append(result["reward"])  # one-step continuing reward
            history["loss"].append(result["actor_loss"])
            history["critic_loss"].append(result["critic_loss"])
            history["simulator_transitions"].append(self.simulator_transitions)
            report_progress(self.iterations - 1, self.n_train, history)
            if self.validation_interval and self.iterations % self.validation_interval == 0:
                started = synchronized_time(self.env.device)
                history["validation_objective"].append(float(self.evaluate(seed=self.config.seed + self.n_train)))
                history["validation_steps"].append(self.iterations)
                history["validation_seconds"].append(synchronized_time(self.env.device) - started)
        return self.policy, history

    def checkpoint(self):
        return {"config": asdict(self.config), "actor": self.policy.export_payload(),
                "critics": self.critics.state_dict(), "states": self.states,
                "population": self.population, "iterations": self.iterations,
                "simulator_transitions": self.simulator_transitions,
                "generator_state": self.generator.get_state(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict()}
