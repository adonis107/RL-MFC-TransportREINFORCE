from dataclasses import dataclass
import copy
import math
import importlib
import inspect

import torch
from torch import nn

from .mfreinforce import MFReinforce
from .mixture import GaussianMixture, MixtureConstraints
from .reinforce import exact_continuous_validation_objective
from .timing import report_progress, synchronized_time


_SAMPLE_TIME_ARGUMENT = {}


def sample_accepts_time(sample):
    """Cache whether an environment's sample method takes a time argument."""
    function = getattr(sample, "__func__", sample)
    if function not in _SAMPLE_TIME_ARGUMENT:
        _SAMPLE_TIME_ARGUMENT[function] = "t" in inspect.signature(sample).parameters
    return _SAMPLE_TIME_ARGUMENT[function]


@dataclass(frozen=True)
class DiscreteTransportConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_logit_gradient: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    lambda_: float | None = None
    eta: float | None = None
    simplex_sigma: float = 1.0
    flow: str = "exact"
    n_flow_particles: int | None = None
    baseline: bool = True
    reuse_state_gradient: bool = True
    seed: int = 0


class DiscreteTransport(MFReinforce):
    def __init__(self, env, policy=None, config=DiscreteTransportConfig()):
        super().__init__(env, policy=policy, config=config)

    @property
    def lambda_(self):
        if self.config.lambda_ is not None:
            return self.config.lambda_
        return getattr(self.env.config, "transport_lambda", 0.2)

    @property
    def eta(self):
        if self.config.eta is not None:
            return self.config.eta
        return getattr(self.env.config, "transport_eta", self.lambda_)

    def sample_simplex(self, generator):
        logits = torch.zeros(self.env.n_states, dtype=self.env.dtype, device=self.env.device)
        logits[:-1] = self.config.simplex_sigma * torch.randn(
            self.env.n_states - 1, dtype=self.env.dtype, device=self.env.device, generator=generator
        )
        return torch.softmax(logits, dim=0)

    def sample_simplex_batch(self, n_particles, generator):
        logits = torch.zeros(n_particles, self.env.n_states, dtype=self.env.dtype, device=self.env.device)
        logits[:, :-1] = self.config.simplex_sigma * torch.randn(
            n_particles,
            self.env.n_states - 1,
            dtype=self.env.dtype,
            device=self.env.device,
            generator=generator,
        )
        return torch.softmax(logits, dim=-1)

    def perturb_law(self, law, q, scale):
        return (1.0 - scale) * law + scale * q

    def simplex_score_h(self, q):
        q = q.clamp_min(1e-12)
        z = torch.log(q[..., :-1] / q[..., -1:])
        a = -z / self.config.simplex_sigma**2
        return a / q[..., :-1] + a.sum(dim=-1, keepdim=True) / q[..., -1:] - 1.0 / q[..., :-1] + 1.0 / q[..., -1:]

    def estimate_state_sensitivities(self, laws, seed, initial_distribution=None, eta=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        eta = self.eta if eta is None else eta

        sensitivities = [
            torch.zeros(self.env.n_states, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        states = [self.initial_states(self.n_logit_gradient, generator, initial_distribution)]
        action_log_probs = []
        h_values = []

        for t in range(self.horizon):
            q = self.sample_simplex_batch(self.n_logit_gradient, generator)
            perturbed_law = self.perturb_law(laws[t], q, eta)
            actions, log_probs = self.sample_actions_with_log_probs(t, states[-1], perturbed_law, generator)
            action_log_probs.append(log_probs)
            h_values.append(self.simplex_score_h(q))

            with torch.no_grad():
                states.append(self.sample_next_state(t, states[-1], perturbed_law, actions, generator))

        factor = (1.0 - eta) / eta
        for target_t in range(1, self.horizon + 1):
            numerator = torch.zeros(self.env.n_states - 1, self.n_parameters, dtype=self.env.dtype, device=self.env.device)

            law_score_sum = torch.zeros(
                self.n_logit_gradient, self.n_parameters, dtype=self.env.dtype, device=self.env.device
            )
            for s in range(target_t):
                law_score_sum = law_score_sum - factor * (h_values[s] @ sensitivities[s][:-1])

            target_states = states[target_t]
            mask = target_states < self.env.n_states - 1
            numerator.index_add_(0, target_states[mask], law_score_sum[mask])
            action_gradients = self.state_indexed_log_prob_gradients(
                torch.stack(action_log_probs[:target_t]), target_states
            )
            numerator = numerator + action_gradients[:-1]

            sensitivities[target_t][:-1] = numerator / self.n_logit_gradient
            sensitivities[target_t][-1] = -sensitivities[target_t][:-1].sum(dim=0)

        return sensitivities

    def trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_state = self.initial_state(generator, initial_distribution)
        state = self.initial_state(generator, initial_distribution)
        base_rewards = []
        rewards = []
        policy_score = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        perturbation_score = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)

        for t in range(self.horizon):
            law = laws[t]
            base_action, _ = self.sample_action(t, base_state, law, generator)
            q = self.sample_simplex(generator)
            perturbed_law = self.perturb_law(law, q, self.lambda_)
            action, _ = self.sample_action(t, state, perturbed_law, generator)
            base_rewards.append(self.env.reward(base_state, law, base_action))
            rewards.append(self.env.reward(state, perturbed_law, action))

            policy_score = policy_score + self.log_prob_gradient(t, state, perturbed_law, action)
            perturbation_score = perturbation_score + self.simplex_score_h(q) @ sensitivities[t][:-1]

            with torch.no_grad():
                base_state = self.sample_next_state(t, base_state, law, base_action, generator)
                state = self.sample_next_state(t, state, perturbed_law, action, generator)

        q_terminal = self.sample_simplex(generator)
        terminal_law = self.perturb_law(laws[-1], q_terminal, self.lambda_)
        terminal_reward = self.env.terminal_reward(state, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_state, laws[-1])
        perturbation_score = perturbation_score + self.simplex_score_h(q_terminal) @ sensitivities[-1][:-1]

        trajectory_score = policy_score - ((1.0 - self.lambda_) / self.lambda_) * perturbation_score
        trajectory_return = self.discounted_returns(rewards, terminal_reward)[0]
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        return trajectory_score, trajectory_return, base_return

    def batched_trajectory_components(self, laws, seed, initial_distribution=None, lambda_=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        base_states = self.initial_states(self.n_particles, generator, initial_distribution)
        states = self.initial_states(self.n_particles, generator, initial_distribution)
        base_rewards = []
        rewards = []
        action_log_probs = []
        law_scores = []

        for t in range(self.horizon):
            law = laws[t]
            base_actions, _ = self.sample_actions_with_log_probs(t, base_states, law.expand(self.n_particles, -1), generator)
            q = self.sample_simplex_batch(self.n_particles, generator)
            perturbed_law = self.perturb_law(law, q, lambda_)
            actions, log_probs = self.sample_actions_with_log_probs(t, states, perturbed_law, generator)
            base_rewards.append(self.env.reward(base_states, law, base_actions))
            rewards.append(self.env.reward(states, perturbed_law, actions))
            action_log_probs.append(log_probs)
            law_scores.append(self.simplex_score_h(q))

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, law, base_actions, generator)
                states = self.sample_next_state(t, states, perturbed_law, actions, generator)

        q_terminal = self.sample_simplex_batch(self.n_particles, generator)
        terminal_law = self.perturb_law(laws[-1], q_terminal, lambda_)
        terminal_reward = self.env.terminal_reward(states, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_states, laws[-1])
        law_scores.append(self.simplex_score_h(q_terminal))

        returns = self.discounted_returns(rewards, terminal_reward)[0]
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean() if self.config.baseline else returns
        action_score = torch.stack(action_log_probs).sum(dim=0)
        action_gradient = self.flat_grad((action_score * advantages.detach()).sum())
        weights = advantages.detach()
        return action_gradient, weights, law_scores, base_return.mean()

    def combine_batched_trajectory_components(self, action_gradient, weights, law_scores, sensitivities, lambda_=None):
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        law_gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        for index, simplex_score in enumerate(law_scores):
            law_gradient = law_gradient + (simplex_score.transpose(0, 1) @ weights) @ sensitivities[index][:-1]
        law_gradient = -((1.0 - lambda_) / lambda_) * law_gradient
        return (action_gradient + law_gradient) / self.n_particles

    def batched_trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        action_gradient, weights, law_scores, base_return = self.batched_trajectory_components(
            laws,
            seed,
            initial_distribution=initial_distribution,
        )
        gradient = self.combine_batched_trajectory_components(action_gradient, weights, law_scores, sensitivities)
        return gradient, base_return

    def estimate_gradient(self, seed):
        law_generator = torch.Generator(device=self.env.device)
        law_generator.manual_seed(seed + 30_000)
        initial_distribution = self.sample_initial_distribution(law_generator)
        laws, _ = self.mean_field_law_flow(seed=seed + 20_000, initial_distribution=initial_distribution)
        shared_sensitivities = None
        if self.config.reuse_state_gradient:
            shared_sensitivities = self.estimate_state_sensitivities(
                laws,
                seed + 10_000,
                initial_distribution=initial_distribution,
            )

        if shared_sensitivities is not None:
            return self.batched_trajectory_gradient(
                laws,
                shared_sensitivities,
                seed,
                initial_distribution=initial_distribution,
            )

        gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        return_sum = torch.zeros((), dtype=self.env.dtype, device=self.env.device)
        scores = []
        returns = []

        for b in range(self.n_particles):
            sensitivities = self.estimate_state_sensitivities(
                laws,
                seed + 10_000 + b,
                initial_distribution=initial_distribution,
            )

            score, trajectory_return, base_return = self.trajectory_gradient(
                laws,
                sensitivities,
                seed + b,
                initial_distribution=initial_distribution,
            )
            if self.config.baseline:
                scores.append(score)
                returns.append(trajectory_return)
                return_sum = return_sum + base_return.detach()
            else:
                gradient = gradient + score * trajectory_return.detach()
                return_sum = return_sum + base_return.detach()

        if self.config.baseline:
            scores = torch.stack(scores)
            objectives = torch.stack(returns)
            advantages = objectives - objectives.mean()
            gradient = (scores * advantages.detach().unsqueeze(-1)).sum(dim=0)

        gradient = gradient / self.n_particles
        objective = return_sum / self.n_particles
        return gradient, objective


def train_discrete_transport(env, policy=None, config=DiscreteTransportConfig()):
    return DiscreteTransport(env, policy=policy, config=config).train()


@dataclass(frozen=True)
class ContinuousTransportConfig:
    """Blocks and chart of the Gaussian-mixture transport estimator.

    n_particles is the main block B, n_law_gradient the auxiliary block n, and
    n_law_particles (through n_flow_particles) the population block M the
    mixture is fitted to. n_components and the bounds below describe the chart
    and keep EM away from a covariance collapsing onto one observation.

    jacobian_floor is the local-identification safeguard: the smallest singular
    value of A, relative to its largest, still treated as identified. Raising K
    past what the population law needs leaves the extra components undetermined
    and the run reports dropped directions in sensitivity_fallbacks.

    n_law_gradient has to grow with the number of policy parameters, not merely
    satisfy n eta^2 >= 1: B is a q_K by d_theta matrix estimated from n
    trajectories, and its error is amplified once per time step.
    """

    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_law_gradient: int | None = None
    n_law_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    lambda_: float | None = None
    eta: float | None = None
    n_components: int = 3
    em_iterations: int = 50
    em_tolerance: float = 1e-6
    weight_floor: float = 1e-4
    sigma_min: float = 1e-2
    sigma_max: float = 1e2
    mean_radius: float | None = 1e3
    weight_randomizer_sigma: float = 1.0
    mean_randomizer_sigma: float = 1.0
    log_scale_randomizer_sigma: float = 1.0
    quadrature_nodes: int = 24
    jacobian_floor: float = 1e-1
    flow: str = "particle"
    n_flow_particles: int | None = None
    baseline: bool = True
    reuse_state_gradient: bool = True
    seed: int = 0


class ContinuousTransport:
    """Transport REINFORCE on the Gaussian-mixture chart of the population law.

    The population law is represented by the fitted parameters of a K-component
    Gaussian mixture. The main perturbation follows the transport construction in
    the reference: component weights are mixed with random weights, component
    means are moved toward random target means, and one-dimensional component
    standard deviations move along their Gaussian Wasserstein geodesics. One
    policy update runs the three independent simulation blocks of Algorithm
    "Transport REINFORCE":

      1. M population particles, whose empirical law is fitted at every time by
         the EM rule R_K, give the coordinates z_t;
      2. n auxiliary particles, allocated across centered shifts of the policy
         coordinates, give the population sensitivities D_t = grad_theta z_t;
      3. B main trajectories at radius lambda give the gradient estimate, whose
         mean-field correction is the randomized-mixture score contracted with
         D_t.

    Nothing in the three blocks uses a density or a derivative of the transition
    kernel: only sampled states and actions, the known policy score, and the
    known perturbation density enter the formulas.
    """

    def __init__(self, env, policy=None, config=ContinuousTransportConfig()):
        if hasattr(env, "n_states"):
            raise TypeError("ContinuousTransport is for continuous state spaces. Use DiscreteTransport instead.")
        if not hasattr(env, "sample_initial"):
            raise TypeError("ContinuousTransport requires an environment with sample_initial.")
        if config.flow not in {"exact", "particle"}:
            raise ValueError("flow must be either 'exact' or 'particle'.")

        self.env = env
        self.config = config
        self.policy = self._make_policy() if policy is None else policy
        self.state_dim = self.measure_state_dim()
        self.mixture = GaussianMixture(
            config.n_components,
            self.state_dim,
            env.dtype,
            env.device,
            MixtureConstraints(
                weight_floor=config.weight_floor,
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                mean_radius=config.mean_radius,
            ),
        )
        self.coordinate_dim = self.mixture.coordinate_dim
        self.sensitivity_fallbacks = 0
        self.fitted_coordinates = None

    def measure_state_dim(self):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed)
        sample = self.env.sample_initial(1, generator)
        return 1 if sample.ndim <= 1 else sample.shape[-1]

    def trainable_parameters(self):
        if isinstance(self.policy, nn.Module):
            return list(self.policy.parameters())
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
    def n_law_gradient(self):
        if self.config.n_law_gradient is not None:
            return self.config.n_law_gradient
        if hasattr(self.env.config, "n_law_gradient"):
            return self.env.config.n_law_gradient
        return getattr(self.env.config, "n_logit_gradient", 10)

    @property
    def n_law_particles_per_shift(self):
        return max(1, self.n_law_gradient // max(2 * self.n_parameters, 1))

    @property
    def effective_n_law_gradient(self):
        return 2 * self.n_parameters * self.n_law_particles_per_shift

    @property
    def n_law_particles(self):
        return self.n_particles if self.config.n_law_particles is None else self.config.n_law_particles

    @property
    def n_flow_particles(self):
        return self.n_law_particles if self.config.n_flow_particles is None else self.config.n_flow_particles

    @property
    def n_population_particles(self):
        """Number M of particles whose empirical law is fitted by the mixture."""
        return self.n_flow_particles

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
    def lambda_(self):
        if self.config.lambda_ is not None:
            return self.config.lambda_
        if hasattr(self.env.config, "transport_lambda"):
            return self.env.config.transport_lambda
        perturbation_scale = getattr(self.env.config, "perturbation_scale", 0.0)
        return perturbation_scale if perturbation_scale > 0.0 else 0.1

    @property
    def eta(self):
        if self.config.eta is not None:
            return self.config.eta
        if hasattr(self.env.config, "transport_eta"):
            return self.env.config.transport_eta
        return self.lambda_

    @property
    def discount(self):
        if hasattr(self.env.config, "discount"):
            return self.env.config.discount
        return getattr(self.env.config, "gamma", 1.0)

    @property
    def n_parameters(self):
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def _make_policy(self):
        if hasattr(self.env, "zero_policy"):
            return nn.Parameter(self.env.zero_policy())

        module = importlib.import_module(type(self.env).__module__)
        policy_class = getattr(module, f"{type(self.env).__name__}Policy")
        return policy_class(self.env.config)

    def policy_time(self, t):
        if isinstance(self.policy, nn.Module):
            return torch.tensor(float(t), dtype=self.env.dtype, device=self.env.device)
        return t

    def set_flat_gradient(self, gradient):
        offset = 0
        for parameter in self.trainable_parameters():
            next_offset = offset + parameter.numel()
            parameter.grad = gradient[offset:next_offset].reshape_as(parameter).clone()
            offset = next_offset

    def flatten_grads(self, grads):
        pieces = []
        for parameter, grad in zip(self.trainable_parameters(), grads):
            if grad is None:
                pieces.append(torch.zeros_like(parameter).reshape(-1))
            else:
                pieces.append(grad.reshape(-1))
        return torch.cat(pieces)

    def flat_grad(self, value, retain_graph=False):
        grads = torch.autograd.grad(value, self.trainable_parameters(), allow_unused=True, retain_graph=retain_graph)
        return self.flatten_grads(grads)

    def initial_state(self, generator):
        return self.env.sample_initial(1, generator).reshape(())

    def initial_states(self, n_particles, generator):
        return self.env.sample_initial(n_particles, generator)

    def particle_coordinates(self, states):
        """Particle states as a (M, d) matrix, whatever shape the environment uses."""
        return states.reshape(states.shape[0], self.state_dim)

    def law_features(self, z):
        """Population statistics of Gamma_K(z), for one coordinate or a batch of them.

        An environment that declares state_law_features is given the exact
        mixture expectation of those features, computed by Gauss-Hermite
        quadrature against the known mixture density. Otherwise the environment
        uses the (mean, variance) convention, which the chart provides in closed
        form.
        """
        if hasattr(self.env, "state_law_features"):
            points, weights = self.mixture.quadrature(z, self.config.quadrature_nodes)
            states = points.squeeze(-1) if self.state_dim == 1 else points
            features = self.env.state_law_features(states)
            return (features * weights.unsqueeze(-1)).sum(dim=-2)

        mean, covariance = self.mixture.mean_covariance(z)
        if self.state_dim > 1:
            raise ValueError(
                "A multidimensional state space needs an environment that defines state_law_features."
            )
        return torch.stack([mean[..., 0], covariance[..., 0, 0]], dim=-1)

    def law_argument(self, features):
        if hasattr(self.env, "law_argument"):
            return self.env.law_argument(features)
        return features[..., 0]

    def population_law(self, z):
        """Population argument handed to the environment for the mixture Gamma_K(z)."""
        return self.law_argument(self.law_features(z))

    def sample_action(self, t, state, law, generator):
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                action = self.env.sample_action(self.policy, t, state, law, generator)
            else:
                action = self.env.policy(self.policy, self.policy_time(t), state, law).sample()
        return action.reshape(())

    def sample_actions_with_log_probs(self, t, states, law, generator):
        action_law = self.env.policy(self.policy, self.policy_time(t), states, law)
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                actions = self.env.sample_action(self.policy, t, states, law, generator)
            else:
                actions = action_law.sample()
        return actions, action_law.log_prob(actions.detach())

    def log_prob_gradient(self, t, state, law, action):
        action_law = self.env.policy(self.policy, self.policy_time(t), state, law)
        log_prob = action_law.log_prob(action)
        grads = torch.autograd.grad(log_prob, self.trainable_parameters(), allow_unused=True)
        return self.flatten_grads(grads)

    def sample_next_state(self, t, state, law, action, generator):
        if sample_accepts_time(self.env.sample):
            return self.env.sample(state, law, action, generator, t=t).reshape_as(state)
        return self.env.sample(state, law, action, generator).reshape_as(state)

    def sample_actions_for_population(self, t, states, law, generator):
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                return self.env.sample_action(self.policy, t, states, law, generator)
            return self.env.policy(self.policy, self.policy_time(t), states, law).sample()

    def sample_next_states_for_population(self, t, states, law, actions, generator):
        if sample_accepts_time(self.env.sample):
            return self.env.sample(states, law, actions, generator, t=t)
        return self.env.sample(states, law, actions, generator)

    def discounted_returns(self, rewards, terminal_reward):
        values = [None] * (len(rewards) + 1)
        values[-1] = terminal_reward
        for t in range(len(rewards) - 1, -1, -1):
            values[t] = rewards[t] + self.discount * values[t + 1]
        return values

    def sample_perturbations(self, shape, generator):
        """Standard Gaussian displacements of the mixture coordinate."""
        return torch.randn(
            *shape, self.coordinate_dim, dtype=self.env.dtype, device=self.env.device, generator=generator
        )

    def _normal_log_density(self, value, sigma):
        return -0.5 * (value / sigma).square() - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    def sample_transport_randomizers(self, shape, generator):
        """Draw the continuous-state randomizer R=(Q,A,B) in encoded form.

        The returned tensor uses the same packing as the mixture coordinate, but
        its entries parameterize the random target mixture: softmax logits for Q,
        target means A, and target log standard deviations for sqrt(B). The
        transport map below converts these target parameters into the actual
        perturbed mixture parameters.
        """
        shape = tuple(shape)
        beta_dim = self.config.n_components - 1
        pieces = []
        if beta_dim:
            pieces.append(
                self.config.weight_randomizer_sigma
                * torch.randn(
                    *shape,
                    beta_dim,
                    dtype=self.env.dtype,
                    device=self.env.device,
                    generator=generator,
                )
            )

        pieces.append(
            self.config.mean_randomizer_sigma
            * torch.randn(
                *shape,
                self.config.n_components * self.state_dim,
                dtype=self.env.dtype,
                device=self.env.device,
                generator=generator,
            )
        )
        pieces.append(
            self.config.log_scale_randomizer_sigma
            * torch.randn(
                *shape,
                self.config.n_components * self.mixture.scale_dim,
                dtype=self.env.dtype,
                device=self.env.device,
                generator=generator,
            )
        )
        return torch.cat(pieces, dim=-1)

    def transport_coordinate(self, coordinate, randomizer, scale):
        """Apply the Gaussian-mixture transport perturbation from the reference.

        The current implementation supports the one-dimensional continuous
        benchmarks in this repository. For d=1, the Wasserstein geodesic between
        Gaussian components moves standard deviations linearly.
        """
        if self.state_dim != 1:
            raise NotImplementedError("Continuous transport randomization currently supports one-dimensional states.")

        target_beta, target_means, target_scales = self.mixture.unpack(randomizer)
        source_weights, source_means, source_scale_tril = self.mixture.decode(coordinate)
        target_weights = self.mixture.weights(target_beta)
        target_std = torch.exp(target_scales[..., :, 0])
        source_std = source_scale_tril[..., :, 0, 0]

        perturbed_weights = (1.0 - scale) * source_weights + scale * target_weights
        perturbed_means = (1.0 - scale) * source_means + scale * target_means
        perturbed_std = (1.0 - scale) * source_std + scale * target_std
        perturbed_scale_tril = torch.zeros(
            perturbed_std.shape + (1, 1),
            dtype=perturbed_std.dtype,
            device=perturbed_std.device,
        )
        perturbed_scale_tril[..., 0, 0] = perturbed_std
        return self.mixture.encode(perturbed_weights, perturbed_means, perturbed_scale_tril)

    def transport_log_density(self, perturbed_coordinate, coordinate, scale):
        """log h^scale(y | z) for the implemented one-dimensional randomizer."""
        if self.state_dim != 1:
            raise NotImplementedError("Continuous transport density currently supports one-dimensional states.")
        if scale <= 0.0 or scale >= 1.0:
            raise ValueError("The transport density score is defined for scale in (0, 1).")

        tiny = torch.finfo(perturbed_coordinate.dtype).tiny
        beta_y, means_y, scales_y = self.mixture.unpack(perturbed_coordinate)
        beta_z, means_z, scales_z = self.mixture.unpack(coordinate)

        log_density = torch.zeros(perturbed_coordinate.shape[:-1], dtype=perturbed_coordinate.dtype, device=perturbed_coordinate.device)
        if self.config.n_components > 1:
            weights_y = self.mixture.weights(beta_y).clamp_min(tiny)
            weights_z = self.mixture.weights(beta_z)
            target_weights = ((weights_y - (1.0 - scale) * weights_z) / scale).clamp_min(tiny)
            target_beta = torch.log(target_weights[..., :-1]) - torch.log(target_weights[..., -1:])
            log_density = log_density + self._normal_log_density(
                target_beta,
                self.config.weight_randomizer_sigma,
            ).sum(dim=-1)
            log_jacobian = (
                torch.log(weights_y).sum(dim=-1)
                - torch.log(target_weights).sum(dim=-1)
                - (self.config.n_components - 1) * math.log(scale)
            )
            log_density = log_density + log_jacobian

        target_means = (means_y - (1.0 - scale) * means_z) / scale
        log_density = log_density + self._normal_log_density(
            target_means,
            self.config.mean_randomizer_sigma,
        ).sum(dim=(-2, -1))
        mean_dim = target_means.shape[-2] * target_means.shape[-1]
        log_density = log_density - mean_dim * math.log(scale)

        y_std = torch.exp(scales_y[..., :, 0])
        z_std = torch.exp(scales_z[..., :, 0])
        target_std = (y_std - (1.0 - scale) * z_std).clamp_min(tiny) / scale
        target_log_std = torch.log(target_std)
        log_density = log_density + self._normal_log_density(
            target_log_std,
            self.config.log_scale_randomizer_sigma,
        ).sum(dim=-1)
        log_density = log_density + (
            torch.log(y_std.clamp_min(tiny)) - torch.log((y_std - (1.0 - scale) * z_std).clamp_min(tiny))
        ).sum(dim=-1)
        return log_density

    def transport_scores(self, perturbed_coordinates, coordinate, scale):
        """Score s^scale(y, z)=grad_z log h^scale(y | z)."""
        if perturbed_coordinates.ndim == 1:
            return torch.func.grad(lambda base: self.transport_log_density(perturbed_coordinates, base, scale))(coordinate)

        def score_one(perturbed):
            return torch.func.grad(lambda base: self.transport_log_density(perturbed, base, scale))(coordinate)

        return torch.func.vmap(score_one)(perturbed_coordinates)

    def population_coordinates(self, horizon=None, seed=None, jacobians=True, n_particles=None, update_cache=True):
        """Fit the mixture coordinate along the represented flow.

        Returns the coordinates z_0, ..., z_T and, when requested, the empirical
        score Jacobians A_t = M^{-1} sum_i D_z psi_K(X_t^i, z_t). With
        flow='exact' and an environment exposing its analytic moment flow, the
        coordinate is read off that flow instead of a particle fit; this is the
        oracle-population variant and is only defined for a single component.
        """
        horizon = self.horizon if horizon is None else horizon
        if self.config.flow == "exact" and hasattr(self.env, "moment_flow"):
            return self.exact_population_coordinates(horizon, jacobians=jacobians)

        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed if seed is None else seed)
        n_particles = self.n_population_particles if n_particles is None else n_particles
        states = self.env.sample_initial(n_particles, generator)

        coordinates = []
        score_jacobians = []
        coordinate = None
        for t in range(horizon + 1):
            particles = self.particle_coordinates(states.detach())
            # Each fit is warm-started from the fit of the same time at the
            # previous policy update when there is one, and from the previous
            # time otherwise. This is what keeps the fitted coordinate on the
            # same local root of the likelihood equation as theta moves, and it
            # starts EM from a nearly converged fit after the first update.
            previous = self.fitted_coordinates
            warm_start = previous[t] if previous is not None and t < len(previous) else coordinate
            coordinate = self.mixture.fit(
                particles,
                warm_start=warm_start,
                iterations=self.config.em_iterations,
                tolerance=self.config.em_tolerance,
            )
            coordinates.append(coordinate)
            if jacobians:
                score_jacobians.append(self.mixture.mean_score_jacobian(particles, coordinate))
            if t == horizon:
                break

            law = self.population_law(coordinate)
            actions = self.sample_actions_for_population(t, states, law, generator)
            with torch.no_grad():
                states = self.sample_next_states_for_population(t, states, law, actions, generator)

        if update_cache:
            self.fitted_coordinates = coordinates
        return coordinates, score_jacobians

    def exact_population_coordinates(self, horizon, jacobians=True):
        """Represented flow of an environment that exposes its analytic moments."""
        if self.config.n_components != 1:
            raise ValueError(
                "flow='exact' reads a single Gaussian off the analytic moment flow; "
                "use flow='particle' with n_components > 1."
            )

        with torch.no_grad():
            means, variances = self.env.moment_flow(self.policy, lambda_=0.0)
        means = means[: horizon + 1].detach()
        variances = variances[: horizon + 1].detach().clamp_min(self.config.sigma_min**2)

        coordinates = []
        score_jacobians = []
        for t in range(horizon + 1):
            weights = torch.ones(1, dtype=self.env.dtype, device=self.env.device)
            mean = means[t].reshape(1, 1)
            scale_tril = variances[t].sqrt().reshape(1, 1, 1)
            coordinate = self.mixture.encode(weights, mean, scale_tril)
            coordinates.append(coordinate)
            if jacobians:
                # The represented law is exactly Gamma_K(z_t) here, so the
                # expectation defining A_t is a quadrature against that mixture.
                points, quadrature_weights = self.mixture.quadrature(coordinate, self.config.quadrature_nodes)
                score_jacobians.append(
                    self.mixture.mean_score_jacobian(points, coordinate, weights=quadrature_weights)
                )

        return coordinates, score_jacobians

    def solve_sensitivity(self, jacobian, b):
        """D = -A^{-1} B, guarded against a mixture fit that is not locally identified.

        A is invertible only where the local-identification assumption holds. It
        degenerates when the represented law does not really need K components:
        the likelihood is then flat along a reparametrization that leaves the
        decoded mixture unchanged. Such a direction carries no information about
        the coordinate, yet inverting it would feed an arbitrarily large
        correction into the gradient. The singular directions below
        jacobian_floor times the largest singular value are therefore dropped,
        which is the zero-inverse rule of the assumption applied one direction at
        a time. Dropped directions are counted so that a run can be diagnosed.
        """
        if not torch.isfinite(jacobian).all():
            self.sensitivity_fallbacks += self.coordinate_dim
            return torch.zeros_like(b)

        left, singular_values, right = torch.linalg.svd(jacobian)
        retained = singular_values > self.config.jacobian_floor * singular_values[0]
        self.sensitivity_fallbacks += int((~retained).sum().item())
        inverse = torch.where(retained, 1.0 / singular_values.clamp_min(torch.finfo(jacobian.dtype).tiny), 0.0)
        sensitivity = -(right.transpose(-1, -2) * inverse) @ (left.transpose(-1, -2) @ b)

        # A sensitivity that has overflowed would otherwise enter the coordinate
        # score of every later time step and turn the whole recursion into NaN,
        # which hides the time step where the estimate actually broke down.
        if not torch.isfinite(sensitivity).all():
            self.sensitivity_fallbacks += self.coordinate_dim
            return torch.zeros_like(b)
        return sensitivity

    def batched_log_prob_gradients(self, action_log_probs, scores, score_dim):
        """Score gradients weighted by the mixture score, batched over time and coordinate.

        Entry [target_t - 1, j] is sum_r psi_K(X_{target_t}^r, z_{target_t})_j times
        the gradient of the log-probabilities of trajectory r up to target_t, which
        is the policy-score part of B_{target_t}. One vmapped backward pass replaces
        the horizon * q_K sequential passes over the same graph.
        """
        log_probs = torch.stack(action_log_probs)
        n_outputs = self.horizon * score_dim
        weights = torch.zeros(n_outputs, *log_probs.shape, dtype=self.env.dtype, device=self.env.device)
        for target_t in range(1, self.horizon + 1):
            for index in range(score_dim):
                weights[(target_t - 1) * score_dim + index, :target_t] = scores[target_t][..., index]

        parameters = self.trainable_parameters()
        grads = torch.autograd.grad(
            log_probs, parameters, grad_outputs=weights, allow_unused=True, is_grads_batched=True
        )
        pieces = []
        for parameter, grad in zip(parameters, grads):
            if grad is None:
                pieces.append(torch.zeros(n_outputs, parameter.numel(), dtype=self.env.dtype, device=self.env.device))
            else:
                pieces.append(grad.reshape(n_outputs, -1))
        return torch.cat(pieces, dim=1).reshape(self.horizon, score_dim, self.n_parameters)

    def shifted_policy(self, parameter_index, amount):
        """Copy the current policy and shift one flattened parameter."""
        if isinstance(self.policy, nn.Module):
            shifted = copy.deepcopy(self.policy)
            shifted.to(self.env.device)
            offset = 0
            for parameter in shifted.parameters():
                next_offset = offset + parameter.numel()
                if offset <= parameter_index < next_offset:
                    with torch.no_grad():
                        parameter.reshape(-1)[parameter_index - offset] += amount
                    return shifted
                offset = next_offset
            raise IndexError("Policy parameter index out of range.")

        shifted = self.policy.detach().clone()
        shifted.reshape(-1)[parameter_index] += amount
        return shifted

    def coordinates_under_policy(self, policy, seed, n_particles, warm_start=None):
        """Fit the represented flow under a temporary policy without changing caches.

        The shifted policy differs from the current one by a single coordinate of size eta, so
        the unshifted fit is a good starting point: warm starting from it keeps the shifted fit
        on the same local root of the likelihood equation, which is what makes the difference of
        the two coordinates a sensitivity rather than a relabelling.
        """
        current_policy = self.policy
        current_cache = self.fitted_coordinates
        try:
            self.policy = policy
            self.fitted_coordinates = warm_start
            coordinates, _ = self.population_coordinates(
                seed=seed,
                jacobians=False,
                n_particles=n_particles,
                update_cache=False,
            )
            return coordinates
        finally:
            self.policy = current_policy
            self.fitted_coordinates = current_cache

    def estimate_coordinate_sensitivities(self, coordinates, score_jacobians=None, seed=0, eta=None):
        """Estimate D_t = grad_theta z_t by centered policy differences.

        This is the continuous-state auxiliary estimator of the reference. The
        configured n_law_gradient is interpreted as the total auxiliary particle
        budget n = 2 d_theta n0, rounded down to at least one particle per shift.
        """
        eta = self.eta if eta is None else eta
        if eta <= 0.0:
            raise ValueError("Continuous transport requires eta > 0 for centered policy differences.")

        n_per_shift = self.n_law_particles_per_shift
        sensitivities = [
            torch.zeros(self.coordinate_dim, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        if self.horizon == 0:
            return sensitivities

        for parameter_index in range(self.n_parameters):
            plus_policy = self.shifted_policy(parameter_index, eta)
            minus_policy = self.shifted_policy(parameter_index, -eta)
            plus = self.coordinates_under_policy(
                plus_policy,
                seed + 10_000 * parameter_index + 1,
                n_per_shift,
                warm_start=coordinates,
            )
            minus = self.coordinates_under_policy(
                minus_policy,
                seed + 10_000 * parameter_index + 2,
                n_per_shift,
                warm_start=coordinates,
            )
            for t in range(1, self.horizon + 1):
                sensitivities[t][:, parameter_index] = (plus[t] - minus[t]) / (2.0 * eta)

        return sensitivities

    def coordinate_score(self, perturbed_coordinate, coordinate, sensitivity, lambda_):
        """Mean-field correction D_t^T s_t^lambda(y, z)."""
        score = self.transport_scores(perturbed_coordinate, coordinate, lambda_)
        return score @ sensitivity

    def trajectory_gradient(self, coordinates, sensitivities, seed, lambda_=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        base_state = self.initial_state(generator)
        state = self.initial_state(generator)
        base_rewards = []
        rewards = []
        score_terms = []

        for t in range(self.horizon):
            law = self.population_law(coordinates[t])
            base_action = self.sample_action(t, base_state, law, generator)
            randomizer = self.sample_transport_randomizers((), generator)
            perturbed_coordinate = self.transport_coordinate(coordinates[t], randomizer, lambda_)
            perturbed_law = self.population_law(perturbed_coordinate)
            action = self.sample_action(t, state, perturbed_law, generator)

            law_score = self.coordinate_score(perturbed_coordinate, coordinates[t], sensitivities[t], lambda_)
            action_score = self.log_prob_gradient(t, state, perturbed_law, action)
            score_terms.append(law_score + action_score)
            base_rewards.append(self.env.reward(base_state, law, base_action))
            rewards.append(self.env.reward(state, perturbed_law, action))

            with torch.no_grad():
                base_state = self.sample_next_state(t, base_state, law, base_action, generator)
                state = self.sample_next_state(t, state, perturbed_law, action, generator)

        terminal_law = self.population_law(coordinates[-1])
        randomizer = self.sample_transport_randomizers((), generator)
        perturbed_terminal_coordinate = self.transport_coordinate(coordinates[-1], randomizer, lambda_)
        perturbed_terminal_law = self.population_law(perturbed_terminal_coordinate)
        terminal_reward = self.env.terminal_reward(state, perturbed_terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_state, terminal_law)
        score_terms.append(
            self.coordinate_score(perturbed_terminal_coordinate, coordinates[-1], sensitivities[-1], lambda_)
        )

        returns = self.discounted_returns(rewards, terminal_reward)
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        return torch.stack(score_terms), torch.stack(returns), base_return

    def batched_trajectory_components(self, coordinates, seed, lambda_=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        lambda_ = self.lambda_ if lambda_ is None else lambda_

        base_states = self.initial_states(self.n_particles, generator)
        states = self.initial_states(self.n_particles, generator)
        base_rewards = []
        rewards = []
        action_log_probs = []
        perturbed_coordinates = []

        for t in range(self.horizon):
            law = self.population_law(coordinates[t])
            base_action, _ = self.sample_actions_with_log_probs(t, base_states, law, generator)
            randomizer = self.sample_transport_randomizers((self.n_particles,), generator)
            perturbed_coordinate = self.transport_coordinate(coordinates[t], randomizer, lambda_)
            perturbed_law = self.population_law(perturbed_coordinate)
            action, log_prob = self.sample_actions_with_log_probs(t, states, perturbed_law, generator)

            perturbed_coordinates.append(perturbed_coordinate)
            action_log_probs.append(log_prob)
            base_rewards.append(self.env.reward(base_states, law, base_action))
            rewards.append(self.env.reward(states, perturbed_law, action))

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, law, base_action, generator)
                states = self.sample_next_state(t, states, perturbed_law, action, generator)

        terminal_law = self.population_law(coordinates[-1])
        randomizer = self.sample_transport_randomizers((self.n_particles,), generator)
        perturbed_terminal_coordinate = self.transport_coordinate(coordinates[-1], randomizer, lambda_)
        perturbed_terminal_law = self.population_law(perturbed_terminal_coordinate)
        terminal_reward = self.env.terminal_reward(states, perturbed_terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_states, terminal_law)
        perturbed_coordinates.append(perturbed_terminal_coordinate)

        returns = torch.stack(self.discounted_returns(rewards, terminal_reward))
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean(dim=1, keepdim=True) if self.config.baseline else returns
        action_gradient = self.flat_grad((torch.stack(action_log_probs) * advantages[:-1].detach()).sum())
        return action_gradient, advantages.detach(), perturbed_coordinates, base_return.mean()

    def combine_batched_trajectory_components(
        self, action_gradient, weights, perturbed_coordinates, sensitivities, coordinates, lambda_=None
    ):
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        law_gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        for index, perturbed_coordinate in enumerate(perturbed_coordinates):
            law_score = self.coordinate_score(perturbed_coordinate, coordinates[index], sensitivities[index], lambda_)
            law_gradient = law_gradient + (law_score * weights[index].unsqueeze(-1)).sum(dim=0)
        return (action_gradient + law_gradient) / self.n_particles

    def batched_trajectory_gradient(self, coordinates, sensitivities, seed):
        action_gradient, weights, perturbed_coordinates, base_return = self.batched_trajectory_components(coordinates, seed)
        gradient = self.combine_batched_trajectory_components(
            action_gradient,
            weights,
            perturbed_coordinates,
            sensitivities,
            coordinates,
        )
        return gradient, base_return

    def estimate_gradient(self, seed):
        self.sensitivity_fallbacks = 0
        coordinates, _ = self.population_coordinates(seed=seed + 20_000, jacobians=False)

        if self.config.reuse_state_gradient:
            sensitivities = self.estimate_coordinate_sensitivities(coordinates, seed=seed + 10_000)
            return self.batched_trajectory_gradient(coordinates, sensitivities, seed)

        gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        scores = []
        returns = []
        objectives = []

        for particle in range(self.n_particles):
            sensitivities = self.estimate_coordinate_sensitivities(
                coordinates,
                seed=seed + 10_000 + particle,
            )
            score, trajectory_returns, objective = self.trajectory_gradient(
                coordinates, sensitivities, seed + particle
            )
            objectives.append(objective.detach())

            if self.config.baseline:
                scores.append(score)
                returns.append(trajectory_returns)
            else:
                gradient = gradient + (score * trajectory_returns.detach().unsqueeze(-1)).sum(dim=0)

        if self.config.baseline:
            scores = torch.stack(scores)
            returns = torch.stack(returns)
            returns = returns - returns.mean(dim=0, keepdim=True)
            gradient = (scores * returns.detach().unsqueeze(-1)).sum(dim=(0, 1))

        gradient = gradient / self.n_particles
        objective = torch.stack(objectives).mean()
        return gradient, objective

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        if hasattr(self.env, "objective"):
            return exact_continuous_validation_objective(self.env, self.policy)

        n_particles = self.n_particles if n_particles is None else n_particles
        horizon = getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon
        coordinates, _ = self.population_coordinates(horizon=horizon, seed=seed, jacobians=False)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed if seed is None else seed)

        states = self.env.sample_initial(n_particles, generator)
        rewards = []
        for t in range(horizon):
            law = self.population_law(coordinates[t])
            actions = self.sample_actions_for_population(t, states, law, generator)
            rewards.append(self.env.reward(states, law, actions))
            with torch.no_grad():
                states = self.sample_next_states_for_population(t, states, law, actions, generator)

        terminal = self.env.terminal_reward(states, self.population_law(coordinates[-1]))
        return self.discounted_returns(rewards, terminal)[0].mean()

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        optimizer = self.optimizer()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at
        history = {
            "objective": [],
            "validation_objective": [],
            "gradient_norm": [],
            "sensitivity_fallbacks": [],
            "train_step_seconds": [],
            "validation_seconds": [],
            "setup_seconds": [setup_seconds],
        }

        for episode in range(self.n_train):
            step_started_at = synchronized_time(self.env.device)
            gradient, objective = self.estimate_gradient(self.config.seed + episode * (self.n_particles + 1))

            optimizer.zero_grad()
            self.set_flat_gradient(-gradient)
            optimizer.step()
            objective_value = float(objective.detach().cpu())
            gradient_norm_value = float(gradient.norm().detach().cpu())
            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)

            history["objective"].append(objective_value)
            history["gradient_norm"].append(gradient_norm_value)
            history["sensitivity_fallbacks"].append(self.sensitivity_fallbacks)
            report_progress(episode, self.n_train, history)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                validation_value = float(validation.detach().cpu())
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(validation_value)

        return self.policy, history


def train_continuous_transport(env, policy=None, config=ContinuousTransportConfig()):
    return ContinuousTransport(env, policy=policy, config=config).train()
