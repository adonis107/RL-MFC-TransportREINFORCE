"""Transport REINFORCE on the Gaussian manifold.

Under a Gaussian policy and linear dynamics the controlled law stays Gaussian,
so the population coordinate is the pair (m_t, sigma_t), perturbed as

    m_t^lambda     = (1 - lambda) m_t + lambda A_t,
    Sigma_t^lambda = ((1 - lambda) + lambda B_t)^2 sigma_t^2,

with one draw (A_t, B_t) per particle and per time. The mean randomization is
the one the K=1 mixture chart of Transport REINFORCE performs, so on a
benchmark that reads the population through its mean the two estimators share a
perturbed objective.

Writing U_t = (1 - lambda) + lambda B_t, the randomized coordinate has a
closed-form density whose score is

    grad log q_t = c_t^m grad m_t + c_t^sigma grad log sigma_t,
    c_t^m     = ((1 - lambda) / lambda) (A_t - mu_A) / sigma_A^2,
    c_t^sigma = (U_t / lambda) (B_t - mu_B) / sigma_B^2 - 1.

Both coefficients are known from the draw, so the only unknowns are the flow
sensitivities. Transport REINFORCE estimates the analogous object by centered
policy differences, costing 2 d_theta shifted sub-blocks. Here they come from a
likelihood ratio instead: the mean and second moment of the state at time t are
themselves objectives of the same control problem, so with
S_t = sum_{tau < t} (grad log pi_tau + grad log q_tau),

    grad m_t        = E[ X_t^lambda S_t ],
    grad E[(X_t)^2] = E[ (X_t^lambda)^2 S_t ],
    grad Var(X_t)   = grad E[(X_t)^2] - 2 m_t grad m_t,

and grad log sigma_t = grad Var(X_t) / (2 sigma_t^2). The scores in S_t involve
only times before t, so one forward pass resolves the recursion. The tau = 0
term vanishes, (m_0, sigma_0) being known and independent of theta.

No policy is ever shifted, so the estimator has one perturbation scale and no
auxiliary radius.
"""

from dataclasses import dataclass
import importlib

import torch
from torch import nn

from ..environments.randomizers import LawRandomizer
from .reinforce import exact_continuous_validation_objective
from .transport import sample_accepts_time
from .timing import report_progress, synchronized_time


@dataclass(frozen=True)
class GaussianTransportConfig:
    """Blocks and randomizer of the Gaussian-manifold estimator.

    n_particles is the main block B, n_law_gradient the auxiliary block n, and
    n_flow_particles the population block M; one update costs T (M + n + B).

    A_t is the random target of the mean, drawn over all of R; its defaults are
    those of the mixture chart, which is what puts the two continuous arms on
    one perturbed objective. B_t is the relative target of the standard
    deviation, truncated to B_t > -(1 - lambda)/lambda so that U_t stays
    positive and the randomized variance keeps a support independent of theta.
    The truncation constant depends on lambda alone and cancels in the score.
    """

    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_law_gradient: int | None = None
    n_flow_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    lambda_: float | None = None
    mean_randomizer_mean: float = 0.0
    mean_randomizer_sigma: float = 1.0
    scale_randomizer_mean: float = 1.0
    scale_randomizer_sigma: float = 0.5
    sigma_min: float = 1e-3
    flow: str = "particle"
    baseline: bool = True
    seed: int = 0


class GaussianTransport:
    """Transport REINFORCE with a probabilistic population-flow sensitivity."""

    def __init__(self, env, policy=None, config=GaussianTransportConfig()):
        if hasattr(env, "n_states"):
            raise TypeError("GaussianTransport is for continuous state spaces.")
        if not hasattr(env, "sample_initial"):
            raise TypeError("GaussianTransport requires an environment with sample_initial.")
        if config.flow not in {"exact", "particle"}:
            raise ValueError("flow must be either 'exact' or 'particle'.")
        if config.flow == "exact" and not hasattr(env, "moment_flow"):
            raise ValueError("flow='exact' requires an environment exposing moment_flow.")

        self.env = env
        self.config = config
        self.policy = self._make_policy() if policy is None else policy


    def _make_policy(self):
        if hasattr(self.env, "zero_policy"):
            return nn.Parameter(self.env.zero_policy())

        module = importlib.import_module(type(self.env).__module__)
        policy_class = getattr(module, f"{type(self.env).__name__}Policy")
        return policy_class(self.env.config)

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
    def n_flow_particles(self):
        return self.n_particles if self.config.n_flow_particles is None else self.config.n_flow_particles

    @property
    def n_population_particles(self):
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
        perturbation_scale = getattr(self.env.config, "perturbation_scale", 0.0)
        return perturbation_scale if perturbation_scale > 0.0 else 0.1

    @property
    def discount(self):
        if hasattr(self.env.config, "discount"):
            return self.env.config.discount
        return getattr(self.env.config, "gamma", 1.0)

    @property
    def n_parameters(self):
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    @property
    def randomizer(self):
        """The randomizer the environment's closed-form J^lambda sees.

        These benchmarks read the population through its mean alone, and the
        mean is carried to (1 - lambda) m_t + lambda A_t with A_t independent of
        the flow. That is the mixture chart's law, so the perturbed objective is
        the same one the zero-order transport arm optimizes.
        """
        return LawRandomizer(kind="mixture", sigma=self.config.mean_randomizer_sigma)


    def policy_time(self, t):
        if isinstance(self.policy, nn.Module):
            return torch.tensor(float(t), dtype=self.env.dtype, device=self.env.device)
        return t

    def sample_actions(self, t, states, law, generator):
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                return self.env.sample_action(self.policy, t, states, law, generator)
            return self.env.policy(self.policy, self.policy_time(t), states, law).sample()

    def sample_next_states(self, t, states, law, actions, generator):
        if sample_accepts_time(self.env.sample):
            return self.env.sample(states, law, actions, generator, t=t)
        return self.env.sample(states, law, actions, generator)

    def set_flat_gradient(self, gradient):
        offset = 0
        for parameter in self.trainable_parameters():
            next_offset = offset + parameter.numel()
            parameter.grad = gradient[offset:next_offset].reshape_as(parameter).clone()
            offset = next_offset

    def flatten_grads(self, grads):
        pieces = []
        for parameter, grad in zip(self.trainable_parameters(), grads):
            pieces.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.reshape(-1))
        return torch.cat(pieces)

    def flat_grad(self, value):
        grads = torch.autograd.grad(value, self.trainable_parameters(), allow_unused=True)
        return self.flatten_grads(grads)

    def policy_scores(self, t, states, laws, actions):
        """Per-sample grad_theta log pi_t, as an (n, d_theta) matrix."""
        t_policy = self.policy_time(t)

        def log_prob(parameters, state, law, action):
            return self.env.policy(parameters, t_policy, state, law).log_prob(action)

        parameters = [parameter.detach() for parameter in self.trainable_parameters()]
        if isinstance(self.policy, nn.Module):
            raise NotImplementedError(
                "GaussianTransport currently supports the tabular continuous-state policies."
            )
        grads = torch.func.vmap(torch.func.grad(log_prob), in_dims=(None, 0, 0, 0))(
            parameters[0], states, laws, actions
        )
        return grads.reshape(states.shape[0], -1)

    def discounted_returns(self, rewards, terminal_reward):
        values = [None] * (len(rewards) + 1)
        values[-1] = terminal_reward
        for t in range(len(rewards) - 1, -1, -1):
            values[t] = rewards[t] + self.discount * values[t + 1]
        return values


    def population_moments(self, seed):
        """The represented Gaussian flow (m_t, sigma_t) under the current policy.

        With flow='particle' this is step (1) of the algorithm: M trajectories
        of the unperturbed process, whose empirical mean and variance at every
        time are the coordinate. With flow='exact' the environment's analytic
        moment recursion is read at lambda = 0 instead.
        """
        if self.config.flow == "exact":
            with torch.no_grad():
                means, variances = self.env.moment_flow(self.policy, lambda_=0.0)
            means = means[: self.horizon + 1].detach()
            variances = variances[: self.horizon + 1].detach()
            return means, variances.clamp_min(self.config.sigma_min**2).sqrt()

        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        states = self.env.sample_initial(self.n_population_particles, generator)

        means, deviations = [], []
        for t in range(self.horizon + 1):
            with torch.no_grad():
                mean = states.mean()
                variance = states.var(unbiased=False)
            means.append(mean)
            deviations.append(variance.clamp_min(self.config.sigma_min**2).sqrt())
            if t == self.horizon:
                break
            actions = self.sample_actions(t, states, mean, generator)
            with torch.no_grad():
                states = self.sample_next_states(t, states, mean, actions, generator)

        return torch.stack(means), torch.stack(deviations)


    def sample_randomizers(self, n, generator, lambda_=None):
        """Draw the pair (A_t, B_t) of one time step, for every particle.

        A_t is the mean target and ranges over all of R. B_t is the relative
        scale target and is truncated to B_t > -(1 - lambda)/lambda, which is
        the support U_lambda: it keeps U_t = (1 - lambda) + lambda B_t positive,
        so the randomized variance never leaves R_+^*. The truncated normal is
        sampled by inverting its conditional distribution function, which is
        exact rather than a rejection loop.
        """
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        dtype, device = self.env.dtype, self.env.device
        config = self.config

        a = config.mean_randomizer_mean + config.mean_randomizer_sigma * torch.randn(
            n, dtype=dtype, device=device, generator=generator
        )

        standard = torch.distributions.Normal(
            torch.zeros((), dtype=dtype, device=device),
            torch.ones((), dtype=dtype, device=device),
        )
        lower = -(1.0 - lambda_) / lambda_
        lower_probability = standard.cdf(
            torch.tensor(
                (lower - config.scale_randomizer_mean) / config.scale_randomizer_sigma,
                dtype=dtype,
                device=device,
            )
        )
        uniform = torch.rand(n, dtype=dtype, device=device, generator=generator)
        uniform = lower_probability + (1.0 - lower_probability) * uniform
        uniform = uniform.clamp(torch.finfo(dtype).tiny, 1.0 - torch.finfo(dtype).eps)
        b = config.scale_randomizer_mean + config.scale_randomizer_sigma * standard.icdf(uniform)
        return a, b

    def score_coefficients(self, a, b, lambda_=None):
        """(c^sigma, c^m) of the Corollary, for one time step and every particle.

        The inverse maps of the density lemma return the draws themselves --
        a_t^{lambda,theta}(m, Sigma) = A_t and b_t^{lambda,theta}(m, Sigma) =
        B_t -- so the two score coefficients are read straight off the Gaussian
        scores of A_t and B_t. Both have mean zero, as the score must.
        """
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        config = self.config
        u = (1.0 - lambda_) + lambda_ * b
        c_mean = (
            (1.0 - lambda_)
            / lambda_
            * (a - config.mean_randomizer_mean)
            / config.mean_randomizer_sigma**2
        )
        c_sigma = (
            u / lambda_ * (b - config.scale_randomizer_mean) / config.scale_randomizer_sigma**2 - 1.0
        )
        return c_sigma, c_mean

    def perturbed_law(self, a, mean, lambda_=None):
        """The randomized population mean m_t^lambda = (1-lambda) m_t + lambda A_t."""
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        return (1.0 - lambda_) * mean + lambda_ * a


    def flow_sensitivities(self, means, deviations, seed):
        """grad_theta m_t and grad_theta log sigma_t by the likelihood ratio.

        The auxiliary block is one batch of n perturbed trajectories. The
        forward pass in t below is the triangular resolution described in the
        module docstring: at time t the running sum S carries
        sum_{tau < t} (grad log pi_tau + grad log q_tau), every term of which
        was built from sensitivities already available at that time. The
        centering of X_t and X_t^2 is the usual score control variate and
        leaves the estimator unbiased, because the running score has mean zero.
        """
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        n = self.n_law_gradient
        horizon = self.horizon

        states = self.env.sample_initial(n, generator)
        trajectory = []
        for t in range(horizon + 1):
            a, b = self.sample_randomizers(n, generator)
            law = self.perturbed_law(a, means[t])
            if t == horizon:
                trajectory.append((states, law, a, b, None))
                break
            actions = self.sample_actions(t, states, law, generator)
            trajectory.append((states, law, a, b, actions))
            with torch.no_grad():
                states = self.sample_next_states(t, states, law, actions, generator)

        zero = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        running = torch.zeros(n, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        mean_gradients, log_deviation_gradients = [zero], [zero]

        for t in range(horizon + 1):
            states, law, a, b, actions = trajectory[t]
            if t > 0:
                centered = states - states.mean()
                centered_square = states.square() - states.square().mean()
                mean_gradient = (centered.unsqueeze(-1) * running).mean(dim=0)
                square_gradient = (centered_square.unsqueeze(-1) * running).mean(dim=0)
                variance_gradient = square_gradient - 2.0 * means[t] * mean_gradient
                mean_gradients.append(mean_gradient)
                log_deviation_gradients.append(variance_gradient / (2.0 * deviations[t].square()))

            if t == horizon:
                break
            c_sigma, c_mean = self.score_coefficients(a, b)
            score = c_sigma.unsqueeze(-1) * log_deviation_gradients[t] + c_mean.unsqueeze(-1) * mean_gradients[t]
            running = running + score + self.policy_scores(t, states, law, actions)

        return mean_gradients, log_deviation_gradients


    def estimate_gradient(self, seed):
        means, deviations = self.population_moments(seed + 20_000)
        mean_gradients, log_deviation_gradients = self.flow_sensitivities(
            means, deviations, seed + 10_000
        )

        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        n = self.n_particles
        horizon = self.horizon

        base_states = self.env.sample_initial(n, generator)
        states = self.env.sample_initial(n, generator)
        base_rewards, rewards = [], []
        action_log_probs, coefficients = [], []

        terminal_law = None
        for t in range(horizon + 1):
            a, b = self.sample_randomizers(n, generator)
            law = self.perturbed_law(a, means[t])
            coefficients.append(self.score_coefficients(a, b))
            if t == horizon:
                terminal_law = law
                break
            base_action = self.sample_actions(t, base_states, means[t], generator)
            action = self.sample_actions(t, states, law, generator)
            action_log_probs.append(
                self.env.policy(self.policy, self.policy_time(t), states, law).log_prob(action)
            )
            base_rewards.append(self.env.reward(base_states, means[t], base_action))
            rewards.append(self.env.reward(states, law, action))
            with torch.no_grad():
                base_states = self.sample_next_states(t, base_states, means[t], base_action, generator)
                states = self.sample_next_states(t, states, law, action, generator)

        terminal_reward = self.env.terminal_reward(states, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_states, means[-1])

        returns = torch.stack(self.discounted_returns(rewards, terminal_reward))
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean(dim=1, keepdim=True) if self.config.baseline else returns
        advantages = advantages.detach()

        gradient = self.flat_grad((torch.stack(action_log_probs) * advantages[:-1]).sum()) / n
        for t in range(horizon + 1):
            c_sigma, c_mean = coefficients[t]
            gradient = gradient + (
                (advantages[t] * c_sigma).sum() * log_deviation_gradients[t]
                + (advantages[t] * c_mean).sum() * mean_gradients[t]
            ) / n

        return gradient, base_return.mean()

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        if hasattr(self.env, "objective"):
            return exact_continuous_validation_objective(self.env, self.policy)
        raise NotImplementedError("GaussianTransport validation needs a closed-form objective.")

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
            gradient, objective = self.estimate_gradient(
                self.config.seed + episode * (self.n_particles + 1)
            )

            optimizer.zero_grad()
            self.set_flat_gradient(-gradient)
            optimizer.step()
            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)
            history["objective"].append(float(objective.detach().cpu()))
            history["gradient_norm"].append(float(gradient.norm().detach().cpu()))
            report_progress(episode, self.n_train, history)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(float(validation.detach().cpu()))

        return self.policy, history


def train_gaussian_transport(env, policy=None, config=GaussianTransportConfig()):
    return GaussianTransport(env, policy=policy, config=config).train()
