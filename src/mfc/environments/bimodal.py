from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BimodalConfig:
    """Bimodal population allocation.

    X_0 ~ 1/2 N(-1, s_0) + 1/2 N(1, s_0), A | x ~ N(theta x, v_pi(theta)) and
    X_1 | a ~ N(a, s_P), with v_pi(theta) = 1 - s_P - (1 + s_0) theta^2 so that
    the terminal law 1/2 N(-theta, 1 - theta^2) + 1/2 N(theta, 1 - theta^2) has
    mean zero and variance one for every theta. The terminal reward reads the
    population only through its Gaussian-kernel mass near zero.
    """

    T: int = 1
    initial_variance: float = 0.01
    transition_variance: float = 0.01
    kernel_width: float = 0.2
    target_mass: float = 0.1
    penalty: float = 100.0
    theta_min: float = 0.75
    theta_max: float = 0.95
    theta_init: float = 0.8
    # Floor on the policy variance, so that a shifted parameter outside the
    # admissible set still defines a policy. Inactive on [-0.99, 0.99].
    policy_variance_floor: float = 1e-6
    discount: float = 1.0
    n_train: int = 1_000
    lr: float = 1e-3
    n_particles: int = 100_000
    validation_interval: int = 10
    validation_particles: int = 100_000
    dtype: torch.dtype = torch.float64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Bimodal:
    # The policy parameter may carry a trailing particle dimension, read elementwise:
    # the transport estimator simulates its shifted systems side by side this way.
    per_particle_parameters = True

    def __init__(self, config=BimodalConfig()):
        if config.T != 1:
            raise ValueError("The bimodal benchmark is defined for a single time step, T = 1.")
        self.config = config
        self.dtype = config.dtype
        self.device = config.device
        self.n_params = 1

    def policy_variance(self, theta):
        config = self.config
        variance = 1.0 - config.transition_variance - (1.0 + config.initial_variance) * theta.square()
        return variance.clamp_min(config.policy_variance_floor)

    def sample_initial(self, n_particles, generator):
        signs = 2.0 * (torch.rand(n_particles, dtype=self.dtype, device=self.device, generator=generator) < 0.5) - 1.0
        noise = torch.randn(n_particles, dtype=self.dtype, device=self.device, generator=generator)
        return signs + self.config.initial_variance**0.5 * noise

    def policy_mean(self, theta, t, states, mu):
        return theta[0] * states

    def policy(self, theta, t, state, mu):
        return torch.distributions.Normal(self.policy_mean(theta, t, state, mu), self.policy_variance(theta[0]).sqrt())

    def sample_action(self, theta, t, states, mu, generator):
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return self.policy_mean(theta, t, states, mu) + self.policy_variance(theta[0]).sqrt() * noise

    def sample(self, states, mu, actions, generator, t=None):
        noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
        return actions + self.config.transition_variance**0.5 * noise

    def kernel(self, states):
        return torch.exp(-states.square() / (2.0 * self.config.kernel_width**2))

    def kernel_mass(self, means, variances):
        """Integral of the kernel against N(mean, variance), in closed form."""
        width2 = self.config.kernel_width**2
        return (width2 / (width2 + variances)).sqrt() * torch.exp(-means.square() / (2.0 * (width2 + variances)))

    # The population argument handed to the policy and rewards is the scalar
    # m -> int exp(-y^2 / (2 w^2)) m(dy). REINFORCE reads it off its particles;
    # the mixture chart integrates it exactly against every component.
    def empirical_law(self, states):
        return self.kernel(states).mean(dim=-1).detach()

    def state_law_features(self, states):
        return self.kernel(states).unsqueeze(-1)

    def mixture_law_features(self, weights, means, covariances):
        mass = self.kernel_mass(means[..., 0], covariances[..., 0, 0])
        return (weights * mass).sum(dim=-1, keepdim=True)

    def reward(self, states, mu, actions):
        return torch.zeros_like(states, dtype=self.dtype)

    def terminal_reward(self, states, mu):
        value = -self.config.penalty * (mu - self.config.target_mass).square()
        return torch.zeros_like(states, dtype=self.dtype) + value

    def terminal_law(self, theta):
        """Component means and common variance of mu_1^theta, a mixture of N(-+theta, v)."""
        theta = theta[0]
        variance = self.config.initial_variance * theta.square() + self.policy_variance(theta) + self.config.transition_variance
        return torch.stack([-theta, theta]), variance

    def objective(self, theta, lambda_=None, randomizer=None):
        """Exact J(theta) of the unperturbed problem."""
        if lambda_:
            raise NotImplementedError("The bimodal benchmark has no closed form for the perturbed objective.")
        means, variance = self.terminal_law(theta)
        mass = self.kernel_mass(means, variance).mean()
        return -self.config.penalty * (mass - self.config.target_mass).square()

    def project_policy(self, theta):
        """Keep the parameter in the admissible set Theta after an update."""
        with torch.no_grad():
            theta.clamp_(self.config.theta_min, self.config.theta_max)

    def zero_policy(self):
        # The algorithms start from zero_policy(); here that is theta_init, not zero.
        return torch.full((self.n_params,), self.config.theta_init, dtype=self.dtype, device=self.device)

    def optimal_theta(self):
        """The unique maximizer on Theta, where the kernel mass equals its target.

        The mass decreases in theta on Theta, so bisection finds the root.
        """
        low, high = self.config.theta_min, self.config.theta_max

        def excess(value):
            means, variance = self.terminal_law(torch.tensor([value], dtype=self.dtype, device=self.device))
            return float(self.kernel_mass(means, variance).mean()) - self.config.target_mass

        if excess(low) * excess(high) > 0.0:
            raise ValueError("The kernel mass does not cross its target on Theta.")
        for _ in range(200):
            middle = 0.5 * (low + high)
            if excess(low) * excess(middle) <= 0.0:
                high = middle
            else:
                low = middle
        return torch.tensor([0.5 * (low + high)], dtype=self.dtype, device=self.device)

    def optimal_policy(self):
        return self.optimal_theta()

    def single_gaussian_objective(self):
        """J_1: the reward evaluated on the moment-matched Gaussian N(0, 1)."""
        one = torch.ones((), dtype=self.dtype, device=self.device)
        mass = self.kernel_mass(0.0 * one, one)
        return -self.config.penalty * (mass - self.config.target_mass).square()

    def terminal_density(self, theta, points):
        """Density of mu_1^theta at the given points."""
        means, variance = self.terminal_law(theta)
        normal = torch.exp(-(points.unsqueeze(-1) - means).square() / (2.0 * variance))
        return (normal / math.sqrt(2.0 * math.pi) / variance.sqrt()).mean(dim=-1)
