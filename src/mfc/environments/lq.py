from dataclasses import dataclass
import torch

from .randomizers import resolve


@dataclass(frozen=True)
class LQConfig:
    a: float = 0.9
    b: float = 0.5
    c: float = 0.05
    q: float = 1.0
    r: float = 0.1
    gamma: float = 5.0
    q_T: float = 1.0
    gamma_T: float = 5.0
    tau: float = 0.2
    sigma: float = 0.1
    rho: float = 1.0
    perturbation_scale: float = 0.0
    mu0: float = 1.0
    Sigma0: float = 0.25
    T: int = 20
    discount: float = 1.0
    n_train: int = 10_000
    lr: float = 1e-3
    n_particles: int = 200
    validation_interval: int = 10
    dtype: torch.dtype = torch.float64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LQ:
    def __init__(self, config=LQConfig()):
        self.config = config
        self.dtype = config.dtype
        self.device = config.device
        self.n_params = 2
        self.initial_law = torch.tensor([config.mu0, config.Sigma0], dtype=self.dtype, device=self.device)

    def sample_initial(self, n_particles, generator):
        # Laplace noise keeps the same mean and variance as the oracle but makes the law non-Gaussian.
        uniform = torch.rand(n_particles, dtype=self.dtype, device=self.device, generator=generator) - 0.5
        scale = (0.5 * self.config.Sigma0) ** 0.5
        magnitude = (1.0 - 2.0 * uniform.abs()).clamp_min(torch.finfo(self.dtype).tiny)
        return self.config.mu0 - scale * uniform.sign() * torch.log(magnitude)

    def policy_mean(self, theta, t, states, mu_mean):
        return theta[t, 0] * states + theta[t, 1] * mu_mean

    def policy(self, theta, t, state, mu):
        return torch.distributions.Normal(self.policy_mean(theta, t, state, mu), self.config.tau)

    def sample_action(self, theta, t, states, mu_mean, generator):
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return self.policy_mean(theta, t, states, mu_mean) + self.config.tau * noise

    def transition(self, states, mu, actions):
        mean = self.config.a * states + self.config.b * actions + self.config.c * mu
        variance = torch.zeros_like(mean) + self.config.sigma**2
        return torch.stack([mean, variance], dim=-1)

    def sample(self, states, mu, actions, generator):
        mean = self.config.a * states + self.config.b * actions + self.config.c * mu
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return mean + self.config.sigma * noise

    def reward(self, states, mu, actions):
        return -(self.config.q * states.square() + self.config.r * actions.square() + self.config.gamma * mu.square())

    def terminal_reward(self, states, mu):
        return -(self.config.q_T * states.square() + self.config.gamma_T * mu.square())

    def flows(self, theta, lambda_=None, randomizer=None):
        """Represented flow m_t and the controlled particle's own moments.

        m_t is the unperturbed mean flow the population block fits; the
        particle is shown M_t with mean kappa m_t and variance v_t instead, so
        its own mean mu_t only coincides with m_t when kappa = 1. Returns
        (m, mu, Sigma), all of length T + 1.
        """
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        randomizer = resolve(randomizer, self.config.rho)
        kappa = randomizer.mean_coefficient(lambda_)

        represented = [torch.tensor(self.config.mu0, dtype=self.dtype, device=self.device)]
        mean = [torch.tensor(self.config.mu0, dtype=self.dtype, device=self.device)]
        variance = [torch.tensor(self.config.Sigma0, dtype=self.dtype, device=self.device)]

        for t in range(self.config.T):
            theta1, theta2 = theta[t]
            law_variance = randomizer.variance(lambda_)
            represented.append(
                (self.config.a + self.config.b * theta1 + self.config.b * theta2 + self.config.c) * represented[t]
            )
            mean.append(
                (self.config.a + self.config.b * theta1) * mean[t]
                + (self.config.b * theta2 + self.config.c) * kappa * represented[t]
            )
            variance.append(
                (self.config.a + self.config.b * theta1).square() * variance[t]
                + (self.config.b * theta2 + self.config.c).square() * law_variance
                + self.config.b**2 * self.config.tau**2
                + self.config.sigma**2
            )

        return represented, mean, variance

    def moment_flow(self, theta, lambda_=None, randomizer=None):
        """The controlled particle's mean and variance flow.

        At lambda = 0, and for any randomizer with kappa = 1, this is also the
        represented flow; use flows() when the two have to be told apart.
        """
        _, mean, variance = self.flows(theta, lambda_, randomizer)
        return torch.stack(mean), torch.stack(variance)

    def objective(self, theta, lambda_=None, randomizer=None):
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        resolved = resolve(randomizer, self.config.rho)
        kappa = resolved.mean_coefficient(lambda_)
        represented, mu, Sigma = self.flows(theta, lambda_, randomizer)
        objective = torch.zeros((), dtype=self.dtype, device=self.device)

        for t in range(self.config.T):
            theta1, theta2 = theta[t]
            # Second moments of the two quantities the cost sees: the state and
            # the randomized population mean, which are independent.
            law_mean = kappa * represented[t]
            law_second = law_mean.square() + resolved.variance(lambda_)
            state_second = Sigma[t] + mu[t].square()
            objective = objective + self.config.q * state_second
            objective = objective + self.config.r * (
                theta1.square() * state_second
                + theta2.square() * law_second
                + 2.0 * theta1 * theta2 * mu[t] * law_mean
                + self.config.tau**2
            )
            objective = objective + self.config.gamma * law_second

        terminal_mean = kappa * represented[-1]
        terminal_second = terminal_mean.square() + resolved.variance(lambda_)
        objective = objective + self.config.q_T * (Sigma[-1] + mu[-1].square())
        objective = objective + self.config.gamma_T * terminal_second
        return objective

    def exact_gradient(self, theta, lambda_=None):
        """Analytic gradient of the objective under the additive convention.

        This adjoint recursion is written for the reference's additive
        randomization, Var(M_t) = lambda^2 rho^2 with the mean left alone. It
        takes no randomizer: for any other one, differentiate objective()
        directly, which is what the estimators and scripts/decomposition.py do.
        """
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        mu, Sigma = self.moment_flow(theta, lambda_)
        p = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)
        s = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)

        p[-1] = 2.0 * (self.config.q_T + self.config.gamma_T) * mu[-1]
        s[-1] = self.config.q_T

        for t in range(self.config.T - 1, -1, -1):
            theta1, theta2 = theta[t]
            p[t] = (
                2.0 * (self.config.q + self.config.r * (theta1 + theta2).square() + self.config.gamma) * mu[t]
                + (self.config.a + self.config.b * theta1 + self.config.b * theta2 + self.config.c) * p[t + 1]
            )
            s[t] = self.config.q + self.config.r * theta1.square() + (
                self.config.a + self.config.b * theta1
            ).square() * s[t + 1]

        grad = torch.empty_like(theta)
        for t in range(self.config.T):
            theta1, theta2 = theta[t]
            grad[t, 0] = (
                2.0 * self.config.r * (theta1 + theta2) * mu[t].square()
                + 2.0 * self.config.r * theta1 * Sigma[t]
                + self.config.b * mu[t] * p[t + 1]
                + 2.0 * self.config.b * (self.config.a + self.config.b * theta1) * Sigma[t] * s[t + 1]
            )
            grad[t, 1] = (
                2.0 * self.config.r * (theta1 + theta2) * mu[t].square()
                + 2.0 * self.config.r * theta2 * lambda_**2 * self.config.rho**2
                + self.config.b * mu[t] * p[t + 1]
                + 2.0
                * self.config.b
                * (self.config.b * theta2 + self.config.c)
                * lambda_**2
                * self.config.rho**2
                * s[t + 1]
            )

        return grad

    def zero_policy(self):
        return torch.zeros(self.config.T, self.n_params, dtype=self.dtype, device=self.device)

    def optimal_theta(self):
        P = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)
        R = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)
        P[-1] = self.config.q_T
        R[-1] = self.config.q_T + self.config.gamma_T

        for t in range(self.config.T - 1, -1, -1):
            P[t] = self.config.q + self.config.r * self.config.a**2 * P[t + 1] / (
                self.config.r + self.config.b**2 * P[t + 1]
            )
            R[t] = self.config.q + self.config.gamma + self.config.r * (self.config.a + self.config.c) ** 2 * R[
                t + 1
            ] / (self.config.r + self.config.b**2 * R[t + 1])

        theta = torch.empty(self.config.T, self.n_params, dtype=self.dtype, device=self.device)
        for t in range(self.config.T):
            k = -self.config.a * self.config.b * P[t + 1] / (self.config.r + self.config.b**2 * P[t + 1])
            ell = -self.config.b * (self.config.a + self.config.c) * R[t + 1] / (
                self.config.r + self.config.b**2 * R[t + 1]
            )
            theta[t, 0] = k
            theta[t, 1] = ell - k

        return theta

    def optimal_policy(self):
        return self.optimal_theta()
