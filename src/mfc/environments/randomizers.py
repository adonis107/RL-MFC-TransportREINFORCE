"""How a randomized population law reaches a controlled particle.

The continuous benchmarks read the population only through its mean, so every
randomization reaches them as

    M_t = kappa(lambda) m_t + lambda B_t,

with m_t the represented flow and B_t the randomizer's own draw:

    additive   kappa = 1,        Var(B_t) = rho^2
    mixture    kappa = 1-lambda, Var(B_t) = sigma^2

`additive` is the analytic convention of the reference and belongs to no
estimator; it is the default so that existing gradients and sweeps are
unchanged. `mixture` is what both continuous estimators run: the mean is
carried to a random target, so the randomized law is centred on a shrunken
mean and the controlled particle no longer reproduces the flow it is shown.
That is why the moment recursions carry the represented flow and the particle's
own mean separately.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LawRandomizer:
    """Mean and variance of the randomized population mean at one time."""

    kind: str = "additive"
    sigma: float = 1.0

    def __post_init__(self):
        if self.kind not in {"additive", "mixture"}:
            raise ValueError(f"Unknown randomizer kind: {self.kind}")

    def mean_coefficient(self, lambda_):
        return 1.0 - lambda_ if self.kind == "mixture" else 1.0

    def variance(self, lambda_):
        return lambda_**2 * self.sigma**2


def resolve(randomizer, rho):
    """The randomizer to use, defaulting to the additive convention."""
    return LawRandomizer(kind="additive", sigma=rho) if randomizer is None else randomizer


def randomizer_for_run(metadata):
    """The randomizer a saved continuous-state run was trained with.

    None where no closed form exists: an unperturbed baseline, or a mixture
    chart with more than one component.
    """
    config = metadata.get("algorithm_config", {})
    algorithm = metadata.get("algorithm")
    if algorithm == "transport" and config.get("n_components", 1) != 1:
        return None
    if algorithm in {"transport", "gaussian"}:
        return LawRandomizer(kind="mixture", sigma=config.get("mean_randomizer_sigma", 1.0))
    return None
