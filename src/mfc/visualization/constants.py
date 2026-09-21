from mfc.environments import (
    Advertising,
    AdvertisingConfig,
    AdvertisingPolicy,
    Cybersecurity,
    CybersecurityConfig,
    CybersecurityPolicy,
    Distribution,
    DistributionConfig,
    DistributionPolicy,
    LQ,
    LQConfig,
    Portfolio,
    PortfolioConfig,
    TwoState,
    TwoStateConfig,
)


ENVIRONMENTS = {
    "advertising": (Advertising, AdvertisingConfig),
    "cybersecurity": (Cybersecurity, CybersecurityConfig),
    "distribution": (Distribution, DistributionConfig),
    "lq": (LQ, LQConfig),
    "portfolio": (Portfolio, PortfolioConfig),
    "twostate": (TwoState, TwoStateConfig),
}

POLICIES = {
    "advertising": AdvertisingPolicy,
    "cybersecurity": CybersecurityPolicy,
    "distribution": DistributionPolicy,
}

STATE_LABELS = {
    "advertising": ["not customer", "customer"],
    "cybersecurity": ["DI", "DS", "UI", "US"],
    "distribution": [str(i) for i in range(10)],
    "twostate": ["0", "1"],
}
