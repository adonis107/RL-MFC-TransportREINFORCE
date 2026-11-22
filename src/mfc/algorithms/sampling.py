import inspect


_SAMPLE_TIME_ARGUMENT = {}


def sample_accepts_time(sample):
    """Cache whether an environment's sample method takes a time argument."""
    function = getattr(sample, "__func__", sample)
    if function not in _SAMPLE_TIME_ARGUMENT:
        _SAMPLE_TIME_ARGUMENT[function] = "t" in inspect.signature(sample).parameters
    return _SAMPLE_TIME_ARGUMENT[function]


def independent_return_advantages(returns, dim=0):
    """Leave-one-out baseline for independent prescribed-flow trajectories.

    Including a trajectory in its own baseline shrinks the expected score
    estimate by (B-1)/B. Each baseline here uses only the other trajectories;
    a single trajectory uses zero as its baseline.
    """
    count = returns.shape[dim]
    if count < 2:
        return returns
    return (returns - returns.mean(dim=dim, keepdim=True)) * (count / (count - 1))
