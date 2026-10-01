import inspect


_SAMPLE_TIME_ARGUMENT = {}


def sample_accepts_time(sample):
    """Cache whether an environment's sample method takes a time argument."""
    function = getattr(sample, "__func__", sample)
    if function not in _SAMPLE_TIME_ARGUMENT:
        _SAMPLE_TIME_ARGUMENT[function] = "t" in inspect.signature(sample).parameters
    return _SAMPLE_TIME_ARGUMENT[function]
