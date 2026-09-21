import pandas as pd
import torch

from .constants import STATE_LABELS
from .io import load_env_and_policy


def policy_time(policy, env, t):
    if isinstance(policy, torch.nn.Module):
        return torch.tensor(float(t), dtype=env.dtype, device=env.device)
    return t


def discrete_next_law(env, policy, t, law):
    next_law = torch.zeros_like(law)
    with torch.no_grad():
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            action_probabilities = env.policy(policy, policy_time(policy, env, t), state, law)
            for action_index in range(env.n_actions):
                action = torch.tensor(action_index, dtype=torch.long, device=env.device)
                transition = env.transition(state, law, action)
                next_law = next_law + law[state_index] * action_probabilities[action_index] * transition
    return next_law / next_law.sum()


def discrete_law_flow(env, policy, horizon):
    laws = [env.initial_distribution.detach()]
    for t in range(horizon):
        laws.append(discrete_next_law(env, policy, t, laws[-1]).detach())
    return torch.stack(laws)


def continuous_moment_flow(env, policy, horizon):
    if hasattr(env, "moment_flow"):
        with torch.no_grad():
            means, variances = env.moment_flow(policy, lambda_=0.0)
        return torch.stack([means[: horizon + 1], variances[: horizon + 1]], dim=-1).detach()
    if hasattr(env, "empirical_law"):
        generator = torch.Generator(device=env.device)
        generator.manual_seed(getattr(env.config, "validation_seed", 0))
        n_particles = getattr(env.config, "n_flow_particles", getattr(env.config, "validation_particles", 1024))
        states = env.sample_initial(n_particles, generator)
        laws = [env.empirical_law(states)]
        with torch.no_grad():
            for t in range(horizon):
                law = laws[-1]
                actions = env.sample_action(policy, t, states, law, generator)
                states = env.sample(states, law, actions, generator, t=t)
                laws.append(env.empirical_law(states))
        return torch.stack(laws).detach()
    raise ValueError("Continuous moment flow is only available for environments exposing moment_flow.")


def learned_flow(run, horizon=None, device=None):
    env, policy = load_env_and_policy(run, device=device)
    horizon = horizon or run["metadata"]["horizon"]
    if hasattr(env, "n_states"):
        return discrete_law_flow(env, policy, horizon).cpu()
    return continuous_moment_flow(env, policy, horizon).cpu()


def final_policy_probabilities(env, policy, law=None, t=0):
    if not hasattr(env, "n_states"):
        raise ValueError("Policy probability tables are only defined for discrete action environments.")
    law = env.initial_distribution if law is None else law
    probabilities = []
    with torch.no_grad():
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            probabilities.append(env.policy(policy, policy_time(policy, env, t), state, law).detach().cpu())
    return torch.stack(probabilities)


def flow_dataframe(run):
    metadata = run["metadata"]
    flow = learned_flow(run)
    if metadata["env"] in {"lq", "portfolio"}:
        return pd.DataFrame(
            {
                "time": range(flow.shape[0]),
                "mean": flow[:, 0].numpy(),
                "variance": flow[:, 1].numpy(),
            }
        )
    labels = STATE_LABELS.get(metadata["env"], [str(i) for i in range(flow.shape[1])])
    df = pd.DataFrame(flow.numpy(), columns=labels)
    df["time"] = range(len(df))
    return df
