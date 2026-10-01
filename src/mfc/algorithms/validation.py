"""Monte Carlo validation of a frozen policy with interacting particles.

Every estimator is validated the same way: M_val particles start from the
benchmark's fixed validation law, each reads the empirical law of the whole
system as its population argument, and the objective is the particle average of
the discounted returns. The seed is held fixed across the updates of a run, so
successive validations of one run differ only through the policy.

In finite state space the particles are exchangeable: those sharing a state
draw their actions from the same law, and those sharing a state and an action
draw their next states from the same law. The system is therefore simulated
through its counts, with multinomial draws per state and per state-action pair.
This has exactly the law of the particle simulation, at a cost independent of M_val.
"""

import torch

from .sampling import sample_accepts_time


def particle_law(env, states):
    """Population argument of the particle system: the empirical law, as each benchmark reads it."""
    if hasattr(env, "empirical_law"):
        return env.empirical_law(states)
    if states.dtype.is_floating_point:
        return states.mean()
    counts = torch.bincount(states, minlength=env.n_states)
    return counts.to(env.dtype) / states.numel()


def particle_actions(env, policy, policy_time, t, states, law, generator):
    if hasattr(env, "sample_action"):
        return env.sample_action(policy, t, states, law, generator)
    action_law = env.policy(policy, policy_time(t), states, law)
    if hasattr(action_law, "sample"):
        return action_law.sample()
    probabilities = action_law.clamp_min(1e-12)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    flat = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator)
    return flat.reshape(states.shape)


def particle_step(env, t, states, law, actions, generator):
    if sample_accepts_time(env.sample):
        return env.sample(states, law, actions, generator, t=t)
    return env.sample(states, law, actions, generator)


def multinomial_counts(totals, probabilities, generator):
    """Category counts of totals[...] draws from probabilities[..., :], by successive binomials."""
    remaining = totals
    mass = torch.ones_like(totals)
    counts = []
    for k in range(probabilities.shape[-1] - 1):
        share = (probabilities[..., k] / mass.clamp_min(torch.finfo(mass.dtype).tiny)).clamp(0.0, 1.0)
        draw = torch.binomial(remaining, share, generator=generator)
        counts.append(draw)
        remaining = remaining - draw
        mass = mass - probabilities[..., k]
    counts.append(remaining)
    return torch.stack(counts, dim=-1)


def finite_objective(env, policy, policy_time, discount, n_particles, horizon, generator):
    """The finite-state particle system, simulated through its state and state-action counts."""
    n_states, n_actions = env.n_states, env.n_actions
    states = torch.arange(n_states, device=env.device)
    pair_states = states.repeat_interleave(n_actions)
    pair_actions = torch.arange(n_actions, device=env.device).repeat(n_states)
    total = torch.tensor(float(n_particles), dtype=env.dtype, device=env.device)

    counts = multinomial_counts(total, env.initial_distribution.to(env.dtype), generator)
    value = torch.zeros((), dtype=env.dtype, device=env.device)
    factor = 1.0
    for t in range(horizon):
        law = counts / total
        probabilities = env.policy(policy, policy_time(t), states, law).clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        pair_counts = multinomial_counts(counts, probabilities, generator).reshape(-1)
        value = value + factor * (pair_counts * env.reward(pair_states, law, pair_actions)).sum() / total
        kernels = env.transition(pair_states, law, pair_actions)
        counts = multinomial_counts(pair_counts, kernels, generator).sum(dim=0)
        factor = factor * discount
    law = counts / total
    return value + factor * (counts * env.terminal_reward(states, law)).sum() / total


@torch.no_grad()
def monte_carlo_objective(env, policy, policy_time, discount, n_particles, horizon, seed):
    """Particle estimate of J(theta) from the benchmark's validation law."""
    generator = torch.Generator(device=env.device)
    generator.manual_seed(seed)
    if hasattr(env, "n_states"):
        return finite_objective(env, policy, policy_time, discount, n_particles, horizon, generator)
    if hasattr(env, "sample_initial"):
        states = env.sample_initial(n_particles, generator)
    else:
        states = torch.multinomial(env.initial_distribution, n_particles, replacement=True, generator=generator)

    value = torch.zeros(n_particles, dtype=env.dtype, device=env.device)
    factor = 1.0
    for t in range(horizon):
        law = particle_law(env, states)
        actions = particle_actions(env, policy, policy_time, t, states, law, generator)
        value = value + factor * env.reward(states, law, actions)
        states = particle_step(env, t, states, law, actions, generator)
        factor = factor * discount
    value = value + factor * env.terminal_reward(states, particle_law(env, states))
    return value.mean()
