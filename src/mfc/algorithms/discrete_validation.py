import torch


def mean_field_next_law(env, policy, policy_time, t, law):
    next_law = torch.zeros_like(law)
    with torch.no_grad():
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            action_probabilities = (policy.distribution(state).probs if getattr(policy, "is_stationary_actor", False)
                                    else env.policy(policy, policy_time(t), state, law))
            for action_index in range(env.n_actions):
                action = torch.tensor(action_index, dtype=torch.long, device=env.device)
                transition = env.transition(state, law, action)
                next_law = next_law + law[state_index] * action_probabilities[action_index] * transition
    return next_law / next_law.sum()


def expected_reward(env, policy, policy_time, t, law):
    value = torch.zeros((), dtype=env.dtype, device=env.device)
    with torch.no_grad():
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            action_probabilities = (policy.distribution(state).probs if getattr(policy, "is_stationary_actor", False)
                                    else env.policy(policy, policy_time(t), state, law))
            for action_index in range(env.n_actions):
                action = torch.tensor(action_index, dtype=torch.long, device=env.device)
                value = value + law[state_index] * action_probabilities[action_index] * env.reward(state, law, action)
    return value


def expected_terminal_reward(env, law):
    value = torch.zeros((), dtype=env.dtype, device=env.device)
    with torch.no_grad():
        for state_index in range(env.n_states):
            state = torch.tensor(state_index, dtype=torch.long, device=env.device)
            value = value + law[state_index] * env.terminal_reward(state, law)
    return value


def evaluate_law(env, policy, policy_time, discount, initial_distribution, horizon):
    law = initial_distribution
    value = torch.zeros((), dtype=env.dtype, device=env.device)
    discount_factor = 1.0
    for t in range(horizon):
        value = value + discount_factor * expected_reward(env, policy, policy_time, t, law)
        law = mean_field_next_law(env, policy, policy_time, t, law)
        discount_factor = discount_factor * discount
    return value + discount_factor * expected_terminal_reward(env, law)


def evaluate_initial_distributions(env, policy, policy_time, discount, initial_distributions, horizon):
    values = [
        evaluate_law(env, policy, policy_time, discount, initial_distribution, horizon)
        for initial_distribution in initial_distributions
    ]
    return torch.stack(values).mean()
