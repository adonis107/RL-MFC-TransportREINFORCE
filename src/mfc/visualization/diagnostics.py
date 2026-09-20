from dataclasses import replace

import pandas as pd
import torch

from mfc.algorithms import (
    ContinuousTransport,
    ContinuousTransportConfig,
    DiscreteTransport,
    DiscreteTransportConfig,
)

from .io import load_env_and_policy, run_label


CONTINUOUS_ENVS = {"lq", "portfolio", "kuramoto"}
IDENTIFICATION_COMPONENTS = (1, 2, 3)
IDENTIFICATION_FLOORS = (1e-4, 1e-2, 1e-1)


def vector_std_norm(values):
    if values.shape[0] < 2:
        return torch.full((), float("nan"), dtype=values.dtype)
    return values.std(dim=0, unbiased=True).norm()


def z_ratio(signal, standard_error):
    """Return a norm-to-SE ratio.

    For vector norms this is not a conventional signed z-score: under a zero
    mean error, its root-mean-square null baseline is about 1 rather than 0.
    Prefer the accompanying chi-square columns for a null-aware readout.
    """
    signal = torch.as_tensor(signal)
    standard_error = torch.as_tensor(standard_error)
    if not torch.isfinite(standard_error).item() or standard_error.item() <= 0:
        return float("nan")
    return float(signal / standard_error)


def norm_chi_square(signal, standard_error, dimension):
    """Approximate a vector norm-to-SE ratio as a chi-square statistic."""
    ratio = z_ratio(signal, standard_error)
    if dimension <= 0 or not torch.isfinite(torch.tensor(ratio)).item():
        return float("nan"), float("nan")

    statistic = dimension * ratio**2
    pvalue = torch.special.gammaincc(
        torch.tensor(0.5 * dimension, dtype=torch.float64),
        torch.tensor(0.5 * statistic, dtype=torch.float64),
    )
    return float(statistic), float(pvalue)


def reward_gradient(env, policy, lambda_):
    gradient = env.exact_gradient(policy, lambda_=lambda_)
    if type(env).__name__ == "LQ":
        return -gradient
    return gradient


def safe_scalar_ratio(numerator, denominator):
    numerator = torch.as_tensor(numerator)
    denominator = torch.as_tensor(denominator)
    if not torch.isfinite(denominator).item() or denominator.item() <= 0:
        return float("nan")
    return float(numerator / denominator)


def build_estimator(run, seed, n_particles=None, baseline=True, **overrides):
    """Rebuild the estimator of a saved transport run, for evaluation only."""
    env, policy = load_env_and_policy(run)
    metadata = run["metadata"]
    algorithm_config = metadata["algorithm_config"]
    if metadata["algorithm"] != "transport":
        raise ValueError("Transport diagnostics are defined for transport runs.")

    common = dict(
        n_particles=n_particles or algorithm_config.get("n_particles") or env.config.n_particles,
        lambda_=metadata["perturbation"],
        eta=algorithm_config.get("eta"),
        horizon=metadata["horizon"],
        flow=metadata["flow"],
        n_flow_particles=algorithm_config.get("n_flow_particles"),
        baseline=baseline,
        reuse_state_gradient=algorithm_config.get("reuse_state_gradient", True),
        seed=seed,
    )
    if metadata["env"] in CONTINUOUS_ENVS:
        config = ContinuousTransportConfig(
            **common,
            n_law_gradient=algorithm_config.get("n_law_gradient"),
            n_law_particles=algorithm_config.get("n_law_particles"),
            n_components=algorithm_config.get("n_components") or ContinuousTransportConfig.n_components,
        )
        return env, policy, ContinuousTransport(env, policy=policy, config=replace(config, **overrides))

    config = DiscreteTransportConfig(
        **common,
        n_logit_gradient=algorithm_config.get("n_logit_gradient"),
        simplex_sigma=algorithm_config.get("simplex_sigma", DiscreteTransportConfig.simplex_sigma),
    )
    return env, policy, DiscreteTransport(env, policy=policy, config=replace(config, **overrides))


def mixture_identification(estimator, seed):
    """Conditioning of the mixture score Jacobian A along the represented flow.

    A is inverted once per time step from t = 1 on, so the summary covers those
    matrices. A singular direction below the estimator's floor is dropped by the
    sensitivity solve, and the retained fraction is therefore how much of the
    chart the population law actually identifies: it falls well below one exactly
    when K asks for more components than the law needs.
    """
    _, jacobians = estimator.population_coordinates(seed=seed)
    spectra = [torch.linalg.svdvals(jacobian) for jacobian in jacobians[1:]]
    if not spectra:
        return {}

    floor = estimator.config.jacobian_floor
    largest = torch.stack([spectrum[0] for spectrum in spectra])
    smallest = torch.stack([spectrum[-1] for spectrum in spectra])
    conditions = largest / smallest.clamp_min(torch.finfo(largest.dtype).tiny)
    retained = sum(int((spectrum > floor * spectrum[0]).sum()) for spectrum in spectra)
    return {
        "n_components": estimator.config.n_components,
        "coordinate_dim": estimator.coordinate_dim,
        "jacobian_floor": floor,
        "jacobian_condition_median": float(conditions.median()),
        "jacobian_condition_max": float(conditions.max()),
        "retained_direction_fraction": retained / (estimator.coordinate_dim * len(spectra)),
    }


def gradient_diagnostics(run, n_replications=20, seed=0, compare_to_unperturbed=True, n_particles=None):
    metadata = run["metadata"]
    env, policy, estimator = build_estimator(run, seed, n_particles=n_particles)
    algorithm_config = metadata["algorithm_config"]
    lambda_ = metadata["perturbation"]
    if not hasattr(env, "exact_gradient"):
        raise ValueError("This environment does not expose exact_gradient.")
    if isinstance(policy, torch.nn.Module):
        raise ValueError("Exact gradient diagnostics currently require direct tensor policies.")

    config = estimator.config
    estimates = []
    for index in range(n_replications):
        gradient, _ = estimator.estimate_gradient(seed + index * 100_000)
        estimates.append(gradient.detach().reshape(-1).cpu())

    estimates = torch.stack(estimates)
    # The continuous environments' analytic lambda-oracles model the older
    # affine mean perturbation, not the mixture-transport perturbation from the
    # reference. For continuous transport diagnostics, compare to the unperturbed
    # gradient, which is the quantity the bias theorem bounds.
    reference_lambda = 0.0 if isinstance(estimator, ContinuousTransport) else lambda_
    exact = reward_gradient(env, policy, lambda_=reference_lambda).detach().reshape(-1).cpu()
    error = estimates - exact
    mean_estimate = estimates.mean(dim=0)
    bias_norm = (mean_estimate - exact).norm()
    estimate_std = vector_std_norm(estimates)
    estimate_se = estimate_std / n_replications**0.5
    n_parameters = exact.numel()
    bias_chi2_stat, bias_chi2_pvalue = norm_chi_square(bias_norm, estimate_se, n_parameters)
    exact_norm = exact.norm()

    row = {
        "env": metadata["env"],
        "label": run_label(metadata),
        "flow": metadata["flow"],
        "horizon": metadata["horizon"],
        "lambda": lambda_,
        "reference_lambda": reference_lambda,
        "eta": algorithm_config.get("eta"),
        "n_parameters": n_parameters,
        "diagnostic_n_particles": estimator.n_particles,
        "n_replications": n_replications,
        "bias_norm": float(bias_norm),
        "estimate_std": float(estimate_std),
        "estimate_se": float(estimate_se),
        "estimate_se_to_exact_norm": safe_scalar_ratio(estimate_se, exact_norm),
        "estimate_snr": safe_scalar_ratio(exact_norm, estimate_se),
        "bias_z": z_ratio(bias_norm, estimate_se),
        "bias_z_null_rms": 1.0,
        "bias_chi2_stat": bias_chi2_stat,
        "bias_chi2_df": n_parameters,
        "bias_chi2_pvalue": bias_chi2_pvalue,
        "mse": float(error.square().mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(mean_estimate, exact, dim=0)),
        "exact_gradient_norm": float(exact_norm),
        "estimate_gradient_norm_mean": float(estimates.norm(dim=1).mean()),
    }
    if isinstance(estimator, ContinuousTransport):
        row.update(mixture_identification(estimator, seed))
    if compare_to_unperturbed:
        # Comparing the reference to the unperturbed gradient only says something
        # when the reference is the perturbed one; otherwise the two coincide and
        # the perturbation bias is simply not available in closed form.
        if reference_lambda == lambda_:
            exact_zero = reward_gradient(env, policy, lambda_=0.0).detach().reshape(-1).cpu()
            row["perturbation_bias_norm"] = float((exact - exact_zero).norm())
        else:
            row["perturbation_bias_norm"] = float("nan")

    return pd.DataFrame([row])


def transport_correction_table(run, n_replications=20, n_particles=None, seed=0):
    metadata = run["metadata"]
    _, _, estimator = build_estimator(run, seed, n_particles=n_particles, baseline=False)
    lambda_ = metadata["perturbation"]
    eta = estimator.eta

    rows = []
    full_gradients = []
    policy_gradients = []
    corrections = []
    for replication in range(n_replications):
        base_seed = seed + replication * 100_000

        if isinstance(estimator, ContinuousTransport):
            flow, _ = estimator.population_coordinates(seed=base_seed + 20_000, jacobians=False)
            sensitivities = estimator.estimate_coordinate_sensitivities(flow, seed=base_seed + 10_000)
        else:
            flow, _ = estimator.mean_field_law_flow(seed=base_seed + 20_000)
            sensitivities = estimator.estimate_state_sensitivities(flow, base_seed + 10_000)

        # Both gradients share the seed, so they share one rollout: only the
        # sensitivities fed to the law-score term differ.
        zeros = [torch.zeros_like(sensitivity) for sensitivity in sensitivities]
        components = estimator.batched_trajectory_components(flow, base_seed)
        action_gradient, weights, law_terms, _ = components
        if isinstance(estimator, ContinuousTransport):
            full_gradient = estimator.combine_batched_trajectory_components(
                action_gradient, weights, law_terms, sensitivities, flow
            )
            policy_gradient = estimator.combine_batched_trajectory_components(
                action_gradient, weights, law_terms, zeros, flow
            )
        else:
            full_gradient = estimator.combine_batched_trajectory_components(
                action_gradient, weights, law_terms, sensitivities
            )
            policy_gradient = estimator.combine_batched_trajectory_components(
                action_gradient, weights, law_terms, zeros
            )

        full_gradient = full_gradient.detach().cpu()
        policy_gradient = policy_gradient.detach().cpu()
        correction = full_gradient - policy_gradient
        full_gradients.append(full_gradient)
        policy_gradients.append(policy_gradient)
        corrections.append(correction)
        rows.append(
            {
                "env": metadata["env"],
                "label": run_label(metadata),
                "flow": metadata["flow"],
                "horizon": metadata["horizon"],
                "seed": metadata["seed"],
                "lambda": lambda_,
                "eta": eta,
                "replication": replication,
                "full_gradient_norm": float(full_gradient.norm()),
                "policy_only_gradient_norm": float(policy_gradient.norm()),
                "correction_norm": float(correction.norm()),
                "correction_fraction": float(correction.norm() / full_gradient.norm().clamp_min(1e-12)),
                "full_policy_cosine": float(torch.nn.functional.cosine_similarity(full_gradient, policy_gradient, dim=0)),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    full_gradients = torch.stack(full_gradients)
    policy_gradients = torch.stack(policy_gradients)
    corrections = torch.stack(corrections)
    correction_mean = corrections.mean(dim=0)
    correction_std = vector_std_norm(corrections)
    correction_se = correction_std / n_replications**0.5
    full_mean = full_gradients.mean(dim=0)
    policy_mean = policy_gradients.mean(dim=0)
    n_parameters = correction_mean.numel()
    correction_chi2_stat, correction_chi2_pvalue = norm_chi_square(
        correction_mean.norm(),
        correction_se,
        n_parameters,
    )

    table["n_parameters"] = n_parameters
    table["full_gradient_mean_norm"] = float(full_mean.norm())
    table["policy_only_gradient_mean_norm"] = float(policy_mean.norm())
    table["correction_mean_norm"] = float(correction_mean.norm())
    table["correction_std"] = float(correction_std)
    table["correction_se"] = float(correction_se)
    table["correction_z"] = z_ratio(correction_mean.norm(), correction_se)
    table["correction_z_null_rms"] = 1.0
    table["correction_chi2_stat"] = correction_chi2_stat
    table["correction_chi2_df"] = n_parameters
    table["correction_chi2_pvalue"] = correction_chi2_pvalue
    table["correction_mean_fraction"] = float(correction_mean.norm() / full_mean.norm().clamp_min(1e-12))
    table["full_policy_mean_cosine"] = float(torch.nn.functional.cosine_similarity(full_mean, policy_mean, dim=0))
    return table


def identification_sweep(
    run,
    components=IDENTIFICATION_COMPONENTS,
    floors=IDENTIFICATION_FLOORS,
    n_replications=20,
    n_particles=None,
    seed=0,
):
    """Mixture size and identification floor swept at the policy a run saved.

    Two questions are answered on the same reference policy, and neither needs
    extra training. How many components does the population law identify, read
    off the conditioning of A and the fraction of chart directions the floor
    keeps? And how much does the floor itself matter, read off the dispersion of
    the estimator and, where an environment has a gradient oracle, its bias?
    """
    metadata = run["metadata"]
    if metadata["env"] not in CONTINUOUS_ENVS:
        raise ValueError("The identification sweep is defined for continuous-state runs.")

    rows = []
    for n_components in components:
        for floor in floors:
            env, policy, estimator = build_estimator(
                run,
                seed,
                n_particles=n_particles,
                n_components=n_components,
                jacobian_floor=floor,
            )
            estimates = torch.stack(
                [
                    estimator.estimate_gradient(seed + index * 100_000)[0].detach().reshape(-1).cpu()
                    for index in range(n_replications)
                ]
            )
            mean_estimate = estimates.mean(dim=0)
            estimate_std = vector_std_norm(estimates)
            row = {
                "env": metadata["env"],
                "label": run_label(metadata),
                "flow": metadata["flow"],
                "horizon": metadata["horizon"],
                "seed": metadata["seed"],
                "lambda": metadata["perturbation"],
                "eta": estimator.eta,
                "n_replications": n_replications,
                "diagnostic_n_particles": estimator.n_particles,
                **mixture_identification(estimator, seed),
                "estimate_gradient_norm_mean": float(estimates.norm(dim=1).mean()),
                "estimate_std": float(estimate_std),
                "estimate_se": float(estimate_std / n_replications**0.5),
            }

            # Kuramoto has no gradient oracle and a module policy, so there the
            # sweep reports conditioning and dispersion only.
            if hasattr(env, "exact_gradient") and not isinstance(policy, torch.nn.Module):
                exact = reward_gradient(env, policy, lambda_=0.0).detach().reshape(-1).cpu()
                bias_norm = (mean_estimate - exact).norm()
                row["exact_gradient_norm"] = float(exact.norm())
                row["bias_norm"] = float(bias_norm)
                row["relative_bias"] = safe_scalar_ratio(bias_norm, exact.norm())
                row["relative_se"] = safe_scalar_ratio(estimate_std / n_replications**0.5, exact.norm())
                row["cosine_similarity"] = float(
                    torch.nn.functional.cosine_similarity(mean_estimate, exact, dim=0)
                )
            rows.append(row)

    return pd.DataFrame(rows)
