from pathlib import Path

import pandas as pd
import torch

from mfc.environments.randomizers import randomizer_for_run

from .flows import discrete_law_flow
from .flows import final_policy_probabilities
from .io import load_env_and_policy, run_label, runs_dataframe


CONTINUOUS_TRANSPORT_ENVS = {"lq", "portfolio"}


def optimize_exact_policy(env, lambda_, randomizer=None):
    if not hasattr(env, "objective") or not hasattr(env, "optimal_policy"):
        raise NotImplementedError("Exact policy optimization requires objective and optimal_policy.")

    theta = env.optimal_policy().detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    minimize = type(env).__name__ == "LQ"

    def closure():
        optimizer.zero_grad()
        value = env.objective(theta, lambda_=lambda_, randomizer=randomizer)
        loss = value if minimize else -value
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach()


def twostate_policy_error_table(runs):
    rows = []
    for run in runs:
        if run["metadata"]["env"] != "twostate":
            continue
        env, policy = load_env_and_policy(run)
        learned = final_policy_probabilities(env, policy)
        optimal = env.optimal_policy().detach().cpu()
        error = (learned - optimal).abs()
        rows.append(
            {
                "label": run_label(run["metadata"]),
                "flow": run["metadata"]["flow"],
                "horizon": run["metadata"]["horizon"],
                "eta": run["metadata"].get("eta"),
                "seed": run["metadata"]["seed"],
                "mean_abs_policy_error": float(error.mean()),
                "max_abs_policy_error": float(error.max()),
            }
        )
    return pd.DataFrame(rows)


def advertising_policy_error_table(runs):
    rows = []
    for run in runs:
        if run["metadata"]["env"] != "advertising":
            continue
        env, policy = load_env_and_policy(run)
        flow = discrete_law_flow(env, policy, run["metadata"]["horizon"]).to(env.device)
        errors = []
        learned_means = []
        optimal_means = []
        for t in range(run["metadata"]["horizon"]):
            learned = final_policy_probabilities(env, policy, law=flow[t], t=t).to(env.device)
            optimal = env.optimal_policy(flow[t])
            learned_mean = learned[:, env.AD].mean()
            optimal_mean = optimal[:, env.AD].mean()
            learned_means.append(learned_mean)
            optimal_means.append(optimal_mean)
            errors.append((learned[:, env.AD] - optimal[:, env.AD]).abs().mean())

        errors = torch.stack(errors)
        rows.append(
            {
                "label": run_label(run["metadata"]),
                "flow": run["metadata"]["flow"],
                "horizon": run["metadata"]["horizon"],
                "eta": run["metadata"].get("eta"),
                "seed": run["metadata"]["seed"],
                "mean_abs_ad_probability_error": float(errors.mean().detach().cpu()),
                "max_abs_ad_probability_error": float(errors.max().detach().cpu()),
                "learned_mean_ad_probability": float(torch.stack(learned_means).mean().detach().cpu()),
                "optimal_mean_ad_probability": float(torch.stack(optimal_means).mean().detach().cpu()),
            }
        )
    return pd.DataFrame(rows)


def objective_table(runs):
    rows = []
    optimum_cache = {}
    for run in runs:
        env, policy = load_env_and_policy(run)
        metadata = run["metadata"]
        row = {
            "env": metadata["env"],
            "label": run_label(metadata),
            "flow": metadata["flow"],
            "horizon": metadata["horizon"],
            "eta": metadata.get("eta"),
            "seed": metadata["seed"],
            "validation_reward": run["summary"].get("last_validation_objective"),
        }

        if hasattr(env, "objective"):
            row["objective_convention"] = "cost" if metadata["env"] == "lq" else "reward"
            theta = policy if not isinstance(policy, torch.nn.Module) else None
            # Each estimator randomizes the population its own way, so J^lambda
            # has to be taken under the randomizer the run was trained with.
            # randomizer_for_run returns None where no closed form exists (a
            # mixture chart with more than one component), and that is the only
            # case the perturbed columns are left out.
            law_randomizer = randomizer_for_run(metadata)
            analytic_perturbation = (
                metadata["env"] not in CONTINUOUS_TRANSPORT_ENVS or law_randomizer is not None
            )
            if metadata["env"] not in CONTINUOUS_TRANSPORT_ENVS:
                law_randomizer = None
            if theta is not None:
                with torch.no_grad():
                    row["J0"] = float(env.objective(theta, lambda_=0.0).detach().cpu())
                    if metadata["perturbation"] is not None and analytic_perturbation:
                        row["Jlambda"] = float(
                            env.objective(
                                theta, lambda_=metadata["perturbation"], randomizer=law_randomizer
                            ).detach().cpu()
                        )
                    if hasattr(env, "optimal_policy"):
                        try:
                            optimal = env.optimal_policy()
                            row["J0_star"] = float(env.objective(optimal, lambda_=0.0).detach().cpu())
                        except NotImplementedError:
                            pass
                if metadata["perturbation"] is not None and analytic_perturbation and hasattr(env, "optimal_policy"):
                    key = (
                        metadata["env"],
                        metadata["horizon"],
                        metadata["perturbation"],
                        repr(law_randomizer),
                        repr(metadata["env_config"]),
                    )
                    try:
                        if key not in optimum_cache:
                            optimum_cache[key] = optimize_exact_policy(
                                env, metadata["perturbation"], law_randomizer
                            )
                        optimal_lambda = optimum_cache[key]
                        with torch.no_grad():
                            row["Jlambda_star"] = float(
                                env.objective(
                                    optimal_lambda,
                                    lambda_=metadata["perturbation"],
                                    randomizer=law_randomizer,
                                ).detach().cpu()
                            )
                    except NotImplementedError:
                        pass

        rows.append(row)

    return pd.DataFrame(rows)


def runtime_table(runs):
    df = runs_dataframe(runs)
    if df.empty:
        return df
    df = df.copy()
    return (
        df.groupby(["env", "algorithm", "perturbation", "eta", "horizon", "flow"], dropna=False, as_index=False)
        .agg(
            setup_seconds_mean=("setup_seconds", "mean"),
            setup_seconds_std=("setup_seconds", "std"),
            train_step_seconds_mean=("train_step_seconds", "mean"),
            train_step_seconds_std=("train_step_seconds", "std"),
            validation_seconds_mean=("validation_seconds", "mean"),
            validation_seconds_std=("validation_seconds", "std"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
            seconds_per_step_mean=("seconds_per_training_step", "mean"),
            wall_seconds_per_step_mean=("wall_seconds_per_training_step", "mean"),
            validation_seconds_per_call_mean=("validation_seconds_per_call", "mean"),
            simulator_budget_mean=("simulator_budget_estimate", "mean"),
            n_runs=("seed", "count"),
        )
        .fillna(
            {
                "setup_seconds_std": 0.0,
                "train_step_seconds_std": 0.0,
                "validation_seconds_std": 0.0,
                "elapsed_seconds_std": 0.0,
            }
        )
    )


def discrete_transport_tv_bound_table(runs):
    rows = []
    for run in runs:
        metadata = run["metadata"]
        if metadata["algorithm"] != "transport" or metadata["env"] not in {
            "twostate",
            "cybersecurity",
            "distribution",
            "advertising",
        }:
            continue
        lambda_ = metadata["perturbation"]
        env, policy = load_env_and_policy(run)
        laws = discrete_law_flow(env, policy, metadata["horizon"])
        max_tv_bound = lambda_ * (1.0 - laws.min(dim=1).values)
        rows.append(
            {
                "env": metadata["env"],
                "label": run_label(metadata),
                "flow": metadata["flow"],
                "horizon": metadata["horizon"],
                "seed": metadata["seed"],
                "lambda": lambda_,
                "eta": metadata.get("eta"),
                "max_tv_upper_bound": float(max_tv_bound.max().detach().cpu()),
                "satisfies_lambda_bound": bool((max_tv_bound <= lambda_ + 1e-12).all().detach().cpu()),
            }
        )
    return pd.DataFrame(rows)


def save_table(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".tex":
        path.write_text(table.to_latex(index=False), encoding="utf-8")
    else:
        table.to_csv(path, index=False)
