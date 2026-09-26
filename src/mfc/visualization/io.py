from dataclasses import fields, replace
import json
from pathlib import Path

import pandas as pd
import torch

from .constants import ENVIRONMENTS, POLICIES


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_dtype(value):
    if value == "torch.float64":
        return torch.float64
    if value == "torch.float32":
        return torch.float32
    return value


def clean_config_values(values):
    return {key: parse_dtype(value) for key, value in values.items()}


def dataclass_from_dict(config_class, values, device=None):
    config = config_class()
    names = {field.name for field in fields(config)}
    cleaned = clean_config_values(values)
    if device is not None and "device" in names:
        cleaned["device"] = device
    updates = {key: value for key, value in cleaned.items() if key in names}
    return replace(config, **updates)


def load_runs(results_root="results", env=None):
    root = Path(results_root)
    search_root = root / env if env is not None else root
    runs = []

    for metadata_path in sorted(search_root.glob("**/metadata.json")):
        run_dir = metadata_path.parent
        history_path = run_dir / "history.json"
        summary_path = run_dir / "summary.json"
        policy_path = run_dir / "policy.pt"

        if not history_path.exists() or not policy_path.exists():
            continue

        metadata = read_json(metadata_path)
        history = read_json(history_path)
        summary = read_json(summary_path) if summary_path.exists() else {}
        runs.append(
            {
                "path": run_dir,
                "metadata": metadata,
                "history": history,
                "summary": summary,
                "policy_path": policy_path,
            }
        )

    return runs


def run_label(metadata):
    algorithm = metadata["algorithm"]
    if algorithm == "reinforce":
        return "REINFORCE"
    if algorithm == "mfqlearning":
        resolution = metadata.get("algorithm_config", {}).get("simplex_resolution")
        return "MFQ-learning" if resolution is None else f"MFQ-learning Nm={resolution}"
    if algorithm == "mfreinforce":
        return f"MF-REINFORCE eps={metadata['perturbation']:g}"
    if algorithm == "gaussian":
        return f"Transport-Proba lambda={metadata['perturbation']:g}"
    label = f"Transport lambda={metadata['perturbation']:g}"
    eta = metadata.get("eta")
    if eta is not None:
        label += f", eta={eta:g}"
    components = metadata.get("algorithm_config", {}).get("n_components")
    if components is not None:
        label += f", K={components:g}"
    return label


def best_runs_by_label(runs, prefer_validation=True):
    best = {}
    for run in runs:
        metadata = run["metadata"]
        summary = run["summary"]
        value = summary.get("last_validation_objective") if prefer_validation else None
        if value is None:
            value = summary.get("last_objective")
        if value is None:
            continue
        eta_key = metadata.get("eta")
        if metadata["algorithm"] == "mfqlearning":
            eta_key = metadata.get("algorithm_config", {}).get("simplex_resolution")
        key = (
            metadata["algorithm"],
            metadata["perturbation"],
            eta_key,
            metadata["horizon"],
            metadata["flow"],
            metadata.get("algorithm_config", {}).get("n_components"),
        )
        if key not in best or value > best[key][0]:
            best[key] = (value, run)
    return [run for _, run in best.values()]


def runs_dataframe(runs):
    rows = []
    for run in runs:
        metadata = run["metadata"]
        summary = run["summary"]
        rows.append(
            {
                "path": str(run["path"]),
                "env": metadata["env"],
                "algorithm": metadata["algorithm"],
                "label": run_label(metadata),
                "perturbation": metadata["perturbation"],
                "eta": metadata.get("eta"),
                "horizon": metadata["horizon"],
                "flow": metadata["flow"],
                "seed": metadata["seed"],
                "elapsed_seconds": summary.get("elapsed_seconds", metadata.get("elapsed_seconds")),
                "setup_seconds": summary.get("setup_seconds", metadata.get("setup_seconds", 0.0)),
                "train_step_seconds": summary.get(
                    "train_step_seconds",
                    metadata.get("train_step_seconds", summary.get("elapsed_seconds", metadata.get("elapsed_seconds"))),
                ),
                "validation_seconds": summary.get("validation_seconds", metadata.get("validation_seconds", 0.0)),
                "unaccounted_seconds": summary.get("unaccounted_seconds", metadata.get("unaccounted_seconds", 0.0)),
                "seconds_per_training_step": summary.get("seconds_per_training_step"),
                "wall_seconds_per_training_step": summary.get(
                    "wall_seconds_per_training_step",
                    summary.get("seconds_per_training_step"),
                ),
                "validation_seconds_per_call": summary.get("validation_seconds_per_call", 0.0),
                "simulator_budget_estimate": metadata.get("simulator_budget_estimate"),
                "last_objective": summary.get("last_objective"),
                "last_validation_objective": summary.get("last_validation_objective"),
            }
        )
    return pd.DataFrame(rows)


def validation_dataframe(runs):
    rows = []
    for run in runs:
        metadata = run["metadata"]
        history = run["history"]
        values = history.get("validation_objective", [])
        steps = history.get("validation_steps")
        if steps is None:
            interval = metadata["algorithm_config"].get("validation_interval") or metadata["env_config"].get(
                "validation_interval", 1
            )
            steps = [(index + 1) * interval for index in range(len(values))]

        for step, value in zip(steps, values):
            rows.append(
                {
                    "env": metadata["env"],
                    "algorithm": metadata["algorithm"],
                    "label": run_label(metadata),
                    "perturbation": metadata["perturbation"],
                    "eta": metadata.get("eta"),
                    "horizon": metadata["horizon"],
                    "flow": metadata["flow"],
                    "seed": metadata["seed"],
                    "step": step,
                    "simulator_transitions": step * (metadata.get("simulator_budget_estimate") or 1),
                    "validation_reward": value,
                }
            )

    return pd.DataFrame(rows)


def load_env_and_policy(run, device=None):
    metadata = run["metadata"]
    env_class, config_class = ENVIRONMENTS[metadata["env"]]
    config = dataclass_from_dict(config_class, metadata["env_config"], device=device)
    env = env_class(config)

    payload = torch.load(run["policy_path"], map_location=env.device, weights_only=False)
    if payload["kind"] == "tensor":
        policy = torch.nn.Parameter(payload["tensor"].to(device=env.device, dtype=env.dtype))
    elif payload["kind"] == "mean_field_q_policy":
        from mfc.algorithms import MeanFieldQPolicy

        policy = MeanFieldQPolicy.from_payload(payload, device=env.device, dtype=env.dtype)
    elif payload["kind"] == "module_state_dict":
        policy = POLICIES[metadata["env"]](config)
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
    else:
        raise ValueError(f"Unknown policy payload kind: {payload['kind']}")

    return env, policy
