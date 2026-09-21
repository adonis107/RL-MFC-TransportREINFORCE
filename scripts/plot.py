import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from mfc.visualization import (
    CONTINUOUS_ENVS,
    advertising_policy_error_table,
    discrete_transport_tv_bound_table,
    gradient_diagnostics,
    identification_sweep,
    best_runs_by_label,
    flow_dataframe,
    load_runs,
    objective_table,
    plot_advertising_diagnostics,
    plot_distribution_comparison,
    plot_flow_comparison,
    plot_state_flow,
    plot_validation_rewards,
    runtime_table,
    save_table,
    transport_correction_table,
    twostate_policy_error_table,
)


def save_current(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()


def perturbation_stem(metadata):
    if metadata["algorithm"] == "mfqlearning":
        return f"Nm_{metadata.get('algorithm_config', {}).get('simplex_resolution', 'na')}"
    perturbation = metadata["perturbation"] if metadata["perturbation"] is not None else "none"
    if metadata["algorithm"] != "transport":
        return str(perturbation)

    eta = metadata.get("algorithm_config", {}).get("eta", metadata.get("eta"))
    if eta is None:
        return str(perturbation)
    return f"{perturbation}_eta_{eta}"


def value_stem(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def sort_key(value):
    if value is None:
        return float("-inf")
    return value


def run_score(run):
    summary = run.get("summary", {})
    value = summary.get("last_validation_objective")
    if value is None:
        value = summary.get("last_objective")
    return float("-inf") if value is None else value


def validation_overview_runs(horizon_runs, flow=None):
    selected = []
    for run in horizon_runs:
        metadata = run["metadata"]
        if metadata["algorithm"] == "mfqlearning":
            continue
        if flow is not None and metadata["algorithm"] != "reinforce" and metadata["flow"] != flow:
            continue
        selected.append(run)
    return selected


def representative_plot_runs(runs):
    return best_runs_by_label(runs)


def save_validation_splits(horizon_runs, env, horizon, output_dir):
    mfq_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] == "mfqlearning"]
    comparison_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] != "mfqlearning"]
    transport_runs = [run for run in comparison_runs if run["metadata"]["algorithm"] == "transport"]

    if not transport_runs:
        plot_validation_rewards(comparison_runs, env=env, horizon=horizon)
        save_current(output_dir / f"validation_T_{horizon}.png")
        plot_validation_rewards(comparison_runs, env=env, horizon=horizon, x_axis="simulator_transitions")
        save_current(output_dir / f"validation_transitions_T_{horizon}.png")
        if mfq_runs:
            plot_validation_rewards(mfq_runs, env=env, horizon=horizon)
            save_current(output_dir / f"validation_mfq_T_{horizon}.png")
            plot_validation_rewards(mfq_runs, env=env, horizon=horizon, x_axis="simulator_transitions")
            save_current(output_dir / f"validation_mfq_transitions_T_{horizon}.png")
        return

    transport_flows = sorted({run["metadata"]["flow"] for run in transport_runs})
    default_flow = "exact" if "exact" in transport_flows else transport_flows[0]

    for flow in transport_flows:
        overview_runs = validation_overview_runs(horizon_runs, flow=flow)
        if not overview_runs:
            continue
        plot_validation_rewards(overview_runs, env=env, horizon=horizon)
        suffix = "" if flow == default_flow else f"_{flow}"
        save_current(output_dir / f"validation_T_{horizon}{suffix}.png")
        plot_validation_rewards(overview_runs, env=env, horizon=horizon, x_axis="simulator_transitions")
        save_current(output_dir / f"validation_transitions_T_{horizon}{suffix}.png")

    if mfq_runs:
        plot_validation_rewards(mfq_runs, env=env, horizon=horizon)
        save_current(output_dir / f"validation_mfq_T_{horizon}.png")
        plot_validation_rewards(mfq_runs, env=env, horizon=horizon, x_axis="simulator_transitions")
        save_current(output_dir / f"validation_mfq_transitions_T_{horizon}.png")


def save_flow_comparison_splits(transport_runs, env, horizon, output_dir):
    flows = {run["metadata"]["flow"] for run in transport_runs}
    if not {"exact", "particle"}.issubset(flows):
        return

    for perturbation in sorted({run["metadata"].get("perturbation") for run in transport_runs}, key=sort_key):
        split_runs = [run for run in transport_runs if run["metadata"].get("perturbation") == perturbation]
        split_flows = {run["metadata"]["flow"] for run in split_runs}
        if not {"exact", "particle"}.issubset(split_flows):
            continue
        plot_flow_comparison(split_runs, env=env, horizon=horizon)
        save_current(
            output_dir
            / (
                f"flow_comparison_T_{horizon}_"
                f"lambda_{value_stem(perturbation)}.png"
            )
        )


def make_standard_outputs(
    env,
    results_root,
    output_root,
    gradient_replications=0,
    correction_replications=0,
    gradient_particles=None,
    correction_particles=None,
    identification_replications=0,
    allow_empty=False,
):
    runs = load_runs(results_root, env=env)
    if not runs:
        message = f"No runs found under {Path(results_root) / env}."
        if allow_empty:
            print(f"warning: {message}", file=sys.stderr)
            return False
        raise ValueError(message)

    output_dir = Path(output_root) / env
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon in sorted({run["metadata"]["horizon"] for run in runs}):
        horizon_runs = [run for run in runs if run["metadata"]["horizon"] == horizon]
        save_validation_splits(horizon_runs, env, horizon, output_dir)

        transport_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] == "transport"]
        save_flow_comparison_splits(transport_runs, env, horizon, output_dir)

    save_table(runtime_table(runs), output_dir / "runtime.csv")
    save_table(objective_table(runs), output_dir / "objectives.csv")

    tv_table = discrete_transport_tv_bound_table(runs)
    if not tv_table.empty:
        save_table(tv_table, output_dir / "transport_tv_bounds.csv")

    if env == "twostate":
        save_table(twostate_policy_error_table(runs), output_dir / "policy_error.csv")
    if env == "advertising":
        save_table(advertising_policy_error_table(runs), output_dir / "policy_error.csv")

    if env in CONTINUOUS_ENVS:
        rows = []
        for run in representative_plot_runs(runs):
            metadata = run["metadata"]
            table = flow_dataframe(run)
            table.insert(0, "seed", metadata["seed"])
            table.insert(0, "flow", metadata["flow"])
            table.insert(0, "horizon", metadata["horizon"])
            table.insert(0, "eta", metadata.get("eta"))
            table.insert(0, "perturbation", metadata["perturbation"])
            table.insert(0, "label", (
                f"{metadata['algorithm']}_"
                f"{perturbation_stem(metadata)}_"
                f"{metadata['flow']}"
            ))
            rows.append(table)
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "moment_flows.csv")

    for run in representative_plot_runs(runs):
        metadata = run["metadata"]
        stem = (
            f"{metadata['algorithm']}_"
            f"{perturbation_stem(metadata)}_"
            f"T_{metadata['horizon']}_{metadata['flow']}"
        )
        if env == "distribution":
            plot_distribution_comparison(run)
            save_current(output_dir / f"distribution_{stem}.png")
        elif env == "advertising":
            plot_advertising_diagnostics(run)
            save_current(output_dir / f"advertising_{stem}.png")
        else:
            plot_state_flow(run)
            save_current(output_dir / f"state_flow_{stem}.png")

    if gradient_replications > 0:
        rows = []
        for run in runs:
            try:
                rows.append(
                    gradient_diagnostics(
                        run,
                        n_replications=gradient_replications,
                        n_particles=gradient_particles,
                    )
                )
            except ValueError:
                continue
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "gradient_diagnostics.csv")

    if identification_replications > 0 and env in CONTINUOUS_ENVS:
        # The sweep characterizes the benchmark rather than one training run, so
        # it runs once, on the representative transport run of the environment.
        reference = next(
            (run for run in representative_plot_runs(runs) if run["metadata"]["algorithm"] == "transport"),
            None,
        )
        if reference is not None:
            try:
                sweep = identification_sweep(
                    reference,
                    n_replications=identification_replications,
                    n_particles=gradient_particles,
                )
            except ValueError:
                sweep = None
            if sweep is not None:
                save_table(sweep, output_dir / "mixture_identification.csv")

    if correction_replications > 0:
        rows = []
        for run in runs:
            try:
                rows.append(
                    transport_correction_table(
                        run,
                        n_replications=correction_replications,
                        n_particles=correction_particles,
                    )
                )
            except ValueError:
                continue
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "transport_correction.csv")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Create standard plots and tables from saved MFC runs.")
    parser.add_argument(
        "--env",
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "all"],
        required=True,
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="results/figures")
    parser.add_argument("--gradient-replications", type=int, default=0)
    parser.add_argument("--correction-replications", type=int, default=0)
    parser.add_argument("--gradient-particles", type=int, default=None)
    parser.add_argument("--correction-particles", type=int, default=None)
    parser.add_argument("--identification-replications", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
    selected_envs = envs if args.env == "all" else [args.env]
    for env in selected_envs:
        make_standard_outputs(
            env,
            args.results_root,
            args.output_root,
            args.gradient_replications,
            args.correction_replications,
            args.gradient_particles,
            args.correction_particles,
            args.identification_replications,
            allow_empty=args.env == "all",
        )


if __name__ == "__main__":
    main()
