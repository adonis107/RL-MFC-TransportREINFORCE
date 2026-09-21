"""Create the final figures and tables from saved runs.

    uv run python scripts/make_outputs.py --results-root results

Outputs are written under ``outputs/`` by default:

    outputs/figures/learning_curves.pdf
    outputs/figures/learned_policies.pdf
    outputs/figures/theory_verification.pdf
    outputs/tables/objective_summary.tex
    outputs/tables/budget_runtime.tex
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run as run_plan
from mfc.algorithms.discrete_validation import mean_field_next_law
from mfc.environments import (
    Distribution,
    DistributionConfig,
    DistributionPolicy,
    Portfolio,
    PortfolioConfig,
    TwoState,
    TwoStateConfig,
)
from mfc.visualization import load_runs, objective_table, runtime_table


BENCHMARKS = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
DISPLAY = {
    "twostate": "Two-state",
    "cybersecurity": "Cybersecurity",
    "distribution": "Distribution",
    "advertising": "Advertising",
    "lq": "Linear--quadratic",
    "portfolio": "Portfolio",
}
MAIN_FLOW = {
    "twostate": "exact",
    "cybersecurity": "exact",
    "distribution": "exact",
    "advertising": "exact",
    "lq": "particle",
    "portfolio": "particle",
}
REFERENCE_OPTIMUM = {
    ("twostate", 5): -2.640,
    ("distribution", 5): -0.056991,
    ("advertising", 5): 1.006168,
}
LEARNING_PANELS = {
    "Linear--quadratic": ("lq", -7.223777),
    "Portfolio": ("portfolio", -13.153766),
    "Two-state": ("twostate", -2.640),
    "Distribution": ("distribution", -0.056991),
}
METHOD_COLOR = {"REINFORCE": "#eb6834", "MF-REINFORCE": "#eda100", "Transport": "#2a78d6"}
METHOD_MARKER = {"REINFORCE": "s", "MF-REINFORCE": "D", "Transport": "o"}
OPTIMAL = "#52514e"
GRID, INK, MUTED = "#d8d7d2", "#0b0b0b", "#52514e"


def ensure_dir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(figure, path):
    path = ensure_dir(path)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def transport_stem(env, flow=None, components=None):
    flow = MAIN_FLOW[env] if flow is None else flow
    lambda_ = run_plan.asymptotic_main_lambda(env)
    eta = run_plan.asymptotic_auxiliary_eta(env)
    stem = f"transport_lambda_{lambda_:g}_eta_{eta:g}"
    if components is not None:
        stem = f"{stem}_K_{components}"
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    return f"{stem}_T_{horizon}_{flow}"


def run_stems(env):
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    stems = {"REINFORCE": f"reinforce_none_T_{horizon}_exact"}
    if env == "twostate":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_0.2_T_{horizon}_exact"
    elif env == "cybersecurity":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_1_T_{horizon}_exact"
    elif env == "distribution":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_2_T_{horizon}_exact"
    elif env == "advertising":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_1_T_{horizon}_exact"
    components = 1 if env in {"lq", "portfolio"} else None
    stems["Transport"] = transport_stem(env, components=components)
    return stems


def curves(directory, stem):
    values = []
    for path in sorted(Path(directory).glob(f"{stem}_seed_*")):
        history = json.loads((path / "history.json").read_text())
        values.append(history["validation_objective"])
    if not values:
        return None
    length = min(len(v) for v in values)
    return np.array([v[:length] for v in values])


def compact_tick(value, _position=None):
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:g}"
    decimals = max(0, int(np.ceil(-np.log10(value))) + 1)
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=3, width=0.5)


def learning_curves(results_root, output):
    figure, axes = plt.subplots(2, 2, figsize=(5.5, 3.9), constrained_layout=True)
    for ax, (title, (env, optimum)) in zip(axes.flat, LEARNING_PANELS.items()):
        panel = []
        for method, stem in run_stems(env).items():
            seeds = curves(Path(results_root) / env, stem)
            if seeds is None:
                continue
            steps = np.arange(1, seeds.shape[1] + 1) * 10
            gaps = np.abs(seeds - optimum)
            gap, deviation = gaps.mean(axis=0), gaps.std(axis=0)
            panel.extend(gap.tolist())
            ax.plot(steps, gap, color=METHOD_COLOR[method], linewidth=1.1, label=method)
            floor = max(gap.min() * 0.25, 1e-12)
            ax.fill_between(steps, np.maximum(gap - deviation, floor), gap + deviation,
                            color=METHOD_COLOR[method], alpha=0.20, linewidth=0)
        ax.set_title(title.replace("--", "-"), fontsize=8, color=INK, pad=12)
        ax.text(
            0.5,
            1.015,
            rf"$\lambda={run_plan.asymptotic_main_lambda(env):g},\ \eta={run_plan.asymptotic_auxiliary_eta(env):g}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=MUTED,
        )
        ax.set_yscale("log")
        ax.set_xlabel("policy updates", fontsize=7.5)
        ax.set_ylabel(r"$|J(\theta)-J(\theta^\star)|$", fontsize=7.5)
        style(ax)
        tail = [v for v in panel[max(len(panel) // 20, 1):] if v > 0]
        if tail:
            ax.set_ylim(min(tail) * 0.6, max(tail) * 1.6)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(compact_tick))

    found = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    order = [name for name in ("REINFORCE", "MF-REINFORCE", "Transport") if name in found]
    figure.legend([found[name] for name in order], order, loc="outside lower center",
                  ncol=3, frameon=False, handlelength=1.8)
    save_figure(figure, output)


def tensor_policy(path):
    blob = torch.load(path / "policy.pt", map_location="cpu", weights_only=False)
    return blob["tensor"] if "tensor" in blob else blob


def learned_policies(results_root, output):
    figure, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), constrained_layout=True)

    env = Portfolio(PortfolioConfig(T=10))
    with torch.no_grad():
        optimal, _ = env.moment_flow(env.optimal_policy(), lambda_=0.0)
    steps = np.arange(optimal.numel())
    axes[0].plot(steps, optimal.cpu().numpy(), color=OPTIMAL, linewidth=1.0, dashes=(3, 2), label=r"$\theta^\star$")
    for method, path in {
        "Transport": Path(results_root) / "portfolio" / f"{transport_stem('portfolio', components=1)}_seed_0",
        "REINFORCE": Path(results_root) / "portfolio" / "reinforce_none_T_10_exact_seed_0",
    }.items():
        with torch.no_grad():
            flow, _ = env.moment_flow(tensor_policy(path), lambda_=0.0)
        axes[0].plot(steps, flow.cpu().numpy(), color=METHOD_COLOR[method], linewidth=1.2, label=method)
    axes[0].set_title("Portfolio: mean wealth flow", fontsize=8, color=INK, pad=3)
    axes[0].set_xlabel("$t$", fontsize=7.5)
    axes[0].set_ylabel(r"$\bar x(\mu_t^\theta)$", fontsize=7.5)
    style(axes[0])

    env = TwoState(TwoStateConfig(T=5))
    runs = {
        "Transport": Path(results_root) / "twostate" / f"{transport_stem('twostate')}_seed_0",
        "MF-REINFORCE": Path(results_root) / "twostate" / "mfreinforce_eps_0.2_T_5_exact_seed_0",
        "REINFORCE": Path(results_root) / "twostate" / "reinforce_none_T_5_exact_seed_0",
    }
    steps = np.arange(env.config.T + 1)
    for name, theta in [(r"$\theta^\star$", env.optimal_theta())] + [(m, tensor_policy(q)) for m, q in runs.items()]:
        law = env.initial_distribution.clone()
        trace = [float(law[1])]
        with torch.no_grad():
            for t in range(env.config.T):
                law = mean_field_next_law(env, theta, lambda step: step, t, law)
                trace.append(float(law[1]))
        colour = OPTIMAL if name.startswith("$") else METHOD_COLOR[name]
        dashes = (3, 2) if name.startswith("$") else (None, None)
        axes[1].plot(steps, trace, color=colour, linewidth=1.1, dashes=dashes, label=name)
    axes[1].set_title("Two-state: population flow", fontsize=8, color=INK, pad=3)
    axes[1].set_xlabel("$t$", fontsize=7.5)
    axes[1].set_ylabel(r"$\mu_t^\theta(1)$", fontsize=7.5)
    style(axes[1])

    env = Distribution(DistributionConfig())
    states = np.arange(env.n_states)

    def terminal(policy):
        law = env.initial_distribution.clone()
        with torch.no_grad():
            for t in range(env.config.T):
                law = env.population_step(law, policy(t, law))
        return law.cpu().numpy()

    axes[2].plot(states, terminal(env.optimal_policy()), color=OPTIMAL, linewidth=1.0,
                 dashes=(3, 2), label=r"$\theta^\star$")
    for method, (folder, stem) in {
        "Transport": (Path(results_root) / "distribution", transport_stem("distribution")),
        "MF-REINFORCE": (Path(results_root) / "distribution", "mfreinforce_eps_2_T_5_exact"),
        "REINFORCE": (Path(results_root) / "distribution", "reinforce_none_T_5_exact"),
    }.items():
        module = DistributionPolicy(env.config)
        blob = torch.load(folder / f"{stem}_seed_0" / "policy.pt", map_location="cpu", weights_only=False)
        module.load_state_dict(blob["state_dict"])
        axes[2].plot(states, terminal(lambda t, mu: module(torch.tensor(float(t)), mu)),
                     color=METHOD_COLOR[method], linewidth=1.1, marker=METHOD_MARKER[method],
                     markersize=3.0, label=method)
    axes[2].set_title("Distribution: terminal law", fontsize=8, color=INK, pad=3)
    axes[2].set_xlabel("state", fontsize=7.5)
    axes[2].set_ylabel(r"$\mu_T^\theta(x)$", fontsize=7.5)
    style(axes[2])

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8)
    save_figure(figure, output)


def theory_verification(estimate_csv, consistency_csv, output):
    display = {"twostate": "Two-state", "cybersecurity": "Cybersecurity", "distribution": "Distribution",
               "advertising": "Advertising", "lq": "Linear-quadratic", "portfolio": "Portfolio"}
    order = ["lq", "portfolio", "twostate", "distribution", "cybersecurity", "advertising"]
    color = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    marker = dict(zip(order, ["o", "s", "^", "D", "v", "P"]))
    ticks = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]

    estimate = pd.read_csv(estimate_csv)
    consistency = pd.read_csv(consistency_csv)
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)
    for name in order:
        rows = estimate[estimate["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        axes[0].plot(rows["lambda"], rows["max_ratio"], color=color[name], marker=marker[name],
                     markersize=3.2, linewidth=1.1, dashes=dashes, label=display[name])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.25)
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("scaled deviation")
    axes[0].set_title("(a) perturbation estimate", fontsize=8, color=INK, pad=4)

    for name in order:
        rows = consistency[consistency["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        axes[1].plot(rows["lambda"], rows["gradient_over_lambda"], color=color[name], marker=marker[name],
                     markersize=3.2, linewidth=1.1, dashes=dashes, label=display[name])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\|\nabla_\theta J^\lambda-\nabla_\theta J\|\,/\,\lambda$")
    axes[1].set_title("(b) perturbation consistency", fontsize=8, color=INK, pad=4)

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        ax.tick_params(axis="x", which="minor", length=0)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False,
                  handlelength=1.8, columnspacing=1.0, borderpad=0.2)
    save_figure(figure, output)


def method_of(label):
    if label.startswith("REINFORCE"):
        return "reinforce"
    if label.startswith("MFQ-learning"):
        return "mfqlearning"
    if label.startswith("MF-REINFORCE"):
        return "mfreinforce"
    return "transport"


def number(value, digits=4):
    return "---" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def with_error(row, digits=4):
    if row is None:
        return "---"
    return f"${number(row['mean'], digits)}\\pm{number(row['std'], digits)}$"


def table_environment(body, caption, label, alignment, size=None):
    return "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            *([size] if size else []),
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            body,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def grouped_objectives(results_root, env):
    table = objective_table(load_runs(results_root, env=env))
    grouped = (
        table.groupby(["label", "flow", "horizon"])["validation_reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["method"] = grouped["label"].map(method_of)
    return grouped, table


def select_headline(grouped, env):
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    flow = MAIN_FLOW[env]
    subset = grouped[grouped["horizon"] == horizon]
    subset = subset[(subset["flow"] == flow) | (subset["method"] == "reinforce")]
    best = {}
    for method, rows in subset.groupby("method"):
        best[method] = rows.loc[rows["mean"].idxmax()]
    return best


def reference_optimum(results_root, env):
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    if (env, horizon) in REFERENCE_OPTIMUM:
        return REFERENCE_OPTIMUM[(env, horizon)]
    table = objective_table(load_runs(results_root, env=env))
    if "J0_star" not in table.columns:
        return None
    values = table.loc[table["horizon"] == horizon, "J0_star"].dropna()
    if values.empty:
        return None
    convention = table["objective_convention"].dropna().iloc[0]
    return -float(values.iloc[0]) if convention == "cost" else float(values.iloc[0])


def objective_summary(results_root):
    lines = [
        "Benchmark & Optimum & REINFORCE & MF-REINFORCE & Transport & $(\\lambda,\\eta)$ \\\\",
        "\\midrule",
    ]
    for env in BENCHMARKS:
        grouped, _ = grouped_objectives(results_root, env)
        best = select_headline(grouped, env)
        optimum = reference_optimum(results_root, env)
        digits = 3 if env == "portfolio" else 4
        transport = best.get("transport")
        scales = "---" if transport is None else f"$({run_plan.asymptotic_main_lambda(env):g},{run_plan.asymptotic_auxiliary_eta(env):g})$"
        lines.append(
            " & ".join(
                [
                    f"{DISPLAY[env]} ($T={run_plan.TRANSPORT_ALLOCATIONS[env]['horizon']}$)",
                    "---" if optimum is None else f"${number(optimum, digits)}$",
                    with_error(best.get("reinforce"), digits),
                    with_error(best.get("mfreinforce"), digits),
                    with_error(transport, digits),
                    scales,
                ]
            )
            + " \\\\"
        )
    caption = (
        "Final validation objective on every benchmark, as mean and standard deviation over seeds. "
        "Higher is better. The transport column shows the best fixed-scale run among the bound-scale "
        "multipliers and the asymptotic anchor scales used for the benchmark."
    )
    return table_environment("\n".join(lines), caption, "tab:objective-summary", "lrrrrl", size="\\small")


def budget_runtime(results_root):
    lines = [
        "Benchmark & Estimator & Simulator budget & Wall clock (s) & Ratio to REINFORCE \\\\",
        "\\midrule",
    ]
    for env in BENCHMARKS:
        runtime = runtime_table(load_runs(results_root, env=env))
        horizon, flow = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"], MAIN_FLOW[env]
        subset = runtime[runtime["horizon"] == horizon]
        subset = subset[(subset["flow"] == flow) | (subset["algorithm"] == "reinforce")]
        reference = subset[subset["algorithm"] == "reinforce"]
        reference_seconds = float(reference["elapsed_seconds_mean"].iloc[0]) if not reference.empty else None
        first = True
        for algorithm, name in [("reinforce", "REINFORCE"), ("mfreinforce", "MF-REINFORCE"), ("transport", "Transport"), ("mfqlearning", "MFQ-learning")]:
            rows = subset[subset["algorithm"] == algorithm]
            if rows.empty:
                continue
            row = rows.iloc[rows["elapsed_seconds_mean"].to_numpy().argmax()]
            ratio = "---" if reference_seconds is None else f"${float(row['elapsed_seconds_mean']) / reference_seconds:.1f}\\times$"
            lines.append(
                " & ".join(
                    [
                        DISPLAY[env] if first else "",
                        name,
                        f"${float(row['simulator_budget_mean']):,.0f}$".replace(",", "\\,"),
                        f"${float(row['elapsed_seconds_mean']):,.0f}$".replace(",", "\\,"),
                        ratio,
                    ]
                )
                + " \\\\"
            )
            first = False
        if env != BENCHMARKS[-1]:
            lines.append("\\midrule")
    caption = (
        "Simulator budget and wall-clock cost per run at the headline configuration. "
        "Simulator budgets are matched by construction; wall-clock cost also reflects estimator arithmetic."
    )
    return table_environment("\n".join(lines), caption, "tab:budget-runtime", "llrrr", size="\\small")


def write_table(body, path):
    path = ensure_dir(path)
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--theory-estimate", default="results/figures/theory/perturbation_estimate.csv")
    parser.add_argument("--theory-consistency", default="results/figures/theory_400/perturbation_consistency.csv")
    args = parser.parse_args()

    results_root = ROOT / args.results_root
    output_root = ROOT / args.output_root
    figures = output_root / "figures"
    tables = output_root / "tables"

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.labelsize": 8, "legend.fontsize": 7.2})

    learning_curves(results_root, figures / "learning_curves.pdf")
    learned_policies(results_root, figures / "learned_policies.pdf")
    theory_verification(ROOT / args.theory_estimate, ROOT / args.theory_consistency, figures / "theory_verification.pdf")
    write_table(objective_summary(results_root), tables / "objective_summary.tex")
    write_table(budget_runtime(results_root), tables / "budget_runtime.tex")


if __name__ == "__main__":
    main()
