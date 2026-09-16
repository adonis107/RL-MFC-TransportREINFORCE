"""Main-body result figures: learning curves, and learned against optimal policies.

    uv run python scripts/results_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from mfc.environments import (Distribution, DistributionConfig, DistributionPolicy, LQ, LQConfig,
                              Portfolio, PortfolioConfig, TwoState, TwoStateConfig)

METHOD_COLOR = {"REINFORCE": "#eb6834", "MF-REINFORCE": "#eda100", "Transport": "#2a78d6"}
METHOD_MARKER = {"REINFORCE": "s", "MF-REINFORCE": "D", "Transport": "o"}
OPTIMAL = "#52514e"
GRID, INK, MUTED = "#d8d7d2", "#0b0b0b", "#52514e"

RUNS = {
    "Linear--quadratic": {"config": r"$\lambda=0.1,\ K=1$", "dir": "lq", "optimum": -7.223777, "runs": {
        "REINFORCE": "reinforce_none_T_20_exact", "MF-REINFORCE": None,
        "Transport": "transport_lambda_0.1_eta_0.85_K_1_T_20_particle"}},

    "Portfolio": {"config": r"$\lambda=0.4,\ K=1$", "dir": "portfolio", "optimum": -13.153766, "runs": {
        "REINFORCE": "reinforce_none_T_10_exact", "MF-REINFORCE": None,
        "Transport": "transport_lambda_0.4_eta_0.85_K_1_T_10_particle"}},

    "Two-state": {"config": r"$\lambda=0.05,\ \eta=0.85$", "corner": ("left", "bottom"), "dir": "twostate", "optimum": -2.640, "runs": {
        "REINFORCE": "reinforce_none_T_5_exact", "MF-REINFORCE": None,
        "Transport": "transport_lambda_0.05_eta_0.85_T_5_exact"}},

    "Distribution": {"config": r"$\lambda=0.2,\ \eta=0.98,\ \sigma=0.5$", "dir": "distribution", "optimum": -0.056991, "runs": {
        "REINFORCE": "reinforce_none_T_5_exact", "MF-REINFORCE": "mfreinforce_eps_2_T_5_exact",
        "Transport": None}},

}
OVERRIDE = {
    ("Two-state", "MF-REINFORCE"): ("results/ts_mfr_tuned/twostate", "mfreinforce_eps_0.2_T_5_exact"),
    ("Distribution", "Transport"): ("results/tuned_scales/distribution", "transport_lambda_0.2_eta_0.98_T_5_exact"),
}


def compact(value, _position=None):
    """Plain decimal for a log tick, without scientific notation."""
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


def curves(directory, stem):
    """Per-seed validation curves, truncated to their common length."""
    values = []
    for path in sorted(Path(directory).glob(f"{stem}_seed_*")):
        history = json.loads((path / "history.json").read_text())
        values.append(history["validation_objective"])
    if not values:
        return None
    length = min(len(v) for v in values)
    return np.array([v[:length] for v in values])


def learning_curves(output):
    figure, axes = plt.subplots(2, 2, figsize=(5.5, 3.9), constrained_layout=True)
    for ax, (name, spec) in zip(axes.flat, RUNS.items()):
        panel = []
        for method, stem in spec["runs"].items():
            directory, resolved = ROOT / "results" / spec["dir"], stem
            if (name, method) in OVERRIDE:
                sub, resolved = OVERRIDE[(name, method)]
                directory = ROOT / sub
            if resolved is None:
                continue
            seeds = curves(directory, resolved)
            if seeds is None:
                continue
            steps = np.arange(1, seeds.shape[1] + 1) * 10
            # Optimality gap on a log axis keeps methods spanning orders of magnitude legible.
            # The band is the spread of the per-seed gaps, not the gap of the mean curve.
            gaps = np.abs(seeds - spec["optimum"])
            gap, deviation = gaps.mean(axis=0), gaps.std(axis=0)
            panel.append(list(gap))
            ax.plot(steps, gap, color=METHOD_COLOR[method], linewidth=1.1, label=method)
            floor = gap.min() * 0.25
            ax.fill_between(steps, np.maximum(gap - deviation, floor), gap + deviation,
                            color=METHOD_COLOR[method], alpha=0.20, linewidth=0)
        ax.set_title(name.replace("--", "\u2013"), fontsize=8, color=INK, pad=12)
        # Configuration as a subtitle above the axes, so it never sits over the data.
        ax.text(0.5, 1.015, spec["config"], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.4, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("policy updates", fontsize=7.5)
        ax.set_ylabel(r"$|J(\theta)-J(\theta^\star)|$", fontsize=7.5)
        style(ax)
        tail = [v for c in panel for v in c[max(len(c) // 20, 1):] if v > 0]
        if tail:
            ax.set_ylim(min(tail) * 0.6, max(tail) * 1.6)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(compact))

    found = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    order = [m for m in ("REINFORCE", "MF-REINFORCE", "Transport") if m in found]
    figure.legend([found[m] for m in order], order, loc="outside lower center",
                  ncol=3, frameon=False, handlelength=1.8)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(Path(output).with_suffix(".png"), dpi=200, bbox_inches="tight")


def tensor_policy(path):
    blob = torch.load(path / "policy.pt", map_location="cpu", weights_only=False)
    return blob["tensor"] if "tensor" in blob else blob


def learned_versus_optimal(output):
    """Population behaviour under the learned policies, against the optimum.

    Parameter recovery is not the right test on the continuous benchmarks: their
    objectives are nearly flat along the directions where the learned policy differs,
    so what is comparable is the population flow the policy induces.
    """
    from mfc.algorithms.discrete_validation import mean_field_next_law

    figure, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), constrained_layout=True)

    env = Portfolio(PortfolioConfig(T=10))
    with torch.no_grad():
        optimal, _ = env.moment_flow(env.optimal_policy(), lambda_=0.0)
    steps = np.arange(optimal.numel())
    axes[0].plot(steps, optimal.numpy(), color=OPTIMAL, linewidth=1.0, dashes=(3, 2), label=r"$\theta^\star$")
    for method, path in {"Transport": "results/portfolio/transport_lambda_0.4_eta_0.85_K_1_T_10_particle_seed_0",
                         "REINFORCE": "results/portfolio/reinforce_none_T_10_exact_seed_0"}.items():
        with torch.no_grad():
            flow, _ = env.moment_flow(tensor_policy(ROOT / path), lambda_=0.0)
        axes[0].plot(steps, flow.numpy(), color=METHOD_COLOR[method], linewidth=1.2, label=method)
    axes[0].set_title("Portfolio: mean wealth flow", fontsize=8, color=INK, pad=3)
    axes[0].set_xlabel("$t$", fontsize=7.5)
    axes[0].set_ylabel(r"$\bar x(\mu_t^\theta)$", fontsize=7.5)
    style(axes[0])

    env = TwoState(TwoStateConfig(T=5))
    runs = {"Transport": "results/twostate/transport_lambda_0.05_eta_0.85_T_5_exact_seed_0",
            "MF-REINFORCE": "results/ts_mfr_tuned/twostate/mfreinforce_eps_0.2_T_5_exact_seed_0",
            "REINFORCE": "results/twostate/reinforce_none_T_5_exact_seed_0"}
    steps = np.arange(env.config.T + 1)
    for name, theta in [(r"$\theta^\star$", env.optimal_theta())] + [
            (m, tensor_policy(ROOT / q)) for m, q in runs.items()]:
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
        return law.numpy()

    axes[2].plot(states, terminal(env.optimal_policy()), color=OPTIMAL, linewidth=1.0,
                 dashes=(3, 2), label=r"$\theta^\star$")
    for method, (folder, stem) in {
        "Transport": ("results/tuned_scales/distribution", "transport_lambda_0.2_eta_0.98_T_5_exact"),
        "MF-REINFORCE": ("results/distribution", "mfreinforce_eps_2_T_5_exact"),
        "REINFORCE": ("results/distribution", "reinforce_none_T_5_exact"),
    }.items():
        module = DistributionPolicy(env.config)
        blob = torch.load(ROOT / folder / f"{stem}_seed_0" / "policy.pt", map_location="cpu", weights_only=False)
        module.load_state_dict(blob["state_dict"])
        axes[2].plot(states, terminal(lambda t, mu: module(torch.tensor(float(t)), mu)),
                     color=METHOD_COLOR[method], linewidth=1.1, marker=METHOD_MARKER[method],
                     markersize=3.0, label=method)
    axes[2].set_title(r"Distribution: terminal law", fontsize=8, color=INK, pad=3)
    axes[2].set_xlabel("state", fontsize=7.5)
    axes[2].set_ylabel(r"$\mu_T^\theta(x)$", fontsize=7.5)
    style(axes[2])

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(Path(output).with_suffix(".png"), dpi=200, bbox_inches="tight")


def appendix_benchmarks(output):
    """Validation objective on the two benchmarks reported in the appendix.

    Neither separates the estimators. The objective is plotted directly rather than as an
    optimality gap because cybersecurity has no closed-form optimum.
    """
    panels = [
        ("Cybersecurity", "results/cybersecurity", None, {
            "REINFORCE": "reinforce_none_T_3_exact",
            "MF-REINFORCE": "mfreinforce_eps_1_T_3_exact",
            "Transport": "transport_lambda_0.4_eta_0.85_T_3_exact"}, r"$\lambda=0.4$"),
        ("Advertising", "results/advertising", 1.006168, {
            "REINFORCE": "reinforce_none_T_5_exact",
            "MF-REINFORCE": "mfreinforce_eps_1_T_5_exact",
            "Transport": "transport_lambda_0.2_eta_0.85_T_5_exact"}, r"$\lambda=0.2$"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), constrained_layout=True)
    for ax, (title, directory, optimum, runs, config) in zip(axes, panels):
        for method, stem in runs.items():
            seeds = curves(ROOT / directory, stem)
            if seeds is None:
                continue
            steps = np.arange(1, seeds.shape[1] + 1) * 10
            mean, deviation = seeds.mean(axis=0), seeds.std(axis=0)
            ax.plot(steps, mean, color=METHOD_COLOR[method], linewidth=1.1, label=method)
            ax.fill_between(steps, mean - deviation, mean + deviation,
                            color=METHOD_COLOR[method], alpha=0.20, linewidth=0)
        if optimum is not None:
            ax.axhline(optimum, color=OPTIMAL, linewidth=0.9, dashes=(3, 2))
        ax.set_title(title, fontsize=8, color=INK, pad=12)
        ax.text(0.5, 1.015, config, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.4, color=MUTED)
        ax.set_xlabel("policy updates", fontsize=7.5)
        ax.set_ylabel(r"$J(\theta)$", fontsize=7.5)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=OPTIMAL, linewidth=0.9, dashes=(3, 2)))
    labels.append(r"$J(\theta^\star)$")
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(Path(output).with_suffix(".png"), dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="files/iclr2027/figures")
    args = parser.parse_args()
    out = ROOT / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.labelsize": 8, "legend.fontsize": 7.2})
    learning_curves(out / "learning_curves.pdf")
    print("wrote learning_curves.pdf")
    learned_versus_optimal(out / "learned_policies.pdf")
    print("wrote learned_policies.pdf")
    appendix_benchmarks(out / "appendix_benchmarks.pdf")
    print("wrote appendix_benchmarks.pdf")


if __name__ == "__main__":
    main()
