"""Create the final figures and tables from saved runs.

    uv run python scripts/make_outputs.py --results-root results

Outputs are written under ``outputs/`` by default:

    outputs/figures/theory_verification.pdf
    outputs/tables/objective_summary.tex
    outputs/tables/budget_runtime.tex
"""

import argparse
import json
import re
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
import verify_bounds
from mfc.algorithms.discrete_validation import mean_field_next_law
from mfc.environments import (
    LQ,
    LQConfig,
    Advertising,
    AdvertisingConfig,
    AdvertisingPolicy,
    Cybersecurity,
    CybersecurityConfig,
    CybersecurityPolicy,
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
# Continuous-state benchmarks carry the mixture chart and the Gaussian-manifold
# arm; the other groups name the figure each set of benchmarks appears on.
CONTINUOUS_ENVS = ["lq", "portfolio"]
MAIN_ENVS = ["distribution", "portfolio"]
APPENDIX_MAIN_ENVS = ["lq", "twostate"]
APPENDIX_ENVS = ["cybersecurity", "advertising"]
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
METHOD_COLOR = {"REINFORCE": "#eb6834", "MF-REINFORCE": "#eda100", "Transport": "#2a78d6",
                "Transport-Proba": "#7b3fbf"}
METHOD_MARKER = {"REINFORCE": "s", "MF-REINFORCE": "D", "Transport": "o", "Transport-Proba": "^"}
OPTIMAL = "#52514e"
MFQ_COLOR = "#1baf7a"
TRADEOFF_COLOR = {"lq": "#2a78d6", "portfolio": "#eb6834", "twostate": "#1baf7a", "distribution": "#eda100"}
TRADEOFF_MARKER = {"lq": "o", "portfolio": "s", "twostate": "^", "distribution": "D"}
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


def env_dir(results_root, env):
    root = Path(results_root)
    direct = root / env
    if direct.exists():
        return direct
    partial = root / f"{env}_partial" / env
    if partial.exists():
        return partial
    partial = root / f"{env}_partial"
    if partial.exists():
        return partial
    return direct


def has_run(results_root, env):
    return any(env_dir(results_root, env).glob("*/history.json"))


def load_env_runs(results_root, env):
    """Budget-matched runs of one benchmark.

    The converged tabular reference is not budget matched, so it is kept out of
    the comparison tables and read only by mfq_reference.
    """
    directory = env_dir(results_root, env)
    runs = load_runs(directory.parent, env=directory.name) if directory.name == env else load_runs(directory)
    return [run for run in runs if not run["path"].name.startswith("mfqlearning_converged")]


def transport_stem(results_root, env, flow=None, components=None):
    """Best transport configuration present for this benchmark.

    The scales are read off the runs rather than recomputed from the grid rule, so that a
    results tree assembled from several sweeps is reported at whichever configuration it
    actually contains.
    """
    flow = MAIN_FLOW[env] if flow is None else flow
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    suffix = f"_T_{horizon}_{flow}"
    scores = {}
    for path in env_dir(results_root, env).glob(f"transport_*{suffix}_seed_*"):
        summary = path / "summary.json"
        if not summary.exists():
            continue
        stem = re.sub(r"_seed_\d+$", "", path.name)
        if components is not None and f"_K_{components}_" not in stem:
            continue
        value = json.loads(summary.read_text()).get("last_validation_objective")
        if value is not None:
            scores.setdefault(stem, []).append(value)
    if not scores:
        return None
    return max(scores, key=lambda stem: sum(scores[stem]) / len(scores[stem]))


def gaussian_stem(results_root, env):
    """Best Gaussian-manifold transport configuration present for this benchmark."""
    if env not in CONTINUOUS_ENVS:
        return None
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    scores = {}
    for path in env_dir(results_root, env).glob(f"gaussian_*_T_{horizon}_*_seed_*"):
        summary = path / "summary.json"
        if not summary.exists():
            continue
        value = json.loads(summary.read_text()).get("last_validation_objective")
        if value is not None:
            scores.setdefault(re.sub(r"_seed_\d+$", "", path.name), []).append(value)
    if not scores:
        return None
    return max(scores, key=lambda stem: sum(scores[stem]) / len(scores[stem]))


def transport_scales(stem):
    """The lambda and eta a run stem was trained at.

    The Gaussian-manifold arm has no auxiliary radius, so its stem carries a
    lambda and no eta; eta comes back as None there rather than failing the
    whole match.
    """
    match = re.search(r"lambda_([0-9.]+)", stem or "")
    if match is None:
        return (None, None)
    eta = re.search(r"_eta_([0-9.]+)", stem)
    return (float(match.group(1)), float(eta.group(1)) if eta else None)


def run_stems(results_root, env, include_gaussian=False):
    """Run stems of the methods drawn for one benchmark.

    The Gaussian-manifold arm is reported on its own appendix figure rather than
    alongside the main comparison, so it is opt-in here: every other figure asks
    for the three published methods and gets exactly those.
    """
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
    components = 1 if env in CONTINUOUS_ENVS else None
    stem = transport_stem(results_root, env, components=components)
    if stem is not None:
        stems["Transport"] = stem
    if include_gaussian:
        stem = gaussian_stem(results_root, env)
        if stem is not None:
            stems["Transport-Proba"] = stem
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


def thousands_tick(value, _position=None):
    """Update counts as 10k rather than 10000, which does not fit four across."""
    if value <= 0:
        return "0"
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


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


def scale_label(results_root, env, include_gaussian=False):
    lambda_, eta = transport_scales(transport_stem(results_root, env))
    parts = [] if lambda_ is None or eta is None else [rf"$\lambda={lambda_:.3g},\ \eta={eta:g}$"]
    if include_gaussian:
        proba, _ = transport_scales(gaussian_stem(results_root, env))
        if proba is not None:
            parts.append(rf"$\lambda_{{\mathrm{{proba}}}}={proba:.3g}$")
    return "   ".join(parts)


TITLE_OF = {env: title for title, (env, _) in LEARNING_PANELS.items()}
OPTIMUM_OF = {env: optimum for _, (env, optimum) in LEARNING_PANELS.items()}


def draw_learning_panel(ax, results_root, env, legend_only=False, include_gaussian=False):
    """Optimality gap against policy updates for one benchmark."""
    optimum = OPTIMUM_OF[env]
    panel = []
    for method, stem in run_stems(results_root, env, include_gaussian).items():
        seeds = curves(env_dir(results_root, env), stem)
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
    ax.set_title(TITLE_OF[env].replace("--", "-"), fontsize=8, color=INK, pad=12)
    ax.text(0.5, 1.015, scale_label(results_root, env, include_gaussian), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.4, color=MUTED)
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


def figure_legend(figure, axes, order, ncol):
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    names = [name for name in order if name in found]
    names += [name for name in found if name not in names]
    figure.legend([found[name] for name in names], names, loc="outside lower center",
                  ncol=ncol, frameon=False, handlelength=1.8)


FLOW_ORDER = (r"$\theta^\star$", "Transport", "MF-REINFORCE", "REINFORCE")
# Only the continuous appendix figure carries the Gaussian-manifold arm.
PROBA_ORDER = (r"$\theta^\star$", "Transport", "Transport-Proba", "REINFORCE")


def tensor_policy(path):
    blob = torch.load(path / "policy.pt", map_location="cpu", weights_only=False)
    return blob["tensor"] if "tensor" in blob else blob


def draw_portfolio_flow(ax, results_root, include_gaussian=False):
    env = Portfolio(PortfolioConfig(T=10, device="cpu"))
    with torch.no_grad():
        optimal, _ = env.moment_flow(env.optimal_policy(), lambda_=0.0)
    steps = np.arange(optimal.numel())
    ax.plot(steps, optimal.cpu().numpy(), color=OPTIMAL, linewidth=1.0, dashes=(3, 2), label=r"$\theta^\star$")
    portfolio_dir = env_dir(results_root, "portfolio")
    drawn = {"Transport": transport_stem(results_root, "portfolio", components=1),
             "REINFORCE": "reinforce_none_T_10_exact"}
    if include_gaussian:
        drawn["Transport-Proba"] = gaussian_stem(results_root, "portfolio")
    for method, stem in drawn.items():
        if stem is None:
            continue
        path = portfolio_dir / f"{stem}_seed_0"
        if not (path / "policy.pt").exists():
            continue
        with torch.no_grad():
            flow, _ = env.moment_flow(tensor_policy(path), lambda_=0.0)
        ax.plot(steps, flow.cpu().numpy(), color=METHOD_COLOR[method], linewidth=1.2, label=method)
    ax.set_title("Portfolio: mean wealth flow", fontsize=8, color=INK, pad=3)
    ax.set_xlabel("$t$", fontsize=7.5)
    ax.set_ylabel(r"$\bar x(\mu_t^\theta)$", fontsize=7.5)
    style(ax)


def draw_lq_flow(ax, results_root, include_gaussian=False):
    """Mean flow of the linear--quadratic population under the learned policies.

    Every estimator drives the mean to zero here, so the panel is read on a log
    scale: what separates the policies is how fast they do it and where the
    flow settles, not the direction it takes.
    """
    env = LQ(LQConfig(T=20, device="cpu"))
    with torch.no_grad():
        optimal, _ = env.moment_flow(env.optimal_policy(), lambda_=0.0)
    steps = np.arange(optimal.numel())
    floor = 1e-6
    ax.plot(steps, np.maximum(np.abs(optimal.cpu().numpy()), floor), color=OPTIMAL,
            linewidth=1.0, dashes=(3, 2), label=r"$\theta^\star$")
    lq_dir = env_dir(results_root, "lq")
    drawn = {"Transport": transport_stem(results_root, "lq", components=1),
             "REINFORCE": "reinforce_none_T_20_exact"}
    if include_gaussian:
        drawn["Transport-Proba"] = gaussian_stem(results_root, "lq")
    for method, stem in drawn.items():
        if stem is None:
            continue
        path = lq_dir / f"{stem}_seed_0"
        if not (path / "policy.pt").exists():
            continue
        with torch.no_grad():
            flow, _ = env.moment_flow(tensor_policy(path), lambda_=0.0)
        ax.plot(steps, np.maximum(np.abs(flow.cpu().numpy()), floor),
                color=METHOD_COLOR[method], linewidth=1.2, label=method)
    ax.set_yscale("log")
    ax.set_title("Linear-quadratic: mean flow", fontsize=8, color=INK, pad=3)
    ax.set_xlabel("$t$", fontsize=7.5)
    ax.set_ylabel(r"$|\bar x(\mu_t^\theta)|$", fontsize=7.5)
    style(ax)


def draw_twostate_flow(ax, results_root, include_gaussian=False):
    env = TwoState(TwoStateConfig(T=5, device="cpu"))
    run_dir = env_dir(results_root, "twostate")
    runs = {
        "Transport": run_dir / f"{transport_stem(results_root, 'twostate')}_seed_0",
        "MF-REINFORCE": run_dir / "mfreinforce_eps_0.2_T_5_exact_seed_0",
        "REINFORCE": run_dir / "reinforce_none_T_5_exact_seed_0",
    }
    steps = np.arange(env.config.T + 1)
    policies = [(m, tensor_policy(q)) for m, q in runs.items() if (q / "policy.pt").exists()]
    for name, theta in [(r"$\theta^\star$", env.optimal_theta())] + policies:
        law = env.initial_distribution.clone()
        trace = [float(law[1])]
        with torch.no_grad():
            for t in range(env.config.T):
                law = mean_field_next_law(env, theta, lambda step: step, t, law)
                trace.append(float(law[1]))
        colour = OPTIMAL if name.startswith("$") else METHOD_COLOR[name]
        dashes = (3, 2) if name.startswith("$") else (None, None)
        ax.plot(steps, trace, color=colour, linewidth=1.1, dashes=dashes, label=name)
    ax.set_title("Two-state: population flow", fontsize=8, color=INK, pad=3)
    ax.set_xlabel("$t$", fontsize=7.5)
    ax.set_ylabel(r"$\mu_t^\theta(1)$", fontsize=7.5)
    style(ax)


def draw_distribution_flow(ax, results_root, include_gaussian=False):
    env = Distribution(DistributionConfig(device="cpu"))
    states = np.arange(env.n_states)

    def terminal(policy):
        law = env.initial_distribution.clone()
        with torch.no_grad():
            for t in range(env.config.T):
                law = env.population_step(law, policy(t, law))
        return law.cpu().numpy()

    ax.plot(states, terminal(env.optimal_policy()), color=OPTIMAL, linewidth=1.0,
            dashes=(3, 2), label=r"$\theta^\star$")
    run_dir = env_dir(results_root, "distribution")
    for method, stem in {
        "Transport": transport_stem(results_root, "distribution"),
        "MF-REINFORCE": "mfreinforce_eps_2_T_5_exact",
        "REINFORCE": "reinforce_none_T_5_exact",
    }.items():
        folder = run_dir / f"{stem}_seed_0"
        if not (folder / "policy.pt").exists():
            continue
        module = DistributionPolicy(env.config)
        blob = torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)
        module.load_state_dict(blob["state_dict"])
        ax.plot(states, terminal(lambda t, mu: module(torch.tensor(float(t)), mu)),
                color=METHOD_COLOR[method], linewidth=1.1, marker=METHOD_MARKER[method],
                markersize=3.0, label=method)
    ax.set_title("Distribution: terminal law", fontsize=8, color=INK, pad=3)
    ax.set_xlabel("state", fontsize=7.5)
    ax.set_ylabel(r"$\mu_T^\theta(x)$", fontsize=7.5)
    style(ax)


FLOW_PANEL = {
    "lq": draw_lq_flow,
    "portfolio": draw_portfolio_flow,
    "twostate": draw_twostate_flow,
    "distribution": draw_distribution_flow,
}


def main_benchmarks(results_root, output):
    """Main-text panel: one horizontal row, two panels per benchmark."""
    envs = [env for env in MAIN_ENVS if has_run(results_root, env)]
    if not envs:
        print("skipping main_benchmarks: no matching runs found")
        return
    figure, axes = plt.subplots(1, 2 * len(envs), figsize=(5.5, 1.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for index, env in enumerate(envs):
        draw_learning_panel(axes[2 * index], results_root, env)
        FLOW_PANEL[env](axes[2 * index + 1], results_root)
    # Four panels across the text width leave no room for the default tick density.
    titles = [f"{TITLE_OF[env].replace('--', '-')}: {suffix}"
              for env in envs for suffix in ("optimality gap", None)]
    for ax, title in zip(axes, titles):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        if title.endswith("None"):
            title = ax.get_title()
        ax.set_title(title, fontsize=6.8, color=INK, pad=ax.title.get_position()[1] * 0 + 12)
        ax.tick_params(labelsize=6.0)
        ax.xaxis.label.set_size(6.8)
        ax.yaxis.label.set_size(6.8)
    figure_legend(figure, axes, FLOW_ORDER, 4)
    save_figure(figure, output)


def appendix_continuous(results_root, output):
    """Appendix panel: the two continuous benchmarks with the Gaussian-manifold arm.

    This is the only figure that draws Transport-Proba: two panels per benchmark,
    the optimality gap and the population behaviour it induces, laid out in one
    row across the text width. The linear-quadratic flow appears only here, the
    main text having no panel for it.
    """
    envs = [env for env in CONTINUOUS_ENVS if has_run(results_root, env)]
    if not envs:
        print("skipping appendix_continuous: no matching runs found")
        return

    panels = [(kind, env) for env in envs for kind in ("gap", "flow")]

    figure, axes = plt.subplots(1, len(panels), figsize=(5.5, 1.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    # Four across the text width leave no room for full names or for thousands
    # written out in the update counts.
    short = {"lq": "Linear-quadratic", "portfolio": "Portfolio"}
    for ax, (kind, env) in zip(axes, panels):
        if kind == "gap":
            draw_learning_panel(ax, results_root, env, include_gaussian=True)
            title = f"{short[env]}: optimality gap"
            # Gaps spanning three decades collide under the shared (1, 2, 5) locator.
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=5))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(thousands_tick))
        else:
            FLOW_PANEL[env](ax, results_root, include_gaussian=True)
            title = ax.get_title().replace("mean wealth flow", "wealth flow")
        ax.set_title(title, fontsize=6.2, color=INK, pad=12)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.tick_params(labelsize=5.8)
        ax.xaxis.label.set_size(6.4)
        ax.yaxis.label.set_size(6.4)
    figure_legend(figure, axes, PROBA_ORDER, 4)
    save_figure(figure, output)


def appendix_finite(results_root, output):
    """Appendix panel: the two benchmarks moved out of the main text."""
    envs = [env for env in APPENDIX_MAIN_ENVS if has_run(results_root, env)]
    if not envs:
        print("skipping appendix_finite: no matching runs found")
        return
    flows = [env for env in envs if env in FLOW_PANEL]
    figure, axes = plt.subplots(1, len(envs) + len(flows), figsize=(2.3 * (len(envs) + len(flows)), 2.1),
                                constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, env in zip(axes, envs):
        draw_learning_panel(ax, results_root, env)
    for ax, env in zip(axes[len(envs):], flows):
        FLOW_PANEL[env](ax, results_root)
    figure_legend(figure, axes, FLOW_ORDER, 4)
    save_figure(figure, output)


# The tabular reference is the converged MFQ run, not the budget-matched one.
MFQ_REFERENCE_STEM = "mfqlearning_converged_*_seed_*"


def mlp_policy(module, folder):
    blob = torch.load(Path(folder) / "policy.pt", map_location="cpu", weights_only=False)
    module.load_state_dict(blob["state_dict"])
    module.eval()
    return lambda t, mu: module(torch.tensor(float(t)), mu)


def mfq_reference(results_root, env):
    values = [
        json.loads(path.read_text())["last_validation_objective"]
        for path in sorted(env_dir(results_root, env).glob(f"{MFQ_REFERENCE_STEM}/summary.json"))
    ]
    return float(np.mean(values)) if values else None


def appendix_benchmarks(results_root, output):
    """Learning curves and induced population flows for the two undiscriminating benchmarks."""
    envs = [env for env in APPENDIX_ENVS if has_run(results_root, env)]
    if not envs:
        print("skipping appendix_benchmarks: no matching runs found")
        return

    figure, axes = plt.subplots(2, len(envs), figsize=(2.75 * len(envs), 4.0), constrained_layout=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 2:
        axes = axes.T

    for column, env in enumerate(envs):
        ax = axes[0, column]
        for method, stem in run_stems(results_root, env).items():
            seeds = curves(env_dir(results_root, env), stem)
            if seeds is None:
                continue
            steps = np.arange(1, seeds.shape[1] + 1) * 10
            mean, deviation = seeds.mean(axis=0), seeds.std(axis=0)
            ax.plot(steps, mean, color=METHOD_COLOR[method], linewidth=1.1, label=method)
            ax.fill_between(steps, mean - deviation, mean + deviation,
                            color=METHOD_COLOR[method], alpha=0.20, linewidth=0)
        optimum = REFERENCE_OPTIMUM.get((env, run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]))
        if optimum is not None:
            ax.axhline(optimum, color=OPTIMAL, linewidth=1.0, dashes=(3, 2), label=r"$\theta^\star$")
        elif env == "cybersecurity":
            reference = mfq_reference(results_root, env)
            if reference is not None:
                # A separate style: the tabular reference is not an optimum, only a baseline.
                ax.axhline(reference, color=MFQ_COLOR, linewidth=1.0, dashes=(1, 1.6), label="MFQ-learning")
        ax.set_title(DISPLAY[env].replace("--", "-"), fontsize=8, color=INK, pad=12)
        ax.text(0.5, 1.015, scale_label(results_root, env), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.4, color=MUTED)
        ax.set_xlabel("policy updates", fontsize=7.5)
        ax.set_ylabel(r"$J(\theta)$", fontsize=7.5)
        style(ax)

    for column, env in enumerate(envs):
        ax = axes[1, column]
        directory = env_dir(results_root, env)
        if env == "cybersecurity":
            simulator = Cybersecurity(CybersecurityConfig(T=3, device="cpu"))
            module_class, config = CybersecurityPolicy, simulator.config
            readout = lambda law: float(law[simulator.DI] + law[simulator.UI])
            title, ylabel = "Cybersecurity: infected mass", r"$\mu_t^\theta(\mathrm{DI})+\mu_t^\theta(\mathrm{UI})$"
            optimal_theta = None
        else:
            simulator = Advertising(AdvertisingConfig(T=5, device="cpu"))
            module_class, config = AdvertisingPolicy, simulator.config
            readout = lambda law: float(law[simulator.CUSTOMER])
            title, ylabel = "Advertising: population flow", r"$\mu_t^\theta(1)$"
            optimal_theta = simulator.optimal_policy()

        horizon = simulator.config.T
        steps = np.arange(horizon + 1)

        def trace(policy):
            law = simulator.initial_distribution.clone()
            values = [readout(law)]
            for t in range(horizon):
                law = mean_field_next_law(simulator, policy, lambda step: step, t, law)
                values.append(readout(law))
            return values

        if optimal_theta is not None:
            ax.plot(steps, trace(optimal_theta), color=OPTIMAL, linewidth=1.0,
                    dashes=(3, 2), label=r"$\theta^\star$")
        for method, stem in run_stems(results_root, env).items():
            folder = directory / f"{stem}_seed_0"
            if not (folder / "policy.pt").exists():
                continue
            ax.plot(steps, trace(mlp_policy(module_class(config), folder)),
                    color=METHOD_COLOR[method], linewidth=1.1,
                    marker=METHOD_MARKER[method], markersize=3.0, label=method)
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        ax.set_xlabel("$t$", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_xticks(steps)
        style(ax)

    found = {}
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    order = [name for name in ("REINFORCE", "MF-REINFORCE", "Transport", r"$\theta^\star$", "MFQ-learning") if name in found]
    order += [name for name in found if name not in order]
    figure.legend([found[name] for name in order], order, loc="outside lower center",
                  ncol=len(order), frameon=False, handlelength=1.8)
    save_figure(figure, output)


# Sweeps run at an auxiliary radius other than the headline one, selected by name.
TRADEOFF_FILTER = {"twostate": "eta_0.85", "distribution": "eta_0.85"}


def lambda_sweep(results_root, env, filter_text, flow, horizon):
    """Optimality gap of every transport lambda available for this benchmark."""
    directory = env_dir(results_root, env)
    points = {}
    for path in directory.glob(f"transport_*_T_{horizon}_{flow}_seed_*"):
        stem = re.sub(r"_seed_\d+$", "", path.name)
        if filter_text is not None and filter_text not in stem:
            continue
        if "_K_" in stem and "_K_1_" not in stem:
            continue
        summary = path / "summary.json"
        if not summary.exists():
            continue
        lambda_ = float(re.search(r"lambda_([0-9.]+)", stem).group(1))
        points.setdefault(lambda_, []).append(
            json.loads(summary.read_text())["last_validation_objective"])
    return points


def lambda_tradeoff(results_root, output):
    figure, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    drawn = 0
    for title, (env, optimum) in LEARNING_PANELS.items():
        horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
        points = lambda_sweep(results_root, env, TRADEOFF_FILTER.get(env), MAIN_FLOW[env], horizon)
        if len(points) < 2:
            print(f"skipping {env} in lambda_tradeoff: fewer than two scales available")
            continue
        scales = sorted(points)
        gaps = [abs(sum(points[s]) / len(points[s]) - optimum) for s in scales]
        best = min(gaps)
        colour = TRADEOFF_COLOR[env]
        ax.plot(scales, [g / best for g in gaps], color=colour, marker=TRADEOFF_MARKER[env],
                markersize=3.4, linewidth=1.1, label=title.replace("--", "\u2013"))
        drawn += 1
    if not drawn:
        print("skipping lambda_tradeoff: no sweeps found")
        return
    ax.axhline(1.0, color=MUTED, linewidth=0.7, dashes=(1.2, 1.6), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("optimality gap / best gap")
    style(ax)
    # Explicit decade-free ticks: the default log locator collides on this narrow range.
    ticks = [0.05, 0.1, 0.2, 0.4, 0.8]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.75, ax.get_ylim()[1] * 2.2)
    ax.legend(frameon=False, fontsize=6.6, handlelength=1.6, loc="upper left", ncol=2,
              columnspacing=1.0, borderaxespad=0.2)
    save_figure(figure, output)


def theory_verification(estimate_csv, consistency_csv, output):
    if not Path(estimate_csv).exists() or not Path(consistency_csv).exists():
        print("skipping theory_verification: diagnostic CSVs not found")
        return

    display = {"twostate": "Two-state", "cybersecurity": "Cybersecurity", "distribution": "Distribution",
               "advertising": "Advertising", "lq": "Linear-quadratic", "portfolio": "Portfolio"}
    order = ["lq", "portfolio", "twostate", "distribution", "cybersecurity", "advertising"]
    color = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    marker = dict(zip(order, ["o", "s", "^", "D", "v", "P"]))
    ticks = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]

    estimate = pd.read_csv(estimate_csv)
    consistency = pd.read_csv(consistency_csv)
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)
    # Both state spaces are divided by lambda, so a flat line means the deviation is
    # proportional to lambda whichever metric it is measured in. The finite-state bound is
    # lambda and is attained; the continuous-state bound is only sqrt(lambda), so a flat
    # continuous line means the measured deviation is of lower order than guaranteed.
    deviation = estimate["max_tv"].where(estimate["space"] == "finite", estimate["w1_rms"])
    estimate = estimate.assign(deviation_over_lambda=deviation / estimate["lambda"])
    for name in order:
        rows = estimate[estimate["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        axes[0].plot(rows["lambda"], rows["deviation_over_lambda"], color=color[name], marker=marker[name],
                     markersize=3.2, linewidth=1.1, dashes=dashes, label=display[name])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, max(3.2, float(estimate["deviation_over_lambda"].max()) * 1.15))
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$d_{\mathrm{TV}}/\lambda$   or   $\mathcal{W}_2/\lambda$")
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


BOUND_COLOR = {"lq": "#2a78d6", "portfolio": "#eb6834"}
BOUND_MARKER = {"lq": "o", "portfolio": "s"}


def bound_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def guide(ax, x, y, exponent, anchor=0):
    """A dashed line of the predicted slope, anchored on the measured series."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    scale = y[anchor] / x[anchor] ** exponent
    ax.plot(x, scale * x**exponent, color=MUTED, linewidth=0.7, dashes=(2, 2), zorder=1)


def _bounds_table(bounds_csv):
    if not Path(bounds_csv).exists():
        return None, []
    table = pd.read_csv(bounds_csv)
    return table, [env for env in ("lq", "portfolio") if (table["env"] == env).any()]


def _annotate_slope(ax, x, y, reference, fitted=None):
    """Print the fitted exponent, in whichever corner the curve leaves free."""
    exponent = bound_slope(x, y) if fitted is None else fitted
    height = 0.88 if reference > 0 else 0.06
    ax.text(
        0.04, height, rf"$p={exponent:+.2f}$   $({reference:+.0f})$",
        transform=ax.transAxes, fontsize=5.6, color=INK,
    )


def _finish(ax, xlabel, ylabel, title=None):
    style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=6.6)
    ax.set_ylabel(ylabel, fontsize=6.6)
    ax.tick_params(labelsize=5.8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if title:
        ax.set_title(title, fontsize=7.0, color=INK, pad=4)


def bounds_radius(bounds_csv, output):
    """The auxiliary radius: total sensitivity error against its centred-difference part."""
    table, envs = _bounds_table(bounds_csv)
    if not envs:
        print("skipping bounds_radius: no bounds CSV")
        return
    figure, axes = plt.subplots(1, len(envs), figsize=(5.5, 2.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, env in zip(axes, envs):
        rows = table[table["env"] == env]
        total = rows[rows["sweep"] == "eta"].sort_values("value")
        taylor = rows[rows["sweep"] == "eta-exact"].sort_values("value")
        ax.plot(total["value"], total["error"], color=BOUND_COLOR[env], marker=BOUND_MARKER[env],
                markersize=3.0, linewidth=1.1, label=r"$\mathbb{E}\|\widehat D-D\|^2$")
        ax.plot(taylor["value"], taylor["error"], color=OPTIMAL, marker="^", markersize=3.0,
                linewidth=1.1, dashes=(3, 2), label=r"$\|D_\eta-D\|^2$")
        auxiliary = verify_bounds.BENCHMARKS[env][5]
        ax.axvline(auxiliary ** (-1 / 6), color=MUTED, linewidth=0.8, dashes=(1.2, 1.6), zorder=1)
        ax.annotate(r"$\eta=n^{-1/6}$", xy=(auxiliary ** (-1 / 6), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -8), textcoords="offset points", fontsize=5.8, color=MUTED)
        _finish(ax, r"$\eta$", "squared error", DISPLAY[env].replace("--", "-"))
    figure_legend(figure, axes, (), 2)
    save_figure(figure, output)


RATE_PANELS = [
    ("n", "n", "error", r"$n$", r"$\mathbb{E}\|\widehat D-D\|^2$", -1.0),
    ("M", "M", "error", r"$M$", r"$\mathbb{E}\|\widehat z-z\|^2$", -1.0),
    ("lambda", "lambda", "bias", r"$\lambda$", r"$\|\mathbb{E}\widehat G-\nabla J\|$", 1.0),
    ("B", "B", "variance", r"$B$", r"$\mathbb{E}\|\widehat G-\mathbb{E}\widehat G\|^2$", -1.0),
]


def bounds_rates(bounds_csv, output):
    """One sweep per panel: each block size, the scale, and the main block."""
    table, envs = _bounds_table(bounds_csv)
    if not envs:
        print("skipping bounds_rates: no bounds CSV")
        return
    figure, axes = plt.subplots(len(envs), len(RATE_PANELS), figsize=(5.5, 1.75 * len(envs)),
                                constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, env in enumerate(envs):
        rows = table[table["env"] == env]
        for column, (sweep, _knob, field, xlabel, ylabel, exponent) in enumerate(RATE_PANELS):
            ax = axes[row][column]
            block = rows[rows["sweep"] == sweep]
            if sweep == "B" and not block.empty:
                block = block[block["lambda_"] == block["lambda_"].min()]
            block = block.sort_values("value")
            if block.empty:
                continue
            x, y = block["value"].to_numpy(), block[field].to_numpy()
            ax.plot(x, y, color=BOUND_COLOR[env], marker=BOUND_MARKER[env], markersize=3.0, linewidth=1.1)
            guide(ax, x, y, exponent)
            # The scale sweep saturates at its top, so its exponent is fitted on
            # the lower half of the grid, as in the table.
            keep = x <= np.median(x) if sweep == "lambda" else np.ones_like(x, dtype=bool)
            _annotate_slope(ax, x, y, exponent, fitted=bound_slope(x[keep], y[keep]))
            _finish(ax, xlabel, ylabel, DISPLAY[env].replace("--", "-") if column == 0 else None)
    save_figure(figure, output)


BOUND_SWEEPS = [
    ("eta-exact", r"$\eta$, analytic flows", r"$\mathbb{E}\|\widehat D-D\|^2$", "error", 4.0),
    ("n", r"$n$", r"$\mathbb{E}\|\widehat D-D\|^2$", "error", -1.0),
    ("lambda", r"$\lambda$", r"$\|\mathbb{E}\widehat G-\nabla J\|$", "bias", 1.0),
    ("B", r"$B$", "conditional variance", "variance", -1.0),
    ("M", r"$M$", r"$\mathbb{E}\|\widehat z-z\|^2$", "error", -1.0),
]


def bounds_exponents(bounds_csv):
    """Fitted log-log exponents of every sweep against the rate the bound predicts."""
    if not Path(bounds_csv).exists():
        return None
    table = pd.read_csv(bounds_csv)
    envs = [env for env in ("lq", "portfolio") if (table["env"] == env).any()]
    header = " & ".join(["Quantity", "Swept"] + [DISPLAY[env] for env in envs] + ["$p_0$"])
    lines = [header + " \\\\", "\\midrule"]
    for sweep, knob, quantity, column, exponent in BOUND_SWEEPS:
        cells = []
        for env in envs:
            rows = table[(table["env"] == env) & (table["sweep"] == sweep)]
            if sweep == "B" and not rows.empty:
                rows = rows[rows["lambda_"] == rows["lambda_"].min()]
            if sweep == "lambda" and not rows.empty:
                rows = rows[rows["value"] <= rows["value"].median()]
            cells.append("---" if rows.empty else f"${bound_slope(rows['value'], rows[column]):+.2f}$")
        lines.append(" & ".join([quantity, knob] + cells + [f"${exponent:+.0f}$"]) + " \\\\")
    caption = (
        "Slope $p$ of each error against the quantity swept, fitted by least squares on a log-log "
        "scale at $\\theta=\\tfrac12\\theta^\\star$, with $p_0$ the slope of the mechanism "
        "producing it. The $\\lambda$ row is fitted on $\\lambda\\leq0.1$."
    )
    return table_environment("\n".join(lines), caption, "tab:bounds-exponents",
                             "ll" + "r" * len(envs) + "r", size="\\small")


def method_of(label):
    if label.startswith("REINFORCE"):
        return "reinforce"
    if label.startswith("Transport-Proba"):
        return "gaussian"
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
    table = objective_table(load_env_runs(results_root, env))
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
    table = objective_table(load_env_runs(results_root, env))
    if "J0_star" not in table.columns:
        return None
    values = table.loc[table["horizon"] == horizon, "J0_star"].dropna()
    if values.empty:
        return None
    convention = table["objective_convention"].dropna().iloc[0]
    return -float(values.iloc[0]) if convention == "cost" else float(values.iloc[0])


def objective_summary(results_root):
    lines = [
        "Benchmark & Optimum & REINFORCE & MF-REINFORCE & Transport & $(\\lambda,\\eta)$ "
        "& Transport-Proba & $\\lambda$ \\\\",
        "\\midrule",
    ]
    for env in [name for name in BENCHMARKS if has_run(results_root, name)]:
        grouped, _ = grouped_objectives(results_root, env)
        best = select_headline(grouped, env)
        optimum = reference_optimum(results_root, env)
        digits = 3 if env == "portfolio" else 4
        transport = best.get("transport")
        # Read off the run being reported rather than recomputing the grid rule, which
        # need not name a configuration this results tree actually contains.
        components = 1 if env in CONTINUOUS_ENVS else None
        lambda_, eta = transport_scales(transport_stem(results_root, env, components=components))
        scales = (
            "---"
            if transport is None or lambda_ is None or eta is None
            else f"$({lambda_:g},{eta:g})$"
        )
        gaussian = best.get("gaussian")
        proba_lambda, _ = transport_scales(gaussian_stem(results_root, env))
        proba_scale = "---" if gaussian is None or proba_lambda is None else f"${proba_lambda:g}$"
        lines.append(
            " & ".join(
                [
                    f"{DISPLAY[env]} ($T={run_plan.TRANSPORT_ALLOCATIONS[env]['horizon']}$)",
                    "---" if optimum is None else f"${number(optimum, digits)}$",
                    with_error(best.get("reinforce"), digits),
                    with_error(best.get("mfreinforce"), digits),
                    with_error(transport, digits),
                    scales,
                    with_error(gaussian, digits),
                    proba_scale,
                ]
            )
            + " \\\\"
        )
    caption = (
        "Final validation objective on every benchmark, as mean and standard deviation over seeds. "
        "Higher is better. The transport column shows the best fixed-scale run among the bound-scale "
        "multipliers and the asymptotic anchor scales used for the benchmark."
    )
    return table_environment("\n".join(lines), caption, "tab:objective-summary", "lrrrrlrl", size="\\small")


ESTIMATORS = [
    ("reinforce", "REINFORCE"),
    ("mfreinforce", "MF-REINFORCE"),
    ("transport", "Transport"),
    ("gaussian", "Transport-Proba"),
    ("mfqlearning", "MFQ-learning"),
]
SCALE_COUNT = {"transport": "$\\lambda,\\eta$", "gaussian": "$\\lambda$"}


def runtime_rows(results_root, env, scales_column):
    """One LaTeX row per estimator present on this benchmark, at its headline scale."""
    runtime = runtime_table(load_env_runs(results_root, env))
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    subset = runtime[runtime["horizon"] == horizon]
    subset = subset[(subset["flow"] == MAIN_FLOW[env]) | (subset["algorithm"] == "reinforce")]
    reference = subset[subset["algorithm"] == "reinforce"]
    reference_seconds = float(reference["elapsed_seconds_mean"].iloc[0]) if not reference.empty else None

    lines = []
    for algorithm, name in ESTIMATORS:
        rows = subset[subset["algorithm"] == algorithm]
        if rows.empty:
            continue
        row = headline_row(rows, results_root, env, algorithm)
        seconds = float(row["elapsed_seconds_mean"])
        cells = [DISPLAY[env] if not lines else "", name]
        if scales_column:
            cells.append(SCALE_COUNT.get(algorithm, "---"))
        cells += [
            f"${float(row['simulator_budget_mean']):,.0f}$".replace(",", "\\,"),
            f"${seconds:,.0f}$".replace(",", "\\,"),
            "---" if reference_seconds is None else f"${seconds / reference_seconds:.1f}\\times$",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def runtime_summary(results_root, envs, scales_column, caption, label):
    header = ["Benchmark", "Estimator"] + (["Scales"] if scales_column else [])
    header += ["Simulator budget", "Wall clock (s)", "Ratio to REINFORCE"]
    lines = [" & ".join(header) + " \\\\", "\\midrule"]
    available = [env for env in envs if has_run(results_root, env)]
    for env in available:
        lines.extend(runtime_rows(results_root, env, scales_column))
        if env != available[-1]:
            lines.append("\\midrule")
    alignment = "ll" + ("l" if scales_column else "") + "rrr"
    return table_environment("\n".join(lines), caption, label, alignment, size="\\small")


def budget_runtime(results_root):
    return runtime_summary(
        results_root,
        BENCHMARKS,
        False,
        "Simulator budget and wall-clock cost per run at the headline configuration. "
        "Budgets are matched by construction; wall clock also reflects estimator arithmetic.",
        "tab:budget-runtime",
    )


def continuous_runtime(results_root):
    return runtime_summary(
        results_root,
        CONTINUOUS_ENVS,
        True,
        "Simulator budget per policy update and wall-clock cost of one run, as a mean over "
        "seeds. Budgets are matched by construction; wall clock also reflects estimator "
        "arithmetic. The scales column counts the perturbation hyperparameters each estimator "
        "carries.",
        "tab:continuous-runtime",
    )


CONTINUOUS_STEMS = [
    ("reinforce", "REINFORCE", "reinforce_none_T_{horizon}_exact"),
    ("transport", "Transport", "transport_lambda_{lambda_}_eta_{eta}_K_1_T_{horizon}_particle"),
    ("gaussian", "Transport-Proba", "gaussian_lambda_{lambda_}_T_{horizon}_particle"),
]


def scale_grid(results_root, env, algorithm):
    """Every perturbation scale this results tree holds for one continuous arm."""
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    found = set()
    for path in env_dir(results_root, env).glob(f"{algorithm}_lambda_*_T_{horizon}_particle_seed_*"):
        if algorithm == "transport" and "_K_1_" not in path.name:
            continue
        found.add(float(re.search(r"lambda_([0-9.]+)", path.name).group(1)))
    return sorted(found)


def seed_values(results_root, env, stem):
    values = [
        json.loads(path.read_text()).get("last_validation_objective")
        for path in sorted(env_dir(results_root, env).glob(f"{stem}_seed_*/summary.json"))
    ]
    return np.array([value for value in values if value is not None])


def continuous_comparison(results_root):
    """Every scale of both continuous arms, against REINFORCE and the optimum."""
    lines = [
        "Benchmark & Estimator & Scale & $J(\\widehat\\theta)$ & "
        "$|J(\\widehat\\theta)-J(\\theta^\\star)|$ \\\\",
        "\\midrule",
    ]
    available = [env for env in CONTINUOUS_ENVS if has_run(results_root, env)]
    for env in available:
        horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
        optimum = OPTIMUM_OF[env]
        digits = 3 if env == "portfolio" else 4
        eta = transport_scales(transport_stem(results_root, env, components=1))[1]
        first = True
        for algorithm, name, template in CONTINUOUS_STEMS:
            if algorithm == "reinforce":
                stems = [(None, template.format(horizon=horizon))]
            else:
                stems = [
                    (scale, template.format(lambda_=f"{scale:g}", eta=f"{eta:g}", horizon=horizon))
                    for scale in scale_grid(results_root, env, algorithm)
                ]
            rows = []
            for scale, stem in stems:
                values = seed_values(results_root, env, stem)
                if values.size == 0:
                    continue
                gaps = np.abs(values - optimum)
                rows.append((scale, values.mean(), values.std(ddof=1), gaps.mean(), gaps.std(ddof=1)))
            if not rows:
                continue
            best = min(row[3] for row in rows)
            for index, (scale, mean, deviation, gap, gap_deviation) in enumerate(rows):
                cell = f"${gap:.{digits}f}\\pm{gap_deviation:.{digits}f}$"
                if len(rows) > 1 and gap == best:
                    cell = f"$\\mathbf{{{gap:.{digits}f}\\pm{gap_deviation:.{digits}f}}}$"
                lines.append(
                    " & ".join([
                        f"{DISPLAY[env]} ($T={horizon}$)" if first else "",
                        name if index == 0 else "",
                        "---" if scale is None else f"${scale:g}$",
                        f"${mean:.{digits}f}\\pm{deviation:.{digits}f}$",
                        cell,
                    ]) + " \\\\"
                )
                first = False
            lines.append("\\addlinespace[2pt]")
        if lines[-1] == "\\addlinespace[2pt]":
            lines.pop()
        if env != available[-1]:
            lines.append("\\midrule")
    caption = (
        "Every perturbation scale of the two continuous-state estimators, at matched simulator "
        "budgets and over five paired seeds. Bold marks the best scale of each estimator. "
        "Higher $J$ is better."
    )
    return table_environment("\n".join(lines), caption, "tab:continuous-comparison", "llrrr", size="\\small")


def headline_row(rows, results_root, env, algorithm):
    """The runtime row of the configuration the objective tables report.

    Several scales of the same estimator share an algorithm, so the row has to
    be selected by the scale, not by whichever happened to run longest.
    """
    if algorithm == "transport":
        components = 1 if env in CONTINUOUS_ENVS else None
        lambda_, eta = transport_scales(transport_stem(results_root, env, components=components))
    elif algorithm == "gaussian":
        lambda_, eta = transport_scales(gaussian_stem(results_root, env))
    else:
        lambda_ = eta = None
    if lambda_ is not None:
        match = rows[np.isclose(rows["perturbation"].astype(float), lambda_)]
        if eta is not None and not match.empty and match["eta"].notna().any():
            match = match[np.isclose(match["eta"].astype(float), eta)]
        if not match.empty:
            rows = match
    return rows.iloc[rows["elapsed_seconds_mean"].to_numpy().argmax()]


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
    parser.add_argument("--bounds", default="results/figures/bounds/bounds.csv")
    args = parser.parse_args()

    results_root = ROOT / args.results_root
    output_root = ROOT / args.output_root
    figures = output_root / "figures"
    tables = output_root / "tables"

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.labelsize": 8, "legend.fontsize": 7.2})

    main_benchmarks(results_root, figures / "main_benchmarks.pdf")
    appendix_finite(results_root, figures / "appendix_finite.pdf")
    appendix_continuous(results_root, figures / "appendix_continuous.pdf")
    appendix_benchmarks(results_root, figures / "appendix_benchmarks.pdf")
    lambda_tradeoff(results_root, figures / "lambda_tradeoff.pdf")
    theory_verification(ROOT / args.theory_estimate, ROOT / args.theory_consistency, figures / "theory_verification.pdf")
    write_table(objective_summary(results_root), tables / "objective_summary.tex")
    write_table(budget_runtime(results_root), tables / "budget_runtime.tex")
    bounds_radius(ROOT / args.bounds, figures / "bounds_radius.pdf")
    bounds_rates(ROOT / args.bounds, figures / "bounds_rates.pdf")
    exponents = bounds_exponents(ROOT / args.bounds)
    if exponents is not None:
        write_table(exponents, tables / "bounds_exponents.tex")
    write_table(continuous_comparison(results_root), tables / "continuous_comparison.tex")
    write_table(continuous_runtime(results_root), tables / "continuous_runtime.tex")


if __name__ == "__main__":
    main()
