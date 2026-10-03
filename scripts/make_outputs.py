"""Create the final figures and tables from saved runs.

    uv run python scripts/make_outputs.py --results-root results

Outputs are written under ``outputs/`` by default:

    outputs/figures/main_benchmarks.pdf      main text, learning curves and flows
    outputs/figures/main_diagnostics.pdf     main text, scale trade-off and perturbation results
    outputs/tables/main_summary.tex          main text, optimality gaps
    outputs/tables/bounds_exponents_main.tex main text, fitted rates
    outputs/figures/theory_verification.pdf  appendix, every other figure and table
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
from mfc.algorithms.validation import monte_carlo_objective
from mfc.environments import (
    LQ,
    LQConfig,
    Advertising,
    AdvertisingConfig,
    AdvertisingPolicy,
    Bimodal,
    BimodalConfig,
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
from mfc.visualization.constants import ENVIRONMENTS as ENVIRONMENT_CLASSES
from mfc.visualization.io import dataclass_from_dict


BENCHMARKS = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "bimodal"]
# Continuous-state benchmarks carry the mixture chart and the Gaussian-manifold
# arm; the other groups name the figure each set of benchmarks appears on.
CONTINUOUS_ENVS = ["lq", "portfolio"]
MAIN_ENVS = ["twostate", "distribution", "lq", "portfolio", "bimodal"]
APPENDIX_ENVS = ["cybersecurity", "advertising"]
DISPLAY = {
    "twostate": "Two-state",
    "cybersecurity": "Cybersecurity",
    "distribution": "Distribution",
    "advertising": "Advertising",
    "lq": "Linear--quadratic",
    "portfolio": "Portfolio",
    "bimodal": "Bimodal allocation",
}
# Mixture size of the transport run reported for each continuous benchmark. The
# bimodal benchmark is reported with two components, the size it is built to need;
# its single-Gaussian arm is drawn beside it as "Transport, K=1".
TRANSPORT_COMPONENTS = {"lq": 1, "portfolio": 1, "bimodal": 2}
MAIN_FLOW = {
    "twostate": "particle",
    "cybersecurity": "particle",
    "distribution": "particle",
    "advertising": "particle",
    "lq": "particle",
    "portfolio": "particle",
    "bimodal": "particle",
}
REFERENCE_OPTIMUM = {
    ("twostate", 5): -2.640,
    ("distribution", 5): -0.056991,
    ("advertising", 5): 1.018750,
}
LEARNING_PANELS = {
    "Linear--quadratic": ("lq", -7.223777),
    "Portfolio": ("portfolio", -13.153766),
    "Two-state": ("twostate", -2.640),
    "Distribution": ("distribution", -0.056991),
    "Bimodal allocation": ("bimodal", 0.0),
    "Advertising": ("advertising", 1.018750),
}
METHOD_COLOR = {"REINFORCE": "#eb6834", "MF-REINFORCE": "#eda100", "Transport": "#2a78d6",
                "Transport-Proba": "#7b3fbf", "Transport, K=1": "#8fb8e8", "Finite differences": "#b5446e"}
METHOD_MARKER = {"REINFORCE": "s", "MF-REINFORCE": "D", "Transport": "o", "Transport-Proba": "^",
                 "Transport, K=1": "v", "Finite differences": "P"}
OPTIMAL = "#52514e"
MFQ_COLOR = "#1baf7a"
# One colour and marker per benchmark, shared by every diagnostic figure.
ENV_COLOR = {"lq": "#2a78d6", "portfolio": "#eb6834", "bimodal": "#7b3fbf", "twostate": "#1baf7a",
             "distribution": "#eda100", "cybersecurity": "#e87ba4", "advertising": "#008300"}
ENV_MARKER = {"lq": "o", "portfolio": "s", "bimodal": "X", "twostate": "^", "distribution": "D",
              "cybersecurity": "v", "advertising": "P"}
ENV_SHORT = {"lq": "Linear-quadratic", "portfolio": "Portfolio", "bimodal": "Bimodal", "twostate": "Two-state",
             "distribution": "Distribution", "cybersecurity": "Cybersecurity", "advertising": "Advertising"}
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
        # The radius is selected before training, so only the lambda grid is read off the runs.
        if f"_eta_{run_plan.auxiliary_eta(env):g}_" not in stem:
            continue
        value = json.loads(summary.read_text()).get("last_validation_objective")
        if value is not None:
            scores.setdefault(stem, []).append(value)
    if not scores:
        return None
    return max(scores, key=lambda stem: sum(scores[stem]) / len(scores[stem]))


def finite_difference_stem(results_root, env):
    """The finite-difference comparator of a continuous benchmark, at its step h = eta."""
    if env not in run_plan.CONTINUOUS_ENVS:
        return None
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    stem = f"finitediff_h_{run_plan.auxiliary_eta(env):g}_T_{horizon}_exact"
    return stem if any(env_dir(results_root, env).glob(f"{stem}_seed_*/summary.json")) else None


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
        stems["MF-REINFORCE"] = f"mfreinforce_eps_0.2_T_{horizon}_particle"
    elif env == "cybersecurity":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_1_T_{horizon}_particle"
    elif env == "distribution":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_2_T_{horizon}_particle"
    elif env == "advertising":
        stems["MF-REINFORCE"] = f"mfreinforce_eps_1_T_{horizon}_particle"
    stem = transport_stem(results_root, env, components=TRANSPORT_COMPONENTS.get(env))
    if stem is not None:
        stems["Transport"] = stem
    if env == "bimodal":
        stem = transport_stem(results_root, env, components=1)
        if stem is not None:
            stems["Transport, K=1"] = stem
    stem = finite_difference_stem(results_root, env)
    if stem is not None:
        stems["Finite differences"] = stem
    if include_gaussian:
        stem = gaussian_stem(results_root, env)
        if stem is not None:
            stems["Transport-Proba"] = stem
    return stems


_OPTIMAL_POLICIES = {}
_RUN_OPTIMA = {}


def optimal_policy_of(name, env):
    """The benchmark's optimal policy, or None where it has no reference (cybersecurity)."""
    key = (name, env.config.T, getattr(env.config, "T_val", None))
    if key not in _OPTIMAL_POLICIES:
        if name == "cybersecurity":
            policy = None
        elif name in {"distribution", "advertising"}:
            policy = env.optimal_policy()
        else:
            policy = env.optimal_theta()
        _OPTIMAL_POLICIES[key] = policy
    return _OPTIMAL_POLICIES[key]


def run_optimum(path):
    """J(theta*) estimated as this run's validation estimates J(theta).

    The optimal policy is evaluated with the run's own validation particles and
    seed, so the finite-particle offset of the estimator, which does not vanish at
    the optimum for a reward with a kink, cancels in the gap. None for runs
    validated before Monte Carlo validation, or without a known optimum.
    """
    path = Path(path)
    if path not in _RUN_OPTIMA:
        metadata = json.loads((path / "metadata.json").read_text())
        value = None
        if "validation_particles" in metadata["env_config"]:
            env_class, config_class = ENVIRONMENT_CLASSES[metadata["env"]]
            env = env_class(dataclass_from_dict(config_class, metadata["env_config"], device="cpu"))
            policy = optimal_policy_of(metadata["env"], env)
            if policy is not None:
                n_train = metadata["algorithm_config"].get("n_train") or metadata["env_config"]["n_train"]
                value = float(monte_carlo_objective(
                    env, policy, lambda t: t, metadata["resolved_discount"], env.config.validation_particles,
                    getattr(env.config, "T_val", env.config.T), metadata["seed"] + n_train,
                ))
        _RUN_OPTIMA[path] = value
    return _RUN_OPTIMA[path]


def seed_optima(directory, stem, fallback):
    """Per-seed reference optima, in the order curves() stacks the seeds."""
    optima = [run_optimum(path) for path in sorted(Path(directory).glob(f"{stem}_seed_*"))]
    return np.array([fallback if value is None else value for value in optima])


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
    lambda_, eta = transport_scales(transport_stem(results_root, env, components=TRANSPORT_COMPONENTS.get(env)))
    parts = [] if lambda_ is None or eta is None else [rf"$\lambda={lambda_:.3g},\ \eta={eta:g}$"]
    if include_gaussian:
        proba, _ = transport_scales(gaussian_stem(results_root, env))
        if proba is not None:
            parts.append(rf"$\lambda_{{\mathrm{{proba}}}}={proba:.3g}$")
    return "   ".join(parts)


TITLE_OF = {env: title for title, (env, _) in LEARNING_PANELS.items()}
OPTIMUM_OF = {env: optimum for _, (env, optimum) in LEARNING_PANELS.items()}


def update_budget(env):
    """Simulated transitions of one update, T (M + n + B), shared by every method of a benchmark."""
    allocation = run_plan.TRANSPORT_ALLOCATIONS[env]
    return allocation["horizon"] * run_plan.transport_per_step_budget(env)


def draw_learning_panel(ax, results_root, env, legend_only=False, include_gaussian=False, against="updates",
                        subtitle=True):
    """Optimality gap against policy updates, or against simulated transitions, for one benchmark."""
    optimum = OPTIMUM_OF[env]
    scale = update_budget(env) / 1e6 if against == "calls" else 1.0
    panel = []
    for method, stem in run_stems(results_root, env, include_gaussian).items():
        seeds = curves(env_dir(results_root, env), stem)
        if seeds is None:
            continue
        steps = np.arange(1, seeds.shape[1] + 1) * 10 * scale
        gaps = np.abs(seeds - seed_optima(env_dir(results_root, env), stem, optimum)[:, None])
        gap, deviation = gaps.mean(axis=0), gaps.std(axis=0)
        panel.extend(gap.tolist())
        ax.plot(steps, gap, color=METHOD_COLOR[method], linewidth=1.1, label=method)
        floor = max(gap.min() * 0.25, 1e-12)
        ax.fill_between(steps, np.maximum(gap - deviation, floor), gap + deviation,
                        color=METHOD_COLOR[method], alpha=0.20, linewidth=0)
    ax.set_title(TITLE_OF[env].replace("--", "-"), fontsize=8, color=INK, pad=12)
    if subtitle:
        ax.text(0.5, 1.015, scale_label(results_root, env, include_gaussian), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.4, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("simulated transitions (millions)" if against == "calls" else "policy updates", fontsize=7.5)
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


FLOW_ORDER = (r"$\theta^\star$", "Transport", "Transport, K=1", "Transport-Proba", "Finite differences",
              "MF-REINFORCE", "REINFORCE", r"$\mathcal{N}(0,1)$")


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
             "Finite differences": finite_difference_stem(results_root, "portfolio"),
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
             "Finite differences": finite_difference_stem(results_root, "lq"),
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
        "MF-REINFORCE": run_dir / "mfreinforce_eps_0.2_T_5_particle_seed_0",
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
        "MF-REINFORCE": "mfreinforce_eps_2_T_5_particle",
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


def draw_bimodal_flow(ax, results_root, include_gaussian=False):
    """Terminal law mu_1^theta of the final policies, against the optimum and its K=1 fit."""
    env = Bimodal(BimodalConfig(device="cpu"))
    points = torch.linspace(-3.0, 3.0, 301, dtype=env.dtype)
    density = lambda theta: env.terminal_density(theta.reshape(1).to(env.dtype), points).numpy()
    ax.plot(points.numpy(), density(env.optimal_theta()), color=OPTIMAL, linewidth=1.0,
            dashes=(3, 2), label=r"$\theta^\star$")
    # Every terminal law has mean zero and variance one, so this is the K=1 fit of all of them.
    ax.plot(points.numpy(), np.exp(-0.5 * points.numpy() ** 2) / np.sqrt(2.0 * np.pi), color=MUTED,
            linewidth=0.8, dashes=(1, 1.6), label=r"$\mathcal{N}(0,1)$")
    directory = env_dir(results_root, "bimodal")
    for method, stem in run_stems(results_root, "bimodal").items():
        folder = directory / f"{stem}_seed_0"
        if not (folder / "policy.pt").exists():
            continue
        ax.plot(points.numpy(), density(tensor_policy(folder)), color=METHOD_COLOR[method],
                linewidth=1.1, label=method)
    ax.set_title("Bimodal allocation: terminal law", fontsize=8, color=INK, pad=3)
    ax.set_xlabel("$x$", fontsize=7.5)
    ax.set_ylabel(r"$\mu_1^\theta(x)$", fontsize=7.5)
    style(ax)


FLOW_PANEL = {
    "lq": draw_lq_flow,
    "portfolio": draw_portfolio_flow,
    "twostate": draw_twostate_flow,
    "distribution": draw_distribution_flow,
    "bimodal": draw_bimodal_flow,
}


def main_benchmarks(results_root, output):
    """Main-text figure: optimality gaps on the top row, induced populations below.

    Every panel is read against simulated transitions, the budget the methods of a
    benchmark share, and the continuous benchmarks carry the Transport-Proba arm.
    The scales of each run are reported in the summary table, not on the panels.
    """
    envs = [env for env in MAIN_ENVS if has_run(results_root, env)]
    if not envs:
        print("skipping main_benchmarks: no matching runs found")
        return
    figure, axes = plt.subplots(2, len(envs), figsize=(6.9, 3.15), constrained_layout=True, squeeze=False)
    for index, env in enumerate(envs):
        draw_learning_panel(axes[0, index], results_root, env, include_gaussian=True, against="calls",
                            subtitle=False)
        FLOW_PANEL[env](axes[1, index], results_root, include_gaussian=True)
        axes[0, index].set_title(ENV_SHORT[env], fontsize=7.4, color=INK, pad=4)
        # The flow panels are identified by their column; keep only what they show.
        axes[1, index].set_title(axes[1, index].get_title().split(": ", 1)[-1], fontsize=6.8, color=MUTED, pad=3)
    for env in ("lq", "twostate"):
        if env in envs:
            # Gaps spanning three decades collide under the shared (1, 2, 5) locator.
            axes[0, envs.index(env)].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=5))
    if "bimodal" in envs:
        # The K=2 gap reaches 1e-4, so the default tail-based limits cut it off.
        ax = axes[0, envs.index("bimodal")]
        ax.set_ylim(3e-5, 0.4)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=6))
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.tick_params(labelsize=5.8)
        ax.xaxis.label.set_size(6.2)
        ax.yaxis.label.set_size(6.4)
    for ax in axes[0, 1:]:
        ax.set_ylabel("")
    axes[0, len(envs) // 2].set_xlabel("simulated transitions (millions)", fontsize=6.2)
    for ax in np.delete(axes[0], len(envs) // 2):
        ax.set_xlabel("")
    figure_legend(figure, axes.ravel(), FLOW_ORDER, 8)
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
            steps = np.arange(1, seeds.shape[1] + 1) * 10 * update_budget(env) / 1e6
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
        ax.set_xlabel("simulated transitions (millions)", fontsize=7.5)
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


def tradeoff_filter(env):
    """The lambda sweep is read at the benchmark's selected auxiliary radius."""
    return f"_eta_{run_plan.auxiliary_eta(env):g}_"


def lambda_sweep(results_root, env, filter_text, flow, horizon, optimum):
    """Optimality gap of every transport lambda available for this benchmark."""
    directory = env_dir(results_root, env)
    points = {}
    for path in directory.glob(f"transport_*_T_{horizon}_{flow}_seed_*"):
        stem = re.sub(r"_seed_\d+$", "", path.name)
        if filter_text is not None and filter_text not in stem:
            continue
        components = TRANSPORT_COMPONENTS.get(env)
        if components is not None and f"_K_{components}_" not in stem:
            continue
        summary = path / "summary.json"
        if not summary.exists():
            continue
        lambda_ = float(re.search(r"lambda_([0-9.]+)", stem).group(1))
        reference = run_optimum(path)
        value = json.loads(summary.read_text())["last_validation_objective"]
        points.setdefault(lambda_, []).append(abs(value - (optimum if reference is None else reference)))
    return points


# The bimodal sweep spans two decades of gap, which flattens every other curve; its
# scales are reported in their own table.
TRADEOFF_ENVS = ("twostate", "distribution", "lq", "portfolio")


def draw_lambda_tradeoff(ax, results_root, envs=TRADEOFF_ENVS):
    """Final optimality gap of every transport lambda, relative to the best lambda."""
    drawn = 0
    for env in envs:
        horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
        points = lambda_sweep(results_root, env, tradeoff_filter(env), MAIN_FLOW[env], horizon, OPTIMUM_OF[env])
        if len(points) < 2:
            print(f"skipping {env} in lambda_tradeoff: fewer than two scales available")
            continue
        scales = sorted(points)
        gaps = [sum(points[s]) / len(points[s]) for s in scales]
        best = min(gaps)
        # On the B^{-1/4} multiplier grid the anchor scale is the fourth point.
        ax.plot([s / run_plan.asymptotic_main_lambda(env) for s in scales], [g / best for g in gaps],
                color=ENV_COLOR[env], marker=ENV_MARKER[env], markersize=3.2, linewidth=1.1, label=ENV_SHORT[env])
        drawn += 1
    ax.axhline(1.0, color=MUTED, linewidth=0.7, dashes=(1.2, 1.6), zorder=1)
    ax.axvline(1.0, color=MUTED, linewidth=0.7, dashes=(1.2, 1.6), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda/\lambda_\star$")
    ax.set_ylabel("optimality gap / best gap")
    ticks = [0.125, 0.25, 0.5, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2"])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(compact_tick))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)
    return drawn


THEORY_TICKS = [0.0125, 0.05, 0.2]


def _theory_axis(ax, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8, color=INK, pad=4)
    ax.set_xticks(THEORY_TICKS)
    ax.set_xticklabels([f"{tick:g}" for tick in THEORY_TICKS])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)


def _slope_guides(ax, lambdas, exponents):
    """Grey reference lines of the given slopes, through the lower left of the panel."""
    lambdas = np.array(sorted(lambdas), float)
    low, _ = ax.get_ylim()
    for exponent, dashes in zip(exponents, ((3, 2), (1, 1.6))):
        ax.plot(lambdas, low * 2.0 * (lambdas / lambdas[0]) ** exponent, color=MUTED, linewidth=0.8,
                dashes=dashes, zorder=1, label={1.0: r"$\propto\lambda$", 0.5: r"$\propto\sqrt{\lambda}$"}[exponent])


def draw_perturbation_estimate(ax, estimate_csv, envs):
    """Distance from the perturbed to the unperturbed law, against lambda.

    Finite state space: the largest realized d_TV, whose pathwise bound is lambda.
    Continuous state space: the root-mean-square W2 distance of the decoded mixtures,
    whose bound is sqrt(lambda) up to the projection error.
    """
    estimate = pd.read_csv(estimate_csv)
    continuous = estimate["w2_rms"] if "w2_rms" in estimate else estimate["w1_rms"]
    estimate = estimate.assign(deviation=estimate["max_tv"].where(estimate["space"] == "finite", continuous))
    for name in envs:
        rows = estimate[estimate["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        ax.plot(rows["lambda"], rows["deviation"], color=ENV_COLOR[name], marker=ENV_MARKER[name],
                markersize=3.0, linewidth=1.1, dashes=dashes, label=ENV_SHORT[name])
    _theory_axis(ax, r"$d_{\mathrm{TV}}$  or  $\mathcal{W}_2$", "perturbation estimate")
    _slope_guides(ax, estimate["lambda"].unique(), (1.0, 0.5))


def draw_perturbation_consistency(ax, consistency_csv, envs):
    consistency = pd.read_csv(consistency_csv)
    for name in envs:
        rows = consistency[consistency["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        ax.plot(rows["lambda"], rows["gradient_gap"], color=ENV_COLOR[name], marker=ENV_MARKER[name],
                markersize=3.0, linewidth=1.1, dashes=dashes, label=ENV_SHORT[name])
    _theory_axis(ax, r"$\|\nabla_\theta J^\lambda-\nabla_\theta J\|$", "perturbation consistency")
    _slope_guides(ax, consistency["lambda"].unique(), (1.0,))


def _benchmark_legend(figure, axes, envs, ncol):
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    names = [ENV_SHORT[env] for env in envs if ENV_SHORT[env] in found]
    names += [name for name in found if name.startswith("$")]
    figure.legend([found[name] for name in names], names, loc="outside lower center", ncol=ncol,
                  frameon=False, handlelength=1.8, columnspacing=1.0, borderpad=0.2)


def main_diagnostics(results_root, estimate_csv, consistency_csv, output):
    """Main-text figure: lambda trade-off in training, then the two perturbation results."""
    if not Path(estimate_csv).exists() or not Path(consistency_csv).exists():
        print("skipping main_diagnostics: diagnostic CSVs not found")
        return
    figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), constrained_layout=True)
    draw_lambda_tradeoff(axes[0], results_root)
    axes[0].set_title("(a) scale in training", fontsize=8, color=INK, pad=4)
    draw_perturbation_estimate(axes[1], estimate_csv, MAIN_ENVS)
    axes[1].set_title("(b) " + axes[1].get_title(), fontsize=8, color=INK, pad=4)
    draw_perturbation_consistency(axes[2], consistency_csv, MAIN_ENVS)
    axes[2].set_title("(c) " + axes[2].get_title(), fontsize=8, color=INK, pad=4)
    for ax in axes:
        ax.tick_params(labelsize=6.2)
        ax.xaxis.label.set_size(7.0)
        ax.yaxis.label.set_size(7.0)
    _benchmark_legend(figure, axes, MAIN_ENVS, 7)
    save_figure(figure, output)


def theory_verification(estimate_csv, consistency_csv, output):
    """Appendix figure: the two perturbation results on every benchmark."""
    if not Path(estimate_csv).exists() or not Path(consistency_csv).exists():
        print("skipping theory_verification: diagnostic CSVs not found")
        return
    order = ["lq", "portfolio", "bimodal", "twostate", "distribution", "cybersecurity", "advertising"]
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)
    draw_perturbation_estimate(axes[0], estimate_csv, order)
    axes[0].set_title("(a) perturbation estimate", fontsize=8, color=INK, pad=4)
    draw_perturbation_consistency(axes[1], consistency_csv, order)
    axes[1].set_title("(b) perturbation consistency", fontsize=8, color=INK, pad=4)
    _benchmark_legend(figure, axes, order, 5)
    save_figure(figure, output)


ETA_ENVS = ("twostate", "cybersecurity", "distribution", "advertising")
ETA_COLOR = dict(zip(ETA_ENVS, ["#1baf7a", "#e87ba4", "#eda100", "#008300"]))
ETA_MARKER = dict(zip(ETA_ENVS, ["^", "v", "D", "s"]))


def discrete_eta(eta_csv, output):
    """Auxiliary-radius diagnostic: estimator error against the exact oracles, per benchmark."""
    if not Path(eta_csv).exists():
        print("skipping discrete_eta: diagnostic CSV not found")
        return
    table = pd.read_csv(eta_csv)
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), constrained_layout=True)
    panels = [("d_mse", r"$\sum_t\mathbb{E}\|\widehat D_t-D_t\|_F^2/\sum_t\|D_t\|_F^2$", "sensitivity"),
              ("g_mse", r"$\mathbb{E}\|\widehat G-\nabla J\|^2/\|\nabla J\|^2$", "gradient")]
    for ax, (column, ylabel, title) in zip(axes, panels):
        for env in [env for env in ETA_ENVS if (table["env"] == env).any()]:
            for batches, dashes in (("reused", None), ("fresh", (3, 2))):
                rows = table[(table["env"] == env) & (table["batches"] == batches)].dropna(subset=[column])
                if rows.empty:
                    continue
                rows = rows.sort_values("eta")
                style_kwargs = {} if dashes is None else {"dashes": dashes, "alpha": 0.7}
                ax.plot(rows["eta"], rows[column], color=ETA_COLOR[env], marker=ETA_MARKER[env], markersize=3.0,
                        linewidth=1.0, label=DISPLAY[env] if batches == "reused" else None, **style_kwargs)
        ax.axvspan(0.85, 0.98, color=GRID, alpha=0.5, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(ylabel, fontsize=6.8)
        ax.set_title(title, fontsize=8, color=INK)
        ticks = [0.15, 0.3, 0.5, 0.95]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax)
    figure_legend(figure, axes, [DISPLAY[env] for env in ETA_ENVS], 4)
    save_figure(figure, output)


def scientific(value):
    if value >= 1e3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${float(mantissa):.2f}\\cdot10^{{{int(exponent)}}}$"
    return f"${value:.3g}$"


def discrete_eta_table(eta_csv):
    """Auxiliary-radius diagnostic as a table: shared against fresh batches, and the gradient error."""
    if not Path(eta_csv).exists():
        return None
    table = pd.read_csv(eta_csv)
    lines = [
        "Benchmark & $n$ & $\\eta$ & \\multicolumn{2}{c}{Sensitivity error} & Gradient error \\\\",
        " & & & shared batch & fresh batches & shared batch \\\\",
        "\\midrule",
    ]
    envs = [env for env in ETA_ENVS if (table["env"] == env).any()]
    for env in envs:
        rows = table[table["env"] == env]
        for index, eta in enumerate(sorted(rows["eta"].unique())):
            reused = rows[(rows["eta"] == eta) & (rows["batches"] == "reused")].iloc[0]
            fresh = rows[(rows["eta"] == eta) & (rows["batches"] == "fresh")]
            lines.append(" & ".join([
                DISPLAY[env] if index == 0 else "",
                str(int(reused["n"])) if index == 0 else "",
                f"${eta:.3g}$",
                scientific(reused["d_mse"]),
                "---" if fresh.empty else scientific(fresh.iloc[0]["d_mse"]),
                scientific(reused["g_mse"]),
            ]) + " \\\\")
        if env != envs[-1]:
            lines.append("\\midrule")
    caption = (
        "Auxiliary radius on the finite benchmarks, at the initial policy and $\\lambda=B^{-1/4}$: relative "
        "mean-square errors $\\sum_t\\E\\|\\widehat D_t-D_t\\|_F^2/\\sum_t\\|D_t\\|_F^2$ and "
        "$\\E\\|\\widehat G-\\nabla_\\theta J\\|^2/\\|\\nabla_\\theta J\\|^2$ against the exact sensitivities "
        "and gradient, over 100 and 50 replications. The first rows of each benchmark are "
        "$\\eta_\\star/2$, $\\eta_\\star$ and $2\\eta_\\star$ when below one, with $\\eta_\\star=n^{-1/4}$."
    )
    return table_environment("\n".join(lines), caption, "tab:discrete-eta", "lrrrrr", size="\\small")


def twostate_eta_sweep(results_root, output):
    """Two-state optimality gap at lambda* across the bound radii and the large grid."""
    env = "twostate"
    if not has_run(results_root, env):
        print("skipping twostate_eta_sweep: no matching runs found")
        return
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    lambda_ = run_plan.asymptotic_main_lambda(env)
    figure, ax = plt.subplots(figsize=(3.2, 2.2), constrained_layout=True)
    drawn = False
    for flow, marker in (("exact", "o"), ("particle", "s")):
        points = []
        for eta in run_plan.bound_eta_grid(env) + run_plan.LARGE_AUXILIARY_ETAS:
            gaps = seed_gaps(results_root, env, f"transport_lambda_{lambda_:g}_eta_{eta:g}_T_{horizon}_{flow}")
            if gaps.size:
                points.append((eta, gaps.mean(), gaps.std(ddof=1) if gaps.size > 1 else 0.0))
        if len(points) < 2:
            continue
        etas, means, deviations = map(np.array, zip(*points))
        ax.errorbar(etas, means, yerr=deviations, color=METHOD_COLOR["Transport"], marker=marker, markersize=3.4,
                    linewidth=1.0, capsize=2, dashes=(3, 2) if flow == "particle" else (None, None),
                    label=f"{flow} flow")
        drawn = True
    if not drawn:
        print("skipping twostate_eta_sweep: fewer than two radii available")
        plt.close(figure)
        return
    ax.axvspan(0.85, 0.98, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$|J(\widehat\theta)-J(\theta^\star)|$")
    ax.set_title(rf"Two-state: auxiliary radius at $\lambda={lambda_:.3g}$", fontsize=8, color=INK)
    ticks = [0.25, 0.5, 0.95]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax)
    ax.legend(frameon=False, fontsize=6.6)
    save_figure(figure, output)


BOUND_COLOR, BOUND_MARKER = ENV_COLOR, ENV_MARKER
BOUND_ENVS = ("lq", "portfolio", "twostate", "cybersecurity", "distribution", "advertising")
# The bimodal benchmark has a scalar parameter and T=1, and is not swept.
MAIN_BOUND_ENVS = ("twostate", "distribution", "lq", "portfolio")


def lambda_fit_mask(block):
    """Points of a lambda sweep on which its exponent is fitted.

    A bias is the norm of a sample mean, so it is not read below about twice its
    Monte Carlo resolution; the finite sweeps are exact and keep every point. Of the
    points left, the lower half is kept, since the bias saturates once it is of the
    order of the gradient itself.
    """
    values = block["value"].to_numpy(float)
    keep = np.ones_like(values, dtype=bool)
    if "bias_resolution" in block and block["bias_resolution"].notna().any():
        keep = block["bias"].to_numpy(float) >= 2.0 * block["bias_resolution"].to_numpy(float)
    if keep.sum() > 2:
        keep &= values <= np.median(values[keep])
    return keep


def load_bounds(bounds_csv):
    """The continuous sweeps and, beside them, every finite-state sweep file present."""
    paths = [Path(bounds_csv)] + sorted(Path(bounds_csv).parent.glob("bounds_discrete*.csv"))
    tables = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(tables, ignore_index=True) if tables else None


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


def _bounds_table(bounds_csv, envs=BOUND_ENVS):
    table = load_bounds(bounds_csv)
    if table is None:
        return None, []
    return table, [env for env in envs if (table["env"] == env).any()]


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
    table, envs = _bounds_table(bounds_csv, ("lq", "portfolio"))
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


def bounds_rates(bounds_csv, output, envs=("lq", "portfolio")):
    """One sweep per panel: each block size, the scale, and the main block."""
    table, envs = _bounds_table(bounds_csv, envs)
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
            keep = lambda_fit_mask(block) if sweep == "lambda" else np.ones_like(x, dtype=bool)
            if sweep == "lambda" and "bias_resolution" in block and block["bias_resolution"].notna().any():
                # Monte Carlo resolution of the bias; hollow markers are not fitted.
                ax.errorbar(x, y, yerr=block["bias_resolution"].to_numpy(), fmt="none",
                            ecolor=BOUND_COLOR[env], elinewidth=0.6, capsize=1.5)
                ax.plot(x[~keep], y[~keep], linestyle="none", marker=BOUND_MARKER[env], markersize=3.0,
                        markerfacecolor="white", markeredgecolor=BOUND_COLOR[env])
            anchor = int(np.flatnonzero(keep)[0]) if keep.any() else 0
            guide(ax, x, y, exponent, anchor=anchor)
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


def bounds_exponents(bounds_csv, envs=BOUND_ENVS, label="tab:bounds-exponents", wide=False):
    """Fitted log-log exponents of every sweep against the rate the bound predicts."""
    table, envs = _bounds_table(bounds_csv, envs)
    if table is None:
        return None
    header = " & ".join(["Quantity", "Swept"] + [DISPLAY[env] for env in envs] + ["$p_0$"])
    lines = [header + " \\\\", "\\midrule"]
    for sweep, knob, quantity, column, exponent in BOUND_SWEEPS:
        cells = []
        for env in envs:
            rows = table[(table["env"] == env) & (table["sweep"] == sweep)]
            if sweep == "B" and not rows.empty:
                rows = rows[rows["lambda_"] == rows["lambda_"].min()]
            if sweep == "lambda" and not rows.empty:
                rows = rows[lambda_fit_mask(rows)]
            cells.append("---" if rows.empty else f"${bound_slope(rows['value'], rows[column]):+.2f}$")
        lines.append(" & ".join([quantity, knob] + cells + [f"${exponent:+.0f}$"]) + " \\\\")
    caption = (
        "Slope $p$ of each error against the quantity swept, fitted by least squares on a log-log "
        "scale, at $\\theta=\\tfrac12\\theta^\\star$ on the continuous benchmarks and at the initial "
        "policy on the finite ones, with $p_0$ the slope of the mechanism producing it. The $\\lambda$ "
        "row is fitted on the lower half of the scales whose bias exceeds twice its Monte Carlo resolution."
    )
    return table_environment("\n".join(lines), caption, label,
                             "ll" + "r" * len(envs) + "r", size="\\footnotesize", wide=wide, tabcolsep="4pt")


def method_of(label):
    if label.startswith("REINFORCE"):
        return "reinforce"
    if label.startswith("Transport-Proba"):
        return "gaussian"
    if label.startswith("MFQ-learning"):
        return "mfqlearning"
    if label.startswith("MF-REINFORCE"):
        return "mfreinforce"
    if label.startswith("Finite differences"):
        return "finitediff"
    return "transport"


def number(value, digits=4):
    return "---" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def with_error(row, digits=4):
    if row is None:
        return "---"
    return f"${number(row['mean'], digits)}\\pm{number(row['std'], digits)}$"


def table_environment(body, caption, label, alignment, size=None, wide=False, tabcolsep=None):
    """A booktabs table; wide tables span both columns of the main text."""
    begin, end = ("\\begin{table*}[t]", "\\end{table*}") if wide else ("\\begin{table}[H]", "\\end{table}")
    return "\n".join(
        [
            begin,
            "\\centering",
            *([size] if size else []),
            *([f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []),
            # AISTATS sets table captions above the table.
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            body,
            "\\bottomrule",
            "\\end{tabular}",
            end,
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
    # REINFORCE and finite differences have no population flow of their own to choose.
    subset = subset[(subset["flow"] == flow) | subset["method"].isin(["reinforce", "finitediff"])]
    best = {}
    for method, rows in subset.groupby("method"):
        rows = rows.dropna(subset=["mean"])
        if not rows.empty:
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
    envs = [name for name in BENCHMARKS if has_run(results_root, name)]
    finite_differences = any(finite_difference_stem(results_root, env) for env in envs)
    lines = [
        "Benchmark & $T$ & Optimum & REINFORCE & MF-REINFORCE & Transport & $(\\lambda,\\eta)$"
        + (" & Finite diff." if finite_differences else "") + " \\\\",
        "\\midrule",
    ]
    for env in envs:
        grouped, _ = grouped_objectives(results_root, env)
        best = select_headline(grouped, env)
        optimum = reference_optimum(results_root, env)
        digits = 3 if env == "portfolio" else 4
        transport = best.get("transport")
        # Read off the run being reported rather than recomputing the grid rule, which
        # need not name a configuration this results tree actually contains.
        components = TRANSPORT_COMPONENTS.get(env)
        lambda_, eta = transport_scales(transport_stem(results_root, env, components=components))
        scales = (
            "---"
            if transport is None or lambda_ is None or eta is None
            else f"$({lambda_:g},{eta:g})$"
        )
        lines.append(
            " & ".join(
                [
                    DISPLAY[env],
                    str(run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]),
                    "---" if optimum is None else f"${number(optimum, digits)}$",
                    with_error(best.get("reinforce"), digits),
                    with_error(best.get("mfreinforce"), digits),
                    with_error(transport, digits),
                    scales,
                ] + ([with_error(best.get("finitediff"), digits)] if finite_differences else [])
            )
            + " \\\\"
        )
    caption = (
        "Final validation objective on every benchmark, as mean and standard deviation over seeds. "
        "Higher is better. The transport column shows the best run of the scale grid; every scale of "
        "Transport and Transport-Proba on the continuous benchmarks is in Table~\\ref{tab:continuous-comparison}."
    )
    alignment = "lrrrrrl" + ("r" if finite_differences else "")
    return table_environment("\n".join(lines), caption, "tab:objective-summary", alignment, size="\\footnotesize",
                             tabcolsep="4pt")


ESTIMATORS = [
    ("reinforce", "REINFORCE"),
    ("mfreinforce", "MF-REINFORCE"),
    ("transport", "Transport"),
    ("gaussian", "Transport-Proba"),
    ("finitediff", "Finite differences"),
    ("mfqlearning", "MFQ-learning"),
]
SCALE_COUNT = {"transport": "$\\lambda,\\eta$", "gaussian": "$\\lambda$"}


def runtime_rows(results_root, env, scales_column):
    """One LaTeX row per estimator present on this benchmark, at its headline scale."""
    runtime = runtime_table(load_env_runs(results_root, env))
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    subset = runtime[runtime["horizon"] == horizon]
    subset = subset[(subset["flow"] == MAIN_FLOW[env]) | subset["algorithm"].isin(["reinforce", "finitediff"])]
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


CONTINUOUS_STEMS = [
    ("reinforce", "REINFORCE", "reinforce_none_T_{horizon}_exact"),
    ("transport", "Transport", "transport_lambda_{lambda_}_eta_{eta}_K_1_T_{horizon}_particle"),
    ("gaussian", "Transport-Proba", "gaussian_lambda_{lambda_}_T_{horizon}_particle"),
    ("finitediff", "Finite differences", "finitediff_h_{lambda_}_T_{horizon}_exact"),
]


def scale_grid(results_root, env, algorithm):
    """Every perturbation scale this results tree holds for one continuous arm."""
    horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
    found = set()
    if algorithm == "finitediff":
        pattern, scale = f"finitediff_h_*_T_{horizon}_exact_seed_*", r"_h_([0-9.]+)_"
    else:
        pattern, scale = f"{algorithm}_lambda_*_T_{horizon}_particle_seed_*", r"lambda_([0-9.]+)"
    for path in env_dir(results_root, env).glob(pattern):
        if algorithm == "transport" and "_K_1_" not in path.name:
            continue
        found.add(float(re.search(scale, path.name).group(1)))
    return sorted(found)


def seed_values(results_root, env, stem):
    values = [
        json.loads(path.read_text()).get("last_validation_objective")
        for path in sorted(env_dir(results_root, env).glob(f"{stem}_seed_*/summary.json"))
    ]
    return np.array([value for value in values if value is not None])


def seed_gaps(results_root, env, stem):
    """Final optimality gap of every seed, each against its own reference optimum."""
    gaps = []
    for path in sorted(env_dir(results_root, env).glob(f"{stem}_seed_*")):
        summary = path / "summary.json"
        value = json.loads(summary.read_text()).get("last_validation_objective") if summary.exists() else None
        if value is None:
            continue
        reference = run_optimum(path)
        gaps.append(abs(value - (OPTIMUM_OF[env] if reference is None else reference)))
    return np.array(gaps)


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
                gaps = seed_gaps(results_root, env, stem)
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


def bimodal_components(results_root):
    """Bimodal allocation: every scale of the one- and two-component transport arms."""
    if not has_run(results_root, "bimodal"):
        return None
    horizon = run_plan.TRANSPORT_ALLOCATIONS["bimodal"]["horizon"]
    directory = env_dir(results_root, "bimodal")
    lines = [
        "Estimator & $K$ & $\\lambda$ & $|J(\\widehat\\theta)-J(\\theta^\\star)|$ & $\\widehat\\theta$ \\\\",
        "\\midrule",
    ]
    configurations = [("REINFORCE", None, f"reinforce_none_T_{horizon}_exact")]
    stem = finite_difference_stem(results_root, "bimodal")
    if stem is not None:
        configurations.append(("Finite differences", None, stem))
    for components in run_plan.CONTINUOUS_COMPONENTS["bimodal"]:
        stems = {
            re.sub(r"_seed_\d+$", "", path.name)
            for path in directory.glob(f"transport_*_K_{components}_T_{horizon}_particle_seed_*")
        }
        configurations.extend(
            ("Transport", components, stem)
            for stem in sorted(stems, key=lambda stem: transport_scales(stem)[0])
        )
    for name, components, stem in configurations:
        values = seed_values(results_root, "bimodal", stem)
        if values.size == 0:
            continue
        gaps = seed_gaps(results_root, "bimodal", stem)
        thetas = np.array([float(tensor_policy(path).reshape(-1)[0])
                           for path in sorted(directory.glob(f"{stem}_seed_*")) if (path / "policy.pt").exists()])
        lambda_ = transport_scales(stem)[0]
        lines.append(" & ".join([
            name,
            "---" if components is None else str(components),
            "---" if lambda_ is None else f"${lambda_:g}$",
            f"${gaps.mean():.4f}\\pm{gaps.std(ddof=1) if gaps.size > 1 else 0.0:.4f}$",
            f"${thetas.mean():.4f}\\pm{thetas.std(ddof=1) if thetas.size > 1 else 0.0:.4f}$",
        ]) + " \\\\")
    env = Bimodal(BimodalConfig(device="cpu"))
    caption = (
        "Bimodal allocation: optimality gap and final parameter over seeds, from $\\theta=0.8$ "
        f"(gap ${-float(env.objective(torch.tensor([0.8], dtype=env.dtype))):.4f}$). "
        f"The optimum is $\\theta^\\star={float(env.optimal_theta()):.6f}$. With $K=1$ the "
        f"projected objective is constant, $J_1={float(env.single_gaussian_objective()):.6f}$, "
        "so the single-Gaussian arm carries no gradient."
    )
    return table_environment("\n".join(lines), caption, "tab:bimodal-components", "lrrrr", size="\\small")


MULTIPLIER = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2", 1.0: "1", 2.0: "2"}


def significant(mean, deviation, digits=3):
    """Mean and deviation to the same decimal place, with three significant figures in the mean."""
    decimals = max(0, digits - 1 - int(np.floor(np.log10(abs(mean))))) if mean else digits
    return f"{mean:.{decimals}f}\\pm{deviation:.{decimals}f}"


def multiplier(scale, anchor):
    ratio = scale / anchor
    closest = min(MULTIPLIER, key=lambda value: abs(np.log(ratio / value)))
    return MULTIPLIER[closest] if abs(np.log(ratio / closest)) < 0.05 else f"{ratio:.2g}"


def main_summary(results_root):
    """Main-text table: final optimality gap of every method on the main benchmarks.

    Transport is reported at the anchor scale lambda_star = B^{-1/4} and at the best
    scale of its grid, Transport-Proba at the best scale of the same grid; the
    smallest gap of each row is set in bold.
    """
    lines = [
        "& & & & \\multicolumn{2}{c}{Transport} & Transport-Proba \\\\",
        "\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}",
        "Benchmark & $J(\\theta^\\star)$ & REINFORCE & MF-REINFORCE & at $\\lambda_\\star$ & best ($\\lambda$) "
        "& best ($\\lambda$) \\\\",
        "\\midrule",
    ]
    etas = []
    for env in [name for name in MAIN_ENVS if has_run(results_root, name)]:
        horizon = run_plan.TRANSPORT_ALLOCATIONS[env]["horizon"]
        anchor = run_plan.asymptotic_main_lambda(env)
        stems = run_stems(results_root, env)
        cells = {}
        for method in ("REINFORCE", "MF-REINFORCE"):
            if method in stems:
                gaps = seed_gaps(results_root, env, stems[method])
                if gaps.size:
                    cells[method] = (gaps.mean(), gaps.std(ddof=1))
        sweep = lambda_sweep(results_root, env, tradeoff_filter(env), MAIN_FLOW[env], horizon, OPTIMUM_OF[env])
        sweep = {scale: np.array(gaps) for scale, gaps in sweep.items()}
        if sweep:
            at_anchor = sweep[min(sweep, key=lambda scale: abs(np.log(scale / anchor)))]
            cells["anchor"] = (at_anchor.mean(), at_anchor.std(ddof=1))
            best = min(sweep, key=lambda scale: sweep[scale].mean())
            cells["best"] = (sweep[best].mean(), sweep[best].std(ddof=1))
        proba = {}
        if env in CONTINUOUS_ENVS:
            for scale in scale_grid(results_root, env, "gaussian"):
                gaps = seed_gaps(results_root, env, f"gaussian_lambda_{scale:g}_T_{horizon}_particle")
                if gaps.size:
                    proba[scale] = gaps
        proba_best = min(proba, key=lambda scale: proba[scale].mean()) if proba else None
        if proba_best is not None:
            cells["proba"] = (proba[proba_best].mean(), proba[proba_best].std(ddof=1))
        winner = min(cells, key=lambda key: cells[key][0])

        def cell(key, scale=None):
            if key not in cells:
                return "---"
            text = significant(*cells[key])
            text = f"$\\mathbf{{{text}}}$" if key == winner else f"${text}$"
            if scale is not None:
                ratio = multiplier(scale, anchor)
                written = {"1": "\\lambda_\\star", "2": "2\\lambda_\\star"}.get(
                    ratio, f"\\lambda_\\star/{ratio[2:]}" if ratio.startswith("1/") else f"{ratio}\\lambda_\\star")
                text += f" {{\\scriptsize$({written})$}}"
            return text

        name = "Bimodal ($K=2$)" if env == "bimodal" else DISPLAY[env]
        lines.append(" & ".join([
            name,
            f"${OPTIMUM_OF[env]:.4g}$".replace("$-0$", "$0$"),
            cell("REINFORCE"),
            cell("MF-REINFORCE"),
            cell("anchor"),
            cell("best", best if sweep else None),
            cell("proba", proba_best),
        ]) + " \\\\")
        etas.append(f"{DISPLAY[env].lower()} ${run_plan.auxiliary_eta(env):g}$")
    caption = (
        "Final optimality gap $|J(\\widehat\\theta)-J(\\theta^\\star)|$ on the main benchmarks, as mean and "
        "standard deviation over five paired seeds at matched simulator budgets; lower is better and the "
        "smallest gap of each row is in bold. Transport is trained on the grid "
        "$\\lambda\\in\\lambda_\\star\\{1/8,1/4,1/2,1,2\\}$, with $\\lambda_\\star=B^{-1/4}$, and Transport-Proba on "
        "the same grid; the scale of each best run is in parentheses. The auxiliary radius is fixed before training: $\\eta=$ " + ", ".join(etas) + "."
    )
    return table_environment("\n".join(lines), caption, "tab:main-summary", "lrrrrrr", size="\\footnotesize",
                             wide=True, tabcolsep="4pt")


def headline_row(rows, results_root, env, algorithm):
    """The runtime row of the configuration the objective tables report.

    Several scales of the same estimator share an algorithm, so the row has to
    be selected by the scale, not by whichever happened to run longest.
    """
    if algorithm == "transport":
        components = TRANSPORT_COMPONENTS.get(env)
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


def allocation_table():
    """Run-plan settings of every benchmark, read from scripts/run.py."""
    lines = [
        "Benchmark & $T$ & $M$ & $n$ & $B$ & Budget & Updates & Step size & $\\lambda_\\star$ & $\\eta$ & $\\sigma$ & $K$ \\\\",
        "\\midrule",
    ]
    for env in BENCHMARKS:
        allocation = run_plan.TRANSPORT_ALLOCATIONS[env]
        n = run_plan.effective_auxiliary_samples(env)
        budget = allocation["horizon"] * (allocation["M"] + n + allocation["B"])
        sigma = allocation.get("simplex_sigma")
        components = run_plan.CONTINUOUS_COMPONENTS.get(env)
        lines.append(" & ".join([
            DISPLAY[env],
            str(allocation["horizon"]),
            f"{allocation['M']:,}".replace(",", "\\,"),
            f"{n:,}".replace(",", "\\,"),
            f"{allocation['B']:,}".replace(",", "\\,"),
            f"{budget:,}".replace(",", "\\,"),
            f"{allocation['updates']:,}".replace(",", "\\,"),
            f"$10^{{{int(round(np.log10(allocation['lr'])))}}}$",
            f"${run_plan.asymptotic_main_lambda(env):.3g}$",
            f"${run_plan.auxiliary_eta(env):g}$",
            "---" if sigma is None else f"${sigma:g}$",
            "---" if components is None else ",".join(str(k) for k in components),
        ]) + " \\\\")
    caption = (
        "Transport allocations and optimization settings. $M$, $n$ and $B$ are the population, auxiliary "
        "and main sample sizes, and the budget is the resulting number of simulated transitions per "
        "update, $T(M+n+B)$, which every method of a benchmark receives. Transport is trained at "
        "$\\lambda\\in\\lambda_\\star\\{1/8,1/4,1/2,1,2\\}$, with $\\lambda_\\star=B^{-1/4}$; $\\sigma$ is the "
        "finite-state simplex randomizer scale and $K$ the number of mixture components."
    )
    return table_environment("\n".join(lines), caption, "tab:allocations", "lrrrrrrrrrrr", size="\\small")


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
    parser.add_argument("--discrete-eta", default="results/figures/discrete_eta/discrete_eta.csv")
    args = parser.parse_args()

    results_root = ROOT / args.results_root
    output_root = ROOT / args.output_root
    figures = output_root / "figures"
    tables = output_root / "tables"

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.labelsize": 8, "legend.fontsize": 7.2})

    main_benchmarks(results_root, figures / "main_benchmarks.pdf")
    main_diagnostics(results_root, ROOT / args.theory_estimate, ROOT / args.theory_consistency,
                     figures / "main_diagnostics.pdf")
    write_table(main_summary(results_root), tables / "main_summary.tex")
    appendix_benchmarks(results_root, figures / "appendix_benchmarks.pdf")
    discrete_eta(ROOT / args.discrete_eta, figures / "discrete_eta.pdf")
    twostate_eta_sweep(results_root, figures / "twostate_eta_sweep.pdf")
    eta_table = discrete_eta_table(ROOT / args.discrete_eta)
    if eta_table is not None:
        write_table(eta_table, tables / "discrete_eta.tex")
    theory_verification(ROOT / args.theory_estimate, ROOT / args.theory_consistency, figures / "theory_verification.pdf")
    write_table(allocation_table(), tables / "allocations.tex")
    write_table(objective_summary(results_root), tables / "objective_summary.tex")
    write_table(budget_runtime(results_root), tables / "budget_runtime.tex")
    bounds_radius(ROOT / args.bounds, figures / "bounds_radius.pdf")
    bounds_rates(ROOT / args.bounds, figures / "bounds_rates.pdf")
    bounds_rates(ROOT / args.bounds, figures / "bounds_rates_finite.pdf",
                 ("twostate", "cybersecurity", "distribution", "advertising"))
    exponents = bounds_exponents(ROOT / args.bounds)
    if exponents is not None:
        write_table(exponents, tables / "bounds_exponents.tex")
    exponents = bounds_exponents(ROOT / args.bounds, MAIN_BOUND_ENVS, "tab:bounds-exponents-main", wide=True)
    if exponents is not None:
        write_table(exponents, tables / "bounds_exponents_main.tex")
    write_table(continuous_comparison(results_root), tables / "continuous_comparison.tex")
    bimodal = bimodal_components(results_root)
    if bimodal is not None:
        write_table(bimodal, tables / "bimodal_components.tex")


if __name__ == "__main__":
    main()
