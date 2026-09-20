"""Figure for the numerical verification of the perturbation results.

Panel (a): the perturbation estimate, as the realized finite-state deviation divided
by lambda and the continuous-state projected deviation divided by sqrt(lambda). A
flat line means the measured deviation follows the corresponding reference scale.
Panel (b): finite-state perturbation consistency of the gradient, log-log against
lambda.

    uv run python scripts/theory_figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {"twostate": "Two-state", "cybersecurity": "Cybersecurity", "distribution": "Distribution",
           "advertising": "Advertising", "lq": "Linear\u2013quadratic", "portfolio": "Portfolio"}
ORDER = ["lq", "portfolio", "twostate", "distribution", "cybersecurity", "advertising"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P"]))
GRID = "#d8d7d2"
INK = "#0b0b0b"
MUTED = "#52514e"


TICKS = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]


def style(ax):
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=3, width=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimate", default="results/figures/theory/perturbation_estimate.csv")
    parser.add_argument("--consistency", default="results/figures/theory_400/perturbation_consistency.csv")
    parser.add_argument("--output", default="files/iclr2027/figures/theory_verification.pdf")
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.labelsize": 8, "legend.fontsize": 7.2})

    estimate = pd.read_csv(ROOT / args.estimate)
    consistency = pd.read_csv(ROOT / args.consistency)
    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)

    for name in ORDER:
        rows = estimate[estimate["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        axes[0].plot(rows["lambda"], rows["max_ratio"], color=COLOR[name], marker=MARKER[name],
                     markersize=3.2, linewidth=1.1, dashes=dashes, label=DISPLAY[name])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.25)
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"scaled deviation")
    axes[0].set_title("(a) perturbation estimate", fontsize=8, color=INK, pad=4)
    style(axes[0])

    for name in ORDER:
        rows = consistency[consistency["benchmark"] == name].sort_values("lambda")
        if rows.empty:
            continue
        dashes = (None, None) if rows["space"].iloc[0] == "finite" else (3.5, 1.8)
        axes[1].plot(rows["lambda"], rows["gradient_over_lambda"], color=COLOR[name], marker=MARKER[name],
                     markersize=3.2, linewidth=1.1, dashes=dashes, label=DISPLAY[name])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\|\nabla_\theta J^\lambda-\nabla_\theta J\|\,/\,\lambda$")
    axes[1].set_title("(b) perturbation consistency", fontsize=8, color=INK, pad=4)
    style(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False,
                  handlelength=1.8, columnspacing=1.0, borderpad=0.2)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
