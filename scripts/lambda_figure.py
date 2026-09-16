"""Optimality gap against the perturbation scale, normalised by the best scale of each benchmark.

    uv run python scripts/lambda_figure.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DISPLAY = {"lq": "Linear\u2013quadratic", "portfolio": "Portfolio",
           "twostate": "Two-state", "distribution": "Distribution"}
ORDER = ["lq", "portfolio", "twostate", "distribution"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D"]))
GRID, INK, MUTED = "#d8d7d2", "#0b0b0b", "#52514e"
TICKS = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", default="results/figures/theory/lambda_gap.csv")
    parser.add_argument("--output", default="files/iclr2027/figures/lambda_tradeoff.pdf")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})

    gap = pd.read_csv(ROOT / args.gap)
    figure, ax = plt.subplots(figsize=(3.6, 2.3), constrained_layout=True)
    for name in ORDER:
        rows = gap[gap["benchmark"] == name].sort_values("lambda")
        ax.plot(rows["lambda"], rows["gap"] / rows["gap"].min(), color=COLOR[name],
                marker=MARKER[name], markersize=3.4, linewidth=1.2, label=DISPLAY[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS], fontsize=7)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\lambda$", fontsize=8)
    ax.set_ylabel("optimality gap / best", fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=3, width=0.5)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center", ncol=2, borderaxespad=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)

    path = ROOT / args.output
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
