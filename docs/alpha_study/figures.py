#!/usr/bin/env python3
"""The figure of the alpha study from data/ (python3 figures.py in this directory).

fig_history_12.pdf: the GCR(2) residual histories on 12^4, alpha 0.6 and 1.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
})
NAMES = {"S": r"setup S ($M_5=1.0$, stout)", "B": r"setup B ($M_5=1.8$)"}


def history() -> None:
    with open(HERE / "data" / "iterations_12.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
    for i, s in enumerate("SB"):
        for j, m in enumerate(("0.001", "0.1")):
            ax = axes[i][j]
            for a, style in (("0.6", dict(color="C0", marker="o")), ("1.0", dict(color="C1", marker="s", ls="--"))):
                r = next(r for r in rows if (r["setup"], r["mass"], r["alpha"]) == (s, m, a))
                h = [float(x) ** 0.5 for x in r["history_r2"].split(",")]
                ax.semilogy(range(len(h)), h, label=rf"$\alpha={float(a):g}$", **style)
            ax.set_title(f"{NAMES[s]}, $m={m}$", fontsize=8.5)
            ax.grid(True, which="major", alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i == 1:
                ax.set_xlabel("GCR step")
            if j == 0:
                ax.set_ylabel(r"$|r|/|b|$")
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "fig_history_12.pdf", dpi=200)


if __name__ == "__main__":
    history()
