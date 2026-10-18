#!/usr/bin/env python3
"""The figures of the MG note from the data in data/ (python3 figures.py in this directory).

fig_massscan.pdf: the 24^3x48 mass scan, CGNE against the multigrid (A30).
fig_spectrum.pdf: the spectrum of A~ M, sigma_min(z - A~ M) on contours, 12^4 (RTX 3080).
fig_history.pdf: the GCR(2) residual histories on 24^3x48 (A30).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})
NAMES = {"S": r"setup S ($M_5=1.0$, stout)", "B": r"setup B ($M_5=1.8$)"}


def rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def massscan():
    data = rows("massscan_24.tsv")
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), sharex=True)
    for col, setup in enumerate("SB"):
        sel = lambda solver, alpha: sorted(
            (float(r["mass"]), r) for r in data
            if r["setup"] == setup and r["solver"] == solver and r["alpha"] == alpha)
        cg, mg, mg1 = sel("CGNE", "0.6"), sel("MG", "0.6"), sel("MG", "1.0")
        mx = sel("CGNE_mixed", "0.6")
        m = [x for x, _ in cg]
        ax = axes[0, col]
        ax.loglog(m, [int(r["iterations"]) for _, r in cg], "o-", color="C3", label="CGNE")
        ax.loglog(m, [int(r["iterations"]) for _, r in mx], "o--", color="C3", alpha=0.6,
                  label="CGNE, mixed")
        ax.loglog(m, [int(r["iterations"]) for _, r in mg], "s-", color="C0",
                  label=r"MG, $\alpha=0.6$")
        ax.loglog(m, [int(r["iterations"]) for _, r in mg1], "^--", color="C0", alpha=0.6,
                  label=r"MG, $\alpha=1$")
        ax.set_title(NAMES[setup], fontsize=9)
        if col == 0:
            ax.set_ylabel("iterations")
            ax.legend(loc="upper left", frameon=False)
        ax.grid(True, which="both", alpha=0.25)
        ax = axes[1, col]
        ax.loglog(m, [float(r["solve_s"]) for _, r in cg], "o-", color="C3", label="CGNE")
        ax.loglog(m, [float(r["solve_s"]) for _, r in mx], "o--", color="C3", alpha=0.6,
                  label="CGNE, mixed")
        ax.loglog(m, [float(r["solve_s"]) for _, r in mg], "s-", color="C0",
                  label="MG, solve")
        ax.loglog(m, [float(r["solve_s"]) + float(r["setup_s"]) / 12 for _, r in mg], "s:",
                  color="C0", alpha=0.7, label="MG, solve + setup/12")
        ax.loglog(m, [float(r["solve_s"]) + float(r["setup_s"]) for _, r in mg], "s:",
                  color="C7", alpha=0.7, label="MG, solve + setup")
        ax.set_xlabel(r"quark mass $m$")
        if col == 0:
            ax.set_ylabel("seconds per right-hand side")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].invert_xaxis()
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_massscan.pdf"))


def spectrum():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    norm = LogNorm(vmin=1e-2, vmax=1.5)
    for ax, setup in zip(axes, ["S12", "B12"]):
        pts = rows(f"pseudo_{setup}.tsv")
        sc = ax.scatter([float(p["re"]) for p in pts], [float(p["im"]) for p in pts],
                        c=[float(p["sigma_min"]) for p in pts], cmap="viridis", norm=norm,
                        s=9, marker="o", linewidths=0, zorder=2)
        spec = rows(f"spectrum_{setup}.tsv")
        low = [(float(s["re"]), float(s["im"])) for s in spec if s["kind"] == "smallest_re"]
        farv = [(float(s["re"]), float(s["im"])) for s in spec if s["kind"] == "farthest_from_1"]
        fov = [float(s["re"]) for s in spec if s["kind"].startswith("fov")]
        ax.plot(*zip(*low), "x", color="C3", ms=5, label=r"$\lambda$, smallest Re")
        inside = [z for z in farv if abs(complex(*z) - 1) < 1.0]
        outside = [z for z in farv if abs(complex(*z) - 1) >= 1.0]
        if inside:
            ax.plot(*zip(*inside), "+", color="C1", ms=6, label=r"$\lambda$, farthest from 1")
        ax.plot([0], [0], "k*", ms=6)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-0.65, 2.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Re $z$")
        title = NAMES["S" if setup.startswith("S") else "B"] + r", $12^4$"
        ax.set_title(title, fontsize=9)
        note = r"Re $W(\tilde A\mathcal{M})\in[%.3g,\,%.3g]$" % (fov[0], fov[1])
        for z in outside:
            ax.annotate(r"outlier $%.2f%+.2fi$" % z, xy=(2.58, 0.86),
                        xytext=(1.2, 1.0), fontsize=7, va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.6))
        ax.text(-0.6, -1.05, note, fontsize=7)
    axes[0].set_ylabel(r"Im $z$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color="k", marker="*", ls="", ms=6)],
               labels + ["origin"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.45, 0.04))
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"$\sigma_{\min}(z-\tilde A\mathcal{M})$")
    fig.savefig(os.path.join(HERE, "fig_spectrum.pdf"), bbox_inches="tight")


def history():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    styles = {("0.001", "0.6"): ("C0", "-"), ("0.001", "1.0"): ("C0", "--"),
              ("0.002", "0.6"): ("C2", "-"), ("0.002", "1.0"): ("C2", "--")}
    for ax, setup in zip(axes, "SB"):
        for r in rows("gcr_steps_24.tsv"):
            if r["setup"] != setup:
                continue
            res = [float(x) for x in r["r2_over_b2"].split()]
            color, ls = styles[(r["mass"], r["alpha"])]
            ax.semilogy(range(len(res)), res, ls, color=color, marker=".", ms=3,
                        label=r"$m=%s$, $\alpha=%s$" % (r["mass"], r["alpha"]))
        ax.axhline(1e-18, color="k", lw=0.5, ls=":")
        ax.set_xlabel("GCR(2) step")
        ax.set_title(NAMES[setup] + r", $24^3\times48$", fontsize=9)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="upper right", frameon=False)
    axes[0].set_ylabel(r"$|r|^2/|b|^2$")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_history.pdf"))


if __name__ == "__main__":
    massscan()
    spectrum()
    history()
