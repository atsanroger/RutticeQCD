#!/usr/bin/env python3
"""The figures of the MG note from the data in data/ (python3 figures.py in this directory).

fig_massscan.pdf: the 24^3x48 mass scan, CGNE, Krylov on A~_L without the multigrid, the multigrid
fig_spectrum.pdf: the spectrum of A~ M, sigma_min(z - A~ M) on contours, 12^4
fig_history.pdf: the GCR(2) residual histories on 24^3x48
fig_cgform.pdf: CG iterations on K^dag K and S^dag S and their ratio against the Moebius c, three volumes
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
    data = rows("massscan_24_configs.tsv")
    nomg = rows("baseline_mass_2448.tsv")
    masses = sorted({float(r["mass"]) for r in data})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.8), sharex=True)

    def band(setup, solver, alpha, value):
        """Mean over the configurations and the distance to the smallest and largest value."""
        mean, lo, hi = [], [], []
        for m in masses:
            v = [value(r) for r in data if r["setup"] == setup and r["solver"] == solver
                 and r["alpha"] == alpha and float(r["mass"]) == m]
            mu = sum(v) / len(v)
            mean.append(mu)
            lo.append(mu - min(v))
            hi.append(max(v) - mu)
        return mean, [lo, hi]

    its = lambda r: int(r["iterations"])
    secs = lambda r: float(r["solve_s"])
    for col, setup in enumerate("SB"):
        other = lambda method: sorted(
            (float(r["mass"]), r) for r in nomg if r["setup"] == setup and r["method"] == method)
        bi, ch = other("bicg"), other("cheb")
        for row, (value, ylabel) in enumerate([(its, "iterations"), (secs, "seconds per right-hand side")]):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_yscale("log")
            for solver, alpha, fmt, color, a, lab in [
                    ("CGNE", "0.6", "o-", "C3", 1.0, "CGNE"),
                    ("CGNE_mixed", "0.6", "o--", "C3", 0.6, "CGNE, mixed"),
                    ("MG", "0.6", "s-", "C0", 1.0, r"MG, $\alpha=0.6$" if row == 0 else "MG, solve"),
                    ("MG", "1.0", "^--", "C0", 0.6, r"MG, $\alpha=1$")]:
                if row == 1 and alpha == "1.0":
                    continue
                y, err = band(setup, solver, alpha, value)
                ax.errorbar(masses, y, yerr=err, fmt=fmt, color=color, alpha=a, capsize=2, label=lab)
            col_bi = [float(r["iterations"]) if row == 0 else float(r["solve_s"]) for _, r in bi]
            col_ch = [float(r["iterations"]) if row == 0 else float(r["solve_s"]) for _, r in ch]
            ax.plot([x for x, _ in bi], col_bi, "D-", color="C2", label=r"BiCGStab on $\tilde A_L$")
            ax.plot([x for x, _ in ch], col_ch, "v-", color="C4", label="GCR(2), Chebyshev")
            if row == 1:
                y, _ = band(setup, "MG", "0.6", lambda r: float(r["solve_s"]) + float(r["setup_s"]) / 12)
                ax.plot(masses, y, "s:", color="C0", alpha=0.7, label="MG, solve + setup/12")
                y, _ = band(setup, "MG", "0.6", lambda r: float(r["solve_s"]) + float(r["setup_s"]))
                ax.plot(masses, y, "s:", color="C7", alpha=0.7, label="MG, solve + setup")
                ax.set_xlabel(r"quark mass $m$")
            else:
                ax.set_title(NAMES[setup], fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(masses)
            ax.set_xticklabels([f"{m:g}" for m in masses], fontsize=6.5)
            ax.minorticks_off()
            ax.grid(True, which="major", alpha=0.25)
    axes[0, 0].invert_xaxis()
    handles, labels = axes[1, 0].get_legend_handles_labels()
    h_top, l_top = axes[0, 0].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h_top, l_top) if l not in labels and "0.6" not in l]
    handles += [h for h, _ in extra]
    labels += [l for _, l in extra]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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


def cgform():
    data = rows("cgne_form.tsv")
    it = {(r["volume"], r["setup"], float(r["c"]), r["alpha"], r["operator"]): int(r["iterations"])
          for r in data}
    vols = {"12^4": ("C0", "o", r"$12^4$"), "16^4": ("C1", "s", r"$16^4$"),
            "24^3x48": ("C2", "^", r"$24^3\times48$")}
    names = {"K_dag_K": r"$\mathcal{K}^\dagger\mathcal{K}$", "S_dag_S": r"$S^\dagger S$"}
    fig, axes = plt.subplots(4, 2, figsize=(6.6, 10.4), sharex=True)
    for blk, a in enumerate(("1.0", "0.6")):
        for col, setup in enumerate("SB"):
            top, bot = axes[2 * blk, col], axes[2 * blk + 1, col]
            for vol, (color, mk, lab) in vols.items():
                sel = lambda op: sorted((c, n) for (v, s, c, al, o), n in it.items()
                                        if v == vol and s == setup and al == a and o == op)
                for op, ls in (("K_dag_K", "-"), ("S_dag_S", "--")):
                    pts = sel(op)
                    top.semilogy([c for c, _ in pts], [n for _, n in pts], mk + ls, color=color,
                                 mfc=color if op == "K_dag_K" else "none", label=lab + ", " + names[op])
                k, sq = dict(sel("K_dag_K")), dict(sel("S_dag_S"))
                cs = sorted(k)
                bot.plot(cs, [sq[c] / k[c] for c in cs], mk + "-", color=color, label=lab)
            bot.axhline(1, color="k", lw=0.5, ls=":")
            top.set_title(NAMES[setup] + rf", $\alpha={float(a):g}$", fontsize=9)
            for ax in (top, bot):
                ax.grid(True, which="both", alpha=0.25)
        for r in (2 * blk, 2 * blk + 1):
            lo = min(ax.get_ylim()[0] for ax in axes[r]); hi = max(ax.get_ylim()[1] for ax in axes[r])
            for ax in axes[r]:
                ax.set_ylim(lo, hi)
        axes[2 * blk, 0].set_ylabel("CG iterations")
        axes[2 * blk + 1, 0].set_ylabel(r"$S^\dagger S$ / $\mathcal{K}^\dagger\mathcal{K}$")
    for ax in axes[-1]:
        ax.set_xlabel(r"Möbius $c$ ($b=1+c$)")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, "fig_cgform.pdf"))

if __name__ == "__main__":
    massscan()
    spectrum()
    history()
    cgform()
