#!/usr/bin/env python3
"""Check data/manifest.tsv: for every row, re-extract the printed value from the .tex at file:line,
and, where the row names a data file or logs, recompute it from them and compare.

    python3 -I check_manifest.py            # all rows
    python3 -I check_manifest.py -v         # also the rows that agree

Output: one line per row, OK / MISMATCH / TEX-CHANGED / NOCHECK (no automatic source), then the
counts. Run directories are found as red_gpu/*/multiconf; paths are printed as red_gpu/X/...
Read only: it writes nothing.
"""
import csv
import glob
import math
import os
import re
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
_mc = sorted(glob.glob("/home/chen/red_gpu/*/multiconf"))
MC = _mc[0] if _mc else None
G = os.path.dirname(MC) if MC else None
AR = MC + "/autoR" if MC else None
VERBOSE = "-v" in sys.argv


def hide(s):
    return re.sub(r"/home/chen/red_gpu/[^/ ]+", "red_gpu/X", str(s))


def real(p):
    """red_gpu/X/... of the manifest -> the path on this host."""
    return p.replace("red_gpu/X", G, 1) if p.startswith("red_gpu/X") else p


# ---------------------------------------------------------------- formatting
def hu(x, nd=0):
    """Round half up (the rule of the note's appendix)."""
    d = Decimal(repr(round(float(x), 9)))  # round(.., 9) removes the float error of a mean (9.15 -> 9.149999...)
    if nd < 0:
        f = Decimal(10) ** (-nd)
        return str(int((d / f).quantize(Decimal(1), rounding=ROUND_HALF_UP) * f))
    return str(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def rng(v, nd=0):
    a, b = hu(min(v), nd), hu(max(v), nd)
    return a if a == b else f"{a}-{b}"


def mean(v):
    return sum(v) / len(v)


def norm(s):
    """A printed cell or value to a comparable string."""
    s = str(s)
    s = re.sub(r"\\textcolor\{red\}\{New!!!\}", "", s)
    s = s.replace("\\,", "").replace("~", " ").replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("--", "-").replace("\\times10^", "e").replace("\\ge", ">=").replace("\\le", "<=")
    s = s.replace("\\", "").replace("n.c.", "nc").replace("  ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------- tex
_tex = {}


def tex_lines(f):
    if f not in _tex:
        _tex[f] = open(os.path.join(HERE, f)).read().split("\n")
    return _tex[f]


def cells(f, line):
    """The cells of the table row at file:line, without the trailing \\\\."""
    t = tex_lines(f)[line - 1].strip()
    t = re.sub(r"\\\\\s*$", "", t)
    return [norm(c) for c in t.split("&")]


# ---------------------------------------------------------------- data files
def tsv(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t"))


# ---------------------------------------------------------------- logs
P_SOLVE = re.compile(r"(GCR|CG(?: \(mixed precision\))?|BiCGStab) (\d+)(!?) iterations on the device; "
                     r"(?:(\d+) coarse applications; )?worst \|Dx−b\|² (\S+); ([\d.]+) s")
P_SETUP = re.compile(r"setup on the device: (\d+) s")
P_ROUND = re.compile(r"bootstrap round (\d+): lowest Ritz value of \S+ ([\d.e+-]+)")
P_PEAK = re.compile(r"device [\d.]+ GB, peak ([\d.]+) GB")
_logc = {}


def resolve(path):
    """The configuration tags differ between run groups (__502 or __000502): try both."""
    path = real(path)
    if os.path.exists(path):
        return path
    for alt in (re.sub(r"__000(\d{3})", r"__\1", path), re.sub(r"__(\d{3})(?=[._])", r"__000\1", path)):
        if os.path.exists(alt):
            return alt
    return path


def has(path):
    return os.path.exists(resolve(path))


def log(path):
    path = resolve(path)
    if path not in _logc:
        _logc[path] = open(path, errors="replace").read()
    return _logc[path]


def solves(path):
    """(iterations, converged, coarse applications, seconds) of every solve line of a ruttice log."""
    out = []
    for m in P_SOLVE.finditer(log(path)):
        out.append(dict(it=int(m.group(2)), ok=m.group(3) != "!", coarse=int(m.group(4) or 0),
                        s=float(m.group(6)), worst=float(m.group(5))))
    return out


def setup_s(path):
    m = P_SETUP.findall(log(path))
    return int(m[-1]) if m else None


def rounds(path):
    return [float(b) for _, b in P_ROUND.findall(log(path))]


def peak(path):
    v = [float(x) for x in P_PEAK.findall(log(path))]
    return max(v) if v else None


CFG = {"12": ["v2", "000502", "000602", "000702"], "16": ["v2", "000502", "000602", "000702"],
       "24": ["v2", "000502", "000602", "000702"], "2448": ["v2", "000702", "000802", "000902"]}


def cfgs(vol):
    return CFG[str(vol)]


# ---------------------------------------------------------------- checks
CHECKS = {}


def check(name):
    def deco(f):
        CHECKS[name] = f
        return f
    return deco


def ms_rows():
    return tsv("massscan_24_configs.tsv")


def ms_sel(setup, solver, alpha, mass):
    return [r for r in ms_rows() if r["setup"] == setup and r["solver"] == solver and r["alpha"] == alpha
            and r["mass"] == mass]


MASSES = ["0.1", "0.05", "0.02", "0.01", "0.005", "0.002", "0.001"]


def nrhs_fmt(x):
    return hu(x, 0) if x >= 10 else hu(x, 1)


def ms_table(setup):
    """The cells of Table massscan for one setup, from massscan_24_configs.tsv."""
    out = []
    for m in MASSES:
        cg, mg, m1 = ms_sel(setup, "CGNE_mixed", "0.6", m), ms_sel(setup, "MG", "0.6", m), ms_sel(setup, "MG", "1.0", m)
        f = lambda L, k: mean([float(r[k]) for r in L])
        nr = f(mg, "setup_s") / (f(cg, "solve_s") - f(mg, "solve_s"))
        sp = [float(c["solve_s"]) / float(x["solve_s"]) for c in cg for x in mg if c["config"] == x["config"]]
        out.append([m, hu(f(cg, "iterations")), hu(f(cg, "solve_s"), 1), hu(f(mg, "iterations")),
                    hu(f(mg, "solve_s"), 1), hu(f(mg, "coarse_apps")), hu(f(m1, "iterations")),
                    hu(f(m1, "solve_s"), 1), nrhs_fmt(nr), rng(sp, 1)])
    return out


def table_rows(f, first, n):
    return [cells(f, first + i) for i in range(n)]


def cols_cmp(tex, comp, cols):
    """Mismatches of the columns `cols` between tex rows and computed rows."""
    bad = []
    for t, c in zip(tex, comp):
        for j in cols:
            if norm(t[j]) != norm(c[j]):
                bad.append(f"row {t[0]} col {j}: tex {t[j]} vs {c[j]}")
    return bad


def ms_check(setup, cols):
    first = 144 if setup == "S" else 153
    return cols_cmp(table_rows("results.tex", first, 7), ms_table(setup), cols)


for s in "SB":
    for name, cols in [("cgit", [1]), ("cgs", [2]), ("mg06", [3, 4, 5]), ("mg10", [6, 7]), ("nrhs", [8]),
                       ("speedup", [9])]:
        CHECKS[f"massscan.{s}.{name}"] = (lambda s=s, cols=cols: ms_check(s, cols))


@check("massscan.logs")
def _():
    """massscan_24_configs.tsv against the logs: MG rows (autoR/rerunT/T_ms_*), CGNE rows (cgneT)."""
    bad = []
    for r in ms_rows():
        c = cfgs("2448")[int(r["config"]) - 1]
        if r["solver"] == "MG":
            p = f"{AR}/rerunT/T_ms_{r['setup']}_m{r['mass']}_a{r['alpha']}__{c}.log"
            s = solves(p)[0]
            got = (s["it"], hu(s["s"], 1), setup_s(p), s["coarse"])
            want = (int(r["iterations"]), hu(float(r["solve_s"]), 1), int(r["setup_s"]), int(r["coarse_apps"]))
        else:
            kind = "mixed" if r["solver"] == "CGNE_mixed" else "double"
            p = f"{MC}/cgneT/C_2448_{r['setup']}_{r['mass']}_{kind}__{r['config']}.log"
            s = solves(p)[0]
            got, want = (s["it"], hu(s["s"], 1)), (int(r["iterations"]), hu(float(r["solve_s"]), 1))
        if got != want:
            bad.append(f"{hide(p)}: log {got} tsv {want}")
    return bad


def ratio_growth(setup, solver):
    v = []
    for c in "1234":
        a = [r for r in ms_sel(setup, solver, "0.6", "0.1") if r["config"] == c][0]
        b = [r for r in ms_sel(setup, solver, "0.6", "0.001") if r["config"] == c][0]
        v.append(int(b["iterations"]) / int(a["iterations"]))
    return v


@check("text.growth.mg")
def _():
    v = ratio_growth("S", "MG") + ratio_growth("B", "MG")
    return [] if rng(v, 1) == "2.0-2.4" else [f"MG growth {rng(v, 1)} (per configuration, S and B)"]


@check("text.growth.cg")
def _():
    v = ratio_growth("S", "CGNE_mixed") + ratio_growth("B", "CGNE_mixed")
    return [] if rng(v, 0) == "20-32" else [f"CGNE mixed growth {rng(v, 0)}"]


@check("text.speedup.2448")
def _():
    v = []
    for s in "SB":
        cg, mg = ms_sel(s, "CGNE_mixed", "0.6", "0.001"), ms_sel(s, "MG", "0.6", "0.001")
        v += [float(c["solve_s"]) / float(x["solve_s"]) for c in cg for x in mg if c["config"] == x["config"]]
    return [] if rng(v, 1) == "6.6-11.1" else [f"speed-up {rng(v, 1)}"]


@check("text.payback")
def _():
    v = []
    for s in "SB":
        cg, mg = ms_sel(s, "CGNE_mixed", "0.6", "0.001"), ms_sel(s, "MG", "0.6", "0.001")
        v += [float(x["setup_s"]) / (float(c["solve_s"]) - float(x["solve_s"])) for c in cg for x in mg
              if c["config"] == x["config"]]
    return [] if rng(v, 1) == "2.5-4.6" else [f"payback {rng(v, 1)}"]


@check("text.payback.ps")
def _():
    bad = []
    for s in "SB":
        for t in ms_table(s):
            if float(t[0]) <= 0.01 and float(t[8]) > 12:
                bad.append(f"{s} m={t[0]}: N_rhs {t[8]} > 12")
    return bad


@check("text.alpha1.extra")
def _():
    d = []
    for s in "SB":
        for t in ms_table(s):
            d.append(int(t[6]) - int(t[3]))
    return [] if (min(d), max(d)) == (0, 3) else [f"alpha=1 minus 0.6: {min(d)}..{max(d)}"]


@check("text.faster.every.mass")
def _():
    bad = []
    for s in "SB":
        for t in ms_table(s):
            if float(t[9].split("-")[0]) <= 1:
                bad.append(f"{s} m={t[0]} speed-up {t[9]}")
    return bad


# ---- memory
def mg2448_logs():
    return sorted(glob.glob(f"{AR}/rerunT/T_ms_*.log"))


@check("text.memory.mg")
def _():
    v = [peak(p) for p in mg2448_logs() + sorted(glob.glob(f"{AR}/rerunT/T_ps_*.log"))]
    r = f"{min(v):.1f}-{max(v):.1f}"
    return [] if (round(min(v)) >= 20 and math.ceil(max(v)) <= 23) else [f"device peak of 24^3x48 MG runs {r} GB"]


@check("text.memory.cg")
def _():
    v = [peak(p) for p in glob.glob(f"{MC}/cgneT/C_2448_*_mixed__*.log")]
    d = [peak(p) for p in glob.glob(f"{MC}/cgneT/C_2448_*_double__*.log")]
    out = []
    if hu(max(v), 1) != "7.7":
        out.append(f"mixed peak {min(v)}-{max(v)}")
    if hu(max(d), 1) != "7.4":
        out.append(f"double peak {min(d)}-{max(d)}")
    return out


# ---- CGNE double vs mixed (appendix)
@check("text.mixed.vs.double")
def _():
    it, tm = [], []
    for r in ms_rows():
        if r["solver"] != "CGNE_mixed":
            continue
        d = [x for x in ms_sel(r["setup"], "CGNE", "0.6", r["mass"]) if x["config"] == r["config"]][0]
        it.append((r["setup"], float(r["mass"]), int(r["iterations"]) / int(d["iterations"])))
        tm.append(float(r["solve_s"]) / float(d["solve_s"]))
    within = [x for s, m, x in it if not (s == "S" and m <= 0.002)]
    out = []
    if max(abs(x - 1) for x in within) > 0.01:
        out.append(f"iterations outside S m<=0.002 differ up to {max(abs(x-1) for x in within):.3f}")
    worst = max(x for s, m, x in it)
    if hu((worst - 1) * 100) != "67":
        out.append(f"up to {(worst-1)*100:.1f}% more")
    if rng(tm, 2) != "0.53-0.88":
        out.append(f"time ratio {rng(tm, 2)}")
    return out


# ---- bandwidth
@check("text.bandwidth")
def _():
    t = log(f"{MC}/cgneT/C_2448_S_0.001_double__1.log")
    m = re.search(r"([\d.]+) GFlop/s, ([\d.]+) Flop/byte", t)
    gbs = float(m.group(1)) / float(m.group(2))
    return [f"counted bandwidth {gbs:.0f} GB/s (to compare with the spec peak by hand)"]


# ---- Bridge++
@check("bridge.table")
def _():
    rows = tsv("bridge_cgnr_24.tsv")
    tex = table_rows("results.tex", 113, 3)
    comp = [[r["mass"], f"{r['bridge_iter_min']}-{r['bridge_iter_max']}", r["ruttice_iter"],
             hu(float(r["bridge_s_per_iter"]) * 1000, 1), hu(float(r["ruttice_s_per_iter"]) * 1000, 1),
             r["ratio_per_iter"]] for r in rows]
    bad = cols_cmp(tex, comp, [1, 2, 3, 4, 5])
    for r in rows:
        p = f"{MC}/cgneT/C_2448_S_{r['mass']}_double__1.log"
        s = solves(p)[0]
        if (s["it"], hu(s["s"], 1)) != (int(r["ruttice_iter"]), hu(float(r["ruttice_s"]), 1)):
            bad.append(f"{hide(p)} {s['it']} {s['s']} vs tsv {r['ruttice_iter']} {r['ruttice_s']}")
    return bad


@check("text.bridge")
def _():
    rows = tsv("bridge_cgnr_24.tsv")
    dev = [abs(int(r["ruttice_iter"]) - int(x)) / int(r["ruttice_iter"]) for r in rows
           for x in (r["bridge_iter_min"], r["bridge_iter_max"])]
    rat = [float(r["ratio_per_iter"]) for r in rows]
    out = []
    if max(dev) >= 0.05:
        out.append(f"iterations differ up to {max(dev):.3f}")
    if rng(rat, 2) != "1.46-1.54":
        out.append(f"ratio {rng(rat, 2)}")
    return out


# ---- point source
def ps_rows():
    return tsv("pointsource_24_configs.tsv") + [dict(r, config="1") for r in tsv("pointsource_24.tsv")]


@check("ps.mg.0001")
def _():
    bad = []
    for s, line in (("S", 184), ("B", 185)):
        t = cells("results.tex", line)
        su, it, sec = [], [], []
        for c in cfgs("2448"):
            p = f"{AR}/rerunT/T_ps_{s}__{c}.log"
            su.append(setup_s(p))
            sv = solves(p)
            it += [x["it"] for x in sv]
            sec += [x["s"] for x in sv]
        comp = [rng(su), rng(it), rng(sec, 1)]
        for j, v in zip((2, 3, 4), comp):
            if norm(t[j]) != v:
                bad.append(f"{s} col {j}: tex {t[j]} vs logs {v}")
    return bad


@check("ps.mg.001")
def _():
    bad = []
    for s, line in (("S", 186), ("B", 187)):
        t = cells("results.tex", line)
        p = f"{AR}/psT/T_ps_{s}_m0.01__v2.log"
        sv = solves(p)
        comp = [str(setup_s(p)), rng([x["it"] for x in sv]), rng([x["s"] for x in sv], 1)]
        for j, v in zip((2, 3, 4), comp):
            if norm(t[j]) != v:
                bad.append(f"{s} col {j}: tex {t[j]} vs logs {v}")
    return bad


def ps_cg_logs(s, m):
    tag = "stout" if s == "S" else "m18"
    if m == "0.01":
        return [f"{AR}/truestop/ps_{tag}_m0.01__v2.log"]
    return [f"{AR}/truestop/ps_{tag}_m0.001__{c}.log" for c in cfgs("2448")]


@check("ps.cg")
def _():
    bad = []
    for s, m, line in (("S", "0.001", 184), ("B", "0.001", 185), ("S", "0.01", 186), ("B", "0.01", 187)):
        t = cells("results.tex", line)
        sv = [x for p in ps_cg_logs(s, m) for x in solves(p)]
        comp = [rng([x["it"] for x in sv]), rng([x["s"] for x in sv], 1)]
        for j, v in zip((5, 6), comp):
            if norm(t[j]) != v:
                bad.append(f"{s} m={m} col {j}: tex {t[j]} vs logs {v}")
    return bad


def ps_wall(s, m):
    """Wall time (setup + 12 solves) of MG and of CGNE per configuration, from the data files."""
    mg = {r["config"]: float(r["wall_s"]) for r in ps_rows() if r["setup"] == s and r["mass"] == m and r["solver"] == "MG"}
    cg = {r["config"]: float(r["wall_s"]) for r in ps_rows() if r["setup"] == s and r["mass"] == m
          and r["solver"] == "CGNE_mixed"}
    return mg, cg


@check("ps.wall")
def _():
    bad = []
    for s, m, line in (("S", "0.001", 184), ("B", "0.001", 185), ("S", "0.01", 186), ("B", "0.01", 187)):
        t = cells("results.tex", line)
        mg, cg = ps_wall(s, m)
        comp = [rng(list(mg.values())), rng(list(cg.values()))]
        for j, v in zip((7, 8), comp):
            if norm(t[j]) != v:
                bad.append(f"{s} m={m} col {j}: tex {t[j]} vs tsv {v}")
        # wall_s of the data file against setup + the seconds of the 12 solves of the logs
        if m == "0.001":
            off = []
            for i, c in enumerate(cfgs("2448")):
                p = f"{AR}/rerunT/T_ps_{s}__{c}.log"
                off.append(mg[str(i + 1)] - setup_s(p) - sum(x["s"] for x in solves(p)))
                off.append(cg[str(i + 1)] - sum(x["s"] for x in solves(ps_cg_logs(s, m)[i])))
            bad.append(f"INFO: {s} wall_s minus (setup + 12 solves) of the logs: {min(off):.0f}..{max(off):.0f} s "
                       "(how wall_s was taken is not recorded)")
    return bad


@check("text.ps.ratio")
def _():
    v = []
    for s in "SB":
        mg, cg = ps_wall(s, "0.001")
        v += [mg[c] / cg[c] for c in mg]
    out = [] if rng(v, 2) == "0.27-0.49" else [f"0.001 ratio {rng(v, 2)}"]
    for s, want in (("S", "0.79"), ("B", "0.84")):
        mg, cg = ps_wall(s, "0.01")
        if hu(mg["1"] / cg["1"], 2) != want:
            out.append(f"{s} m=0.01 ratio {mg['1']/cg['1']:.3f}")
    return out


@check("ps.correlator")
def _():
    v = [float(r.get("PS_rel_diff_t0_3") or r.get("PS_rel_diff_t0-3") or "nan") for r in ps_rows()
         if r["solver"] == "MG" and (r.get("PS_rel_diff_t0_3") or r.get("PS_rel_diff_t0-3"))]
    return [f"pseudoscalar correlator MG vs CGNE, relative difference t=0..3: max {max(v):.1e} ({len(v)} runs)"]


# ---- volume
def vol_rows():
    return tsv("volume_configs.tsv")


def vol_sel(setup, vol, method):
    if vol == "2448":
        solver = {"cgm": "CGNE_mixed", "mg": "MG", "cgd": "CGNE"}[method]
        return [dict(iterations=r["iterations"], solve_s=r["solve_s"], setup_s=r["setup_s"]) for r in
                ms_sel(setup, solver, "0.6", "0.001")]
    return [r for r in vol_rows() if r["setup"] == setup and r["volume"] == vol and r["mass"] == "0.001"
            and r["method"] == method]


def vol_cells(setup):
    out = {"cgit": [], "cgs": [], "bit": [], "bs": [], "mgit": [], "mgs": []}
    bl = {r["volume"]: r for r in tsv("baseline_volume.tsv") if r["setup"] == setup and r["mass"] == "0.001"
          and r["method"] == "bicg"}
    bl["2448"] = [r for r in tsv("baseline_mass_2448.tsv") if r["setup"] == setup and r["mass"] == "0.001"
                  and r["method"] == "bicg"][0]
    for v in ("12", "16", "24", "2448"):
        cg, mg = vol_sel(setup, v, "cgm"), vol_sel(setup, v, "mg")
        out["cgit"].append(rng([float(r["iterations"]) for r in cg]))
        out["cgs"].append(rng([float(r["solve_s"]) for r in cg], 1))
        out["bit"].append(bl[v]["iterations"])
        out["bs"].append(hu(float(bl[v]["solve_s"]), 1))
        out["mgit"].append(rng([float(r["iterations"].strip("!")) for r in mg]))
        out["mgs"].append(rng([float(r["solve_s"]) for r in mg], 1) + f" (+{hu(mean([float(r['setup_s']) for r in mg]))})")
    return out


def vol_check(setup, key, line):
    t = cells("results_tables.tex", line)[3:]
    c = vol_cells(setup)[key]
    return [f"{setup} {key} col {j}: tex {a} vs {b}" for j, (a, b) in enumerate(zip(t, c)) if norm(a) != norm(b)]


for s, base in (("S", 15), ("B", 22)):
    for k, off in (("cgit", 0), ("cgs", 1), ("bit", 2), ("bs", 3), ("mgit", 4), ("mgs", 5)):
        CHECKS[f"volume.{s}.{k}"] = (lambda s=s, k=k, l=base + off: vol_check(s, k, l))


@check("volume.logs")
def _():
    bad = []
    for r in vol_rows():
        if r["mass"] != "0.001":
            continue
        c = cfgs(r["volume"])[int(r["config"]) - 1]
        if r["method"] == "mg":
            p = f"{AR}/rerunT/T_vol_{r['setup']}_{r['volume']}__{c}.log"
        else:
            kind = "mixed" if r["method"] == "cgm" else "double"
            p = f"{MC}/cgneT/C_{r['volume']}_{r['setup']}_0.001_{kind}__{r['config']}.log"
        s = solves(p)[0]
        if (s["it"], hu(s["s"], 1)) != (int(r["iterations"].strip("!")), hu(float(r["solve_s"]), 1)):
            bad.append(f"{hide(p)}: log {s['it']} {s['s']} tsv {r['iterations']} {r['solve_s']}")
    return bad


@check("text.volume.growth")
def _():
    out = []
    for s in "SB":
        a = med([float(r["iterations"]) for r in vol_sel(s, "12", "cgm")])
        b = med([float(r["iterations"]) for r in vol_sel(s, "2448", "cgm")])
        c = med([float(r["iterations"]) for r in vol_sel(s, "12", "mg")])
        d = med([float(r["iterations"]) for r in vol_sel(s, "2448", "mg")])
        out.append(f"{s}: CGNE median x{b/a:.1f}, MG median x{d/c:.2f}")
    return out


@check("text.volume.order")
def _():
    out = []
    for s in "SB":
        for v in ("12", "16", "24", "2448"):
            cg = [float(r["solve_s"]) for r in vol_sel(s, v, "cgm")]
            mg = [float(r["solve_s"]) for r in vol_sel(s, v, "mg")]
            out.append(f"{s} {v}: CGNE {min(cg)}-{max(cg)} s, MG {min(mg)}-{max(mg)} s")
    return out


# ---- baselines without the multigrid
@check("text.nomg")
def _():
    out = []
    bl = tsv("baseline_mass_2448.tsv")
    for s in "SB":
        r = []
        slower = []
        for m in MASSES:
            b = [x for x in bl if x["setup"] == s and x["mass"] == m and x["method"] == "bicg"][0]
            ch = [x for x in bl if x["setup"] == s and x["mass"] == m and x["method"] == "cheb"][0]
            d = [x for x in ms_sel(s, "CGNE", "0.6", m) if x["config"] == "1"][0]
            mx = [x for x in ms_sel(s, "CGNE_mixed", "0.6", m) if x["config"] == "1"][0]
            r.append(int(b["iterations"]) / int(d["iterations"]))
            slower.append(float(b["solve_s"]) > float(mx["solve_s"]))
            if not (int(ch["iterations"]) < int(b["iterations"]) and float(ch["solve_s"]) > float(b["solve_s"])):
                out.append(f"{s} m={m}: GCR(2)-Chebyshev {ch['iterations']} it {ch['solve_s']} s vs BiCGStab {b['iterations']} {b['solve_s']}")
        want = "0.53-0.99" if s == "S" else "0.40-0.50"
        if rng(r, 2) != want:
            out.append(f"{s} BiCGStab/CGNE-double iterations {rng(r, 2)} (tex {want})")
        if not all(slower):
            out.append(f"{s}: BiCGStab not slower than mixed CGNE at every mass")
    return out


# ---- sensitivity, costs
def sens(setup, change):
    return [r for r in tsv("sensitivity_12_configs.tsv") if r["setup"] == setup and r["change"] == change]


@check("sens.logs")
def _():
    bad = []
    for r in tsv("sensitivity_12_configs.tsv"):
        c = cfgs("12")[int(r["config"]) - 1]
        p = f"{AR}/rerunT/T_sens_{r['change']}__{c}.log"
        s = solves(p)[0]
        it = f"{s['it']}{'' if s['ok'] else '!'}"
        if (it, setup_s(p), hu(s["s"], 1)) != (r["iterations"], int(r["setup_s"]), hu(float(r["solve_s"]), 1)):
            bad.append(f"{hide(p)}: log {it} {setup_s(p)} {s['s']} vs tsv {r['iterations']} {r['setup_s']} {r['solve_s']}")
    return bad


@check("sens.rounds")
def _():
    out = []
    for s, pre, want in (("S", "stout", 2), ("B", "m18", 3)):
        n = [len(rounds(f"{AR}/rerunT/T_sens_{pre}_base__{c}.log")) for c in cfgs("12")]
        if set(n) != {want}:
            out.append(f"{s}: rounds {n}")
    return out


@check("sens.cap")
def _():
    v = set()
    for p in glob.glob(f"{AR}/rerunT/T_sens_*.yaml"):
        m = re.search(r"maximum_number_of_iteration\s*:\s*(\d+)", open(p).read())
        v.add(m.group(1))
    return [f"outer iteration limit in the T_sens yamls: {sorted(v)} (caption and figures.py use 1000)"]


COSTS = [("base", "base"), ("block2", "block_2"), ("block3", "block_3"), ("block6", "block_6"),
         ("vectors16", "vectors_16"), ("vectors24", "vectors_24"), ("vectors48", "vectors_48"),
         ("steps4", "steps_4"), ("steps6", "steps_6"), ("steps12", "steps_12"), ("steps16", "steps_16"),
         ("bootstrap1", "bootstrap_1"), ("bootstrap3", "bootstrap_3")]


def costs_cell(setup, change):
    pre = "stout" if setup == "S" else "m18"
    rows = sens(setup, f"{pre}_{change}")
    if not rows:
        return None
    it = [r["iterations"] for r in rows]
    nc = sum("!" in x for x in it)
    if nc >= 3:
        return "stag."
    good = [int(x) for x in it if "!" not in x]
    if nc == 2:
        iv = rng(good)
    else:
        iv = hu(med([int(x.strip("!")) if "!" not in x else 10 ** 9 for x in it]))
    iv = iv + (f" [{nc}]" if nc else "")
    return (f"{iv} / {hu(med([float(r['setup_s']) for r in rows]))} / "
            f"{hu(med([float(r['solve_s']) for r in rows]), 1)}")


@check("costs.table")
def _():
    bad = []
    for i, (_, ch) in enumerate(COSTS):
        t = cells("results_tables.tex", 92 + i)
        for j, s in ((1, "S"), (2, "B")):
            c = costs_cell(s, ch)
            if c is None:
                c = "(base)" if (s == "B" and ch == "steps_12") else "?"
            if norm(t[j]) != norm(c):
                bad.append(f"{ch} {s}: tex {t[j]} vs sensitivity tsv {c}")
    return bad


@check("costs.tsv")
def _():
    bad = []
    for r in tsv("costs_12.tsv"):
        rows = sens(r["setup"], ("stout_" if r["setup"] == "S" else "m18_") + r["change"].replace("block", "block_")
                    .replace("vectors", "vectors_").replace("steps", "steps_").replace("bootstrap", "bootstrap_"))
        if not rows:
            continue
        m = med([float(x["setup_s"]) for x in rows])
        if hu(m) != hu(float(r["setup_s"])):
            bad.append(f"{r['setup']} {r['change']}: costs setup {r['setup_s']} vs median {m}")
    return bad


@check("text.blocks6")
def _():
    th = rounds(f"{AR}/rerunT/T_sens_stout_block_6__502.log")
    g2, g3 = th[0] / th[1], th[1] / th[2]
    return [] if (hu(g2, 1), hu(g3, -1)) == ("3.5", "280") else [f"gains {g2:.2f} {g3:.0f}"]


# ---- probe pairs (two.sh reran the jobs that ended after one round; old logs in old1/)
P_END = re.compile(r"bootstrap round (\d+) not needed: its cycle lowers ([\d.e+-]+) to ([\d.e+-]+)")


def pairs():
    out = []
    for l in open(f"{AR}/two_jobs.txt"):
        g, t = l.strip().split("/")
        e = P_END.findall(log(f"{AR}/{g}/old1/{t}.log"))
        if not e or e[0][0] != "2":
            continue
        th1, pr = float(e[0][1]), float(e[0][2])
        th2 = rounds(f"{AR}/{g}/{t}.log")[1]
        out.append((g, t, (th1 / th2) / (th1 / pr)))
    return out


@check("text.probe")
def _():
    rr = [x for g, t, x in pairs() if g == "rerunT"]
    msr = [x for g, t, x in pairs() if t.startswith("T_ms")]
    allp = [x for g, t, x in pairs()]
    out = [f"rerunT pairs {len(rr)}, median {med(rr):.3f}; mass scan {rng(msr, 2)}; all groups {len(allp)} "
           f"pairs, median {med(allp):.3f}; all > 1: {all(x > 1 for x in allp)}"]
    return out


@check("text.autorounds")
def _():
    out = []
    for s in "SB":
        for m in MASSES:
            n = [len(rounds(f"{AR}/rerunT/T_ms_{s}_m{m}_a0.6__{c}.log")) for c in cfgs("2448")]
            fm = float(m)
            if fm >= 0.02 and set(n) != {2} or fm <= 0.002 and set(n) != {3}:
                out.append(f"{s} m={m}: rounds {n}")
    its = {s: [solves(f"{AR}/rerunT/T_ms_{s}_m0.001_a0.6__{c}.log")[0]["it"] for c in cfgs("2448")] for s in "SB"}
    if (rng(its["S"]), rng(its["B"])) != ("18-22", "16-18"):
        out.append(f"m=0.001 iterations {its}")
    two = []
    for p in glob.glob(f"{MC}/rerunT/T_ms_*_m0.001_*.log"):
        two += [x["it"] for x in solves(p)]
    out.append(f"INFO: two rounds (multiconf/rerunT, 0a124d0, old stopping rule): max {max(two)} (tex 44)")
    return out


@check("text.theta")
def _():
    out = []
    ff = re.compile(r"filter on the device: Ritz values of \S+ ([\d.e+-]+) \.\.")
    for s in "SB":
        th0 = [float(ff.search(log(p)).group(1)) for p in glob.glob(f"{AR}/rerunT/T_ms_{s}_*_a0.6__*.log")]
        out.append(f"{s}: theta_0 {min(th0):.1e}..{max(th0):.1e}")
    for m in ("0.1", "0.001"):
        last = [rounds(f"{AR}/rerunT/T_ms_S_m{m}_a0.6__{c}.log")[-1] for c in cfgs("2448")]
        out.append(f"S m={m}: last theta {min(last):.1e}..{max(last):.1e}")
    return out


# ---- GCR(k)
@check("gcrk.table")
def _():
    rows = tsv("gcrk12_configs.tsv")
    bad = []
    spec = [(188, "S_L_1.0e-2"), (189, "S_R_1.0e-2"), (190, "B_L_1.0e-2"), (191, "B_R_1.0e-2"), (192, "S_L_1.0e-8")]
    for line, case in spec:
        t = cells("results_tables.tex", line)
        for j, k in zip(range(2, 7), (2, 8, 16, 32, 64)):
            v = [int(r["iterations"]) for r in rows if r["case"] == f"{case}_k{k}"]
            c = hu(med(v)) if v else ""
            if norm(t[j]) != c:
                bad.append(f"{case} k={k}: tex '{t[j]}' vs median {c}")
            for r in rows:
                if r["case"] == f"{case}_k{k}":
                    p = f"{AR}/rerunT/T_gcrk12_{case}_k{k}__{cfgs('12')[int(r['config'])-1]}.log"
                    if solves(p)[0]["it"] != int(r["iterations"]):
                        bad.append(f"{hide(p)} {solves(p)[0]['it']} vs tsv {r['iterations']}")
    return bad


# ---- test vectors
@check("tvscan.table")
def _():
    bad = []
    for i, (s, nv) in enumerate([("S", 16), ("S", 24), ("S", 32), ("S", 48), ("B", 16), ("B", 24), ("B", 32), ("B", 48)]):
        t = cells("results_tables.tex", 210 + i)
        for j, vol in ((2, "24"), (3, "2448")):
            if nv == 32:
                ps = ([f"{AR}/rerunT/T_vol_{s}_24__{c}.log" for c in cfgs("24")] if vol == "24" else
                      [f"{AR}/rerunT/T_ms_{s}_m0.001_a0.6__{c}.log" for c in cfgs("2448")])
            else:
                ps = [f"{AR}/tvscanT/T_tv{vol}_{s}_nv{nv}__{c}.log" for c in cfgs(vol)]
            ps = [p for p in ps if has(p)]
            if not ps:
                c = "not run"
            else:
                sv = [solves(p)[0] for p in ps]
                nc = sum(not x["ok"] for x in sv)
                good = [x["it"] for x in sv if x["ok"]]
                c = (rng(good) if good else "") + (f", nc ({nc})" if nc and good else f"nc ({nc})" if nc else "")
            if norm(t[j]) != norm(c):
                bad.append(f"{s} {nv} {vol}: tex {t[j]} vs logs {c}")
    return bad


# ---- basis lifts
@check("basis.table")
def _():
    rows = tsv("basis_configs.tsv")
    bad = []
    for i, (lift, y) in enumerate([("st", "16"), ("sds", "16"), ("s", "16"), ("s", "32")]):
        t = cells("results.tex", 276 + i)
        comp = []
        for lat in ("12", "16"):
            for s in "SB":
                it = [r["iterations"] for r in rows if r["lattice"] == lat and r["setup"] == s and r["lifts"] == lift
                      and r["y_vectors"] == y]
                nc = sum("!" in x for x in it)
                good = [int(x) for x in it if "!" not in x]
                if nc == 2:
                    c = rng(good) + " [2]"
                else:
                    c = hu(med([int(x) if "!" not in x else 10 ** 9 for x in it])) + (f" [{nc}]" if nc else "")
                comp.append(c)
        for s in "SB":
            r = [r for r in rows if r["lattice"] == "2448" and r["setup"] == s and r["lifts"] == lift and r["y_vectors"] == y][0]
            comp.append(r["iterations"])
        su = [r["setup_s"] for s in "SB" for r in rows if r["lattice"] == "2448" and r["setup"] == s
              and r["lifts"] == lift and r["y_vectors"] == y]
        comp.append(", ".join(su))
        for j, c in zip(range(3, 10), comp):
            if norm(t[j]) != norm(c):
                bad.append(f"{lift} {y} col {j}: tex {t[j]} vs tsv {c}")
    for r in rows:
        p = re.search(r"(multiconf/\S+\.log)", r["source"])
        if p:
            p = MC + "/" + p.group(1)[len("multiconf/"):]
            s = solves(p)[0]
            if f"{s['it']}{'' if s['ok'] else '!'}" != r["iterations"]:
                bad.append(f"{hide(p)} {s['it']} vs tsv {r['iterations']}")
    return bad


@check("text.basis.setup")
def _():
    rows = tsv("basis_configs.tsv")
    v = []
    for s in "SB":
        a = [int(r["setup_s"]) for r in rows if r["lattice"] == "2448" and r["setup"] == s and r["lifts"] == "s" and r["y_vectors"] == "32"][0]
        b = [int(r["setup_s"]) for r in rows if r["lattice"] == "2448" and r["setup"] == s and r["lifts"] == "st"][0]
        v.append(a / b)
    return [] if rng(v, 1) == "1.2-1.7" else [f"setup ratio {rng(v, 1)}"]


# ---- Tikhonov
TK_ROWS = [("1", "NS12_S__v2", None), ("1a", None, "tk_v2_s2"), ("1b", None, "tk_v2_s5"), ("1c", None, "tk_v2_s6"),
           ("2", "NS12_S__000502", None), ("2a", None, "tk_502_s10"), ("3", "NS12_S__000602", None),
           ("3a", None, "tk_602_s3"), ("3b", None, "tk_602_s10"), ("4", "NS12_S__000702", "tk_702_s21")]
TAUS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]


def it_str(p):
    s = solves(p)[0]
    return str(s["it"]) if s["ok"] else "nc"


@check("tikhonov.table")
def _():
    bad = []
    for i, (row, ns, tk) in enumerate(TK_ROWS):
        t = cells("results.tex", 317 + i)
        comp = [it_str(f"{AR}/noshift16/{ns}.log") if ns else it_str(f"{AR}/tikhonov/{tk}_none.log")]
        for tau in TAUS:
            if ns and tk is None:
                p = f"{AR}/tikhonov2/{ns}_{tau}.log"
            else:
                pre = "" if tau in ("1e-3", "1e-2", "3e-2") else "e"
                p = f"{AR}/tikhonov/{tk}_{pre}{tau}.log"
            comp.append(it_str(p))
        for j, c in zip(range(1, 8), comp):
            if norm(t[j]) != c:
                bad.append(f"row {row} col {j}: tex {t[j]} vs log {c}")
    for i, (sm, pre) in enumerate([("16", "NS16_S"), ("12none", "NS12_S0")]):
        for k, c in enumerate(cfgs("12")):
            line = (329 if sm == "16" else 335) + k
            t = cells("results.tex", line)
            comp = [it_str(f"{AR}/noshift16/{pre}__{c}.log")] + [it_str(f"{AR}/tikhonov2/{pre}__{c}_{tau}.log") for tau in TAUS]
            for j, cc in zip(range(1, 8), comp):
                if norm(t[j]) != cc:
                    bad.append(f"{sm} config {k+1} col {j}: tex {t[j]} vs log {cc}")
    return bad


@check("text.tikhonov.seeds")
def _():
    v = [int(r["iterations"]) for r in tsv("tikhonov_seeds_12.tsv") if r["setup"] == "S"]
    inside = [x for x in v if 11 <= x <= 21]
    out = [x for x in v if not 11 <= x <= 21]
    return [] if (len(v), len(inside), out) == (40, 39, [46]) else [f"{len(v)} seeds, {len(inside)} in 11-21, others {out}"]


@check("text.tikhonov.tau")
def _():
    v = []
    for i in range(10):
        v += cells("results.tex", 317 + i)[2:8]
    for i in range(4):
        v += cells("results.tex", 329 + i)[2:8] + cells("results.tex", 335 + i)[2:8]
    n = [int(x) for x in v]
    out = [x for x in n if not 11 <= x <= 23]
    return [] if len(out) == 2 else [f"outside 11-23: {out}"]


# ---- singular values
@check("pvsv.table")
def _():
    bad = []
    sv = re.compile(r"P\^dag D_PV P: singular vector 0: sigma = ([\d.e+-]+)")
    lam = re.compile(r"^\s+0\s+\+([\d.e+-]+)", re.M)
    for line, coarse, fine in ((73, "pvsvD12_sm1", "fs_sm1_m1.0"), (74, "pvsvD12_sm0", "fs_sm0_m1.0"),
                               (75, "pvsvD16_sm1", "fs16_sm1_m1.0")):
        t = cells("results_tables.tex", line)[2:]
        for j, c in enumerate(["v2", "502", "602", "702"]):
            cc = "v2" if c == "v2" else "000" + c
            a = float(sv.search(log(f"{AR}/pvsvD/{coarse}__{cc}.log")).group(1))
            b = math.sqrt(float(lam.search(log(f"{MC}/finesigmaA1/{fine}__{c}.log")).group(1)))
            ta = norm(t[j]).split(" / ")
            nd = [len(x.split(".")[1]) if "." in x else 0 for x in ta]
            if (hu(a, nd[0]), hu(b, nd[1])) != (ta[0], ta[1]):
                bad.append(f"line {line} config {j+1}: tex {t[j]} vs logs {a:.4g} / {b:.4g}")
    return bad


@check("text.pvsv.ratio")
def _():
    r = []
    for line in (73, 74, 75):
        for c in cells("results_tables.tex", line)[2:]:
            a, b = [float(x) for x in norm(c).split(" / ")]
            r.append((line, b / a))
    st = [x for l, x in r if l != 74]
    no = [x for l, x in r if l == 74]
    return [] if rng(st) == "9-106" else [f"stout ratio {rng(st)}, no smearing {rng(no, 1)}"]


# ---- field of values, spectrum, pseudospectra
P_PART = re.compile(r"parts A~ (S|E|M): Re W in \[([-+\d.e]+), ([-+\d.e]+)\], \|I - A~ \w\|_2 = ([\d.e+-]+)")
P_FOVA = re.compile(r"^A~   : Re of the field of values in \[([-+\d.e]+),", re.M)


def parts(tag):
    t = log(f"{AR}/fovT2/{tag}.log")
    d = {k: (float(a), float(b), float(n)) for k, a, b, n in P_PART.findall(t)}
    d["A"] = float(P_FOVA.search(t).group(1))
    return d


def sci(x):
    e = math.floor(math.log10(abs(x)))
    return f"{hu(x / 10 ** e, 1)}e{e}" if e >= 3 else (hu(x) if x >= 10 else hu(x, 2) if x < 10 else hu(x, 1))


@check("fov.table")
def _():
    bad = []
    for line, pre in ((124, "parts12_S"), (125, "parts12_S0"), (126, "parts12_B")):
        P = [parts(f"{pre}__{c}") for c in ("v2", "502", "602", "702")]
        t = cells("results_tables.tex", line)
        comp = [">=" + hu(min(p["A"] for p in P), 3),
                rng([p["S"][2] for p in P], 2),
                None,
                f"{hu(min(p['M'][0] for p in P), 2)} to {hu(max(p['M'][0] for p in P), 2)}",
                rng([p["M"][2] for p in P], 1)]
        e = [p["E"][2] for p in P]
        comp[2] = (f"{hu(min(e)/1e3,1)}e3-{hu(max(e)/1e3,1)}e3" if min(e) > 1e3 else rng(e))
        for j, c in zip(range(1, 6), comp):
            if norm(t[j]).replace(" ", "") != norm(c).replace(" ", ""):
                bad.append(f"{pre} col {j}: tex {t[j]} vs logs {c}")
    for r in tsv("fov_12_configs.tsv"):
        pre = {"S": "parts12_S", "S0": "parts12_S0", "B": "parts12_B"}.get(r["setup"])
        if pre:
            c = {"1": "v2", "2": "502", "3": "602", "4": "702"}[r["config"]]
            if abs(parts(f"{pre}__{c}")["M"][0] - float(r["re_W_AM_min"])) > 1e-6:
                bad.append(f"fov_12_configs {r['setup']} {r['config']} differs from the log")
    return bad


@check("text.fov.regulator")
def _():
    rg = re.compile(r"regulator: coarse solves to (1e-20|1e-2):  x: Re x\^dag A~ E x \S+, \|\(I - A~ E\) x\| ([\d.e+-]+), "
                    r"Re x\^dag A~ M x ([-+\d.e]+)")
    out = []
    pos = tot = 0
    fac = []
    for p in sorted(glob.glob(f"{AR}/fovT2/parts12_*.log")):
        d = {k: (float(a), float(b)) for k, a, b in rg.findall(log(p))}
        if "1e-2" in d and "1e-20" in d:
            fac.append(d["1e-20"][0] / d["1e-2"][0])
            tot += 1
            pos += d["1e-2"][1] > 0
    out.append(f"{tot} setups: |(I-AE)x| exact/1e-2 = {min(fac):.1f}..{max(fac):.1f}; Re x^dag A M x > 0 with 1e-2 in {pos} of {tot}")
    return out


@check("text.adjoint")
def _():
    v = []
    for p in glob.glob(f"{AR}/fovT2/*.log") + glob.glob(f"{AR}/pseudoT/*.log"):
        m = re.search(r"relative difference ([\d.e+-]+)", log(p))
        if m:
            v.append(float(m.group(1)))
    return [f"adjoint check, relative difference {min(v):.1e}..{max(v):.1e} ({len(v)} runs)"]


P_ARN = re.compile(r"lambda_\d+ = ([-+\d.e]+) ([-+\d.e]+)i .*?kappa = 1/\|y\^dag x\| = ([\d.e+-]+)")
P_FAR = re.compile(r"largest \|lambda - \(\+1\.000 \+0\.000i\)\| = ([\d.e+-]+): lambda = ([-+\d.e]+) ([-+\d.e]+)i .*?kappa = 1/\|y\^dag x\| = ([\d.e+-]+)")


def far(tag):
    t = log(f"{AR}/fovT2/{tag}.log")
    return P_ARN.findall(t), P_FAR.findall(t)


@check("spectrum.table")
def _():
    out = []
    for line, tag in ((146, "far12_S_R__v2"), (147, "far12_S_L__v2"), (148, "far12_B_R__v2"), (149, "far12_B_L__v2")):
        arn, fr = far(tag)
        t = cells("results_tables.tex", line)
        mn = hu(min(float(a) for a, b, k in arn), 3)
        ks = [float(k) for a, b, k in arn]
        out.append(f"{tag}: tex {t[1]} | {t[2]} | {t[3]} | {t[4]} ;; logs min Re {mn}, kappa {rng(ks,1)} ({len(ks)}), "
                   f"far {[(round(float(d),3), a, b, k) for d, a, b, k in fr[:3]]}")
    for s in "SB":
        mins = []
        for c in ("v2", "502", "602", "702"):
            arn, fr = far(f"far16_{s}__{c}")
            if arn:
                mins.append(min(float(a) for a, b, k in arn))
        out.append(f"16^4 {s}: min Re {rng(mins,3)} ({len(mins)} configurations)")
    return out


@check("spectrum.tsv")
def _():
    bad = []
    for s, tag in (("S12", "far12_S_R__v2"), ("B12", "far12_B_R__v2")):
        arn, fr = far(tag)
        rows = [r for r in tsv(f"spectrum_{s}.tsv") if r["kind"] == "smallest_re"]
        if [float(r["re"]) for r in rows] != [float(a) for a, b, k in arn]:
            bad.append(f"spectrum_{s}.tsv smallest_re differ from {tag}")
    return bad


@check("pseudo.table")
def _():
    out = []
    pz = re.compile(r"pseudo z = ([+-][\d.]+) ([+-][\d.]+)i: sigma_min\(z - A~ M\) = (\S+)")
    for s, c, r, n, line in (("S", 1.0, 0.6, 160, 171), ("B", 1.4, 0.93, 96, 172)):
        pts = [(float(a), float(b), float(x)) for a, b, x in pz.findall(log(f"{AR}/pseudoT/ps12_{s}__v2.log"))]
        circ = [x for a, b, x in pts if abs(math.hypot(a - c, b) - r) < 2e-3]
        h = 2 * math.pi * r / n
        mn = min(circ)
        t = cells("results_tables.tex", line)
        out.append(f"{s}: {len(circ)} points on the circle, min sigma {mn:.4f}, eps >= {mn - h/2:.4f}; tex {t[3]}")
    return out


@check("pseudo.tsv")
def _():
    bad = []
    pz = re.compile(r"pseudo z = ([+-][\d.]+) ([+-][\d.]+)i: sigma_min\(z - A~ M\) = (\S+)")
    for s in ("S", "B"):
        pts = pz.findall(log(f"{AR}/pseudoT/ps12_{s}__v2.log"))
        rows = tsv(f"pseudo_{s}12.tsv")
        if len(rows) != len(pts) or any(abs(float(r["sigma_min"]) - float(p[2])) > 1e-9 for r, p in zip(rows, pts)):
            bad.append(f"pseudo_{s}12.tsv differs from the log ({len(rows)} rows, {len(pts)} points)")
    return bad


# ---- history
@check("history.tsv")
def _():
    bad = []
    st = re.compile(r"seed 91 GCR step\s+\d+: \|r\|\^2/\|b\|\^2 (\S+)")
    tag = {"S": "stout", "B": "m18"}
    for r in tsv("gcr_steps_24_configs.tsv"):
        c = cfgs("2448")[int(r["config"]) - 1]
        p = f"{AR}/rerunT/T_hist_mg_{tag[r['setup']]}_m{r['mass']}_a{r['alpha']}__{c}.log"
        h = st.findall(log(p))
        if [float(x) for x in h] != [float(x) for x in r["r2_over_b2"].split()]:
            bad.append(f"{hide(p)}: {len(h)} steps vs tsv {len(r['r2_over_b2'].split())}")
    return bad


@check("text.history.factor")
def _():
    r = [x for x in tsv("gcr_steps_24.tsv") if x["setup"] == "S" and x["mass"] == "0.001" and x["alpha"] == "1.0"][0]
    v = [float(x) for x in r["r2_over_b2"].split()]
    f = [math.sqrt(v[i + 1] / v[i]) for i in range(len(v) - 1)]
    return [] if hu(med(f), 2) == "0.42" else [f"median factor per step of |r|: {med(f):.3f}"]


# ---- D-dagger
P_DT = re.compile(r"^(random b|sweep residual r)\s+(D by GCR\(2\) with M_D|GCR\((\d)\) on D† with Γ5 M_D Γ5, (\d) inner|"
                  r"dual cycle on Ã†, GCR\((\d)\)):\s+(\d+) its( \(not converged\))?", re.M)


def dagT(tag):
    d = {}
    for m in P_DT.finditer(log(f"{AR}/daggerT/{tag}.log")):
        rhs = "rand" if m.group(1) == "random b" else "res"
        if m.group(2).startswith("D by"):
            key = "D"
        elif m.group(2).startswith("GCR"):
            key = f"g{m.group(3)}_{m.group(4)}"
        else:
            key = f"dual{m.group(5)}"
        d[(key, rhs)] = (int(m.group(6)), m.group(7) is None)
    return d


def stall_cell(v, stall_at=3, halves=False):
    ok = [i for i, c in v if c]
    nc = len(v) - len(ok)
    if nc >= stall_at:
        return "stall"
    if nc == 2:
        return rng(ok) + " [2]"
    x = med([i if c else 10 ** 9 for i, c in v])
    s = hu(x, 1) if halves and x != int(x) else hu(x)  # Table b5 prints the halves, Table c4 rounds them up
    return s + (f" [{nc}]" if nc else "")


@check("c4.table")
def _():
    bad = []
    keys = [("D", 50), ("g2_1", 51), ("g8_1", 52), ("g2_2", 53), ("g8_2", 54)]
    for key, line in keys:
        t = cells("results_tables.tex", line)
        j = 1
        for lat in ("16", "12"):
            for s in "SB":
                for rhs in ("rand", "res"):
                    v = [dagT(f"T_dag{lat}_{s}__{c}")[(key, rhs)] for c in ("v2", "502", "602", "702")]
                    c = stall_cell(v)
                    if norm(t[j]) != c:
                        bad.append(f"{key} {lat} {s} {rhs}: tex {t[j]} vs logs {c}")
                    j += 1
    return bad


@check("text.dual")
def _():
    out = []
    for lat in ("16", "12"):
        for s in "SB":
            v = [dagT(f"T_dag{lat}_{s}__{c}")[("dual2", "res")] for c in ("v2", "502", "602", "702")]
            out.append(f"{lat} {s} dual GCR(2) res: {v}")
    return out


P_B5 = re.compile(r"(random b|sweep residual r): (D  by GCR\(2\) with M|D† by GCR\(2\) with Γ5 M Γ5|D† by D, Γ5 B5 D⁻¹ B5⁻¹ Γ5):\s+"
                  r"(\d+) its( \(not converged\))?  \S+ \S+ \S+ \S+\s+([\d.]+) s")


def b5(tag):
    d = {}
    for m in P_B5.finditer(log(f"{AR}/daggerB5/{tag}.log")):
        rhs = "rand" if m.group(1) == "random b" else "res"
        key = "D" if m.group(2).startswith("D  by") else ("G" if "Γ5 M Γ5" in m.group(2) else "B5")
        d[(key, rhs)] = (int(m.group(3)), m.group(4) is None, float(m.group(5)))
    return d


@check("b5.table")
def _():
    bad = []
    line = 1528
    for lat in ("12", "16", "24", "2448"):
        for s in "SB":
            t = cells("MDWF_MULTIGRID_NOTE.tex", line)
            line += 1
            j = 2
            for key in ("D", "G", "B5"):
                for rhs in ("rand", "res"):
                    v = [b5(f"B5_{lat}_{s}__{c}")[(key, rhs)][:2] for c in cfgs(lat)]
                    c = stall_cell(v, stall_at=4, halves=True)
                    if norm(t[j]) != c:
                        bad.append(f"{lat} {s} {key} {rhs}: tex {t[j]} vs logs {c}")
                    j += 1
    return bad


@check("text.b5")
def _():
    out = []
    diff, faster, s12 = [], True, []
    for lat in ("12", "16", "24", "2448"):
        for s in "SB":
            for c in cfgs(lat):
                d = b5(f"B5_{lat}_{s}__{c}")
                for rhs in ("rand", "res"):
                    diff.append(d[("B5", rhs)][0] - d[("D", rhs)][0])
                    if not d[("B5", rhs)][2] <= d[("G", rhs)][2]:
                        faster = False
                if lat == "12" and s == "S":
                    s12.append(d[("B5", "res")][0])
    out.append(f"through D minus D: {min(diff)}..{max(diff)}; through D faster than Gamma5 M Gamma5 in every run: {faster}; "
               f"12^4 S res through D: {rng(s12)}")
    return out


@check("text.hmc24")
def _():
    v = []
    for c in cfgs("24"):
        m = re.search(r"Action_F_1: 5 solves, (\d+) iterations", log(f"{AR}/hmcT/hmc24_{c}.log"))
        v.append(int(m.group(1)))
    per = [x / 5 for x in v]
    return [] if (rng(v), rng(per)) == ("217-247", "43-49") else [f"{v}"]


@check("text.dagger.tworounds")
def _():
    out = []
    pat = re.compile(r"D x0 = w0 (\d+) its \(converged (\w+)\)")
    for lat in ("24", "2448"):
        r = []
        for c in cfgs(lat):
            c = c if c == "v2" else "000" + c[-3:]
            got = []
            for p in sorted(glob.glob(f"{MC}/daggerA/A{lat}_S__{c}*.log")):
                m = pat.search(log(p))
                if m:
                    got.append(f"{os.path.basename(p)} {m.group(1)} {m.group(2)}")
            r.append("; ".join(got) or "no D x0 line")
        out.append(f"{lat} S two rounds: " + " | ".join(r))
    return out


@check("text.g5check")
def _():
    t = log(f"{MC}/g5check/g5check_12_S.log")
    return [l for l in t.split("\n") if l.startswith("b ")]


# ---- configurations
@check("configs.cfg1")
def _():
    bad = []
    for lat, f, line in (("12", "q12", 1875), ("16", "q16", 1879), ("24", "q24", 1883), ("2448", "q2448", 1887)):
        t = cells("MDWF_MULTIGRID_NOTE.tex", line)
        m = re.search(r"_v2\.bin\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([-+\d.]+)\s+([-+\d.]+)", log(f"{G}/topo/{f}.log"))
        comp = [hu(float(m.group(1)), 6), hu(float(m.group(2)), 2), "%+.2f" % float(m.group(3)), "%+.2f" % float(m.group(4))]
        for j, c in zip((2, 3, 4, 5), comp):
            if norm(t[j]) != c:
                bad.append(f"{lat} config 1 col {j}: tex {t[j]} vs topo log {c}")
    return bad


@check("configs.plaq")
def _():
    """The plaquettes of configurations 2-4 against the plaquette printed by the run logs."""
    bad = []
    pl = re.compile(r"plaquette ([\d.]+) →")
    line = {"12": 1875, "16": 1879, "24": 1883, "2448": 1887}
    for lat, tag in (("12", "T_vol_S_12"), ("16", "T_vol_S_16"), ("24", "T_vol_S_24"), ("2448", "T_ms_S_m0.001_a0.6")):
        for k, c in enumerate(cfgs(lat)):
            t = cells("MDWF_MULTIGRID_NOTE.tex", line[lat] + k)
            v = float(pl.search(log(f"{AR}/rerunT/{tag}__{c}.log")).group(1))
            if norm(t[2]) != hu(v, 6):
                bad.append(f"{lat} config {k+1}: tex {t[2]} vs log {v}")
    return bad


@check("configs.q2to4")
def _():
    t = open(sorted(glob.glob("/home/chen/red_gpu/*/gauge_field/ENSEMBLES.md"))[0]).read()
    return [f"t0 and Q of configurations 2-4: only in gauge_field/ENSEMBLES.md ({len(re.findall('more/conf_quench', t))} rows); no flow log on this host"]


@check("text.configs.Qdiff")
def _():
    d = []
    for line in range(1875, 1891):
        t = cells("MDWF_MULTIGRID_NOTE.tex", line)
        d.append(abs(float(t[5]) - float(t[4])))
    return [f"largest |Q_imp - Q_clover| in Table configs: {max(d):.2f} (text: 'up to 1')"]


# ---- CG forms
@check("cgform.claims")
def _():
    rows = tsv("cgne_form.tsv")
    it = {(r["volume"], r["setup"], r["b"], r["c"], r["alpha"], r["operator"]): int(r["iterations"]) for r in rows}
    out = []
    sham = [it[k] / it[k[:5] + ("S_dag_S",)] for k in it if k[2:5] == ("1.0", "0.0", "1.0") and k[5] == "K_dag_K"]
    out.append(f"Shamir (b,c)=(1,0) alpha 1: K^dag K / S^dag S iterations {rng(sham, 2)} ({len(sham)} cases)")
    for a in ("1.0", "0.6"):
        for s in "SB":
            r = sorted((float(k[3]), k[0], it[k] / it[k[:5] + ("S_dag_S",)]) for k in it
                       if k[1] == s and k[4] == a and k[5] == "K_dag_K" and k[2:4] != ("1.0", "0.0"))
            out.append(f"alpha {a} {s} (b,c) scan, K/S: " + " ".join(f"c={c:g}/{v}:{x:.2f}" for c, v, x in r))
    return out


@check("params.setups")
def _():
    bad = []
    for s in "SB":
        y = open(f"{AR}/rerunT/T_ms_{s}_m0.001_a0.6__v2.yaml").read()
        want = {"domain_wall_height": "1.0" if s == "S" else "1.8", "number_of_steps": "8" if s == "S" else "12",
                "interval_ratio": "10" if s == "S" else "30", "coefficient_b": "1.5", "coefficient_c": "0.5",
                "extent_of_5th_dimension": "8", "quark_mass_PauliVillars": "1.0", "neff_alpha": "0.6",
                "block_size": "[4, 4, 4, 4]", "number_of_test_vectors": "32", "filter_number_of_vectors": "16",
                "filter_number_of_rounds": "2", "bootstrap_number_of_steps": "4", "bootstrap_number_of_rounds": "auto",
                "tikhonov_epsilon": "1.0e-3", "number_of_kept_directions": "2", "seed": "21",
                "boundary_condition": "[1, 1, 1, -1]"}
        for k, v in want.items():
            m = re.search(rf"^\s*{k}\s*:\s*(.+?)\s*(#.*)?$", y, re.M)
            if not m or m.group(1) != v:
                bad.append(f"{s} {k}: yaml {m.group(1) if m else None} vs {v}")
        if ("Stout" in y) != (s == "S"):
            bad.append(f"{s}: smearing")
    seeds = set()
    for p in glob.glob(f"{AR}/rerunT/*.yaml") + glob.glob(f"{AR}/*/[A-Z]*.yaml"):
        m = re.search(r"^\s*seed\s*:\s*(\d+)", open(p).read(), re.M)
        seeds.add(m.group(1) if m else "none")
    bad.append(f"setup seeds in the autoR yamls: {sorted(seeds)}")
    return bad


@check("text.borici")
def _():
    out = []
    for f in sorted(glob.glob(f"{G}/therm16/regcheck_20261106/borici/bor24_*.log")):
        s = solves(f)
        out.append(f"{os.path.basename(f)}: {s[0]['it']}{'' if s[0]['ok'] else '!'} ({len(rounds(f))} rounds)")
    return out


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "manifest.tsv")), delimiter="\t"))
    count = {}
    for r in rows:
        status, notes = "NOCHECK", []
        # the printed value still at file:line
        f, ln = r["file:line"].split(":")
        lines = ln.split("-")
        lo, hi = int(lines[0]), int(lines[-1])
        text = norm(" ".join(tex_lines(f)[lo - 1:hi]))
        miss = [tok for tok in re.findall(r"[-+]?\d+(?:\.\d+)?", norm(r["value(s)"])) if tok.lstrip("+-") not in text]
        tex_ok = not miss
        fn = CHECKS.get(r["id"])
        if fn is not None and MC is not None:
            try:
                res = fn()
            except Exception as e:  # a missing log is a finding, not a crash
                res = [f"ERROR {type(e).__name__}: {hide(e)}"]
            mism = [x for x in res if not x.startswith("INFO:")]
            if r["id"] in INFO_ONLY:
                status, notes = "INFO", res
            else:
                status, notes = ("MISMATCH", res) if mism else ("OK", res)
        if not tex_ok:
            status = "TEX-CHANGED" if status in ("OK", "NOCHECK") else status
            notes = [f"value tokens not found at {r['file:line']}: {miss}"] + notes
        count[status] = count.get(status, 0) + 1
        if status != "OK" or VERBOSE:
            print(f"{status:11s} {r['id']:24s} {r['file:line']:28s} {r['quantity']}")
            for n in notes:
                print(f"{'':12s}{hide(n)}")
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(count.items())) + f" ({len(rows)} rows)")


# Checks that print what the logs give instead of an automatic comparison (claims, loose numbers).
INFO_ONLY = {"text.bandwidth", "ps.correlator", "text.volume.growth", "text.volume.order", "sens.cap", "text.probe",
             "text.theta", "text.fov.regulator", "text.adjoint", "spectrum.table", "pseudo.table", "text.dual",
             "text.b5", "text.dagger.tworounds", "text.g5check", "configs.q2to4", "text.configs.Qdiff", "cgform.claims",
             "params.setups", "text.borici"}

if __name__ == "__main__":
    main()
