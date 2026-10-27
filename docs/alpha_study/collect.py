#!/usr/bin/env python3
"""Collect the logs of the alpha study (12^4, A30) into data/*.tsv.

python3 -I collect.py RUNS, with RUNS the run directory holding alpha12, alpha12_mg, alpha12_hist,
alpha12_fovM and alpha12_lowdefect.
"""
import argparse
import re
from pathlib import Path

SETS = [(s, m, a) for s in "SB" for m in ("0.001", "0.1") for a in ("0.6", "1.0")]
NUM = r"([+-]?[\d.]+(?:e[+-]?\d+)?)"


def write(path: Path, header: list[str], rows: list[list]) -> None:
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def iterations(runs: Path) -> list[list]:
    rows = []
    for s, m, a in SETS:
        t = (runs / "alpha12_hist" / f"{s}_m{m}_a{a}.log").read_text()
        n = int(re.search(r"GCR (\d+) iterations", t)[1])
        hist = [float(x) for x in re.findall(r"GCR step +\d+: \|r\|\^2/\|b\|\^2 " + NUM, t)]
        rows.append([s, m, a, n, ",".join(f"{x:.3e}" for x in hist)])
    return rows


def spectrum_a(runs: Path) -> list[list]:
    rows = []
    for s, m, a in SETS:
        d = runs / "alpha12"
        asc = (d / f"{s}_m{m}_a{a}_ascending.log").read_text()
        des = (d / f"{s}_m{m}_a{a}_abs_descending.log").read_text()
        pairs = re.findall(r"^\s+(\d+)\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s*$", asc, re.M)
        top = re.findall(r"^\s+(\d+)\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s*$", des, re.M)
        lo, hi48 = complex(float(pairs[0][1]), float(pairs[0][2])), complex(float(pairs[-1][1]), float(pairs[-1][2]))
        tp = complex(float(top[0][1]), float(top[0][2]))
        rows.append([s, m, a, f"{abs(lo):.4e}", f"{abs(hi48):.4e}", f"{abs(tp):.4e}"])
    return rows


def spectrum_am(runs: Path) -> list[list]:
    rows = []
    for s, m, a in SETS:
        t = (runs / "alpha12_fovM" / f"{s}_m{m}_a{a}.log").read_text()
        lam = re.findall(r"lambda_\d+ = " + NUM + " " + NUM + r"i .*?kappa = 1/\|y\^dag x\| = " + NUM, t)
        big = max(abs(complex(float(x), float(y))) for x, y in re.findall(r"largest \|lambda\|: " + NUM + " " + NUM + "i", t))
        ns = re.search(r"\|I - A~ S\|_2 = " + NUM, t)[1]
        ne = re.search(r"\|I - A~ E\|_2 = " + NUM, t)[1]
        rows.append([s, m, a, lam[0][0], lam[0][2], lam[1][0], f"{big:.4e}", ns, ne])
    return rows


def low_defect(runs: Path) -> list[list]:
    pat = re.compile(r"lowdefect coarse (\S+): lambda_(\d+) = " + NUM + " " + NUM + r"i  \|A x - l x\| " + NUM
                     + r"  \|\(I - A~ E\) x\| " + NUM + "  min_c " + NUM + r"  \|\(I - A~ M\) x\| " + NUM + "  min_c " + NUM)
    rows = []
    for s, m, a in SETS:
        for r in pat.findall((runs / "alpha12_lowdefect" / f"{s}_m{m}_a{a}.log").read_text()):
            rows.append([s, m, a, *r])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    runs = ap.parse_args().runs
    out = Path(__file__).resolve().parent / "data"
    write(out / "iterations_12.tsv", ["setup", "mass", "alpha", "iterations", "history_r2"], iterations(runs))
    write(out / "spectrum_A_12.tsv", ["setup", "mass", "alpha", "abs_l0", "abs_l47", "abs_lmax"], spectrum_a(runs))
    write(out / "spectrum_AM_12.tsv", ["setup", "mass", "alpha", "re_l0", "kappa_l0", "re_l1", "abs_lmax", "norm_I_AS", "norm_I_AE"], spectrum_am(runs))
    write(out / "lowdefect_12.tsv", ["setup", "mass", "alpha", "coarse_tol", "k", "re_l", "im_l", "eig_res", "I_AE", "I_AE_minc", "I_AM", "I_AM_minc"], low_defect(runs))


if __name__ == "__main__":
    main()
