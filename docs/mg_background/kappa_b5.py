#!/usr/bin/env python3
# kappa(B5) of Sec. IV.A: B5 = b - cY on one chirality, Y the hop in s of Eq. (Y) with the corner m.
# Prints kappa for (b,c) = (1,1) and (1.5,0.5), Ls = 8 and 16, m = 0.001 (the text: 11, 21, 1.9, 2.0).
import numpy as np

def y_minus(ls, m):
    y = np.zeros((ls, ls))
    for s in range(ls):
        if s + 1 < ls:
            y[s, s + 1] = -1.0
        else:
            y[s, 0] = m
    return y

for b, c in [(1.0, 1.0), (1.5, 0.5)]:
    for ls in (8, 16):
        for y in (y_minus(ls, 0.001), y_minus(ls, 0.001).T):
            sv = np.linalg.svd(b * np.eye(ls) - c * y, compute_uv=False)
            print(f"b={b} c={c} Ls={ls}: kappa(B5) = {sv[0] / sv[-1]:.2f}")
