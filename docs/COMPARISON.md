# Numerical comparisons

One entry per comparison against a reference code (FLOW.md §D1). A feature gets its own
regression baseline only after an entry here shows it agreeing with the reference.

## 1. Plaquette of the Bridge++ test configuration

| | |
|---|---|
| Date | 2026-10-06 |
| What | Average plaquette of `conf_04040408.txt` (4×4×4×8, plain-text format) |
| Reference | Bridge++ test `Gauge_lex`, `expected_result` in `tests/test_Gauge_lex.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | md5 `a8cd38fbb5711577001d85bd12d29422`, 36864 values |
| RutticeQCD | stage 0; `ruttice-validation` test `bridge_gauge_lex_plaquette` |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process, periodic neighbours |

| | Value |
|---|---|
| Bridge++ | 0.625129946974942 |
| RutticeQCD | 0.625129946974942 |
| Tolerance | 1e-13 absolute |
| Result | pass |

Also checked on the same file: every link is unitary to 1e-12.

What this validates: the text reader (site outermost, then direction, then 18 reals
row-major), lexicographic site numbering with x fastest, the nearest-neighbour stencil,
3×3 complex multiplication and adjoint, and the fixed-order reduction.

What it does not validate: anything with a boundary phase, any fermion code, layouts other
than `Aos`, more than one process.

## 2. Wilson operator applied to a constant spinor

| | |
|---|---|
| Date | 2026-10-06 |
| What | 2-norm of D b, where every real of b is 1 and D is applied once |
| Reference | Bridge++ test `Mult_Wilson`, `expected_result` in `tests/test_Mult_Wilson.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | κ = 0.12, Dirac representation, antiperiodic in all four directions |
| RutticeQCD | stage 1; `ruttice-validation` test `bridge_mult_wilson`; fused kernel, variant `D` |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |

| | Value |
|---|---|
| Bridge++ | 1.23160709257508e+02 |
| RutticeQCD | 1.23160709257508e+02 |
| Tolerance | 1e-13 relative (the Bridge++ test itself uses 1e-14) |
| Result | pass |

What this validates: the gamma matrices and the spinor component order, the sign and
orientation of both hopping terms, the link and its adjoint, the antiperiodic phase on
hops across the boundary, and the normalisation D = 1 - κ (hops).

What it does not validate: D† and γ5 D against a reference (they are checked against D by
adjointness and by γ5 hermiticity only), other boundary types, other gamma representations,
`f32`, layouts other than `Aos`, more than one process.

Independent checks that passed alongside (no reference needed): free-field plane wave
against the analytic result, gauge covariance, <χ, Dψ> = <D†χ, ψ>, the fused sweep against
the sum of the eight hopping terms, and several spinors stored together against each one
applied separately.

## 3. Iteration count of CG on the Wilson normal operator

| | |
|---|---|
| Date | 2026-10-06 |
| What | round(Nconv / 10), Nconv = 1 + number of CG steps |
| Reference | Bridge++ test `Solver_Wilson_CGNE`, `expected_result` in `tests/test_Solver_Wilson_CGNE.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | κ = 0.12, antiperiodic in all directions; point source at the origin, spin 0, colour 0; CG on D D† starting from x = b; stop at \|r\|²/\|b\|² < 1e-28; true residual and restart every 100 steps |
| RutticeQCD | stage 1; `ruttice-validation` test `bridge_solver_wilson_cgne_iteration_count`; `Cg` on `Product(D, Adjoint(D))` |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |

| | Value |
|---|---|
| Bridge++ | 11 |
| RutticeQCD | 11 (Nconv = 109, one restart; final \|D x − b\|²/\|b\|² = 7.1e-29) |
| Tolerance | exact match of the rounded value |
| Result | pass |

This is a coarse comparison: any Nconv from 105 to 114 gives the same rounded value, so it
shows the solver follows the same procedure, not that it reproduces it step for step.
The Bridge++ test fixes the gauge to Landau gauge first; that step was skipped here because
the CG residual history of a point source is gauge covariant. This was not checked by
running both.

Not compared: BiCGStab (the Bridge++ variants count iterations differently and have extra
parameters), CGNR, any solution vector element by element.

## 4. Even-odd preconditioned Wilson operator applied to a constant spinor

| | |
|---|---|
| Date | 2026-10-06 |
| What | 2-norm of (1 - M_eo M_oe) b on the even sites, every real of b equal to 1 |
| Reference | Bridge++ test `Mult_Wilson_eo`, `expected_result` in `tests/test_Mult_Wilson_eo.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | κ = 0.12, Dirac representation, antiperiodic in all four directions |
| RutticeQCD | stage 1; `ruttice-validation` test `bridge_mult_wilson_eo`; `Schur` on `Parity::Even` |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |

| | Value |
|---|---|
| Bridge++ | 8.07908371651806e+01 |
| RutticeQCD | 8.07908371651807e+01 |
| Tolerance | 1e-13 relative (the Bridge++ test itself uses 1e-14) |
| Result | pass |

The same quantity on the odd sites is 8.13216769729418e+01, so the agreement also fixes
which parity Bridge++ calls even: x + y + z + t even.

What this validates: the hopping blocks between the two parities, the half-lattice
stencil, and the Schur complement for a unit diagonal.

What it does not validate: the ordering of sites within a parity (a norm does not see
it), the right-hand-side preparation and the reconstruction of the other parity (those
are checked against the full operator only), a non-trivial diagonal block.

## 5. Pion correlator from a point source

| | |
|---|---|
| Date | 2026-10-06 |
| What | Real part of the pseudoscalar-pseudoscalar correlator at t = 0 |
| Reference | Bridge++ test `Spectrum_Wilson_Hadron2ptFunction`, `expected_result` in the yaml file of that name |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | κ = 0.12, antiperiodic in all four directions, point source at the origin, all 12 spin-colour components |
| RutticeQCD | stage 1; `ruttice-validation` test `bridge_spectrum_wilson_pion`; 12 right-hand sides solved together by even-odd preconditioned CGNE to \|r\|²/\|b\|² < 1e-28, then `pion_correlator` |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |

| | Value |
|---|---|
| Bridge++ | 14.2250394502083 |
| RutticeQCD | 14.2250394502083 |
| Tolerance | 1e-12 relative |
| Result | pass |

Worst full-system residual \|D S − b\|² over the 12 columns: 8.8e-29.

The Bridge++ test fixes the gauge to Landau gauge before solving; that step was skipped.
The pseudoscalar correlator from a point source is gauge invariant, and the agreement to
all printed digits is consistent with that.

What this validates: the point source, the solve for several right-hand sides at once,
the even-odd solve end to end (preparation, Schur solve, reconstruction), and the
time-slice reduction.

What it does not validate: the other time slices against Bridge++ (only t = 0 is in the
yaml file), any channel other than the pseudoscalar. `meson_correlator` with other gamma
matrices has only been checked to reduce to the pion case for γ5; its sign and
normalisation conventions for other channels are unverified.

