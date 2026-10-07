# Numerical comparisons

One entry per comparison against a reference code. A feature gets its own
regression baseline only after an entry here shows it agreeing with the reference.

## 1. Plaquette of the Bridge++ test configuration

| | |
|---|---|
| Date | 2026-10-06 |
| What | Average plaquette of `conf_04040408.txt` (4×4×4×8, plain-text format) |
| Reference | Bridge++ test `Gauge_lex`, `expected_result` in `tests/test_Gauge_lex.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | md5 `a8cd38fbb5711577001d85bd12d29422`, 36864 values |
| RutticeQCD | test `bridge_gauge_lex_plaquette` |
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
| RutticeQCD | test `bridge_mult_wilson`; fused kernel, variant `D` |
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
| RutticeQCD | test `bridge_solver_wilson_cgne_iteration_count`; `Cg` on `Product(D, Adjoint(D))` |
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
| RutticeQCD | test `bridge_mult_wilson_eo`; `Schur` on `Parity::Even` |
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
| RutticeQCD | test `bridge_spectrum_wilson_pion`; 12 right-hand sides solved together by even-odd preconditioned CGNE to \|r\|²/\|b\|² < 1e-28, then `pion_correlator` |
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

The yaml file holds only this one number. The other time slices and the other twelve
channels are compared in entry 8.

## 6. Clover operator applied to a constant spinor

| | |
|---|---|
| Date | 2026-10-06 |
| What | 2-norm of D b, every real of b equal to 1, D applied once |
| Reference | Bridge++ test `Mult_Clover`, `expected_result` in `tests/test_Mult_Clover.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | kappa = 0.12, c_SW = 1, Dirac representation, periodic in space, antiperiodic in time |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |
| RutticeQCD | test `bridge_mult_clover`; `Clover`, variant `D` |

| | Value |
|---|---|
| Bridge++ | 1.23817935905285e+02 |
| RutticeQCD | 1.23817935905285e+02 |
| Tolerance | 1e-13 relative (the Bridge++ test itself uses 1e-14) |
| Result | pass |

What this validates: the clover-leaf field strength (orientation of the four leaves, the
1/8, no trace removal), the sign and normalisation of the clover term, the spin matrices
that multiply it, and a boundary that differs between directions.

What it does not validate against Bridge++: c_SW other than 1, other gamma
representations. For c_SW = 1.7 the stored diagonal block is checked against the defining
formula applied directly (test `diagonal_block_matches_the_direct_formula`); together with
this entry, which fixes the sign and normalisation of that formula at c_SW = 1, that covers
other values without a second reference number.

## 7. Even-odd preconditioned clover operator applied to a constant spinor

| | |
|---|---|
| Date | 2026-10-06 |
| What | 2-norm of (1 - T_ee^-1 M_eo T_oo^-1 M_oe) b on the even sites, every real of b equal to 1 |
| Reference | Bridge++ test `Mult_Clover_eo`, `expected_result` in `tests/test_Mult_Clover_eo.yaml` |
| Reference code | Bridge++ 2.0 series working tree at git `5540a26` with local modifications; the value is the one shipped in the yaml file, not recomputed |
| Input file | `conf_04040408.txt`, md5 `a8cd38fbb5711577001d85bd12d29422` |
| Parameters | kappa = 0.12, c_SW = 1, Dirac representation, periodic in space, antiperiodic in time |
| Toolchain | rustc 1.99.0, x86-64 Linux (WSL2) |
| Settings | `f64`, `Aos`, single process |
| RutticeQCD | test `bridge_mult_clover_eo`; `Schur` on `Parity::Even` followed by the inverse diagonal block |

| | Value |
|---|---|
| Bridge++ | 8.10390521901559e+01 |
| RutticeQCD | 8.10390521901559e+01 |
| Tolerance | 1e-13 relative (the Bridge++ test itself uses 1e-14) |
| Result | pass |

Bridge++ normalises the preconditioned operator by the inverse diagonal block; the Schur
complement here is T_ee - M_eo T_oo^-1 M_oe, so one more multiplication by T_ee^-1 is
needed to compare.

What this validates: the inverse of the 12 x 12 diagonal block on both parities, and the
generic Schur complement with a non-trivial diagonal.

## Reference runs made for this project

Entries 1 to 7 use numbers shipped with Bridge++. From here on the numbers come from
running the Bridge++ test program ourselves (CPU build of the same source tree, git
`5540a26` with local modifications, one process, one thread, 2026-10-06), sometimes with a
parameter changed so that one feature is exercised at a time. The values are stored in
`tests/data/` with their provenance in the file headers; nothing from Bridge++
itself is in this repository.

## 8. All meson channels and time slices, Wilson, point source

| | |
|---|---|
| Date | 2026-10-06 |
| What | 13 channels (PS, V1-3, A4-PS, PS-A4, S, γ5γ1-3, σ12, σ23, σ31) on 8 time slices: 104 complex numbers |
| Reference | output of the unmodified Bridge++ test `Spectrum.Wilson.Hadron2ptFunction` |
| Parameters | κ = 0.12, antiperiodic in all directions, point source at the origin; Bridge++ fixes the gauge to Landau gauge first, RutticeQCD does not |
| RutticeQCD | test `wilson_meson_channels`; `meson_correlator` |
| Precision of the reference | 13 significant digits |

| | |
|---|---|
| Largest difference | 4.3e-12 |
| Tolerance | 2.9e-11 absolute (2e-12 of the largest value) |
| Result | pass, all 104 |

What this validates: the sign and normalisation conventions of `meson_correlator` for
every channel the Bridge++ spectrum test prints, including the composite gamma matrices.

## 9. Clover, two flavours, exponential source

| | |
|---|---|
| Date | 2026-10-06 |
| What | the same 13 channels and 8 time slices for the flavour pairs (1,1), (2,2) and (1,2): 312 complex numbers |
| Reference | output of the Bridge++ test `Spectrum.Clover.Hadron2ptFunction` with `gauge_fixing_type` changed from `Coulomb` to `None` |
| Parameters | c_SW = 1, periodic in space, antiperiodic in time; exponential source at the origin with power 2. Flavour 1: κ = 0.12, slope 0.25. Flavour 2: κ = 0.115, slope 0.5 |
| RutticeQCD | test `clover_two_flavours_exponential_source`; `exponential_source`, even-odd CGNE, `meson_correlator` with two propagators |
| Precision of the reference | 13 significant digits |

| | |
|---|---|
| Largest difference | 4.9e-12 |
| Tolerance | 3.1e-11 absolute |
| Result | pass, all 312 |

What this validates: the exponential source, clover solves for two hopping parameters,
and correlators built from two different propagators.

What it does not validate: gauge fixing; the unmodified test fixes the configuration to
Coulomb gauge first (entry 11).

## 10. Clover operator with another coefficient

| | |
|---|---|
| Date | 2026-10-06 |
| What | 2-norm of D b, every real of b equal to 1 |
| Reference | the Bridge++ test `Mult.Clover` with `clover_coefficient` changed from 1.0 to 1.7 |
| Parameters | κ = 0.12, c_SW = 1.7, periodic in space, antiperiodic in time |
| RutticeQCD | test `clover_mult_with_another_coefficient` |

| | Value |
|---|---|
| Bridge++ | 1.24728260729018e+02 |
| RutticeQCD | 1.24728260729018e+02 |
| Tolerance | 1e-13 relative |
| Result | pass |

## 11. Clover, two flavours, exponential source, Coulomb gauge

| | |
|---|---|
| Date | 2026-10-06 |
| What | the same 312 numbers as entry 9, and the number of gauge-fixing sweeps |
| Reference | output of the unmodified Bridge++ test `Spectrum.Clover.Hadron2ptFunction` |
| Parameters | as entry 9, after Coulomb gauge fixing: overrelaxation ω = 1.6 after 50 sweeps with ω = 1, violation measured every 10 sweeps, stop below 1e-24 on every time slice, random reset every 1000 sweeps |
| RutticeQCD | test `clover_two_flavours_coulomb_gauge`; `fix_coulomb`, then as entry 9. Also through the program: test `clover_example_in_coulomb_gauge_reproduces_the_reference_table` |
| Precision of the reference | 13 significant digits |

| | Bridge++ | RutticeQCD |
|---|---|---|
| Sweeps until converged | 120 | 120 |
| Plaquette change | 4.9e-15 | below 1e-13 (checked) |
| `expected_result` (PS, flavours 1 1, t = 0) | 81.2796854325709 | within 6.5e-12 |

| | |
|---|---|
| Largest difference | 6.5e-12 over all 312 values |
| Tolerance | 1.6e-10 absolute (2e-12 of the largest value) |
| Result | pass, all 312 |

What this validates: the Coulomb gauge-fixing algorithm and its convergence measure
(stopping at the same sweep requires the same normalisation of the violation and the
same sequence of sweeps), and that the gauge it reaches is the same one: the exponential
source is gauge dependent: the first value is 81.28 here and 15.24 without gauge fixing
(entry 9).

What it does not validate: the random gauge transformations after `reset_interval`
sweeps, which this run never reached. They use RutticeQCD's own random numbers, so a run
that needs them will not follow Bridge++ sweep for sweep. Independent checks
(`stage2_gauge_fixing.rs`): convergence from a random configuration, the plaquette being
unchanged, the same configuration on every process grid with and without resets, and the
gauge transformation against a separate reference implementation.

## 12. Domain-wall operator (Shamir), one application

| | |
|---|---|
| Reference | Bridge++ test `Mult_Domainwall` (`test_Mult_Domainwall.yaml`), its `expected_result` |
| What | the norm of D applied to a field with every real equal to 1 |
| Parameters | 4×4×4×8, `conf_04040408.txt`, Ls = 8, M5 = 1.6, m = 0.1, b = 1, c = 0, boundary (1, 1, 1, -1), Dirac representation |
| RutticeQCD | test `bridge_mult_domainwall`; `Mdwf::mult` with `Variant::D` |
| Precision of the reference | 15 significant digits |

| | Bridge++ | RutticeQCD |
|---|---|---|
| \|D 1\| | 1.02352320377108e+03 | agrees to 1e-12 relative (the tolerance of the test) |

What this validates: the normalisation of the Wilson kernel inside the operator
(κ = 1/(8 - 2 M5), the factor 4 - M5), the projectors and the sign of the hops in the fifth
dimension, and the mass term across the walls.

What it does not validate: the Möbius coefficients (c = 0 here), D†, and Γ5 D. Those have
only checks that need no reference (`stage3_mdwf.rs`): D† is the adjoint of D for
(b, c) = (1.5, 0.5); Γ5 D is Hermitian for c = 0 and is not for c ≠ 0; two vectors in one
field are independent; several processes give the bits of one.

