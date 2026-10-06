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
