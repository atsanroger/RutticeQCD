# RutticeQCD

Lattice QCD in Rust. Early development. Code comments cite sections of the design notes
(`DESIGN.md`, `FLOW.md`), which are kept outside this repository for now.

## Status

Stages refer to DESIGN.md §9.1.

- Stage 0, done: fields, nearest-neighbour stencil, site-parallel entry points, reading a
  plain-text gauge configuration, the plaquette.
- Stage 1, done except FGMRES: the Wilson operator (fused sweep for D, D†, γ5 D; hopping
  terms one direction at a time; even-odd blocks); CG and BiCGStab for one or several
  right-hand sides; mixed precision by defect correction; even-odd preconditioning written
  once for all fermions; point sources and meson correlators.

Single node, array-of-structures layout only. Comparisons with reference codes are
recorded in [docs/COMPARISON.md](docs/COMPARISON.md).

## Layout

| Directory | Crate | Contents |
|---|---|---|
| `kernel/` | `ruttice-kernel` | arithmetic on small by-value objects, `no_std` |
| `core/` | `ruttice-core` | fields, lattice, stencil, site-parallel entry points |
| `io/` | `ruttice-io` | gauge configuration I/O |
| `gauge/` | `ruttice-gauge` | gauge observables |
| `fermion/` | `ruttice-fermion` | fermion operators, one per file |
| `solver/` | `ruttice-solver` | Krylov solvers and the pieces mixed precision is built from |
| `measure/` | `ruttice-measure` | sources and hadron correlators |
| `validation/` | `ruttice-validation` | physics checks and comparisons with reference codes |

## Tests

```bash
cargo test
```

Comparisons against Bridge++ need its test inputs, which are not part of this repository.
Point `RUTTICE_BRIDGE_TESTS` at a directory containing them:

```bash
RUTTICE_BRIDGE_TESTS=/path/to/bridge/tests cargo test
```

Without it those comparisons are skipped and say so.

## License

Not decided yet.
