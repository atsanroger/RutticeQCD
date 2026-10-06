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

- Stage 2, in progress: clover-leaf field strength; the clover operator, which is the
  Wilson sweep with a non-trivial site-diagonal block, including its even-odd form.
  Not yet: FGMRES, multigrid.

Single node, array-of-structures layout only. Comparisons with reference codes are
recorded in [docs/COMPARISON.md](docs/COMPARISON.md).

## Layout

| Directory | Crate | Contents |
|---|---|---|
| `kernel/` | `ruttice-kernel` | arithmetic on small by-value objects, `no_std` |
| `numerics/` | `ruttice-numerics` | small dense linear algebra, no lattice fields |
| `core/` | `ruttice-core` | fields, lattice, stencil, site-parallel entry points |
| `io/` | `ruttice-io` | gauge configuration I/O |
| `gauge/` | `ruttice-gauge` | gauge observables |
| `fermion/` | `ruttice-fermion` | fermion operators, one per file |
| `solver/` | `ruttice-solver` | Krylov solvers and the pieces mixed precision is built from |
| `measure/` | `ruttice-measure` | sources and hadron correlators |
| `validation/` | `ruttice-validation` | physics checks and comparisons with reference codes |

## Running something

There is no command-line program yet; the crates are libraries. One example runs the whole
chain on a single configuration (plain-text format, one real per line):

```bash
cargo run --release -p ruttice-validation --example pion -- CONFIG LX LY LZ LT KAPPA
```

It prints the plaquette, solves for the propagator from a point source at the origin and
prints the pion correlator for every time slice. Its source,
[validation/examples/pion.rs](validation/examples/pion.rs), is the shortest complete
illustration of how the pieces fit together.

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
