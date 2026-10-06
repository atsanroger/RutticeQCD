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
  Wilson sweep with a non-trivial site-diagonal block, including its even-odd form;
  exponential sources; meson correlators from two different propagators; a command-line
  program driven by parameter files. Not yet: FGMRES, multigrid, gauge fixing.

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
| `cli/` | `ruttice-cli` | the `ruttice` program: parameter files in, correlators out |
| `validation/` | `ruttice-validation` | physics checks and comparisons with reference codes |

## Running something

The `ruttice` program reads a parameter file and prints meson two-point functions:

```bash
cargo run --release -p ruttice-cli -- PARAMETER_FILE
```

or build it once with `cargo build --release -p ruttice-cli` and run
`target/release/ruttice PARAMETER_FILE`.

The parameter file is TOML (`.toml`) or YAML (`.yaml`, `.yml`), with the same structure in
both. It names the lattice, the gauge configuration, the solver, one or more quarks (each
with its fermion action and source) and the channels to measure. Two commented examples:

- [cli/examples/wilson_pion.toml](cli/examples/wilson_pion.toml): Wilson, point source, pion
- [cli/examples/clover_two_flavours.yaml](cli/examples/clover_two_flavours.yaml): clover,
  two flavours, exponential sources, all channels

Paths inside a parameter file are relative to that file. The output is a few comment lines
(plaquette, iterations and residual of each solve) followed by one line per quark pair,
channel and time slice: `<i><j> <channel> <t> <re> <im>`.

What a parameter file can choose today:

| | Options |
|---|---|
| fermion | `wilson` (kappa), `clover` (kappa, csw); boundary per direction: `periodic`, `antiperiodic`, `dirichlet` |
| source | `point` (position), `exponential` (position, slope, power) |
| solver | `cg_normal`, `bicgstab`; `even_odd` true or false; tolerance and iteration limits |
| channels | PS V1 V2 V3 A4_PS PS_A4 S G51 G52 G53 S12 S23 S31, or all when omitted |
| gauge configuration | plain text, one real per line |

Unknown keys and unknown names are errors when the file is read. Single process, double
precision only.

To use the crates as a library instead, see
[validation/examples/pion.rs](validation/examples/pion.rs).

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
