# RutticeQCD

Lattice QCD in Rust. Early development. Code comments cite sections of the design notes
(`DESIGN.md`, `FLOW.md`), which are kept outside this repository for now.

## Status

Stage 0 of DESIGN.md §9.1: fields, nearest-neighbour stencil, site-parallel entry
points, reading a plain-text gauge configuration, and the plaquette. Single node, `f64`,
array-of-structures layout only.

## Layout

| Directory | Crate | Contents |
|---|---|---|
| `kernel/` | `ruttice-kernel` | arithmetic on small by-value objects, `no_std` |
| `core/` | `ruttice-core` | fields, lattice, stencil, site-parallel entry points |
| `io/` | `ruttice-io` | gauge configuration I/O |
| `gauge/` | `ruttice-gauge` | gauge observables |
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
