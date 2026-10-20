# Development

For those who change the code: how it is laid out, how it is checked, and what is not there
yet.

## Structure

One Cargo package, `ruttice`.

| Directory | Contents |
|---|---|
| `src/` | the library, one directory per subject (below) |
| `apps/` | the programs: `ruttice`, `ruttice-bench`, and with the feature `mpi` `ruttice-mpi`, `ruttice-bench-mpi` and `ruttice-mpi-check` |
| `tests/` | the checks described above, one file per subject; `tests/data/` holds reference values |
| `examples/` | the parameter-file examples, `pion.rs` (the library used directly), and programs that time kernels (`*_bench.rs`, `hmc_eo_measure.rs`) |
| `docs/` | `COMPARISON.md` |

| Directory under `src/` | Contents |
|---|---|
| `field` | fields, storage types, the vector and linear-operator interfaces |
| `lattice` | lattice geometry, neighbour tables, even-odd decomposition, loops over sites in parallel |
| `comm` | communication between processes |
| `backend` | backends that need an external library: the communicator on MPI (feature `mpi`) |
| `kernel` | arithmetic on single sites: complex numbers, colour matrices, spinors, gamma matrices, SIMD lanes |
| `numerics` | small dense linear algebra; rational approximations |
| `io` | reading and writing gauge configurations |
| `gauge` | plaquette, field strength, gauge fixing, smearing, products of links along paths |
| `fermion` | the Wilson, clover and Möbius domain-wall operators; even-odd preconditioning |
| `solver` | Krylov solvers, mixed precision, and under `preconditioner` the polynomial preconditioners and the multigrids |
| `hmc` | actions and their forces, integrators |
| `measure` | sources and meson correlators |
| `app` | parameter files, building what they name, running and reporting |

Files with the same role have the same name. Under `fermion`, each fermion has

| File | Role |
|---|---|
| `mod.rs` | the operator, its parameters, what it is |
| `site.rs` | the part that acts within a site (the clover term; the fifth dimension) |
| `lexical.rs` | the operator on all sites, in lexical order |
| `even_odd.rs` | its even-odd blocks and the operators made of them |

and what only it has: `wilson/` the sweep of the hopping term (`sweep*.rs`), which the other
two are built on; `mdwf/quark_4d.rs` the four-dimensional quark fields of five-dimensional
ones. Clover fermions are the Wilson operator with another site term, so `clover/` has only
`mod.rs` and `site.rs`. The multigrids are named the same way
(`multigrid/wilson.rs`, `mdwf_even_odd.rs`, `mdwf_lexical.rs`), and so are the actions of
HMC (`hmc/gauge_action.rs`, `fermion_action.rs`, `even_odd.rs`, `rational.rs`).

A field is `Field<T, Kind, Sites, Layout>`: the storage type (`f32`, `f64`), what it holds
(`Gauge`, `Spinor`, `Coarse`), which sites (`Lexical`: all of them in lexical order;
`Half`: those of one parity) and how they are laid out (`Aos`: site by site; `Aosoa<W>`:
blocks of W sites, one per SIMD lane). The solvers know only vectors and linear maps;
nothing outside `solver::preconditioner::multigrid` knows about fermions.

### Using it as a library

The program is a thin layer over the library; anything it does can be done from Rust code.
The shortest complete example is [examples/pion.rs](../examples/pion.rs). Its core:

```rust
let lat = Lattice::single_node([4, 4, 4, 8]);
let u = read_bridge_text(&path, &lat)?;

let mut op = Wilson::<f64>::new(&lat, WilsonParams { kappa, boundary });
op.set_gauge(&u)?;

let b = point_source::<f64>(&lat, [0, 0, 0, 0]);   // twelve right-hand sides
let mut s = b.zeros_like();
solve_even_odd(&op, index, &mut s, &b, |schur, x_e, rhs| {
    solve_cgne(&mut Cg::new(stop), schur, x_e, rhs)
})?;

let c = pion_correlator(&s);                        // one value per time slice
```

## Tests

Two kinds of checks run with `cargo test`.

**Checks that need nothing else.** Properties the result must have whatever the numbers
are: gauge covariance, the free-field result for a plane wave, D† being the adjoint of D,
the solution of a linear system satisfying it when put back, the same answer from every
solver and every layout of the fields, the same bits for any number of threads, forces
being the derivatives of their actions.

**Comparisons with Bridge++.** The plaquette; the Wilson, clover and domain-wall operators
applied to a vector; solver iteration counts; meson correlators in thirteen channels on
every time slice, for one and two quark flavours and for point and smeared sources, with
and without Coulomb gauge fixing; the domain-wall pion correlator; and one trajectory of
Bridge++'s HMC tests of the actions Ruttice has (quenched, Wilson, clover, domain-wall over
Pauli-Villars fields, RHMC, even-odd Wilson). The comparisons of the solves are written down
in [docs/COMPARISON.md](COMPARISON.md). They are our comparisons with the numbers Bridge++
prints; the Bridge++ developers have not reviewed them. What Bridge++ does not compute (the
domain-wall multigrid, the low-mode probe, `A4_A4`) has only the checks above.

The comparisons read a Bridge++ test configuration, `conf_04040408.txt` (4³ × 8), that ships
with Bridge++ under the GPL; a copy is in `examples/`, where `cargo test` finds it. A file of
the same name in `run/` takes its place, and a directory of your own can be named instead:

```bash
RUTTICE_BRIDGE_TESTS=/path/to/that/directory cargo test
```

**Without the file those comparisons print `SKIPPED` and count as passed**, and the rest
still run: to know that the comparisons ran, look for `SKIPPED` in the output of
`cargo test -- --nocapture`.

## Status and plans

Version 0.2.0. See [CHANGELOG.md](../CHANGELOG.md). What is known and not done (structure,
performance, continuous integration, the multigrid paper, open questions) is listed with the
reasons and how to check it in [TODO.md](TODO.md).

Not there yet:

- Landau gauge fixing; baryons
- the device backend (`backend::cuda_bridge`) on more than one GPU, and the Wilson and clover
  multigrids on the device
- the domain-wall multigrid measured on volumes beyond 24³×48, and on more than one
  configuration per volume

The repository holds programs that time the kernels (`ruttice-bench`,
`examples/*_bench.rs`), not timings: those depend on the machine.
