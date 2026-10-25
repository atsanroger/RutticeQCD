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
| `docs/` | `COMPARISON.md`; `mg_background/`, the notes on the multigrid; `implementation/`, how the program is built: its architecture (`ARCHITECTURE.md`), writing a GPU kernel (`GPU_KERNELS.md`), every CUDA kernel (`KERNELS_NOTE`), several GPUs (`MULTI_GPU_NOTE`, with an evaluation of NVSHMEM) |

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
| `measure` | sources; meson and proton correlators, with momentum; three-point functions by a sequential source through the sink (`Sink`, `sequential_source`, `current_insertion`, `conserved_charge`) |
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

### One algorithm, two backends

The CPU and the device (`backend::cuda_bridge`) do the same things because each algorithm is
written once, against traits, and a backend supplies only the parts that touch its memory:
operators, fields, kernels. A feature added to the algorithm is then on both; a backend refuses
an option only where it lacks a kernel. For the domain-wall fermions (decided 2026-10-23, done
2026-10-24):

| Algorithm, written once | What a backend supplies |
|---|---|
| the operators of the even sites: `PvRatio`, `SchurFactor`, `AfterQ`, `adjoint_by_gamma5` (`fermion::even_odd`) | `EvenOdd` for its operator (`Mdwf`, `CudaMdwf`) |
| the pseudofermion actions of Wilson and clover fermions, Rational included, and how a parameter file wires them (`hmc::TwoFlavor`, `TwoFlavorEo`, `TwoFlavorRatio`, `Rational`; `app::hmc::wilson_type`) | `ForceOp`, `EvenOddForceOp`: the hop derivative, the clover term's insertion and leaves, ln det T (`Wilson`, `Clover`, `CudaWilson`, `CudaClover`) |
| the coarse correction and the cycle: `Correction`, `TwoGrid`, `Chebyshev` | P, the coarse operators, the levels below the coarse lattice |
| the setup of the multigrid as the propagator runs and HMC take it: what the file asks (`MdwfMultigridSettings`, from `MultigridBlock::mdwf_settings` for both), the test vectors from a checkpoint or the filter and the bootstrap, P, the levels, the coarse parts, the tops of the smoothers, all of the cycle in its precision (`solver::preconditioner::multigrid::MdwfBackend::rebuild`) | `MdwfBackend`: its fields and the kernels of those steps, the cycles made of the parts (`OnCpu`, `cuda_bridge::mdwf_backend::OnDevice`) |
| the multigrid of HMC: when to set up, bootstrap or rebuild, the cache, the retries before CG, the solves with D (Neff's α by the left form and Q) and with D† (`hmc::dwf_multigrid::with_solves`) | — |
| the multigrid of Wilson and clover fermions: the setup (test vectors, setup iterations, P, the coarse operator, the levels) and the cycle (`multigrid::WilsonMultigrid`); its kept test vectors in HMC (`MultigridSetup::cycle_on`); the cycle on the even sites (`fermion::OnParity`) | `MultigridBackend`: P from test vectors, the coarse operator, the smoother (`WilsonOnCpu`, `cuda_bridge::WilsonOnDevice`) |
| SAP: the cycles, colours and minimal-residual steps (`multigrid::Sap`) | `SapBlocks`: the block operator, a step on every block, keeping one colour (`CpuBlocks`, `cuda_bridge::DeviceBlocks`) |
| mixed precision: CG on D†D or K†K in single precision, with Q for Neff's α, for the defect correction of HMC and the reliable updates of the CGNE of the propagators (`fermion::SingleAll`, `SingleEven`) | `SingleOperator`: its operator in single precision |
| the eigensolver run: the operator of the mode, the start vector, Lanczos or Arnoldi (`app::eigen`) | `EigenBackend`: its operators, the start vector on it |
| the propagator run: sources, solves, the four-dimensional fields, the measurements (`app::mdwf::solve_sources`) | where the fields live |

Before this (2026-10-24), the multigrid of HMC, the mixed solvers and the setup of the
propagator runs were written once per backend, and the copies had drifted apart (Neff's α, the
levels and the precision of the cycle in the multigrid of HMC on the device only; mixed CGNE of
the propagators on the device only; the coarse operators of a cycle in single precision probed
in single precision by the propagator runs and in double by HMC on the device). The propagator
runs keep only what is theirs: the smoother of the CPU in the SIMD layouts
(`PvEoCycle::from_parts_in_layout`) and their reports.

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
being the derivatives of their actions, the charge of the conserved vector current in
three-point functions (constant between source and sink, its jump the number of quarks
times the two-point function).

**Comparisons with Bridge++.** The plaquette; the Wilson, clover and domain-wall operators
applied to a vector; solver iteration counts; meson correlators in thirteen channels on
every time slice, for one and two quark flavours and for point and smeared sources, with
and without Coulomb gauge fixing; meson correlators at ten momenta and the proton correlator
(numbers no Bridge++ test checks, printed by a driver linked to Bridge++); the domain-wall pion
correlator; and one trajectory of
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
