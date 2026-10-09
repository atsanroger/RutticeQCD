# RutticeQCD

Lattice QCD in Rust.

RutticeQCD computes quark propagators and meson two-point functions on a given gauge
configuration, with Wilson and clover fermions. It is at an early stage: it has not been
tuned for speed and cannot generate gauge configurations yet. It runs on one process or,
with MPI, on several (see [Running on several processes](#running-on-several-processes)). What
it does compute has been checked number by number against
[Bridge++](https://bridge.kek.jp/Lattice-code/) (see [Correctness](#correctness)).

## Quick start

You need a stable Rust toolchain (1.89 or newer) and a gauge configuration.

```bash
git clone https://github.com/atsanroger/RutticeQCD.git
cd RutticeQCD
cargo build --release
```

This produces the program `target/release/ruttice`. It takes one argument, a parameter
file. A parameter file names the lattice, the gauge configuration, the solver, the quarks
and what to measure; paths inside it are relative to the file itself.

Keep your own runs in `run/` at the top of the repository: git ignores that directory, so
configurations, parameter files and results can live there without ever being committed.
Copy an example into it, put the configuration next to it, and run:

```bash
mkdir -p run
cp examples/wilson_pion.toml run/
cp /path/to/conf_04040408.txt run/
target/release/ruttice run/wilson_pion.toml
```

The two commented examples to start from:

- [examples/wilson_pion.toml](examples/wilson_pion.toml): Wilson fermions, point
  source, pion correlator
- [examples/clover_two_flavours.yaml](examples/clover_two_flavours.yaml): clover
  fermions, two flavours, smeared sources, every channel

The output for the first example, on a 4×4×4×8 lattice, looks like this:

```text
# lattice    [4, 4, 4, 8]
# plaquette  0.625129946974942
# quark 1    40 iterations, 0 restarts, worst |D S - b|^2/|b|^2 = 8.782e-29, 1.245 s
# quarks channel t re im
11 PS 0 1.422503945020827e1 0.000000000000000e0
11 PS 1 6.530367660173256e-1 0.000000000000000e0
...
```

Lines starting with `#` describe the run. Every other line is one value of a correlator:
the two quarks it was built from (`11` is quark 1 with itself, `12` is quark 1 with quark
2), the channel, the time slice, and the real and imaginary parts.

### Running on several processes

The MPI version is a second program, `ruttice-mpi`, built when the feature `mpi` is turned
on. Building it needs an MPI installation (Open MPI, MPICH or Intel MPI) and libclang:

```bash
cargo build --release --features mpi
mpirun -np 4 target/release/ruttice-mpi run/wilson_pion.toml
```

It reads the same parameter files and prints the same table. How the lattice is split is
set in the parameter file:

```toml
process_grid = [1, 1, 2, 2]     # processes in x, y, z, t
```

The product must be the number of processes, and each number must divide the lattice
extent in its direction. Without `process_grid` the lattice is split automatically. With
`even_odd = true`, the part of the lattice on each process must have even extents.

`mpirun` must come from the same MPI installation the program was built with. If it does
not, every process believes it is alone, and the program stops with a message saying that
`process_grid` needs more processes than there are. After switching to another MPI
installation, run `cargo clean -p mpi-sys` before building again.

Every process reads the whole configuration file and keeps its part, so this is meant
for files that fit in memory. The results agree with a single process up to rounding.

### The gauge configuration file

`gauge.format` names the format of the file:

| `gauge.format` | File |
|---|---|
| `bridge_text` | plain text with one real number per line, the format Bridge++ calls `Text` |
| `ildg` | the ILDG format: a LIME container with the records `ildg-format` and `ildg-binary-data` |
| `scidac` | a lattice-colour-matrix record written by the SciDAC QIO library: a LIME container with `scidac-binary-data` |
| `hdf5` | the dataset `gauge` of an HDF5 file, double precision, of shape `[Lt, Lz, Ly, Lx, 4, 3, 3, 2]`; needs a program built with `--features hdf5` and the HDF5 library |

All four keep the numbers in the same order, from the outermost loop to the innermost:

1. site, with x running fastest: `site = x + Lx * (y + Ly * (z + Lz * t))`
2. direction mu = x, y, z, t
3. the 3×3 link matrix row by row, real part then imaginary part of each entry

That is `Lx * Ly * Lz * Lt * 4 * 18` numbers. A file of the wrong length or for another
lattice is rejected. The binary formats hold big-endian reals in single or double precision;
`ildg` and `scidac` are read by the same code, which accepts either kind of binary record
and verifies the SciDAC checksum when the file has one.

The LIME and HDF5 readers and writers were written from the descriptions of the formats and
have not yet been tried on files from other programs; the HDF5 layout is this program's own.

## The parameter file

The file is TOML (`.toml`) or YAML (`.yaml`, `.yml`); both have the same structure. A key
that is not listed here, or a name that is not one of the options, is an error reported
when the file is read, with the line and the accepted alternatives.

| Key | Meaning | Default |
|---|---|---|
| `lattice` | `[Lx, Ly, Lz, Lt]` | required |
| `gauge.file` | path of the configuration | required |
| `gauge.format` | `bridge_text`, `ildg`, `scidac` or `hdf5`; see below | `bridge_text` |
| `gauge.fixing` | fix the gauge first; see below | none |
| `solver.type` | `cg_normal` (conjugate gradient on the normal equations), `bicgstab`, or `fgmres` (flexible GMRES; `max_iterations` is then the Krylov space kept before a restart) | required |
| `solver.even_odd` | solve through the even-odd (red-black) Schur complement; needs even extents | `true` |
| `solver.tolerance_squared` | stop when \|r\|²/\|b\|² is below this for every right-hand side | required |
| `solver.max_iterations` | iterations before the true residual is recomputed and the solver restarts | required |
| `solver.max_restarts` | number of such restarts allowed | `3` |
| `solver.simd` | lattice sites computed together as the lanes of one SIMD register: `1`, `2`, `4` or `8`; see below | `1` |
| `solver.multigrid` | precondition with two-level multigrid; see below | none |
| `quark` | a list; one entry per quark flavour, each with `fermion` and `source` | at least one |
| `measure.channels` | `"all"` or a list of channel names | `"all"` |
| `output.file` | also write the table to this file | none |

With `solver.simd` above 1 the fields are kept in blocks of that many sites, and the
operator and the solvers work on whole blocks. In double precision `4` fills a 256-bit
register (AVX2) and `8` a 512-bit one (AVX-512); the program must have been built for a
processor that has them (`RUSTFLAGS="-C target-cpu=native"`), otherwise the blocks are
computed with ordinary instructions. The results are those of `simd = 1` up to the order
of the sums in the solver. The lattice is cut into that many equal parts by halving it,
t first, then z, y, x; with `even_odd = true` the parts must have even extents, so a
4×4×4×8 lattice allows up to `8`. On several processes it is the part of the lattice on
each process that is cut. Not together with multigrid.

Multigrid needs a solver that tolerates a preconditioner changing from one application to
the next, `solver.type = "fgmres"` or `"bicgstab"`, and `solver.even_odd = false`; it runs on
one process only so far. Only `block` is required; the defaults are those of the Bridge++
test `test_alt_Multigrid.yaml` (SAP smoother, BiCGStab on the coarse lattice). The other
choices are those of the CUDA domain-wall multigrid (GMRES(m) smoother before and after,
GMRES(m) on the coarse lattice, coarse operator preconditioned by its diagonal blocks):

| `solver.multigrid` key | Meaning | Default |
|---|---|---|
| `block` | `[bx, by, bz, bt]`: the blocks of the coarse lattice and of the smoother; must divide the lattice | required |
| `test_vectors` | number of test vectors (at most 6 × the block volume) | `16` |
| `setup_iterations` | rounds that improve the test vectors after the initial smoothing | `4` |
| `coarse_solver` | `bicgstab` or `gmres` | `bicgstab` |
| `coarse_tolerance_squared` | the coarse solve stops below this \|r\|²/\|b\|² … | `2.0e-3` |
| `coarse_max_iterations` | … or after this many iterations (`gmres`: the m of GMRES(m)) | `200` |
| `precondition_coarse_diagonal` | multiply the coarse operator by the inverse of its diagonal blocks | `false` |
| `smoother` | `sap` or `gmres` | `sap` |
| `pre_smoothing` | smooth before the coarse correction too, instead of starting from x = b | `false` |
| `smoother_cycles` | `sap`: cycles | `4` |
| `smoother_block_steps` | `sap`: minimal-residual steps per block in each cycle | `6` |
| `smoother_iterations` | `gmres`: the m of GMRES(m) | `4` |
| `smoother_tolerance_squared` | `gmres`: stop earlier below this \|r\|²/\|b\|² | `1.0e-3` |
| `precision` | `single` or `double`: what the preconditioner works in; the outer solve is always double | `single` |
| `seed` | seed of the random initial test vectors | `1` |

```toml
[solver]
type = "fgmres"
even_odd = false
tolerance_squared = 1.0e-28
max_iterations = 30
max_restarts = 30
multigrid = { block = [2, 2, 2, 2] }
```

Each quark has a fermion action:

| `fermion.type` | Other keys |
|---|---|
| `wilson` | `kappa`, `boundary` |
| `clover` | `kappa`, `csw`, `boundary` |

`boundary` is a list of four entries, one per direction, each `periodic`, `antiperiodic` or
`dirichlet`.

and a source:

| `source.type` | Other keys | What it is |
|---|---|---|
| `point` | `position` | a unit source at one site |
| `exponential` | `position`, `slope`, `power` | `exp(-slope * r^power)` on the time slice of `position`, normalised to unit norm. It depends on the gauge the configuration is in, so it is normally used with `gauge.fixing`. |

Gauge fixing has one type, `coulomb`; every other key is optional:

| `gauge.fixing` key | Meaning | Default |
|---|---|---|
| `type` | `coulomb` | required |
| `tolerance_squared` | stop when the violation is below this on every time slice | `1.0e-24` |
| `max_iterations` | give up after this many sweeps | `5000` |
| `naive_iterations` | sweeps without overrelaxation at the start and after every reset | `50` |
| `measure_interval` | measure the violation every this many sweeps | `10` |
| `reset_interval` | every this many sweeps, randomly transform the time slices that have not converged | `1000` |
| `overrelaxation` | overrelaxation parameter ω | `1.6` |
| `seed` | seed for those random transformations | `1234567` |

The defaults are those of the Bridge++ test `Spectrum_Clover_Hadron2ptFunction`, and so is
the measure of the violation, so the same parameters stop after the same number of sweeps.
In YAML:

```yaml
gauge:
  file: conf_04040408.txt
  fixing:
    type: coulomb
```

For every quark the propagator is computed for all twelve spin-colour components of the
source at once. Correlators are then formed for every pair of quarks (i, j) with i ≤ j.

The channels are named after the gamma matrices at the sink and at the source:

| Name | Sink ← source | Name | Sink ← source |
|---|---|---|---|
| `PS` | γ5 ← γ5 | `S` | 1 ← 1 |
| `V1` `V2` `V3` | γ_k ← γ_k | `G51` `G52` `G53` | γ5γ_k ← γ5γ_k |
| `A4_PS` | γ4γ5 ← γ5 | `S12` `S23` `S31` | iγ2γ1, iγ3γ2, iγ1γ3 at both ends |
| `PS_A4` | γ5 ← γ5γ4 | `A4_A4` | γ4γ5 ← γ5γ4 |

They are printed in the order of the Bridge++ spectrum test, with `A4_A4` after the other
axial channels. All but `A4_A4` were checked against Bridge++; `A4_A4` is not in Bridge++
and has no reference values from another program.

## Conventions

So that numbers can be compared with other codes:

- **Wilson operator.** `D ψ(x) = ψ(x) − κ Σ_μ [ (1 − γ_μ) U_μ(x) ψ(x+μ) + (1 + γ_μ) U_μ(x−μ)† ψ(x−μ) ]`.
  The hopping parameter is related to the bare mass by `κ = 1 / (2 m0 + 8)`.
- **Clover operator.** `D = D_Wilson − κ c_SW Σ_{μ<ν} F_μν γ_μ γ_ν`, with the clover-leaf
  field strength `F_μν = (Q − Q†)/8`, Q being the sum of the four plaquettes in the μ-ν
  plane that start at x.
- **Gamma matrices.** Dirac representation (γ4 diagonal), Hermitian, with
  γ5 = [[0, 1], [1, 0]] in 2×2 blocks.
- **Boundary conditions** apply to the fermions only; the gauge field is always periodic.
- **Meson correlator.** `C(t) = Σ_x Tr[ (γ5 Γ_sink) S1(x,t) (Γ_source γ5) S2(x,t)† ]`,
  where S1 and S2 are the propagators of the two quarks from the same source and the trace
  is over spin and colour. No further sign or normalisation is applied.
- **Plaquette.** Normalised so that a unit gauge field gives 1.
- **Coulomb gauge.** Maximises Σ_x Σ_i Re tr U_i(x) over the spatial directions i on every
  time slice. The violation of a time slice is Σ_x |Δ(x)|² / (4 Nc V_t), where Δ(x) is the
  traceless part of M − M† with M = Σ_i [U_i(x−i) − U_i(x)], |·|² sums the squares of all
  real components and V_t is the number of sites in a time slice.

These follow Bridge++, which is what the results are checked against.

## Correctness

Two kinds of checks run with `cargo test`.

**Checks that need nothing else.** Properties the result must have whatever the numbers
are: gauge covariance, the free-field result for a plane wave, D† being the adjoint of D,
the solution of a linear system satisfying it when put back, the same answer from every
solver, the same bits for any number of threads.

**Comparisons with Bridge++.** The plaquette; the Wilson and clover operators applied to a
vector, with and without even-odd preconditioning; solver iteration counts; and meson
correlators in thirteen channels on every time slice, for one and two quark flavours and
for point and smeared sources, with and without Coulomb gauge fixing. Every comparison is written down in
[docs/COMPARISON.md](docs/COMPARISON.md): what was compared, the two numbers, the
tolerance, and what the comparison does and does not establish.

The comparisons read a Bridge++ test configuration, `conf_04040408.txt`, that ships with
Bridge++ and is not part of this repository. Put it in `run/` and `cargo test` finds it;
or keep it elsewhere and name the directory:

```bash
RUTTICE_BRIDGE_TESTS=/path/to/that/directory cargo test
```

Without the file those comparisons print `SKIPPED` and the rest still run.

## Using it as a library

The program is a thin layer over the library (`ruttice`, in `src/`); anything it does can be done from Rust
code. The shortest complete example is
[examples/pion.rs](examples/pion.rs). Its core:

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

One Cargo package, `ruttice`. Where things go is fixed by the design document (§8):

| Directory | Contents |
|---|---|
| `src/` | the library, one directory per subject (below) |
| `apps/` | the programs: `ruttice`, and with the feature `mpi` `ruttice-mpi` and `ruttice-mpi-check` |
| `tests/` | the checks described above; `tests/data/` holds the reference values |
| `examples/` | `pion.rs` and the parameter-file examples |
| `docs/` | `COMPARISON.md` |

| Directory under `src/` | Contents |
|---|---|
| `field` | fields, storage types, the vector and linear-operator interfaces |
| `lattice` | lattice geometry, neighbour tables, even-odd decomposition, site-parallel loops |
| `comm` | communication between processes |
| `backend` | backends that need an external library: the communicator on MPI (`backend::mpi`, feature `mpi`) |
| `kernel` | arithmetic on single sites: complex numbers, colour matrices, spinors, gamma matrices |
| `numerics` | small dense linear algebra |
| `io` | reading and writing gauge configurations |
| `gauge` | plaquette, field strength, gauge transformations, Coulomb gauge fixing |
| `fermion` | Wilson and clover operators, even-odd preconditioning |
| `solver` | conjugate gradient, BiCGStab, flexible GMRES, mixed precision by defect correction, multigrid |
| `measure` | sources and meson correlators |
| `app` | parameter files, building what they name, running and reporting |

Things the library can do that the program does not expose yet: single-precision solves
and double-precision accuracy from them by defect correction (multigrid already works in
single precision inside a double outer solve).

## Limitations and plans

Not there yet, roughly in the order planned:

- fast multigrid: it works and cuts the outer iterations (FGMRES from 157 to 8 on the
  Bridge++ test configuration, test `multigrid_on_the_bridge_configuration`), but its smoother
  is not tuned, so on small lattices it is slower than CG
- domain-wall fermions
- vectorised (SIMD) kernels and running on more than one process (MPI)
- generating configurations (HMC), Landau gauge fixing, smearing
- GPUs

No timing in this repository should be read as a performance figure.

## License

RutticeQCD is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version
(SPDX: `GPL-3.0-or-later`). It is distributed in the hope that it will be useful, but
without any warranty; see the file [LICENSE](LICENSE) for the full text.
