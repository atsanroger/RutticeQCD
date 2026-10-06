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
cargo build --release -p ruttice-cli
```

This produces the program `target/release/ruttice`. It takes one argument, a parameter
file:

```bash
target/release/ruttice my_run/wilson_pion.toml
```

A parameter file names the lattice, the gauge configuration, the solver, the quarks and
what to measure. Start from one of the two commented examples and put it next to your
configuration file (paths inside a parameter file are relative to that file):

- [cli/examples/wilson_pion.toml](cli/examples/wilson_pion.toml): Wilson fermions, point
  source, pion correlator
- [cli/examples/clover_two_flavours.yaml](cli/examples/clover_two_flavours.yaml): clover
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

The MPI version is a second program, `ruttice-mpi`, built from the `ruttice-mpi` package.
Building it needs an MPI installation (Open MPI or MPICH) and libclang:

```bash
cargo build --release -p ruttice-mpi
mpirun -np 4 target/release/ruttice-mpi --grid 1,1,2,2 my_run/wilson_pion.toml
```

It reads the same parameter files and prints the same table. `--grid` gives the number of
processes in x, y, z and t; their product must be the number of processes, and each must
divide the lattice extent in its direction. Without `--grid` the lattice is split
automatically. With `even_odd = true`, the part of the lattice on each process must have
even extents.

Every process reads the whole configuration file and keeps its part, so this is meant
for files that fit in memory. The results agree with a single process up to rounding.

### The gauge configuration file

One format is read so far: plain text with one real number per line, the format Bridge++
calls `Text`. The numbers are ordered, from the outermost loop to the innermost:

1. site, with x running fastest: `site = x + Lx * (y + Ly * (z + Lz * t))`
2. direction mu = x, y, z, t
3. the 3×3 link matrix row by row, real part then imaginary part of each entry

That is `Lx * Ly * Lz * Lt * 4 * 18` numbers. A file of the wrong length is rejected.

## The parameter file

The file is TOML (`.toml`) or YAML (`.yaml`, `.yml`); both have the same structure. A key
that is not listed here, or a name that is not one of the options, is an error reported
when the file is read, with the line and the accepted alternatives.

| Key | Meaning | Default |
|---|---|---|
| `lattice` | `[Lx, Ly, Lz, Lt]` | required |
| `gauge.file` | path of the configuration | required |
| `gauge.format` | `bridge_text` | `bridge_text` |
| `solver.type` | `cg_normal` (conjugate gradient on the normal equations) or `bicgstab` | required |
| `solver.even_odd` | solve through the even-odd (red-black) Schur complement; needs even extents | `true` |
| `solver.tolerance_squared` | stop when \|r\|²/\|b\|² is below this for every right-hand side | required |
| `solver.max_iterations` | iterations before the true residual is recomputed and the solver restarts | required |
| `solver.max_restarts` | number of such restarts allowed | `3` |
| `quark` | a list; one entry per quark flavour, each with `fermion` and `source` | at least one |
| `measure.channels` | list of channel names | all thirteen |
| `output.file` | also write the table to this file | none |

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
| `exponential` | `position`, `slope`, `power` | `exp(-slope * r^power)` on the time slice of `position`, normalised to unit norm. It depends on the gauge the configuration is in; no gauge fixing is applied. |

For every quark the propagator is computed for all twelve spin-colour components of the
source at once. Correlators are then formed for every pair of quarks (i, j) with i ≤ j.

The channels are named after the gamma matrices at the sink and at the source:

| Name | Sink ← source | Name | Sink ← source |
|---|---|---|---|
| `PS` | γ5 ← γ5 | `S` | 1 ← 1 |
| `V1` `V2` `V3` | γ_k ← γ_k | `G51` `G52` `G53` | γ5γ_k ← γ5γ_k |
| `A4_PS` | γ4γ5 ← γ5 | `S12` `S23` `S31` | iγ2γ1, iγ3γ2, iγ1γ3 at both ends |
| `PS_A4` | γ5 ← γ5γ4 | | |

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
for point and smeared sources. Every comparison is written down in
[docs/COMPARISON.md](docs/COMPARISON.md): what was compared, the two numbers, the
tolerance, and what the comparison does and does not establish.

The comparisons read a Bridge++ test configuration that is not part of this repository.
Point `RUTTICE_BRIDGE_TESTS` at a directory containing `conf_04040408.txt` (it ships with
Bridge++):

```bash
RUTTICE_BRIDGE_TESTS=/path/to/that/directory cargo test
```

Without it those comparisons print `SKIPPED` and the rest still run.

## Using it as a library

The program is a thin layer over a set of crates; anything it does can be done from Rust
code. The shortest complete example is
[validation/examples/pion.rs](validation/examples/pion.rs). Its core:

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

| Directory | Crate | Contents |
|---|---|---|
| `kernel/` | `ruttice-kernel` | arithmetic on single sites: complex numbers, colour matrices, spinors, gamma matrices. No standard library |
| `numerics/` | `ruttice-numerics` | small dense linear algebra |
| `core/` | `ruttice-core` | lattice, fields, neighbour tables, even-odd decomposition, the vector and linear-operator interfaces |
| `io/` | `ruttice-io` | reading and writing gauge configurations |
| `gauge/` | `ruttice-gauge` | plaquette, field strength |
| `fermion/` | `ruttice-fermion` | Wilson and clover operators, even-odd preconditioning |
| `solver/` | `ruttice-solver` | conjugate gradient, BiCGStab, flexible GMRES, mixed precision by defect correction |
| `measure/` | `ruttice-measure` | sources and meson correlators |
| `cli/` | `ruttice-cli` | the `ruttice` program |
| `validation/` | `ruttice-validation` | the checks described above |

Things the library can do that the program does not expose yet: single precision, and
double-precision accuracy from single-precision solves.

## Limitations and plans

Not there yet, roughly in the order planned:

- multigrid solvers
- domain-wall fermions
- vectorised (SIMD) kernels and running on more than one process (MPI)
- generating configurations (HMC), gauge fixing, smearing
- GPUs

No timing in this repository should be read as a performance figure.

## License

No license has been chosen yet.
