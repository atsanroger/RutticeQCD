# Usage

The parameter files, the ways of running, and the files of gauge configurations. The
README has the short version and says which solver to choose.

### Parameter files

A parameter file has the sections and keys of a Bridge++ parameter file, in YAML (`.yaml`,
`.yml`) or TOML (`.toml`). Keywords (the values of choices) are read in any case and with or
without `_`: `BiCGStab`, `bicgstab` and `BICG_STAB` are the same. A key that does not exist
is an error when the file is read, with the alternatives; a key that is there but not used
by the run is named in a note.

The commented examples show every key:

| File | What it runs |
|---|---|
| [examples/wilson_pion.yaml](../examples/wilson_pion.yaml) | Wilson fermions, point source, pion correlator |
| [examples/clover_two_flavours.yaml](../examples/clover_two_flavours.yaml) | clover fermions, two quarks, smeared sources, every channel |
| [examples/clover_mg.yaml](../examples/clover_mg.yaml) | clover fermions with the multigrid |
| [examples/mdwf_pv_eo_mg.yaml](../examples/mdwf_pv_eo_mg.yaml) | Möbius domain-wall fermions with the multigrid (and `.toml`: the same in TOML, the multigrid from a file of its own) |
| [examples/hmc_domainwall_pv.yaml](../examples/hmc_domainwall_pv.yaml) | HMC: two flavours of domain-wall fermions over Pauli-Villars, stout links |
| [examples/hmc_clover_rhmc_nf2p1.yaml](../examples/hmc_clover_rhmc_nf2p1.yaml) | HMC: 2 + 1 flavours of clover fermions, the strange quark by RHMC |

The sections of a file that solves for propagators:

| Section | What it holds |
|---|---|
| `Main` | `lattice_size`; `grid_size` (processes per direction); `backend` (`cpu` or `simd`) and `simd_lanes` |
| `Gauge` | `config_filename_input` and `gauge_config_type_input` (`Text`, `ILDG`, `SciDAC`, `HDF5`) |
| `GaugeFixing` | optional: `gauge_fixing_type : Coulomb` and its parameters |
| `GaugeSmearing` | optional: stout or APE smearing of the links the fermions see |
| `Fopr` | the fermion: `fermion_type` (`Wilson`, `Clover`, `Mobius`) and its parameters |
| `Source` | `source_type` (`Local`, `Exponential`), `source_position` |
| `Quark_2` … `Quark_4` | Wilson and clover: further quarks, each a `Fopr` and a `Source` |
| `Solver` | `solver_type`, `operator_type`, the stopping rule, `preconditioner_type` |
| `MultiGrid` | the multigrid, used with `preconditioner_type : MG` |
| `Measure` | Wilson and clover: `channels` (`all`, or a list) |
| `Output` | Wilson and clover: `file`, to write the table to as well |

`Solver.operator_type` says where the system is solved, for the fine lattice and for the
coarse lattice of the multigrid alike: `Even_odd` (the default) solves the Schur complement
on the even sites, and the coarse system on the even coarse sites; `Lexical` solves on all
sites of both. Domain-wall fermions are always solved on the even sites
(`PV_right`, `PV_left`, `Normal_left` choose the form of the operator).

An HMC file has a format of its own for now (`lattice`, `start`, `hmc`, `actions`,
`output`; see the two examples). It will take the sections of Bridge++'s `HMC_General` in
the next version.

### What is computed

For every quark the propagator is computed for all twelve spin-colour components of the
source. Wilson and clover runs print the meson correlators of every pair of quarks (i, j)
with i ≤ j; domain-wall runs print the pion correlator and the residual mass.

The channels are named after the gamma matrices at the sink and at the source:

| Name | Sink ← source | Name | Sink ← source |
|---|---|---|---|
| `PS` | γ5 ← γ5 | `S` | 1 ← 1 |
| `V1` `V2` `V3` | γ_k ← γ_k | `G51` `G52` `G53` | γ5γ_k ← γ5γ_k |
| `A4_PS` | γ4γ5 ← γ5 | `S12` `S23` `S31` | iγ2γ1, iγ3γ2, iγ1γ3 at both ends |
| `PS_A4` | γ5 ← γ5γ4 | `A4_A4` | γ4γ5 ← γ5γ4 |

All but `A4_A4` were checked against Bridge++; `A4_A4` is not in Bridge++ and has no
reference values from another program.

### SIMD

With `backend : simd` the fields are kept in blocks of `simd_lanes` sites, and the operator
and the solvers work on whole blocks. In double precision 4 lanes fill a 256-bit register
(AVX2) and 8 a 512-bit one (AVX-512); the program must have been built for a processor that
has them (`target-cpu=native`, above), otherwise the blocks are computed with ordinary
instructions. The results are those of `cpu` up to the order of the sums in the solver. The
lattice is cut into that many equal parts by halving it, t first, then z, y, x; with
`Even_odd` the parts must have even extents, so a 4×4×4×8 lattice allows up to 8.

Whether blocks are faster depends on the machine: the kernels are limited by the memory
they move before they are limited by arithmetic.

### Several processes

The MPI version is a second program, `ruttice-mpi`, built when the feature `mpi` is turned
on. Building it needs an MPI installation (Open MPI, MPICH or Intel MPI) and libclang:

```bash
cargo build --release --features mpi
mpirun -np 4 target/release/ruttice-mpi run/wilson_pion.yaml
```

It reads the same parameter files and prints the same table. `Main.grid_size` is the number
of processes in x, y, z, t; the product must be the number of processes, and each number
must divide the lattice extent in its direction. Without it the lattice is split
automatically. With `Even_odd`, the part of the lattice on each process must have even
extents; with `simd`, it is that part that is cut into blocks.

`mpirun` must come from the same MPI installation the program was built with. If it does
not, every process believes it is alone, and the program stops with a message saying that
the process grid needs more processes than there are. After switching to another MPI
installation, run `cargo clean -p mpi-sys` before building again.

Every process reads the whole configuration file and keeps its part, so this is meant
for files that fit in memory. The results agree with a single process up to rounding.

### Generating configurations

`ruttice` given an HMC file runs trajectories and saves configurations. The actions: the
plaquette and rectangle (Iwasaki, …) gauge actions; two flavours of Wilson, clover or
domain-wall fermions, on all sites or on the even sites; their ratio to Pauli-Villars
fermions; and fractional powers of the determinant by a rational approximation (RHMC), with
the coefficients computed by the program. The links the fermions see can be stout smeared.
The integrator is leapfrog or Omelyan on several time scales; the force solves can start
from the solutions before them.

The momenta, the pseudofermions and the Metropolis test use the random numbers of
Bridge++, in its order, so a run of Bridge++'s `HMC_General` can be repeated number by
number.

### The gauge configuration file

| `gauge_config_type_input` | File |
|---|---|
| `Text` | plain text with one real number per line, the format Bridge++ calls `Text` |
| `ILDG` | the ILDG format: a LIME container with the records `ildg-format` and `ildg-binary-data` |
| `SciDAC` | a lattice-colour-matrix record written by the SciDAC QIO library: a LIME container with `scidac-binary-data` |
| `HDF5` | the dataset `gauge` of an HDF5 file, double precision, of shape `[Lt, Lz, Ly, Lx, 4, 3, 3, 2]`; needs a program built with `--features hdf5` and the HDF5 library |

All four keep the numbers in the same order, from the outermost loop to the innermost:

1. site, with x running fastest: `site = x + Lx * (y + Ly * (z + Lz * t))`
2. direction mu = x, y, z, t
3. the 3×3 link matrix row by row, real part then imaginary part of each entry

A file of the wrong length or for another lattice is rejected. The binary formats hold
big-endian reals in single or double precision; `ILDG` and `SciDAC` are read by the same
code, which accepts either kind of binary record and verifies the SciDAC checksum when the
file has one.

The LIME and HDF5 readers and writers were written from the descriptions of the formats and
have not yet been tried on files from other programs; the HDF5 layout is this program's own.
