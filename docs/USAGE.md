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
| `Main` | `lattice_size`; `grid_size` (processes per direction); `backend` (`cpu`, `simd` or `cuda_bridge`) and `simd_lanes` |
| `Gauge` | `config_filename_input` and `gauge_config_type_input` (`Text`, `Binary`, `NERSC`, `ILDG`, `SciDAC`, `HDF5`) |
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

An HMC file has the sections of Bridge++'s HMC files (`HMC_General`, `Builder_Integrator`,
`Action_G`) and, where Bridge++ fixes the actions in the test program, lists them:
`Action_F_1`, `Action_F_2`, … each with its `Fopr`, its level and its `Solver`. `Main`, `Fopr`
and `GaugeSmearing` are the sections of the propagator files. See the two examples.

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

### Eigenvalues

A file with an `Eigensolver` section in place of `Solver` prints eigenvalues of the fermion
operator instead of propagators (`Main`, `Gauge`, `GaugeSmearing` and `Fopr` as above). The keys
are those of Bridge++'s `Eigensolver`:

| Key | Values |
|---|---|
| `eigensolver_type` | `IRLanczos` (thick-restart Lanczos; Hermitian modes) or `IRArnoldi` (Krylov-Schur; any) |
| `fermion_mult_mode` | `D`, `Ddag`, `DdagD`, `DDdag`, `H` (γ5 D; for domain-wall fermions Γ5 D, Γ5 = γ5 R, Hermitian for `coefficient_c: 0` only) |
| `operator_type` | `Lexical` (default: D on the whole lattice, five-dimensional for domain-wall fermions); for domain-wall fermions `PV_right` or `PV_left`, the even-odd Pauli-Villars operator Ã |
| `eigensolver_mode` | `ascending`, `descending` (by the real part with Arnoldi), `abs_ascending`, `abs_descending` |
| `number_of_wanted_eigenvectors` | Nk |
| `number_of_working_eigenvectors` | Np: the Krylov space holds Nk + Np vectors |
| `number_of_kept_vectors` | kept at a restart, Nk + Np/2 by default (not in Bridge++) |
| `convergence_criterion_squared` | converged when \|A x − λ x\|² is below it |
| `maximum_number_of_iteration` | restarts |
| `threshold_value` | stop once the eigenvalues up to the first one at or beyond it have converged |
| `seed` | of the random start vector |
| `Chebyshev` | Lanczos on T_d of the operator, small on [`threshold_value`, `upper_bound`] (`degree_of_polynomial`, even); for `abs_ascending` on H², on [t², u²], as Bridge++'s `Fopr_Chebyshev` on D†D |

The magnitudes of an indefinite H come from Lanczos on H² followed by the Rayleigh-Ritz
procedure of H. The table has λ (Re and Im with Arnoldi) and \|A x − λ x\|² of each pair; with
`Output.file` it is also written there. `examples/eigen_clover.yaml` is Bridge++'s test
`Eigensolver` (expected result 0.15203194923405; Ruttice 0.15203194923408772);
`examples/eigen_mdwf_arnoldi.yaml` is Ã of Möbius fermions by Arnoldi. CPU only so far; the
eigenvectors are not written.

### Threads

A run uses every core of the machine unless told otherwise, by `Main.cpu_thread` in the
file or by the environment variable
`RAYON_NUM_THREADS`; the file wins. With MPI this is the number of threads of each process, so
processes × threads should not exceed the cores. The kernels are limited by memory bandwidth:
more threads than a few per memory channel gain little.

### GPU

With `backend : cuda_bridge` the solve runs on an NVIDIA GPU, with the CUDA kernels extracted
from Bridge++ (`cuda_bridge/`). The program must be built with the feature, which needs the CUDA
toolkit and cmake:

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda-bridge
```

(`RUTTICE_CUDA_ARCH` is the list of architectures the kernels are compiled for, as cmake takes
it; the default is `80;86`, the A100 and A30 and the RTX 30 series. One architecture builds
faster, e.g. `RUTTICE_CUDA_ARCH=86`.)
Wilson and clover fermions take the solvers and operator types of the CPU; domain-wall fermions
the outer solvers and operator types of the CPU, without a preconditioner, with `Chebyshev` or
with the multigrid, which is on the device with its setup: above the coarse lattice in single
precision as on the CPU (`MultiGrid.fine_precision : double` for double; the outer solver is in
double precision either way), its prolongation kept in half precision as on the CPU
(`basis_precision : single` for single, on the device only), which is most of its memory. `Setup.filter_number_in_block` vectors are filtered at a
time: fewer take less memory. The levels below the coarse lattice (`MultiGrid_Level2` and on,
`levels_for_pauli_villars`) are on the device too, the last of them inverted densely by
cuSOLVER (any size that fits in memory), and so is the exact adjoint of the second factor
(`Coarse_Solver_PV` other than `MINRES`).
With `RUTTICE_CHECKSUM=1` in the environment a run on the device prints checksums of what its
stages made (the test vectors, P, the coarse operators, the solutions), to compare two runs bit
by bit. HMC files with `backend : cuda_bridge` run the domain-wall Standard (Lexical) and Ratio
(Lexical or Even_odd) actions on the device, the Ratio action with the multigrid too (its test
vectors, P in half precision, the coarse operators and the cycles there, and the cache of
`MultiGrid_HMC.rebuild_iteration_growth`), and mixed-precision CG with the operator in single
precision on the device. Not on the device yet: the
multigrid of Wilson and clover fermions, several processes, boundary phases other than ±1 and 0.

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

The `Solver` section of a fermion action says how its forces and energies are solved:

| `Solver` key | Values |
|---|---|
| `solver_type` | `CG` (the default): CG on D†D. `BiCGStab`, `FGMRES`, `GCR`: D† y = φ and then D x = y; Wilson and clover fermions only. A `Rational` action without it uses multi-shift CG; with `CG` one solve per pole; with the others the poles through γ5 D: D†D + σ = γ5 (D + i√σ γ5) γ5 (D − i√σ γ5) |
| `operator_type` | what the solver works on, as in the propagator files; only what the action solves is accepted: `Lexical` (on all sites), `Even_odd` (Wilson and clover, `operator_type : Even_odd` of the action), `Normal_left` (the even-odd domain-wall Ratio), `PV_right` (the domain-wall multigrid) |
| `preconditioner_type` | `Chebyshev`, with `CG`, for any fermions (`Chebyshev: {number_of_steps: 8, interval_ratio: 30}`, a polynomial of D†D). `MG`, with `BiCGStab`, `FGMRES` or `GCR` for Wilson and clover fermions, with `GCR` for the domain-wall Ratio action on all sites over Pauli-Villars fields |
| `precision` | `double` (the default) or `mixed`: CG in single precision on the same operator, corrected in double on the residual of D†D (K†K even-odd) until the tolerance; domain-wall fermions, the Standard (Lexical) and Ratio actions, CG without a preconditioner, on the CPU and on the device |
| `convergence_criterion_squared_inner` | `mixed`: where each CG in single precision stops (1e-10) |
| `number_of_initial_guess` | `Standard`: start the force solves from the last n solutions |
| `number_of_kept_directions` | the cycle of FGMRES, the directions GCR keeps (32) |
| `convergence_criterion_squared`, `convergence_criterion_squared_H`, `maximum_number_of_iteration` | the stopping rule of the force solves and of the energy solves |

`Fopr.neff_alpha` solves domain-wall fermions with Neff's D Q at that α (CG only).

With `preconditioner_type : MG` the action has a `MultiGrid` section beside its `Solver`: the
`MultiGrid` section of the propagator files, the same keys and defaults (`MultiGrid_Level1`
with `block_size`, `number_of_test_vectors` and the coarse solvers, `Setup`, `Smoother`), so a
multigrid tuned on propagators is copied over as it is. The domain-wall coarse solves are in
single precision unless `coarse_precision : double`, as in the propagator files; the rest of
the cycle is in double precision unless `fine_precision : single`, which is on the device only
(the cycle on the operators in single precision, the outer GCR in double). The levels below
the coarse lattice, `basis_precision` and `Setup.checkpoint` are not in HMC yet and are
refused. What only HMC has is in `MultiGrid_HMC`:

| `MultiGrid_HMC` key | Values |
|---|---|
| `refresh_interval` | improve the test vectors again after this many force solves (0: when a trajectory starts) |
| `reuse_vectors`, `number_of_reuse_iterations` | Wilson and clover: start each setup but the first from the vectors before, improved this many times (`false`: from random vectors) |
| `convergence_criterion_squared_inner` | domain-wall fermions: where each of the two solves on the even sites stops (1e-10), before they are repeated on the residual of D†D |
| `dual_with_gamma5` | domain-wall fermions: make the P of the cycle of Ã† (the solves of D†) from the test vectors y and Γ5 y instead of sharing that of Ã: twice the vectors, four times the coarse operators; for when the solves of D† stall (`false`) |
| `rebuild_iteration_growth` | domain-wall fermions: keep P and the coarse operators (and the tops of the smoothers) from solve to solve, the cycles made from them for the links of each solve, and make them again when the GCR iterations per solve have grown to this many times those after the last rebuild (above 1, e.g. 1.5); left out: made for every solve. The test vectors are improved, and everything made again, as `refresh_interval` says |

The multigrid keeps its test vectors along a trajectory and makes the cycle of every solve
from them for the operator on the links of that solve. The domain-wall multigrid solves
(D†D)⁻¹ as D† through the Pauli-Villars operator with D and D_PV exchanged and D through Ã,
both by GCR on the even sites. Without the multigrid, domain-wall fermions are solved by CG:
BiCGStab and GMRES do not converge on D itself.

The momenta, the pseudofermions and the Metropolis test use the random numbers of
Bridge++, in its order, so a run of Bridge++'s `HMC_General` can be repeated number by
number.

### The gradient flow

`ruttice-flow` (`ruttice-flow-mpi` on several processes, built with `--features mpi`) runs
the Wilson flow on saved configurations, for the history of a Monte Carlo chain:

```bash
ruttice-flow 24 24 24 48 binary conf_*.bin              # --step 0.01 --t-max 10 by default
mpirun -np 4 ruttice-flow-mpi 32 32 32 32 binary conf_*.bin
```

For every file it prints the plaquette, t0/a² (where t² E = 0.3, E of the clover field
strength) and the topological charge there, of the clover field strength and improved with
rectangles. The flow (Lüscher's third-order Runge-Kutta), E and Q follow the conventions of
Bridge++ and agree with its tests.

An HMC file measures the same along the run with a `GradientFlow` section; the chain goes on
from the configuration, the flow works on a copy:

```yaml
GradientFlow:
  measurement_interval : 10        # after every 10 trajectories
  step_size            : 0.01      # the default
  max_flow_time        : 10.0      # the default: give up on t0 there
  skip_trajectories    : 200       # thermalization: measured, left out of the analysis
```

Each measurement is a line `# flow N  ... t0/a^2 t Q_clover Q_imp`. At the end of the run
(and of `ruttice-flow`, over its files in the order given) the series of t0, Q_clover and Q_imp
get the Γ-method of Wolff (Comput. Phys. Commun. 156 (2004) 143, S_τ = 1.5): the mean, its
error with the autocorrelations, and τ_int with its error, in trajectories (in files for
`ruttice-flow`). "window not found" means the series is too short for τ_int. A chain is
thermalized from where t0 and Q no longer drift: leave out the trajectories before that and
several τ_int more, by `skip_trajectories` (for `ruttice-flow`, `--skip N` files). A drift left
in biases the mean and shows up as a long τ_int.

### The gauge configuration file

| `gauge_config_type_input` | File |
|---|---|
| `Text` | plain text with one real number per line, the format Bridge++ calls `Text` |
| `Binary` | big-endian double precision reals without a header, the format Bridge++ calls `Binary` |
| `NERSC` | the NERSC (Gauge Connection) format: a text header, then whole matrices (`4D_SU3_GAUGE_3x3`) or their first two rows (`4D_SU3_GAUGE`), single or double precision, either byte order; the checksum is verified. Written as whole matrices in big-endian double precision. Not yet checked against a file of another program |
| `ILDG` | the ILDG format: a LIME container with the records `ildg-format` and `ildg-binary-data` |
| `SciDAC` | a lattice-colour-matrix record written by the SciDAC QIO library: a LIME container with `scidac-binary-data` |
| `HDF5` | the dataset `gauge` of an HDF5 file, double precision, of shape `[Lt, Lz, Ly, Lx, 4, 3, 3, 2]`; needs a program built with `--features hdf5` and the HDF5 library |

All six keep the numbers in the same order, from the outermost loop to the innermost:

1. site, with x running fastest: `site = x + Lx * (y + Ly * (z + Lz * t))`
2. direction mu = x, y, z, t
3. the 3×3 link matrix row by row, real part then imaginary part of each entry

A file of the wrong length or for another lattice is rejected. The LIME formats hold
big-endian reals in single or double precision; `ILDG` and `SciDAC` are read by the same
code, which accepts either kind of binary record and verifies the SciDAC checksum when the
file has one.

The LIME and HDF5 readers and writers were written from the descriptions of the formats and
have not yet been tried on files from other programs; the HDF5 layout is this program's own.
