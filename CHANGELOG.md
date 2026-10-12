# Changelog

## 0.1.0 (2026-10-12)

The first tagged version.

### What is in it

- **Fermions.** Wilson, clover and Möbius domain-wall operators, on all sites and in
  even-odd form; fields in the usual layout or in blocks of 2, 4 or 8 sites for SIMD; on one
  process or on several (MPI).
- **Solvers.** CG, BiCGStab, FGMRES, GCR, MINRES, multi-shift CG; mixed precision.
- **Multigrid.** For Wilson and clover fermions (SAP or GMRES smoother), and for domain-wall
  fermions on the even sites (the Pauli-Villars preconditioned operator, a factorized
  coarse correction, Chebyshev smoother, levels below the coarse lattice). The coarse
  systems are solved on the even coarse sites.
- **Measurements.** Meson two-point functions in fourteen channels for any pair of quarks;
  the pion correlator and the residual mass of domain-wall fermions; point and smeared
  sources; Coulomb gauge fixing; stout and APE smearing of the links.
- **Generating configurations.** HMC with the plaquette and rectangle gauge actions; two
  flavours of Wilson, clover or domain-wall fermions, on all sites or on the even sites;
  ratios to Pauli-Villars fermions; RHMC with rational approximations computed by a Remez
  iteration; leapfrog and Omelyan integrators on several time scales; initial guesses from
  earlier solutions; saving configurations on several processes.
- **Input and output.** Gauge configurations as Bridge++ text and binary, NERSC, ILDG, SciDAC and HDF5.
- **Solvers of HMC.** CG, BiCGStab, FGMRES and GCR for the two-flavour actions, multi-shift CG
  or CG pole by pole for RHMC; Chebyshev and (Wilson, clover) multigrid preconditioners; Neff's
  α for domain-wall fermions.
- **Threads.** `cpu_thread` in a parameter file, or `RAYON_NUM_THREADS`.
- **Checked against Bridge++**: operators, solver iteration counts, correlators, and one
  trajectory of each of its HMC tests.

### Changed since the last state of `main` (0.0.1, 2026-10-10)

- **Parameter files.** Solves are described in the sections of Bridge++ only (`Main`,
  `Gauge`, `Fopr`, `Solver`, `Source`, `MultiGrid`, …). The earlier format (`lattice`,
  `[solver]`, `[[quark]]`) is no longer read; what it could say that the Bridge++ sections
  could not has been added to them: `Quark_2` … `Quark_4`, `GaugeFixing`, `Measure`,
  `Output`, the `Exponential` source, Dirichlet boundaries, the `SciDAC` and `HDF5` formats.
  HMC files keep their own format in this version.
- **Even-odd or lexical is one choice** (`Solver.operator_type : Even_odd | Lexical`), for
  the fine lattice and the coarse lattice of the multigrid alike. `Full` is still read for
  `Lexical`.
- **Names.** In the library the marker of a field on all sites is `Lexical` (it was
  `Full`). Files are named by their role: under `fermion`, every fermion has `mod.rs`,
  `site.rs`, `lexical.rs`, `even_odd.rs`; `lattice/for_sites.rs`; the polynomial
  preconditioners are under `solver/preconditioner`; the tests are named by subject
  (`tests/wilson.rs`, `tests/mdwf.rs`, …) instead of `stage` numbers.
- **Speed of the domain-wall operator.** The even-odd Pauli-Villars operator is five passes
  over the field instead of ten: the factor in the fifth dimension, the inverse of the
  site-local part and the difference with the input are done in the sweeps of the hopping
  term, and large outputs are written past the cache.
- **Speed of the multigrid.** Coarse systems are solved on the even coarse sites (the Schur
  complement of the coarse operator); inner solves no longer compute a first and a last
  residual.
- **HMC.** One driver (`Hmc`); the earlier quenched-only one is gone.
- **Repository.** The study programs of the domain-wall multigrid left `examples/` (tag
  `mdwf-mg-studies`); the working notes are on the branch `notes`.

### Known limits

- No GPU backend.
- The multigrids were developed and measured on 12⁴ and 16⁴ lattices.
- The LIME and HDF5 files have not been exchanged with other programs.
