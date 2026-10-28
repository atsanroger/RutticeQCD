# Reproducing the MG note

How the numbers of `MDWF_MULTIGRID_NOTE.pdf` were produced and how to extend them. Table, figure
and section numbers are those of the note as of 2026-10-28.

## Build

RutticeQCD, branch `cuda-bridge`; Rust (stable) and, for the GPU, CUDA 12:

```
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda-bridge
./target/release/ruttice mg_24.yaml
```

The GPU backend runs on one GPU. Parameter files are in the style of Bridge++ (`docs/USAGE.md`);
configurations are read in the Bridge++ text and binary formats, NERSC, ILDG and SciDAC.

On the device the fine operators, the prolongation, the coarse operators and the coarse solvers stay
on the GPU. P is stored in half precision, the cycle above the coarse lattice runs in single
precision, the outer solver in double. Coarse solvers: BiCGStab for the quark factor, MINRES on H_c
for the Pauli–Villars factor, both to |r|²/|b|² < 1e-2 on the even coarse sites.

## Where each result comes from

Data files are in `data/`. A blank entry is not recorded.

| result | commit | program | data |
|---|---|---|---|
| Tables 4, 5 (mass scan) | `6d5bf18` (configuration 1), `e044050` (configurations 2–4) | `ruttice`, `runs/mg_24.yaml`, `runs/cgne_24.yaml` | `massscan_24.tsv` (configuration 1), `massscan_24_configs.tsv` (all four; configurations 2–4 at `e044050`, runs `massscan_newconf`, `multiconf/p1`) |
| Fig. 4 | `6d5bf18`, `e044050` | `figures.py` | `massscan_24_configs.tsv`, `baseline_mass_2448.tsv` (configuration 1) |
| Table 6 (point source) | `6d5bf18` | `ruttice` | `pointsource_24.tsv` (configuration 1), `pointsource_24_configs.tsv` (configurations 2–4, m 0.001, `e044050`) |
| Table 7 (volume) | `e044050` | `ruttice` | `baseline_volume.tsv`, `baseline_mass_2448.tsv` |
| Table 8 (D†) | | example `dagger_precond` | |
| Table 9 (no MG) | `e044050` | `ruttice` | `baseline_mass_2448.tsv` (all seven masses), `baseline_volume.tsv` |
| Table 10 (lifts) | `e044050`, branch `exp/basis-lifts` | `ruttice` with `RUTTICE_BASIS_LIFTS` (default = `e044050` bit for bit) | `basis_12.tsv`, `basis_16.tsv`, `basis_2448.tsv` (right form on 12⁴, 16⁴) |
| Table 11 (Γ5-Hermiticity) | `e044050` | example `coarse_hermiticity` (CPU, 8 threads; run dir `multiconf/herm`) | `herm_12_configs.tsv` |
| Table 13 (sensitivity) | `e044050` (`fflowHPC_multiconf`, yaml rebuilt from `scan12/*_base.yaml`) | `ruttice` | `sensitivity_12_configs.tsv` (four configurations; the earlier single-configuration run is `sensitivity_12.tsv`, `6d5bf18`) |
| Table 14 (costs) | `e044050` | `ruttice` | `sensitivity_12_configs.tsv` (setup and solve seconds) |
| Sec. V.C.4, 16 and 24 vectors on 24³×48 | `6d5bf18` | `ruttice`, point source | |
| Table 15 (field of values) | | `preconditioned_fov_device --parts` | |
| Table 16 (spectrum) | | `preconditioned_fov_device --arnoldi --far 1,0` (`--left` for Ã_L) | `spectrum_*.tsv` |
| Table 17, Fig. 5 (pseudospectra) | | `preconditioned_fov_device --pseudo` | `pseudo_*.tsv`, `spectrum_*.tsv` |
| Table 21, Fig. 6 (GCR history) | a build before `6d5bf18` | `ruttice` | `gcr_steps_24.tsv` |
| Appendix D, CG on K†K and S†S (Table 24, Fig. 8) | `e044050` + branch `exp/schur-cgne`, not yet in a commit | `ruttice` with `RUTTICE_CGNE_SCHUR=1` for S†S, double | `cgne_form.tsv`; `figures.py` |
| Table 19 (slow setups), Sec. V.D.4 | `e044050` (`fflowHPC_multiconf`) | `ruttice`, `MultiGrid.Setup.seed` 1–40, `Solver.number_of_kept_directions`, `number_of_test_vectors` 48, `Coarse_Solver_D` 1e-4, setup B from `scan12/m18_base.yaml` | run dir `ruttice_a30_20261016/multiconf/p1`: `tail_*`, `more_*`, `fix_*`, `tailB_*`, `lsseed_*`, `ctls_*`, `ct24_*`, `fix24_*`, `dg_*` (histories, point source, right form), `nost_*` (no smearing), `s16_*` (16⁴, dynamical = `therm16/conf16_m0.01_000102.bin`), `s24_*`; not yet in `data/` |
| Table 20 (stall) | not yet in a commit | `preconditioned_fov_device --gcrtest` | `gcr_stall.tsv` |
| Table 23 (configurations) | | `ruttice-flow` | |
| Eq. (3.8) checks (Sec. III.A.3) | | example `correction_exactness` (CPU) | |
| Eqs. (2.10), (2.13) | | example `mobius_gamma5_check` (CPU) | |
| D₋ inverse (Sec. IV.D), dual-cycle prolongations | | examples `dminus_cost`, `dual_basis` | |

Builds add the device reductions in different orders. Against `6c7ff42` the multigrid iterations
of Table 4 differ by one or two in six of the 28 runs, and those of Table 13 differed by up to six
with an earlier build; whether a code change or the order causes the first is not established.

The eigensolvers (thick-restart Lanczos, Krylov–Schur Arnoldi with left eigenvectors, σ_min(z−X)
by Lanczos on (z−X)†(z−X)) are in the module `eigen` and are checked against Bridge++'s
eigensolver test.

## The mass scan

`runs/mg_24.yaml` (multigrid) and `runs/cgne_24.yaml` (double or mixed CGNE); the comments say how
to switch to setup B and to a point source. Each run prints

```
seed 91: GCR 28 iterations on the device; 3014 coarse applications;
    worst |Dx-b|^2 2e-18; 13.2 s; ... device 19.91 GB
```

and, for the multigrid, `setup on the device: 206 s`. One configuration takes about 50 minutes for
the multigrid (seven masses, both setups, α = 0.6) and under half an hour for the two CGNE scans.

Memory: on 24³×48 the multigrid needs 19.9 GB of device memory (CGNE 6.9 GB double, 7.2 GB
mixed), roughly proportional to the volume: 32³×64 about 63 GB. The 4⁴ aggregates must divide the
lattice, and each level below must divide the extent of the level above (`MultiGrid_Level2`,
`_Level3`; on 32³×64, super-blocks 2 and 2).

## Spectral quantities

```
cargo run --release --features cuda-bridge \
    --example preconditioned_fov_device -- \
    16 16 binary conf.bin 1.0 0.001 1 --ratio 10 \
    --arnoldi --far 1,0 --pseudo --contour 1,0,0.6,16
```

(arguments: L, T, format, file, M5, m, stout 0/1). It fits 16⁴ in 24 GB and takes about an hour;
24³×48 does not fit. The CPU examples `preconditioned_fov`, `coarse_hermiticity` and
`correction_exactness` need no GPU:

```
cargo run --release --example correction_exactness -- \
    12 binary conf.bin 1.0 0.001 1 --block 4
```

prints ‖e(r)−y‖/‖y‖ for the right, left and dual corrections and the GCR(2) iterations of Ã_R and
Ã_R† (setup S smoother by default, `--steps 12 --ratio 30` for setup B; about 12 minutes on 12⁴ on
a desktop CPU).

## What would help most

1. The mass scan on three to five further 24³×48 configurations, for the spread of the iterations
   and of the speed-up.
2. The same on 32³×64 (an 80 GB GPU), for the volume dependence of the iterations and the setup.
3. The mass scan on a dynamical 2+1-flavor ensemble, e.g. that of KCM2024.
4. The spectral run on further 16⁴ configurations, for how the outliers of setup S and the
   pseudospectra vary.

For each run, the result lines above and the commit of the build are enough.
