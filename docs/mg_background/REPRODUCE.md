# Reproducing the MG note

How the numbers of `MDWF_MULTIGRID_NOTE.pdf` were produced and how to extend them. Table, figure
and section numbers are those of the note as of 2026-10-31 (the tables of Sec. V are in Appendix E).

## Build

RutticeQCD, the commits below (branch `exp/pv-coarse-spectrum`, merged into `main` on 2026-11-01); Rust
(stable) and, for the GPU, CUDA 12:

```
export CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda-bridge
./target/release/ruttice mg_24.yaml
```

The GPU backend runs on one GPU. Parameter files are in the style of Bridge++ (`docs/USAGE.md`);
configurations are read in the Bridge++ text and binary formats, NERSC, ILDG and SciDAC.

On the device the fine operators, the prolongation, the coarse operators and the coarse solvers stay
on the GPU. P is stored in half precision, the cycle above the coarse lattice runs in single
precision, the outer solver in double. Coarse solvers: BiCGStab for the quark factor on the even
coarse sites, and for the Pauli–Villars factor CG on the regularized normal equation on all coarse
sites, both to |r|²/|b|² < 1e-2:

```
Coarse_Solver_PV:
  solver_type                   : CG
  tikhonov_epsilon              : 1.0e-3
```

## Commits

| commit (branch `exp/pv-coarse-spectrum`) | what |
|---|---|
| `0a124d0` | the regularized solve (`Coarse_Solver_PV: CG`, `tikhonov_epsilon`); every `ruttice` run of the tables |
| `684189d` | `--tikhonov eps` in the examples `dagger_precond`, `preconditioned_fov_device` |
| `8ed045b` | the adjoint cycle with the regularized solve (needed for the field of values and the pseudospectra) |
| `facb62b` | eigensolver mode `ReD`, the Hermitian part (D + D†)/2 |
| `d61fe3c` | `dagger_precond` prints the solve with D of the same right-hand sides |
| `d9ef1a7` | `coarse_hermiticity --bicgstab`: the bootstrap solves the Pauli–Villars factor by BiCGStab (Table 13, coarse) |

Table 10 (lifts) uses branch `exp/basis-lifts-tikhonov` (`36f9079`, `RUTTICE_BASIS_LIFTS` = `st`, `sds`, `s`).

## Where each result comes from

Data files are in `data/` (`data/unused/` holds files of earlier runs that no table uses); run directories under `ruttice_a30_20261016/multiconf/` on the A30 machine.

| result | program | data, run directory |
|---|---|---|
| Table 7 (mass scan), Fig. 4 | `ruttice` | `massscan_24_configs.tsv`, `massscan_24.tsv` (configuration 1); CGNE rows from `cgneT/` (`feat/true-residual-stop` 8ae9255, `update_data.py`); `baseline_mass_2448.tsv` from earlier builds; `rerunT/T_ms_*` |
| Table 4 (point source) | `ruttice` | `pointsource_24.tsv` (configuration 1), `pointsource_24_configs.tsv`; `rerunT/T_ps_*` (m 0.001), `psT/` (m 0.01); correlators against the CGNE logs of `p1/t5cg_*` and `final_fbc8f00/pointsource/` |
| Table 8 (volume), Fig. 5 | `ruttice` | `volume_configs.tsv` (CGNE from `cgneT/`), `baseline_volume.tsv`; `rerunT/T_vol_*` |
| Fig. 6 (sensitivity), Table 14 (costs) | `ruttice` | `sensitivity_12_configs.tsv`, `costs_12.tsv` (medians over the four configurations); `rerunT/T_sens_*` |
| Table 9 (D†) | example `dagger_precond --tikhonov 1e-3 --pl 0` | `daggerT2/`, `daggerT3/` (the solve with D) |
| Table 10 (lifts) | `ruttice` with `RUTTICE_BASIS_LIFTS` | `basis_configs.tsv`; `liftsT/` |
| Table 12 (with and without the shift) | `ruttice` | `noshift16/` (no shift: `Coarse_Solver_PV` BiCGStab; with the shift: `rerunT/T_vol_S_*`) |
| Table 11 (τ) | `ruttice` | `tikhonov/` (status, status3); rows 1a, 1b, 1c, 2a, 3a, 3b = seeds 2, 5, 6, 10, 3, 10; the 40 seeds (1–10) of Sec. V.C.3: `tikhonov/status2.txt` |
| Table 13 (singular values) | example `coarse_hermiticity --nsv 48` (coarse, CPU); `ruttice` eigensolver, `DdagD`, `Lexical` (fine; the logs print σ²) | coarse: `pvsvB/` (`--bicgstab`: the bootstrap solves the Pauli–Villars factor by BiCGStab, as the no-shift rows of Table 12); fine: `finesigmaA1/fs*` (`neff_alpha 1.0`; the earlier `finesigma*` had 0.6, i.e. D(m=1)Q) |
| Table 15 (field of values) | example `preconditioned_fov_device --parts --tikhonov 1e-3` | `fovT2/parts*` |
| Table 16 (spectrum) | example `preconditioned_fov_device --arnoldi --far 1,0 --tikhonov 1e-3` (`--left` for Ã_L) | `fovT2/far*`; `spectrum_*12.tsv` |
| Table 17, Fig. 7 (pseudospectra) | example `preconditioned_fov_device --pseudo --contour c,0,r,n --tikhonov 1e-3` | `pseudoT/`; `pseudo_*12.tsv` |
| Table 18 (GCR(k)) | `ruttice` | `gcrk12_configs.tsv`; `rerunT/T_gcrk12_*` |
| Table 19 (test vectors) | `ruttice` | `tvscanT/` (status, status_redo) |
| Fig. 8 (GCR history) | `ruttice`, `verbose_level: Detailed` | `gcr_steps_24.tsv`, `gcr_steps_24_configs.tsv`; `rerunT/T_hist_*` |
| Sec. V.C.4, the one GCR(2) plateau | `ruttice` | `diagB10/` |
| Open problems, field of values of the fine D_PV | `ruttice` eigensolver, `fermion_mult_mode: ReD` | `finesigmaA1/reD*` (`neff_alpha 1.0`) |
| Appendix D, CG on K†K and S†S (Fig. 10) | `ruttice` with `RUTTICE_CGNE_SCHUR=1` for S†S, double | `cgne_form.tsv`; `figures.py` |
| Table 6 (configurations) | `ruttice-flow` | |
| Sec. II.B.2, Γ5 D Γ5 against D† and Eq. (2.13) | example `mobius_gamma5_check 12 Binary conf_quench_12x12x12x12_v2.bin 1.0 0.001 1` (CPU) | `g5check/g5check_12_S.log` |

`multiconf/rerunT/extract.py` writes the multigrid rows of the data files from the logs and prints each
change (`extract_diff.txt`); `tables.py` computes the rows of Table 7.

Builds add the device reductions in different orders, so the iterations of otherwise equal runs can
differ by a few. The default setup seed (`Setup: seed`) is 21. `RUTTICE_CHECKSUM=1` prints checksums of the stages of a device run to compare builds.

## The mass scan

The yaml files of `rerunT/T_ms_*` (multigrid) and the CGNE files of the earlier runs. Each multigrid
run prints

```
setup on the device: 203 s
seed 91: GCR 29 iterations on the device; 3958 coarse applications; worst |Dx−b|² 1e-18; 13.9 s, ...
```

Memory: on 24³×48 the multigrid needs 20.3–21.1 GB of device memory (23.0 GB for a point source;
CGNE 6.9 GB double, 7.2 GB mixed), measured with a copy of the adjoint of the coarse Pauli–Villars
operator; `main` from 0bced8a applies it from the operator's blocks instead, about 0.9 GB less
(24³×48, configuration 1, setup S, m = 0.001: device peak of the run 21.06 → 20.11 GB). The 4⁴ aggregates must divide the lattice, and
each level below must divide the extent of the level above (`MultiGrid_Level2`, `_Level3`).

## Spectral quantities

```
cargo run --release --features cuda-bridge \
    --example preconditioned_fov_device -- \
    16 16 binary conf.bin 1.0 0.001 1 --block 4 --vectors 16 --steps 8 --ratio 10 \
    --tikhonov 1e-3 --pseudo --contour 1,0,0.6,16
```

(arguments: L, T, format, file, M5, m, stout 0/1). It fits 16⁴ in 24 GB and takes about an hour;
24³×48 does not fit. Each run prints the adjoint check `<z, A M w>` against `<M† A† z, w>`; it must
agree to about 1e-10, or the field of values and the pseudospectra are not those of a Hermitian part.

## What would help most

1. The mass scan on further 24³×48 configurations, for the spread of the iterations and of the speed-up.
2. The same on 32³×64 (an 80 GB GPU), for the volume dependence of the iterations and the setup.
3. The mass scan on a dynamical 2+1-flavor ensemble.
4. The solves with D† in HMC, with Γ5 M Γ5 and with the exchanged (dual) cycle.
