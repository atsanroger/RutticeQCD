# RutticeQCD

```text
 ____          _    _    _               ___    ____  ____
|  _ \  _   _ | |_ | |_ (_)  ___   ___  / _ \  / ___||  _ \
| |_) || | | || __|| __|| | / __| / _ \| | | || |    | | | |
|  _ < | |_| || |_ | |_ | || (__ |  __/| |_| || |___ | |_| |
|_| \_\ \__,_| \__| \__||_| \___| \___| \__\_\ \____||____/
```

Lattice QCD in Rust.

## Introduction

Ruttice is a lattice QCD code written in Rust. Its idea is to use what Rust offers as far
as it goes to make the solvers and HMC fast, with the aim of simulations in pure Rust on
every kind of hardware.

- **Fermions**: Wilson, clover and Möbius domain-wall.
- **Solvers**: a multigrid written for each kind, to compute two-point correlation
  functions quickly.
- **Configurations**: pure gauge, HMC and RHMC.
- **GPUs**: the CUDA kernels of Bridge++ behind `backend : cuda_bridge` (propagators; no
  multigrid and no HMC on the device yet). A backend in Rust is planned once CUDA is officially
  supported there, keeping both.

The code is written for a clear structure and for speed, at some cost in readability where
the two pull apart: fused kernels, fields laid out for SIMD, low-level control of memory.

Everything it computes is checked number by number against
[Bridge++](https://bridge.kek.jp/Lattice-code/).

## Quick start

You need a stable Rust toolchain.

```bash
git clone https://github.com/atsanroger/RutticeQCD.git
cd RutticeQCD
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run an example. It reads `examples/conf_04040408.txt`, the 4³ × 8 test configuration of
Bridge++, which comes with the repository:

```bash
target/release/ruttice examples/wilson_pion.yaml
```

```text
# Wilson: κ = 0.12
# CG on Even_odd (the Schur complement on the even sites, CGNE), to |r|²/|b|² < 1e-28
# plaquette  0.625129946974942
# quark 1    40 iterations, 0 restarts, worst |D S - b|^2/|b|^2 = 8.835e-29, 0.119 s
# quarks channel t re im
11 PS 0 1.422503945020827e1 0.000000000000000e0
11 PS 1 6.530367660173256e-1 0.000000000000000e0
...
```

## Usage

A parameter file has the sections and keys of a Bridge++ parameter file (YAML or TOML).
Start from an example; each one is commented.

| Example | What it runs |
|---|---|
| [wilson_pion.yaml](examples/wilson_pion.yaml) | Wilson fermions, pion correlator |
| [clover_two_flavours.yaml](examples/clover_two_flavours.yaml) | clover fermions, two quarks, smeared sources, every channel |
| [clover_mg.yaml](examples/clover_mg.yaml) | clover fermions with the multigrid |
| [mdwf_pv_eo_mg.yaml](examples/mdwf_pv_eo_mg.yaml) | domain-wall fermions with the multigrid |
| [hmc_domainwall_pv.yaml](examples/hmc_domainwall_pv.yaml) | HMC, two flavours of domain-wall fermions |
| [hmc_clover_rhmc_nf2p1.yaml](examples/hmc_clover_rhmc_nf2p1.yaml) | HMC, 2 + 1 flavours of clover fermions (RHMC) |

All of them run as they are on the configuration in `examples/`, except the domain-wall
multigrid, which wants a 12⁴ configuration of your own (`Gauge.config_filename_input`).

[docs/USAGE.md](docs/USAGE.md) has every section of the file.

### Backends

`Main.backend` in the file chooses how the kernels are computed:

| Backend | How |
|---|---|
| `cpu` | the default: one site at a time |
| `simd` | `simd_lanes: 4` (2, 4 or 8) sites in one SIMD register; build with `target-cpu=native` |
| `cuda_bridge` | on an NVIDIA GPU, with the CUDA kernels of Bridge++; a program built with `--features cuda-bridge` |

### Parallelism

Either backend runs on threads, on several processes, or both:

| | How |
|---|---|
| Threads | every core by default; `Main.cpu_thread` in the file or `RAYON_NUM_THREADS` sets how many |
| MPI | build with `--features mpi`, run `ruttice-mpi`; `Main.grid_size` is the process grid |

```bash
cargo build --release --features mpi        # needs an MPI installation and libclang
mpirun -np 4 target/release/ruttice-mpi run/wilson_pion.yaml
```

### Which solver

| Fermions | Solvers | Preconditioners |
|---|---|---|
| Wilson, clover | CG, BiCGStab, FGMRES; on all sites (`Lexical`) or on the even sites (`Even_odd`) | multigrid |
| Möbius domain-wall | GCR, block GCR, BiCGStab, FGMRES, CG; always on the even sites | multigrid, Chebyshev polynomial |

What to choose for propagators. The lattices tried so far go up to 16⁴ on one machine; the
rows marked *expected* have not been measured:

| Fermions | Lattice | Choice |
|---|---|---|
| Wilson, clover | small | `CG` on `Even_odd` |
| Wilson, clover | large | `FGMRES` or `BiCGStab` with the multigrid (*expected*) |
| Domain-wall | up to 12⁴ | `CG` on `Normal_left` with `neff_alpha : 0.6` |
| Domain-wall | 16⁴ | `GCR` or `FGMRES` with the multigrid |
| Domain-wall | larger | the multigrid (*expected*) |

In HMC the default is CG on D†D for two flavours and multi-shift CG for RHMC. Wilson and
clover fermions also take BiCGStab, FGMRES or GCR, with the multigrid; domain-wall fermions
take CG only, with Neff's α. [docs/USAGE.md](docs/USAGE.md) has the keys.

Test gauge configuration could be download from [here](https://drive.google.com/drive/folders/1-OUysaybUd8ViKQbliD7gu7_wPPNpM4S?usp=drive_link)

### The domain-wall multigrid

- It works on the **Pauli-Villars preconditioned operator** of Kanamori, Chen and
  Matsufuru, PoS(LATTICE2024)414, in its even-odd form (`operator_type : PV_right` or
  `PV_left`). The multigrid here is built for that operator.
- **Neff's α** is supported (`neff_alpha`): the better conditioned form of the domain-wall
  operator of Neff, [arXiv:1501.04950](https://arxiv.org/abs/1501.04950), studied for the
  Möbius operator by Chen, Kanamori, Matsufuru and Neff,
  [arXiv:2602.14454](https://arxiv.org/abs/2602.14454). It speeds up `CG` on `Normal_left`
  and the solves on `PV_left`, with or without the multigrid.

## More

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | the parameter files and the ways of running |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | operators, gamma matrices, correlators: for comparing numbers |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | the structure of the code, the tests, the library, what is not there yet |
| [docs/COMPARISON.md](docs/COMPARISON.md) | the comparisons with Bridge++, number by number |
| [CHANGELOG.md](CHANGELOG.md) | what changed |

## Acknowledgements

This project draws heavily on [Bridge++](https://bridge.kek.jp/Lattice-code/): its
algorithms, its conventions, its parameter files and its tests. Much has been changed, as
the Rust language asks for, and the aims differ: Bridge++ is written to be read and to be
general. But without Bridge++ there would be no Ruttice.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE).
