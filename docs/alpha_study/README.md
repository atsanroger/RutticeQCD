# Neff's α in the multigrid: the runs

12⁴ (`conf_quench_12x12x12x12_v2.bin`), run directory `ruttice_a30_20261016/`:

- `alpha12_mg/`, `alpha12_hist/`: MG solves (`fflowHPC_head`, main e044050), the yamls of
  `basis12/*_left_32.yaml` with the mass and α changed; `alpha12_hist` adds
  `Main.verbose_level : Detailed` for the GCR history.
- `alpha12/`: Arnoldi on Ã_L^(α) without the multigrid (eigensolver yamls).
- `alpha12_fovM/`, `alpha12_lowdefect/`: `examples/preconditioned_fov_device` with
  `--left --alpha A --arnoldi --parts` and `--lowdefect 8 --dim 64`; the options come from
  `alpha12_fovM/fov_alpha_lowdefect2.patch` on main 6b6e91a (not committed).

`python3 -I collect.py <run directory>` writes `data/`, `python3 figures.py` the figure.
