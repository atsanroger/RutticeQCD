//! Möbius domain-wall solves on the even sites, with or without the even-odd
//! Pauli-Villars multigrid (`pv_eo`), from a parameter file in the style of Bridge++
//! (`params`): what `ruttice` runs for `fermion_type: Mobius` (and the example
//! `mdwf_pv_eo_mg`, which can choose the seeds of random right-hand sides).

use super::memory_line;
use super::params::{Backend, Params, PrecisionType, PreconditionerType, SolverType};
use crate::fermion::{FermionOp, Mdwf, PvEvenOdd, Variant};
use crate::io::read_bridge_text;
use crate::kernel::{Dirac, GammaRepre};
use crate::measure::{
    channel, meson_correlator, pion_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use crate::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, Aggregation, CycleLayout,
    PvEoBasis, PvEoCycle, SharedOperator,
};
use crate::solver::{
    largest_magnitude, BiCgStab, Cg, Chebyshev, Fgmres, Gcr, SolveError, SolveInfo, Solver, Stop,
};
use crate::{
    extract_parity, random_spinor, Aos, Aosoa, Comm, ConvertFrom, Field, Gauge, Half, Lattice,
    Lexical, LinearFn, Parity, SingleComm, Spinor, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type H<L = Aos> = Field<f64, Spinor, Half, L>;
type F<L = Aos> = Field<f64, Spinor, Lexical, L>;

/// The first `count` vectors of a low-mode checkpoint, read site by site.
fn load(path: &Path, like: &H, count: usize) -> Result<H, Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u64> {
        f.read_exact(&mut word)?;
        Ok(u64::from_le_bytes(word))
    };
    let _ = next(&mut f)?;
    let m = next(&mut f)? as usize;
    let len = next(&mut f)? as usize;
    let c = count.min(m);
    let mut x = like.zeros_with(c);
    let sites = x.num_sites();
    let reals = like.site_len() / like.num_vectors();
    if sites * m * reals != len {
        return Err("checkpoint for another shape".into());
    }
    for _ in 0..m {
        let _ = next(&mut f)?;
    }
    let mut buf = vec![0u8; 8 * m * reals];
    for site in 0..sites {
        f.read_exact(&mut buf)?;
        for (k, v) in x.site_mut(0, site).iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[8 * k..8 * k + 8].try_into().expect("8 bytes"));
        }
    }
    Ok(x)
}

/// The preconditioner of the outer solver, in the layout `L` of the fine fields.
enum Preconditioner<'a, L: CycleLayout> {
    None,
    Chebyshev(Box<Chebyshev<PvEvenOdd<'a, f64, Dirac, L>, H<L>>>),
    Single(Box<PvEoCycle<'a, Dirac, f32, L>>),
    Double(Box<PvEoCycle<'a, Dirac, f64, L>>),
}

impl<L: CycleLayout> Preconditioner<'_, L> {
    fn coarse_applications(&self) -> usize {
        match self {
            Preconditioner::Single(c) => c.coarse_applications(),
            Preconditioner::Double(c) => c.coarse_applications(),
            _ => 0,
        }
    }

    /// Applications of the operator inside one application of the preconditioner.
    fn inner_applications(&self, steps: usize) -> usize {
        match self {
            Preconditioner::None => 0,
            Preconditioner::Chebyshev(_) => steps - 1,
            // Two smoother applications of steps − 1 each, two residuals.
            _ => 2 * (steps - 1) + 2,
        }
    }

    fn as_fn(&mut self) -> Option<&mut dyn LinearFn<H<L>>> {
        match self {
            Preconditioner::None => None,
            Preconditioner::Chebyshev(c) => Some(c.as_mut()),
            Preconditioner::Single(c) => Some(c.as_mut()),
            Preconditioner::Double(c) => Some(c.as_mut()),
        }
    }
}

/// Counts the applications of the map it wraps.
struct Counted<'a, V> {
    a: &'a mut dyn LinearFn<V>,
    n: &'a Cell<usize>,
}

impl<V: Vector> LinearFn<V> for Counted<'_, V> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

/// Run what a domain-wall parameter file (`Params`) describes: random right-hand sides with seeds
/// `seeds` (inclusive) without a `Source` block, the propagator, the meson correlators and the
/// residual mass with one.
pub fn run(
    params: &Params,
    file: &Path,
    seeds: (u64, u64),
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    run_on(params, file, seeds, out, Arc::new(SingleComm))
}

/// `run` on the processes of `comm`, split as `Main.grid_size` asks (or as chosen here when it is
/// left at one process).
pub fn run_on(
    params: &Params,
    file: &Path,
    seeds: (u64, u64),
    out: &mut dyn Write,
    comm: Arc<dyn Comm>,
) -> Result<(), Box<dyn Error>> {
    let relative = |p: &PathBuf| file.parent().map_or(p.clone(), |d| d.join(p));
    let size = params.main.lattice_size;
    let requested = (params.main.grid_size != [1; ND]).then_some(params.main.grid_size);
    let grid = super::grid_for(size, requested, comm.size())?;
    let lat = Lattice::new(size, grid, comm);
    let u = read_bridge_text(&relative(&params.gauge.config_filename_input), &lat)?;
    // Gauge smearing: the operators are made of the smeared links.
    let u = match &params.gauge_smearing {
        None => u,
        Some(g) => {
            let st = crate::Stencil::nearest(&lat, [crate::Boundary::Periodic; ND]);
            let v = g.to_config().smearing().apply(&u);
            writeln!(
                out,
                "gauge smearing {:?}: plaquette {:.15} → {:.15}",
                g.to_config(),
                crate::gauge::plaquette(&u, &st),
                crate::gauge::plaquette(&v, &st)
            )?;
            v
        }
    };
    match (params.main.backend, params.main.simd_lanes) {
        (Backend::Simd, 2) => run_in::<Aosoa<2>>(params, file, &lat, &u, seeds, out),
        (Backend::Simd, 4) => run_in::<Aosoa<4>>(params, file, &lat, &u, seeds, out),
        (Backend::Simd, 8) => run_in::<Aosoa<8>>(params, file, &lat, &u, seeds, out),
        (Backend::Cpu, _) => run_in::<Aos>(params, file, &lat, &u, seeds, out),
        _ => unreachable!("refused when the file is read"),
    }
}

/// Everything after reading the configuration, with the fine fields in the layout `L`.
fn run_in<L: CycleLayout>(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    (seed_lo, seed_hi): (u64, u64),
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>>
where
    F<L>: ConvertFrom<F>,
    F: ConvertFrom<F<L>>,
{
    let relative = |p: &PathBuf| file.parent().map_or(p.clone(), |d| d.join(p));
    let lat = lat.clone();
    let fo = &params.fopr;
    let make = |mass: f64, alpha: f64| -> Result<Mdwf<f64>, Box<dyn Error>> {
        let mut op = Mdwf::<f64>::new(&lat, params.mdwf_params(mass));
        op.set_gauge(u)?;
        op.set_alpha(alpha);
        Ok(op)
    };
    // The operators in the layout of the fine fields (the outer operator and the smoother).
    let make_l = |mass: f64, alpha: f64| -> Result<Mdwf<f64, Dirac, L>, Box<dyn Error>> {
        let mut op = Mdwf::<f64, Dirac, L>::in_layout(&lat, params.mdwf_params(mass))
            .ok_or("the lattice cannot be cut for the SIMD layout")?;
        op.set_gauge(u)?;
        op.set_alpha(alpha);
        Ok(op)
    };
    let op = make(fo.mass(), 1.0)?;
    let pv_plain = make(fo.quark_mass_pauli_villars, 1.0)?;
    let op_a = make(fo.mass(), fo.neff_alpha)?;
    let pv_alpha = make(fo.quark_mass_pauli_villars, fo.neff_alpha_pauli_villars)?;
    let op_l = make_l(fo.mass(), fo.neff_alpha)?;
    let pv_alpha_l = make_l(fo.quark_mass_pauli_villars, fo.neff_alpha_pauli_villars)?;
    // The plain D for the check of the solution, when the operator carries α.
    let op_check_l = if fo.neff_alpha != 1.0 {
        Some(make_l(fo.mass(), 1.0)?)
    } else {
        None
    };
    let op_check = op_check_l.as_ref().unwrap_or(&op_l);
    // The normal equations are the operator with D in place of D_PV.
    let (form, normal) = params.form();
    let (pv, pv_a, pv_l) = if normal {
        (&op, &op_a, &op_l)
    } else {
        (&pv_plain, &pv_alpha, &pv_alpha_l)
    };
    let ls = fo.ls();
    let mut like = H::with_shape(&lat, 24, ls, 1, 1);
    like.set_parity(Parity::Even);
    let mut like_l = H::<L>::with_shape(&lat, 24, ls, 1, 1);
    like_l.set_parity(Parity::Even);
    let no_map = "the lattice cannot be cut for the SIMD layout";
    let index = op
        .even_odd_index()
        .ok_or("no even-odd decomposition")?
        .clone();

    // The multigrid basis lives as long as the cycle that borrows it.
    let start = Instant::now();
    let s = &params.solver;
    let mg_settings = params.multigrid();
    // With the vectors y it was built from (for the levels below the coarse lattice).
    let basis: Option<(PvEoBasis, H)> = match (&mg_settings, &params.multigrid) {
        (Some((_, low, boot)), Some(mg)) => {
            let keep = mg.level1.test_vectors(true) / 2;
            let mut y = match &mg.setup.checkpoint {
                Some(_) if lat.comm.size() > 1 => {
                    return Err("MultiGrid.Setup.checkpoint is read on one process only".into())
                }
                Some(path) => load(&relative(path), &like, keep)?,
                None => {
                    let (x, theta) = pv_eo_low_modes(&op, pv, &lat, *low);
                    writeln!(
                        out,
                        "  filter: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s)",
                        theta[0],
                        theta[theta.len() - 1],
                        start.elapsed().as_secs_f64()
                    )?;
                    x.select(&(0..keep).collect::<Vec<_>>())
                }
            };
            let agg = Aggregation::new(&lat, mg.level1.block_size);
            let basis = if boot.rounds > 0 {
                pv_eo_bootstrap(&op, pv, &agg, &mut y, *boot)?
            } else {
                pv_eo_basis(&op, pv, &agg, &y)?
            };
            Some((basis, y))
        }
        _ => None,
    };
    let mut coarse_flops = 0.0;
    let (mut pre, steps) = match (s.preconditioner_type, &mg_settings, &basis) {
        (PreconditionerType::Multigrid, Some((cycle_params, _, _)), Some((basis, y))) => {
            let mg = params.multigrid.as_ref().expect("multigrid() found it");
            let dim = basis.coarse_dim() as f64;
            let coarse_sites = basis.aggregation().coarse_lattice().num_sites() as f64;
            coarse_flops = 72.0 * dim * dim * coarse_sites;
            let outer = (&op_a, pv_a, form);
            let map = L::map(&lat).ok_or(no_map)?;
            let mut cycle = match mg.coarse_precision {
                PrecisionType::Single => {
                    Preconditioner::Single(Box::new(PvEoCycle::<Dirac, f32, L>::with_layout(
                        &op,
                        pv,
                        outer,
                        (&op_l, pv_l),
                        map,
                        basis,
                        *cycle_params,
                    )?))
                }
                PrecisionType::Double => {
                    Preconditioner::Double(Box::new(PvEoCycle::<Dirac, f64, L>::with_layout(
                        &op,
                        pv,
                        outer,
                        (&op_l, pv_l),
                        map,
                        basis,
                        *cycle_params,
                    )?))
                }
            };
            let super_blocks = mg.super_blocks();
            if !super_blocks.is_empty() {
                let levels_start = Instant::now();
                let vectors = pv_eo_coarse_vectors(&op, pv, basis, y);
                let applications = match &mut cycle {
                    Preconditioner::Single(c) => {
                        c.add_levels_with(&vectors, &super_blocks, &mg.level_bootstrap())?
                    }
                    Preconditioner::Double(c) => {
                        c.add_levels_with(&vectors, &super_blocks, &mg.level_bootstrap())?
                    }
                    _ => unreachable!(),
                };
                writeln!(
            out,
                    "  levels below the coarse lattice {super_blocks:?} (refinement {:?}): {applications} coarse applications ({:.0} s)",
                    mg.level_bootstrap().iter().map(|b| (b.rounds, b.iterations)).collect::<Vec<_>>(),
                    levels_start.elapsed().as_secs_f64()
                )?;
            }
            (cycle, cycle_params.chebyshev_steps)
        }
        (PreconditionerType::Chebyshev, _, _) => {
            let c = s.chebyshev_settings();
            let mut start_sites = like.zeros_like();
            extract_parity(
                &mut start_sites,
                &random_spinor(&lat, ls, 5),
                &index,
                Parity::Even,
            );
            let mut start_vector = like_l.zeros_like();
            L::from_sites(
                &L::map(&lat).ok_or(no_map)?,
                &mut start_vector,
                &start_sites,
            );
            let hi = 1.1
                * largest_magnitude(
                    &mut PvEvenOdd::new(&op_l, pv_l, form, &like_l),
                    &mut start_vector,
                    50,
                );
            let cheb = Chebyshev::new(
                PvEvenOdd::new(&op_l, pv_l, form, &like_l),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                &like_l,
            );
            (Preconditioner::Chebyshev(Box::new(cheb)), c.number_of_steps)
        }
        _ => (Preconditioner::None, 1),
    };
    writeln!(
        out,
        "  setup: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory_line()
    )?;

    // With the multigrid the outer solver uses the cycle's own operator: one set of its
    // work fields for both.
    let mut a = match &mut pre {
        Preconditioner::Single(c) => c.operator(),
        Preconditioner::Double(c) => c.operator(),
        _ => SharedOperator::new(PvEvenOdd::new(&op_l, pv_l, form, &like_l)),
    };
    let fine_flops = 2.0 * (op.flops_per_site() * lat.num_sites()) as f64;
    let stop: Stop = s.stop();
    let k = s.number_of_kept_directions;
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let mut solve = |b: &F<L>, label: &str, out: &mut dyn Write| -> F<L> {
        let nv = b.num_vectors();
        let mut rhs = like_l.zeros_with(nv);
        a.right_hand_side(&mut rhs, b);
        let (outer, applied) = (Cell::new(0), Cell::new(0));
        let coarse_before = pre.coarse_applications();
        let inner_per = pre.inner_applications(steps);
        let mut sol = rhs.zeros_like();
        let start = Instant::now();
        let result: Result<SolveInfo, SolveError> = {
            let mut counted_a = Counted {
                a: &mut a,
                n: &outer,
            };
            let mut counted_pre = pre.as_fn().map(|m| Counted { a: m, n: &applied });
            let m = counted_pre.as_mut().map(|c| c as &mut dyn LinearFn<H<L>>);
            match s.solver_type {
                SolverType::Gcr => Gcr::new(stop, k).solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::BiCgStab => {
                    BiCgStab::new(stop).solve(&mut counted_a, m, &mut sol, &rhs)
                }
                SolverType::Fgmres => Fgmres::new(Stop {
                    max_iterations: k,
                    max_restarts: stop.max_iterations / k.max(1),
                    ..stop
                })
                .solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::Cg => Cg::new(stop).solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::Minres => unreachable!("refused when the file is read"),
            }
        };
        let seconds = start.elapsed().as_secs_f64();
        let iterations = match &result {
            Ok(i) => format!("{}", i.iterations),
            Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
            Err(e) => format!("failed: {e}"),
        };
        // Applications of the operator, per right-hand side.
        let applications = (outer.get() + applied.get() * inner_per) / nv;
        let coarse = (pre.coarse_applications() - coarse_before) / nv;
        let gflops = (applications as f64 * fine_flops + coarse as f64 * coarse_flops) * nv as f64
            / seconds
            * 1e-9;
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, b);
        let mut r = b.zeros_like();
        op_check.mult(&mut op_check.work(), &mut r, &x, Variant::D);
        r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
        let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        let worst = rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b));
        let _ = writeln!(
            out,
            "  {label}: {} {iterations} iterations; per right-hand side {} preconditioner applications, {applications} applications of the operator = {} Wilson sweeps, {coarse} coarse; worst |Dx−b|² {worst:.0e}; {seconds:.1} s, {gflops:.1} GFlop/s; {}",
            s.solver_type,
            applied.get() / nv,
            2 * applications,
            memory_line()
        );
        x
    };

    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            let mut b = F::with_shape(&lat, 24, ls, 1, 1);
            b.as_mut_slice()
                .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
            let mut b_l = F::<L>::with_shape(&lat, 24, ls, 1, 1);
            b_l.convert_from(&b);
            drop(b);
            solve(&b_l, &format!("seed {seed}"), out);
        }
        return Ok(());
    };

    // The propagator from the point source, `number_of_rhs_per_solve` columns at a time:
    // the four-dimensional quark field and the midpoint field of every column.
    let eta = point_source::<f64>(&lat, src.source_position);
    let (mut prop, mut mid) = (eta.zeros_like(), eta.zeros_like());
    let columns: Vec<usize> = (0..NUM_SOURCES).collect();
    for chunk in columns.chunks(src.number_of_rhs_per_solve) {
        let eta_c = eta.select(chunk);
        let mut eta_l = F::<L>::with_shape(&lat, 24, 1, chunk.len(), 1);
        eta_l.convert_from(&eta_c);
        let mut b = F::<L>::with_shape(&lat, 24, ls, chunk.len(), 1);
        op_check.source_5d(&mut op_check.work(), &mut b, &eta_l);
        let psi = solve(&b, &format!("columns {chunk:?}"), out);
        let (mut q_l, mut m_l) = (eta_l.zeros_like(), eta_l.zeros_like());
        op_check.propagator_4d(&mut q_l, &psi);
        op_check.midpoint_4d(&mut m_l, &psi);
        let (mut q, mut m) = (eta_c.zeros_like(), eta_c.zeros_like());
        q.convert_from(&q_l);
        m.convert_from(&m_l);
        prop.assign(chunk, &q);
        mid.assign(chunk, &m);
    }
    // Bridge++'s layout of the output: one block per channel, t from the source.
    let t0 = src.source_position[3];
    let nt = params.main.lattice_size[3];
    let shift = |c: &[C64]| -> Vec<C64> { (0..nt).map(|t| c[(t + t0) % nt]).collect() };
    for name in CHANNELS {
        let (sink, source) = channel::<Dirac>(name).expect("a channel of the list");
        let c = shift(&meson_correlator(
            &prop,
            &prop,
            &sink,
            &source,
            &Dirac::GAMMA5,
        ));
        writeln!(out, "  {name}:")?;
        for (t, z) in c.iter().enumerate() {
            writeln!(out, "    {t:3}  {:+.15e}  {:+.15e}", z.re, z.im)?;
        }
    }
    let (pp, j5q) = (pion_correlator(&prop), pion_correlator(&mid));
    writeln!(out, "  residual mass m_res(t) = C_J5q(t) / C_PS(t):")?;
    for t in 0..nt {
        let at = (t + t0) % nt;
        writeln!(
            out,
            "    {t:3}  {:.6e}   (C_J5q {:.6e}, C_PS {:.6e})",
            j5q[at] / pp[at],
            j5q[at],
            pp[at]
        )?;
    }
    Ok(())
}
