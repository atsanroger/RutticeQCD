//! Möbius domain-wall solves on the even sites, with or without the even-odd
//! Pauli-Villars multigrid (`pv_eo`), from a parameter file in the style of Bridge++
//! (`params`): what `ruttice` runs for `fermion_type: Mobius` (and the example
//! `mdwf_pv_eo_mg`, which can choose the seeds of random right-hand sides).

use super::memory_line;
use super::params::{Backend, Params, PrecisionType, PreconditionerType, SolverType};
use crate::fermion::{
    EoForm, FermionOp, Mdwf, MdwfLayout, PvEvenOdd, SingleEven, SingleOperator, Variant,
};
use crate::kernel::{Dirac, GammaRepre};
use crate::measure::{
    channel, meson_correlator, pion_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use crate::solver::preconditioner::multigrid::{
    random_even, AnyParts, CpuParts, CycleLayout, MdwfBackend, OnCpu, PartsIn, PvEoCycle,
    PvEoParams, SetupStage, SharedOperator,
};
use crate::solver::{largest_magnitude, Chebyshev, Convert, SolveError, SolveInfo};
use crate::{
    random_spinor, Aos, Aosoa, Comm, ConvertFrom, EvenOddIndex, Field, Gauge, Half, Lattice,
    Lexical, LinearFn, Parity, SingleComm, Spinor, Storage, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type H<L = Aos, T = f64> = Field<T, Spinor, Half, L>;
type Smoother<'a, L, T> = Chebyshev<PvEvenOdd<'a, T, Dirac, L>, H<L, T>>;
/// A cycle in single precision as a preconditioner of fields in double precision.
type SinglePre<'a, L> = Convert<Preconditioner<'a, L, f32>, H<L, f32>>;
type F<L = Aos> = Field<f64, Spinor, Lexical, L>;

/// The first `count` vectors of a low-mode checkpoint, read site by site. The file holds the
/// even sites of the whole lattice in the order of one process; every process reads it and
/// keeps its sites.
pub(super) fn load(path: &Path, like: &H, count: usize) -> Result<H, Box<dyn Error>> {
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
    let lat = like.lattice().clone();
    let (whole, mine) = (
        EvenOddIndex::of_extents(lat.global),
        EvenOddIndex::new(&lat),
    );
    let sites = whole.num_half_sites();
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
        let coords = lat.global_site_coords(whole.full_site(Parity::Even, site));
        let Some(local) = lat.local_site(coords) else {
            continue;
        };
        let (parity, half) = mine.half_site(local);
        debug_assert!(parity == Parity::Even);
        for (k, v) in x.site_mut(0, half).iter_mut().enumerate() {
            *v = f64::from_le_bytes(buf[8 * k..8 * k + 8].try_into().expect("8 bytes"));
        }
    }
    Ok(x)
}

/// A preconditioner of the operator in the layout `L` of the fine fields, stored as `T`.
pub(super) enum Preconditioner<'a, L: CycleLayout<T>, T: Storage = f64> {
    None,
    Chebyshev(Box<Smoother<'a, L, T>>),
    Single(Box<PvEoCycle<'a, Dirac, f32, L, T>>),
    Double(Box<PvEoCycle<'a, Dirac, f64, L, T>>),
}

impl<L: CycleLayout<T>, T: Storage> Preconditioner<'_, L, T> {
    pub(super) fn coarse_applications(&self) -> usize {
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

    fn as_fn(&mut self) -> Option<&mut dyn LinearFn<H<L, T>>> {
        match self {
            Preconditioner::None => None,
            Preconditioner::Chebyshev(c) => Some(c.as_mut()),
            Preconditioner::Single(c) => Some(c.as_mut()),
            Preconditioner::Double(c) => Some(c.as_mut()),
        }
    }
}

/// As a map: for `Convert`, around a preconditioner stored otherwise than the solver's fields.
impl<L: CycleLayout<T>, T: Storage> LinearFn<H<L, T>> for Preconditioner<'_, L, T> {
    fn mult(&mut self, out: &mut H<L, T>, inp: &H<L, T>) {
        self.as_fn().expect("a preconditioner").mult(out, inp);
    }
}

/// The cycle of `parts` for `plain` and the operator `outer`, its smoother on the operators
/// `layout` in the layout `L` (`map`).
fn cycle_of<'a, L: CycleLayout<T>, T: Storage>(
    plain: (&'a Mdwf<T>, &'a Mdwf<T>),
    outer: (&'a Mdwf<T>, &'a Mdwf<T>, EoForm),
    layout: (&'a Mdwf<T, Dirac, L>, &'a Mdwf<T, Dirac, L>),
    map: L::Map,
    p: &'a PartsIn<T>,
    params: PvEoParams,
) -> Preconditioner<'a, L, T> {
    let (op, pv) = plain;
    let hi = p.tops[0];
    match &p.coarse {
        AnyParts::Single(c) => Preconditioner::Single(Box::new(PvEoCycle::from_parts_in_layout(
            op,
            pv,
            outer,
            layout,
            map,
            &p.basis,
            params,
            c.clone().with_top(hi),
        ))),
        AnyParts::Double(c) => Preconditioner::Double(Box::new(PvEoCycle::from_parts_in_layout(
            op,
            pv,
            outer,
            layout,
            map,
            &p.basis,
            params,
            c.clone().with_top(hi),
        ))),
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
    let size = params.main.lattice_size;
    let requested = (params.main.grid_size != [1; ND]).then_some(params.main.grid_size);
    let grid = super::grid_for(size, requested, comm.size())?;
    let lat = Lattice::new(size, grid, comm);
    // Gauge smearing: the operators are made of the smeared links (stout on the device for
    // `cuda_bridge`).
    let smearing = params.gauge_smearing.as_ref();
    let u = match params.main.backend {
        #[cfg(feature = "cuda-bridge")]
        Backend::CudaBridge => {
            super::load_links_on_device(&lat, &params.gauge, smearing, file, out)?
        }
        _ => super::load_links(&lat, &params.gauge, smearing, file, out)?,
    };
    match (params.main.backend, params.main.simd_lanes) {
        (Backend::Simd, 2) => run_in::<Aosoa<2>>(params, file, &lat, &u, seeds, out),
        (Backend::Simd, 4) => run_in::<Aosoa<4>>(params, file, &lat, &u, seeds, out),
        (Backend::Simd, 8) => run_in::<Aosoa<8>>(params, file, &lat, &u, seeds, out),
        (Backend::Cpu, _) => run_in::<Aos>(params, file, &lat, &u, seeds, out),
        #[cfg(feature = "cuda-bridge")]
        (Backend::CudaBridge, _) => super::mdwf_cuda::run(params, file, &lat, &u, seeds, out),
        _ => unreachable!("refused when the file is read"),
    }
}

/// The operator stored as `T` in the layout `L`, at a mass and with Neff's α.
fn operator<T: Storage, L: MdwfLayout<T>>(
    params: &Params,
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    mass: f64,
    alpha: f64,
) -> Result<Mdwf<T, Dirac, L>, Box<dyn Error>> {
    let mut op = Mdwf::<T, Dirac, L>::in_layout(lat, params.mdwf_params(mass))
        .ok_or("the lattice cannot be cut for the SIMD layout")?;
    op.set_gauge(u)?;
    op.set_alpha(alpha);
    Ok(op)
}

/// D and the second operator of the preconditioned one (the Pauli-Villars operator, or D itself
/// for the normal equations), stored as `T` in the layout `L`.
struct Pair<T: Storage, L: MdwfLayout<T>> {
    d: Mdwf<T, Dirac, L>,
    other: Option<Mdwf<T, Dirac, L>>,
}

impl<T: Storage, L: MdwfLayout<T>> Pair<T, L> {
    /// With the α of the file if `alpha`, else the plain operators.
    fn new(
        params: &Params,
        lat: &Arc<Lattice>,
        u: &Field<f64, Gauge>,
        alpha: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let fo = &params.fopr;
        let of = |a: f64| if alpha { a } else { 1.0 };
        let (_, normal) = params.form();
        Ok(Self {
            d: operator(params, lat, u, fo.mass(), of(fo.neff_alpha))?,
            other: if normal {
                None
            } else {
                Some(operator(
                    params,
                    lat,
                    u,
                    fo.quark_mass_pauli_villars,
                    of(fo.neff_alpha_pauli_villars),
                )?)
            },
        })
    }

    fn both(&self) -> (&Mdwf<T, Dirac, L>, &Mdwf<T, Dirac, L>) {
        (&self.d, self.other.as_ref().unwrap_or(&self.d))
    }
}

/// The operators of a multigrid stored as `T`: those of its coarse correction, which is in the
/// usual layout, plain and with α, and those of its smoother, in the layout `L`.
struct CycleOperators<T: Storage, L: MdwfLayout<T>> {
    plain: Pair<T, Aos>,
    alpha: Pair<T, Aos>,
    smoother: Option<Pair<T, L>>,
}

impl<T: Storage, L: MdwfLayout<T>> CycleOperators<T, L> {
    /// `smoother`: with its own operators (else those of the outer solver serve).
    fn new(
        params: &Params,
        lat: &Arc<Lattice>,
        u: &Field<f64, Gauge>,
        smoother: bool,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            plain: Pair::new(params, lat, u, false)?,
            alpha: Pair::new(params, lat, u, true)?,
            smoother: if smoother {
                Some(Pair::new(params, lat, u, true)?)
            } else {
                None
            },
        })
    }
}

/// Everything after reading the configuration, with the fine fields in the layout `L`.
fn run_in<L: CycleLayout<f64> + CycleLayout<f32>>(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    u: &Field<f64, Gauge>,
    seeds: (u64, u64),
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>>
where
    F<L>: ConvertFrom<F>,
    F: ConvertFrom<F<L>>,
    Mdwf<f32, Dirac, L>: SingleOperator<Half = H<L, f32>>,
    H<L>: ConvertFrom<H<L, f32>>,
    H<L, f32>: ConvertFrom<H<L>>,
{
    let lat = lat.clone();
    let fo = &params.fopr;
    let s = &params.solver;
    let detailed = matches!(
        params.main.verbose_level,
        super::params::VerboseLevel::Detailed | super::params::VerboseLevel::Paranoiac
    );
    // The operator of the outer solver, in double precision, with the Pauli-Villars one (the
    // normal equations are the operator with D in place of D_PV).
    let outer = Pair::<f64, L>::new(params, &lat, u, true)?;
    let (op_l, pv_l) = outer.both();
    // The plain D for the check of the solution, when the operator carries α.
    let op_check_l = if fo.neff_alpha != 1.0 {
        Some(operator::<f64, L>(params, &lat, u, fo.mass(), 1.0)?)
    } else {
        None
    };
    let op_check = op_check_l.as_ref().unwrap_or(op_l);
    let (form, _) = params.form();
    let ls = fo.ls();
    let mut like_l = H::<L>::with_shape(&lat, 24, ls, 1, 1);
    like_l.set_parity(Parity::Even);
    let index = op_l
        .even_odd_index()
        .ok_or("no even-odd decomposition")?
        .clone();

    // The multigrid: the test vectors (from a checkpoint, or filtered, and improved by the
    // bootstrap), P, the levels and the coarse parts by the CPU's backend (`OnCpu`, as in HMC),
    // then the cycle made of them with its smoother in the layout `L`, in single precision with
    // operators of its own if the file says.
    let start = Instant::now();
    let mg_settings = params.multigrid();
    let mg = params.multigrid.as_ref().filter(|_| mg_settings.is_some());
    let settings = mg.map(|mg| mg.mdwf_settings(PrecisionType::Single));
    // The plain operators the parts are made for and the outer operator, in the usual layout.
    let aos = match settings {
        Some(_) => Some((
            Pair::<f64, Aos>::new(params, &lat, u, false)?,
            Pair::<f64, Aos>::new(params, &lat, u, true)?,
        )),
        None => None,
    };
    let single_ops = match settings.is_some_and(|m| m.fine_single) {
        true => Some(CycleOperators::<f32, L>::new(params, &lat, u, true)?),
        false => None,
    };
    let parts = match (mg, &settings, &aos) {
        (Some(mg), Some(m), Some((plain, alpha))) => {
            let backend = OnCpu::<Dirac>::new(&plain.d);
            let kept = match &mg.setup.checkpoint {
                Some(path) => {
                    let path = file.parent().map_or(path.clone(), |d| d.join(path));
                    let mut like = H::with_shape(&lat, 24, ls, 1, 1);
                    like.set_parity(Parity::Even);
                    Some(backend.vectors_from_host(&load(&path, &like, m.test_vectors)?, m)?)
                }
                None => None,
            };
            let mut failed: Option<std::io::Error> = None;
            let mut report = |stage: SetupStage<'_>| {
                let line = match stage {
                    SetupStage::Filtered(theta) => writeln!(
                        out,
                        "  filter: Ritz values of Ã†Ã {:.3e} .. {:.3e} ({:.0} s)",
                        theta[0],
                        theta[theta.len() - 1],
                        start.elapsed().as_secs_f64()
                    ),
                    SetupStage::Levels {
                        pauli_villars: false,
                        applications,
                        seconds,
                    } => writeln!(
                        out,
                        "  levels below the coarse lattice {:?} (refinement {:?}): {applications} coarse applications ({seconds:.0} s)",
                        mg.super_blocks(),
                        mg.level_bootstrap().iter().map(|b| (b.rounds, b.iterations)).collect::<Vec<_>>(),
                    ),
                    SetupStage::Levels {
                        pauli_villars: true,
                        applications,
                        seconds,
                    } => writeln!(
                        out,
                        "  the same levels for the Pauli-Villars factor: {applications} coarse applications ({seconds:.0} s)"
                    ),
                    SetupStage::Round(_) | SetupStage::Checksum(..) => Ok(()),
                };
                if let Err(e) = line {
                    failed.get_or_insert(e);
                }
            };
            let (d, pv) = alpha.both();
            let outers = [(d, pv, form)];
            let (_, parts, _) =
                backend.rebuild(plain.both(), &outers, &lat, m, kept, true, &mut report)?;
            if let Some(e) = failed {
                return Err(e.into());
            }
            Some(parts)
        }
        _ => None,
    };
    let mut steps = 1;
    let mut pre: Preconditioner<L> = Preconditioner::None;
    let mut single_pre: Option<SinglePre<L>> = None;
    match (s.preconditioner_type, &mg_settings) {
        (PreconditionerType::Multigrid, Some((cycle_params, _, _))) => {
            steps = cycle_params.chebyshev_steps;
            match (&parts, &aos, &single_ops) {
                (Some(CpuParts::Double(p)), Some((plain, alpha)), _) => {
                    let (d, pv) = alpha.both();
                    let map = <L as CycleLayout<f64>>::map(&lat)
                        .ok_or("the lattice cannot be cut for the SIMD layout")?;
                    let outer = (d, pv, form);
                    pre = cycle_of(plain.both(), outer, (op_l, pv_l), map, p, *cycle_params);
                }
                (Some(CpuParts::Single(p)), _, Some(o)) => {
                    let (d, pv) = o.alpha.both();
                    let smoother = o.smoother.as_ref().expect("made with them");
                    let map = <L as CycleLayout<f32>>::map(&lat)
                        .ok_or("the lattice cannot be cut for the SIMD layout")?;
                    let outer = (d, pv, form);
                    let cycle = cycle_of(
                        o.plain.both(),
                        outer,
                        smoother.both(),
                        map,
                        p,
                        *cycle_params,
                    );
                    let mut like = H::<L, f32>::with_shape(&lat, 24, ls, 1, 1);
                    like.set_parity(Parity::Even);
                    single_pre = Some(Convert::new(cycle, &like));
                }
                _ => unreachable!("the parts are made in the precision of the cycle"),
            }
        }
        (PreconditionerType::Chebyshev, _) => {
            let c = s.chebyshev_settings();
            let map = <L as CycleLayout<f64>>::map(&lat)
                .ok_or("the lattice cannot be cut for the SIMD layout")?;
            let mut start_vector = like_l.zeros_like();
            <L as CycleLayout<f64>>::from_sites(
                &map,
                &mut start_vector,
                &random_even(&lat, &index, ls, 1, 5),
            );
            let hi = 1.1
                * largest_magnitude(
                    &mut PvEvenOdd::new(op_l, pv_l, form, &like_l),
                    &mut start_vector,
                    50,
                );
            let cheb = Chebyshev::new(
                PvEvenOdd::new(op_l, pv_l, form, &like_l),
                c.number_of_steps,
                hi / c.interval_ratio,
                hi,
                &like_l,
            );
            pre = Preconditioner::Chebyshev(Box::new(cheb));
            steps = c.number_of_steps;
        }
        _ => {}
    }
    writeln!(
        out,
        "  setup: {:.0} s; {}",
        start.elapsed().as_secs_f64(),
        memory_line()
    )?;

    // With the multigrid in double precision the outer solver uses the cycle's own operator:
    // one set of its work fields for both.
    let mut a = match &mut pre {
        Preconditioner::Single(c) => c.operator(),
        Preconditioner::Double(c) => c.operator(),
        _ => SharedOperator::new(PvEvenOdd::new(op_l, pv_l, form, &like_l)),
    };
    let coarse_applications = |pre: &Preconditioner<L>, single_pre: &Option<SinglePre<L>>| {
        single_pre.as_ref().map_or(pre.coarse_applications(), |c| {
            c.inner().coarse_applications()
        })
    };
    // Mixed precision (CGNE only): CG in single precision on K†K, K the left Schur factor of the
    // operator in single precision, with reliable updates in double precision each time |r|² has
    // fallen by `drop_sqr` (`SingleEven::reliable_normal`, as on the device).
    let mut single = match s.mixed() {
        Some(drop_sqr) => {
            let op32 = operator::<f32, L>(params, &lat, u, fo.mass(), fo.neff_alpha)?;
            Some((SingleEven::new(op32, &lat, drop_sqr), drop_sqr))
        }
        None => None,
    };
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let solve = |b: F<L>, label: &str, out: &mut dyn Write| -> Result<F<L>, Box<dyn Error>> {
        let b = &b;
        let nv = b.num_vectors();
        let mut rhs = like_l.zeros_with(nv);
        a.right_hand_side(&mut rhs, b);
        let (outer, applied) = (Cell::new(0), Cell::new(0));
        let coarse_before = coarse_applications(&pre, &single_pre);
        let inner_per = match &single_pre {
            Some(c) => c.inner().inner_applications(steps),
            None => pre.inner_applications(steps),
        };
        let mut sol = rhs.zeros_like();
        let counted = crate::meter::Counts::now();
        let start = Instant::now();
        let (result, history): (Result<SolveInfo, SolveError>, _) = {
            let mut counted_a = Counted {
                a: &mut a,
                n: &outer,
            };
            let m: Option<&mut dyn LinearFn<H<L>>> = match single_pre.as_mut() {
                Some(c) => Some(c),
                None => pre.as_fn(),
            };
            let mut counted_pre = m.map(|m| Counted { a: m, n: &applied });
            let m = counted_pre.as_mut().map(|c| c as &mut dyn LinearFn<H<L>>);
            match (s.solver_type, single.as_mut()) {
                (SolverType::Cg, Some((sc, drop_sqr))) => {
                    let stop = s.stop();
                    let solved =
                        sc.reliable_normal(&mut counted_a, &mut sol, &rhs, stop, *drop_sqr);
                    (solved, None)
                }
                _ => s.solve(&mut counted_a, m, &mut sol, &rhs, detailed),
            }
        };
        let seconds = start.elapsed().as_secs_f64();
        let work = crate::meter::Counts::now().since(&counted);
        let iterations = iterations_text(&result);
        // Applications of the operator, per right-hand side.
        let applications = (outer.get() + applied.get() * inner_per) / nv;
        let coarse = (coarse_applications(&pre, &single_pre) - coarse_before) / nv;
        let mut x = b.zeros_like();
        a.solution(&mut x, &sol, b);
        let mut r = b.zeros_like();
        op_check.mult(&mut op_check.work(), &mut r, &x, Variant::D);
        r.aypx(&vec![C64::new(-1.0, 0.0); nv], b);
        let (mut rr, mut bb) = (vec![0.0; nv], vec![0.0; nv]);
        r.norm_sqr(&mut rr);
        b.norm_sqr(&mut bb);
        let worst = rr.iter().zip(&bb).fold(0.0_f64, |m, (r, b)| m.max(r / b));
        writeln!(
            out,
            "  {label}: {}{} {iterations} iterations; per right-hand side {} preconditioner applications, {applications} applications of the operator = {} Wilson sweeps, {coarse} coarse; worst |Dx−b|² {worst:.0e}; {seconds:.1} s, {}; {}",
            s.solver_type,
            if single.is_some() { " (mixed precision)" } else { "" },
            applied.get() / nv,
            2 * applications,
            work.line(seconds),
            memory_line()
        )?;
        super::history_lines(out, label, history);
        if result.is_ok() {
            super::check_solution(label, worst, s.stop().tolerance_sqr)?;
        }
        Ok(x)
    };
    let mut fields = CpuFields {
        lat: &lat,
        op: op_check,
        ls,
    };
    solve_sources(params, &lat, seeds, &mut fields, solve, out)
}

/// The iterations of a solve as the report shows them: with `!` if it did not converge.
pub(super) fn iterations_text(result: &Result<SolveInfo, SolveError>) -> String {
    match result {
        Ok(i) => format!("{}", i.iterations),
        Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
        Err(e) => format!("failed: {e}"),
    }
}

/// Where a backend keeps the five-dimensional fields of a propagator run, and how it makes
/// them: the right-hand sides and the four-dimensional fields of the solutions, which the
/// host collects (`solve_sources`).
pub(super) trait SourceFields {
    /// A five-dimensional field of right-hand sides or solutions.
    type Fifth;
    /// The random right-hand side of `seed`.
    fn random(&mut self, seed: u64) -> Result<Self::Fifth, Box<dyn Error>>;
    /// The five-dimensional source of the four-dimensional columns `eta`.
    fn source(&mut self, eta: &F) -> Result<Self::Fifth, Box<dyn Error>>;
    /// The four-dimensional quark field and the midpoint field of `psi`, shaped like `eta`.
    fn project(&mut self, psi: &Self::Fifth, eta: &F) -> Result<(F, F), Box<dyn Error>>;
}

/// What a propagator run does once its operators and preconditioner are made, on any backend:
/// without a `Source` block, `solve` (which solves D ψ = b and reports it in one line under its
/// label) on the random right-hand sides of the seeds `seeds` (inclusive); with one, on the
/// point source, `number_of_rhs_per_solve` columns at a time, and the meson correlators and
/// the residual mass of the propagator.
pub(super) fn solve_sources<S: SourceFields>(
    params: &Params,
    lat: &Arc<Lattice>,
    (seed_lo, seed_hi): (u64, u64),
    fields: &mut S,
    mut solve: impl FnMut(S::Fifth, &str, &mut dyn Write) -> Result<S::Fifth, Box<dyn Error>>,
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let Some(src) = &params.source else {
        for seed in seed_lo..=seed_hi {
            let b = fields.random(seed)?;
            solve(b, &format!("seed {seed}"), out)?;
        }
        return Ok(());
    };
    let eta = point_source::<f64>(lat, src.source_position);
    let (mut prop, mut mid) = (eta.zeros_like(), eta.zeros_like());
    let columns: Vec<usize> = (0..NUM_SOURCES).collect();
    for chunk in columns.chunks(src.number_of_rhs_per_solve) {
        let eta_c = eta.select(chunk);
        let b = fields.source(&eta_c)?;
        let psi = solve(b, &format!("columns {chunk:?}"), out)?;
        let (q, m) = fields.project(&psi, &eta_c)?;
        prop.assign(chunk, &q);
        mid.assign(chunk, &m);
    }
    write_measurements(
        src.source_position,
        params.main.lattice_size[3],
        &prop,
        &mid,
        out,
    )
}

/// The fields of a run on the CPU, in the layout `L`, made with the plain operator `op`.
struct CpuFields<'a, L: MdwfLayout<f64>> {
    lat: &'a Arc<Lattice>,
    op: &'a Mdwf<f64, Dirac, L>,
    ls: usize,
}

impl<L: MdwfLayout<f64>> SourceFields for CpuFields<'_, L>
where
    F<L>: ConvertFrom<F>,
    F: ConvertFrom<F<L>>,
{
    type Fifth = F<L>;

    fn random(&mut self, seed: u64) -> Result<F<L>, Box<dyn Error>> {
        let mut b = F::with_shape(self.lat, 24, self.ls, 1, 1);
        b.as_mut_slice()
            .copy_from_slice(random_spinor(self.lat, self.ls, seed).as_slice());
        let mut b_l = F::<L>::with_shape(self.lat, 24, self.ls, 1, 1);
        b_l.convert_from(&b);
        Ok(b_l)
    }

    fn source(&mut self, eta: &F) -> Result<F<L>, Box<dyn Error>> {
        let n = eta.num_vectors();
        let mut eta_l = F::<L>::with_shape(self.lat, 24, 1, n, 1);
        eta_l.convert_from(eta);
        let mut b = F::<L>::with_shape(self.lat, 24, self.ls, n, 1);
        self.op.source_5d(&mut self.op.work(), &mut b, &eta_l);
        Ok(b)
    }

    fn project(&mut self, psi: &F<L>, eta: &F) -> Result<(F, F), Box<dyn Error>> {
        let like = F::<L>::with_shape(self.lat, 24, 1, eta.num_vectors(), 1);
        let (mut q_l, mut m_l) = (like.zeros_like(), like);
        self.op.propagator_4d(&mut q_l, psi);
        self.op.midpoint_4d(&mut m_l, psi);
        let (mut q, mut m) = (eta.zeros_like(), eta.zeros_like());
        q.convert_from(&q_l);
        m.convert_from(&m_l);
        Ok((q, m))
    }
}

/// The meson correlators of the propagator `prop` from a source at `position`, and the residual
/// mass from it and the field at the middle of the fifth dimension `mid`.
pub(super) fn write_measurements(
    position: [usize; ND],
    nt: usize,
    prop: &F,
    mid: &F,
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    // Bridge++'s layout of the output: one block per channel, t from the source.
    let t0 = position[3];
    let shift = |c: &[C64]| -> Vec<C64> { (0..nt).map(|t| c[(t + t0) % nt]).collect() };
    for name in CHANNELS {
        let (sink, source) = channel::<Dirac>(name).expect("a channel of the list");
        let c = shift(&meson_correlator(
            prop,
            prop,
            &sink,
            &source,
            &Dirac::GAMMA5,
        ));
        writeln!(out, "  {name}:")?;
        for (t, z) in c.iter().enumerate() {
            writeln!(out, "    {t:3}  {:+.15e}  {:+.15e}", z.re, z.im)?;
        }
    }
    let (pp, j5q) = (pion_correlator(prop), pion_correlator(mid));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{local_part, LocalComm};

    /// A checkpoint written on one process is read on several: every process gets its sites.
    #[test]
    fn checkpoint_is_read_on_several_processes() {
        let global = [4, 4, 4, 8];
        let (ls, m, keep) = (2, 3, 2);
        let whole_lat = Lattice::single_node(global);
        let index = EvenOddIndex::new(&whole_lat);
        // Vectors on all sites, so that the parts of the processes can be cut out of them.
        let full = random_spinor(&whole_lat, ls * m, 77);
        let mut like = H::with_shape(&whole_lat, 24, ls, 1, 1);
        like.set_parity(Parity::Even);
        let reals = 24 * ls;
        let mut bytes = Vec::new();
        for word in [0u64, m as u64, (index.num_half_sites() * m * reals) as u64] {
            bytes.extend(word.to_le_bytes());
        }
        bytes.extend((0..m).flat_map(|_| 0u64.to_le_bytes()));
        // The file: per even site, the m vectors one after the other.
        for half in 0..index.num_half_sites() {
            let site = full.site(0, index.full_site(Parity::Even, half));
            bytes.extend(site.iter().flat_map(|v| v.to_le_bytes()));
        }
        let path = std::env::temp_dir().join(format!("ruttice_ckpt_{}.bin", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();

        let one = load(&path, &like, keep).unwrap();
        for grid in [[1, 1, 1, 2], [2, 1, 2, 2], [2, 2, 2, 2]] {
            let worst = LocalComm::run(global, grid, |lat| {
                let mut like = H::with_shape(&lat, 24, ls, 1, 1);
                like.set_parity(Parity::Even);
                let got = load(&path, &like, keep).unwrap();
                // The same from the vectors on all sites: this process's part, its even sites.
                let part = local_part(&full, &lat);
                let mine = EvenOddIndex::new(&lat);
                let mut worst = 0.0f64;
                for half in 0..mine.num_half_sites() {
                    let want = &part.site(0, mine.full_site(Parity::Even, half))[..keep * reals];
                    for (a, b) in got.site(0, half).iter().zip(want) {
                        worst = worst.max((a - b).abs());
                    }
                }
                worst
            });
            assert!(worst.iter().all(|w| *w == 0.0), "{grid:?}: {worst:?}");
        }
        assert_eq!(one.num_vectors(), keep);
        let _ = std::fs::remove_file(path);
    }
}
