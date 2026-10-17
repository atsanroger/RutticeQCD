//! Möbius domain-wall solves on the even sites, with or without the even-odd
//! Pauli-Villars multigrid (`pv_eo`), from a parameter file in the style of Bridge++
//! (`params`): what `ruttice` runs for `fermion_type: Mobius` (and the example
//! `mdwf_pv_eo_mg`, which can choose the seeds of random right-hand sides).

use super::memory_line;
use super::params::{Backend, Params, PrecisionType, PreconditionerType, SolverType};
use crate::fermion::{EoForm, FermionOp, Mdwf, MdwfLayout, PvEvenOdd, Variant};
use crate::kernel::{Dirac, GammaRepre};
use crate::measure::{
    channel, meson_correlator, pion_correlator, point_source, CHANNELS, NUM_SOURCES,
};
use crate::solver::preconditioner::multigrid::{
    pv_eo_basis, pv_eo_bootstrap, pv_eo_coarse_vectors, pv_eo_low_modes, random_even, Aggregation,
    CycleLayout, PvEoBasis, PvEoCycle, PvEoParams, SharedOperator,
};
use crate::solver::{
    largest_magnitude, BiCgStab, BlockGcr, Cg, Chebyshev, Convert, Fgmres, Gcr, SolveError,
    SolveInfo, Solver, Stop,
};
use crate::{
    random_spinor, Aos, Aosoa, Comm, ConvertFrom, EvenOddIndex, Field, Gauge, Half, Lattice,
    Lexical, LinearFn, Parity, SingleComm, Spinor, Storage, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type H<L = Aos, T = f64> = Field<T, Spinor, Half, L>;
/// The basis of a multigrid with the vectors it was built from.
pub(super) type Tested<T = f64> = (PvEoBasis<T>, H<Aos, T>);
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

/// The vectors y on the even sites the multigrid is built from (from a checkpoint, or
/// filtered here, and improved by the bootstrap if the file asks) with the basis on them, for
/// the operators `op` and `pv` and stored as they are. `start`: when the setup began.
pub(super) fn multigrid_basis<T: Storage>(
    params: &Params,
    file: &Path,
    lat: &Arc<Lattice>,
    (op, pv): (&Mdwf<T>, &Mdwf<T>),
    start: Instant,
    out: &mut dyn Write,
) -> Result<Option<Tested<T>>, Box<dyn Error>> {
    let relative = |p: &PathBuf| file.parent().map_or(p.clone(), |d| d.join(p));
    let (Some((_, low, boot)), Some(mg)) = (params.multigrid(), &params.multigrid) else {
        return Ok(None);
    };
    let keep = mg.level1.test_vectors(true) / 2;
    let ls = op.params().ls;
    let mut y = match &mg.setup.checkpoint {
        Some(path) => {
            // Checkpoints are in double precision.
            let mut like = H::with_shape(lat, 24, ls, 1, 1);
            like.set_parity(Parity::Even);
            let read = load(&relative(path), &like, keep)?;
            let mut y = H::<Aos, T>::with_shape(lat, 24, ls, read.num_vectors(), 1);
            y.set_parity(Parity::Even);
            y.convert_from(&read);
            y
        }
        None => {
            let (x, theta) = pv_eo_low_modes(op, pv, lat, low);
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
    let agg = Aggregation::new(lat, mg.level1.block_size);
    let basis = if boot.rounds > 0 {
        pv_eo_bootstrap(op, pv, &agg, &mut y, boot)?
    } else {
        pv_eo_basis(op, pv, &agg, &y)?
    };
    Ok(Some((basis, y)))
}

/// The cycle on a basis of `multigrid_basis` for the operator `outer`, with the levels below
/// the coarse lattice the file asks for, and the flops of one application of a coarse
/// operator. `fine`: the operators of `outer` in the layout of the fine fields.
#[allow(clippy::too_many_arguments)]
pub(super) fn multigrid_cycle<'a, L: CycleLayout<T>, T: Storage>(
    params: &Params,
    lat: &Lattice,
    (op, pv): (&'a Mdwf<T>, &'a Mdwf<T>),
    outer: (&'a Mdwf<T>, &'a Mdwf<T>, EoForm),
    fine: (&'a Mdwf<T, Dirac, L>, &'a Mdwf<T, Dirac, L>),
    basis: &'a PvEoBasis<T>,
    y: &H<Aos, T>,
    cycle_params: PvEoParams,
    out: &mut dyn Write,
) -> Result<(Preconditioner<'a, L, T>, f64), Box<dyn Error>> {
    let mg = params.multigrid.as_ref().expect("multigrid() found it");
    let dim = basis.coarse_dim() as f64;
    let coarse_sites = basis.aggregation().coarse_lattice().num_sites() as f64;
    let coarse_flops = 72.0 * dim * dim * coarse_sites;
    let map = L::map(lat).ok_or("the lattice cannot be cut for the SIMD layout")?;
    let mut cycle = match mg.coarse_precision {
        PrecisionType::Single => {
            Preconditioner::Single(Box::new(PvEoCycle::<Dirac, f32, L, T>::with_layout(
                op,
                pv,
                outer,
                fine,
                map,
                basis,
                cycle_params,
            )?))
        }
        PrecisionType::Double => {
            Preconditioner::Double(Box::new(PvEoCycle::<Dirac, f64, L, T>::with_layout(
                op,
                pv,
                outer,
                fine,
                map,
                basis,
                cycle_params,
            )?))
        }
    };
    let super_blocks = mg.super_blocks();
    if !super_blocks.is_empty() {
        let levels_start = Instant::now();
        let vectors = pv_eo_coarse_vectors(op, pv, basis, y);
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
        if mg.levels_for_pauli_villars {
            if cycle_params.pv_solver
                != crate::solver::preconditioner::multigrid::CoarseKrylov::Minres
            {
                return Err("MultiGrid.levels_for_pauli_villars needs MINRES for the Pauli-Villars factor (Coarse_Solver_PV)".into());
            }
            let levels_start = Instant::now();
            let applications = match &mut cycle {
                Preconditioner::Single(c) => c.add_pv_levels(&vectors, &super_blocks)?,
                Preconditioner::Double(c) => c.add_pv_levels(&vectors, &super_blocks)?,
                _ => unreachable!(),
            };
            writeln!(
                out,
                "  the same levels for the Pauli-Villars factor: {applications} coarse applications ({:.0} s)",
                levels_start.elapsed().as_secs_f64()
            )?;
        }
    }
    Ok((cycle, coarse_flops))
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
    let format = super::config::GaugeFormat::from(params.gauge.gauge_config_type_input);
    let u =
        super::gauge_file(format)?.read(&relative(&params.gauge.config_filename_input), &lat)?;
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
    (seed_lo, seed_hi): (u64, u64),
    out: &mut dyn Write,
) -> Result<(), Box<dyn Error>>
where
    F<L>: ConvertFrom<F>,
    F: ConvertFrom<F<L>>,
{
    let lat = lat.clone();
    let fo = &params.fopr;
    let s = &params.solver;
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

    // The multigrid above the coarse lattice is stored as the file says, in single precision
    // with operators of its own. They and the basis live as long as the cycle that borrows
    // them.
    let start = Instant::now();
    let mg_settings = params.multigrid();
    let single = mg_settings.is_some()
        && params
            .multigrid
            .as_ref()
            .is_some_and(|mg| mg.fine_precision == PrecisionType::Single);
    let (double_ops, single_ops) = match (mg_settings.is_some(), single) {
        (false, _) => (None, None),
        (true, false) => (
            Some(CycleOperators::<f64, L>::new(params, &lat, u, false)?),
            None,
        ),
        (true, true) => (
            None,
            Some(CycleOperators::<f32, L>::new(params, &lat, u, true)?),
        ),
    };
    // The test vectors are dropped once the cycle is made from them.
    let (double_basis, double_y) = match &double_ops {
        Some(o) => multigrid_basis(params, file, &lat, o.plain.both(), start, out)?,
        None => None,
    }
    .unzip();
    let (single_basis, single_y) = match &single_ops {
        Some(o) => multigrid_basis(params, file, &lat, o.plain.both(), start, out)?,
        None => None,
    }
    .unzip();
    let (mut coarse_flops, mut steps) = (0.0, 1);
    let mut pre: Preconditioner<L> = Preconditioner::None;
    let mut single_pre: Option<SinglePre<L>> = None;
    match (s.preconditioner_type, &mg_settings) {
        (PreconditionerType::Multigrid, Some((cycle_params, _, _))) => {
            steps = cycle_params.chebyshev_steps;
            if let (Some(o), Some(basis), Some(y)) = (&double_ops, &double_basis, &double_y) {
                let (d, pv) = o.alpha.both();
                (pre, coarse_flops) = multigrid_cycle(
                    params,
                    &lat,
                    o.plain.both(),
                    (d, pv, form),
                    (op_l, pv_l),
                    basis,
                    y,
                    *cycle_params,
                    out,
                )?;
            }
            if let (Some(o), Some(basis), Some(y)) = (&single_ops, &single_basis, &single_y) {
                let (d, pv) = o.alpha.both();
                let smoother = o.smoother.as_ref().expect("made with them");
                let (cycle, flops) = multigrid_cycle(
                    params,
                    &lat,
                    o.plain.both(),
                    (d, pv, form),
                    smoother.both(),
                    basis,
                    y,
                    *cycle_params,
                    out,
                )?;
                let mut like = H::<L, f32>::with_shape(&lat, 24, ls, 1, 1);
                like.set_parity(Parity::Even);
                single_pre = Some(Convert::new(cycle, &like));
                coarse_flops = flops;
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
    drop((double_y, single_y));
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
    let fine_flops = 2.0 * (op_l.flops_per_site() * lat.num_sites()) as f64;
    let stop: Stop = s.stop();
    let k = s.number_of_kept_directions;
    // Solves D ψ = b (any number of vectors) and reports it in one line, under `label`.
    let mut solve = |b: &F<L>, label: &str, out: &mut dyn Write| -> F<L> {
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
        let start = Instant::now();
        let result: Result<SolveInfo, SolveError> = {
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
            match s.solver_type {
                SolverType::Gcr => Gcr::new(stop, k).solve(&mut counted_a, m, &mut sol, &rhs),
                SolverType::BlockGcr => {
                    BlockGcr::new(stop, k).solve(&mut counted_a, m, &mut sol, &rhs)
                }
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
        let iterations = iterations_text(&result);
        // Applications of the operator, per right-hand side.
        let applications = (outer.get() + applied.get() * inner_per) / nv;
        let coarse = (coarse_applications(&pre, &single_pre) - coarse_before) / nv;
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
    write_measurements(
        src.source_position,
        params.main.lattice_size[3],
        &prop,
        &mid,
        out,
    )?;
    Ok(())
}

/// The iterations of a solve as the report shows them: with `!` if it did not converge.
pub(super) fn iterations_text(result: &Result<SolveInfo, SolveError>) -> String {
    match result {
        Ok(i) => format!("{}", i.iterations),
        Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
        Err(e) => format!("failed: {e}"),
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
