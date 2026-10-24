//! The domain-wall multigrid of the even sites (`mdwf_even_odd`) as the propagator runs and HMC
//! set it up, the same on every backend: the settings a parameter file gives
//! (`MdwfMultigridSettings`), what a backend supplies (`MdwfBackend`: the test vectors, P, the
//! coarse parts with their levels, the cycles made of them), and the CPU as one (`OnCpu`). The
//! device is another (`backend::cuda_bridge::multigrid::OnDevice`).

use super::{
    pv_eo_basis_stored, pv_eo_bootstrap_stored, pv_eo_coarse_vectors, pv_eo_low_modes, random_even,
    Aggregation, BasisStorage, BootstrapParams, CoarseParts, LevelBootstrap, LowModeParams,
    PvEoBasis, PvEoCycle, PvEoParams,
};
use crate::fermion::{EoForm, EvenOdd, Mdwf, PvEvenOdd, SplitParity};
use crate::kernel::GammaRepre;
use crate::solver::{largest_magnitude, Convert};
use crate::{
    ConvertFrom, EvenOddIndex, Field, Half, Lattice, LinearFn, Parity, Spinor, Storage, Vector, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::sync::Arc;

/// The levels below the coarse lattice, as the parameter files give them:
/// the first `count` super-blocks and refinements, and whether the Pauli-Villars factor has
/// them too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoarseLevels {
    pub count: usize,
    pub super_blocks: [usize; 3],
    pub bootstrap: [LevelBootstrap; 3],
    pub for_pauli_villars: bool,
}

impl CoarseLevels {
    /// From the lists of the file (at most three levels below the coarse one).
    pub fn new(super_blocks: &[usize], bootstrap: &[LevelBootstrap], for_pv: bool) -> Self {
        assert!(super_blocks.len() <= 3 && bootstrap.len() == super_blocks.len());
        let mut l = Self {
            count: super_blocks.len(),
            for_pauli_villars: for_pv,
            ..Self::default()
        };
        l.super_blocks[..l.count].copy_from_slice(super_blocks);
        l.bootstrap[..l.count].copy_from_slice(bootstrap);
        l
    }
}

/// What a parameter file asks of the domain-wall multigrid, the same for the propagator runs
/// and HMC, on every backend.
#[derive(Clone, Copy, Debug)]
pub struct MdwfMultigridSettings {
    /// Aggregates of the coarse lattice.
    pub block: [usize; ND],
    /// Test vectors kept (P has twice as many columns).
    pub test_vectors: usize,
    pub low_modes: LowModeParams,
    pub bootstrap: BootstrapParams,
    /// The cycle of Ã.
    pub cycle: PvEoParams,
    /// The coarse solves in double precision; false: in single precision.
    pub coarse_double: bool,
    /// The setup and the cycle above the coarse lattice in single precision (on the operators
    /// converted to it), the outer solver in double.
    pub fine_single: bool,
    /// P kept in single precision; false: in half precision.
    pub basis_single: bool,
    /// Levels below the coarse lattice for the coarse solves.
    pub levels: CoarseLevels,
}

impl MdwfMultigridSettings {
    /// Why these settings cannot run on several processes on the device, if they cannot: the
    /// levels below the coarse lattice and the exact adjoint of the second factor (a solver of
    /// it other than MINRES) are on one process only so far; and an aggregate must lie within
    /// the local lattice `local` of a process.
    pub fn refused_on_several(&self, local: [usize; ND]) -> Option<String> {
        if self.levels.count > 0 {
            return Some("the levels below the coarse lattice (MultiGrid_Level2 and on) run on one process on the device so far".into());
        }
        if self.cycle.pv_solver != super::CoarseKrylov::Minres {
            return Some("MultiGrid_Level1.Coarse_Solver_PV other than MINRES runs on one process on the device so far".into());
        }
        if (0..ND).any(|mu| !local[mu].is_multiple_of(self.block[mu])) {
            return Some(format!(
                "MultiGrid_Level1.block_size {:?} must divide the local lattice {local:?} of a process",
                self.block
            ));
        }
        None
    }

    /// The settings of `cycle`, the cycle in double precision, P in half precision, no levels.
    pub fn new(block: [usize; ND], test_vectors: usize, cycle: PvEoParams) -> Self {
        Self {
            block,
            test_vectors,
            low_modes: LowModeParams::default(),
            bootstrap: BootstrapParams {
                cycle,
                ..BootstrapParams::default()
            },
            cycle,
            coarse_double: false,
            fine_single: false,
            basis_single: false,
            levels: CoarseLevels::default(),
        }
    }

    /// The super-blocks and refinements of the levels below the coarse lattice.
    pub fn level_lists(&self) -> (&[usize], &[LevelBootstrap]) {
        let l = &self.levels;
        (&l.super_blocks[..l.count], &l.bootstrap[..l.count])
    }

    fn storage(&self) -> BasisStorage {
        match self.basis_single {
            true => BasisStorage::Single,
            false => BasisStorage::Half,
        }
    }
}

/// What the setup has done, for the report of a run.
pub enum SetupStage<'s> {
    /// The filter: the Ritz values of Ã†Ã of the test vectors.
    Filtered(&'s [f64]),
    /// A round of the bootstrap.
    Round(usize),
    /// The levels below the coarse lattice made, for the quark factor or the Pauli-Villars one:
    /// the applications of the coarse operator that took, and the seconds.
    Levels {
        pauli_villars: bool,
        applications: usize,
        seconds: f64,
    },
    /// A checksum of what a stage made, computed when asked (on the device, to compare runs bit
    /// by bit).
    Checksum(&'s str, &'s dyn Fn() -> u64),
}

/// The operators a cycle is for: Ã of the first over the second, in the form given.
pub type Outer<'a, O> = (&'a O, &'a O, EoForm);

/// What a backend supplies to the domain-wall multigrid of the even sites; the rest (the
/// propagator runs, `hmc::dwf_multigrid`) is written once.
pub trait MdwfBackend {
    /// D (and D_PV) in double precision.
    type Op: EvenOdd<Self::Half>;
    /// Fields on all sites and on the sites of one parity.
    type Field: Vector;
    type Half: Vector;
    type Split: SplitParity<Self::Field, Self::Half>;
    /// The test vectors as the backend keeps them.
    type Vectors: 'static;
    /// What the cycles are made of: P, the coarse parts with their levels, the tops of the
    /// smoothers of the outer operators. Kept from solve to solve by the cache of HMC.
    type Parts: 'static;

    fn split(&self) -> &Self::Split;
    /// Zero fields of one vector on the even and on the odd sites, for `op`.
    fn halves(&self, op: &Self::Op, lat: &Arc<Lattice>) -> (Self::Half, Self::Half);
    fn gamma5(op: &Self::Op, out: &mut Self::Half, inp: &Self::Half);
    /// `out = Q inp` on all sites, Q at the α of `op`.
    fn mult_q(op: &Self::Op, out: &mut Self::Field, inp: &Self::Field);
    /// D^(α) on the links of `op`.
    fn with_alpha(op: &Self::Op, alpha: f64) -> Result<Self::Op, Box<dyn Error>>;
    /// Test vectors given on the host in double precision (a checkpoint), kept as the backend
    /// keeps them.
    fn vectors_from_host(
        &self,
        y: &Field<f64, Spinor, Half>,
        settings: &MdwfMultigridSettings,
    ) -> Result<Self::Vectors, Box<dyn Error>>;
    /// The test vectors for the plain operators from those `kept` (a full setup without any,
    /// improved by a bootstrap when `due`, else as they are), and from them P, the coarse parts
    /// with the levels the settings ask for, and the tops of the smoothers of `outers`; with the
    /// seconds the vectors took.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn rebuild(
        &self,
        plain: (&Self::Op, &Self::Op),
        outers: &[Outer<'_, Self::Op>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        kept: Option<Self::Vectors>,
        due: bool,
        report: &mut dyn FnMut(SetupStage<'_>),
    ) -> Result<(Self::Vectors, Self::Parts, f64), Box<dyn Error>>;
    /// The cycles of `outers` (those `rebuild` made the parts for), made of `parts` for the
    /// operators on the links of now.
    fn cycles<'a>(
        &'a self,
        plain: (&'a Self::Op, &'a Self::Op),
        outers: &[Outer<'a, Self::Op>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        parts: &'a Self::Parts,
    ) -> Result<Box<dyn Cycles<Self::Half> + 'a>, Box<dyn Error>>;
}

/// The cycles of `MdwfBackend::cycles`.
pub trait Cycles<H> {
    /// The cycle of the `which`-th outer operator.
    fn cycle(&self, which: usize) -> Box<dyn LinearFn<H> + '_>;
    /// Applications of the coarse operators by the cycles so far.
    fn coarse_applications(&self) -> usize;
}

/// The CPU as an `MdwfBackend`: the cycles of `PvEoCycle`, in double precision or
/// (`fine_single`) on the operators converted to single precision.
pub struct OnCpu<G: GammaRepre> {
    index: Arc<EvenOddIndex>,
    _gamma: std::marker::PhantomData<G>,
}

impl<G: GammaRepre> OnCpu<G> {
    pub fn new(op: &Mdwf<f64, G>) -> Self {
        Self {
            index: op.even_odd_index().expect("even extents").clone(),
            _gamma: std::marker::PhantomData,
        }
    }
}

type HalfOf<F> = Field<F, Spinor, Half>;

/// The test vectors of the CPU, in the precision of the cycle.
pub enum CpuVectors {
    Double(HalfOf<f64>),
    Single(HalfOf<f32>),
}

/// The coarse parts in the precision of the coarse solves.
pub enum AnyParts {
    Single(CoarseParts<f32>),
    Double(CoarseParts<f64>),
}

/// What the cycles of the CPU are made of, in the precision `F` of the cycle.
pub struct PartsIn<F: Storage> {
    pub basis: PvEoBasis<F>,
    pub coarse: AnyParts,
    /// The tops of the smoothers of the outer operators.
    pub tops: Vec<f64>,
}

pub enum CpuParts {
    Double(PartsIn<f64>),
    Single(PartsIn<f32>),
}

/// The test vectors for `ops` stored as `F` (`MdwfBackend::rebuild`).
fn vectors_in<F: Storage, G: GammaRepre>(
    (op, pv): (&Mdwf<F, G>, &Mdwf<F, G>),
    lat: &Arc<Lattice>,
    settings: &MdwfMultigridSettings,
    kept: Option<HalfOf<F>>,
    due: bool,
    report: &mut dyn FnMut(SetupStage<'_>),
) -> Result<HalfOf<F>, Box<dyn Error>> {
    let agg = Aggregation::new(lat, settings.block);
    Ok(match kept {
        None => {
            let (y, theta) = pv_eo_low_modes(op, pv, lat, settings.low_modes);
            report(SetupStage::Filtered(&theta));
            let mut y = y.select(&(0..settings.test_vectors).collect::<Vec<_>>());
            pv_eo_bootstrap_stored(op, pv, &agg, &mut y, settings.bootstrap, settings.storage())?;
            y
        }
        Some(mut y) if due => {
            pv_eo_bootstrap_stored(op, pv, &agg, &mut y, settings.bootstrap, settings.storage())?;
            y
        }
        Some(y) => y,
    })
}

/// 1.1 |λ_max| of Ã of `op` over `pv` in `form`: the top of a smoother's interval, as
/// `PvEoCycle` takes it.
pub fn top_of<F: Storage, G: GammaRepre>(
    (op, pv, form): Outer<'_, Mdwf<F, G>>,
    lat: &Arc<Lattice>,
    index: &EvenOddIndex,
    params: PvEoParams,
) -> f64 {
    let mut start = random_even::<F>(lat, index, op.params().ls, 1, params.seed);
    let like = start.zeros_like();
    1.1 * largest_magnitude(
        &mut PvEvenOdd::new(op, pv, form, &like),
        &mut start,
        params.power_steps,
    )
}

/// P and the coarse parts with their levels for `ops` stored as `F` (`MdwfBackend::rebuild`),
/// the tops given.
fn parts_in<F: Storage, G: GammaRepre>(
    (op, pv): (&Mdwf<F, G>, &Mdwf<F, G>),
    lat: &Arc<Lattice>,
    settings: &MdwfMultigridSettings,
    y: &HalfOf<F>,
    tops: Vec<f64>,
    report: &mut dyn FnMut(SetupStage<'_>),
) -> Result<PartsIn<F>, Box<dyn Error>> {
    if settings.levels.for_pauli_villars && settings.cycle.pv_solver != super::CoarseKrylov::Minres
    {
        return Err("MultiGrid.levels_for_pauli_villars needs MINRES for the Pauli-Villars factor (Coarse_Solver_PV)".into());
    }
    let agg = Aggregation::new(lat, settings.block);
    let basis = pv_eo_basis_stored(op, pv, &agg, y, settings.storage())?;
    let (super_blocks, refine) = settings.level_lists();
    let vectors = (!super_blocks.is_empty()).then(|| pv_eo_coarse_vectors(op, pv, &basis, y));
    let outer = (op, pv, EoForm::Right);
    macro_rules! made {
        ($ct:ty, $variant:ident) => {{
            let mut c = PvEoCycle::<G, $ct, crate::Aos, F>::with_coarse_storage(
                op,
                pv,
                outer,
                &basis,
                settings.cycle,
            )?;
            if let Some(v) = &vectors {
                let start = std::time::Instant::now();
                let applications = c.add_levels_with(v, super_blocks, refine)?;
                report(SetupStage::Levels {
                    pauli_villars: false,
                    applications,
                    seconds: start.elapsed().as_secs_f64(),
                });
                if settings.levels.for_pauli_villars {
                    let start = std::time::Instant::now();
                    let applications = c.add_pv_levels(v, super_blocks)?;
                    report(SetupStage::Levels {
                        pauli_villars: true,
                        applications,
                        seconds: start.elapsed().as_secs_f64(),
                    });
                }
            }
            AnyParts::$variant(c.into_parts())
        }};
    }
    let coarse = match settings.coarse_double {
        false => made!(f32, Single),
        true => made!(f64, Double),
    };
    Ok(PartsIn {
        basis,
        coarse,
        tops,
    })
}

/// The cycle of `parts` for `plain` and the outer operator `outer` with the top `hi`, acting
/// on fields stored as `F`.
fn cycle_in<'a, F: Storage, G: GammaRepre>(
    (d, pv): (&'a Mdwf<F, G>, &'a Mdwf<F, G>),
    outer: Outer<'a, Mdwf<F, G>>,
    hi: f64,
    parts: &'a PartsIn<F>,
    params: PvEoParams,
    count: &'a Cell<usize>,
) -> Box<dyn LinearFn<HalfOf<F>> + 'a> {
    macro_rules! counted {
        ($p:expr) => {{
            let c =
                PvEoCycle::from_parts(d, pv, outer, &parts.basis, params, $p.clone().with_top(hi));
            Box::new(CountedCycle {
                cycle: c,
                count,
                before: 0,
            })
        }};
    }
    match &parts.coarse {
        AnyParts::Single(p) => counted!(p),
        AnyParts::Double(p) => counted!(p),
    }
}

/// A cycle that adds its applications of the coarse operators to `count`.
struct CountedCycle<'a, C> {
    cycle: C,
    count: &'a Cell<usize>,
    before: usize,
}

impl<'a, G: GammaRepre, CT: Storage, F: Storage> LinearFn<HalfOf<F>>
    for CountedCycle<'a, PvEoCycle<'a, G, CT, crate::Aos, F>>
{
    fn mult(&mut self, out: &mut HalfOf<F>, inp: &HalfOf<F>) {
        self.cycle.mult(out, inp);
        let now = self.cycle.coarse_applications();
        self.count.set(self.count.get() + now - self.before);
        self.before = now;
    }
}

/// D, D_PV and the outer operators in single precision, on the links of now.
type SingleOps<G> = (
    Mdwf<f32, G>,
    Mdwf<f32, G>,
    Vec<(Mdwf<f32, G>, Mdwf<f32, G>, EoForm)>,
);

/// The cycles of one solve on the CPU: the operators of the cycle (converted to single
/// precision with `fine_single`) and the parts.
struct CpuCycles<'a, G: GammaRepre> {
    plain: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>),
    outers: Vec<Outer<'a, Mdwf<f64, G>>>,
    single: Option<SingleOps<G>>,
    parts: &'a CpuParts,
    params: PvEoParams,
    like: HalfOf<f32>,
    count: Cell<usize>,
}

impl<G: GammaRepre> Cycles<HalfOf<f64>> for CpuCycles<'_, G> {
    fn cycle(&self, which: usize) -> Box<dyn LinearFn<HalfOf<f64>> + '_> {
        match (self.parts, &self.single) {
            (CpuParts::Double(p), _) => {
                let hi = p.tops[which];
                cycle_in(
                    self.plain,
                    self.outers[which],
                    hi,
                    p,
                    self.params,
                    &self.count,
                )
            }
            (CpuParts::Single(p), Some((d, pv, outers))) => {
                let (od, opv, form) = &outers[which];
                let hi = p.tops[which];
                let m = cycle_in((d, pv), (od, opv, *form), hi, p, self.params, &self.count);
                Box::new(Convert::new(m, &self.like))
            }
            (CpuParts::Single(_), None) => unreachable!("made with fine_single"),
        }
    }

    fn coarse_applications(&self) -> usize {
        self.count.get()
    }
}

impl<G: GammaRepre> MdwfBackend for OnCpu<G> {
    type Op = Mdwf<f64, G>;
    type Field = Field<f64, Spinor>;
    type Half = HalfOf<f64>;
    type Split = EvenOddIndex;
    type Vectors = CpuVectors;
    type Parts = CpuParts;

    fn split(&self) -> &EvenOddIndex {
        &self.index
    }

    fn halves(&self, op: &Mdwf<f64, G>, lat: &Arc<Lattice>) -> (HalfOf<f64>, HalfOf<f64>) {
        let half = |parity| {
            let mut h = HalfOf::<f64>::with_shape(lat, 24, op.params().ls, 1, 1);
            h.set_parity(parity);
            h
        };
        (half(Parity::Even), half(Parity::Odd))
    }

    fn gamma5(op: &Mdwf<f64, G>, out: &mut HalfOf<f64>, inp: &HalfOf<f64>) {
        op.mult_gamma5_on(out, inp);
    }

    fn mult_q(op: &Mdwf<f64, G>, out: &mut Field<f64, Spinor>, inp: &Field<f64, Spinor>) {
        op.mult_q(out, inp, false);
    }

    fn with_alpha(op: &Mdwf<f64, G>, alpha: f64) -> Result<Mdwf<f64, G>, Box<dyn Error>> {
        let mut a = Mdwf::converted_from(op);
        a.set_alpha(alpha);
        Ok(a)
    }

    fn vectors_from_host(
        &self,
        y: &HalfOf<f64>,
        settings: &MdwfMultigridSettings,
    ) -> Result<CpuVectors, Box<dyn Error>> {
        Ok(match settings.fine_single {
            false => CpuVectors::Double(y.select(&(0..y.num_vectors()).collect::<Vec<_>>())),
            true => {
                let mut y32 =
                    HalfOf::<f32>::with_shape(y.lattice(), 24, y.ls(), y.num_vectors(), 1);
                y32.set_parity(Parity::Even);
                y32.convert_from(y);
                CpuVectors::Single(y32)
            }
        })
    }

    fn rebuild(
        &self,
        (d, pv): (&Mdwf<f64, G>, &Mdwf<f64, G>),
        outers: &[Outer<'_, Mdwf<f64, G>>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        kept: Option<CpuVectors>,
        due: bool,
        report: &mut dyn FnMut(SetupStage<'_>),
    ) -> Result<(CpuVectors, CpuParts, f64), Box<dyn Error>> {
        let start = std::time::Instant::now();
        let index = &self.index;
        // The tops of the smoothers in the precision of the cycle.
        let tops = |outers: &[Outer<'_, Mdwf<_, G>>]| {
            outers
                .iter()
                .map(|o| top_of(*o, lat, index, settings.cycle))
                .collect::<Vec<_>>()
        };
        Ok(match settings.fine_single {
            false => {
                let kept = match kept {
                    Some(CpuVectors::Double(y)) => Some(y),
                    _ => None,
                };
                let y = vectors_in((d, pv), lat, settings, kept, due, report)?;
                let seconds = start.elapsed().as_secs_f64();
                let parts = parts_in((d, pv), lat, settings, &y, tops(outers), report)?;
                (CpuVectors::Double(y), CpuParts::Double(parts), seconds)
            }
            true => {
                let ops = (Mdwf::converted_from(d), Mdwf::converted_from(pv));
                let outers32: Vec<_> = outers
                    .iter()
                    .map(|(a, b, form)| (Mdwf::converted_from(*a), Mdwf::converted_from(*b), *form))
                    .collect();
                let outers32: Vec<_> = outers32.iter().map(|(a, b, f)| (a, b, *f)).collect();
                let kept = match kept {
                    Some(CpuVectors::Single(y)) => Some(y),
                    _ => None,
                };
                let y = vectors_in((&ops.0, &ops.1), lat, settings, kept, due, report)?;
                let seconds = start.elapsed().as_secs_f64();
                let parts = parts_in((&ops.0, &ops.1), lat, settings, &y, tops(&outers32), report)?;
                (CpuVectors::Single(y), CpuParts::Single(parts), seconds)
            }
        })
    }

    fn cycles<'a>(
        &'a self,
        plain: (&'a Mdwf<f64, G>, &'a Mdwf<f64, G>),
        outers: &[Outer<'a, Mdwf<f64, G>>],
        lat: &Arc<Lattice>,
        settings: &MdwfMultigridSettings,
        parts: &'a CpuParts,
    ) -> Result<Box<dyn Cycles<HalfOf<f64>> + 'a>, Box<dyn Error>> {
        let single = match parts {
            CpuParts::Single(_) => Some((
                Mdwf::converted_from(plain.0),
                Mdwf::converted_from(plain.1),
                outers
                    .iter()
                    .map(|(a, b, form)| (Mdwf::converted_from(*a), Mdwf::converted_from(*b), *form))
                    .collect(),
            )),
            CpuParts::Double(_) => None,
        };
        let mut like = HalfOf::<f32>::with_shape(lat, 24, plain.0.params().ls, 1, 1);
        like.set_parity(Parity::Even);
        Ok(Box::new(CpuCycles {
            plain,
            outers: outers.to_vec(),
            single,
            parts,
            params: settings.cycle,
            like,
            count: Cell::new(0),
        }))
    }
}
