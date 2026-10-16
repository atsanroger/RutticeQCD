//! The lowest singular values of the even-odd Pauli-Villars operator Ã of Möbius domain-wall
//! fermions on the GPU, and the chirality of their vectors: the lowest eigenpairs of Ã†Ã by the
//! thick-restart Lanczos method on a Chebyshev polynomial (`eigen::lowest`); for each, σ = √λ,
//! the residual, ⟨ψ|Γ5 ψ⟩ of the five-dimensional field ψ the vector lifts to (Γ5 = γ5 R) and the
//! chirality ⟨q|γ5 q⟩/⟨q|q⟩ of its four-dimensional quark field q = P− ψ_0 + P+ ψ_(Ls−1).
//! |chirality| ≈ 1 alone does not mark a zero mode: the non-zero modes come in degenerate pairs of
//! opposite chirality (Ã†Ã nearly commutes with γ5 on q). The topological near-zero modes are the
//! unpaired ones below the pairs, and the sum of their chiralities is the charge (v2 12⁴: one
//! unpaired mode, +1, Q = +1; v2 16⁴: none, Q ≈ 0).
//!
//! With `--coherence N1,N2,...`, Lüscher's test of local coherence (JHEP 0707:081) on these
//! WANTED = K modes, with the prolongation of the multigrid (`DeviceBasis`: aggregates of
//! `--block` b⁴ sites on the even sites, every s, split by Γ5 = γ5 R) made of the lowest N:
//! - right: P from the right singular vectors v_0..v_(N−1); |P P† v_j| for j = N..K−1;
//! - left: the same with the left ones, u_j = Ã v_j / σ_j;
//! - cross: P from v, |P P† u_j| for all j < K (what the coarse operator P† Ã P needs: Ã maps
//!   the near-null space of the right vectors onto that of the left ones);
//! - random: |P P† t| of a random vector t.
//!
//! Locally coherent modes give right and left near 1 for N much smaller than K.
//!
//! With `--fov`, first two properties of Ã that the choice of its solvers rests on:
//! - how far it is from Γ5-Hermitian: |Γ5 Ã Γ5 x − Ã† x| / |Ã† x| for a random x, and the same
//!   for the five-dimensional D (|Γ5 D Γ5 x − D† x| / |D† x|), which for c ≠ 0 is not zero;
//! - the real part of its field of values, [λ_min, λ_max] of the Hermitian part (Ã + Ã†)/2 by
//!   Lanczos (no polynomial; `--fov-tol`, 1e-4): λ_min < 0 means the field of values reaches into
//!   the left half plane, where GCR and FGMRES can stagnate and a Chebyshev smoother on a real
//!   interval cannot be trusted.
//!
//! WANTED 0 with `--fov` skips the singular values.
//!
//! ```text
//! ruttice-low-modes LX LY LZ LT FORMAT FILE M5 MASS SMEAR WANTED [--single] [--ls N]
//!     [--b B] [--c C] [--degree D] [--lo R] [--keep K] [--dim M] [--tol T] [--restarts N]
//!     [--coherence N1,N2,...] [--block B] [--fov] [--fov-tol T]
//! ```
//!
//! FORMAT: text, binary, nersc, ildg or scidac; SMEAR 1 for stout ρ = 0.1 × 3 (on the GPU), 0
//! for none. Ls 8, b 1.5, c 0.5, m_PV 1, boundary antiperiodic in t unless given. The polynomial
//! has degree `--degree` (even, 100) on [lo, hi], hi = 1.1 |λ_max| of Ã†Ã by 50 power steps and
//! lo = hi × `--lo` (1e-3). `--single` keeps the fields in single precision (half the memory of
//! the `--dim` + `--keep` Lanczos vectors; the residuals then stop near 1e-6).
//!
//! Memory: about `--dim` + `--keep` + 8 fields on the even sites, each V/2 × Ls × 24 reals (V/2 ×
//! Ls × 192 bytes in double precision): on 24³×48 with Ls 8 that is 0.51 GB a field, so WANTED 8
//! (keep 16, dim 32) needs 28 GB; `--single --keep 10 --dim 20` about 5 GB. The program prints
//! its estimate first.

use ruttice::backend::cuda_bridge::{
    CudaMdwf, DeviceBasis, DeviceLinks, DeviceParities, DeviceReal, DeviceSpinor,
};
use ruttice::eigen::{lowest, Chebyshev, LanczosParams};
use ruttice::fermion::{EoForm, FermionOp, MdwfParams, PvRatio, SplitParity, Variant};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::kernel::Dirac;
use ruttice::measure::chirality;
use ruttice::solver::largest_magnitude;
use ruttice::solver::preconditioner::multigrid::Aggregation;
use ruttice::{
    random_spinor, Boundary, Field, Lattice, LinearFn, LinearOp, Normal, Parity, Spinor, Vector,
    C64, ND,
};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: ruttice-low-modes LX LY LZ LT FORMAT FILE M5 MASS SMEAR WANTED [--single] [--ls N] [--b B] [--c C] [--degree D] [--lo R] [--keep K] [--dim M] [--tol T] [--restarts N] [--coherence N1,N2,...] [--block B] [--fov] [--fov-tol T]";

struct Options {
    dims: [usize; ND],
    format: String,
    file: String,
    m5: f64,
    mass: f64,
    smear: bool,
    wanted: usize,
    single: bool,
    ls: usize,
    b: f64,
    c: f64,
    degree: usize,
    lo: f64,
    keep: usize,
    dim: usize,
    tol: f64,
    restarts: usize,
    coherence: Vec<usize>,
    block: usize,
    fov: bool,
    fov_tol: f64,
}

fn options() -> Result<Options, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 10 {
        return Err(USAGE.into());
    }
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args.get(i + 1));
    let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
        match flag(name) {
            Some(Some(v)) => Ok(v.parse()?),
            Some(None) => Err(format!("{name} needs a value").into()),
            None => Ok(default),
        }
    };
    let wanted: usize = args[9].parse()?;
    Ok(Options {
        dims: [
            args[0].parse()?,
            args[1].parse()?,
            args[2].parse()?,
            args[3].parse()?,
        ],
        format: args[4].clone(),
        file: args[5].clone(),
        m5: args[6].parse()?,
        mass: args[7].parse()?,
        smear: args[8] == "1",
        wanted,
        single: args.iter().any(|a| a == "--single"),
        ls: value("--ls", 8.0)? as usize,
        b: value("--b", 1.5)?,
        c: value("--c", 0.5)?,
        degree: value("--degree", 100.0)? as usize,
        lo: value("--lo", 1e-3)?,
        keep: value("--keep", (2 * wanted).max(wanted + 4) as f64)? as usize,
        dim: value("--dim", (4 * wanted).max(wanted + 12) as f64)? as usize,
        tol: value("--tol", 1e-6)?,
        restarts: value("--restarts", 100.0)? as usize,
        coherence: match flag("--coherence") {
            Some(Some(list)) => list
                .split(',')
                .map(|n| n.trim().parse())
                .collect::<Result<_, _>>()?,
            Some(None) => return Err("--coherence needs a list".into()),
            None => Vec::new(),
        },
        block: value("--block", 4.0)? as usize,
        fov: args.iter().any(|a| a == "--fov"),
        fov_tol: value("--fov-tol", 1e-4)?,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let o = options()?;
    let lat = Lattice::single_node(o.dims);
    let u = ruttice::app::read_gauge_named(&o.format, Path::new(&o.file), &lat)?;
    let mut links = DeviceLinks::from_host(&u)?;
    if o.smear {
        let s = GaugeSmearing {
            smear: Smear::Stout(Stout { rho: uniform(0.1) }),
            steps: 3,
        };
        links = s.levels_on_device(&links)?.pop().expect("three steps");
    }
    println!(
        "# {:?} {} {}; M5 {}, m {}, Ls {}, b {}, c {}; stout 0.1 x 3: {}; {} precision",
        o.dims,
        o.format,
        o.file,
        o.m5,
        o.mass,
        o.ls,
        o.b,
        o.c,
        o.smear,
        if o.single { "single" } else { "double" }
    );
    match o.single {
        true => run::<f32>(&o, &lat, &links),
        false => run::<f64>(&o, &lat, &links),
    }
}

fn run<T: DeviceReal>(
    o: &Options,
    lat: &Arc<Lattice>,
    links: &DeviceLinks,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let fields = o.dim + o.keep + 8;
    let bytes = (lat.num_sites() / 2 * o.ls * 24 * std::mem::size_of::<T>()) as f64;
    println!(
        "# memory: about {fields} fields of {:.2} GB on the even sites, {:.1} GB",
        bytes / 1e9,
        fields as f64 * bytes / 1e9
    );
    let params = |mass: f64| MdwfParams {
        ls: o.ls,
        m5: o.m5,
        mass,
        b: o.b,
        c: o.c,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut d = CudaMdwf::<T>::new(lat, params(o.mass))?;
    d.set_gauge_on_device(links)?;
    let pv = CudaMdwf::on_the_links_of(&d, params(1.0))?;
    let even = DeviceSpinor::<T>::shape_half(lat, o.ls, Parity::Even)?;
    let odd = DeviceSpinor::<T>::shape_half(lat, o.ls, Parity::Odd)?;
    let ratio = || PvRatio::new(&d, &pv, EoForm::Right, &even, &odd);
    let mut normal = Normal::new(ratio(), &even);

    // A random start on the even sites.
    let mut full = Field::<f64, Spinor>::with_shape(lat, 24, o.ls, 1, 1);
    full.as_mut_slice()
        .copy_from_slice(random_spinor(lat, o.ls, 41).as_slice());
    let full = DeviceSpinor::<T>::from_host(&full)?;
    let mut x0 = even.zeros_with(1);
    DeviceParities.extract(&mut x0, &full, Parity::Even);
    let mut power = x0.zeros_like();
    power.copy_from(&x0);
    let hi = 1.1 * largest_magnitude(&mut normal, &mut power, 50);
    let chebyshev = Chebyshev {
        degree: o.degree,
        lo: hi * o.lo,
        hi,
    };
    println!("# hi = 1.1 |lambda_max(A^dag A)| = {hi:.6e}; {chebyshev:?}");
    if o.fov {
        field_of_values(o, lat, &d, &mut ratio(), &x0)?;
        if o.wanted == 0 {
            return Ok(());
        }
    }
    let e = lowest(
        &mut normal,
        &x0,
        LanczosParams {
            wanted: o.wanted,
            keep: o.keep,
            max_dim: o.dim,
            tolerance: o.tol,
            max_restarts: o.restarts,
            chebyshev: Some(chebyshev),
        },
    );
    println!(
        "# Lanczos: converged {}, {} restarts, {} applications of A^dag A, {:.0} s",
        e.converged,
        e.restarts,
        e.applications,
        start.elapsed().as_secs_f64()
    );
    println!(
        "# k  sigma  lambda=sigma^2  |A^dag A x - lambda x|/lambda  <psi|Gamma5 psi>  chirality_4d"
    );
    let zero = DeviceSpinor::<T>::zeros_fifth(lat, o.ls)?;
    let mut r = ratio();
    for (k, ((lambda, res), y)) in e
        .values
        .iter()
        .zip(&e.residuals)
        .zip(&e.vectors)
        .enumerate()
    {
        // ψ = (C_e⁻¹ y, −C_o⁻¹ D_oe C_e⁻¹ y), q = P− ψ_0 + P+ ψ_(Ls−1).
        let mut psi = zero.zeros_like();
        r.solution(&DeviceParities, &mut psi, y, &zero);
        let mut q = DeviceSpinor::<T>::zeros_fifth(lat, 1)?;
        d.propagator_4d(&mut q, &psi);
        println!(
            "{k:3}  {:.6e}  {lambda:.6e}  {:.1e}  {:+.6}  {:+.6}",
            lambda.max(0.0).sqrt(),
            res / lambda.abs(),
            chirality::<Dirac>(&psi.to_host()?, true),
            chirality::<Dirac>(&q.to_host()?, false)
        );
    }
    if !o.coherence.is_empty() {
        coherence(o, lat, &mut r, &e.values, &e.vectors, &x0)?;
    }
    Ok(())
}

/// `s` (Ã + Ã†) / 2: the Hermitian part of Ã, or minus it.
struct HermitianPart<'a, A> {
    a: &'a mut A,
    sign: f64,
}

impl<T: DeviceReal, A: LinearOp<DeviceSpinor<T>>> LinearFn<DeviceSpinor<T>>
    for HermitianPart<'_, A>
{
    fn mult(&mut self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        let mut t = inp.zeros_like();
        self.a.mult(out, inp);
        self.a.mult_dag(&mut t, inp);
        out.axpy(&vec![C64::new(1.0, 0.0); inp.num_vectors()], &t);
        out.scale(&vec![0.5 * self.sign; inp.num_vectors()]);
    }
}

fn norm<T: DeviceReal>(v: &DeviceSpinor<T>) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n[0].sqrt()
}

/// The distance from Γ5-Hermiticity and the real part of the field of values (module
/// documentation).
fn field_of_values<T: DeviceReal>(
    o: &Options,
    lat: &Arc<Lattice>,
    d: &CudaMdwf<T>,
    a: &mut PvRatio<'_, CudaMdwf<T>, DeviceSpinor<T>>,
    x0: &DeviceSpinor<T>,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    // |Γ5 Ã Γ5 x − Ã† x| / |Ã† x| on the even sites.
    let (mut g, mut ag, mut gag, mut adag) = (
        x0.zeros_like(),
        x0.zeros_like(),
        x0.zeros_like(),
        x0.zeros_like(),
    );
    d.mult_gamma5_on(&mut g, x0);
    a.mult(&mut ag, &g);
    d.mult_gamma5_on(&mut gag, &ag);
    a.mult_dag(&mut adag, x0);
    let scale = norm(&adag);
    gag.axpy(&[C64::new(-1.0, 0.0)], &adag);
    let even_defect = norm(&gag) / scale;
    // The same for D on the whole lattice.
    let mut full = Field::<f64, Spinor>::with_shape(lat, 24, o.ls, 1, 1);
    full.as_mut_slice()
        .copy_from_slice(random_spinor(lat, o.ls, 43).as_slice());
    let x = DeviceSpinor::<T>::from_host(&full)?;
    let mut work = FermionOp::work(d);
    let (mut gx, mut dgx, mut gdgx, mut ddag) = (
        x.zeros_like(),
        x.zeros_like(),
        x.zeros_like(),
        x.zeros_like(),
    );
    d.mult_gamma5_on(&mut gx, &x);
    d.mult(&mut work, &mut dgx, &gx, Variant::D);
    d.mult_gamma5_on(&mut gdgx, &dgx);
    d.mult(&mut work, &mut ddag, &x, Variant::Dagger);
    let scale = norm(&ddag);
    gdgx.axpy(&[C64::new(-1.0, 0.0)], &ddag);
    let full_defect = norm(&gdgx) / scale;
    println!(
        "# Gamma5-Hermiticity: |G5 A G5 x - A^dag x|/|A^dag x| = {even_defect:.3e} (even sites, A~); |G5 D G5 x - D^dag x|/|D^dag x| = {full_defect:.3e} (D, whole lattice)"
    );
    // The ends of the spectrum of (Ã + Ã†)/2: the lowest of it and of minus it.
    let params = LanczosParams {
        wanted: 1,
        keep: 8,
        max_dim: 40,
        tolerance: o.fov_tol,
        max_restarts: 2000,
        chebyshev: None,
    };
    let mut ends = Vec::new();
    for sign in [1.0, -1.0] {
        let mut h = HermitianPart { a: &mut *a, sign };
        let e = lowest(&mut h, x0, params);
        ends.push((sign * e.values[0], e.converged, e.restarts, e.applications));
    }
    println!(
        "# field of values: Re in [{:+.6e}, {:+.6e}] (lambda_min, lambda_max of (A + A^dag)/2; converged {} {}, {} + {} restarts, {} + {} applications; {:.0} s)",
        ends[0].0,
        ends[1].0,
        ends[0].1,
        ends[1].1,
        ends[0].2,
        ends[1].2,
        ends[0].3,
        ends[1].3,
        start.elapsed().as_secs_f64()
    );
    if ends[0].0 < 0.0 {
        println!("# lambda_min < 0: the field of values reaches into the left half plane");
    }
    Ok(())
}

/// |π P P† ι w| / |w| for the prolongation `basis`.
fn kept<T: DeviceReal>(basis: &DeviceBasis, w: &DeviceSpinor<T>) -> f64 {
    let mut c = basis.coarse_field::<T>(1);
    basis.restrict(&mut c, w);
    let mut back = w.zeros_like();
    basis.prolong(&mut back, &c);
    let (mut a, mut b) = ([0.0], [0.0]);
    back.norm_sqr(&mut a);
    w.norm_sqr(&mut b);
    (a[0] / b[0]).sqrt()
}

/// The prolongation of the multigrid from the vectors `v` (on the even sites).
fn basis_of<T: DeviceReal>(
    agg: &Arc<Aggregation>,
    ls: usize,
    v: &[DeviceSpinor<T>],
    whole: &DeviceSpinor<T>,
) -> Result<DeviceBasis, Box<dyn Error>> {
    let mut basis = DeviceBasis::empty(agg, ls, v.len(), false)?;
    for (j, y) in v.iter().enumerate() {
        let mut s = whole.zeros_like();
        DeviceParities.insert(&mut s, y);
        basis.add(j, &s);
    }
    basis.finish()?;
    Ok(basis)
}

/// Lüscher's test (module documentation).
fn coherence<T: DeviceReal>(
    o: &Options,
    lat: &Arc<Lattice>,
    a: &mut impl LinearFn<DeviceSpinor<T>>,
    values: &[f64],
    right: &[DeviceSpinor<T>],
    random: &DeviceSpinor<T>,
) -> Result<(), Box<dyn Error>> {
    let k = right.len();
    let agg = Aggregation::new(lat, [o.block; ND]);
    let whole = DeviceSpinor::<T>::zeros_fifth(lat, o.ls)?;
    // u_j = Ã v_j / σ_j, normalised.
    let left: Vec<DeviceSpinor<T>> = right
        .iter()
        .map(|v| {
            let mut u = v.zeros_like();
            a.mult(&mut u, v);
            let mut n = [0.0];
            u.norm_sqr(&mut n);
            u.scale(&[1.0 / n[0].sqrt()]);
            u
        })
        .collect();
    println!(
        "# local coherence: blocks of {}^4, K = {k} modes (sigma {:.3e} .. {:.3e})",
        o.block,
        values[0].max(0.0).sqrt(),
        values[k - 1].max(0.0).sqrt()
    );
    println!("# N  right: mean min at_N  left: mean min at_N  cross: mean min  random");
    let stats = |basis: &DeviceBasis, w: &[DeviceSpinor<T>]| -> (f64, f64, f64) {
        let f: Vec<f64> = w.iter().map(|x| kept(basis, x)).collect();
        let mean = f.iter().sum::<f64>() / f.len() as f64;
        let min = f.iter().copied().fold(f64::INFINITY, f64::min);
        (mean, min, f[0])
    };
    for &n in &o.coherence {
        if n == 0 || n >= k {
            println!("{n:3}  (needs 0 < N < K)");
            continue;
        }
        let p_right = basis_of(&agg, o.ls, &right[..n], &whole)?;
        let p_left = basis_of(&agg, o.ls, &left[..n], &whole)?;
        let r = stats(&p_right, &right[n..]);
        let l = stats(&p_left, &left[n..]);
        let c = stats(&p_right, &left);
        println!(
            "{n:3}  {:.4} {:.4} {:.4}  {:.4} {:.4} {:.4}  {:.4} {:.4}  {:.4}",
            r.0,
            r.1,
            r.2,
            l.0,
            l.1,
            l.2,
            c.0,
            c.1,
            kept(&p_right, random)
        );
    }
    Ok(())
}
