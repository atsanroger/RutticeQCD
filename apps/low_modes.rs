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
//! ```text
//! ruttice-low-modes LX LY LZ LT FORMAT FILE M5 MASS SMEAR WANTED [--single] [--ls N]
//!     [--b B] [--c C] [--degree D] [--lo R] [--keep K] [--dim M] [--tol T] [--restarts N]
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
    CudaMdwf, DeviceLinks, DeviceParities, DeviceReal, DeviceSpinor,
};
use ruttice::eigen::{lowest, Chebyshev, LanczosParams};
use ruttice::fermion::{EoForm, MdwfParams, PvRatio, SplitParity};
use ruttice::gauge::smearing::{uniform, Stout};
use ruttice::gauge::{GaugeSmearing, Smear};
use ruttice::kernel::Dirac;
use ruttice::measure::chirality;
use ruttice::solver::largest_magnitude;
use ruttice::{random_spinor, Boundary, Field, Lattice, Normal, Parity, Spinor, Vector, ND};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: ruttice-low-modes LX LY LZ LT FORMAT FILE M5 MASS SMEAR WANTED [--single] [--ls N] [--b B] [--c C] [--degree D] [--lo R] [--keep K] [--dim M] [--tol T] [--restarts N]";

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
    Ok(())
}
