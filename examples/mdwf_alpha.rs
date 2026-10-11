//! Neff's form D^(α) = D Q (arXiv:2602.14454) on the even sites, without multigrid.
//!
//! ```text
//! cargo run --release --example mdwf_alpha -- CONFIG L M5 MASS [LS] [ALPHAS] [plainpv] [cheb] [cgne]
//! ```
//!
//! Q is diagonal in the fifth dimension (Q_0 = P- + α P+, Q_s = α, Q_(Ls-1) = P+ + α P-)
//! and leaves the 4D quark field alone: D x = b is solved as x = Q x^(α) with
//! D^(α) x^(α) = b. With C^(α) = C Q the site-local blocks and S^(α) = S Q_e the Schur
//! complement on the even sites (S = C_e - D_eo C_o^-1 D_oe), the right-preconditioned
//! S^(α) C_e^(α)-1 = S C_e^-1 does not depend on α: the operator Ã of the multigrid runs
//! sees no α. The left-preconditioned
//!
//!   L^(α) = C_e^(α)-1 S^(α) = Q_e^-1 C_e^-1 S Q_e,   Ã_L = L_PV^(α)† L^(α)
//!
//! does: with the same α in the Pauli-Villars operator Ã_L is similar to
//! (C_PV,e^-1 S_PV)† Q_e^-2 (C_e^-1 S). Printed for each α (default 1,0.8,0.6,0.5,0.4)
//! and three right-hand sides: BiCGStab on Ã_L without a preconditioner, to
//! |r|^2/|b|^2 < 1e-16 of its own system, and even-odd CGNE on S^(α) (normal equations
//! Q S† S Q) and on L^(α) (Q L† Q^-2 L Q, the form of 2602.14454), each to 1e-16 of its
//! own system; the iterations, the Wilson sweeps (Ã_L is two, CGNE two per iteration) and
//! the worst |D x - b|^2/|b|^2 checked with D itself. The right-preconditioned Ã of the
//! multigrid runs once for comparison. `plainpv` keeps α = 1 in the Pauli-Villars
//! operator; `cheb` adds BiCGStab with a Chebyshev polynomial (17 steps, ratio 300; cut off
//! at 300 iterations: on 12⁴ the polynomial for Ã did not suit Ã_L); `cgne` runs only CGNE.

use ruttice::fermion::{solve_even_odd, Dagger, EvenOdd, FermionOp, Mdwf, MdwfParams, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::{
    largest_magnitude, solve_cgne, BiCgStab, Cg, Chebyshev, SolveError, SolveInfo, Solver, Stop,
};
use ruttice::{
    extract_parity, insert_parity, random_spinor, Boundary, EvenOddIndex, Field, Half, Lattice,
    LinearFn, LinearOp, Parity, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_alpha CONFIG L M5 MASS [LS] [ALPHAS] [plainpv] [cheb] [cgne]";

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;
type Work = <Mdwf<f64> as EvenOdd<H>>::Work;
/// α, the right-preconditioned Ã instead of Ã_L, and the Chebyshev steps and ratio.
type Case = (f64, bool, Option<(usize, f64)>);

/// `out = 1 - C_e^-1 D_eo C_o^-1 D_oe` of `op`, or its adjoint
/// `1 - (D†)_eo C_o†^-1 (D†)_oe C_e†^-1`; with `right`, the right-preconditioned
/// `1 - D_eo C_o^-1 D_oe C_e^-1` and its adjoint instead.
fn schur(
    op: &Mdwf<f64>,
    work: &mut Work,
    scratch: (&mut H, &mut H, &mut H),
    out: &mut H,
    inp: &H,
    dagger: Dagger,
    right: bool,
) {
    let (o1, o2, even) = scratch;
    let e = Parity::Even;
    let o = Parity::Odd;
    match (dagger, right) {
        (Dagger::No, false) => {
            op.hop(work, o1, inp, o, Dagger::No);
            op.diag_inv(work, o2, o1, o, Dagger::No);
            op.hop(work, even, o2, e, Dagger::No);
            op.diag_inv(work, out, even, e, Dagger::No);
        }
        (Dagger::Yes, false) => {
            op.diag_inv(work, even, inp, e, Dagger::Yes);
            op.hop(work, o1, even, o, Dagger::Yes);
            op.diag_inv(work, o2, o1, o, Dagger::Yes);
            op.hop(work, out, o2, e, Dagger::Yes);
        }
        (Dagger::No, true) => {
            op.diag_inv(work, even, inp, e, Dagger::No);
            op.hop(work, o1, even, o, Dagger::No);
            op.diag_inv(work, o2, o1, o, Dagger::No);
            op.hop(work, out, o2, e, Dagger::No);
        }
        (Dagger::Yes, true) => {
            op.hop(work, o1, inp, o, Dagger::Yes);
            op.diag_inv(work, o2, o1, o, Dagger::Yes);
            op.hop(work, even, o2, e, Dagger::Yes);
            op.diag_inv(work, out, even, e, Dagger::Yes);
        }
    }
    out.aypx(&vec![C64::new(-1.0, 0.0); inp.num_vectors()], inp);
}

/// Ã_L = L_PV† L (or, with `right`, Ã = S_PV† S of the multigrid runs) on the even sites.
struct HalfA<'a> {
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    right: bool,
    work: Work,
    work_pv: Work,
    o1: H,
    o2: H,
    e1: H,
    e2: H,
}

impl<'a> HalfA<'a> {
    fn new(op: &'a Mdwf<f64>, pv: &'a Mdwf<f64>, right: bool, even: &H, odd: &H) -> Self {
        Self {
            op,
            pv,
            right,
            work: op.even_odd_work(),
            work_pv: pv.even_odd_work(),
            o1: odd.zeros_like(),
            o2: odd.zeros_like(),
            e1: even.zeros_like(),
            e2: even.zeros_like(),
        }
    }

    fn apply(&mut self, out: &mut H, inp: &H, pv: bool, dagger: Dagger) {
        let (op, work) = if pv {
            (self.pv, &mut self.work_pv)
        } else {
            (self.op, &mut self.work)
        };
        schur(
            op,
            work,
            (&mut self.o1, &mut self.o2, &mut self.e1),
            out,
            inp,
            dagger,
            self.right,
        );
    }
}

impl LinearFn<H> for HalfA<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        self.apply(&mut y, inp, false, Dagger::No);
        self.apply(out, &y, true, Dagger::Yes);
        self.e2 = y;
    }
}

impl LinearOp<H> for HalfA<'_> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        self.apply(&mut y, inp, true, Dagger::No);
        self.apply(out, &y, false, Dagger::Yes);
        self.e2 = y;
    }
}

/// The left-preconditioned Schur complement L = 1 - C_e^-1 D_eo C_o^-1 D_oe of the quark
/// operator alone (the form of 2602.14454), for CGNE.
struct Left<'a>(HalfA<'a>);

impl LinearFn<H> for Left<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        self.0.apply(out, inp, false, Dagger::No);
    }
}

impl LinearOp<H> for Left<'_> {
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        self.0.apply(out, inp, false, Dagger::Yes);
    }
}

/// Counts the applications of the operator it wraps.
struct Counted<'a, A> {
    a: &'a mut A,
    count: usize,
}

impl<W: Vector, A: LinearFn<W>> LinearFn<W> for Counted<'_, A> {
    fn mult(&mut self, out: &mut W, inp: &W) {
        self.count += 1;
        self.a.mult(out, inp);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let plain_pv = args.iter().any(|a| a == "plainpv");
    let with_chebyshev = args.iter().any(|a| a == "cheb");
    let only_cgne = args.iter().any(|a| a == "cgne");
    args.retain(|a| a != "plainpv" && a != "cheb" && a != "cgne");
    if !(4..=6).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args
        .get(4)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let alphas: Vec<f64> = args
        .get(5)
        .map_or("1,0.8,0.6,0.5,0.4", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| USAGE)?;

    let lat = Lattice::single_node([l; ND]);
    let u = read_bridge_text(&path, &lat)?;
    let params = MdwfParams {
        ls,
        m5,
        mass,
        b: 1.5,
        c: 0.5,
        boundary: [
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Periodic,
            Boundary::Antiperiodic,
        ],
    };
    let mut plain = Mdwf::<f64>::new(&lat, params);
    plain.set_gauge(&u)?;
    let index: &EvenOddIndex = plain.even_odd_index().ok_or("no even-odd decomposition")?;
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, Pauli-Villars α {}",
        if plain_pv { "1" } else { "= α" }
    );

    let full = |seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, 1, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls, seed).as_slice());
        v
    };
    let half = |parity: Parity| {
        let mut h = H::with_shape(&lat, 24, ls, 1, 1);
        h.set_parity(parity);
        h
    };
    let half_random = |seed: u64| {
        let v = full(seed);
        let mut h = half(Parity::Even);
        extract_parity(&mut h, &v, index, Parity::Even);
        h
    };
    let physical = |x: &V, b: &V| {
        let mut r = b.zeros_like();
        plain.mult(&mut plain.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };
    let (even, odd) = (half(Parity::Even), half(Parity::Odd));
    let stop = Stop {
        tolerance_sqr: 1e-16,
        max_iterations: 4000,
        max_restarts: 1,
    };
    let outcome = |r: &Result<SolveInfo, SolveError>| match r {
        Ok(i) => format!("{}", i.iterations),
        Err(SolveError::NotConverged(i)) => format!("{}*", i.iterations),
        Err(SolveError::Breakdown { iterations, .. }) => format!("{iterations}!"),
        Err(e) => format!("({e})"),
    };
    let seeds = 91..=93;
    println!("α     method                         iterations      sweeps (mean)  worst |Dx-b|^2  |λ_max|  s");

    let mut cases: Vec<Case> = if only_cgne {
        Vec::new()
    } else {
        vec![(1.0, true, None)]
    };
    for &alpha in alphas.iter().filter(|_| !only_cgne) {
        cases.push((alpha, false, None));
        if with_chebyshev {
            cases.push((alpha, false, Some((17, 300.0))));
        }
    }
    for &(alpha, right, preconditioner) in &cases {
        let mut op = Mdwf::<f64>::new(&lat, params);
        op.set_gauge(&u)?;
        op.set_alpha(alpha);
        let mut pv = Mdwf::<f64>::new(
            &lat,
            MdwfParams {
                mass: 1.0,
                ..params
            },
        );
        pv.set_gauge(&u)?;
        if !plain_pv {
            pv.set_alpha(alpha);
        }
        let hi = 1.1
            * largest_magnitude(
                &mut HalfA::new(&op, &pv, right, &even, &odd),
                &mut half_random(0x70B),
                30,
            );
        let start = Instant::now();
        let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
        for seed in seeds.clone() {
            let b = full(seed);
            let (mut b_e, mut b_o) = (half(Parity::Even), half(Parity::Odd));
            extract_parity(&mut b_e, &b, index, Parity::Even);
            extract_parity(&mut b_o, &b, index, Parity::Odd);
            let mut work = op.even_odd_work();
            // b'_e = b_e - D_eo C_o^-1 b_o
            let (mut t_o, mut t_e) = (half(Parity::Odd), half(Parity::Even));
            op.diag_inv(&mut work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
            op.hop(&mut work, &mut t_e, &t_o, Parity::Even, Dagger::No);
            t_e.aypx(&[C64::new(-1.0, 0.0)], &b_e);
            let mut a = HalfA::new(&op, &pv, right, &even, &odd);
            // Left: L x_e = C_e^-1 b'_e; right: S C_e^-1 y = b'_e with x_e = C_e^-1 y.
            let mut lhs = half(Parity::Even);
            if right {
                lhs.as_mut_slice().copy_from_slice(t_e.as_slice());
            } else {
                op.diag_inv(&mut work, &mut lhs, &t_e, Parity::Even, Dagger::No);
            }
            let mut rhs = half(Parity::Even);
            a.apply(&mut rhs, &lhs, true, Dagger::Yes);
            let mut y = half(Parity::Even);
            let mut cheb = preconditioner.map(|(steps, ratio)| {
                Chebyshev::new(
                    HalfA::new(&op, &pv, right, &even, &odd),
                    steps,
                    hi / ratio,
                    hi,
                    &even,
                )
            });
            let mut counted = Counted {
                a: &mut a,
                count: 0,
            };
            // The polynomial assumes a spectrum in [hi/ratio, hi]; where Ã_L leaves it the
            // iteration stalls, so it is cut off early.
            let max_iterations = if preconditioner.is_some() { 300 } else { 4000 };
            let result = BiCgStab::new(Stop {
                max_iterations,
                ..stop
            })
            .solve(
                &mut counted,
                cheb.as_mut().map(|c| c as &mut dyn LinearFn<H>),
                &mut y,
                &rhs,
            );
            let applications =
                counted.count + preconditioner.map_or(0, |(steps, _)| (steps - 1) * counted.count);
            let x_e = if right {
                let mut x_e = half(Parity::Even);
                op.diag_inv(&mut work, &mut x_e, &y, Parity::Even, Dagger::No);
                x_e
            } else {
                y
            };
            // x_o = C_o^-1 (b_o - D_oe x_e), x = Q x^(α)
            op.hop(&mut work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
            t_o.aypx(&[C64::new(-1.0, 0.0)], &b_o);
            let mut x_o = half(Parity::Odd);
            op.diag_inv(&mut work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
            let mut x_alpha = b.zeros_like();
            insert_parity(&mut x_alpha, &x_e, index);
            insert_parity(&mut x_alpha, &x_o, index);
            let mut x = b.zeros_like();
            op.mult_q(&mut x, &x_alpha, false);
            counts.push(outcome(&result));
            sweeps += 2 * applications;
            worst = worst.max(physical(&x, &b));
        }
        let name = format!(
            "BiCGStab {}, {}",
            if right { "Ã (right)" } else { "Ã_L" },
            match preconditioner {
                None => "none".to_string(),
                Some((steps, ratio)) => format!("Cheb {steps}/{ratio}"),
            }
        );
        println!(
            "{alpha:<6}{name:<31}{:<16}{:>8}       {worst:.0e}        {:.3}    {:.0}",
            counts.join(" "),
            sweeps / 3,
            hi / 1.1,
            start.elapsed().as_secs_f64()
        );
    }

    // Even-odd CGNE on S^(α) = S Q_e: its normal equations Q_e S† S Q_e do see α.
    for &alpha in &alphas {
        let mut op = Mdwf::<f64>::new(&lat, params);
        op.set_gauge(&u)?;
        op.set_alpha(alpha);
        let start = Instant::now();
        let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
        for seed in seeds.clone() {
            let b = full(seed);
            let mut x_alpha = b.zeros_like();
            let result = solve_even_odd(&op, index, &mut x_alpha, &b, |s, x_e, r| {
                solve_cgne(&mut Cg::new(stop), s, x_e, r)
            });
            if let Ok(i) | Err(SolveError::NotConverged(i)) = &result {
                sweeps += 2 * i.iterations;
            }
            let mut x = b.zeros_like();
            op.mult_q(&mut x, &x_alpha, false);
            counts.push(outcome(&result));
            worst = worst.max(physical(&x, &b));
        }
        println!(
            "{alpha:<6}{:<31}{:<16}{:>8}       {worst:.0e}                 {:.0}",
            "CGNE, S = C_e - D_eo C_o^-1 D_oe",
            counts.join(" "),
            sweeps / 3,
            start.elapsed().as_secs_f64()
        );

        // CGNE on L^(α) = Q_e^-1 L Q_e (2602.14454): normal equations Q L† Q^-2 L Q.
        let pv = Mdwf::<f64>::new(
            &lat,
            MdwfParams {
                mass: 1.0,
                ..params
            },
        );
        let start = Instant::now();
        let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
        for seed in seeds.clone() {
            let b = full(seed);
            let (mut b_e, mut b_o) = (half(Parity::Even), half(Parity::Odd));
            extract_parity(&mut b_e, &b, index, Parity::Even);
            extract_parity(&mut b_o, &b, index, Parity::Odd);
            let mut work = op.even_odd_work();
            let (mut t_o, mut t_e) = (half(Parity::Odd), half(Parity::Even));
            op.diag_inv(&mut work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
            op.hop(&mut work, &mut t_e, &t_o, Parity::Even, Dagger::No);
            t_e.aypx(&[C64::new(-1.0, 0.0)], &b_e);
            // L x_e = C_e^-1 b'_e
            let mut rhs = half(Parity::Even);
            op.diag_inv(&mut work, &mut rhs, &t_e, Parity::Even, Dagger::No);
            let mut left = Left(HalfA::new(&op, &pv, false, &even, &odd));
            let mut x_e = half(Parity::Even);
            let result = solve_cgne(&mut Cg::new(stop), &mut left, &mut x_e, &rhs);
            if let Ok(i) | Err(SolveError::NotConverged(i)) = &result {
                sweeps += 2 * i.iterations;
            }
            op.hop(&mut work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
            t_o.aypx(&[C64::new(-1.0, 0.0)], &b_o);
            let mut x_o = half(Parity::Odd);
            op.diag_inv(&mut work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
            let mut x_alpha = b.zeros_like();
            insert_parity(&mut x_alpha, &x_e, index);
            insert_parity(&mut x_alpha, &x_o, index);
            let mut x = b.zeros_like();
            op.mult_q(&mut x, &x_alpha, false);
            counts.push(outcome(&result));
            worst = worst.max(physical(&x, &b));
        }
        println!(
            "{alpha:<6}{:<31}{:<16}{:>8}       {worst:.0e}                 {:.0}",
            "CGNE, L = C_e^-1 S",
            counts.join(" "),
            sweeps / 3,
            start.elapsed().as_secs_f64()
        );
    }
    println!("* not converged, ! breakdown");
    Ok(())
}
