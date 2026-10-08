//! An even-odd version of the Pauli-Villars preconditioned operator, on half the lattice.
//!
//! ```text
//! cargo run --release --example mdwf_evenodd_a -- CONFIG L M5 MASS [LS] [cgne|ladder]
//! ```
//!
//! With C the site-local diagonal blocks of D and Y = D C^-1, X = D_PV C_PV^-1 (the paper's
//! A' = X† Y), the Schur complements on the even sites are
//!
//!   S_Y = 1 - D_eo C_o^-1 D_oe C_e^-1,    S_X = the same for D_PV,
//!
//! and the Schur complement of D is S_D = S_Y C_e. The half-lattice operator is
//!
//!   Ã = S_X† S_Y,
//!
//! of the same form as A' (S_X† S_X at m = M_PV). D x = b is solved as
//! Ã y = S_X† b'_e with b'_e = b_e - D_eo C_o^-1 b_o, then x_e = C_e^-1 y,
//! x_o = C_o^-1 (b_o - D_oe x_e). One application of Ã is four half-lattice hops, two
//! Wilson sweeps, the same as A'.
//!
//! Printed for five right-hand sides: BiCGStab on A' and on Ã without a preconditioner and
//! with Chebyshev polynomials (17 steps, ratio 300; 33 steps, ratio 1000), to
//! |r|^2/|b|^2 < 1e-16 of the own system, with |D x - b|^2/|b|^2 and the Wilson sweeps
//! (one application of either operator is two). Even-odd CGNE for comparison, to the
//! physical residual 1e-16. Also Re<v, Ã v>/|v|^2 for random and low-mode vectors.
//! `cgne` runs only the CGNE comparison; `ladder` runs only BiCGStab on Ã without a
//! preconditioner to the tolerances 1e-16 .. 1e-19, to match the physical residual of CGNE.

use ruttice::fermion::{
    solve_even_odd, Dagger, EvenOdd, FermionOp, Mdwf, MdwfParams, PvPreconditioned, Variant,
};
use ruttice::io::read_bridge_text;
use ruttice::solver::{
    largest_magnitude, solve_cgne, BiCgStab, Cg, Chebyshev, ChebyshevFilter, SolveError, SolveInfo,
    Solver, Stop,
};
use ruttice::{
    extract_parity, insert_parity, random_spinor, Boundary, EvenOddIndex, Field, Half, Lattice,
    LinearFn, LinearOp, Normal, Parity, Spinor, Vector, C64, ND,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_evenodd_a CONFIG L M5 MASS [LS] [cgne|ladder]";

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;

/// `out = 1 - D_eo C_o^-1 D_oe C_e^-1` of `op` (S_Y for the quark mass, S_X for the
/// Pauli-Villars operator), or its adjoint `1 - C_e†^-1 (D†)_eo C_o†^-1 (D†)_oe`.
fn schur(
    op: &Mdwf<f64>,
    work: &mut <Mdwf<f64> as EvenOdd<H>>::Work,
    odd: (&mut H, &mut H),
    even: &mut H,
    out: &mut H,
    inp: &H,
    dagger: Dagger,
) {
    let (o1, o2) = odd;
    match dagger {
        Dagger::No => {
            op.diag_inv(work, even, inp, Parity::Even, Dagger::No);
            op.hop(work, o1, even, Parity::Odd, Dagger::No);
            op.diag_inv(work, o2, o1, Parity::Odd, Dagger::No);
            op.hop(work, out, o2, Parity::Even, Dagger::No);
        }
        Dagger::Yes => {
            op.hop(work, o1, inp, Parity::Odd, Dagger::Yes);
            op.diag_inv(work, o2, o1, Parity::Odd, Dagger::Yes);
            op.hop(work, even, o2, Parity::Even, Dagger::Yes);
            op.diag_inv(work, out, even, Parity::Even, Dagger::Yes);
        }
    }
    out.aypx(&vec![C64::new(-1.0, 0.0); inp.num_vectors()], inp);
}

/// Ã = S_X† S_Y on the even sites.
struct HalfA<'a> {
    op: &'a Mdwf<f64>,
    pv: &'a Mdwf<f64>,
    work: <Mdwf<f64> as EvenOdd<H>>::Work,
    work_pv: <Mdwf<f64> as EvenOdd<H>>::Work,
    o1: H,
    o2: H,
    e1: H,
    e2: H,
}

impl<'a> HalfA<'a> {
    fn new(op: &'a Mdwf<f64>, pv: &'a Mdwf<f64>, even: &H, odd: &H) -> Self {
        Self {
            op,
            pv,
            work: op.even_odd_work(),
            work_pv: pv.even_odd_work(),
            o1: odd.zeros_like(),
            o2: odd.zeros_like(),
            e1: even.zeros_like(),
            e2: even.zeros_like(),
        }
    }

    /// `out = S_X† inp`.
    fn x_dagger(&mut self, out: &mut H, inp: &H) {
        schur(
            self.pv,
            &mut self.work_pv,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            out,
            inp,
            Dagger::Yes,
        );
    }
}

impl LinearFn<H> for HalfA<'_> {
    fn mult(&mut self, out: &mut H, inp: &H) {
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        schur(
            self.op,
            &mut self.work,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            &mut y,
            inp,
            Dagger::No,
        );
        self.x_dagger(out, &y);
        self.e2 = y;
    }
}

impl LinearOp<H> for HalfA<'_> {
    /// `out = Ã† inp = S_Y† S_X inp`.
    fn mult_dag(&mut self, out: &mut H, inp: &H) {
        let mut y = std::mem::replace(&mut self.e2, inp.zeros_like());
        schur(
            self.pv,
            &mut self.work_pv,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            &mut y,
            inp,
            Dagger::No,
        );
        schur(
            self.op,
            &mut self.work,
            (&mut self.o1, &mut self.o2),
            &mut self.e1,
            out,
            &y,
            Dagger::Yes,
        );
        self.e2 = y;
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
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let mut op = Mdwf::<f64>::new(&lat, params);
    op.set_gauge(&u)?;
    let mut pv = Mdwf::<f64>::new(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    pv.set_gauge(&u)?;
    let index: &EvenOddIndex = op.even_odd_index().ok_or("no even-odd decomposition")?;
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}");

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
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };
    let (even, odd) = (half(Parity::Even), half(Parity::Odd));

    let mode = args.get(5).map(String::as_str);
    let only_cgne = mode == Some("cgne") || mode == Some("ladder");
    // Positivity: Re<v, Ã v>/|v|^2 on random and low-mode vectors.
    if !only_cgne {
        let mut a = HalfA::new(&op, &pv, &even, &odd);
        let hi = 1.1
            * largest_magnitude(
                &mut Normal::new(HalfA::new(&op, &pv, &even, &odd), &even),
                &mut half_random(5),
                50,
            );
        let mut filter = ChebyshevFilter::new(
            Normal::new(HalfA::new(&op, &pv, &even, &odd), &even),
            200,
            hi / 1000.0,
            hi,
            &even,
        );
        let mut line = String::from("Re<v, Ã v>/|v|^2:");
        for seed in [301, 302, 303] {
            let r = half_random(seed);
            let mut low = r.zeros_like();
            filter.mult(&mut low, &r);
            for (name, v) in [("random", &r), ("low", &low)] {
                let mut av = v.zeros_like();
                a.mult(&mut av, v);
                let (mut d, mut n) = ([C64::ZERO], [0.0]);
                v.dot(&av, &mut d);
                v.norm_sqr(&mut n);
                line += &format!("  {name} {:+.4}", d[0].re / n[0]);
            }
        }
        println!("{line}");
    }

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
    let hi_full = 1.1
        * largest_magnitude(
            &mut PvPreconditioned::new(&op, &full(1)),
            &mut full(0x70B),
            30,
        );
    let hi_half = 1.1
        * largest_magnitude(
            &mut HalfA::new(&op, &pv, &even, &odd),
            &mut half_random(0x70B),
            30,
        );
    println!(
        "|lambda_max|: A' {:.3}, Ã {:.3}",
        hi_full / 1.1,
        hi_half / 1.1
    );
    println!("method                                  iterations (seeds 91..95)   sweeps (mean)  worst |Dx-b|^2  s");
    let preconditioners: &[Option<(usize, f64)>] = if only_cgne {
        &[]
    } else {
        &[None, Some((17, 300.0)), Some((33, 1000.0))]
    };
    for &preconditioner in preconditioners {
        for on_half in [false, true] {
            let start = Instant::now();
            let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
            for seed in 91..=95 {
                let b = full(seed);
                let mut x = b.zeros_like();
                let (result, applications) = if on_half {
                    let (mut b_e, mut b_o) = (half(Parity::Even), half(Parity::Odd));
                    extract_parity(&mut b_e, &b, index, Parity::Even);
                    extract_parity(&mut b_o, &b, index, Parity::Odd);
                    let mut work = op.even_odd_work();
                    // b'_e = b_e - D_eo C_o^-1 b_o
                    let (mut t_o, mut t_e) = (half(Parity::Odd), half(Parity::Even));
                    op.diag_inv(&mut work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
                    op.hop(&mut work, &mut t_e, &t_o, Parity::Even, Dagger::No);
                    t_e.aypx(&[C64::new(-1.0, 0.0)], &b_e);
                    let mut a = HalfA::new(&op, &pv, &even, &odd);
                    let mut rhs = half(Parity::Even);
                    a.x_dagger(&mut rhs, &t_e);
                    let mut y = half(Parity::Even);
                    let mut cheb = preconditioner.map(|(steps, ratio)| {
                        Chebyshev::new(
                            HalfA::new(&op, &pv, &even, &odd),
                            steps,
                            hi_half / ratio,
                            hi_half,
                            &even,
                        )
                    });
                    let mut counted = Counted {
                        a: &mut a,
                        count: 0,
                    };
                    let result = BiCgStab::new(stop).solve(
                        &mut counted,
                        cheb.as_mut().map(|c| c as &mut dyn LinearFn<H>),
                        &mut y,
                        &rhs,
                    );
                    let applications = counted.count
                        + preconditioner.map_or(0, |(steps, _)| (steps - 1) * counted.count);
                    // x_e = C_e^-1 y, x_o = C_o^-1 (b_o - D_oe x_e)
                    let mut x_e = half(Parity::Even);
                    op.diag_inv(&mut work, &mut x_e, &y, Parity::Even, Dagger::No);
                    op.hop(&mut work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
                    t_o.aypx(&[C64::new(-1.0, 0.0)], &b_o);
                    let mut x_o = half(Parity::Odd);
                    op.diag_inv(&mut work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
                    insert_parity(&mut x, &x_e, index);
                    insert_parity(&mut x, &x_o, index);
                    (result, applications)
                } else {
                    let mut a = PvPreconditioned::new(&op, &b);
                    let mut rhs = b.zeros_like();
                    a.right_hand_side(&mut rhs, &b);
                    let mut y = b.zeros_like();
                    let mut cheb = preconditioner.map(|(steps, ratio)| {
                        Chebyshev::new(
                            PvPreconditioned::new(&op, &b),
                            steps,
                            hi_full / ratio,
                            hi_full,
                            &b,
                        )
                    });
                    let (result, count) = {
                        let mut counted = Counted {
                            a: &mut a,
                            count: 0,
                        };
                        let result = BiCgStab::new(stop).solve(
                            &mut counted,
                            cheb.as_mut().map(|c| c as &mut dyn LinearFn<V>),
                            &mut y,
                            &rhs,
                        );
                        (result, counted.count)
                    };
                    a.solution(&mut x, &y);
                    let applications =
                        count + preconditioner.map_or(0, |(steps, _)| (steps - 1) * count);
                    (result, applications)
                };
                counts.push(outcome(&result));
                sweeps += 2 * applications;
                worst = worst.max(physical(&x, &b));
            }
            let name = format!(
                "BiCGStab, {}, {}",
                if on_half { "Ã (half)" } else { "A' (full)" },
                match preconditioner {
                    None => "no preconditioner".to_string(),
                    Some((steps, ratio)) => format!("Chebyshev {steps}/{ratio}"),
                }
            );
            println!(
                "{name:<40}{:<28}{:>8}       {worst:.0e}    {:.0}",
                counts.join(" "),
                sweeps / 5,
                start.elapsed().as_secs_f64()
            );
        }
    }

    if mode == Some("ladder") {
        println!("BiCGStab on Ã, no preconditioner:  tol   iterations (seeds 91..95)   sweeps (mean)  worst |Dx-b|^2");
        for tol in [1e-16, 1e-17, 1e-18, 1e-19] {
            let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
            for seed in 91..=95 {
                let b = full(seed);
                let (mut b_e, mut b_o) = (half(Parity::Even), half(Parity::Odd));
                extract_parity(&mut b_e, &b, index, Parity::Even);
                extract_parity(&mut b_o, &b, index, Parity::Odd);
                let mut work = op.even_odd_work();
                let (mut t_o, mut t_e) = (half(Parity::Odd), half(Parity::Even));
                op.diag_inv(&mut work, &mut t_o, &b_o, Parity::Odd, Dagger::No);
                op.hop(&mut work, &mut t_e, &t_o, Parity::Even, Dagger::No);
                t_e.aypx(&[C64::new(-1.0, 0.0)], &b_e);
                let mut a = HalfA::new(&op, &pv, &even, &odd);
                let mut rhs = half(Parity::Even);
                a.x_dagger(&mut rhs, &t_e);
                let mut y = half(Parity::Even);
                let result = BiCgStab::new(Stop {
                    tolerance_sqr: tol,
                    ..stop
                })
                .solve(&mut a, None, &mut y, &rhs);
                if let Ok(i) | Err(SolveError::NotConverged(i)) = &result {
                    sweeps += 4 * i.iterations;
                }
                let mut x = b.zeros_like();
                let mut x_e = half(Parity::Even);
                op.diag_inv(&mut work, &mut x_e, &y, Parity::Even, Dagger::No);
                op.hop(&mut work, &mut t_o, &x_e, Parity::Odd, Dagger::No);
                t_o.aypx(&[C64::new(-1.0, 0.0)], &b_o);
                let mut x_o = half(Parity::Odd);
                op.diag_inv(&mut work, &mut x_o, &t_o, Parity::Odd, Dagger::No);
                insert_parity(&mut x, &x_e, index);
                insert_parity(&mut x, &x_o, index);
                counts.push(outcome(&result));
                worst = worst.max(physical(&x, &b));
            }
            println!(
                "                                   {tol:.0e}   {:<28}{:>8}       {worst:.0e}",
                counts.join(" "),
                sweeps / 5
            );
        }
        return Ok(());
    }

    // Even-odd CGNE: an iteration is two Wilson sweeps.
    let start = Instant::now();
    let (mut counts, mut sweeps, mut worst) = (Vec::new(), 0usize, 0.0_f64);
    for seed in 91..=95 {
        let b = full(seed);
        let mut x = b.zeros_like();
        let result = solve_even_odd(&op, index, &mut x, &b, |s, x_e, r| {
            solve_cgne(&mut Cg::new(stop), s, x_e, r)
        });
        if let Ok(i) | Err(SolveError::NotConverged(i)) = &result {
            sweeps += 2 * i.iterations;
        }
        counts.push(outcome(&result));
        worst = worst.max(physical(&x, &b));
    }
    println!(
        "{:<40}{:<28}{:>8}       {worst:.0e}    {:.0}",
        "CGNE, even-odd",
        counts.join(" "),
        sweeps / 5,
        start.elapsed().as_secs_f64()
    );
    println!("* not converged, ! breakdown");
    Ok(())
}
