//! The Pauli-Villars preconditioned operator as the paper defines it against the one the
//! library implements.
//!
//! - Library (`PvPreconditioned`): A = (D_PV C_PV^-1)† C^-1 D; solve A x = P b with
//!   P = (D_PV C_PV^-1)† C^-1.
//! - Paper (Kanamori, Chen, Matsufuru, PoS(LATTICE2024)414, Fig. 4 left):
//!   A' = (D_PV C_PV^-1)† D C^-1, C^-1 a right preconditioner; solve A' y = (D_PV C_PV^-1)† b,
//!   then x = C^-1 y. With X = D_PV C_PV^-1 and Y = D C^-1 it is X† Y, which is X† X at
//!   m = M_PV.
//!
//! ```text
//! cargo run --release --example mdwf_paper_a -- CONFIG L M5 MASS [LS] [BLOCK] [library|paper]
//! ```
//!
//! For both: BiCGStab without a preconditioner to tolerances 1e-10, 1e-14, 1e-18 of its own
//! system, with |D x - b|^2/|b|^2; what GMRES(4) and two-grid cycles leave of a random and
//! a smooth vector, |v - A M v|; and BiCGStab and FGMRES(20) preconditioned by the cycles
//! to 1e-16 of the own system. The cycles smooth with GMRES(4) on the operator and take
//! 8 random test vectors; the coarse operator is either built from the parts of D with a
//! site-local right factor (C_PV^-1 for the library's A, as the reference does; C^-1 for
//! A') or the probed Galerkin projection of the operator.

use ruttice::fermion::{
    FermionOp, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, SiteInverse, Variant,
};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, CycleParts, Multigrid, SetupHooks,
};
use ruttice::solver::{BiCgStab, Fgmres, Solve, SolveError, Solver, Stop};
use ruttice::{fit, random_spinor, Boundary, Field, Lattice, LinearFn, Spinor, Vector, C64, ND};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: mdwf_paper_a CONFIG L M5 MASS [LS] [BLOCK] [library|paper]";

type V = Field<f64, Spinor>;

/// The paper's A' = (D_PV C_PV^-1)† D C^-1 = C_PV†^-1 D_PV† D C^-1.
struct PaperA<'a> {
    op: &'a Mdwf<f64>,
    work: ruttice::fermion::MdwfWork<f64>,
    first: V,
    second: V,
}

impl<'a> PaperA<'a> {
    fn new(op: &'a Mdwf<f64>, like: &V) -> Self {
        Self {
            op,
            work: op.work(),
            first: like.zeros_like(),
            second: like.zeros_like(),
        }
    }

    /// `out = (D_PV C_PV^-1)† inp`: the right-hand side of A' y = X† b.
    fn x_dagger(&mut self, out: &mut V, inp: &V) {
        fit(&mut self.first, inp);
        self.op
            .mult_at(&mut self.work, &mut self.first, inp, Variant::Dagger, 1.0);
        self.op.mult_site_inv(out, &self.first, 1.0, true);
    }
}

impl LinearFn<V> for PaperA<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        let mass = self.op.params().mass;
        fit(&mut self.first, inp);
        fit(&mut self.second, inp);
        self.op.mult_site_inv(&mut self.second, inp, mass, false);
        self.op
            .mult_at(&mut self.work, &mut self.first, &self.second, Variant::D, mass);
        self.op
            .mult_at(&mut self.work, &mut self.second, &self.first, Variant::Dagger, 1.0);
        self.op.mult_site_inv(out, &self.second, 1.0, true);
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Which {
    Library,
    Paper,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=7).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let ls: usize = args.get(4).map_or(Ok(8), |s| s.parse()).map_err(|_| USAGE)?;
    let block: usize = args.get(5).map_or(Ok(3), |s| s.parse()).map_err(|_| USAGE)?;

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
    println!("{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}, aggregates {block}^4");

    let random = |n: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, n, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * n, seed).as_slice());
        v
    };
    let normalise = |v: &mut V| {
        let mut n = [0.0];
        v.norm_sqr(&mut n);
        v.scale(&[1.0 / n[0].sqrt()]);
    };
    let b = random(1, 91);
    let physical = |x: &V| {
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut left, mut size) = ([0.0], [0.0]);
        r.norm_sqr(&mut left);
        b.norm_sqr(&mut size);
        left[0] / size[0]
    };
    let operator = |which: Which| -> Box<dyn LinearFn<V> + '_> {
        match which {
            Which::Library => Box::new(PvPreconditioned::new(&op, &b)),
            Which::Paper => Box::new(PaperA::new(&op, &b)),
        }
    };
    // The right-hand side of the system and the map from its solution to x.
    let rhs = |which: Which| {
        let mut r = b.zeros_like();
        match which {
            Which::Library => PvPreconditioned::new(&op, &b).precondition(&mut r, &b),
            Which::Paper => PaperA::new(&op, &b).x_dagger(&mut r, &b),
        }
        r
    };
    let solution = |which: Which, y: &V| {
        let mut x = y.zeros_like();
        match which {
            Which::Library => x.copy_from(y),
            Which::Paper => op.mult_site_inv(&mut x, y, mass, false),
        }
        x
    };
    let outcome = |result: Result<ruttice::solver::SolveInfo, SolveError>| match result {
        Ok(i) => format!("{:4}", i.iterations),
        Err(SolveError::NotConverged(i)) => format!("{:4} (not converged)", i.iterations),
        Err(e) => format!("failed: {e}"),
    };

    let only = args.get(6).map(String::as_str);
    for which in [Which::Library, Which::Paper] {
        let skip = match which {
            Which::Library => only == Some("paper"),
            Which::Paper => only == Some("library"),
        };
        if skip {
            continue;
        }
        let name = match which {
            Which::Library => "library A = (D_PV C_PV^-1)† C^-1 D",
            Which::Paper => "paper   A' = (D_PV C_PV^-1)† D C^-1",
        };
        println!("{name}");
        let r = rhs(which);
        for tol in [1e-10, 1e-14, 1e-18] {
            let mut y = b.zeros_like();
            let mut a = operator(which);
            let result = BiCgStab::new(Stop::simple(tol, 4000)).solve(&mut *a, None, &mut y, &r);
            println!(
                "  BiCGStab, no preconditioner, {tol:.0e}: {} iterations, |Dx-b|^2/|b|^2 {:.1e}",
                outcome(result),
                physical(&solution(which, &y))
            );
        }

        let mut rough = random(1, 301);
        normalise(&mut rough);
        let mut smooth = rough.zeros_like();
        {
            let v = random(1, 302);
            let mut a = operator(which);
            let _ = BiCgStab::new(Stop::simple(1e-12, 4000)).solve(&mut *a, None, &mut smooth, &v);
            normalise(&mut smooth);
        }
        let leaves = |m: &mut dyn LinearFn<V>| {
            let mut a = operator(which);
            let mut out = String::new();
            for v in [&rough, &smooth] {
                let (mut mv, mut amv) = (v.zeros_like(), v.zeros_like());
                m.mult(&mut mv, v);
                a.mult(&mut amv, &mv);
                let mut along = [C64::ZERO];
                v.dot(&amv, &mut along);
                amv.aypx(&[C64::new(-1.0, 0.0)], v);
                let mut left = [0.0];
                amv.norm_sqr(&mut left);
                out += &format!("  {:.3} (Re<v,AMv> {:+.3})", left[0].sqrt(), along[0].re);
            }
            out
        };
        let gmres4 = || -> Box<dyn LinearFn<V> + '_> {
            Box::new(Solve::new(Fgmres::new(Stop::simple(1e-6, 4)), Boxed(operator(which))))
        };
        println!("  |v-AMv| on a random and a smooth vector:");
        println!("    none        {}", leaves(&mut Identity));
        println!("    GMRES(4)    {}", leaves(&mut *gmres4()));

        for galerkin in [false, true] {
            let parts = CycleParts {
                operator: operator(which),
                pre_smoother: Some(gmres4()),
                post_smoother: gmres4(),
                coarse_solver: Box::new(Fgmres::new(Stop {
                    tolerance_sqr: 1e-4,
                    max_iterations: 5,
                    max_restarts: 1,
                })),
            };
            let mut hooks = SetupHooks {
                precondition_coarse_diagonal: true,
                ..SetupHooks::default()
            };
            let coarse_name = if galerkin {
                hooks.galerkin_of = Some(operator(which));
                "Galerkin, probed".to_string()
            } else {
                // The reference multiplies the test vectors of its A by D_PV; for A' the
                // natural right factor is C^-1 at the quark mass.
                let factor_mass = match which {
                    Which::Library => {
                        hooks.after_smoothing = Some(Box::new(MdwfAt::new(&op, Variant::D, 1.0)));
                        1.0
                    }
                    Which::Paper => mass,
                };
                hooks.coarse_right = Some(Box::new(SiteInverse {
                    op: &op,
                    mass: factor_mass,
                    dagger: false,
                }));
                format!("P†D C^-1 P at mass {factor_mass}")
            };
            let agg = Aggregation::new(&lat, [block; ND]);
            let mut mg = Multigrid::with_parts(&op, &agg, random(8, 1), 0, parts, hooks)?;
            println!("    cycle, {coarse_name:<22}{}", leaves(&mut mg));
            let stop = Stop {
                tolerance_sqr: 1e-16,
                max_iterations: 1500,
                max_restarts: 1,
            };
            let mut line = String::new();
            for flexible in [false, true] {
                let mut y = b.zeros_like();
                let mut a = operator(which);
                let result = if flexible {
                    Fgmres::new(Stop {
                        max_iterations: 20,
                        max_restarts: 100,
                        ..stop
                    })
                    .solve(&mut *a, Some(&mut mg), &mut y, &r)
                } else {
                    BiCgStab::new(stop).solve(&mut *a, Some(&mut mg), &mut y, &r)
                };
                line += &format!(
                    "  {}: {}, |Dx-b|^2/|b|^2 {:.1e}",
                    if flexible { "FGMRES(20)" } else { "BiCGStab" },
                    outcome(result),
                    physical(&solution(which, &y))
                );
            }
            println!("     {line}");
        }
    }
    Ok(())
}

/// M = 1.
struct Identity;

impl LinearFn<V> for Identity {
    fn mult(&mut self, out: &mut V, inp: &V) {
        out.copy_from(inp);
    }
}

/// A boxed map as a map, for members that take one by value.
struct Boxed<'a>(Box<dyn LinearFn<V> + 'a>);

impl LinearFn<V> for Boxed<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.0.mult(out, inp);
    }
}
