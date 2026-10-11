//! The domain-wall multigrid with genuine low modes of A as test vectors: the lowest right
//! singular vectors computed by `mdwf_coherence_eig`, read from its checkpoint.
//!
//! ```text
//! cargo run --release --example mdwf_eigmg -- CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK] [N...]
//! ```
//!
//! For each number N of test vectors (default 8 16), P is built from the lowest N Ritz
//! vectors of the checkpoint, split by Γ5 over aggregates BLOCK^4 (default 3) that keep
//! every s. Printed:
//! - the exact coarse correction P (R A P)⁻¹ R A on the next eight Ritz vectors (low modes
//!   not in P): what it leaves of each and of its residual, against the floor
//!   √(1 − |PP†t|²), for three restrictions: Galerkin R = P†; Petrov-Galerkin R = Q† with Q
//!   from the left singular vectors u = A v / |A v|; and R = P†A†, which minimises the
//!   residual over span(P). Galerkin is the test the coarse correction failed with random
//!   and inverse-iteration vectors (log §4: 0.99 left against a floor of 0.88).
//! - BiCGStab and FGMRES(20) on A y = X† b to |r|²/|b|² < 1e-16 preconditioned by
//!   (s) the smoother alone, as a cycle without coarse correction;
//!   (p) the library cycle with the probed Galerkin coarse operator (nearest neighbours);
//!   (e) the same cycle with the exact coarse operator R A P, applied through the fine
//!   lattice, for each of the three restrictions;
//!   with the outer iterations, the applications of A on the fine lattice (all of them:
//!   outer, smoothers, residuals, exact coarse operator) and the residual of D x = b.
//!   The smoother is GMRES(4) before and after, as in the log §3.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{
    Aggregation, Basis, CycleParts, Multigrid, SetupHooks,
};
use ruttice::solver::{BiCgStab, Fgmres, Solve, SolveError, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, LinearOp, Spinor, Vector, C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_eigmg CONFIG L M5 MASS CHECKPOINT [LS] [BLOCK] [N...]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

type Count = Rc<Cell<usize>>;

/// A that counts the vectors it is applied to.
struct Counted<'a> {
    a: PvPreconditioned<'a, f64>,
    n: Count,
}

impl<'a> Counted<'a> {
    fn new(op: &'a Mdwf<f64>, like: &V, n: &Count) -> Self {
        Self {
            a: PvPreconditioned::new(op, like),
            n: n.clone(),
        }
    }
}

impl LinearFn<V> for Counted<'_> {
    fn mult(&mut self, out: &mut V, inp: &V) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

impl Counted<'_> {
    fn mult_dag(&mut self, out: &mut V, inp: &V) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult_dag(out, inp);
    }
}

/// The restriction R of the coarse correction e ← e − P (R A P)⁻¹ R A e.
#[derive(Clone, Copy, PartialEq)]
enum Restriction {
    /// R = P†: Galerkin.
    Galerkin,
    /// R = Q†, Q from the left singular vectors u = A v / σ: Petrov-Galerkin, with A
    /// mapping span(P) onto span(Q) for exact singular vectors.
    Left,
    /// R = P† A†: the correction minimises |A e| over span(P); R A P = P†A†AP is
    /// Hermitian positive definite.
    MinRes,
}

impl Restriction {
    fn name(self) -> &'static str {
        match self {
            Restriction::Galerkin => "G  R = P†",
            Restriction::Left => "L  R = Q† (left vectors)",
            Restriction::MinRes => "M  R = P†A†",
        }
    }
}

/// GMRES(4) on A, the smoother of the log §3.
fn gmres4<'a>(op: &'a Mdwf<f64>, like: &V, n: &Count) -> Box<dyn LinearFn<V> + 'a> {
    Box::new(Solve::new(
        Fgmres::new(Stop::simple(1e-6, 4)),
        Counted::new(op, like, n),
    ))
}

/// R A P through the fine lattice.
struct Exact<'a, 'b> {
    a: Counted<'a>,
    p: &'b Basis<f64>,
    q: &'b Basis<f64>,
    kind: Restriction,
    fine: V,
    a_fine: V,
}

/// `out = R r`, with `tmp` as scratch.
fn restrict(
    kind: Restriction,
    [p, q]: [&Basis<f64>; 2],
    a: &mut Counted<'_>,
    tmp: &mut V,
    out: &mut C,
    r: &V,
) {
    match kind {
        Restriction::Galerkin => p.restrict(out, r),
        Restriction::Left => q.restrict(out, r),
        Restriction::MinRes => {
            a.mult_dag(tmp, r);
            p.restrict(out, tmp);
        }
    }
}

impl Exact<'_, '_> {
    /// `out = R r`.
    fn restrict(&mut self, out: &mut C, r: &V) {
        restrict(
            self.kind,
            [self.p, self.q],
            &mut self.a,
            &mut self.fine,
            out,
            r,
        );
    }
}

impl LinearFn<C> for Exact<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.p.prolong(&mut self.fine, inp);
        self.a.mult(&mut self.a_fine, &self.fine);
        restrict(
            self.kind,
            [self.p, self.q],
            &mut self.a,
            &mut self.fine,
            out,
            &self.a_fine,
        );
    }
}

/// The two-level cycle written out: x = S r; x += P c with (R A P) c = R (r − A x) when
/// `coarse` is given; x += S (r − A x).
struct TwoLevel<'a, 'b> {
    a: Counted<'a>,
    smoother: Box<dyn LinearFn<V> + 'a>,
    coarse: Option<(Exact<'a, 'b>, Fgmres<C>)>,
    r: V,
    e: V,
}

impl LinearFn<V> for TwoLevel<'_, '_> {
    fn mult(&mut self, x: &mut V, b: &V) {
        let minus = vec![C64::new(-1.0, 0.0); b.num_vectors()];
        let one = vec![C64::new(1.0, 0.0); b.num_vectors()];
        self.smoother.mult(x, b);
        if let Some((exact, solver)) = self.coarse.as_mut() {
            self.a.mult(&mut self.r, x);
            self.r.aypx(&minus, b);
            let mut rc = exact.p.coarse_field(b.num_vectors());
            exact.restrict(&mut rc, &self.r);
            let mut c = rc.zeros_like();
            let _ = solver.solve(exact, None, &mut c, &rc);
            exact.p.prolong(&mut self.e, &c);
            x.axpy(&one, &self.e);
        }
        self.a.mult(&mut self.r, x);
        self.r.aypx(&minus, b);
        self.smoother.mult(&mut self.e, &self.r);
        x.axpy(&one, &self.e);
    }
}

/// What the outer solves need besides the preconditioner.
struct Problem<'a> {
    op: &'a Mdwf<f64>,
    b: V,
    rhs: V,
    stop: Stop,
}

impl Problem<'_> {
    /// |D x − b|² / |b|².
    fn physical(&self, x: &V) -> f64 {
        let mut r = self.b.zeros_like();
        self.op.mult(&mut self.op.work(), &mut r, x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &self.b);
        norm(&r).powi(2) / norm(&self.b).powi(2)
    }

    /// BiCGStab and FGMRES(20) on A y = X† b with the preconditioner `make` gives; the count
    /// is that of every application of A.
    fn outer<'m>(&self, name: &str, make: &dyn Fn(&Count) -> Box<dyn LinearFn<V> + 'm>) {
        let mut line = format!("    {name:<32}");
        for flexible in [false, true] {
            let count = Count::default();
            let mut m = make(&count);
            let (mut y, mut x) = (self.b.zeros_like(), self.b.zeros_like());
            let mut a = Counted::new(self.op, &self.b, &count);
            let start = Instant::now();
            let result = if flexible {
                Fgmres::new(Stop {
                    max_iterations: 20,
                    max_restarts: 100,
                    ..self.stop
                })
                .solve(&mut a, Some(&mut *m), &mut y, &self.rhs)
            } else {
                BiCgStab::new(self.stop).solve(&mut a, Some(&mut *m), &mut y, &self.rhs)
            };
            PvPreconditioned::new(self.op, &self.b).solution(&mut x, &y);
            let it = match result {
                Ok(i) => format!("{}", i.iterations),
                Err(SolveError::NotConverged(i)) => format!("{} (not conv.)", i.iterations),
                Err(e) => format!("failed: {e}"),
            };
            line += &format!(
                "  {} {it:>5}, {:>6} A, |Dx-b|² {:.1e}, {:.0} s;",
                if flexible { "FGMRES(20)" } else { "BiCGStab" },
                count.get(),
                self.physical(&x),
                start.elapsed().as_secs_f64()
            );
        }
        println!("{line}");
    }
}

/// The Ritz vectors of a `mdwf_coherence_eig` checkpoint (ascending) and their θ = σ².
fn load(path: &PathBuf, like: &V) -> Result<(V, Vec<f64>), Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<f64> {
        f.read_exact(&mut word)?;
        Ok(f64::from_le_bytes(word))
    };
    let mut header = [0u64; 3];
    for h in header.iter_mut() {
        *h = next(&mut f)?.to_bits();
    }
    let [done, m, len] = header.map(|h| h as usize);
    let mut x = like.zeros_with(m);
    if x.as_slice().len() != len {
        return Err(format!("{} is for another lattice or Ls", path.display()).into());
    }
    let mut theta = vec![0.0; m];
    for t in theta.iter_mut() {
        *t = next(&mut f)?;
    }
    for v in x.as_mut_slice() {
        *v = next(&mut f)?;
    }
    println!(
        "{}: {m} Ritz vectors after {done} iterations",
        path.display()
    );
    Ok((x, theta))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(&args[0]);
    let l: usize = args[1].parse().map_err(|_| USAGE)?;
    let m5: f64 = args[2].parse().map_err(|_| USAGE)?;
    let mass: f64 = args[3].parse().map_err(|_| USAGE)?;
    let checkpoint = PathBuf::from(&args[4]);
    let ls: usize = args
        .get(5)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let block: usize = args
        .get(6)
        .map_or(Ok(3), |s| s.parse())
        .map_err(|_| USAGE)?;
    let counts: Vec<usize> = if args.len() > 7 {
        args[7..]
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()
            .map_err(|_| USAGE)?
    } else {
        vec![8, 16]
    };

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
    let like = random(1, 1);
    let (ritz, theta) = load(&checkpoint, &like)?;
    let mut a = PvPreconditioned::new(&op, &like);

    // The same right-hand side as the log's measurements (seed 91).
    let b = random(1, 91);
    let mut rhs = b.zeros_like();
    a.right_hand_side(&mut rhs, &b);
    let problem = Problem {
        op: &op,
        b,
        rhs,
        stop: Stop {
            tolerance_sqr: 1e-16,
            max_iterations: 1500,
            max_restarts: 1,
        },
    };

    println!("  smoother alone:");
    problem.outer("(s) GMRES(4) before and after", &|count| {
        Box::new(TwoLevel {
            a: Counted::new(&op, &like, count),
            smoother: gmres4(&op, &like, count),
            coarse: None,
            r: like.zeros_like(),
            e: like.zeros_like(),
        })
    });

    let agg = Aggregation::new(&lat, [block; ND]);
    for n in counts {
        if n + 8 > ritz.num_vectors() {
            println!(
                "N = {n}: the checkpoint has only {} vectors",
                ritz.num_vectors()
            );
            continue;
        }
        let vectors = ritz.select(&(0..n).collect::<Vec<_>>());
        let mut g5 = vectors.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5, &vectors);
        let p = Basis::<f64>::from_vectors(&agg, &vectors, &g5)?;
        println!(
            "N = {n} (σ_0 … σ_{} = {:.2e} … {:.2e}), coarse dimension {} per aggregate:",
            n - 1,
            theta[0].sqrt(),
            theta[n - 1].sqrt(),
            2 * n
        );

        // Q from the left singular vectors u = A v / |A v|.
        let mut u = vectors.zeros_like();
        Counted::new(&op, &like, &Count::default()).mult(&mut u, &vectors);
        let mut un = vec![0.0; n];
        u.norm_sqr(&mut un);
        u.scale(&un.iter().map(|x| 1.0 / x.sqrt()).collect::<Vec<_>>());
        let mut g5u = u.zeros_like();
        op.mult_gamma5(&mut op.work(), &mut g5u, &u);
        let q = Basis::<f64>::from_vectors(&agg, &u, &g5u)?;

        // The exact coarse correction on the next eight Ritz vectors: what it leaves of the
        // error t and of the residual A t, against the floor √(1 − |PP†t|²) of the error.
        let none = Count::default();
        let mut floor = String::from("    floor √(1 − |PP†t|²):        ");
        for j in n..n + 8 {
            let t = ritz.select(&[j]);
            let mut c = p.coarse_field(1);
            p.restrict(&mut c, &t);
            let mut inside = like.zeros_like();
            p.prolong(&mut inside, &c);
            let held = norm(&inside) / norm(&t);
            floor += &format!(" {:.3}", (1.0 - held * held).max(0.0).sqrt());
        }
        println!("{floor}");
        for kind in [
            Restriction::Galerkin,
            Restriction::Left,
            Restriction::MinRes,
        ] {
            let mut line = format!("    {:<28} error / residual:", kind.name());
            for j in n..n + 8 {
                let t = ritz.select(&[j]);
                let mut a_t = t.zeros_like();
                a.mult(&mut a_t, &t);
                let mut exact = Exact {
                    a: Counted::new(&op, &like, &none),
                    p: &p,
                    q: &q,
                    kind,
                    fine: like.zeros_like(),
                    a_fine: like.zeros_like(),
                };
                let mut rc = p.coarse_field(1);
                exact.restrict(&mut rc, &a_t);
                let mut y = rc.zeros_like();
                let solved = Fgmres::new(Stop {
                    tolerance_sqr: 1e-14,
                    max_iterations: 100,
                    max_restarts: 10,
                })
                .solve(&mut exact, None, &mut y, &rc);
                let mut left = like.zeros_like();
                p.prolong(&mut left, &y);
                left.aypx(&[C64::new(-1.0, 0.0)], &t);
                let mut a_left = left.zeros_like();
                a.mult(&mut a_left, &left);
                line += &format!(
                    " {:.3}/{:.3}{}",
                    norm(&left) / norm(&t),
                    norm(&a_left) / norm(&a_t),
                    if solved.is_err() { "*" } else { "" }
                );
            }
            println!("{line}");
        }

        problem.outer("(p) cycle, probed P†AP", &|count| {
            let parts = CycleParts {
                operator: Box::new(Counted::new(&op, &like, count)),
                pre_smoother: Some(gmres4(&op, &like, count)),
                post_smoother: gmres4(&op, &like, count),
                coarse_solver: Box::new(Fgmres::new(Stop {
                    tolerance_sqr: 1e-4,
                    max_iterations: 5,
                    max_restarts: 1,
                })),
            };
            let hooks = SetupHooks {
                precondition_coarse_diagonal: true,
                galerkin_of: Some(Box::new(PvPreconditioned::new(&op, &like))),
                ..SetupHooks::default()
            };
            let mut copy = vectors.zeros_like();
            copy.copy_from(&vectors);
            Box::new(
                Multigrid::with_parts(&op, &agg, copy, 0, parts, hooks)
                    .expect("the multigrid setup"),
            )
        });
        for kind in [
            Restriction::Galerkin,
            Restriction::Left,
            Restriction::MinRes,
        ] {
            let name = format!("(e) cycle, exact, {}", kind.name());
            problem.outer(&name, &|count| {
                Box::new(TwoLevel {
                    a: Counted::new(&op, &like, count),
                    smoother: gmres4(&op, &like, count),
                    coarse: Some((
                        Exact {
                            a: Counted::new(&op, &like, count),
                            p: &p,
                            q: &q,
                            kind,
                            fine: like.zeros_like(),
                            a_fine: like.zeros_like(),
                        },
                        Fgmres::new(Stop {
                            tolerance_sqr: 1e-4,
                            max_iterations: 30,
                            max_restarts: 1,
                        }),
                    )),
                    r: like.zeros_like(),
                    e: like.zeros_like(),
                })
            });
        }
    }
    Ok(())
}
