//! BiCGStab on A′ with a two-level cycle whose coarse operator is the Γ5-Hermitian P†DP
//! (`mdwf_g5galerkin`): the full solve, counting the work.
//!
//! ```text
//! cargo run --release --example mdwf_g5cycle -- CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [K]
//! ```
//!
//! P from C⁻¹ v for the lowest N (default 8) Ritz vectors v of a `mdwf_coherence_eig`
//! checkpoint, on BLOCK^4 aggregates (default 3), split by Γ5. The coarse operator is the
//! exact nearest-neighbour P†DP built from the hops of D (`CoarseOp::build`), checked against
//! P†DP applied through the fine lattice. One cycle for A′ y = r:
//!
//!   y = S r;  y += C P (P†DP)⁻¹ P† X⁻†(r − A′y);  y += S (r − A′y),
//!
//! with X⁻† = D_PV⁻† C_PV† by K (default 5) BiCGStab iterations on D_PV† (K = 0: the identity,
//! i.e. D_PV ≈ C_PV), the coarse system by FGMRES to 1e-4 (at most 40 iterations), and the
//! smoother S either GMRES(4) or a Chebyshev polynomial (9 steps, ratio 100, hi = 4.905).
//! Printed for BiCGStab and FGMRES(20) on A′ y = X† b to |r|²/|b|² < 1e-16 (seed 91): the
//! outer iterations, the applications of A′ and of D_PV†, the coarse matrix applications,
//! the Wilson sweeps (A′ is two, D_PV one; the coarse ones not counted), and |Dx − b|²/|b|²;
//! for the smoother alone (no coarse correction) and with it.

use ruttice::fermion::{FermionOp, HopTerms, Mdwf, MdwfAt, MdwfParams, PvPreconditioned, Variant};
use ruttice::io::read_bridge_text;
use ruttice::solver::preconditioner::multigrid::{Aggregation, Basis, CoarseOp};
use ruttice::solver::{BiCgStab, Cg, Chebyshev, Fgmres, Solve, SolveError, Solver, Stop};
use ruttice::{
    random_spinor, Boundary, Coarse, Field, Lattice, LinearFn, LinearOp, Normal, Spinor, Vector,
    C64, ND,
};
use std::cell::Cell;
use std::error::Error;
use std::io::Read;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

const USAGE: &str = "usage: mdwf_g5cycle CONFIG L M5 MASS CHECKPOINT [LS] [N] [BLOCK] [K]";

type V = Field<f64, Spinor>;
type C = Field<f64, Coarse>;
type Count = Rc<Cell<usize>>;

fn norm(v: &V) -> f64 {
    let mut n = vec![0.0; v.num_vectors()];
    v.norm_sqr(&mut n);
    n.iter().sum::<f64>().sqrt()
}

fn load(path: &PathBuf, like: &V, n: usize) -> Result<V, Box<dyn Error>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut word = [0u8; 8];
    let mut next = |f: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<[u8; 8]> {
        f.read_exact(&mut word)?;
        Ok(word)
    };
    let _ = next(&mut f)?;
    let m = u64::from_le_bytes(next(&mut f)?) as usize;
    let len = u64::from_le_bytes(next(&mut f)?) as usize;
    let mut x = like.zeros_with(m);
    if x.as_slice().len() != len {
        return Err("checkpoint for another shape".into());
    }
    for _ in 0..m {
        let _ = next(&mut f)?;
    }
    for v in x.as_mut_slice() {
        *v = f64::from_le_bytes(next(&mut f)?);
    }
    Ok(x.select(&(0..n).collect::<Vec<_>>()))
}

/// A linear map that counts the vectors it is applied to.
struct Counted<A> {
    a: A,
    n: Count,
}

impl<W: Vector, A: LinearFn<W>> LinearFn<W> for Counted<A> {
    fn mult(&mut self, out: &mut W, inp: &W) {
        self.n.set(self.n.get() + inp.num_vectors());
        self.a.mult(out, inp);
    }
}

/// P†DP through the fine lattice, for the check.
struct Galerkin<'a, 'b> {
    d: MdwfAt<'a, f64>,
    p: &'b Basis<f64>,
    f: V,
    g: V,
}

impl LinearFn<C> for Galerkin<'_, '_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.p.prolong(&mut self.f, inp);
        self.d.mult(&mut self.g, &self.f);
        self.p.restrict(out, &self.g);
    }
}

/// The cycle.
struct Cycle<'a, 'b> {
    op: &'a Mdwf<f64>,
    mass: f64,
    a: Counted<PvPreconditioned<'a, f64>>,
    smoother: Box<dyn LinearFn<V> + 'a>,
    coarse: Option<(&'b Basis<f64>, Counted<&'b mut CoarseOp<f64>>)>,
    pv_dag: Counted<MdwfAt<'a, f64>>,
    k: usize,
    cgnr: bool,
    /// The factorized coarse correction: (P†D_PV†P)(P†DP) c = P† C_PV† r, with this P†D_PV P.
    pv_coarse: Option<&'b mut CoarseOp<f64>>,
    coarse_tol: f64,
    /// Conjugate residuals on the Γ5-Hermitian coarse factors instead of FGMRES.
    cr: bool,
    r: V,
    e: V,
}

/// Γ5 on the coarse lattice: +1 on the first half of the components of every vector
/// (positive chirality), −1 on the second.
fn coarse_gamma5(v: &mut C) {
    let nv = v.num_vectors();
    let per_vector = v.site_len() / nv;
    for x in v.as_mut_slice().chunks_mut(per_vector) {
        for r in x[per_vector / 2..].iter_mut() {
            *r = -*r;
        }
    }
}

/// Γ5 A_c (or Γ5 A_c† with `adjoint`) for a coarse operator A_c with Γ5 A_c Γ5 = A_c†: a
/// Hermitian, indefinite map.
struct Gamma5Coarse<'a> {
    a: &'a mut CoarseOp<f64>,
    adjoint: bool,
}

impl LinearFn<C> for Gamma5Coarse<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        if self.adjoint {
            self.a.mult_dag(out, inp);
        } else {
            self.a.mult(out, inp);
        }
        coarse_gamma5(out);
    }
}

/// Conjugate residuals for a Hermitian (indefinite) map: five vectors, short recurrences,
/// the residual minimised over the Krylov space. Solves A_c x = b as (Γ5 A_c) x = Γ5 b.
fn conjugate_residual(
    h: &mut Gamma5Coarse<'_>,
    x: &mut C,
    b: &C,
    tol_sqr: f64,
    max_it: usize,
) -> usize {
    let nv = b.num_vectors();
    let mut rhs = b.zeros_like();
    rhs.copy_from(b);
    coarse_gamma5(&mut rhs);
    x.scale(&vec![0.0; nv]);
    let mut r = rhs.zeros_like();
    r.copy_from(&rhs);
    let mut p = r.zeros_like();
    p.copy_from(&r);
    let mut hr = r.zeros_like();
    h.mult(&mut hr, &r);
    let mut hp = hr.zeros_like();
    hp.copy_from(&hr);
    let mut bb = vec![0.0; nv];
    rhs.norm_sqr(&mut bb);
    let mut rhr = vec![C64::ZERO; nv];
    r.dot(&hr, &mut rhr);
    let (mut hphp, mut rr) = (vec![0.0; nv], vec![0.0; nv]);
    for it in 0..max_it {
        hp.norm_sqr(&mut hphp);
        let alpha: Vec<C64> = (0..nv)
            .map(|j| C64::new(rhr[j].re / hphp[j].max(1e-300), 0.0))
            .collect();
        x.axpy(&alpha, &p);
        let minus: Vec<C64> = alpha.iter().map(|a| C64::new(-a.re, 0.0)).collect();
        r.axpy(&minus, &hp);
        r.norm_sqr(&mut rr);
        if (0..nv).all(|j| rr[j] <= tol_sqr * bb[j]) {
            return it + 1;
        }
        h.mult(&mut hr, &r);
        let mut rhr_new = vec![C64::ZERO; nv];
        r.dot(&hr, &mut rhr_new);
        let beta: Vec<C64> = (0..nv)
            .map(|j| C64::new(rhr_new[j].re / rhr[j].re, 0.0))
            .collect();
        p.aypx(&beta, &r);
        hp.aypx(&beta, &hr);
        rhr = rhr_new;
    }
    max_it
}

/// The adjoint of a coarse operator as a linear map.
struct Adjoint<'a>(&'a mut CoarseOp<f64>);

impl LinearFn<C> for Adjoint<'_> {
    fn mult(&mut self, out: &mut C, inp: &C) {
        self.0.mult_dag(out, inp);
    }
}

/// The coarse solves of the factorized correction, to `tolerance_sqr` (argument 13,
/// default 1e-8).
fn fact_stop(tolerance_sqr: f64) -> Stop {
    Stop {
        tolerance_sqr,
        max_iterations: 40,
        max_restarts: 10,
    }
}

impl LinearFn<V> for Cycle<'_, '_> {
    fn mult(&mut self, y: &mut V, b: &V) {
        let nv = b.num_vectors();
        let minus = vec![C64::new(-1.0, 0.0); nv];
        let one = vec![C64::new(1.0, 0.0); nv];
        self.smoother.mult(y, b);
        if let Some((p, coarse)) = self.coarse.as_mut() {
            self.a.mult(&mut self.r, y);
            self.r.aypx(&minus, b);
            if let Some(cpv) = self.pv_coarse.as_mut() {
                // (P†D_PV†P) w = P† C_PV† r, then (P†DP) c = w; nothing on the fine lattice.
                let mut z = self.r.zeros_like();
                self.op.mult_site(&mut z, &self.r, 1.0, true);
                let mut g = p.coarse_field(nv);
                p.restrict(&mut g, &z);
                let mut w = g.zeros_like();
                let mut c = w.zeros_like();
                if self.cr {
                    // Both factors are Γ5-Hermitian: conjugate residuals on Γ5 A_c, five
                    // coarse vectors instead of the 80 of FGMRES(40).
                    let it1 = conjugate_residual(
                        &mut Gamma5Coarse {
                            a: cpv,
                            adjoint: true,
                        },
                        &mut w,
                        &g,
                        self.coarse_tol,
                        2000,
                    );
                    let it2 = conjugate_residual(
                        &mut Gamma5Coarse {
                            a: &mut *coarse.a,
                            adjoint: false,
                        },
                        &mut c,
                        &w,
                        self.coarse_tol,
                        2000,
                    );
                    coarse.n.set(coarse.n.get() + it1 + it2);
                } else {
                    let _ = Fgmres::new(fact_stop(self.coarse_tol)).solve(
                        &mut Adjoint(cpv),
                        None,
                        &mut w,
                        &g,
                    );
                    let _ = Fgmres::new(fact_stop(self.coarse_tol)).solve(coarse, None, &mut c, &w);
                }
                let mut pc = self.r.zeros_like();
                p.prolong(&mut pc, &c);
                self.op.mult_site(&mut self.e, &pc, self.mass, false);
                y.axpy(&one, &self.e);
                self.a.mult(&mut self.r, y);
                self.r.aypx(&minus, b);
                self.smoother.mult(&mut self.e, &self.r);
                y.axpy(&one, &self.e);
                return;
            }
            // z = X⁻† r = D_PV⁻† C_PV† r.
            let mut z = self.r.zeros_like();
            if self.k == 0 {
                z.copy_from(&self.r);
            } else {
                let mut w = self.r.zeros_like();
                self.op.mult_site(&mut w, &self.r, 1.0, true);
                let _ = BiCgStab::new(Stop::simple(1e-30, self.k)).solve(
                    &mut self.pv_dag,
                    None,
                    &mut z,
                    &w,
                );
            }
            let mut rc = p.coarse_field(nv);
            p.restrict(&mut rc, &z);
            let mut c = rc.zeros_like();
            if self.cgnr {
                // The normal equations of the coarse D: its spectrum has negative real
                // parts, as D's has, which GMRES(40) does not get through.
                let mut nrhs = rc.zeros_like();
                coarse.a.mult_dag(&mut nrhs, &rc);
                coarse.n.set(coarse.n.get() + nv);
                let mut normal = Normal::new(&mut *coarse.a, &rc);
                let _ = Cg::new(Stop {
                    tolerance_sqr: 1e-8,
                    max_iterations: 2000,
                    max_restarts: 0,
                })
                .solve(&mut normal, None, &mut c, &nrhs);
            } else {
                let _ = Fgmres::new(Stop {
                    tolerance_sqr: 1e-4,
                    max_iterations: 40,
                    max_restarts: 0,
                })
                .solve(coarse, None, &mut c, &rc);
            }
            let mut pc = self.r.zeros_like();
            p.prolong(&mut pc, &c);
            self.op.mult_site(&mut self.e, &pc, self.mass, false);
            y.axpy(&one, &self.e);
        }
        self.a.mult(&mut self.r, y);
        self.r.aypx(&minus, b);
        self.smoother.mult(&mut self.e, &self.r);
        y.axpy(&one, &self.e);
    }
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
    let n: usize = args
        .get(6)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|_| USAGE)?;
    let block: usize = args
        .get(7)
        .map_or(Ok(3), |s| s.parse())
        .map_err(|_| USAGE)?;
    let k: usize = args
        .get(8)
        .map_or(Ok(5), |s| s.parse())
        .map_err(|_| USAGE)?;
    let cgnr = args.get(9).is_some_and(|s| s == "cgnr");
    // `fact`: the factorized coarse correction, no D_PV on the fine lattice.
    let fact = args.get(9).is_some_and(|s| s == "fact");
    // `sds`: the basis from s = C⁻¹v for the lowest N/2 and D s for the same N/2.
    let sds = args.get(11).is_some_and(|s| s == "sds");
    let coarse_tol: f64 = args
        .get(13)
        .map_or(Ok(1e-8), |s| s.parse())
        .map_err(|_| USAGE)?;
    // `cr`: conjugate residuals on Γ5 times each coarse factor (argument 14).
    let cr = args.get(14).is_some_and(|s| s == "cr");
    let smoothers: Vec<&str> = match args.get(10).map(String::as_str) {
        Some("cheb") => vec!["Chebyshev 9/100"],
        Some("gmres") => vec!["GMRES(4)"],
        _ => vec!["GMRES(4)", "Chebyshev 9/100"],
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
    let random = |nv: usize, seed: u64| {
        let mut v = V::with_shape(&lat, 24, ls, nv, 1);
        v.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, ls * nv, seed).as_slice());
        v
    };
    let like = random(1, 1);
    // The test vectors: the Ritz vectors of the checkpoint, or (CHECKPOINT `setup:PASSES:TOL`)
    // a realistic setup: random vectors through PASSES inverse iterations x ← A′⁻¹ x, each a
    // BiCGStab solve of all vectors together, preconditioned by the Chebyshev polynomial, to
    // |r|²/|b|² < TOL. Its Wilson sweeps are printed.
    let count = if sds { n / 2 } else { n };
    let setup = args[4].strip_prefix("setup:").map(|s| {
        let mut it = s.split(':');
        let passes: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(3);
        let tol: f64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-2);
        (passes, tol)
    });
    let ritz = if let Some((passes, tol)) = setup {
        let mut x = random(count, 4242);
        let na = Count::default();
        let start = Instant::now();
        for _ in 0..passes {
            let mut nx = vec![0.0; count];
            x.norm_sqr(&mut nx);
            x.scale(&nx.iter().map(|v| 1.0 / v.sqrt()).collect::<Vec<_>>());
            let mut y = x.zeros_like();
            let mut a = Counted {
                a: PvPreconditioned::new(&op, &x),
                n: na.clone(),
            };
            let mut cheb = Chebyshev::new(
                Counted {
                    a: PvPreconditioned::new(&op, &x),
                    n: na.clone(),
                },
                9,
                4.905 / 100.0,
                4.905,
                &x,
            );
            let _ = BiCgStab::new(Stop {
                tolerance_sqr: tol,
                max_iterations: 200,
                max_restarts: 0,
            })
            .solve(&mut a, Some(&mut cheb), &mut y, &x);
            x = y;
        }
        println!(
            "  setup: {count} random vectors, {passes} inverse iterations to {tol:e}: {} A′ = {} sweeps, {:.0} s",
            na.get(),
            2 * na.get(),
            start.elapsed().as_secs_f64()
        );
        x
    } else {
        load(&checkpoint, &like, count)?
    };
    let mut xs = ritz.zeros_like();
    op.mult_site_inv(&mut xs, &ritz, mass, false);
    drop(ritz);
    if sds {
        let half = xs.num_vectors();
        let mut ds = xs.zeros_like();
        MdwfAt::new(&op, Variant::D, mass).mult(&mut ds, &xs);
        let mut both = xs.zeros_with(2 * half);
        both.assign(&(0..half).collect::<Vec<_>>(), &xs);
        both.assign(&(half..2 * half).collect::<Vec<_>>(), &ds);
        xs = both;
    }
    let mut pv = Mdwf::<f64>::new(
        &lat,
        MdwfParams {
            mass: 1.0,
            ..params
        },
    );
    pv.set_gauge(&u)?;
    let mut g5 = xs.zeros_like();
    op.mult_gamma5(&mut op.work(), &mut g5, &xs);
    let agg = Aggregation::new(&lat, [block; ND]);
    let p = Basis::from_vectors(&agg, &xs, &g5)?;
    drop((xs, g5));
    let mut coarse = CoarseOp::build(&op, &mut op.work(), &p);
    let mut pv_coarse = CoarseOp::build(&pv, &mut pv.work(), &p);
    println!(
        "{l}^4, Ls = {ls}, M5 = {m5}, m = {mass}; P from {} on {block}^4 (coarse dimension {}); {}",
        if sds {
            format!("s = C⁻¹v and Ds for the lowest {}", n / 2)
        } else {
            format!("s = C⁻¹v for the lowest {n}")
        },
        coarse.dim(),
        if fact {
            "factorized coarse correction (P†D_PV†P)(P†DP), no fine D_PV".to_string()
        } else {
            format!("X⁻† by {k} BiCGStab iterations")
        }
    );
    // Check the nearest-neighbour P†DP against the one through the fine lattice.
    {
        let mut c = p.coarse_field(1);
        let rnd = random(1, 77);
        p.restrict(&mut c, &rnd);
        let (mut a1, mut a2) = (c.zeros_like(), c.zeros_like());
        coarse.mult(&mut a1, &c);
        Galerkin {
            d: MdwfAt::new(&op, Variant::D, mass),
            p: &p,
            f: like.zeros_like(),
            g: like.zeros_like(),
        }
        .mult(&mut a2, &c);
        let mut n1 = vec![0.0; 1];
        a2.norm_sqr(&mut n1);
        a1.axpy(&[C64::new(-1.0, 0.0)], &a2);
        let mut n2 = vec![0.0; 1];
        a1.norm_sqr(&mut n2);
        println!(
            "  check: |P†DP (built) − P†DP (through the fine lattice)| / |…| = {:.1e}",
            (n2[0] / n1[0]).sqrt()
        );
    }

    // Spurious small singular values of the coarse operator? The lowest of P†DP by inverse
    // iteration on its normal operator (CG solves, cheap on the coarse lattice), against
    // |D s|/|s| for the x-space low modes s = C⁻¹ v the basis was built from. Only in the
    // `diag` mode: on large coarse lattices the tight CG solves take hours.
    if args.get(10).map(String::as_str) == Some("diag") {
        let mut c = p.coarse_field(1);
        p.restrict(&mut c, &random(1, 55));
        let mut sigma = 0.0;
        for _ in 0..30 {
            let mut nn = vec![0.0; 1];
            c.norm_sqr(&mut nn);
            c.scale(&[1.0 / nn[0].sqrt()]);
            let mut y = c.zeros_like();
            let _ = Cg::new(Stop {
                tolerance_sqr: 1e-20,
                max_iterations: 20000,
                max_restarts: 0,
            })
            .solve(&mut Normal::new(&mut coarse, &c), None, &mut y, &c);
            let mut ny = vec![0.0; 1];
            y.norm_sqr(&mut ny);
            sigma = (1.0 / ny[0].sqrt()).sqrt();
            c = y;
        }
        let ritz = load(&checkpoint, &like, n)?;
        let mut s = ritz.zeros_like();
        op.mult_site_inv(&mut s, &ritz, mass, false);
        let mut ds = s.zeros_like();
        MdwfAt::new(&op, Variant::D, mass).mult(&mut ds, &s);
        let (mut ns, mut nds) = (vec![0.0; n], vec![0.0; n]);
        s.norm_sqr(&mut ns);
        ds.norm_sqr(&mut nds);
        let fine: Vec<f64> = ns.iter().zip(&nds).map(|(a, b)| (b / a).sqrt()).collect();
        println!(
            "  lowest singular value of P†DP (per unit |P c|): {sigma:.3e};  |D s|/|s| of the basis modes: {:.3e} … {:.3e}",
            fine.iter().cloned().fold(f64::INFINITY, f64::min),
            fine.iter().cloned().fold(0.0, f64::max)
        );
    }
    if args.get(10).map(String::as_str) == Some("diag") {
        return Ok(());
    }
    // Check conjugate residuals on both coarse factors: solve, then the true residual.
    if cr {
        let mut g = p.coarse_field(1);
        p.restrict(&mut g, &random(1, 31));
        for (name, adjoint) in [("P†DP", false), ("P†D_PV†P", true)] {
            let a: &mut CoarseOp<f64> = if adjoint { &mut pv_coarse } else { &mut coarse };
            let mut x = g.zeros_like();
            let it = conjugate_residual(&mut Gamma5Coarse { a, adjoint }, &mut x, &g, 1e-16, 5000);
            let mut ax = g.zeros_like();
            let a: &mut CoarseOp<f64> = if adjoint { &mut pv_coarse } else { &mut coarse };
            if adjoint {
                a.mult_dag(&mut ax, &x);
            } else {
                a.mult(&mut ax, &x);
            }
            ax.axpy(&[C64::new(-1.0, 0.0)], &g);
            let (mut r, mut b) = (vec![0.0; 1], vec![0.0; 1]);
            ax.norm_sqr(&mut r);
            g.norm_sqr(&mut b);
            println!(
                "  check: conjugate residuals on Γ5 {name}: {it} iterations, |A x − b|/|b| = {:.1e}",
                (r[0] / b[0]).sqrt()
            );
        }
    }

    let b = random(1, 91);
    let mut rhs = b.zeros_like();
    PvPreconditioned::new(&op, &b).right_hand_side(&mut rhs, &b);
    let physical = |y: &V| {
        let mut x = y.zeros_like();
        PvPreconditioned::new(&op, y).solution(&mut x, y);
        let mut r = b.zeros_like();
        op.mult(&mut op.work(), &mut r, &x, Variant::D);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        (norm(&r) / norm(&b)).powi(2)
    };
    let hi = 4.905;
    let stop = Stop {
        tolerance_sqr: 1e-16,
        max_iterations: 1000,
        max_restarts: 1,
    };
    for smoother_name in smoothers {
        // `noalone`: skip the smoother-alone baselines (they do not depend on P).
        let skip_alone = args.get(12).is_some_and(|s| s == "noalone");
        for with_coarse in [false, true].into_iter().filter(|w| *w || !skip_alone) {
            let mut line = format!(
                "  {smoother_name:<15} {:<15}",
                if with_coarse { "+ Γ5 coarse" } else { "alone" }
            );
            for flexible in [false, true] {
                let (na, npv, nc) = (Count::default(), Count::default(), Count::default());
                let smoother: Box<dyn LinearFn<V>> = if smoother_name == "GMRES(4)" {
                    Box::new(Solve::new(
                        Fgmres::new(Stop::simple(1e-6, 4)),
                        Counted {
                            a: PvPreconditioned::new(&op, &like),
                            n: na.clone(),
                        },
                    ))
                } else {
                    Box::new(Chebyshev::new(
                        Counted {
                            a: PvPreconditioned::new(&op, &like),
                            n: na.clone(),
                        },
                        9,
                        hi / 100.0,
                        hi,
                        &like,
                    ))
                };
                let mut cycle = Cycle {
                    op: &op,
                    mass,
                    a: Counted {
                        a: PvPreconditioned::new(&op, &like),
                        n: na.clone(),
                    },
                    smoother,
                    coarse: if with_coarse {
                        Some((
                            &p,
                            Counted {
                                a: &mut coarse,
                                n: nc.clone(),
                            },
                        ))
                    } else {
                        None
                    },
                    pv_dag: Counted {
                        a: MdwfAt::new(&op, Variant::Dagger, 1.0),
                        n: npv.clone(),
                    },
                    k,
                    cgnr,
                    pv_coarse: if fact && with_coarse {
                        Some(&mut pv_coarse)
                    } else {
                        None
                    },
                    coarse_tol,
                    cr,
                    r: like.zeros_like(),
                    e: like.zeros_like(),
                };
                let mut outer = Counted {
                    a: PvPreconditioned::new(&op, &b),
                    n: na.clone(),
                };
                let mut y = b.zeros_like();
                let start = Instant::now();
                let result = if flexible {
                    Fgmres::new(Stop {
                        max_iterations: 20,
                        max_restarts: 100,
                        ..stop
                    })
                    .solve(&mut outer, Some(&mut cycle), &mut y, &rhs)
                } else {
                    BiCgStab::new(stop).solve(&mut outer, Some(&mut cycle), &mut y, &rhs)
                };
                let it = match result {
                    Ok(i) => format!("{}", i.iterations),
                    Err(SolveError::NotConverged(i)) => format!("{}!", i.iterations),
                    Err(e) => format!("failed: {e}"),
                };
                line += &format!(
                    " | {} {it:>5}: {:>5} A′, {:>5} D_PV, {:>5} coarse, {:>6} sweeps, |Dx−b|² {:.0e}, {:.0} s",
                    if flexible { "FGMRES(20)" } else { "BiCGStab" },
                    na.get(),
                    npv.get(),
                    nc.get(),
                    2 * na.get() + npv.get(),
                    physical(&y),
                    start.elapsed().as_secs_f64()
                );
            }
            println!("{line}");
        }
    }
    Ok(())
}
