//! The field of values of Ã and of Ã M on the device (`preconditioned_fov` on the CPU):
//! [λ_min, λ_max] of the Hermitian parts (Ã + Ã†)/2 and (Ã M + M† Ã†)/2 by Lanczos, for the
//! large lattices. M is the two-grid cycle of the solver (Chebyshev smoother, the PV coarse
//! correction); M† the cycle of Ã† on the same P with the same top of the smoother's interval,
//! both coarse factors solved on their exact adjoints (BiCGStab, coarse solves in double
//! precision) to |r|²/|b|² < `--coarse` (1e-20), so that M† is the adjoint of M as a map. The
//! program checks ⟨z, Ã M w⟩ = ⟨M† Ã† z, w⟩ first.
//!
//! ```text
//! cargo run --release --features cuda-bridge --example preconditioned_fov_device --
//!     L T FORMAT FILE M5 MASS SMEAR [--block B] [--vectors N] [--steps S] [--ratio R]
//!     [--coarse T] [--tol T] [--dim D] [--parts]
//! ```
//!
//! `--arnoldi`: the eigenvalues of Ã M of smallest real part by the Krylov-Schur method, with
//! their condition numbers (the left eigenvectors from M† Ã†), and that of largest magnitude.
//!
//! `--left`: everything in the left even-odd form (PV_left) instead of the right one.
//!
//! `--pseudo`: σ_min(z − Ã M), the norm of the resolvent, at the points of `--contour
//! cx,cy,r,n` (n on a circle, repeatable) and `--points x0,x1,..` (on the real axis); by default
//! on the real axis from −0.5 to 0.7 and on the circle |z − 1| = 0.8. With `--arnoldi`,
//! `--far cx,cy` adds the eigenvalues farthest from cx + i cy.
//!
//! `--parts`: the smoother S and the coarse correction E apart as well, the field of values and
//! ‖I − Ã X‖₂ of Ã S, Ã E and Ã M (I − Ã M = (I − Ã S)(I − Ã E)(I − Ã S)), and the vector at
//! λ_min of the Hermitian part of Ã M: its part in the range of P and its Rayleigh quotients.
//!
//! L³×T lattice, Ls 8, b 1.5, c 0.5; SMEAR 1 for stout 0.1 × 3. Setup as the solver's: 16
//! vectors filtered in 2 rounds, the bootstrap 2 × 4, P in half precision. Defaults: blocks of
//! 4⁴, 16 test vectors, Chebyshev 8 steps on [hi/30, hi], Lanczos to 1e-4 with at most `--dim`
//! (24) vectors (each a field on the even sites, on the device).

#[cfg(feature = "cuda-bridge")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    device::main()
}

#[cfg(not(feature = "cuda-bridge"))]
fn main() {
    eprintln!("preconditioned_fov_device needs --features cuda-bridge");
}

#[cfg(feature = "cuda-bridge")]
mod device {
    use ruttice::backend::cuda_bridge::multigrid::{
        bootstrap, device_correction, filtered_test_vectors, top_of, two_grid, DeviceCycle, Stage,
    };
    use ruttice::backend::cuda_bridge::{
        CudaMdwf, DeviceBasis, DeviceCoarse, DeviceCorrection, DeviceParities, DeviceParts,
        DeviceSpinor,
    };
    use ruttice::eigen::{lowest, LanczosParams};
    use ruttice::fermion::{EoForm, FermionOp, MdwfParams, PvRatio, SplitParity};
    use ruttice::gauge::smearing::{uniform, Stout};
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::solver::preconditioner::multigrid::Transfer;
    use ruttice::solver::preconditioner::multigrid::{
        Aggregation, BootstrapParams, CoarseKrylov, LowModeParams, PvEoParams,
    };
    use ruttice::solver::Chebyshev;
    use ruttice::{
        random_spinor, Boundary, EvenOddIndex, Field, Lattice, LinearFn, Parity, Spinor, Vector,
        C64,
    };
    use std::error::Error;
    use std::path::Path;
    use std::rc::Rc;
    use std::time::Instant;

    type V = DeviceSpinor<f64>;

    /// s (B + B†)/2 for B = Ã M (or Ã alone without the cycles), B† = M† Ã†.
    struct HermitianPart<'a, 'c> {
        a: PvRatio<'a, CudaMdwf<f64>, V>,
        a_dag: PvRatio<'a, CudaMdwf<f64>, V>,
        cycles: Option<(&'c mut DeviceCycle<'a, f64>, &'c mut DeviceCycle<'a, f64>)>,
        sign: f64,
    }

    impl HermitianPart<'_, '_> {
        /// B x and B† x.
        fn both(&mut self, x: &V) -> (V, V) {
            let (mut bx, mut bdx) = (x.zeros_like(), x.zeros_like());
            match self.cycles.as_mut() {
                Some((m, m_dag)) => {
                    let mut t = x.zeros_like();
                    m.mult(&mut t, x);
                    self.a.mult(&mut bx, &t);
                    let mut s = x.zeros_like();
                    self.a_dag.mult(&mut s, x);
                    m_dag.mult(&mut bdx, &s);
                }
                None => {
                    self.a.mult(&mut bx, x);
                    self.a_dag.mult(&mut bdx, x);
                }
            }
            (bx, bdx)
        }
    }

    /// The Hermitian part for `d` over `pv`, with the cycles or without.
    fn part<'a, 'c>(
        (d, pv): (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
        (even, odd): (&'a V, &'a V),
        cycles: Option<(&'c mut DeviceCycle<'a, f64>, &'c mut DeviceCycle<'a, f64>)>,
        sign: f64,
    ) -> HermitianPart<'a, 'c> {
        HermitianPart {
            a: PvRatio::new(d, pv, form(), even, odd),
            a_dag: PvRatio::new(pv, d, form(), even, odd),
            cycles,
            sign,
        }
    }

    impl LinearFn<V> for HermitianPart<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            let (bx, bdx) = self.both(inp);
            out.copy_from(&bx);
            out.axpy(&[C64::new(1.0, 0.0)], &bdx);
            out.scale(&[0.5 * self.sign]);
        }
    }

    /// The even-odd form of Ã and of the cycle: right (PV_right) unless `--left` (PV_left).
    static LEFT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    fn form() -> EoForm {
        if LEFT.load(std::sync::atomic::Ordering::Relaxed) {
            EoForm::Left
        } else {
            EoForm::Right
        }
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        LEFT.store(
            args.iter().any(|a| a == "--left"),
            std::sync::atomic::Ordering::Relaxed,
        );
        if args.len() < 7 {
            return Err("L T FORMAT FILE M5 MASS SMEAR [--block B] [--vectors N] [--steps S] [--ratio R] [--coarse T] [--tol T] [--dim D]".into());
        }
        let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
            match args.iter().position(|a| a == name) {
                Some(i) => Ok(args.get(i + 1).ok_or("a value")?.parse()?),
                None => Ok(default),
            }
        };
        let (l, nt): (usize, usize) = (args[0].parse()?, args[1].parse()?);
        let (m5, mass): (f64, f64) = (args[4].parse()?, args[5].parse()?);
        let smear = args[6] == "1";
        let block = value("--block", 4.0)? as usize;
        let vectors = value("--vectors", 16.0)? as usize;
        let coarse = value("--coarse", 1e-20)?;
        let tol = value("--tol", 1e-4)?;
        let dim = value("--dim", 24.0)? as usize;
        let far_from = match args.iter().position(|a| a == "--far") {
            Some(i) => {
                let c: Vec<f64> = args
                    .get(i + 1)
                    .ok_or("--far cx,cy")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                Some(C64::new(c[0], *c.get(1).unwrap_or(&0.0)))
            }
            None => None,
        };
        let start = Instant::now();
        let lat = Lattice::single_node([l, l, l, nt]);
        let mut u = ruttice::app::read_gauge_named(&args[2], Path::new(&args[3]), &lat)?;
        if smear {
            let s = GaugeSmearing {
                smear: Smear::Stout(Stout { rho: uniform(0.1) }),
                steps: 3,
            };
            u = s.levels(&u).pop().expect("three steps");
        }
        let params = |m: f64| MdwfParams {
            ls: 8,
            m5,
            mass: m,
            b: 1.5,
            c: 0.5,
            boundary: [
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Periodic,
                Boundary::Antiperiodic,
            ],
        };
        let mut d = CudaMdwf::<f64>::new(&lat, params(mass))?;
        d.set_gauge(&u)?;
        let mut pv = CudaMdwf::<f64>::new(&lat, params(1.0))?;
        pv.set_gauge(&u)?;
        let (d, pv) = (&d, &pv);
        let even = DeviceSpinor::<f64>::shape_half(&lat, 8, Parity::Even)?;
        let odd = DeviceSpinor::<f64>::shape_half(&lat, 8, Parity::Odd)?;
        let cycle = PvEoParams {
            chebyshev_steps: value("--steps", 8.0)? as usize,
            chebyshev_ratio: value("--ratio", 30.0)?,
            ..PvEoParams::default()
        };
        // The test vectors as the solver makes them, P from them in half precision.
        let index = EvenOddIndex::new(&lat);
        let quiet = &mut |_: Stage<'_, f64>| {};
        let mut y = filtered_test_vectors(
            (d, pv),
            (&even, &odd),
            &lat,
            &index,
            LowModeParams::default(),
            vectors,
            quiet,
        )?;
        let agg = Aggregation::new(&lat, [block; 4]);
        let boot = BootstrapParams {
            cycle,
            ..BootstrapParams::default()
        };
        let mut with = device_correction((d, pv), &agg, cycle, &even, true);
        bootstrap((d, pv), (&even, &odd), &lat, &mut y, boot, &mut with, quiet)?;
        let basis = Rc::new(DeviceBasis::pv_eo(d, pv, &agg, &y, true)?);
        drop(y);
        let exact = PvEoParams {
            pv_solver: CoarseKrylov::BiCgStab,
            coarse_tolerance_sqr: coarse,
            pv_tolerance_sqr: coarse,
            coarse_max_iterations: 20_000,
            pv_max_iterations: 20_000,
            ..cycle
        };
        let mut c = DeviceCorrection::<f64, f64>::shared(
            basis.clone(),
            (d, pv),
            (d, pv),
            form(),
            exact,
            &even,
        )?;
        let mut c_dag = DeviceCorrection::<f64, f64>::shared(
            basis.clone(),
            (pv, d),
            (pv, d),
            form(),
            exact,
            &even,
        )?;
        let (parts, parts_dag) = (c.parts(), c_dag.parts());
        let hi = top_of(
            &mut PvRatio::new(d, pv, form(), &even, &odd),
            &lat,
            &even,
            exact.seed,
            exact.power_steps,
        )?;
        let mut m = two_grid(
            (d, pv),
            form(),
            (&even, &odd),
            &lat,
            exact,
            Box::new(move |e, r| c.correct(e, r)),
            Some(hi),
        )?;
        let mut m_dag = two_grid(
            (pv, d),
            form(),
            (&even, &odd),
            &lat,
            exact,
            Box::new(move |e, r| c_dag.correct(e, r)),
            Some(hi),
        )?;
        println!(
            "# {l}^3x{nt} {} on the device; M5 {m5}, m {mass}, stout 0.1x3: {smear}; blocks {block}^4, {vectors} vectors, Chebyshev {} on [hi/{}, hi], hi {hi:.6}; coarse solves exact adjoints to {coarse:e} in double precision; setup {:.0} s",
            args[3],
            cycle.chebyshev_steps,
            cycle.chebyshev_ratio,
            start.elapsed().as_secs_f64()
        );
        let random_even = |seed: u64| -> Result<V, Box<dyn Error>> {
            let mut host = Field::<f64, Spinor>::with_shape(&lat, 24, 8, 1, 1);
            host.as_mut_slice()
                .copy_from_slice(random_spinor(&lat, 8, seed).as_slice());
            let mut v = even.zeros_with(1);
            DeviceParities.extract(
                &mut v,
                &DeviceSpinor::<f64>::from_host(&host)?,
                Parity::Even,
            );
            Ok(v)
        };
        // ⟨z, Ã M w⟩ = ⟨M† Ã† z, w⟩.
        {
            let (z, w) = (random_even(61)?, random_even(62)?);
            let mut h = part((d, pv), (&even, &odd), Some((&mut m, &mut m_dag)), 1.0);
            let (bw, _) = h.both(&w);
            let (_, bdz) = h.both(&z);
            let (mut left, mut right) = ([C64::ZERO], [C64::ZERO]);
            z.dot(&bw, &mut left);
            bdz.dot(&w, &mut right);
            let rel = ((left[0] - right[0]).norm_sqr() / left[0].norm_sqr()).sqrt();
            println!(
                "# adjoint check: <z, A M w> = {:.10e} {:+.10e}i, <M^dag A^dag z, w> = {:.10e} {:+.10e}i, relative difference {rel:.1e}",
                left[0].re, left[0].im, right[0].re, right[0].im
            );
            if rel > 1e-6 {
                println!("# WARNING: M^dag is not the adjoint of M to 1e-6; the numbers below are not those of a Hermitian part");
            }
        }
        let lanczos = LanczosParams {
            wanted: 1,
            keep: 6.min(dim - 2),
            max_dim: dim,
            tolerance: (tol).into(),
            order: ruttice::eigen::Order::Ascending,
            threshold: None,
            max_restarts: 2000,
            chebyshev: None,
        };
        let x0 = random_even(63)?;
        let mut lowest_of_am: Option<V> = None;
        for preconditioned in [false, true] {
            let mut ends = Vec::new();
            let t = Instant::now();
            for sign in [1.0, -1.0] {
                let cycles = preconditioned.then_some((&mut m, &mut m_dag));
                let mut h = part((d, pv), (&even, &odd), cycles, sign);
                let mut e = lowest(&mut h, &x0, lanczos);
                ends.push((sign * e.values[0], e.converged, e.restarts));
                if preconditioned && sign > 0.0 {
                    lowest_of_am = e.vectors.drain(..).next();
                }
            }
            println!(
                "{}: Re of the field of values in [{:+.6e}, {:+.6e}]  (converged {} {}, {} + {} restarts, {:.0} s){}",
                if preconditioned { "A~ M " } else { "A~   " },
                ends[0].0,
                ends[1].0,
                ends[0].1,
                ends[1].1,
                ends[0].2,
                ends[1].2,
                t.elapsed().as_secs_f64(),
                if ends[0].0 < 0.0 { "  lambda_min < 0" } else { "  in the right half plane" }
            );
        }
        if args.iter().any(|a| a == "--arnoldi") {
            spectrum(
                (d, pv),
                (&even, &odd),
                &lat,
                (parts.clone(), parts_dag.clone()),
                (exact, hi),
                (&x0, dim, tol, far_from),
            )?;
        }
        if args.iter().any(|a| a == "--pseudo") {
            let points = contour_points(&args)?;
            pseudospectrum(
                (d, pv),
                (&even, &odd),
                &lat,
                (parts.clone(), parts_dag.clone()),
                (exact, hi),
                (&x0, lanczos, points),
            )?;
        }
        if args.iter().any(|a| a == "--parts") {
            let x = lowest_of_am.ok_or("the vector at lambda_min of A~ M")?;
            dissect(
                (d, pv),
                (&even, &odd),
                &lat,
                &basis,
                (parts, parts_dag),
                (exact, hi),
                (&x0, &x, lanczos),
            )?;
        }
        Ok(())
    }

    /// B (or B†) of a pair as an operator.
    struct Apply<'p, 'a>(&'p mut Pair<'a>, bool);

    impl LinearFn<V> for Apply<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            match self.1 {
                false => (self.0.b)(out, inp),
                true => (self.0.b_dag)(out, inp),
            }
        }
    }

    /// M and M† (D and D_PV exchanged on the same P, the same top of the smoother's interval)
    /// of the exact cycle, rebuilt from the coarse parts.
    fn exact_cycles<'a>(
        (d, pv): (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
        (even, odd): (&'a V, &'a V),
        lat: &std::sync::Arc<Lattice>,
        (parts, parts_dag): (&DeviceParts<f64>, &DeviceParts<f64>),
        (exact, hi): (PvEoParams, f64),
    ) -> Result<(DeviceCycle<'a, f64>, DeviceCycle<'a, f64>), Box<dyn Error>> {
        let correction = |p: &DeviceParts<f64>, a, b| {
            DeviceCorrection::<f64, f64>::from_parts(p.clone(), (a, b), (a, b), form(), exact, even)
        };
        let (mut cm, mut cm_dag) = (correction(parts, d, pv)?, correction(parts_dag, pv, d)?);
        let m = two_grid(
            (d, pv),
            form(),
            (even, odd),
            lat,
            exact,
            Box::new(move |e, r| cm.correct(e, r)),
            Some(hi),
        )?;
        let m_dag = two_grid(
            (pv, d),
            form(),
            (even, odd),
            lat,
            exact,
            Box::new(move |e, r| cm_dag.correct(e, r)),
            Some(hi),
        )?;
        Ok((m, m_dag))
    }

    /// The spectrum of Ã M by the Krylov-Schur method: the eigenvalues of smallest real part
    /// with the condition number 1/|y† x| of each (x and y the right and left eigenvectors,
    /// y from M† Ã†), the eigenvalue of largest magnitude, and where the Ritz values of the last
    /// projection lie.
    fn spectrum(
        (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
        (even, odd): (&V, &V),
        lat: &std::sync::Arc<Lattice>,
        (parts, parts_dag): (DeviceParts<f64>, DeviceParts<f64>),
        (exact, hi): (PvEoParams, f64),
        (x0, dim, tol, far_from): (&V, usize, f64, Option<C64>),
    ) -> Result<(), Box<dyn Error>> {
        use ruttice::eigen::{arnoldi, ArnoldiParams, Which};
        let (mut m, mut m_dag) =
            exact_cycles((d, pv), (even, odd), lat, (&parts, &parts_dag), (exact, hi))?;
        let mut pair = product(
            (d, pv),
            (even, odd),
            Box::new(|o, i| m.mult(o, i)),
            Box::new(|o, i| m_dag.mult(o, i)),
        );
        // At most `dim` vectors on the device (--dim), 6 wanted.
        let params = ArnoldiParams {
            wanted: 6,
            keep: (2 * dim / 5).clamp(7, dim.saturating_sub(2).max(7)),
            max_dim: dim.max(10),
            tolerance: (tol.min(1e-6)).into(),
            threshold: None,
            max_restarts: 400,
            which: Which::SmallestReal,
        };
        let t = Instant::now();
        let right = arnoldi(&mut Apply(&mut pair, false), x0, params);
        let left = arnoldi(&mut Apply(&mut pair, true), x0, params);
        println!(
            "arnoldi A~ M, smallest real parts (converged {} {}, {} + {} restarts, {} applications, {:.0} s):",
            right.converged,
            left.converged,
            right.restarts,
            left.restarts,
            right.applications + left.applications,
            t.elapsed().as_secs_f64()
        );
        for (k, (theta, x)) in right.values.iter().zip(&right.vectors).enumerate() {
            // The left eigenvector: that of M† Ã† for the conjugate eigenvalue.
            let j = (0..left.values.len())
                .min_by(|&p, &q| {
                    (left.values[p].conj() - *theta)
                        .norm_sqr()
                        .total_cmp(&(left.values[q].conj() - *theta).norm_sqr())
                })
                .expect("left pairs");
            let mut d = [C64::ZERO];
            left.vectors[j].dot(x, &mut d);
            let mismatch = (left.values[j].conj() - *theta).norm_sqr().sqrt();
            println!(
                "arnoldi   lambda_{k} = {:+.6e} {:+.6e}i  |A x - l x| {:.1e}  kappa = 1/|y^dag x| = {:.3e}  (left value off by {:.1e})",
                theta.re,
                theta.im,
                right.residuals[k],
                1.0 / d[0].norm_sqr().sqrt(),
                mismatch
            );
        }
        let far = right
            .ritz
            .iter()
            .map(|z| (*z - C64::new(1.0, 0.0)).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        let leftmost = right
            .ritz
            .iter()
            .map(|z| z.re)
            .fold(f64::INFINITY, f64::min);
        println!(
            "arnoldi   Ritz values of the last projection ({}): Re >= {leftmost:+.4e}, |theta - 1| <= {far:.4e}",
            right.ritz.len()
        );
        let lm = arnoldi(
            &mut Apply(&mut pair, false),
            x0,
            ArnoldiParams {
                wanted: 2,
                which: Which::LargestMagnitude,
                ..params
            },
        );
        for (k, theta) in lm.values.iter().enumerate() {
            println!(
                "arnoldi   largest |lambda|: {:+.6e} {:+.6e}i  (|A x - l x| {:.1e}, converged {})",
                theta.re, theta.im, lm.residuals[k], lm.converged
            );
        }
        if let Some(c) = far_from {
            // The eigenvalues of Ã M − c of largest magnitude: every other eigenvalue of Ã M is
            // in the disc around c through the last of them. The left eigenvectors from
            // (Ã M)† − c̄ give the condition numbers 1/|y†x| (‖P‖ of the spectral projector,
            // what an outlier deflated by the GMRES polynomial costs in the bound).
            let far = ArnoldiParams {
                wanted: 4,
                which: Which::LargestMagnitude,
                ..params
            };
            let right = arnoldi(
                &mut ruttice::eigen::Shifted {
                    a: Apply(&mut pair, false),
                    z: c,
                },
                x0,
                far,
            );
            let left = arnoldi(
                &mut ruttice::eigen::Shifted {
                    a: Apply(&mut pair, true),
                    z: c.conj(),
                },
                x0,
                far,
            );
            for (k, (theta, x)) in right.values.iter().zip(&right.vectors).enumerate() {
                let lambda = *theta + c;
                let j = (0..left.values.len())
                    .min_by(|&p, &q| {
                        (left.values[p].conj() - *theta)
                            .norm_sqr()
                            .total_cmp(&(left.values[q].conj() - *theta).norm_sqr())
                    })
                    .expect("left pairs");
                let mut d = [C64::ZERO];
                left.vectors[j].dot(x, &mut d);
                println!(
                    "arnoldi   largest |lambda - ({:+.3} {:+.3}i)| = {:.6e}: lambda = {:+.6e} {:+.6e}i  (|A x - l x| {:.1e}, converged {} {}; kappa = 1/|y^dag x| = {:.3e}, left value off by {:.1e})",
                    c.re,
                    c.im,
                    theta.norm_sqr().sqrt(),
                    lambda.re,
                    lambda.im,
                    right.residuals[k],
                    right.converged,
                    left.converged,
                    1.0 / d[0].norm_sqr().sqrt(),
                    (left.values[j].conj() - *theta).norm_sqr().sqrt()
                );
            }
        }
        Ok(())
    }

    /// The norm of the resolvent of Ã M, σ_min(z − Ã M), on the real axis between the origin
    /// and the spectrum and on a circle around 1 (Lanczos on (z − Ã M)†(z − Ã M)): the
    /// ε-pseudospectrum is where it is below ε.
    fn pseudospectrum(
        (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
        (even, odd): (&V, &V),
        lat: &std::sync::Arc<Lattice>,
        (parts, parts_dag): (DeviceParts<f64>, DeviceParts<f64>),
        (exact, hi): (PvEoParams, f64),
        (x0, lanczos, points): (&V, LanczosParams, Vec<C64>),
    ) -> Result<(), Box<dyn Error>> {
        use ruttice::eigen::resolvent_norms;
        let (mut m, mut m_dag) =
            exact_cycles((d, pv), (even, odd), lat, (&parts, &parts_dag), (exact, hi))?;
        let mut pair = product(
            (d, pv),
            (even, odd),
            Box::new(|o, i| m.mult(o, i)),
            Box::new(|o, i| m_dag.mult(o, i)),
        );
        // σ_min² to 1e-3 relative (σ_min to 5e-4) is ample for a bound; each point is printed
        // as it is done.
        let params = LanczosParams {
            wanted: 1,
            tolerance: (1e-3).into(),
            max_restarts: 400,
            ..lanczos
        };
        let mut op = PairOp(&mut pair);
        for z in points {
            let t = Instant::now();
            let r = resolvent_norms(&mut op, &[z], x0, params)[0];
            println!(
                "pseudo z = {:+.3} {:+.3}i: sigma_min(z - A~ M) = {:.6e}  (converged {}, {} applications of the normal operator, {:.0} s)",
                r.z.re,
                r.z.im,
                r.sigma_min,
                r.converged,
                r.applications,
                t.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    /// The points of `--pseudo`: those of every `--contour cx,cy,r,n[,k0,k1]` (n points on the
    /// circle |z − (cx + i cy)| = r, or only points k0..k1 of them) and `--points x0,x1,..` on the real axis; without either, the real
    /// axis at −0.5, 0, 0.25, 0.5, 0.7 and 8 points on |z − 1| = 0.8.
    fn contour_points(args: &[String]) -> Result<Vec<C64>, Box<dyn Error>> {
        let lists = |flag: &str| -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
            let mut out = Vec::new();
            for (i, a) in args.iter().enumerate() {
                if a == flag {
                    let s = args.get(i + 1).ok_or("a value")?;
                    out.push(s.split(',').map(str::parse).collect::<Result<_, _>>()?);
                }
            }
            Ok(out)
        };
        let (mut contours, reals) = (lists("--contour")?, lists("--points")?);
        let mut points: Vec<C64> = reals.concat().iter().map(|&x| C64::new(x, 0.0)).collect();
        if contours.is_empty() && points.is_empty() {
            points = [-0.5, 0.0, 0.25, 0.5, 0.7]
                .iter()
                .map(|&x| C64::new(x, 0.0))
                .collect();
            contours.push(vec![1.0, 0.0, 0.8, 8.0]);
        }
        for c in contours {
            // `cx,cy,r,n[,k0,k1]`: points k0..k1 of the n (all by default), to resume a run.
            let (cx, cy, r, n, k0, k1) = match c[..] {
                [cx, cy, r, n] => (cx, cy, r, n, 0, n as usize),
                [cx, cy, r, n, k0, k1] => (cx, cy, r, n, k0 as usize, k1 as usize),
                _ => return Err("--contour cx,cy,r,n[,k0,k1]".into()),
            };
            for k in k0..k1.min(n as usize) {
                let phi = 2.0 * std::f64::consts::PI * k as f64 / n;
                points.push(C64::new(cx + r * phi.cos(), cy + r * phi.sin()));
            }
        }
        Ok(points)
    }

    /// A pair as an operator with its adjoint.
    struct PairOp<'p, 'a>(&'p mut Pair<'a>);

    impl LinearFn<V> for PairOp<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            (self.0.b)(out, inp);
        }
    }

    impl ruttice::LinearOp<V> for PairOp<'_, '_> {
        fn mult_dag(&mut self, out: &mut V, inp: &V) {
            (self.0.b_dag)(out, inp);
        }
    }

    /// A map of fields on the even sites.
    type Map<'a> = Box<dyn FnMut(&mut V, &V) + 'a>;

    /// B and B† as maps, for any product of Ã with a part of the cycle.
    struct Pair<'a> {
        b: Map<'a>,
        b_dag: Map<'a>,
    }

    /// s (B + B†)/2.
    struct Herm<'p, 'a>(&'p mut Pair<'a>, f64);

    impl LinearFn<V> for Herm<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            let mut t = inp.zeros_like();
            (self.0.b)(out, inp);
            (self.0.b_dag)(&mut t, inp);
            out.axpy(&[C64::new(1.0, 0.0)], &t);
            out.scale(&[0.5 * self.1]);
        }
    }

    /// −R†R, R = I − B: its lowest eigenvalue is −‖I − B‖².
    struct MinusNorm<'p, 'a>(&'p mut Pair<'a>);

    impl LinearFn<V> for MinusNorm<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            let (mut r, mut s) = (inp.zeros_like(), inp.zeros_like());
            (self.0.b)(&mut r, inp);
            r.aypx(&[C64::new(-1.0, 0.0)], inp); // r = x − B x
            (self.0.b_dag)(&mut s, &r);
            s.aypx(&[C64::new(-1.0, 0.0)], &r); // s = r − B† r
            out.copy_from(&s);
            out.scale(&[-1.0]);
        }
    }

    /// Ã X and X† Ã† for a part X of the cycle and its adjoint.
    fn product<'a>(
        (d, pv): (&'a CudaMdwf<f64>, &'a CudaMdwf<f64>),
        (even, odd): (&'a V, &'a V),
        mut x: Map<'a>,
        mut x_dag: Map<'a>,
    ) -> Pair<'a> {
        let mut a = PvRatio::new(d, pv, form(), even, odd);
        let mut a_dag = PvRatio::new(pv, d, form(), even, odd);
        Pair {
            b: Box::new(move |out: &mut V, inp: &V| {
                let mut t = inp.zeros_like();
                x(&mut t, inp);
                a.mult(out, &t);
            }),
            b_dag: Box::new(move |out: &mut V, inp: &V| {
                let mut t = inp.zeros_like();
                a_dag.mult(&mut t, inp);
                x_dag(out, &t);
            }),
        }
    }

    /// Re x† B x for a normalised x.
    fn rayleigh(pair: &mut Pair<'_>, x: &V) -> f64 {
        let mut bx = x.zeros_like();
        (pair.b)(&mut bx, x);
        let mut d = [C64::ZERO];
        x.dot(&bx, &mut d);
        d[0].re
    }

    /// The smoother S and the coarse correction E of the cycle apart: for Ã S, Ã E and Ã M the
    /// field of values and ‖I − Ã X‖₂ (I − Ã M = (I − Ã S)(I − Ã E)(I − Ã S)), and the vector
    /// x at λ_min of the Hermitian part of Ã M: its part in the range of P and its Rayleigh
    /// quotients.
    fn dissect(
        (d, pv): (&CudaMdwf<f64>, &CudaMdwf<f64>),
        (even, odd): (&V, &V),
        lat: &std::sync::Arc<Lattice>,
        basis: &Rc<DeviceBasis>,
        (parts, parts_dag): (DeviceParts<f64>, DeviceParts<f64>),
        (exact, hi): (PvEoParams, f64),
        (x0, x, lanczos): (&V, &V, LanczosParams),
    ) -> Result<(), Box<dyn Error>> {
        let ratio = |a, b| PvRatio::new(a, b, form(), even, odd);
        let smoother = |a, b| {
            Chebyshev::new(
                ratio(a, b),
                exact.chebyshev_steps,
                hi / exact.chebyshev_ratio,
                hi,
                even,
            )
        };
        let correction = |p: &DeviceParts<f64>, a, b| {
            DeviceCorrection::<f64, f64>::from_parts(p.clone(), (a, b), (a, b), form(), exact, even)
        };
        let (mut s, mut s_dag) = (smoother(d, pv), smoother(pv, d));
        let mut with_s = product(
            (d, pv),
            (even, odd),
            Box::new(move |o, i| s.mult(o, i)),
            Box::new(move |o, i| s_dag.mult(o, i)),
        );
        let (mut e, mut e_dag) = (correction(&parts, d, pv)?, correction(&parts_dag, pv, d)?);
        let mut with_e = product(
            (d, pv),
            (even, odd),
            Box::new(move |o, i| e.correct(o, i)),
            Box::new(move |o, i| e_dag.correct(o, i)),
        );
        let (mut m, mut m_dag) =
            exact_cycles((d, pv), (even, odd), lat, (&parts, &parts_dag), (exact, hi))?;
        let mut with_m = product(
            (d, pv),
            (even, odd),
            Box::new(move |o, i| m.mult(o, i)),
            Box::new(move |o, i| m_dag.mult(o, i)),
        );
        for (name, pair) in [
            ("A~ S", &mut with_s),
            ("A~ E", &mut with_e),
            ("A~ M", &mut with_m),
        ] {
            let t = Instant::now();
            let lo = lowest(&mut Herm(pair, 1.0), x0, lanczos);
            let up = lowest(&mut Herm(pair, -1.0), x0, lanczos);
            let norm = lowest(&mut MinusNorm(pair), x0, lanczos);
            println!(
                "parts {name}: Re W in [{:+.6e}, {:+.6e}], |I - {name}|_2 = {:.6e}  (converged {} {} {}, {:.0} s)",
                lo.values[0],
                -up.values[0],
                (-norm.values[0]).max(0.0).sqrt(),
                lo.converged,
                up.converged,
                norm.converged,
                t.elapsed().as_secs_f64()
            );
        }
        // The vector at λ_min of the Hermitian part of Ã M.
        let mut coarse: DeviceCoarse<f64> =
            Transfer::<V, DeviceCoarse<f64>>::coarse_field(&**basis, 1);
        Transfer::restrict(&**basis, &mut coarse, x);
        let mut px = x.zeros_like();
        Transfer::prolong(&**basis, &mut px, &coarse);
        let (mut nx, mut npx) = ([0.0], [0.0]);
        x.norm_sqr(&mut nx);
        px.norm_sqr(&mut npx);
        let mut a = ratio(d, pv);
        let mut ax = x.zeros_like();
        a.mult(&mut ax, x);
        let mut dax = [C64::ZERO];
        x.dot(&ax, &mut dax);
        println!(
            "parts: the vector at lambda_min of A~ M: |P P^dag x|/|x| = {:.4}; Re x^dag A~ x = {:+.4e}, A~ S x {:+.4e}, A~ E x {:+.4e}, A~ M x {:+.4e}",
            (npx[0] / nx[0]).sqrt(),
            dax[0].re,
            rayleigh(&mut with_s, x),
            rayleigh(&mut with_e, x),
            rayleigh(&mut with_m, x)
        );
        // The same correction and cycle with the coarse solves stopped early, as the solver
        // stops them (1e-2), on x and on a random vector v (|x| = |v| = 1): whether a loose
        // coarse solve leaves out the directions in which the exact correction amplifies.
        let unit = |v: &V| {
            let mut n = [0.0];
            v.norm_sqr(&mut n);
            let mut u = v.zeros_like();
            u.copy_from(v);
            u.scale(&[1.0 / n[0].sqrt()]);
            u
        };
        let (ux, uv) = (unit(x), unit(x0));
        for tol in [exact.coarse_tolerance_sqr, 1e-8, 1e-4, 1e-2] {
            let loose = PvEoParams {
                coarse_tolerance_sqr: tol,
                pv_tolerance_sqr: tol,
                ..exact
            };
            let new_correction = || {
                DeviceCorrection::<f64, f64>::from_parts(
                    parts.clone(),
                    (d, pv),
                    (d, pv),
                    form(),
                    loose,
                    even,
                )
            };
            let mut corr = new_correction()?;
            let mut in_cycle = new_correction()?;
            let mut cycle = two_grid(
                (d, pv),
                form(),
                (even, odd),
                lat,
                loose,
                Box::new(move |e, r| in_cycle.correct(e, r)),
                Some(hi),
            )?;
            let mut line = format!("regulator: coarse solves to {tol:.0e}:");
            for (name, v) in [("x", &ux), ("v", &uv)] {
                let (mut ev, mut aev) = (v.zeros_like(), v.zeros_like());
                corr.correct(&mut ev, v);
                a.mult(&mut aev, &ev);
                let mut re_e = [C64::ZERO];
                v.dot(&aev, &mut re_e);
                let mut rest = v.zeros_like();
                rest.copy_from(v);
                rest.axpy(&[C64::new(-1.0, 0.0)], &aev);
                let mut n_rest = [0.0];
                rest.norm_sqr(&mut n_rest);
                let (mut mv, mut amv) = (v.zeros_like(), v.zeros_like());
                cycle.mult(&mut mv, v);
                a.mult(&mut amv, &mv);
                let mut re_m = [C64::ZERO];
                v.dot(&amv, &mut re_m);
                line += &format!(
                    "  {name}: Re {name}^dag A~ E {name} {:+.4e}, |(I - A~ E) {name}| {:.4e}, Re {name}^dag A~ M {name} {:+.4e};",
                    re_e[0].re,
                    n_rest[0].sqrt(),
                    re_m[0].re
                );
            }
            println!("{line}");
        }
        Ok(())
    }
}
