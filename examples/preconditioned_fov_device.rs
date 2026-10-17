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
//!     [--coarse T] [--tol T] [--dim D]
//! ```
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
        CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceSpinor,
    };
    use ruttice::eigen::{lowest, LanczosParams};
    use ruttice::fermion::{EoForm, FermionOp, MdwfParams, PvRatio, SplitParity};
    use ruttice::gauge::smearing::{uniform, Stout};
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::solver::preconditioner::multigrid::{
        Aggregation, BootstrapParams, CoarseKrylov, LowModeParams, PvEoParams,
    };
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
            a: PvRatio::new(d, pv, EoForm::Right, even, odd),
            a_dag: PvRatio::new(pv, d, EoForm::Right, even, odd),
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

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
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
            EoForm::Right,
            exact,
            &even,
        )?;
        let mut c_dag = DeviceCorrection::<f64, f64>::shared(
            basis,
            (pv, d),
            (pv, d),
            EoForm::Right,
            exact,
            &even,
        )?;
        let hi = top_of(
            &mut PvRatio::new(d, pv, EoForm::Right, &even, &odd),
            &lat,
            &even,
            exact.seed,
            exact.power_steps,
        )?;
        let mut m = two_grid(
            (d, pv),
            EoForm::Right,
            (&even, &odd),
            &lat,
            exact,
            Box::new(move |e, r| c.correct(e, r)),
            Some(hi),
        )?;
        let mut m_dag = two_grid(
            (pv, d),
            EoForm::Right,
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
            tolerance: tol,
            max_restarts: 2000,
            chebyshev: None,
        };
        let x0 = random_even(63)?;
        for preconditioned in [false, true] {
            let mut ends = Vec::new();
            let t = Instant::now();
            for sign in [1.0, -1.0] {
                let cycles = preconditioned.then_some((&mut m, &mut m_dag));
                let mut h = part((d, pv), (&even, &odd), cycles, sign);
                let e = lowest(&mut h, &x0, lanczos);
                ends.push((sign * e.values[0], e.converged, e.restarts));
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
        Ok(())
    }
}
