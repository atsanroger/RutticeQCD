//! Solves with D† on the device on a right-hand side like those that stagnate in HMC: the
//! residual r = b − D†D x₀ of one sweep of the normal equations, after D† and D solved to 1e-10.
//! Compared: the cycle of Ã† on the P of Ã (as the HMC multigrid makes it), GCR(2) and GCR(8);
//! and flexible GCR on D† on the whole lattice preconditioned by Γ5 M_D Γ5, M_D a few GCR(2)
//! steps of the multigrid of D (D⁻† = (D₋†)⁻¹ Γ5 D⁻¹ Γ5 D₋†, so Γ5 D⁻¹ Γ5 is D⁻† up to the
//! similarity by D₋† = c W† − 1).
//!
//! ```text
//! cargo run --release --features cuda-bridge --example dagger_precond -- L FORMAT FILE M5 SMEAR
//!     [--steps S] [--ratio R] [--mass M] [--seed S]
//! ```
//!
//! On the L⁴ configuration FILE; SMEAR 1 for stout 0.1 × 3; Ls 8, b 1.5, c 0.5, blocks of 4⁴,
//! 32 test vectors (16 filtered in 2 rounds, the bootstrap 2 × 4), Chebyshev 8 steps on
//! [hi/10, hi] (setup S; B: `--steps 12 --ratio 30`), m 0.01 by default.

#[cfg(feature = "cuda-bridge")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    device::main()
}

#[cfg(not(feature = "cuda-bridge"))]
fn main() {
    eprintln!("dagger_precond needs --features cuda-bridge");
}

#[cfg(feature = "cuda-bridge")]
mod device {
    use ruttice::backend::cuda_bridge::multigrid::{
        bootstrap, device_correction, filtered_test_vectors, two_grid, DeviceCycle, Stage,
    };
    use ruttice::backend::cuda_bridge::{
        CudaMdwf, DeviceBasis, DeviceCorrection, DeviceParities, DeviceSpinor,
    };
    use ruttice::fermion::{
        adjoint_right_hand_side, adjoint_solution, EoForm, FermionOp, MdwfParams, PvRatio, Variant,
    };
    use ruttice::gauge::smearing::{uniform, Stout};
    use ruttice::gauge::{GaugeSmearing, Smear};
    use ruttice::hmc::LinkField;
    use ruttice::solver::preconditioner::multigrid::{
        Aggregation, BootstrapParams, CoarseKrylov, LowModeParams, PvEoParams,
    };
    use ruttice::solver::{Gcr, SolveError, Solver, Stop};
    use ruttice::{
        random_spinor, Boundary, EvenOddIndex, Field, Lattice, LinearFn, Parity, Spinor, Vector,
        C64, ND,
    };
    use std::cell::{Cell, RefCell};
    use std::error::Error;
    use std::path::Path;
    use std::rc::Rc;
    use std::time::Instant;

    type V = DeviceSpinor<f64>;

    /// D or D† on the whole lattice.
    struct Whole<'a> {
        d: &'a CudaMdwf<f64>,
        variant: Variant,
    }

    impl LinearFn<V> for Whole<'_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            self.d.mult(&mut self.d.work(), out, inp, self.variant);
        }
    }

    /// x ≈ D⁻¹ b on the whole lattice: GCR(2) on Ã with the cycle M, at most `its` steps.
    struct DSolve<'a> {
        d: &'a CudaMdwf<f64>,
        pv: &'a CudaMdwf<f64>,
        even: &'a V,
        odd: &'a V,
        m: RefCell<DeviceCycle<'a, f64>>,
        cycles: Cell<usize>,
    }

    impl DSolve<'_> {
        fn solve(&self, x: &mut V, b: &V, its: usize, tol: f64) {
            let mut a = PvRatio::new(self.d, self.pv, EoForm::Right, self.even, self.odd);
            let mut rhs = self.even.zeros_with(1);
            a.right_hand_side(&DeviceParities, &mut rhs, b);
            let mut y = rhs.zeros_like();
            let mut m = self.m.borrow_mut();
            let info = Gcr::new(Stop::simple(tol, its), 2).solve(
                &mut a,
                Some(&mut *m as &mut dyn LinearFn<V>),
                &mut y,
                &rhs,
            );
            self.cycles.set(
                self.cycles.get()
                    + match info {
                        Ok(i) | Err(SolveError::NotConverged(i)) => i.iterations,
                        Err(_) => 0,
                    },
            );
            a.solution(&DeviceParities, x, &y, b);
        }
    }

    /// Γ5 M_D Γ5 with `its` steps of the solve of D.
    struct Gamma5Prec<'a, 'b> {
        solve: &'b DSolve<'a>,
        d: &'a CudaMdwf<f64>,
        its: usize,
    }

    impl LinearFn<V> for Gamma5Prec<'_, '_> {
        fn mult(&mut self, out: &mut V, inp: &V) {
            let mut t = inp.zeros_like();
            self.d.mult_gamma5_on(&mut t, inp);
            let mut x = inp.zeros_like();
            self.solve.solve(&mut x, &t, self.its, 1e-30);
            self.d.mult_gamma5_on(out, &x);
        }
    }

    fn outcome(info: Result<ruttice::solver::SolveInfo, SolveError>) -> (usize, bool) {
        match info {
            Ok(i) => (i.iterations, true),
            Err(SolveError::NotConverged(i)) => (i.iterations, false),
            Err(_) => (0, false),
        }
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let value = |name: &str, default: f64| -> Result<f64, Box<dyn Error>> {
            match args.iter().position(|a| a == name) {
                Some(i) => Ok(args.get(i + 1).ok_or("a value")?.parse()?),
                None => Ok(default),
            }
        };
        let l: usize = args[0].parse()?;
        let m5: f64 = args[3].parse()?;
        let smear = args[4] == "1";
        let quark = value("--mass", 0.01)?;
        let cycle = PvEoParams {
            chebyshev_steps: value("--steps", 8.0)? as usize,
            chebyshev_ratio: value("--ratio", 10.0)?,
            ..PvEoParams::default()
        };
        let dual = PvEoParams {
            pv_solver: CoarseKrylov::BiCgStab,
            ..cycle
        };
        let lat = Lattice::single_node([l; ND]);
        let u = ruttice::app::read_gauge_named(&args[1], Path::new(&args[2]), &lat)?;
        let v = match smear {
            true => GaugeSmearing {
                smear: Smear::Stout(Stout { rho: uniform(0.1) }),
                steps: 3,
            }
            .levels(&u)
            .pop()
            .unwrap(),
            false => u.duplicate(),
        };
        let mk = |mass: f64| -> Result<CudaMdwf<f64>, Box<dyn Error>> {
            let mut op = CudaMdwf::<f64>::new(
                &lat,
                MdwfParams {
                    ls: 8,
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
                },
            )?;
            op.set_gauge(&v)?;
            Ok(op)
        };
        let (d, pv) = (mk(quark)?, mk(1.0)?);
        let even = V::shape_half(&lat, 8, Parity::Even)?;
        let odd = V::shape_half(&lat, 8, Parity::Odd)?;
        let index = EvenOddIndex::new(&lat);
        let agg = Aggregation::new(&lat, [4; ND]);
        let quiet = &mut |_: Stage<'_, f64>| {};
        println!(
            "# {l}^4 M5 {m5} stout {smear}, m {quark}, Ls 8, b 1.5 c 0.5; 32 vectors, blocks 4^4, Chebyshev {} on [hi/{}, hi]",
            cycle.chebyshev_steps, cycle.chebyshev_ratio
        );
        let t = Instant::now();
        let mut y = filtered_test_vectors(
            (&d, &pv),
            (&even, &odd),
            &lat,
            &index,
            LowModeParams::default(),
            32,
            quiet,
        )?;
        let ritz = {
            let mut c = y.zeros_like();
            c.copy_from(&y);
            c
        };
        let mut with = device_correction((&d, &pv), &agg, cycle, &even, true);
        bootstrap(
            (&d, &pv),
            (&even, &odd),
            &lat,
            &mut y,
            BootstrapParams {
                cycle,
                ..BootstrapParams::default()
            },
            &mut with,
            quiet,
        )?;
        drop(with);
        let basis = Rc::new(DeviceBasis::pv_eo(&d, &pv, &agg, &y, true)?);
        println!("# setup {:.1} s", t.elapsed().as_secs_f64());
        let mut c = DeviceCorrection::<f64, f32>::shared(
            basis.clone(),
            (&d, &pv),
            (&d, &pv),
            EoForm::Right,
            cycle,
            &even,
        )?;
        let m = two_grid(
            (&d, &pv),
            EoForm::Right,
            (&even, &odd),
            &lat,
            cycle,
            Box::new(move |e, r| c.correct(e, r)),
            None,
        )?;
        let dsolve = DSolve {
            d: &d,
            pv: &pv,
            even: &even,
            odd: &odd,
            m: RefCell::new(m),
            cycles: Cell::new(0),
        };
        let mut c_dag = DeviceCorrection::<f64, f32>::shared(
            basis.clone(),
            (&pv, &d),
            (&pv, &d),
            EoForm::Right,
            dual,
            &even,
        )?;
        let mut m_dag = two_grid(
            (&pv, &d),
            EoForm::Right,
            (&even, &odd),
            &lat,
            dual,
            Box::new(move |e, r| c_dag.correct(e, r)),
            None,
        )?;
        // D† w = b on Ã† by GCR(k) with the cycle `m`.
        let by_cycle =
            |m: &mut dyn LinearFn<V>, w: &mut V, b: &V, k: usize, tol: f64, max: usize| {
                let mut rhs = even.zeros_with(1);
                adjoint_right_hand_side(&d, &DeviceParities, &mut rhs, b);
                let mut a = PvRatio::new(&pv, &d, EoForm::Right, &even, &odd);
                let mut yv = rhs.zeros_like();
                let info =
                    Gcr::new(Stop::simple(tol, max), k).solve(&mut a, Some(m), &mut yv, &rhs);
                adjoint_solution(&d, &pv, &DeviceParities, w, &yv, b);
                outcome(info)
            };
        // D† w = b by the dual cycle on Ã†, GCR(k).
        let mut dual_solve = |w: &mut V, b: &V, k: usize, tol: f64, max: usize| {
            let mut rhs = even.zeros_with(1);
            adjoint_right_hand_side(&d, &DeviceParities, &mut rhs, b);
            let mut a = PvRatio::new(&pv, &d, EoForm::Right, &even, &odd);
            let mut yv = rhs.zeros_like();
            let info = Gcr::new(Stop::simple(tol, max), k).solve(
                &mut a,
                Some(&mut m_dag as &mut dyn LinearFn<V>),
                &mut yv,
                &rhs,
            );
            adjoint_solution(&d, &pv, &DeviceParities, w, &yv, b);
            outcome(info)
        };
        let relative = |op: Variant, x: &V, b: &V| {
            let mut ax = b.zeros_like();
            d.mult(&mut d.work(), &mut ax, x, op);
            ax.axpy(&[C64::new(-1.0, 0.0)], b);
            let (mut n, mut nb) = ([0.0], [0.0]);
            ax.norm_sqr(&mut n);
            b.norm_sqr(&mut nb);
            n[0] / nb[0]
        };

        // The right-hand sides: random b, and the residual of one sweep of D†D x = b.
        let mut host = Field::<f64, Spinor>::with_shape(&lat, 24, 8, 1, 1);
        host.as_mut_slice()
            .copy_from_slice(random_spinor(&lat, 8, value("--seed", 91.0)? as u64).as_slice());
        let b = V::from_host(&host)?;
        let mut w0 = b.zeros_like();
        // Sweep 1: D† by flexible GCR(8) with Γ5 M_D Γ5 (as HMC would solve it), or the dual.
        let nodual = value("--nodual", 0.0)? != 0.0;
        let (it, ok) = if nodual {
            let mut prec = Gamma5Prec {
                solve: &dsolve,
                d: &d,
                its: 1,
            };
            let mut a = Whole {
                d: &d,
                variant: Variant::Dagger,
            };
            outcome(Gcr::new(Stop::simple(1e-10, 2000), 8).solve(
                &mut a,
                Some(&mut prec as &mut dyn LinearFn<V>),
                &mut w0,
                &b,
            ))
        } else {
            dual_solve(&mut w0, &b, 8, 1e-10, 2000)
        };
        let mut x0 = b.zeros_like();
        dsolve.solve(&mut x0, &w0, 2000, 1e-10);
        let mut dx = b.zeros_like();
        d.mult(&mut d.work(), &mut dx, &x0, Variant::D);
        let mut r = b.zeros_like();
        d.mult(&mut d.work(), &mut r, &dx, Variant::Dagger);
        r.aypx(&[C64::new(-1.0, 0.0)], &b);
        let (mut nr, mut nb) = ([0.0], [0.0]);
        r.norm_sqr(&mut nr);
        b.norm_sqr(&mut nb);
        println!(
            "# sweep 1: D† by the dual GCR(8) {it} its (converged {ok}); |b - D†D x0|²/|b|² = {:.1e}",
            nr[0] / nb[0]
        );
        for (name, rhs) in [("random b", &b), ("sweep residual r", &r)] {
            for k in [2, 8].into_iter().filter(|_| !nodual) {
                let t = Instant::now();
                let mut w = rhs.zeros_like();
                let (its, ok) = dual_solve(&mut w, rhs, k, 1e-10, 300);
                println!(
                    "{name:18} dual cycle on Ã†, GCR({k}): {its:4} its{}  |D†w - b|²/|b|² {:.1e}  {:.2} s",
                    if ok { "" } else { " (not converged)" },
                    relative(Variant::Dagger, &w, rhs),
                    t.elapsed().as_secs_f64()
                );
            }
            let ks: Vec<usize> = match nodual {
                true => vec![2, 4],
                false => vec![2, 8],
            };
            let js: Vec<usize> = match nodual {
                true => vec![1],
                false => vec![1, 2, 4],
            };
            for &k in &ks {
                for &j in &js {
                    let t = Instant::now();
                    dsolve.cycles.set(0);
                    let mut prec = Gamma5Prec {
                        solve: &dsolve,
                        d: &d,
                        its: j,
                    };
                    let mut a = Whole {
                        d: &d,
                        variant: Variant::Dagger,
                    };
                    let mut w = rhs.zeros_like();
                    let (its, ok) = outcome(Gcr::new(Stop::simple(1e-10, 300), k).solve(
                        &mut a,
                        Some(&mut prec as &mut dyn LinearFn<V>),
                        &mut w,
                        rhs,
                    ));
                    println!(
                        "{name:18} GCR({k}) on D† with Γ5 M_D Γ5, {j} inner: {its:4} its{} ({} cycles of D)  |D†w - b|²/|b|² {:.1e}  {:.2} s",
                        if ok { "" } else { " (not converged)" },
                        dsolve.cycles.get(),
                        relative(Variant::Dagger, &w, rhs),
                        t.elapsed().as_secs_f64()
                    );
                }
            }
        }
        if value("--cheap", 0.0)? != 0.0 {
            // Cheap variants on the shared P: the coarse solves of M† and its smoother, and a
            // deflated initial guess from U = Ã_R V.
            let mut u_c = ritz.zeros_like();
            PvRatio::new(&d, &pv, EoForm::Right, &even, &odd).mult(&mut u_c, &ritz);
            let fixed = |kind: CoarseKrylov, its: usize| (kind, 1e-30, its);
            let variants: Vec<(&str, PvEoParams)> = vec![
                (
                    "PV BiCGStab 1e-1",
                    PvEoParams {
                        coarse_tolerance_sqr: 1e-1,
                        ..dual
                    },
                ),
                (
                    "PV BiCGStab 1e-4",
                    PvEoParams {
                        coarse_tolerance_sqr: 1e-4,
                        ..dual
                    },
                ),
                ("PV BiCGStab 8 its fixed", {
                    let (k, t, i) = fixed(CoarseKrylov::BiCgStab, 8);
                    PvEoParams {
                        d_solver: k,
                        coarse_tolerance_sqr: t,
                        coarse_max_iterations: i,
                        ..dual
                    }
                }),
                ("PV GCR(4) 4 its fixed", {
                    let (k, t, i) = fixed(CoarseKrylov::Gcr { k: 4 }, 4);
                    PvEoParams {
                        d_solver: k,
                        coarse_tolerance_sqr: t,
                        coarse_max_iterations: i,
                        ..dual
                    }
                }),
                (
                    "PV GCR(4) 4 its, quark BiCGStab 16 its, fixed",
                    PvEoParams {
                        d_solver: CoarseKrylov::Gcr { k: 4 },
                        coarse_tolerance_sqr: 1e-30,
                        coarse_max_iterations: 4,
                        pv_tolerance_sqr: 1e-30,
                        pv_max_iterations: 16,
                        ..dual
                    },
                ),
                (
                    "smoother 12 steps",
                    PvEoParams {
                        chebyshev_steps: 12,
                        ..dual
                    },
                ),
            ];
            for (vname, params) in variants {
                let mut c_v = DeviceCorrection::<f64, f32>::shared(
                    basis.clone(),
                    (&pv, &d),
                    (&pv, &d),
                    EoForm::Right,
                    params,
                    &even,
                )?;
                let mut m_v = two_grid(
                    (&pv, &d),
                    EoForm::Right,
                    (&even, &odd),
                    &lat,
                    params,
                    Box::new(move |e, r| c_v.correct(e, r)),
                    None,
                )?;
                for (name, rhs) in [("random b", &b), ("sweep residual r", &r)] {
                    let t = Instant::now();
                    let mut w = rhs.zeros_like();
                    let (its, ok) = by_cycle(&mut m_v, &mut w, rhs, 2, 1e-10, 300);
                    println!(
                        "{name:18} shared P, {vname}, GCR(2): {its:4} its{}  |D†w - b|²/|b|² {:.1e}  {:.2} s",
                        if ok { "" } else { " (not converged)" },
                        relative(Variant::Dagger, &w, rhs),
                        t.elapsed().as_secs_f64()
                    );
                }
            }
            // Deflated start on Ã†: z0 = Z (Z† Ã† Z)⁻¹ Z† ρ with Z = U, then GCR(2) with M†.
            use ruttice::Subspace;
            let n = u_c.num_vectors();
            let mut az = u_c.zeros_like();
            PvRatio::new(&pv, &d, EoForm::Right, &even, &odd).mult(&mut az, &u_c);
            let gmat = u_c.gram(&az);
            for (name, rhs) in [("random b", &b), ("sweep residual r", &r)] {
                let t = Instant::now();
                let mut rho = even.zeros_with(1);
                adjoint_right_hand_side(&d, &DeviceParities, &mut rho, rhs);
                let zb = u_c.gram(&rho);
                // Solve gmat c = zb (n×n, complex) by Gaussian elimination with pivoting.
                let mut a: Vec<Vec<C64>> = (0..n)
                    .map(|i| {
                        let mut row: Vec<C64> = (0..n).map(|j| gmat[i * n + j]).collect();
                        row.push(zb[i]);
                        row
                    })
                    .collect();
                for col in 0..n {
                    let p = (col..n)
                        .max_by(|&x, &y| {
                            a[x][col]
                                .norm_sqr()
                                .partial_cmp(&a[y][col].norm_sqr())
                                .unwrap()
                        })
                        .unwrap();
                    a.swap(col, p);
                    for row in 0..n {
                        if row != col {
                            let f = a[row][col] / a[col][col];
                            for k in col..=n {
                                let v = a[col][k];
                                a[row][k] = a[row][k] - f * v;
                            }
                        }
                    }
                }
                let coef: Vec<C64> = (0..n).map(|i| a[i][n] / a[i][i]).collect();
                let mut rot = vec![C64::new(0.0, 0.0); n * n];
                for i in 0..n {
                    rot[i * n] = coef[i];
                }
                let mut z = u_c.zeros_like();
                z.copy_from(&u_c);
                z.rotate(&rot);
                let mut yv = z.select(&[0]);
                let mut ad = PvRatio::new(&pv, &d, EoForm::Right, &even, &odd);
                let info = Gcr::new(Stop::simple(1e-10, 300), 2).solve(
                    &mut ad,
                    Some(&mut m_dag as &mut dyn LinearFn<V>),
                    &mut yv,
                    &rho,
                );
                let (its, ok) = outcome(info);
                let mut w = rhs.zeros_like();
                adjoint_solution(&d, &pv, &DeviceParities, &mut w, &yv, rhs);
                println!(
                    "{name:18} shared P, deflated start from U ({n}), GCR(2): {its:4} its{}  |D†w - b|²/|b|² {:.1e}  {:.2} s",
                    if ok { "" } else { " (not converged)" },
                    relative(Variant::Dagger, &w, rhs),
                    t.elapsed().as_secs_f64()
                );
            }
            return Ok(());
        }
        if value("--pl", 1.0)? == 0.0 {
            return Ok(());
        }
        // The left side: P_L from U = Ã_R V (the Ritz vectors of the filter), from U after a
        // bootstrap of Ã†, and from a second setup of Ã†; the cycle of Ã† on P_L with the
        // Pauli-Villars factor by BiCGStab or GCR(4).
        let mut u = ritz.zeros_like();
        PvRatio::new(&d, &pv, EoForm::Right, &even, &odd).mult(&mut u, &ritz);
        let dual_boot = |mut z: V, params: PvEoParams| -> Result<V, Box<dyn Error>> {
            let mut with = device_correction((&pv, &d), &agg, params, &even, true);
            bootstrap(
                (&pv, &d),
                (&even, &odd),
                &lat,
                &mut z,
                BootstrapParams {
                    cycle: params,
                    ..BootstrapParams::default()
                },
                &mut with,
                &mut |_: Stage<'_, f64>| {},
            )?;
            Ok(z)
        };
        let gcr4 = PvEoParams {
            d_solver: CoarseKrylov::Gcr { k: 4 },
            ..dual
        };
        let copy = |z: &V| {
            let mut c = z.zeros_like();
            c.copy_from(z);
            c
        };
        for (pname, params) in [("PV factor BiCGStab", dual), ("PV factor GCR(4)", gcr4)] {
            for which in [
                "U = Ã V",
                "U after a bootstrap of Ã†",
                "a second setup of Ã†",
            ] {
                let t = Instant::now();
                let z = match which {
                    "U = Ã V" => copy(&u),
                    "U after a bootstrap of Ã†" => dual_boot(copy(&u), params)?,
                    _ => {
                        let z = filtered_test_vectors(
                            (&pv, &d),
                            (&even, &odd),
                            &lat,
                            &index,
                            LowModeParams::default(),
                            32,
                            &mut |_: Stage<'_, f64>| {},
                        )?;
                        dual_boot(z, params)?
                    }
                };
                let basis_l = Rc::new(DeviceBasis::pv_eo(&pv, &d, &agg, &z, true)?);
                let made = t.elapsed().as_secs_f64();
                let mut c_l = DeviceCorrection::<f64, f32>::shared(
                    basis_l,
                    (&pv, &d),
                    (&pv, &d),
                    EoForm::Right,
                    params,
                    &even,
                )?;
                let mut m_l = two_grid(
                    (&pv, &d),
                    EoForm::Right,
                    (&even, &odd),
                    &lat,
                    params,
                    Box::new(move |e, r| c_l.correct(e, r)),
                    None,
                )?;
                for (name, rhs) in [("random b", &b), ("sweep residual r", &r)] {
                    let t = Instant::now();
                    let mut w = rhs.zeros_like();
                    let (its, ok) = by_cycle(&mut m_l, &mut w, rhs, 2, 1e-10, 300);
                    println!(
                        "{name:18} P_L from {which}, {pname}, GCR(2): {its:4} its{}  |D†w - b|²/|b|² {:.1e}  {:.2} s (P_L {made:.1} s)",
                        if ok { "" } else { " (not converged)" },
                        relative(Variant::Dagger, &w, rhs),
                        t.elapsed().as_secs_f64()
                    );
                }
            }
        }
        Ok(())
    }
}
