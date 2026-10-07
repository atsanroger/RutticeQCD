//! Checks: the Wilson operator.

mod common;

use common::{
    bridge_test_file, gauge_transform, gauge_transform_spinor, max_abs_diff, BRIDGE_TESTS_ENV,
};
use ruttice::fermion::{FermionOp, HopTerms, Variant, Wilson, WilsonParams};
use ruttice::gauge::set_unit;
use ruttice::gauge::{random_gauge, random_su3};
use ruttice::io::read_bridge_text;
use ruttice::kernel::{Cplx, Dirac, GammaRepre, Spinor as SiteSpinor, SPINOR_REALS};
use ruttice::{dot, fill, norm_sqr, Boundary, Field, Gauge, Lattice, Spinor, ND};
use ruttice::{random_spinor, TestRng};
use std::sync::Arc;

const PERIODIC: [Boundary; ND] = [Boundary::Periodic; ND];
const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];
/// One of each kind, to exercise every branch of the boundary handling.
const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

fn wilson(
    lat: &Arc<Lattice>,
    kappa: f64,
    boundary: [Boundary; ND],
    u: &Field<f64, Gauge>,
) -> Wilson<f64> {
    let mut op = Wilson::new(lat, WilsonParams { kappa, boundary });
    op.set_gauge(u).unwrap();
    op
}

fn mult(op: &Wilson<f64>, inp: &Field<f64, Spinor>, variant: Variant) -> Field<f64, Spinor> {
    let mut out = Field::<f64, Spinor>::spinor(inp.lattice(), inp.num_vectors());
    op.mult(&mut op.work(), &mut out, inp, variant);
    out
}

/// The operator and the inner product give the same bits for every number of threads, on a
/// lattice large enough for the site loops to be split over the thread pool.
#[test]
fn results_do_not_depend_on_thread_count() {
    let lat = Lattice::single_node([8, 8, 8, 8]);
    assert!(lat.num_sites() > 2 * ruttice::min_sites_per_task());
    let op = wilson(&lat, 0.12, MIXED, &random_gauge(&lat, 21));
    let psi = random_spinor(&lat, 2, 22);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let out = mult(&op, &psi, Variant::Dagger);
                let overlap = dot(&psi, &out);
                (out, overlap)
            })
    };
    let (out1, dot1) = run(1);
    for threads in [3, 8] {
        let (out, overlap) = run(threads);
        assert!(out.as_slice() == out1.as_slice(), "{threads} threads");
        assert_eq!(overlap, dot1, "{threads} threads");
    }
}

// ---- Physics checks: no reference values needed ----

/// With unit links and periodic boundaries a plane wave is an eigenvector of the hopping
/// term up to a spin matrix:
/// D ψ = (1 - 2κ Σ cos p_mu) ψ + 2iκ Σ sin p_mu γ_mu ψ.
#[test]
fn free_field_plane_wave() {
    let dims = [4, 6, 4, 8];
    let lat = Lattice::single_node(dims);
    let mut u = Field::<f64, Gauge>::gauge(&lat);
    set_unit(&mut u);
    let kappa = 0.11;
    let op = wilson(&lat, kappa, PERIODIC, &u);

    let n = [1.0, 2.0, 0.0, 3.0];
    let p: [f64; ND] =
        core::array::from_fn(|mu| 2.0 * std::f64::consts::PI * n[mu] / dims[mu] as f64);
    let mut rng = TestRng::new(3);
    let chi_reals: [f64; SPINOR_REALS] = core::array::from_fn(|_| rng.uniform());
    let chi = SiteSpinor::from_reals(&chi_reals);
    let scaled = |s: &SiteSpinor<f64>, z: Cplx<f64>| {
        SiteSpinor(core::array::from_fn(|a| {
            core::array::from_fn(|c| z * s.0[a][c])
        }))
    };

    let mut psi = Field::<f64, Spinor>::spinor(&lat, 1);
    let mut want = Field::<f64, Spinor>::spinor(&lat, 1);
    for site in 0..lat.num_sites() {
        let x = lat.coords(site);
        let phase: f64 = (0..ND).map(|mu| p[mu] * x[mu] as f64).sum();
        let wave = scaled(&chi, Cplx::new(phase.cos(), phase.sin()));
        wave.write_reals(psi.site_mut(0, site));

        let cos_sum: f64 = p.iter().map(|q| q.cos()).sum();
        let mut d = scaled(&wave, Cplx::new(1.0 - 2.0 * kappa * cos_sum, 0.0));
        for mu in 0..ND {
            let g = Dirac::GAMMA[mu].apply(&wave);
            d = d.add(&scaled(&g, Cplx::new(0.0, 2.0 * kappa * p[mu].sin())));
        }
        d.write_reals(want.site_mut(0, site));
    }

    let got = mult(&op, &psi, Variant::D);
    assert!(norm_sqr(&got) > 1.0);
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 1e-14, "max difference {diff}");
}

/// D[U^g] (g ψ) = g (D[U] ψ).
#[test]
fn gauge_covariance() {
    let lat = Lattice::single_node([4, 4, 6, 8]);
    let u = random_gauge(&lat, 101);
    let psi = random_spinor(&lat, 1, 7);
    let mut rng = TestRng::new(55);
    let g: Vec<_> = (0..lat.num_sites()).map(|_| random_su3(&mut rng)).collect();
    // The transformed links wrap periodically; the fermion phase is separate, so this holds
    // for any boundary condition.
    let st = ruttice::Stencil::nearest(&lat, PERIODIC);
    let ug = gauge_transform(&u, &g, &st);

    for boundary in [ANTIPERIODIC, MIXED] {
        let lhs = mult(
            &wilson(&lat, 0.13, boundary, &ug),
            &gauge_transform_spinor(&psi, &g),
            Variant::D,
        );
        let rhs = gauge_transform_spinor(
            &mult(&wilson(&lat, 0.13, boundary, &u), &psi, Variant::D),
            &g,
        );
        let diff = max_abs_diff(&lhs, &rhs);
        assert!(diff < 1e-13, "max difference {diff}");
    }
}

/// <χ, D ψ> = <D† χ, ψ>, and γ5 D is exactly γ5 applied to D ψ.
#[test]
fn dagger_is_the_adjoint_and_gamma5d_is_gamma5_times_d() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 21);
    let psi = random_spinor(&lat, 1, 8);
    let chi = random_spinor(&lat, 1, 9);
    for boundary in [PERIODIC, ANTIPERIODIC, MIXED] {
        let op = wilson(&lat, 0.12, boundary, &u);
        let d_psi = mult(&op, &psi, Variant::D);
        let ddag_chi = mult(&op, &chi, Variant::Dagger);
        let (a_re, a_im) = dot(&chi, &d_psi);
        let (b_re, b_im) = dot(&ddag_chi, &psi);
        assert!(a_re.abs() > 1.0, "the test should not be trivially zero");
        assert!(
            (a_re - b_re).abs() < 1e-11 && (a_im - b_im).abs() < 1e-11,
            "{a_re} {b_re} {a_im} {b_im}"
        );

        let mut g5_d = Field::<f64, Spinor>::spinor(&lat, 1);
        op.mult_gamma5(&mut op.work(), &mut g5_d, &d_psi);
        assert_eq!(
            mult(&op, &psi, Variant::Gamma5D).as_slice(),
            g5_d.as_slice()
        );
    }
}

/// The fused sweep equals the diagonal term plus the eight hopping terms applied one by one.
#[test]
fn fused_kernel_equals_sum_of_hop_terms() {
    let lat = Lattice::single_node([4, 6, 4, 8]);
    let u = random_gauge(&lat, 33);
    let psi = random_spinor(&lat, 2, 12);
    for boundary in [ANTIPERIODIC, MIXED] {
        let op = wilson(&lat, 0.125, boundary, &u);
        let mut work = op.work();

        let mut hops = Field::<f64, Spinor>::spinor(&lat, 2);
        for mu in 0..ND {
            op.hop_forward(&mut work, mu, &mut hops, &psi);
            op.hop_backward(&mut work, mu, &mut hops, &psi);
        }
        let mut parts = Field::<f64, Spinor>::spinor(&lat, 2);
        op.mult_diag(&mut work, &mut parts, &psi);
        let c = op.hop_coefficient();
        for (p, h) in parts.as_mut_slice().iter_mut().zip(hops.as_slice()) {
            *p += c * h;
        }

        let fused = mult(&op, &psi, Variant::D);
        let diff = max_abs_diff(&fused, &parts);
        assert!(diff < 1e-14, "max difference {diff}");
    }
}

/// Several spinor fields stored together give the same bits as applying the operator to
/// each one separately.
#[test]
fn vectors_stored_together_are_independent() {
    let lat = Lattice::single_node([4, 4, 4, 4]);
    let u = random_gauge(&lat, 2);
    let op = wilson(&lat, 0.12, ANTIPERIODIC, &u);
    let nv = 3;
    let all = random_spinor(&lat, nv, 40);
    let all_out = mult(&op, &all, Variant::Dagger);
    for r in 0..nv {
        let column = |f: &Field<f64, Spinor>| {
            let mut one = Field::<f64, Spinor>::spinor(&lat, 1);
            for site in 0..lat.num_sites() {
                one.site_mut(0, site)
                    .copy_from_slice(&f.site(0, site)[r * SPINOR_REALS..(r + 1) * SPINOR_REALS]);
            }
            one
        };
        assert_eq!(
            mult(&op, &column(&all), Variant::Dagger).as_slice(),
            column(&all_out).as_slice()
        );
    }
}

#[test]
fn single_precision_agrees_with_double() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = random_gauge(&lat, 77);
    let psi = random_spinor(&lat, 1, 78);
    let params = WilsonParams {
        kappa: 0.12,
        boundary: ANTIPERIODIC,
    };
    let want = mult(
        &wilson(&lat, params.kappa, params.boundary, &u),
        &psi,
        Variant::D,
    );

    let mut op = Wilson::<f32>::new(&lat, params);
    op.set_gauge(&u).unwrap();
    let mut inp = Field::<f32, Spinor>::spinor(&lat, 1);
    for (dst, src) in inp.as_mut_slice().iter_mut().zip(psi.as_slice()) {
        *dst = *src as f32;
    }
    let mut out = Field::<f32, Spinor>::spinor(&lat, 1);
    op.mult(&mut op.work(), &mut out, &inp, Variant::D);

    let worst = out
        .as_slice()
        .iter()
        .zip(want.as_slice())
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - y).abs()));
    assert!(worst < 2e-6, "max difference {worst}");
    assert!(
        worst > 1e-10,
        "single precision should not reproduce double exactly"
    );
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Mult_Wilson` (tests/test_Mult_Wilson.yaml):
/// the 2-norm of D applied once to a spinor whose reals are all 1, with κ = 0.12 and
/// antiperiodic boundaries in all four directions.
const BRIDGE_MULT_WILSON_NORM: f64 = 1.23160709257508e+02;

#[test]
fn bridge_mult_wilson() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let u = read_bridge_text(&path, &lat).unwrap();
    let op = wilson(&lat, 0.12, ANTIPERIODIC, &u);

    let mut b = Field::<f64, Spinor>::spinor(&lat, 1);
    fill(&mut b, 1.0);
    let norm = norm_sqr(&mult(&op, &b, Variant::D)).sqrt();
    eprintln!("|D b| = {norm:.14e}, Bridge++ = {BRIDGE_MULT_WILSON_NORM:.14e}");
    let relative = (norm - BRIDGE_MULT_WILSON_NORM).abs() / BRIDGE_MULT_WILSON_NORM;
    assert!(relative < 1e-13, "relative difference {relative:e}");
}
