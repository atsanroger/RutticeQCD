//! Checks: even-odd preconditioning of the Wilson operator.

use ruttice_core::{
    extract_parity, fill, insert_parity, norm_sqr, Boundary, Field, Gauge, Half, Lattice, LinearFn,
    LinearOp, Parity, Spinor, Vector, C64, ND,
};
use ruttice_fermion::{
    solve_even_odd, Bound, Dagger, EvenOdd, FermionOp, Schur, Variant, Wilson, WilsonParams,
};
use ruttice_io::read_bridge_text;
use ruttice_solver::{solve_cgne, Cg, Stop};
use ruttice_validation::{
    bridge_test_file, max_abs_diff, random_gauge, random_spinor, BRIDGE_TESTS_ENV,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;
type H = Field<f64, Spinor, Half>;

const ANTIPERIODIC: [Boundary; ND] = [Boundary::Antiperiodic; ND];
const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];

fn wilson(lat: &Arc<Lattice>, boundary: [Boundary; ND], u: &Field<f64, Gauge>) -> Wilson<f64> {
    let mut op = Wilson::new(
        lat,
        WilsonParams {
            kappa: 0.12,
            boundary,
        },
    );
    op.set_gauge(u).unwrap();
    op
}

fn halves(op: &Wilson<f64>, full: &V) -> (H, H) {
    let idx = op.even_odd_index().unwrap();
    let nv = full.num_vectors();
    let mut e = H::spinor_half(full.lattice(), nv, Parity::Even);
    let mut o = H::spinor_half(full.lattice(), nv, Parity::Odd);
    extract_parity(&mut e, full, idx, Parity::Even);
    extract_parity(&mut o, full, idx, Parity::Odd);
    (e, o)
}

/// Assembling M ψ from the four blocks gives the same bits as the full-lattice sweep.
#[test]
fn blocks_reproduce_the_full_operator() {
    let lat = Lattice::single_node([4, 6, 4, 8]);
    let u = random_gauge(&lat, 61);
    let psi = random_spinor(&lat, 2, 62);
    let one = vec![C64::new(1.0, 0.0); 2];
    for boundary in [ANTIPERIODIC, MIXED] {
        let op = wilson(&lat, boundary, &u);
        let idx = op.even_odd_index().unwrap();
        let (psi_e, psi_o) = halves(&op, &psi);
        let mut w = op.even_odd_work();

        for (dagger, variant) in [(Dagger::No, Variant::D), (Dagger::Yes, Variant::Dagger)] {
            let mut want = psi.zeros_like();
            op.mult(&mut op.work(), &mut want, &psi, variant);

            let (mut out_e, mut out_o) = (psi_e.zeros_like(), psi_o.zeros_like());
            let (mut hop_e, mut hop_o) = (psi_e.zeros_like(), psi_o.zeros_like());
            op.diag(&mut w, &mut out_e, &psi_e, Parity::Even, dagger);
            op.hop(&mut w, &mut hop_e, &psi_o, Parity::Even, dagger);
            out_e.axpy(&one, &hop_e);
            op.diag(&mut w, &mut out_o, &psi_o, Parity::Odd, dagger);
            op.hop(&mut w, &mut hop_o, &psi_e, Parity::Odd, dagger);
            out_o.axpy(&one, &hop_o);

            let mut got = psi.zeros_like();
            insert_parity(&mut got, &out_e, idx);
            insert_parity(&mut got, &out_o, idx);
            assert_eq!(got.as_slice(), want.as_slice(), "{variant:?}");
        }
    }
}

/// <χ, S ψ> = <S† χ, ψ> for the Schur complement on either parity.
#[test]
fn schur_dagger_is_the_adjoint() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson(&lat, MIXED, &random_gauge(&lat, 63));
    let (psi_e, psi_o) = halves(&op, &random_spinor(&lat, 1, 64));
    let (chi_e, chi_o) = halves(&op, &random_spinor(&lat, 1, 65));
    for (parity, psi, chi, other) in [
        (Parity::Even, &psi_e, &chi_e, &psi_o),
        (Parity::Odd, &psi_o, &chi_o, &psi_e),
    ] {
        let mut s = Schur::new(&op, parity, psi, other);
        let (mut s_psi, mut sdag_chi) = (psi.zeros_like(), psi.zeros_like());
        s.mult(&mut s_psi, psi);
        s.mult_dag(&mut sdag_chi, chi);
        let (mut a, mut b) = ([C64::ZERO], [C64::ZERO]);
        chi.dot(&s_psi, &mut a);
        sdag_chi.dot(psi, &mut b);
        assert!(a[0].re.abs() > 1.0);
        assert!(
            (a[0] - b[0]).norm_sqr().sqrt() < 1e-11,
            "{:?} {:?}",
            a[0],
            b[0]
        );
    }
}

/// Solving through the Schur complement gives the solution of the full system, in fewer
/// iterations than solving the full system directly.
#[test]
fn even_odd_solve_solves_the_full_system() {
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson(&lat, ANTIPERIODIC, &random_gauge(&lat, 66));
    let b = random_spinor(&lat, 2, 67);
    let stop = Stop {
        tolerance_sqr: 1e-26,
        max_iterations: 500,
        max_restarts: 2,
    };

    let mut x = b.zeros_like();
    let info = solve_even_odd(
        &op,
        op.even_odd_index().unwrap(),
        &mut x,
        &b,
        |schur, x_e, rhs| solve_cgne(&mut Cg::new(stop), schur, x_e, rhs),
    )
    .unwrap();

    // Residual of the full system, computed with the full operator.
    let mut d = Bound::new(&op, Variant::D);
    let mut r = b.zeros_like();
    d.mult(&mut r, &x);
    r.scale(&[-1.0; 2]);
    r.axpy(&[C64::new(1.0, 0.0); 2], &b);
    let (mut rr, mut bb) = ([0.0; 2], [0.0; 2]);
    r.norm_sqr(&mut rr);
    b.norm_sqr(&mut bb);
    for j in 0..2 {
        assert!(
            rr[j] / bb[j] < 1e-24,
            "vector {j}: residual {:e}",
            rr[j] / bb[j]
        );
    }

    let mut x_full = b.zeros_like();
    let info_full = solve_cgne(&mut Cg::new(stop), &mut d, &mut x_full, &b).unwrap();
    let diff = max_abs_diff(&x, &x_full);
    assert!(
        diff < 1e-11,
        "max difference between the two solutions: {diff}"
    );
    eprintln!(
        "CG steps: {} with even-odd, {} without",
        info.iterations, info_full.iterations
    );
    assert!(info.iterations < info_full.iterations);
}

// ---- Comparison with Bridge++ (skipped unless the test inputs are provided) ----

/// `expected_result` of the Bridge++ test `Mult_Wilson_eo` (tests/test_Mult_Wilson_eo.yaml):
/// the 2-norm of the even-odd preconditioned operator 1 - M_eo M_oe applied once to a
/// half-lattice spinor whose reals are all 1; κ = 0.12, antiperiodic in all directions.
const BRIDGE_MULT_WILSON_EO_NORM: f64 = 8.07908371651806e+01;

#[test]
fn bridge_mult_wilson_eo() {
    let Some(path) = bridge_test_file("conf_04040408.txt") else {
        eprintln!("SKIPPED: set {BRIDGE_TESTS_ENV} to a directory containing conf_04040408.txt");
        return;
    };
    let lat = Lattice::single_node([4, 4, 4, 8]);
    let op = wilson(&lat, ANTIPERIODIC, &read_bridge_text(&path, &lat).unwrap());

    let norm_on = |parity: Parity| {
        let mut b = H::spinor_half(&lat, 1, parity);
        fill(&mut b, 1.0);
        let other = H::spinor_half(&lat, 1, parity.flip());
        let mut y = b.zeros_like();
        Schur::new(&op, parity, &b, &other).mult(&mut y, &b);
        norm_sqr(&y).sqrt()
    };
    let (even, odd) = (norm_on(Parity::Even), norm_on(Parity::Odd));
    eprintln!(
        "|S b| = {even:.14e} on even sites, {odd:.14e} on odd sites; Bridge++ = {BRIDGE_MULT_WILSON_EO_NORM:.14e}"
    );
    let relative = (even - BRIDGE_MULT_WILSON_EO_NORM).abs() / BRIDGE_MULT_WILSON_EO_NORM;
    assert!(
        relative < 1e-13,
        "relative difference on even sites {relative:e}"
    );
}
