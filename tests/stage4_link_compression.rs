//! Checks: operators that keep two rows of each link (`LinkFormat::TwoRows`) agree with
//! the ones that keep the whole link.

mod common;

use common::max_abs_diff;
use ruttice::fermion::{
    Clover, CloverParams, Dagger, EvenOdd, FermionOp, GaugeError, HopTerms, LinkFormat, Variant,
    Wilson, WilsonParams,
};
use ruttice::gauge::{link, random_gauge, set_link};
use ruttice::{
    assemble, extract_parity, insert_parity, local_part, random_spinor, Boundary, ConvertFrom,
    Field, Gauge, Half, Lattice, LocalComm, Parity, Spinor, Storage, Vector, ND,
};
use std::sync::Arc;

type V = Field<f64, Spinor>;

const MIXED: [Boundary; ND] = [
    Boundary::Phase(0.6, 0.8),
    Boundary::Periodic,
    Boundary::Dirichlet,
    Boundary::Antiperiodic,
];
const DIMS: [usize; ND] = [4, 4, 4, 8];

/// A gauge field whose links are products of two random SU(3) matrices. The links that
/// `random_gauge` makes have a third row that is exactly the cross product of the other
/// two, so rebuilding it would give back the same bits and test nothing.
fn products(lat: &Arc<Lattice>, seed: u64) -> Field<f64, Gauge> {
    let (mut u, v) = (random_gauge(lat, seed), random_gauge(lat, seed + 5000));
    for mu in 0..ND {
        for site in 0..lat.num_sites() {
            let product = link(&u, mu, site).mul(&link(&v, mu, site));
            set_link(&mut u, mu, site, &product);
        }
    }
    u
}

fn wilson<T: Storage>(lat: &Arc<Lattice>, u: &Field<f64, Gauge>, format: LinkFormat) -> Wilson<T> {
    let params = WilsonParams {
        kappa: 0.12,
        boundary: MIXED,
    };
    let mut op = Wilson::new(lat, params).with_link_format(format);
    assert_eq!(op.link_format(), format);
    op.set_gauge(u).unwrap();
    op
}

/// The rebuilt third row differs from the stored one by rounding only.
#[test]
fn wilson_agrees_with_full_links() {
    let lat = Lattice::single_node(DIMS);
    let u = products(&lat, 801);
    let psi = random_spinor(&lat, 2, 802);
    let (full, two) = (
        wilson::<f64>(&lat, &u, LinkFormat::Full),
        wilson::<f64>(&lat, &u, LinkFormat::TwoRows),
    );
    for variant in [Variant::D, Variant::Dagger, Variant::Gamma5D] {
        let (mut a, mut b) = (psi.zeros_like(), psi.zeros_like());
        full.mult(&mut full.work(), &mut a, &psi, variant);
        two.mult(&mut two.work(), &mut b, &psi, variant);
        let diff = max_abs_diff(&a, &b);
        assert!(diff < 1e-13, "{variant:?}: {diff:e}");
        assert!(
            diff > 0.0,
            "{variant:?}: the two formats took the same path"
        );
    }
}

#[test]
fn single_precision_agrees_with_full_links() {
    let lat = Lattice::single_node(DIMS);
    let u = products(&lat, 811);
    let mut psi = Field::<f32, Spinor>::spinor(&lat, 1);
    psi.convert_from(&random_spinor(&lat, 1, 812));
    let apply = |format| {
        let op = wilson::<f32>(&lat, &u, format);
        let mut out = psi.zeros_like();
        op.mult(&mut op.work(), &mut out, &psi, Variant::D);
        let mut wide = V::spinor(&lat, 1);
        wide.convert_from(&out);
        wide
    };
    let diff = max_abs_diff(&apply(LinkFormat::Full), &apply(LinkFormat::TwoRows));
    assert!(diff < 1e-5, "{diff:e}");
}

/// The hopping terms one direction at a time and the even-odd blocks use the same links.
#[test]
fn hopping_terms_and_even_odd_blocks_agree() {
    let lat = Lattice::single_node(DIMS);
    let u = products(&lat, 821);
    let psi = random_spinor(&lat, 2, 822);
    let (full, two) = (
        wilson::<f64>(&lat, &u, LinkFormat::Full),
        wilson::<f64>(&lat, &u, LinkFormat::TwoRows),
    );
    for mu in 0..ND {
        let hops = |op: &Wilson<f64>| {
            let (mut f, mut b) = (psi.zeros_like(), psi.zeros_like());
            op.hop_forward(&mut op.work(), mu, &mut f, &psi);
            op.hop_backward(&mut op.work(), mu, &mut b, &psi);
            (f, b)
        };
        let ((f1, b1), (f2, b2)) = (hops(&full), hops(&two));
        assert!(max_abs_diff(&f1, &f2) < 1e-13 && max_abs_diff(&b1, &b2) < 1e-13);
    }
    let block = |op: &Wilson<f64>, to: Parity| {
        let index = op.even_odd_index().unwrap();
        let mut inp = Field::<f64, Spinor, Half>::spinor_half(&lat, 2, to.flip());
        let mut out = Field::<f64, Spinor, Half>::spinor_half(&lat, 2, to);
        extract_parity(&mut inp, &psi, index, to.flip());
        op.hop(&mut op.even_odd_work(), &mut out, &inp, to, Dagger::Yes);
        let mut whole = psi.zeros_like();
        insert_parity(&mut whole, &out, index);
        whole
    };
    for to in [Parity::Even, Parity::Odd] {
        assert!(max_abs_diff(&block(&full, to), &block(&two, to)) < 1e-13);
    }
}

#[test]
fn clover_agrees_with_full_links() {
    let lat = Lattice::single_node(DIMS);
    let u = products(&lat, 831);
    let psi = random_spinor(&lat, 1, 832);
    let apply = |format| {
        let params = CloverParams {
            kappa: 0.12,
            csw: 1.3,
            boundary: [Boundary::Antiperiodic; ND],
        };
        let mut op = Clover::<f64>::new(&lat, params).with_link_format(format);
        op.set_gauge(&u).unwrap();
        let mut out = psi.zeros_like();
        op.mult(&mut op.work(), &mut out, &psi, Variant::D);
        out
    };
    let diff = max_abs_diff(&apply(LinkFormat::Full), &apply(LinkFormat::TwoRows));
    assert!(diff < 1e-13, "{diff:e}");
}

/// On several processes the faces are packed with the same links: bit for bit the result
/// of one process.
#[test]
fn several_processes_give_the_same_bits() {
    let lat = Lattice::single_node(DIMS);
    let u = products(&lat, 841);
    let psi = random_spinor(&lat, 2, 842);
    let op = wilson::<f64>(&lat, &u, LinkFormat::TwoRows);
    let mut want = psi.zeros_like();
    op.mult(&mut op.work(), &mut want, &psi, Variant::Dagger);
    for grid in [[2, 1, 1, 2], [2, 2, 2, 2]] {
        let parts = LocalComm::run(DIMS, grid, |lat| {
            let op = wilson::<f64>(&lat, &local_part(&u, &lat), LinkFormat::TwoRows);
            let inp = local_part(&psi, &lat);
            let mut out = inp.zeros_like();
            op.mult(&mut op.work(), &mut out, &inp, Variant::Dagger);
            out
        });
        let got = assemble(&lat, &parts);
        assert!(got.as_slice() == want.as_slice(), "grid {grid:?}");
    }
}

/// Two rows are only enough for links in SU(3). Anything else is refused when the gauge
/// field is set, with the place of the first bad link; the full format takes it.
#[test]
fn links_outside_su3_are_refused() {
    let lat = Lattice::single_node(DIMS);
    let mut u = products(&lat, 851);
    let (mu, site) = (2, lat.site([1, 3, 0, 5]));
    let stretched = link(&u, mu, site).scale(1.0 + 1e-6);
    set_link(&mut u, mu, site, &stretched);

    let params = WilsonParams {
        kappa: 0.12,
        boundary: MIXED,
    };
    let mut two = Wilson::<f64>::new(&lat, params).with_link_format(LinkFormat::TwoRows);
    match two.set_gauge(&u) {
        Err(GaugeError::NotSpecialUnitary {
            mu: m,
            site: s,
            deviation,
        }) => {
            assert_eq!((m, s), (mu, site));
            assert!(deviation > 1e-6 && deviation < 1e-5, "{deviation:e}");
        }
        other => panic!("expected a refusal, got {:?}", other.err()),
    }
    let mut full = Wilson::<f64>::new(&lat, params);
    assert!(full.set_gauge(&u).is_ok());
}
