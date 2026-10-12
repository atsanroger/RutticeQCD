//! The even-odd blocks of the operator, in the usual layout and in blocks of W sites: the
//! hops between the sites of the two parities (a sweep of the hopping term without the
//! diagonal block) and the site-diagonal block and its inverse on one parity.
//! `fermion::even_odd` makes the Schur complement and the solves of them.

use super::sweep_blocks::BlockField;
use super::*;

/// With M = [[T_ee, M_eo], [M_oe, T_oo]], T the site-diagonal block and
/// M_eo = -κ (hops from odd to even sites). M† = γ5 M γ5 and γ5 is site-diagonal, so the
/// blocks of M† are γ5 (block of M) γ5; T is Hermitian and commutes with γ5.
impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> EvenOdd<Field<T, Spinor, Half>>
    for WilsonOp<T, G, D>
{
    type Work = WilsonWork<T>;

    fn even_odd_work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn hop(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
    ) {
        let eo = self
            .state
            .even_odd
            .as_ref()
            .expect("Wilson: even-odd blocks need a lattice with even extents");
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        let g5 = dagger == Dagger::Yes;
        self.sweep(
            work,
            &eo.to[to.index()],
            out,
            inp,
            Diagonal::Absent,
            (Pre::gamma5(g5), Post::gamma5(g5)),
            None,
        );
    }

    fn diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let eo = self
                .state
                .even_odd
                .as_ref()
                .expect("Wilson: even-odd needs even extents");
            let geo = &eo.to[parity.index()];
            self.site_block(out, inp, |h| geo.out_site(h), false);
        }
    }

    fn diag_inv(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let eo = self
                .state
                .even_odd
                .as_ref()
                .expect("Wilson: even-odd needs even extents");
            let geo = &eo.to[parity.index()];
            self.site_block(out, inp, |h| geo.out_site(h), true);
        }
    }
}

/// The even-odd blocks for fields in blocks of sites; see the implementation for the usual
/// layout above. A half field holds the blocks of one parity of the lattice of blocks.
impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, const W: usize> EvenOdd<BlockField<T, Half, W>>
    for WilsonOp<T, G, D, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    type Work = WilsonWork<T>;

    fn even_odd_work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn hop(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        to: Parity,
        dagger: Dagger,
    ) {
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        let g5 = dagger == Dagger::Yes;
        self.block_sweep(
            work,
            self.parity_geometry(to),
            out,
            inp,
            Diagonal::Absent,
            (Pre::gamma5(g5), Post::gamma5(g5)),
            None,
        );
    }

    fn diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let geo = self.parity_geometry(parity);
            self.block_diag(out, inp, |h| geo.out_block(h), false);
        }
    }

    fn diag_inv(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Half, W>,
        inp: &BlockField<T, Half, W>,
        parity: Parity,
        _dagger: Dagger,
    ) {
        assert_eq!(inp.parity(), parity);
        assert_eq!(out.parity(), parity);
        if D::IS_UNIT {
            out.copy_from(inp);
        } else {
            let geo = self.parity_geometry(parity);
            self.block_diag(out, inp, |h| geo.out_block(h), true);
        }
    }
}
