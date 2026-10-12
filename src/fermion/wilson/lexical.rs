//! The operator on the whole lattice: D, D† and γ5 D, each one sweep of the hopping term
//! with the site-diagonal block, in the usual layout and in blocks of W sites; setting the
//! gauge field; and in the usual layout the hopping terms one direction at a time.

use super::sweep_blocks::BlockField;
use super::*;

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> FermionOp<Field<T, Spinor>> for WilsonOp<T, G, D> {
    type Work = WilsonWork<T>;

    /// The hopping term and the unit diagonal are counted as Bridge++ counts its Wilson operator in
    /// the Dirac representation: `Nc Nd (4 + 6 (4 Nc + 2) + 2 (4 Nc + 1))`, 1368 for three colours.
    fn flops_per_site(&self) -> usize {
        NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1)) + D::FLOPS_PER_SITE
    }

    fn work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        // Every link in the operator's precision, `[mu][site]` as in the gauge field.
        let convert = |x: &f64| T::store(Number::from_f64(*x));
        let stored: &[[f64; FULL_LINK_REALS]] = u.as_slice().as_chunks().0;
        let links = match self.link_format {
            LinkFormat::Full => LinkStore::Full(
                stored
                    .par_iter()
                    .with_min_len(LINKS_PER_TASK)
                    .map(|link| core::array::from_fn(|k| convert(&link[k])))
                    .collect(),
            ),
            LinkFormat::TwoRows => {
                let n = self.lat.num_sites();
                for (i, link) in stored.iter().enumerate() {
                    let deviation = su3_deviation(&ColorMatrix::from_reals(link));
                    if deviation.is_nan() || deviation > SU3_TOLERANCE {
                        return Err(GaugeError::NotSpecialUnitary {
                            mu: i / n,
                            site: i % n,
                            deviation,
                        });
                    }
                }
                LinkStore::TwoRows(
                    stored
                        .par_iter()
                        .with_min_len(LINKS_PER_TASK)
                        .map(|link| core::array::from_fn(|k| convert(&link[k])))
                        .collect(),
                )
            }
        };
        self.diag.set_gauge(&self.lat, u)?;
        self.state.links = links;
        Ok(())
    }

    fn mult(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
        variant: Variant,
    ) {
        // γ5 applied to the input (pre) and to the result (post) of the same sweep.
        let (pre, post) = match variant {
            Variant::D => (Pre::Plain, Post::Plain),
            Variant::Gamma5D => (Pre::Plain, Post::Gamma5),
            Variant::Dagger => (Pre::Gamma5, Post::Gamma5),
        };
        self.sweep(
            work,
            &self.state.full,
            out,
            inp,
            Diagonal::Present,
            (pre, post),
            None,
        );
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> HopTerms<Field<T, Spinor>> for WilsonOp<T, G, D> {
    fn hop_forward(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop_term(work, mu, Direction::Forward, out, inp);
    }

    fn hop_backward(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.hop_term(work, mu, Direction::Backward, out, inp);
    }

    fn mult_diag(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(&self.state.full, out, inp);
        let copies = inp.copies();
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            let term = self
                .diag
                .apply(i / copies, &load_spinor::<T>(stored_spinor(s)));
            let sum = load_spinor::<T>(stored_spinor(o)).add(&term);
            store_spinor::<T>(&sum, o);
        });
    }

    fn mult_gamma5(
        &self,
        _work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        self.check(&self.state.full, out, inp);
        for_sites(out.slab_mut(0), SPINOR_REALS, |i, o| {
            let s = &inp.as_slice()[i * SPINOR_REALS..(i + 1) * SPINOR_REALS];
            store_spinor::<T>(&G::GAMMA5.apply(&load_spinor::<T>(stored_spinor(s))), o);
        });
    }

    fn hop_coefficient(&self) -> f64 {
        -self.params.kappa
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>, const W: usize> FermionOp<BlockField<T, Lexical, W>>
    for WilsonOp<T, G, D, Aosoa<W>>
where
    T::Compute: LaneOf<W>,
{
    type Work = WilsonWork<T>;

    fn work(&self) -> WilsonWork<T> {
        WilsonWork::new()
    }

    fn flops_per_site(&self) -> usize {
        NC * NS * (4 + 6 * (4 * NC + 2) + 2 * (4 * NC + 1)) + D::FLOPS_PER_SITE
    }

    fn set_gauge(&mut self, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        if u.num_sites() != self.lat.num_sites() || u.lattice().local != self.lat.local {
            return Err(GaugeError::LatticeMismatch);
        }
        let blocks = self.state.full.stencil.blocks();
        let num_blocks = blocks.num_blocks();
        // Filled by the threads of the pool, like the fields (`Cpu::alloc`).
        let links = (0..ND * num_blocks)
            .into_par_iter()
            .with_min_len(LINKS_PER_TASK)
            .map(|i| {
                let (mu, block) = (i / num_blocks, i % num_blocks);
                let mut link = [[T::default(); W]; FULL_LINK_REALS];
                for lane in 0..W {
                    let from = u.site(mu, blocks.site(block, lane));
                    for c in 0..FULL_LINK_REALS {
                        link[c][lane] = T::store(Number::from_f64(from[c]));
                    }
                }
                link
            })
            .collect();
        self.diag.set_gauge(&self.lat, u)?;
        self.state.links = links;
        Ok(())
    }

    fn mult(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut BlockField<T, Lexical, W>,
        inp: &BlockField<T, Lexical, W>,
        variant: Variant,
    ) {
        let gamma5 = match variant {
            Variant::D => (Pre::Plain, Post::Plain),
            Variant::Gamma5D => (Pre::Plain, Post::Gamma5),
            Variant::Dagger => (Pre::Gamma5, Post::Gamma5),
        };
        self.block_sweep(
            work,
            &self.state.full,
            out,
            inp,
            Diagonal::Present,
            gamma5,
            None,
        );
    }
}
