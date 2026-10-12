//! The sweep over sites of the usual layout, and what goes with it: the operator in that
//! layout, its single hopping terms and even-odd blocks.

use super::*;

impl<T: Storage, G: GammaRepre> WilsonOp<T, G, UnitDiag> {
    pub fn new(lat: &Arc<Lattice>, params: WilsonParams) -> Self {
        Self::with_diag(lat, params, UnitDiag)
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> WilsonOp<T, G, D> {
    /// An operator of Wilson type with the given site-diagonal block.
    pub fn with_diag(lat: &Arc<Lattice>, params: WilsonParams, diag: D) -> Self {
        let even_odd = lat.local.iter().all(|l| l % 2 == 0).then(|| {
            let index = Arc::new(EvenOddIndex::new(lat));
            let to = [Parity::Even, Parity::Odd]
                .map(|p| Geometry::parity(lat, params.boundary, &index, p));
            EvenOddGeometry { index, to }
        });
        Self {
            params,
            lat: lat.clone(),
            state: SiteState {
                full: Geometry::full(lat, params.boundary),
                even_odd,
                links: LinkStore::Unset,
            },
            link_format: LinkFormat::Full,
            diag,
            _gamma: PhantomData,
        }
    }

    pub fn params(&self) -> &WilsonParams {
        &self.params
    }

    /// The even-odd hop with what goes around it within a site in the same sweep: `out =
    /// after(M_(to, other) before(inp))`, or `after(M†_(to, other) inp)` with `Dagger::Yes`.
    /// `before` is a factor in the fifth dimension on the input (for M only); `after` is
    /// applied to every site of the result (see `After`). One pass over the field.
    #[allow(clippy::too_many_arguments)]
    pub fn even_odd_hop_fifth(
        &self,
        work: &mut WilsonWork<T>,
        out: &mut Field<T, Spinor, Half>,
        inp: &Field<T, Spinor, Half>,
        to: Parity,
        dagger: Dagger,
        before: Option<Fifth>,
        after: After<'_, T>,
    ) {
        let eo = self
            .state
            .even_odd
            .as_ref()
            .expect("Wilson: even-odd blocks need a lattice with even extents");
        assert_eq!(out.parity(), to, "output field has the wrong parity");
        assert_eq!(inp.parity(), to.flip(), "input field has the wrong parity");
        let g5 = dagger == Dagger::Yes;
        let pre = match before {
            Some(fifth) => {
                assert!(!g5, "the factor goes before the kernel of M, not of M†");
                assert_eq!(
                    inp.copies() % fifth.ls,
                    0,
                    "the fields hold ls spinors per vector"
                );
                Pre::Fifth(fifth)
            }
            None => Pre::gamma5(g5),
        };
        let ends = (pre, Post::gamma5(g5));
        self.sweep(
            work,
            &eo.to[to.index()],
            out,
            inp,
            Diagonal::Absent,
            ends,
            after,
        );
    }

    /// The even-odd numbering this operator uses, if the lattice allows one.
    pub fn even_odd_index(&self) -> Option<&Arc<EvenOddIndex>> {
        self.state.even_odd.as_ref().map(|eo| &eo.index)
    }

    /// Choose how the links are kept. The gauge field has to be set again afterwards.
    pub fn with_link_format(mut self, format: LinkFormat) -> Self {
        self.link_format = format;
        self.state.links = LinkStore::Unset;
        self
    }

    pub fn link_format(&self) -> LinkFormat {
        self.link_format
    }

    /// The phase a hop in direction (mu, dir) picks up when it crosses the boundary of the
    /// global lattice. A hop backward picks up the conjugate.
    fn boundary_phase(&self, mu: usize, dir: Direction) -> Option<Cplx<T::Compute>> {
        boundary_phase(self.params.boundary[mu], dir)
    }

    /// The leg in direction (mu, dir) of output site `site`.
    ///
    /// Forward legs use the link at the output site. Backward legs use the link at the
    /// neighbour; when the neighbour is on another process, that process applies it (and
    /// the phase) before sending.
    #[inline(always)]
    fn resolve<'a, R: LinkRepr>(
        geo: &Geometry,
        fixed: &Fixed<'a, T, R>,
        site: usize,
        mu: usize,
        dir: Direction,
    ) -> Hop<'a, T, R> {
        let phase = |wrapped: bool| {
            if wrapped {
                fixed.phase[Stencil::leg(mu, dir)]
            } else {
                None
            }
        };
        match (geo.stencil.source(site, mu, dir), dir) {
            (Source::Zero, _) => Hop::Absent,
            (Source::Local { site: n, wrapped }, _) => Hop::Local {
                site: n,
                link: match dir {
                    Direction::Forward => &fixed.links[mu][geo.out_site(site)],
                    Direction::Backward => &fixed.links[mu][geo.inp_site(n)],
                },
                phase: phase(wrapped),
            },
            (Source::Buffer { index, wrapped }, Direction::Forward) => Hop::Projected {
                index,
                link: &fixed.links[mu][geo.out_site(site)],
                phase: phase(wrapped),
            },
            (Source::Buffer { index, .. }, Direction::Backward) => Hop::Transported { index },
        }
    }

    /// What `resolve` needs and does not change during a sweep.
    fn fixed<R: LinkRepr>(&self) -> Fixed<'_, T, R> {
        let mut phase = [None; FACES];
        for mu in 0..ND {
            for dir in [Direction::Forward, Direction::Backward] {
                phase[Stencil::leg(mu, dir)] = self.boundary_phase(mu, dir);
            }
        }
        let (all, n) = (R::of(&self.state.links), self.lat.num_sites());
        Fixed {
            links: core::array::from_fn(|mu| &all[mu * n..(mu + 1) * n]),
            phase,
        }
    }

    /// The half spinor that arrives along one leg, ready for the sink, for spinor `r` of
    /// the site. `local(site)` reads the input field; `received` is what came in on the
    /// face of this leg.
    #[inline(always)]
    fn arrive<R: LinkRepr>(
        hop: &Hop<'_, T, R>,
        mu: usize,
        dir: Direction,
        input: &Input<'_, T, G>,
        received: &[T],
    ) -> Option<HalfSpinor<T::Compute>> {
        let (sign, link_use) = leg_kind(dir);
        match hop {
            Hop::Absent => None,
            Hop::Local { site, link, phase } => Some(transport(
                &G::GAMMA[mu],
                sign,
                &input.local(*site),
                &R::load::<T>(link),
                link_use,
                *phase,
            )),
            Hop::Projected { index, link, phase } => Some(transport_half(
                &load_half::<T>(received, input.slot(*index)),
                &R::load::<T>(link),
                link_use,
                *phase,
            )),
            Hop::Transported { index } => Some(load_half::<T>(received, input.slot(*index))),
        }
    }

    /// Add the forward and the backward leg of direction `MU` to `hops`.
    ///
    /// The direction is a compile-time constant so that the gamma matrix is one too: its
    /// table of columns and phases then folds into straight-line code. With the direction
    /// as a loop variable the compiler kept the loop and the table lookups (seen in the
    /// generated code as indirect jumps on every component).
    #[inline(always)]
    fn add_legs<const MU: usize, R: LinkRepr>(
        hops: &mut SiteSpinor<T::Compute>,
        forward: &Hop<'_, T, R>,
        backward: &Hop<'_, T, R>,
        input: &Input<'_, T, G>,
        received: &[&[T]; FACES],
    ) {
        let (fwd, bwd) = (Direction::Forward, Direction::Backward);
        let gamma = &G::GAMMA[MU];
        let face = received[Stencil::leg(MU, fwd)];
        if let Some(h) = Self::arrive(forward, MU, fwd, input, face) {
            sink_spinor(hops, &h, gamma, Sign::Minus);
        }
        let face = received[Stencil::leg(MU, bwd)];
        if let Some(h) = Self::arrive(backward, MU, bwd, input, face) {
            sink_spinor(hops, &h, gamma, Sign::Plus);
        }
    }

    /// Send the faces of `inp` that the neighbouring processes need and return what
    /// arrived, by face. Does nothing on a single process.
    fn exchange<'w, R: LinkRepr, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &Geometry,
        inp: &Field<T, Spinor, SI>,
        pre: Pre,
        only: Option<(usize, Direction)>,
    ) -> [&'w [T]; FACES] {
        match self.start_exchange::<R, SI>(work, geo, inp, pre, only) {
            Some(exchange) => finish_exchange(exchange),
            None => [&[]; FACES],
        }
    }

    /// Fill the faces of `inp` that the neighbouring processes need and start sending
    /// them. `None` on a single process.
    ///
    /// Through a forward face goes what the neighbour's backward legs need: this side
    /// holds the link, so the spinor is fully transported here. Through a backward face
    /// goes the projection only; the neighbour holds the link of its forward legs.
    ///
    /// `only` restricts the packing to what the legs in one direction need.
    fn start_exchange<'w, R: LinkRepr, SI: Sites>(
        &self,
        work: &'w mut WilsonWork<T>,
        geo: &Geometry,
        inp: &Field<T, Spinor, SI>,
        pre: Pre,
        only: Option<(usize, Direction)>,
    ) -> Option<&'w mut dyn HaloExchange<T>> {
        if !geo.stencil.has_buffers() {
            return None;
        }
        let copies = inp.copies();
        let exchange = work.exchange(geo.id, copies, &self.lat, || {
            geo.stencil.halo_spec(copies * HALF_SPINOR_REALS)
        });
        for mu in 0..ND {
            for send in [Direction::Forward, Direction::Backward] {
                // What leaves through a face serves the legs pointing the other way.
                let serves = match send {
                    Direction::Forward => Direction::Backward,
                    Direction::Backward => Direction::Forward,
                };
                let sites = geo.stencil.send_sites(mu, send);
                if sites.is_empty() || only.is_some_and(|leg| leg != (mu, serves)) {
                    continue;
                }
                let (sign, link_use) = leg_kind(serves);
                let phase = if geo.stencil.sends_across_boundary(mu, send) {
                    self.boundary_phase(mu, serves)
                } else {
                    None
                };
                let buffer = exchange.send_buffer(Stencil::leg(mu, send));
                let (len, work) = (copies * HALF_SPINOR_REALS, copies * SPINOR_REALS);
                for_sites_as(buffer, len, work, |k, o| {
                    let site = sites[k] as usize;
                    let link = match send {
                        Direction::Forward => Some(self.link::<R>(mu, geo.inp_site(site))),
                        Direction::Backward => None,
                    };
                    let o = o.as_chunks_mut::<HALF_SPINOR_REALS>().0;
                    for (r, o) in o.iter_mut().enumerate() {
                        let psi = Self::input(inp, site, r, pre);
                        let h = match link {
                            Some(link) => {
                                let link = R::load::<T>(link);
                                transport(&G::GAMMA[mu], sign, &psi, &link, link_use, phase)
                            }
                            None => project(&G::GAMMA[mu], sign, &psi),
                        };
                        store_half::<T>(&h, o);
                    }
                });
            }
        }
        exchange.start();
        Some(exchange)
    }

    /// Spinor `r` at one site of the input, transformed by `pre`.
    #[inline(always)]
    fn input<SI: Sites>(
        inp: &Field<T, Spinor, SI>,
        site: usize,
        r: usize,
        pre: Pre,
    ) -> SiteSpinor<T::Compute> {
        let all = inp.site(0, site);
        pre.spinor::<T::Compute, G>(r, |k| {
            load_spinor::<T>(stored_spinor(
                &all[k * SPINOR_REALS..(k + 1) * SPINOR_REALS],
            ))
        })
    }

    /// The link U_mu at a full-lattice site.
    #[inline(always)]
    fn link<R: LinkRepr>(&self, mu: usize, full_site: usize) -> &R::Stored<T> {
        &R::of(&self.state.links)[mu * self.lat.num_sites() + full_site]
    }

    fn check<SO: Sites, SI: Sites>(
        &self,
        geo: &Geometry,
        out: &Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
    ) {
        assert_eq!(
            out.num_sites(),
            geo.stencil.num_sites(),
            "output on the wrong sites"
        );
        assert!(out.outer() == 1 && inp.outer() == 1);
        assert_eq!(
            out.copies(),
            inp.copies(),
            "out and inp hold different numbers of spinors"
        );
    }

    /// `out = B inp` site by site, where B is the diagonal block or its inverse. `full_site`
    /// gives the full-lattice index of each site of the fields.
    fn site_block<S: Sites>(
        &self,
        out: &mut Field<T, Spinor, S>,
        inp: &Field<T, Spinor, S>,
        full_site: impl Fn(usize) -> usize + Sync,
        inverse: bool,
    ) {
        assert_eq!(out.as_slice().len(), inp.as_slice().len());
        let copies = inp.copies();
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let site = full_site(x);
            let o = o.as_chunks_mut::<SPINOR_REALS>().0;
            for (o, s) in o
                .iter_mut()
                .zip(inp.site(0, x).as_chunks::<SPINOR_REALS>().0)
            {
                let phi = load_spinor::<T>(s);
                let result = if inverse {
                    self.diag.apply_inv(site, &phi)
                } else {
                    self.diag.apply(site, &phi)
                };
                store_spinor::<T>(&result, o);
            }
        });
    }

    /// One fused sweep: `out = post( diagonal φ - κ Σ hops φ )` with `φ = pre(inp)`. The
    /// full operator and the even-odd hopping blocks are both this, on different
    /// geometries. The last arguments are (pre, post) and what is done to each site of the
    /// result after them.
    #[allow(clippy::too_many_arguments)]
    fn sweep<SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &Geometry,
        out: &mut Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
        diagonal: Diagonal,
        gamma5: (Pre, Post),
        after: After<'_, T>,
    ) {
        match self.state.links {
            LinkStore::Full(_) => {
                self.sweep_with::<FullLinks, SO, SI>(work, geo, out, inp, diagonal, gamma5, after)
            }
            LinkStore::TwoRows(_) => {
                self.sweep_with::<TwoRowLinks, SO, SI>(work, geo, out, inp, diagonal, gamma5, after)
            }
            LinkStore::Unset => panic!("set_gauge must be called before the operator is applied"),
        }
    }

    /// `sweep` for one representation of the links.
    #[allow(clippy::too_many_arguments)]
    fn sweep_with<R: LinkRepr, SO: Sites, SI: Sites>(
        &self,
        work: &mut WilsonWork<T>,
        geo: &Geometry,
        out: &mut Field<T, Spinor, SO>,
        inp: &Field<T, Spinor, SI>,
        diagonal: Diagonal,
        (pre, post): (Pre, Post),
        after: After<'_, T>,
    ) {
        self.check(geo, out, inp);
        let kappa: T::Compute = Number::from_f64(self.params.kappa);
        let copies = inp.copies();
        let exchange = self.start_exchange::<R, SI>(work, geo, inp, pre, None);
        let fixed = self.fixed::<R>();
        let spinors: &[[T; SPINOR_REALS]] = inp.slab(0).as_chunks().0;
        let (fwd, bwd) = (Direction::Forward, Direction::Backward);
        let (out, site_len) = (out.slab_mut(0), copies * SPINOR_REALS);
        // What is done to a site after the sweep reads it back: it stays in the cache.
        let stream = streams(out, site_len) && after.is_none();
        let fence = || {
            if stream {
                stream_fence();
            }
        };

        overlapped(exchange, geo.parts.as_ref(), |received, part| {
            let site = |x: usize, o: &mut [T]| {
                // Geometry and links are resolved once per site and shared by all copies.
                let forward: [_; ND] =
                    core::array::from_fn(|mu| Self::resolve(geo, &fixed, x, mu, fwd));
                let backward: [_; ND] =
                    core::array::from_fn(|mu| Self::resolve(geo, &fixed, x, mu, bwd));
                let x_full = geo.out_site(x);

                for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                    let input = Input {
                        spinors,
                        r,
                        copies,
                        pre,
                        _gamma: PhantomData::<G>,
                    };

                    let mut hops = SiteSpinor::zero();
                    let (f, b) = (&forward, &backward);
                    Self::add_legs::<0, R>(&mut hops, &f[0], &b[0], &input, received);
                    Self::add_legs::<1, R>(&mut hops, &f[1], &b[1], &input, received);
                    Self::add_legs::<2, R>(&mut hops, &f[2], &b[2], &input, received);
                    Self::add_legs::<3, R>(&mut hops, &f[3], &b[3], &input, received);

                    let diag = match diagonal {
                        Diagonal::Present => self.diag.apply(x_full, &input.local(x)),
                        Diagonal::Absent => SiteSpinor::zero(),
                    };
                    let mut result = diag.sub_scaled(kappa, &hops);
                    if matches!(post, Post::Gamma5) {
                        result = G::GAMMA5.apply(&result);
                    }
                    if stream {
                        let mut stored = [T::default(); SPINOR_REALS];
                        store_spinor::<T>(&result, &mut stored);
                        T::stream(o, &stored);
                    } else {
                        store_spinor::<T>(&result, o);
                    }
                }
                if let Some(after) = after {
                    after(x, o);
                }
            };
            match part {
                None => for_sites_then(out, site_len, site, fence),
                Some(runs) => for_site_runs(out, site_len, runs, site, fence),
            }
        });
    }

    /// Add one hopping term to `out`. Shared by `hop_forward` and `hop_backward`.
    fn hop_term(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        dir: Direction,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        match self.state.links {
            LinkStore::Full(_) => self.hop_term_with::<FullLinks>(work, mu, dir, out, inp),
            LinkStore::TwoRows(_) => self.hop_term_with::<TwoRowLinks>(work, mu, dir, out, inp),
            LinkStore::Unset => panic!("set_gauge must be called before the operator is applied"),
        }
    }

    /// `hop_term` for one representation of the links.
    fn hop_term_with<R: LinkRepr>(
        &self,
        work: &mut WilsonWork<T>,
        mu: usize,
        dir: Direction,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) {
        let geo = &self.state.full;
        self.check(geo, out, inp);
        let (sign, _) = leg_kind(dir);
        let copies = inp.copies();
        let received = self.exchange::<R, crate::Full>(work, geo, inp, Pre::Plain, Some((mu, dir)));
        let fixed = self.fixed::<R>();
        let spinors: &[[T; SPINOR_REALS]] = inp.slab(0).as_chunks().0;
        let face = received[Stencil::leg(mu, dir)];
        for_sites(out.slab_mut(0), copies * SPINOR_REALS, |x, o| {
            let hop = Self::resolve(geo, &fixed, x, mu, dir);
            for (r, o) in o.as_chunks_mut::<SPINOR_REALS>().0.iter_mut().enumerate() {
                let input = Input {
                    spinors,
                    r,
                    copies,
                    pre: Pre::Plain,
                    _gamma: PhantomData::<G>,
                };
                if let Some(h) = Self::arrive(&hop, mu, dir, &input, face) {
                    let mut acc = load_spinor::<T>(o);
                    sink_spinor(&mut acc, &h, &G::GAMMA[mu], sign);
                    store_spinor::<T>(&acc, o);
                }
            }
        });
    }
}

impl<T: Storage, G: GammaRepre, D: DiagTerm<T>> FermionOp<Field<T, Spinor>> for WilsonOp<T, G, D> {
    type Work = WilsonWork<T>;

    /// The hopping term and the unit diagonal are counted as Bridge++ counts its Wilson
    /// operator in the Dirac representation: `Nc Nd (4 + 6 (4 Nc + 2) + 2 (4 Nc + 1))`,
    /// 1368 for three colours. Six of the eight legs have an off-diagonal gamma matrix
    /// and two a diagonal one, which needs no additions to project.
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
