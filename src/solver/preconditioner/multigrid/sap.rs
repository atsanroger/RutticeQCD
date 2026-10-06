use super::Aggregation;
use crate::fermion::{HopTerms, Variant};
use crate::kernel::Real;
use crate::{for_sites, Direction, Field, LinearFn, Spinor, Storage, Vector, C64, ND};
use std::sync::Arc;

/// The eight hopping directions.
const DIRECTIONS: [(usize, Direction); 2 * ND] = [
    (0, Direction::Forward),
    (1, Direction::Forward),
    (2, Direction::Forward),
    (3, Direction::Forward),
    (0, Direction::Backward),
    (1, Direction::Backward),
    (2, Direction::Backward),
    (3, Direction::Backward),
];

/// Schwarz alternating procedure: an approximate inverse of D built from the inverses of D
/// restricted to blocks (Lüscher, Comput. Phys. Commun. 156 (2004) 209).
///
/// The blocks are coloured like a checkerboard by their position in the block lattice. One
/// cycle solves the block systems of one colour against the current residual, updates the
/// solution and the residual, then does the same for the other colour. The block systems
/// are solved approximately by a fixed number of minimal-residual steps; D restricted to a
/// block drops every hop that leaves the block.
///
/// The number of cycles and of block steps are fixed, but the minimal-residual
/// coefficients depend on the residual, so the result is not a linear function of the
/// right-hand side: use it inside flexible Krylov solvers (FGMRES) and as the smoother of a
/// multigrid cycle. Starts from zero; `mult(x, b)` returns the approximation to D^{-1} b.
/// Every vector of a multi-vector is treated independently.
///
/// The parameters follow the smoother of the Bridge++ multigrid solver (`ASolver_SAP`).
pub struct Sap<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> {
    op: &'a F,
    work: F::Work,
    blocks: Blocks,
    cycles: usize,
    block_steps: usize,
    /// Allocated on first use, and again when the number of vectors changes.
    scratch: Option<Scratch<T>>,
}

struct Scratch<T: Storage> {
    /// Residual.
    r: Field<T, Spinor>,
    /// Update from one colour's block solves.
    e: Field<T, Spinor>,
    /// Block operator applied to the block residual.
    p: Field<T, Spinor>,
    /// One hopping term.
    hop: Field<T, Spinor>,
}

/// The blocks and their colours.
struct Blocks {
    agg: Arc<Aggregation>,
    /// Colour (0 or 1) of each block.
    colour: Vec<u8>,
    hop_coefficient: f64,
}

impl<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> Sap<'a, F, T> {
    /// `agg` gives the blocks; `cycles` is the number of SAP cycles, `block_steps` the
    /// number of minimal-residual steps per block solve.
    pub fn new(op: &'a F, agg: &Arc<Aggregation>, cycles: usize, block_steps: usize) -> Self {
        let coarse = agg.coarse_lattice();
        let colour = (0..coarse.num_sites())
            .map(|c| (coarse.global_coords(c).iter().sum::<usize>() % 2) as u8)
            .collect();
        Self {
            op,
            work: op.work(),
            blocks: Blocks {
                agg: agg.clone(),
                colour,
                hop_coefficient: op.hop_coefficient(),
            },
            cycles,
            block_steps,
            scratch: None,
        }
    }
}

impl Blocks {
    #[inline(always)]
    fn colour_of(&self, site: usize) -> u8 {
        self.colour[self.agg.coarse_site(site)]
    }

    /// Set `v` to zero outside the blocks of colour `c`.
    fn keep_colour<T: Storage>(&self, c: u8, v: &mut Field<T, Spinor>) {
        let site_len = v.site_len();
        for_sites(v.slab_mut(0), site_len, |x, o| {
            if self.colour_of(x) != c {
                o.fill(T::default());
            }
        });
    }

    /// `out = D_c inp` on the blocks of colour `c`, where D_c is D without the hops that
    /// leave a block; zero on the other blocks. `hop` is scratch.
    fn mult<F, T>(
        &self,
        op: &F,
        work: &mut F::Work,
        hop: &mut Field<T, Spinor>,
        c: u8,
        out: &mut Field<T, Spinor>,
        inp: &Field<T, Spinor>,
    ) where
        F: HopTerms<Field<T, Spinor>>,
        T: Storage,
    {
        out.set_zero();
        op.mult_diag(work, out, inp);
        let k: T::Compute = Real::from_f64(self.hop_coefficient);
        let site_len = out.site_len();
        for (mu, dir) in DIRECTIONS {
            hop.set_zero();
            match dir {
                Direction::Forward => op.hop_forward(work, mu, hop, inp),
                Direction::Backward => op.hop_backward(work, mu, hop, inp),
            }
            let hop = &*hop;
            for_sites(out.slab_mut(0), site_len, |x, o| {
                if self.colour_of(x) == c && !self.agg.crosses(x, mu, dir) {
                    for (o, h) in o.iter_mut().zip(hop.site(0, x)) {
                        *o = T::store(o.load() + k * h.load());
                    }
                }
            });
        }
        self.keep_colour(c, out);
    }

    /// `e ≈ D_c^{-1} r` on the blocks of colour `c`, by `steps` minimal-residual steps on
    /// every block and vector independently; zero on the other blocks. `r` is left holding
    /// the remaining residual of the block systems.
    #[allow(clippy::too_many_arguments)]
    fn solve<F, T>(
        &self,
        op: &F,
        work: &mut F::Work,
        hop: &mut Field<T, Spinor>,
        c: u8,
        steps: usize,
        e: &mut Field<T, Spinor>,
        r: &mut Field<T, Spinor>,
        p: &mut Field<T, Spinor>,
    ) where
        F: HopTerms<Field<T, Spinor>>,
        T: Storage,
    {
        e.set_zero();
        self.keep_colour(c, r);
        let nv = r.num_vectors();
        let reals = r.site_len() / nv;
        let mut alpha = vec![C64::ZERO; self.colour.len() * nv];
        for _ in 0..steps {
            self.mult(op, work, hop, c, p, r);
            // α = <p, r> / <p, p> per block and vector, which minimises |r - α p|.
            let (p_ref, r_ref) = (&*p, &*r);
            for_sites(&mut alpha, nv, |block, a| {
                a.fill(C64::ZERO);
                if self.colour[block] != c {
                    return;
                }
                for v in 0..nv {
                    let (mut pr, mut pp) = (C64::ZERO, 0.0);
                    for x in self.agg.fine_sites(block) {
                        let ps = &p_ref.site(0, *x as usize)[v * reals..(v + 1) * reals];
                        let rs = &r_ref.site(0, *x as usize)[v * reals..(v + 1) * reals];
                        for (pz, rz) in ps.as_chunks::<2>().0.iter().zip(rs.as_chunks::<2>().0) {
                            let pz = C64::new(pz[0].load().to_f64(), pz[1].load().to_f64());
                            let rz = C64::new(rz[0].load().to_f64(), rz[1].load().to_f64());
                            pr += pz.conj() * rz;
                            pp += pz.norm_sqr();
                        }
                    }
                    if pp > 0.0 {
                        a[v] = pr.scale(1.0 / pp);
                    }
                }
            });
            self.block_axpy(e, &alpha, 1.0, r);
            self.block_axpy(r, &alpha, -1.0, p);
        }
    }

    /// `y += sign * α(block, vector) x`.
    fn block_axpy<T: Storage>(
        &self,
        y: &mut Field<T, Spinor>,
        alpha: &[C64],
        sign: f64,
        x: &Field<T, Spinor>,
    ) {
        let nv = y.num_vectors();
        let site_len = y.site_len();
        let reals = site_len / nv;
        for_sites(y.slab_mut(0), site_len, |site, ys| {
            let a = &alpha[self.agg.coarse_site(site) * nv..][..nv];
            let xs = x.site(0, site);
            for v in 0..nv {
                let (re, im): (T::Compute, T::Compute) = (
                    Real::from_f64(sign * a[v].re),
                    Real::from_f64(sign * a[v].im),
                );
                let range = v * reals..(v + 1) * reals;
                for (yz, xz) in ys[range.clone()]
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(xs[range].as_chunks::<2>().0)
                {
                    let (xr, xi) = (xz[0].load(), xz[1].load());
                    yz[0] = T::store(yz[0].load() + re * xr - im * xi);
                    yz[1] = T::store(yz[1].load() + re * xi + im * xr);
                }
            }
        });
    }
}

impl<F: HopTerms<Field<T, Spinor>>, T: Storage> LinearFn<Field<T, Spinor>> for Sap<'_, F, T> {
    fn mult(&mut self, x: &mut Field<T, Spinor>, b: &Field<T, Spinor>) {
        assert!(x.same_shape(b));
        if !self.scratch.as_ref().is_some_and(|s| s.r.same_shape(b)) {
            self.scratch = Some(Scratch {
                r: b.zeros_like(),
                e: b.zeros_like(),
                p: b.zeros_like(),
                hop: b.zeros_like(),
            });
        }
        let nv = b.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let Sap {
            op,
            work,
            blocks,
            cycles,
            block_steps,
            scratch,
        } = self;
        let Some(Scratch { r, e, p, hop }) = scratch.as_mut() else {
            unreachable!("the scratch fields were just created");
        };
        x.set_zero();
        r.copy_from(b);
        for _ in 0..*cycles {
            for c in [0, 1] {
                blocks.solve(*op, work, hop, c, *block_steps, e, r, p);
                x.axpy(&one, e);
                // r = b - D x
                op.mult(work, r, x, Variant::D);
                r.aypx(&minus_one, b);
            }
        }
    }
}
