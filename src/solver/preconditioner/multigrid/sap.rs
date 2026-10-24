use super::Aggregation;
use crate::fermion::{HopTerms, Variant};
use crate::kernel::{Number, Real};
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
/// restricted to blocks (Lüscher, Comput. Phys. Commun. 156 (2004) 209), written once for any
/// backend: `blocks` does what touches the fields (`SapBlocks`). The blocks are the aggregates,
/// coloured by the parity of their block coordinates; a cycle solves the blocks of one colour
/// by `block_steps` minimal-residual steps each, adds the update and recomputes the residual
/// with the whole operator, then the same for the other colour.
pub struct Sap<B, V> {
    blocks: B,
    cycles: usize,
    block_steps: usize,
    /// Residual, the update of one colour's block solves, and the block operator applied to
    /// the block residual; made on first use, and again when the number of vectors changes.
    scratch: Option<[V; 3]>,
}

/// What `Sap` needs of a backend.
pub trait SapBlocks<V> {
    /// `out = D_c inp` on the blocks of colour `c`, D_c being D without the hops that leave a
    /// block; zero on the other blocks.
    fn block_mult(&mut self, c: u8, out: &mut V, inp: &V);
    /// `v = 0` outside the blocks of colour `c`.
    fn keep_colour(&mut self, c: u8, v: &mut V);
    /// One minimal-residual step on every block of colour `c` and every vector:
    /// α = <p, r> / <p, p> (which minimises |r − α p|), `e += α r`, `r −= α p`.
    fn step(&mut self, c: u8, e: &mut V, r: &mut V, p: &V);
    /// `out = D inp`.
    fn mult(&mut self, out: &mut V, inp: &V);
}

impl<B, V> Sap<B, V> {
    /// `cycles` SAP cycles of `block_steps` minimal-residual steps per block solve.
    pub fn with_blocks(blocks: B, cycles: usize, block_steps: usize) -> Self {
        Self {
            blocks,
            cycles,
            block_steps,
            scratch: None,
        }
    }
}

impl<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> Sap<CpuBlocks<'a, F, T>, Field<T, Spinor>> {
    /// SAP on the CPU: `agg` gives the blocks; `cycles` is the number of SAP cycles,
    /// `block_steps` the number of minimal-residual steps per block solve.
    pub fn new(op: &'a F, agg: &Arc<Aggregation>, cycles: usize, block_steps: usize) -> Self {
        let coarse = agg.coarse_lattice();
        let colour = (0..coarse.num_sites())
            .map(|c| (coarse.global_coords(c).iter().sum::<usize>() % 2) as u8)
            .collect();
        let blocks = CpuBlocks {
            op,
            work: op.work(),
            blocks: Blocks {
                agg: agg.clone(),
                colour,
                hop_coefficient: op.hop_coefficient(),
            },
            hop: None,
        };
        Self::with_blocks(blocks, cycles, block_steps)
    }
}

/// The parts of SAP on the CPU: the operator, its blocks, and a field for one hopping term.
pub struct CpuBlocks<'a, F: HopTerms<Field<T, Spinor>>, T: Storage> {
    op: &'a F,
    work: F::Work,
    blocks: Blocks,
    hop: Option<Field<T, Spinor>>,
}

/// The blocks and their colours.
struct Blocks {
    agg: Arc<Aggregation>,
    /// Colour (0 or 1) of each block.
    colour: Vec<u8>,
    hop_coefficient: f64,
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
        let k: T::Compute = Number::from_f64(self.hop_coefficient);
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

    /// One minimal-residual step on the blocks of colour `c`, every vector independently:
    /// α = <p, r> / <p, p>, `e += α r`, `r −= α p`.
    fn step<T: Storage>(
        &self,
        c: u8,
        e: &mut Field<T, Spinor>,
        r: &mut Field<T, Spinor>,
        p: &Field<T, Spinor>,
    ) {
        let nv = r.num_vectors();
        let reals = r.site_len() / nv;
        let mut alpha = vec![C64::ZERO; self.colour.len() * nv];
        // α = <p, r> / <p, p> per block and vector, which minimises |r - α p|.
        let r_ref = &*r;
        for_sites(&mut alpha, nv, |block, a| {
            a.fill(C64::ZERO);
            if self.colour[block] != c {
                return;
            }
            for v in 0..nv {
                let (mut pr, mut pp) = (C64::ZERO, 0.0);
                for x in self.agg.fine_sites(block) {
                    let ps = &p.site(0, *x as usize)[v * reals..(v + 1) * reals];
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
                    Number::from_f64(sign * a[v].re),
                    Number::from_f64(sign * a[v].im),
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

impl<F: HopTerms<Field<T, Spinor>>, T: Storage> SapBlocks<Field<T, Spinor>>
    for CpuBlocks<'_, F, T>
{
    fn block_mult(&mut self, c: u8, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        if !self.hop.as_ref().is_some_and(|h| h.same_shape(inp)) {
            self.hop = Some(inp.zeros_like());
        }
        let hop = self.hop.as_mut().expect("the field was just made");
        self.blocks.mult(self.op, &mut self.work, hop, c, out, inp);
    }
    fn keep_colour(&mut self, c: u8, v: &mut Field<T, Spinor>) {
        self.blocks.keep_colour(c, v);
    }
    fn step(
        &mut self,
        c: u8,
        e: &mut Field<T, Spinor>,
        r: &mut Field<T, Spinor>,
        p: &Field<T, Spinor>,
    ) {
        self.blocks.step(c, e, r, p);
    }
    fn mult(&mut self, out: &mut Field<T, Spinor>, inp: &Field<T, Spinor>) {
        self.op.mult(&mut self.work, out, inp, Variant::D);
    }
}

impl<B: SapBlocks<V>, V: Vector> LinearFn<V> for Sap<B, V> {
    fn mult(&mut self, x: &mut V, b: &V) {
        assert!(x.same_shape(b));
        if !self.scratch.as_ref().is_some_and(|s| s[0].same_shape(b)) {
            self.scratch = Some([b.zeros_like(), b.zeros_like(), b.zeros_like()]);
        }
        let nv = b.num_vectors();
        let one = vec![C64::new(1.0, 0.0); nv];
        let minus_one = vec![C64::new(-1.0, 0.0); nv];
        let Some([r, e, p]) = self.scratch.as_mut() else {
            unreachable!("the scratch fields were just made");
        };
        x.set_zero();
        r.copy_from(b);
        for _ in 0..self.cycles {
            for c in [0, 1] {
                // e ≈ D_c⁻¹ r on the blocks of colour c; r left holding the remaining residual
                // of the block systems.
                e.set_zero();
                self.blocks.keep_colour(c, r);
                for _ in 0..self.block_steps {
                    self.blocks.block_mult(c, p, r);
                    self.blocks.step(c, e, r, p);
                }
                x.axpy(&one, e);
                // r = b - D x
                self.blocks.mult(r, x);
                r.aypx(&minus_one, b);
            }
        }
    }
}
