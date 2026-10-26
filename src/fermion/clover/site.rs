//! The clover term: the site-diagonal block of the clover operator and its inverse.

use super::*;
use rayon::prelude::*;

/// Rows of the dense site matrix (spin ⊗ colour) and of one chiral block.
const DENSE: usize = 12;
const BLOCK: usize = 6;
/// Reals stored per site for one matrix: the two chiral blocks, γ5 = +1 first.
const SITE_REALS: usize = 2 * HERMITIAN6_REALS;
/// Sites a task of the pool makes the blocks of.
const SITES_PER_TASK: usize = 256;

/// The site-diagonal block of the clover operator and its inverse, each stored as two
/// packed Hermitian 6 × 6 blocks per site.
pub struct CloverDiag<T: Storage> {
    kappa_csw: f64,
    /// γ_mu γ_nu for each plane, in the order of `ruttice::gauge::PLANES`.
    gamma_pairs: [Gamma; 6],
    t: Vec<T>,
    t_inv: Vec<T>,
}

impl<T: Storage> CloverDiag<T> {
    pub(super) fn new<G: GammaRepre>(kappa: f64, csw: f64) -> Self {
        // The chiral split below is written for γ5 = [[0, 1], [1, 0]] in 2 × 2 spin blocks,
        // whose eigenvectors are (e_a ± e_{a+2}) / √2.
        assert!(
            G::GAMMA5.col == [2, 3, 0, 1] && G::GAMMA5.val == [Unit::One; 4],
            "the clover term is implemented for representations with gamma5 = [[0, 1], [1, 0]]"
        );
        Self {
            kappa_csw: kappa * csw,
            gamma_pairs: PLANES.map(|(mu, nu)| G::GAMMA[mu].mul(&G::GAMMA[nu])),
            t: Vec::new(),
            t_inv: Vec::new(),
        }
    }

    /// The blocks of T, `2 * HERMITIAN6_REALS` reals per site of the lattice: the block on
    /// γ5 = +1 and the one on γ5 = −1, each packed (`hermitian6_mul`). Empty before `set_gauge`.
    pub fn blocks(&self) -> &[T] {
        &self.t
    }

    /// Apply the two stored blocks of one site to a spinor.
    #[inline(always)]
    fn apply_blocks(
        matrices: &[T],
        full_site: usize,
        phi: &SiteSpinor<T::Compute>,
    ) -> SiteSpinor<T::Compute> {
        let s = &matrices[full_site * SITE_REALS..(full_site + 1) * SITE_REALS];
        let plus_block: [T::Compute; HERMITIAN6_REALS] = core::array::from_fn(|k| s[k].load());
        let minus_block: [T::Compute; HERMITIAN6_REALS] =
            core::array::from_fn(|k| s[HERMITIAN6_REALS + k].load());

        // Components along (e_a ± e_{a+2}), without the 1/√2.
        let plus: [Cplx<T::Compute>; BLOCK] =
            core::array::from_fn(|k| phi.0[k / NC][k % NC] + phi.0[k / NC + 2][k % NC]);
        let minus: [Cplx<T::Compute>; BLOCK] =
            core::array::from_fn(|k| phi.0[k / NC][k % NC] - phi.0[k / NC + 2][k % NC]);
        let (p, m) = (
            hermitian6_mul(&plus_block, &plus),
            hermitian6_mul(&minus_block, &minus),
        );

        // Back to spin components; the two 1/√2 combine into 1/2.
        let half: T::Compute = Number::from_f64(0.5);
        let mut out = SiteSpinor::zero();
        for k in 0..BLOCK {
            out.0[k / NC][k % NC] = (p[k] + m[k]).scale(half);
            out.0[k / NC + 2][k % NC] = (p[k] - m[k]).scale(half);
        }
        out
    }

    /// The dense 12 × 12 matrix T at one site, row-major (re, im), index `colour + 3 spin`.
    fn dense_block(&self, f: &Field<f64, Gauge>, site: usize) -> [f64; 2 * DENSE * DENSE] {
        let mut t = [0.0f64; 2 * DENSE * DENSE];
        for d in 0..DENSE {
            t[2 * (DENSE * d + d)] = 1.0;
        }
        for (plane, gamma) in self.gamma_pairs.iter().enumerate() {
            let fs = f.site(plane, site);
            // Row (spin a, colour i), column (spin gamma.col[a], colour j):
            // -κ c_SW (γγ)_{a, col[a]} F_{ij}
            for a in 0..4 {
                let b = gamma.col[a];
                for i in 0..NC {
                    for j in 0..NC {
                        let fij = Cplx::new(fs[2 * (NC * i + j)], fs[2 * (NC * i + j) + 1]);
                        let z = gamma.val[a].times(fij).scale(self.kappa_csw);
                        let k = 2 * (DENSE * (i + NC * a) + (j + NC * b));
                        t[k] -= z.re;
                        t[k + 1] -= z.im;
                    }
                }
            }
        }
        t
    }
}

/// Element (r, c) of a dense row-major complex matrix with `n` rows.
fn element(m: &[f64], n: usize, r: usize, c: usize) -> Cplx<f64> {
    Cplx::new(m[2 * (n * r + c)], m[2 * (n * r + c) + 1])
}

/// The block of a dense 12 × 12 matrix between γ5 eigenvectors of signs `left` and `right`
/// (+1 or -1): `<e^left_r| M |e^right_c>`, as a dense 6 × 6 matrix.
fn chiral_block(m: &[f64], left: f64, right: f64) -> [f64; 2 * BLOCK * BLOCK] {
    let mut out = [0.0; 2 * BLOCK * BLOCK];
    for r in 0..BLOCK {
        for c in 0..BLOCK {
            let z = element(m, DENSE, r, c)
                + element(m, DENSE, r, c + BLOCK).scale(right)
                + element(m, DENSE, r + BLOCK, c).scale(left)
                + element(m, DENSE, r + BLOCK, c + BLOCK).scale(left * right);
            out[2 * (BLOCK * r + c)] = 0.5 * z.re;
            out[2 * (BLOCK * r + c) + 1] = 0.5 * z.im;
        }
    }
    out
}

/// Pack a Hermitian 6 × 6 matrix: six real diagonal entries, then the upper triangle row
/// by row as (re, im). Returns the largest deviation from Hermiticity alongside.
fn pack_hermitian(m: &[f64]) -> ([f64; HERMITIAN6_REALS], f64) {
    let mut packed = [0.0; HERMITIAN6_REALS];
    let mut defect = 0.0f64;
    let mut k = BLOCK;
    for r in 0..BLOCK {
        let d = element(m, BLOCK, r, r);
        packed[r] = d.re;
        defect = defect.max(d.im.abs());
        for c in r + 1..BLOCK {
            let (upper, lower) = (element(m, BLOCK, r, c), element(m, BLOCK, c, r));
            defect = defect.max((upper - lower.conj()).norm_sqr().sqrt());
            packed[k] = upper.re;
            packed[k + 1] = upper.im;
            k += 2;
        }
    }
    (packed, defect)
}

impl<T: Storage> DiagTerm<T> for CloverDiag<T> {
    const IS_UNIT: bool = false;
    /// The count Bridge++ uses for its clover term, `Nc Nd (2 + 48 Nc)`, plus the
    /// `2 Nc Nd` additions that combine it with the hopping term: 1776 for three colours.
    const FLOPS_PER_SITE: usize = NC * NS * (2 + 48 * NC) + 2 * NC * NS;

    #[inline(always)]
    fn apply(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        Self::apply_blocks(&self.t, full_site, phi)
    }

    #[inline(always)]
    fn apply_inv(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        Self::apply_blocks(&self.t_inv, full_site, phi)
    }

    fn set_gauge(&mut self, lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        crate::hmc::count::clover_blocks(lat.num_sites());
        // The field strength is built from the links alone; fermion boundary phases do not
        // enter it.
        let f = field_strength(u, &Stencil::nearest(lat, [Boundary::Periodic; ND]));
        let n = lat.num_sites();
        let mut t = vec![T::default(); n * SITE_REALS];
        let mut t_inv = vec![T::default(); n * SITE_REALS];
        let store = |dst: &mut [T], src: &[f64; HERMITIAN6_REALS]| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = T::store(Number::from_f64(*s));
            }
        };

        // The sites one by one, each on its own: by the threads of the pool, so that the
        // pages of the blocks are spread over the memory nodes like those of the fields.
        // The first singular site is the one reported, whatever the threads.
        let singular = t
            .par_chunks_mut(SITE_REALS)
            .zip(t_inv.par_chunks_mut(SITE_REALS))
            .enumerate()
            .with_min_len(SITES_PER_TASK)
            .filter_map(|(site, (t, t_inv))| {
                let dense = self.dense_block(&f, site);
                // T commutes with γ5, so the blocks between opposite chiralities vanish and
                // the diagonal blocks are Hermitian. Both are properties the packed storage
                // relies on; a violation here is a bug in the construction above.
                let mixing = chiral_block(&dense, 1.0, -1.0)
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(
                    mixing < 1e-12,
                    "clover block mixes chiralities at site {site}: {mixing:e}"
                );

                for (which, sign) in [1.0, -1.0].into_iter().enumerate() {
                    let block = chiral_block(&dense, sign, sign);
                    let (packed, defect) = pack_hermitian(&block);
                    assert!(
                        defect < 1e-12,
                        "clover block is not Hermitian at site {site}: {defect:e}"
                    );

                    let mut scratch = block;
                    let mut inverse = [0.0; 2 * BLOCK * BLOCK];
                    if invert_complex(BLOCK, &mut scratch, &mut inverse).is_err() {
                        return Some(site);
                    }
                    // The inverse of a Hermitian matrix is Hermitian up to rounding; pack it as
                    // such.
                    let (packed_inverse, _) = pack_hermitian(&inverse);

                    let at = which * HERMITIAN6_REALS;
                    store(&mut t[at..at + HERMITIAN6_REALS], &packed);
                    store(&mut t_inv[at..at + HERMITIAN6_REALS], &packed_inverse);
                }
                None
            })
            .min();
        if let Some(site) = singular {
            return Err(GaugeError::SingularDiagonal { site });
        }
        self.t = t;
        self.t_inv = t_inv;
        Ok(())
    }
}
