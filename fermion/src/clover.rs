//! Clover (Sheikholeslami-Wohlert) fermions.
//!
//! D = D_Wilson - κ c_SW Σ_{mu<nu} F_mu_nu ⊗ γ_mu γ_nu
//!
//! where F_mu_nu = (Q - Q^†) / 8 is the clover-leaf field strength (`ruttice_gauge`). The
//! extra term acts within a site, so the operator is the Wilson sweep with a non-trivial
//! site-diagonal block T(x) = 1 - κ c_SW Σ F ⊗ γγ, a Hermitian 12 × 12 matrix.
//!
//! Provides: everything `Wilson` provides; T and its inverse for the even-odd blocks.
//! Solvers: as for Wilson. Tools needed: field strength (`ruttice_gauge`), small dense
//! inversion (`ruttice_numerics`).
//!
//! The normalisation and sign conventions follow Bridge++ so that results can be compared.
//! Written from the formula above.

use crate::wilson::{DiagTerm, WilsonOp, WilsonParams};
use crate::GaugeError;
use ruttice_core::{Boundary, Field, Gauge, Lattice, Stencil, Storage, ND};
use ruttice_gauge::{field_strength, PLANES};
use ruttice_kernel::{
    site_matrix_mul, Dirac, Gamma, GammaRepre, Real, Spinor as SiteSpinor, NC, SITE_MATRIX_DIM,
    SITE_MATRIX_REALS,
};
use ruttice_numerics::invert_complex;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct CloverParams {
    /// Hopping parameter κ.
    pub kappa: f64,
    /// Clover coefficient c_SW.
    pub csw: f64,
    pub boundary: [Boundary; ND],
}

/// The site-diagonal block of the clover operator and its inverse, one dense 12 × 12
/// complex matrix per site each.
pub struct CloverDiag<T: Storage> {
    kappa_csw: f64,
    /// γ_mu γ_nu for each plane, in the order of `ruttice_gauge::PLANES`.
    gamma_pairs: [Gamma; 6],
    t: Vec<T>,
    t_inv: Vec<T>,
}

impl<T: Storage> CloverDiag<T> {
    fn new<G: GammaRepre>(kappa: f64, csw: f64) -> Self {
        Self {
            kappa_csw: kappa * csw,
            gamma_pairs: PLANES.map(|(mu, nu)| G::GAMMA[mu].mul(&G::GAMMA[nu])),
            t: Vec::new(),
            t_inv: Vec::new(),
        }
    }

    #[inline(always)]
    fn block(matrices: &[T], full_site: usize) -> [T::Compute; SITE_MATRIX_REALS] {
        let s = &matrices[full_site * SITE_MATRIX_REALS..(full_site + 1) * SITE_MATRIX_REALS];
        core::array::from_fn(|k| s[k].load())
    }
}

impl<T: Storage> DiagTerm<T> for CloverDiag<T> {
    const IS_UNIT: bool = false;

    #[inline(always)]
    fn apply(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        site_matrix_mul(&Self::block(&self.t, full_site), phi)
    }

    #[inline(always)]
    fn apply_inv(&self, full_site: usize, phi: &SiteSpinor<T::Compute>) -> SiteSpinor<T::Compute> {
        site_matrix_mul(&Self::block(&self.t_inv, full_site), phi)
    }

    fn set_gauge(&mut self, lat: &Arc<Lattice>, u: &Field<f64, Gauge>) -> Result<(), GaugeError> {
        // The field strength is built from the links alone; fermion boundary phases do not
        // enter it.
        let f = field_strength(u, &Stencil::nearest(lat, [Boundary::Periodic; ND]));
        let n = lat.num_sites();
        let dim = SITE_MATRIX_DIM;
        let mut t = vec![T::default(); n * SITE_MATRIX_REALS];
        let mut t_inv = vec![T::default(); n * SITE_MATRIX_REALS];
        let mut block = [0.0f64; SITE_MATRIX_REALS];
        let mut scratch = [0.0f64; SITE_MATRIX_REALS];
        let mut inverse = [0.0f64; SITE_MATRIX_REALS];

        for site in 0..n {
            block.fill(0.0);
            for d in 0..dim {
                block[2 * (dim * d + d)] = 1.0;
            }
            for (plane, gamma) in self.gamma_pairs.iter().enumerate() {
                let fs = f.site(plane, site);
                // Row (spin a, colour i), column (spin gamma.col[a], colour j):
                // -κ c_SW (γγ)_{a, col[a]} F_{ij}
                for a in 0..4 {
                    let b = gamma.col[a];
                    for i in 0..NC {
                        for j in 0..NC {
                            let fij = ruttice_kernel::Cplx::new(
                                fs[2 * (NC * i + j)],
                                fs[2 * (NC * i + j) + 1],
                            );
                            let z = gamma.val[a].times(fij).scale(self.kappa_csw);
                            let k = 2 * (dim * (i + NC * a) + (j + NC * b));
                            block[k] -= z.re;
                            block[k + 1] -= z.im;
                        }
                    }
                }
            }
            scratch.copy_from_slice(&block);
            invert_complex(dim, &mut scratch, &mut inverse)
                .map_err(|_| GaugeError::SingularDiagonal { site })?;

            let range = site * SITE_MATRIX_REALS..(site + 1) * SITE_MATRIX_REALS;
            for (dst, src) in t[range.clone()].iter_mut().zip(block) {
                *dst = T::store(Real::from_f64(src));
            }
            for (dst, src) in t_inv[range].iter_mut().zip(inverse) {
                *dst = T::store(Real::from_f64(src));
            }
        }
        self.t = t;
        self.t_inv = t_inv;
        Ok(())
    }
}

/// The clover operator: the Wilson sweep with the clover term as its site-diagonal block.
pub type Clover<T, G = Dirac> = WilsonOp<T, G, CloverDiag<T>>;

impl<T: Storage, G: GammaRepre> WilsonOp<T, G, CloverDiag<T>> {
    pub fn new(lat: &Arc<Lattice>, params: CloverParams) -> Self {
        WilsonOp::with_diag(
            lat,
            WilsonParams {
                kappa: params.kappa,
                boundary: params.boundary,
            },
            CloverDiag::new::<G>(params.kappa, params.csw),
        )
    }
}
