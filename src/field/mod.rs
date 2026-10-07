pub(crate) mod blas;
pub(crate) mod linear;
pub(crate) mod random;
pub(crate) mod storage;
pub(crate) mod vector;
use crate::field::storage::Storage;
use crate::lattice::evenodd::Parity;
use crate::lattice::Lattice;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

/// What lives on a site.
pub trait Kind: 'static {
    const NAME: &'static str;
    /// Internal degrees of freedom per site, in reals. `None` means it is set when the
    /// field is created (coarse grids).
    const INTERNAL_DOF: Option<usize>;
}

pub struct Gauge;
impl Kind for Gauge {
    const NAME: &'static str = "gauge";
    const INTERNAL_DOF: Option<usize> = Some(18);
}

pub struct Spinor;
impl Kind for Spinor {
    const NAME: &'static str = "spinor";
    const INTERNAL_DOF: Option<usize> = Some(24);
}

pub struct Coarse;
impl Kind for Coarse {
    const NAME: &'static str = "coarse";
    const INTERNAL_DOF: Option<usize> = None;
}

/// Which sites a field covers.
pub trait Sites: 'static {
    fn num_sites(lat: &Lattice) -> usize;
}

pub struct Full;
impl Sites for Full {
    fn num_sites(lat: &Lattice) -> usize {
        lat.num_sites()
    }
}

/// One parity of an even-odd decomposition. Declared so types can name it; not usable yet.
pub struct Half;
impl Sites for Half {
    fn num_sites(lat: &Lattice) -> usize {
        lat.num_sites() / 2
    }
}

/// Maps (component, copy, site) to a position inside one outer slab of the buffer.
pub trait Layout: Copy + Send + Sync + 'static {
    /// Sites processed together by one kernel iteration.
    const LANES: usize;
    fn offset(
        internal_dof: usize,
        copies: usize,
        num_sites: usize,
        c: usize,
        r: usize,
        site: usize,
    ) -> usize;
}

/// Array of structures: all reals of a site are contiguous.
#[derive(Clone, Copy)]
pub struct Aos;
impl Layout for Aos {
    const LANES: usize = 1;
    #[inline(always)]
    fn offset(
        internal_dof: usize,
        copies: usize,
        _num_sites: usize,
        c: usize,
        r: usize,
        site: usize,
    ) -> usize {
        c + internal_dof * (r + copies * site)
    }
}

/// Buffers of at least this many reals are first written in parallel (see `Cpu::alloc`).
const FIRST_TOUCH_MIN: usize = 1 << 20;
/// Reals per task of that first write: a few pages.
const FIRST_TOUCH_CHUNK: usize = 1 << 14;

/// Where memory lives.
pub trait Backend: Send + Sync + 'static {
    type Buf<T: Storage>: Send + Sync;
    fn alloc<T: Storage>(len: usize) -> Self::Buf<T>;
}

pub struct Cpu;
impl Backend for Cpu {
    type Buf<T: Storage> = Vec<T>;
    /// A zeroed buffer. Large ones are written once by the threads of the pool before
    /// they are returned: on a machine with several memory nodes a page is placed on the
    /// node of the thread that first writes to it, and a buffer written first by the
    /// calling thread alone ends up on one node, out of reach of the memory bandwidth of
    /// the others.
    fn alloc<T: Storage>(len: usize) -> Vec<T> {
        let mut buf = vec![T::default(); len];
        if len >= FIRST_TOUCH_MIN {
            buf.par_chunks_mut(FIRST_TOUCH_CHUNK)
                .for_each(|chunk| chunk.fill(T::default()));
        }
        buf
    }
}

/// Carries the marker types without owning them; `fn() -> ...` keeps a field Send + Sync
/// whatever the markers are.
type Marker<K, S, L> = PhantomData<fn() -> (K, S, L)>;

/// A lattice field.
///
/// Each site holds `ls * num_vectors` copies of `internal_dof` reals. `outer` is an index
/// outside the site (the four directions of a gauge field); each outer slab is laid out by `L`.
pub struct Field<T: Storage, K: Kind, S: Sites = Full, L: Layout = Aos, B: Backend = Cpu> {
    lat: Arc<Lattice>,
    buf: B::Buf<T>,
    internal_dof: usize,
    ls: usize,
    num_vectors: usize,
    outer: usize,
    num_sites: usize,
    /// Which parity a `Half` field lives on. Always `Even` for `Full` fields.
    parity: Parity,
    _marker: Marker<K, S, L>,
}

impl<T: Storage, K: Kind, S: Sites, L: Layout, B: Backend> Field<T, K, S, L, B> {
    /// A zero field with every shape parameter given. `internal_dof` must agree with the
    /// kind when the kind fixes it.
    pub fn with_shape(
        lat: &Arc<Lattice>,
        internal_dof: usize,
        ls: usize,
        num_vectors: usize,
        outer: usize,
    ) -> Self {
        if let Some(fixed) = K::INTERNAL_DOF {
            assert_eq!(
                internal_dof,
                fixed,
                "{} fields have {} internal dof",
                K::NAME,
                fixed
            );
        }
        assert!(internal_dof > 0 && ls > 0 && num_vectors > 0 && outer > 0);
        let num_sites = S::num_sites(lat);
        let len = internal_dof * ls * num_vectors * num_sites * outer;
        Self {
            lat: lat.clone(),
            buf: B::alloc(len),
            internal_dof,
            ls,
            num_vectors,
            outer,
            num_sites,
            parity: Parity::Even,
            _marker: PhantomData,
        }
    }

    pub fn parity(&self) -> Parity {
        self.parity
    }

    pub(crate) fn copy_parity_from(&mut self, other: &Self) {
        self.parity = other.parity;
    }

    pub fn lattice(&self) -> &Arc<Lattice> {
        &self.lat
    }
    pub fn internal_dof(&self) -> usize {
        self.internal_dof
    }
    pub fn ls(&self) -> usize {
        self.ls
    }
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    /// `ls * num_vectors`.
    pub fn copies(&self) -> usize {
        self.ls * self.num_vectors
    }
    pub fn outer(&self) -> usize {
        self.outer
    }
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }
    /// Reals per site within one outer slab.
    pub fn site_len(&self) -> usize {
        self.internal_dof * self.copies()
    }
    /// Reals in one outer slab.
    pub fn slab_len(&self) -> usize {
        self.site_len() * self.num_sites
    }
}

impl<T: Storage> Field<T, Gauge> {
    /// A gauge field: one link per direction per site, all zero.
    pub fn gauge(lat: &Arc<Lattice>) -> Self {
        Self::with_shape(lat, 18, 1, 1, crate::lattice::ND)
    }
}

impl<T: Storage, K: Kind, L: Layout, B: Backend> Field<T, K, Half, L, B> {
    pub fn set_parity(&mut self, parity: Parity) {
        self.parity = parity;
    }
}

impl<T: Storage> Field<T, Spinor, Half> {
    /// `num_vectors` spinor fields on the sites of one parity, all zero.
    pub fn spinor_half(lat: &Arc<Lattice>, num_vectors: usize, parity: Parity) -> Self {
        let mut f = Self::with_shape(lat, 24, 1, num_vectors, 1);
        f.parity = parity;
        f
    }
}

impl<T: Storage> Field<T, Spinor> {
    /// `num_vectors` four-dimensional spinor fields stored together, all zero.
    pub fn spinor(lat: &Arc<Lattice>, num_vectors: usize) -> Self {
        Self::with_shape(lat, 24, 1, num_vectors, 1)
    }
}

impl<T: Storage, K: Kind, S: Sites, L: Layout> Field<T, K, S, L, Cpu> {
    pub fn as_slice(&self) -> &[T] {
        &self.buf
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf
    }
    pub fn slab(&self, outer: usize) -> &[T] {
        let n = self.slab_len();
        &self.buf[outer * n..(outer + 1) * n]
    }
    pub fn slab_mut(&mut self, outer: usize) -> &mut [T] {
        let n = self.slab_len();
        &mut self.buf[outer * n..(outer + 1) * n]
    }
}

impl<T: Storage, K: Kind, S: Sites> Field<T, K, S, Aos, Cpu> {
    /// All reals of one site in one outer slab.
    #[inline]
    pub fn site(&self, outer: usize, site: usize) -> &[T] {
        let n = self.site_len();
        let start = outer * self.slab_len() + site * n;
        &self.buf[start..start + n]
    }
    #[inline]
    pub fn site_mut(&mut self, outer: usize, site: usize) -> &mut [T] {
        let n = self.site_len();
        let start = outer * self.slab_len() + site * n;
        &mut self.buf[start..start + n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauge_field_shape() {
        let lat = Lattice::single_node([4, 4, 4, 8]);
        let u = Field::<f64, Gauge>::gauge(&lat);
        assert_eq!(u.as_slice().len(), 4 * 512 * 18);
        assert_eq!(u.site(3, 511).len(), 18);
    }

    #[test]
    fn aos_site_matches_layout_offset() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let mut f = Field::<f64, Coarse>::with_shape(&lat, 6, 2, 3, 1);
        for (i, x) in f.as_mut_slice().iter_mut().enumerate() {
            *x = i as f64;
        }
        let (site, r, c) = (5, 4, 3);
        let k = Aos::offset(f.internal_dof(), f.copies(), f.num_sites(), c, r, site);
        assert_eq!(f.site(0, site)[c + f.internal_dof() * r], f.as_slice()[k]);
    }

    #[test]
    #[should_panic]
    fn fixed_kind_rejects_wrong_dof() {
        let lat = Lattice::single_node([2, 2, 2, 2]);
        let _ = Field::<f64, Spinor>::with_shape(&lat, 12, 1, 1, 1);
    }
}
