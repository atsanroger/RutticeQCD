use crate::comm::{Comm, HaloExchange, HaloSpec};
use crate::kernel::Real;

/// What a field stores. Kernels compute in `Compute`; only load/store touch the stored type,
/// so a storage format that differs from the compute format needs no new kernels.
pub trait Storage: Copy + Default + Send + Sync + 'static {
    type Compute: Real;
    fn load(self) -> Self::Compute;
    fn store(x: Self::Compute) -> Self;
    /// Buffers for exchanging faces of fields stored as this type.
    fn halo(comm: &dyn Comm, spec: &HaloSpec) -> Box<dyn HaloExchange<Self>>;
    /// `dst.copy_from_slice(src)` with stores that bypass the cache where the processor has
    /// them, an ordinary copy otherwise.
    ///
    /// After the last call of a task, `stream_fence` must be called before other threads may read
    /// the data.
    #[inline(always)]
    fn stream(dst: &mut [Self], src: &[Self]) {
        dst.copy_from_slice(src);
    }
}

/// Make the data written by `Storage::stream` on this thread visible to the others in
/// order with everything written later. Does nothing where `stream` is an ordinary copy.
#[inline(always)]
pub fn stream_fence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SFENCE has no operands and no preconditions; it is part of SSE, which every
    // x86-64 processor has.
    unsafe {
        core::arch::x86_64::_mm_sfence()
    }
}

/// `Storage::stream` for one type on x86-64, in pieces of one 128-bit register: `$lanes`
/// values, stored with `$store` after loading with `$load`. SSE2 is part of x86-64, so
/// this needs no check of the processor.
#[cfg(target_arch = "x86_64")]
macro_rules! stream_x86_64 {
    ($lanes:literal, $load:ident, $store:ident) => {
        #[inline(always)]
        fn stream(dst: &mut [Self], src: &[Self]) {
            use core::arch::x86_64::{$load, $store};
            assert_eq!(dst.len(), src.len());
            if dst.as_ptr().align_offset(16) != 0 || dst.len() % $lanes != 0 {
                dst.copy_from_slice(src);
                return;
            }
            for (d, s) in dst.chunks_exact_mut($lanes).zip(src.chunks_exact($lanes)) {
                // SAFETY: `d` and `s` are `$lanes` values each, the size of the register.
                // `dst` starts at a multiple of 16 bytes (checked above) and the pieces
                // are 16 bytes long, so every `d` is aligned as the store requires; the
                // load has no alignment requirement.
                unsafe { $store(d.as_mut_ptr(), $load(s.as_ptr())) }
            }
        }
    };
}

impl Storage for f32 {
    type Compute = f32;
    #[inline(always)]
    fn load(self) -> f32 {
        self
    }
    #[inline(always)]
    fn store(x: f32) -> f32 {
        x
    }
    fn halo(comm: &dyn Comm, spec: &HaloSpec) -> Box<dyn HaloExchange<f32>> {
        comm.halo_f32(spec)
    }
    #[cfg(target_arch = "x86_64")]
    stream_x86_64!(4, _mm_loadu_ps, _mm_stream_ps);
}

impl Storage for f64 {
    type Compute = f64;
    #[inline(always)]
    fn load(self) -> f64 {
        self
    }
    #[inline(always)]
    fn store(x: f64) -> f64 {
        x
    }
    fn halo(comm: &dyn Comm, spec: &HaloSpec) -> Box<dyn HaloExchange<f64>> {
        comm.halo_f64(spec)
    }
    #[cfg(target_arch = "x86_64")]
    stream_x86_64!(2, _mm_loadu_pd, _mm_stream_pd);
}
