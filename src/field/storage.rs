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

/// A number in IEEE half precision (binary16), computed with as `f32`: for what is read far
/// more often than it is written and needs about three digits, the prolongation of the
/// multigrid. Not exchanged between processes.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// The nearest number in half precision, ties to even (as `__float2half_rn` on a device).
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7f_ffff;
        if exponent == 0xff {
            let nan = if mantissa != 0 { 0x200 } else { 0 };
            return Self(sign | 0x7c00 | nan);
        }
        let e = exponent - 127 + 15;
        if e >= 31 {
            return Self(sign | 0x7c00);
        }
        // Round `value >> shift` to the nearest, ties to even.
        let round = |value: u32, shift: u32| {
            let (kept, rest, halfway) =
                (value >> shift, value & ((1 << shift) - 1), 1 << (shift - 1));
            if rest > halfway || (rest == halfway && kept & 1 == 1) {
                kept + 1
            } else {
                kept
            }
        };
        if e <= 0 {
            // Subnormal in half precision (or zero): units of 2^-24.
            if e < -10 {
                return Self(sign);
            }
            return Self(sign | round(mantissa | 0x80_0000, (14 - e) as u32) as u16);
        }
        // A carry out of the mantissa goes into the exponent, as it should.
        Self(sign | round(((e as u32) << 23) | mantissa, 13) as u16)
    }

    /// Without branches but the one for infinities and NaN: the exponent and mantissa moved
    /// into place and multiplied by 2^112, which is exact for normal and subnormal numbers.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        let h = u32::from(self.0);
        let (sign, magnitude) = ((h & 0x8000) << 16, h & 0x7fff);
        let scaled = f32::from_bits(magnitude << 13) * f32::from_bits(0x7780_0000);
        let value = if magnitude >= 0x7c00 {
            f32::from_bits((magnitude << 13) | 0x7f80_0000)
        } else {
            scaled
        };
        f32::from_bits(value.to_bits() | sign)
    }
}

/// `dst = src` in single precision: eight at a time with F16C where the processor has it.
pub fn widen_halves(src: &[F16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") {
        // SAFETY: the processor has F16C and AVX (F16C implies it), checked just above.
        unsafe { widen_f16c(src, dst) };
        return;
    }
    for (d, h) in dst.iter_mut().zip(src) {
        *d = h.to_f32();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn widen_f16c(src: &[F16], dst: &mut [f32]) {
    use core::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128};
    let whole = src.len() / 8 * 8;
    for k in (0..whole).step_by(8) {
        // SAFETY: eight values from `k` on in both slices (k + 8 <= whole <= len); unaligned
        // loads and stores.
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(k).cast());
            _mm256_storeu_ps(dst.as_mut_ptr().add(k), _mm256_cvtph_ps(h));
        }
    }
    for k in whole..src.len() {
        dst[k] = src[k].to_f32();
    }
}

impl Storage for F16 {
    type Compute = f32;
    #[inline(always)]
    fn load(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn store(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn halo(_: &dyn Comm, _: &HaloSpec) -> Box<dyn HaloExchange<Self>> {
        unimplemented!("fields in half precision are not exchanged between processes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every number in half precision goes to `f32` and back unchanged, and rounding goes to
    /// the nearest, ties to even.
    #[test]
    fn half_precision_round_trips_and_rounds_to_nearest() {
        for bits in 0..=u16::MAX {
            let h = F16(bits);
            let x = h.to_f32();
            if x.is_nan() {
                assert!(F16::from_f32(x).to_f32().is_nan());
            } else {
                assert_eq!(F16::from_f32(x), h, "{bits:#06x}");
            }
        }
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: to 1 (even); 1 + 3 * 2^-11 to 1 + 2^-9.
        assert_eq!(F16::from_f32(1.0 + 2f32.powi(-11)).to_f32(), 1.0);
        assert_eq!(
            F16::from_f32(1.0 + 3.0 * 2f32.powi(-11)).to_f32(),
            1.0 + 2f32.powi(-9)
        );
        assert_eq!(F16::from_f32(65520.0).to_f32(), f32::INFINITY);
        assert_eq!(F16::from_f32(2f32.powi(-24)).to_f32(), 2f32.powi(-24));
        assert_eq!(F16::from_f32(2f32.powi(-26)).to_f32(), 0.0);
        assert_eq!(F16::from_f32(-0.1).to_f32(), -0.099975586);
        // The fast path of the processor gives the same.
        let all: Vec<F16> = (0..=u16::MAX).map(F16).collect();
        let mut wide = vec![0.0f32; all.len()];
        widen_halves(&all, &mut wide);
        for (h, w) in all.iter().zip(&wide) {
            let x = h.to_f32();
            assert!(
                x.to_bits() == w.to_bits() || (x.is_nan() && w.is_nan()),
                "{:#06x}",
                h.0
            );
        }
    }
}
