//! Numbers in several lanes: the same arithmetic on W values at once.

use super::{Number, Real};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A number made of `W` scalars, one per lane.
pub trait Lane<const W: usize>: Number {
    type Scalar: Real;
    fn from_array(a: [Self::Scalar; W]) -> Self;
    fn to_array(self) -> [Self::Scalar; W];

    /// Lane `i` of the result is lane `from[i]` of `self`.
    #[inline(always)]
    fn permute(self, from: &[usize; W]) -> Self {
        let a = self.to_array();
        let mut r = a;
        for i in 0..W {
            r[i] = a[from[i]];
        }
        Self::from_array(r)
    }
}

/// The lane type to use for `W` values of this scalar type.
pub trait LaneOf<const W: usize>: Real {
    type Lane: Lane<W, Scalar = Self>;
}

/// `W` scalars in an array, with element-wise arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Lanes<T, const W: usize>(pub [T; W]);

macro_rules! array_binary {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Real, const W: usize> $trait for Lanes<T, W> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..W {
                    r[i] = self.0[i] $op o.0[i];
                }
                Lanes(r)
            }
        }
    };
}
array_binary!(Add, add, +);
array_binary!(Sub, sub, -);
array_binary!(Mul, mul, *);

impl<T: Real, const W: usize> Neg for Lanes<T, W> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut r = self.0;
        for i in 0..W {
            r[i] = -self.0[i];
        }
        Lanes(r)
    }
}

/// The compound assignments of a lane type, from its binary operators.
macro_rules! assign_ops {
    ([$($generics:tt)*] $name:ty) => {
        impl<$($generics)*> AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl<$($generics)*> SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }
        impl<$($generics)*> MulAssign for $name {
            #[inline(always)]
            fn mul_assign(&mut self, o: Self) {
                *self = *self * o;
            }
        }
    };
}
assign_ops!([T: Real, const W: usize] Lanes<T, W>);

impl<T: Real, const W: usize> Number for Lanes<T, W> {
    const ZERO: Self = Lanes([T::ZERO; W]);
    const ONE: Self = Lanes([T::ONE; W]);
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        Lanes([T::from_f64(x); W])
    }
}

impl<T: Real, const W: usize> Lane<W> for Lanes<T, W> {
    type Scalar = T;
    #[inline(always)]
    fn from_array(a: [T; W]) -> Self {
        Lanes(a)
    }
    #[inline(always)]
    fn to_array(self) -> [T; W] {
        self.0
    }
}

/// A lane type that wraps a register type of the processor. It exists only in builds that
/// enable `$feature`, which is what makes the calls to the intrinsics sound: a program
/// built that way cannot run on a processor without the feature.
macro_rules! register_lane {
    ($name:ident, $reg:ident, $scalar:ty, $w:expr, $feature:literal,
     $add:ident, $sub:ident, $mul:ident, $set1:ident) => {
        #[cfg(target_feature = $feature)]
        #[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct $name(core::arch::x86_64::$reg);

        #[cfg(target_feature = $feature)]
        const _: () = {
            use core::arch::x86_64::*;

            impl Add for $name {
                type Output = Self;
                #[inline(always)]
                fn add(self, o: Self) -> Self {
                    // SAFETY: the build enables the feature (see `register_lane`).
                    $name(unsafe { $add(self.0, o.0) })
                }
            }
            impl Sub for $name {
                type Output = Self;
                #[inline(always)]
                fn sub(self, o: Self) -> Self {
                    // SAFETY: the build enables the feature (see `register_lane`).
                    $name(unsafe { $sub(self.0, o.0) })
                }
            }
            impl Mul for $name {
                type Output = Self;
                #[inline(always)]
                fn mul(self, o: Self) -> Self {
                    // SAFETY: the build enables the feature (see `register_lane`).
                    $name(unsafe { $mul(self.0, o.0) })
                }
            }
            impl Neg for $name {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    // Times −1, which is exact and turns +0 into −0 as `-x` does for a scalar;
                    // 0 − x would give +0 there and differ from the scalar in the sign bit.
                    // SAFETY: the build enables the feature (see `register_lane`).
                    $name(unsafe { $mul(self.0, $set1(-1.0)) })
                }
            }
    assign_ops!([] $name);

            impl Number for $name {
                // SAFETY: the register type and an array of `W` scalars have the same size,
                // and every bit pattern is valid for both.
                const ZERO: Self = unsafe { core::mem::transmute([0.0 as $scalar; $w]) };
                // SAFETY: as for ZERO.
                const ONE: Self = unsafe { core::mem::transmute([1.0 as $scalar; $w]) };
                #[inline(always)]
                fn from_f64(x: f64) -> Self {
                    // SAFETY: the build enables the feature (see `register_lane`).
                    $name(unsafe { $set1(<$scalar as Number>::from_f64(x)) })
                }
            }

            impl Lane<$w> for $name {
                type Scalar = $scalar;
                #[inline(always)]
                fn from_array(a: [$scalar; $w]) -> Self {
                    // SAFETY: as for ZERO.
                    unsafe { core::mem::transmute(a) }
                }
                #[inline(always)]
                fn to_array(self) -> [$scalar; $w] {
                    // SAFETY: as for ZERO.
                    unsafe { core::mem::transmute(self) }
                }
            }
        };
    };
}
register_lane!(
    F64x4,
    __m256d,
    f64,
    4,
    "avx2",
    _mm256_add_pd,
    _mm256_sub_pd,
    _mm256_mul_pd,
    _mm256_set1_pd
);
register_lane!(
    F32x8,
    __m256,
    f32,
    8,
    "avx2",
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_mul_ps,
    _mm256_set1_ps
);
register_lane!(
    F64x8,
    __m512d,
    f64,
    8,
    "avx512f",
    _mm512_add_pd,
    _mm512_sub_pd,
    _mm512_mul_pd,
    _mm512_set1_pd
);
register_lane!(
    F32x16,
    __m512,
    f32,
    16,
    "avx512f",
    _mm512_add_ps,
    _mm512_sub_ps,
    _mm512_mul_ps,
    _mm512_set1_ps
);

/// `LaneOf` with a register type when the build has `$feature`, the array otherwise.
macro_rules! lane_of {
    ($scalar:ty, $w:expr, array) => {
        impl LaneOf<$w> for $scalar {
            type Lane = Lanes<$scalar, $w>;
        }
    };
    ($scalar:ty, $w:expr, $feature:literal => $name:ident) => {
        #[cfg(target_feature = $feature)]
        impl LaneOf<$w> for $scalar {
            type Lane = $name;
        }
        #[cfg(not(target_feature = $feature))]
        impl LaneOf<$w> for $scalar {
            type Lane = Lanes<$scalar, $w>;
        }
    };
}
lane_of!(f64, 1, array);
lane_of!(f64, 2, array);
lane_of!(f64, 4, "avx2" => F64x4);
lane_of!(f64, 8, "avx512f" => F64x8);
lane_of!(f32, 1, array);
lane_of!(f32, 2, array);
lane_of!(f32, 4, array);
lane_of!(f32, 8, "avx2" => F32x8);
lane_of!(f32, 16, "avx512f" => F32x16);

#[cfg(test)]
mod tests {
    use super::*;

    /// A value computed with lanes and the scalar function it should equal in each lane.
    type Case<V, S> = (V, fn(S, S) -> S);

    /// Every operation of a lane type gives, in each lane, what the scalar gives.
    fn check<const W: usize, V: Lane<W>>() {
        let scalar = |k: usize, shift: f64| V::Scalar::from_f64(0.37 * k as f64 - 1.1 + shift);
        let a: [V::Scalar; W] = core::array::from_fn(|k| scalar(k, 0.0));
        let b: [V::Scalar; W] = core::array::from_fn(|k| scalar(3 * k + 1, 0.25));
        let (va, vb) = (V::from_array(a), V::from_array(b));
        assert!(va.to_array() == a);
        let cases: [Case<V, V::Scalar>; 5] = [
            (va + vb, |x, y| x + y),
            (va - vb, |x, y| x - y),
            (va * vb, |x, y| x * y),
            (-va, |x, _| -x),
            (va * V::from_f64(0.5) + V::ONE - V::ZERO, |x, _| {
                x * V::Scalar::from_f64(0.5) + V::Scalar::ONE - V::Scalar::ZERO
            }),
        ];
        for (n, (got, want)) in cases.into_iter().enumerate() {
            let got = got.to_array();
            for k in 0..W {
                assert!(got[k] == want(a[k], b[k]), "case {n}, lane {k}");
            }
        }
        // −(+0) is −0 and −(−0) is +0, as for the scalars: the same bits, not only the same
        // value.
        let zeros: [V::Scalar; W] = core::array::from_fn(|k| {
            let z = V::Scalar::ZERO;
            if k % 2 == 0 {
                z
            } else {
                -z
            }
        });
        let negated = (-V::from_array(zeros)).to_array();
        for k in 0..W {
            assert!(
                negated[k].to_f64().to_bits() == (-zeros[k]).to_f64().to_bits(),
                "the sign of a negated zero, lane {k}"
            );
        }
        let mut c = va;
        c += vb;
        c -= va;
        c *= vb;
        let c = c.to_array();
        for k in 0..W {
            assert!(c[k] == (a[k] + b[k] - a[k]) * b[k]);
        }
        let from: [usize; W] = core::array::from_fn(|k| (k + 1) % W);
        let turned = va.permute(&from).to_array();
        for k in 0..W {
            assert!(turned[k] == a[(k + 1) % W]);
        }
    }

    #[test]
    fn lane_types_agree_with_scalars() {
        check::<1, Lanes<f64, 1>>();
        check::<4, Lanes<f64, 4>>();
        check::<8, Lanes<f32, 8>>();
        // Whatever the build selects, register types included.
        check::<2, <f64 as LaneOf<2>>::Lane>();
        check::<4, <f64 as LaneOf<4>>::Lane>();
        check::<8, <f64 as LaneOf<8>>::Lane>();
        check::<4, <f32 as LaneOf<4>>::Lane>();
        check::<8, <f32 as LaneOf<8>>::Lane>();
        check::<16, <f32 as LaneOf<16>>::Lane>();
    }
}
