use ruttice_kernel::Real;

/// What a field stores. Kernels compute in `Compute`; only load/store touch the stored type,
/// so a storage format that differs from the compute format needs no new kernels.
pub trait Storage: Copy + Default + Send + Sync + 'static {
    type Compute: Real;
    fn load(self) -> Self::Compute;
    fn store(x: Self::Compute) -> Self;
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
}
