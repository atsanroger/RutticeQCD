//! The multigrid of the Wilson and clover operators on the device: what `WilsonMultigrid`
//! needs of a backend (`MultigridBackend`), with P made on the host as on the CPU and kept on
//! the device folded in spin (`DeviceBasis`), the coarse operator made on the device
//! (`coarse::coarse_blocks`), and SAP on the device (`sap`).

use super::sap::{device_sap, BlockOperator};
use super::*;
use crate::fermion::{Bound, Variant};
use crate::solver::preconditioner::multigrid::{
    random_vectors, Aggregation, Basis, CycleLevels, MultigridBackend, MultigridError,
    MultigridParams, SetupHooks, SmootherChoice, WilsonMultigrid,
};
use crate::solver::{Fgmres, Solve, Stop};
use crate::{LinearFn, Vector, C64};

/// What a four-dimensional operator on the device gives its multigrid: the block operator of
/// SAP and the coarse operator on a P.
pub trait MultigridOperator<T: DeviceReal>: BlockOperator<T> {
    /// P† D P on `basis`, made and kept on the device (`even_odd`: with the even-odd form).
    fn coarse_on_device(
        &self,
        basis: &DeviceBasis,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<T>, Box<dyn std::error::Error>>;
}

impl<T: DeviceReal> MultigridOperator<T> for CudaWilson<T> {
    fn coarse_on_device(
        &self,
        basis: &DeviceBasis,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<T>, Box<dyn std::error::Error>> {
        self.coarse_operator_on_device(basis, even_odd)
    }
}

impl<T: DeviceReal> MultigridOperator<T> for CudaClover<T> {
    fn coarse_on_device(
        &self,
        basis: &DeviceBasis,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<T>, Box<dyn std::error::Error>> {
        self.coarse_operator_on_device(basis, even_odd)
    }
}

fn backend_error(e: impl std::fmt::Display) -> MultigridError {
    MultigridError::Backend(format!("the device: {e}"))
}

/// The multigrid of the operator `op` on the device.
pub struct WilsonOnDevice<'a, O: MultigridOperator<T>, T: DeviceReal> {
    op: &'a O,
    precision: PhantomData<T>,
}

impl<'a, O: MultigridOperator<T>, T: DeviceReal> WilsonOnDevice<'a, O, T> {
    pub fn new(op: &'a O) -> Self {
        Self {
            op,
            precision: PhantomData,
        }
    }
}

impl<'a, O: MultigridOperator<T> + 'a, T: DeviceReal> MultigridBackend<'a>
    for WilsonOnDevice<'a, O, T>
{
    type Fine = DeviceSpinor<T>;
    type Coarse = DeviceCoarse<T>;
    type Basis = DeviceBasis;
    type CoarseOp = DeviceCoarseOp<T>;
    type CoarseTests = ();

    fn operator(&self) -> Box<dyn LinearFn<DeviceSpinor<T>> + 'a> {
        Box::new(Bound::new(self.op, Variant::D))
    }

    fn smoother(
        &self,
        agg: &Arc<Aggregation>,
        choice: SmootherChoice,
    ) -> Box<dyn LinearFn<DeviceSpinor<T>> + 'a> {
        match choice {
            SmootherChoice::Sap {
                cycles,
                block_steps,
            } => Box::new(must(device_sap(self.op, agg, cycles, block_steps))),
            SmootherChoice::Gmres {
                iterations,
                tolerance_sqr,
            } => Box::new(Solve::new(
                Fgmres::new(Stop::simple(tolerance_sqr, iterations)),
                Bound::new(self.op, Variant::D),
            )),
        }
    }

    /// As on the CPU: the vectors and γ5 of them brought to the host, orthonormalised there
    /// (`Basis::from_vectors`, the code of the CPU), P kept on the device in single precision.
    fn basis(
        &mut self,
        agg: &Arc<Aggregation>,
        x: &DeviceSpinor<T>,
    ) -> Result<DeviceBasis, MultigridError> {
        let mut g5x = x.zeros_like();
        super::wilson::gamma5(&mut g5x, x, false);
        let (x64, g64) = (
            x.to_host().map_err(backend_error)?,
            g5x.to_host().map_err(backend_error)?,
        );
        let basis = Basis::<f64, f32>::from_vectors(agg, &x64, &g64)?;
        DeviceBasis::new(&basis).map_err(backend_error)
    }

    fn coarse_operator(
        &mut self,
        basis: &DeviceBasis,
        restriction: Option<&DeviceBasis>,
        hooks: &mut SetupHooks<'a, DeviceSpinor<T>>,
        even_odd: bool,
    ) -> Result<DeviceCoarseOp<T>, MultigridError> {
        assert!(
            restriction.is_none()
                && hooks.galerkin_of.is_none()
                && hooks.coarse_right.is_none()
                && !hooks.precondition_coarse_diagonal,
            "the coarse operator on the device is P† D P (refused when the file is read)"
        );
        self.op
            .coarse_on_device(basis, even_odd)
            .map_err(backend_error)
    }

    fn test_vectors(&self, basis: &DeviceBasis) -> DeviceSpinor<T> {
        let n = basis.num_test_vectors();
        let (sites, dim) = (basis.coarse_sites, basis.coarse_dim());
        // Ones at components i and n + i of every coarse site, for vector i.
        let mut c = must(DeviceCoarse::<T>::units(sites, dim, 0, n)).on(&basis.coarse);
        let upper = must(DeviceCoarse::<T>::units(sites, dim, n, n)).on(&basis.coarse);
        c.axpy(&vec![C64::new(1.0, 0.0); n], &upper);
        let mut fine = must(basis.fine_field::<T>(n));
        basis.prolong(&mut fine, &c);
        fine
    }

    fn coarse_tests(&self, _basis: &DeviceBasis, _x: &DeviceSpinor<T>) {}

    fn diagonal_preconditioned(_coarse: &DeviceCoarseOp<T>) -> bool {
        false
    }

    fn apply_diagonal_inverse(
        _coarse: &DeviceCoarseOp<T>,
        _out: &mut DeviceCoarse<T>,
        _inp: &DeviceCoarse<T>,
    ) {
        unreachable!("the coarse operator on the device is not preconditioned by its blocks")
    }

    fn levels(
        _coarse: &mut DeviceCoarseOp<T>,
        _tests: &(),
        _super_blocks: &[usize],
    ) -> Result<CycleLevels<'a, DeviceCoarse<T>>, MultigridError> {
        Err(MultigridError::Backend(
            "the levels below the coarse lattice of the Wilson and clover multigrid are not on the device yet".into(),
        ))
    }
}

/// The multigrid of a Wilson or clover operator on the device.
pub type DeviceMultigrid<'a, O, T> = WilsonMultigrid<'a, WilsonOnDevice<'a, O, T>>;

/// The multigrid of `op` on the device from the choices in `params`: the initial test vectors
/// those of the CPU (`random_vectors`, made on the host), improved `params.setup_iterations`
/// times.
pub fn device_multigrid<'a, O: MultigridOperator<T> + 'a, T: DeviceReal>(
    op: &'a O,
    lat: &Arc<Lattice>,
    params: &MultigridParams,
) -> Result<DeviceMultigrid<'a, O, T>, MultigridError> {
    let host = random_vectors::<f64>(lat, 1, params.num_test_vectors, params.seed);
    let initial = DeviceSpinor::<T>::from_host(&host).map_err(backend_error)?;
    DeviceMultigrid::from_parameters(
        WilsonOnDevice::new(op),
        lat,
        params,
        initial,
        true,
        params.setup_iterations,
    )
}
