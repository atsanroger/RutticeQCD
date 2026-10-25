//! The Schwarz smoother of the Wilson and clover multigrid on the device: the parts of `Sap`
//! that touch the fields (`multigrid/sap.cuh`), the block operator of the operators
//! (`hop_kernel` with a `BlockCut`).

use super::coarse::DeviceTable;
use super::*;
use crate::fermion::{FermionOp, Variant};
use crate::solver::preconditioner::multigrid::{Aggregation, Sap, SapBlocks};
use crate::Vector;

/// The kernels of SAP in one precision.
pub trait SapReal: Sized + Copy + 'static {
    /// `bridge_cuda_hop_block`.
    ///
    /// # Safety
    /// Fields of one spinor a site on the whole lattice of `geometry`, `vectors` of them; `out`
    /// none of the others; `clover` null or the blocks of every site; `cut` six numbers.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hop_block(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        from: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        kappa: f64,
        vectors: i32,
        cut: *const i32,
    );
    /// `bridge_cuda_sap_keep_colour`.
    ///
    /// # Safety
    /// `v` a field of `vectors` vectors on `sites` sites; the tables those of its blocks.
    unsafe fn keep_colour(
        v: *mut Self,
        blocks: *const u32,
        block_of: *const u32,
        sites: i32,
        vectors: i32,
        c: i32,
    );
    /// `bridge_cuda_sap_alpha`.
    ///
    /// # Safety
    /// As `keep_colour`; `alpha` two numbers for every block and vector.
    #[allow(clippy::too_many_arguments)]
    unsafe fn alpha(
        alpha: *mut f64,
        p: *const Self,
        r: *const Self,
        blocks: *const u32,
        block_sites: *const u32,
        n_blocks: i32,
        block_volume: i32,
        sites: i32,
        vectors: i32,
        c: i32,
    );
    /// `bridge_cuda_sap_update`.
    ///
    /// # Safety
    /// As `keep_colour`, three distinct fields; `alpha` as `alpha` made it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn update(
        e: *mut Self,
        r: *mut Self,
        p: *const Self,
        alpha: *const f64,
        block_of: *const u32,
        sites: i32,
        vectors: i32,
    );
}

macro_rules! sap_real {
    ($t:ty: $block:ident, $keep:ident, $alpha:ident, $update:ident) => {
        extern "C" {
            fn $block(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                dagger: i32,
                vectors: i32,
                cut: *const i32,
                stream: *mut c_void,
            );
            fn $keep(
                v: *mut $t,
                blocks: *const u32,
                block_of: *const u32,
                sites: i32,
                vectors: i32,
                c: i32,
                stream: *mut c_void,
            );
            fn $alpha(
                alpha: *mut f64,
                p: *const $t,
                r: *const $t,
                blocks: *const u32,
                block_sites: *const u32,
                n_blocks: i32,
                block_volume: i32,
                sites: i32,
                vectors: i32,
                c: i32,
                stream: *mut c_void,
            );
            fn $update(
                e: *mut $t,
                r: *mut $t,
                p: *const $t,
                alpha: *const f64,
                block_of: *const u32,
                sites: i32,
                vectors: i32,
                stream: *mut c_void,
            );
        }

        impl SapReal for $t {
            unsafe fn hop_block(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                vectors: i32,
                cut: *const i32,
            ) {
                $block(
                    out, inp, gauge, from, clover, geometry, kappa, 0, vectors, cut, STREAM,
                )
            }
            unsafe fn keep_colour(
                v: *mut $t,
                blocks: *const u32,
                block_of: *const u32,
                sites: i32,
                vectors: i32,
                c: i32,
            ) {
                $keep(v, blocks, block_of, sites, vectors, c, STREAM)
            }
            unsafe fn alpha(
                alpha: *mut f64,
                p: *const $t,
                r: *const $t,
                blocks: *const u32,
                block_sites: *const u32,
                n_blocks: i32,
                block_volume: i32,
                sites: i32,
                vectors: i32,
                c: i32,
            ) {
                $alpha(
                    alpha,
                    p,
                    r,
                    blocks,
                    block_sites,
                    n_blocks,
                    block_volume,
                    sites,
                    vectors,
                    c,
                    STREAM,
                )
            }
            unsafe fn update(
                e: *mut $t,
                r: *mut $t,
                p: *const $t,
                alpha: *const f64,
                block_of: *const u32,
                sites: i32,
                vectors: i32,
            ) {
                $update(e, r, p, alpha, block_of, sites, vectors, STREAM)
            }
        }
    };
}

sap_real!(f32: bridge_cuda_hop_block_f32, bridge_cuda_sap_keep_colour_f32,
    bridge_cuda_sap_alpha_f32, bridge_cuda_sap_update_f32);
sap_real!(f64: bridge_cuda_hop_block_f64, bridge_cuda_sap_keep_colour_f64,
    bridge_cuda_sap_alpha_f64, bridge_cuda_sap_update_f64);

/// The block operator a four-dimensional operator on the device gives SAP.
pub trait BlockOperator<T: DeviceReal>: FermionOp<DeviceSpinor<T>> {
    /// `out = D_c inp`: D without the hops that leave a block, on the blocks `cut` keeps
    /// (extent[4], colour, shift: `BlockCut` of `fermion/hops.cuh`), zero on the others.
    fn block_mult(&self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>, cut: &[i32; 6]);
}

/// The parts of SAP on the device for the operator `op` and the blocks of `agg`.
pub struct DeviceBlocks<'a, O: BlockOperator<T>, T: DeviceReal> {
    op: &'a O,
    work: O::Work,
    /// The colour of every block, its sites, and the block of every site.
    colours: DeviceTable,
    block_sites: DeviceTable,
    block_of: DeviceTable,
    n_blocks: usize,
    block_volume: usize,
    sites: usize,
    extent: [i32; ND],
    /// The parity of the block coordinates of the origin of this process.
    shift: i32,
    /// α of every block and vector.
    alpha: Option<DeviceBuffer<f64>>,
}

impl<'a, O: BlockOperator<T>, T: DeviceReal> DeviceBlocks<'a, O, T> {
    pub fn new(op: &'a O, agg: &Arc<Aggregation>) -> Result<Self, CudaError> {
        let (fine, coarse) = (agg.fine_lattice(), agg.coarse_lattice());
        let colours: Vec<u32> = (0..coarse.num_sites())
            .map(|c| (coarse.global_coords(c).iter().sum::<usize>() % 2) as u32)
            .collect();
        let block_sites: Vec<u32> = (0..coarse.num_sites())
            .flat_map(|c| agg.fine_sites(c).iter().copied())
            .collect();
        let block_of: Vec<u32> = (0..fine.num_sites())
            .map(|x| agg.coarse_site(x) as u32)
            .collect();
        let block = agg.block();
        let origin = fine.global_coords(0);
        let shift = (0..ND).map(|mu| origin[mu] / block[mu]).sum::<usize>() % 2;
        Ok(Self {
            op,
            work: op.work(),
            colours: DeviceTable::from_host(&colours)?,
            block_sites: DeviceTable::from_host(&block_sites)?,
            block_of: DeviceTable::from_host(&block_of)?,
            n_blocks: coarse.num_sites(),
            block_volume: agg.block_volume(),
            sites: fine.num_sites(),
            extent: block.map(|b| b as i32),
            shift: shift as i32,
            alpha: None,
        })
    }

    fn check(&self, v: &DeviceSpinor<T>) {
        assert!(v.ls == 1 && v.parity.is_none() && v.sites() == self.sites);
    }
}

impl<O: BlockOperator<T>, T: DeviceReal> SapBlocks<DeviceSpinor<T>> for DeviceBlocks<'_, O, T> {
    fn block_mult(&mut self, c: u8, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        self.check(inp);
        let e = self.extent;
        self.op.block_mult(
            out,
            inp,
            &[e[0], e[1], e[2], e[3], i32::from(c), self.shift],
        );
    }

    fn keep_colour(&mut self, c: u8, v: &mut DeviceSpinor<T>) {
        self.check(v);
        crate::meter::vector(v.lattice() * v.vectors, std::mem::size_of::<T>(), 0.0, 1.0);
        // SAFETY: a field of this lattice and the tables of its blocks.
        unsafe {
            T::keep_colour(
                v.data.ptr,
                self.colours.ptr,
                self.block_of.ptr,
                self.sites as i32,
                v.vectors as i32,
                i32::from(c),
            );
        }
    }

    fn step(
        &mut self,
        c: u8,
        e: &mut DeviceSpinor<T>,
        r: &mut DeviceSpinor<T>,
        p: &DeviceSpinor<T>,
    ) {
        self.check(r);
        assert!(e.same_shape(r) && p.same_shape(r));
        let nv = r.vectors;
        let len = 2 * self.n_blocks * nv;
        if !self.alpha.as_ref().is_some_and(|a| a.len == len) {
            self.alpha = Some(must(DeviceBuffer::zeros(len)));
        }
        let alpha = self.alpha.as_ref().expect("the buffer was just made");
        // <p, r> and |p|² (6 flops a real, both read), then e and r (8, five reals).
        crate::meter::vector(r.lattice() * nv, std::mem::size_of::<T>(), 6.0, 2.0);
        crate::meter::vector(r.lattice() * nv, std::mem::size_of::<T>(), 8.0, 5.0);
        // SAFETY: three fields of this lattice with `nv` vectors, e and r distinct from each
        // other and from p (borrowed mutably), α two numbers a block and vector.
        unsafe {
            T::alpha(
                alpha.ptr,
                p.data.ptr,
                r.data.ptr,
                self.colours.ptr,
                self.block_sites.ptr,
                self.n_blocks as i32,
                self.block_volume as i32,
                self.sites as i32,
                nv as i32,
                i32::from(c),
            );
            T::update(
                e.data.ptr,
                r.data.ptr,
                p.data.ptr,
                alpha.ptr,
                self.block_of.ptr,
                self.sites as i32,
                nv as i32,
            );
        }
    }

    fn mult(&mut self, out: &mut DeviceSpinor<T>, inp: &DeviceSpinor<T>) {
        self.op.mult(&mut self.work, out, inp, Variant::D);
    }
}

/// SAP on the device: `cycles` cycles of `block_steps` minimal-residual steps per block.
pub fn device_sap<'a, O: BlockOperator<T>, T: DeviceReal>(
    op: &'a O,
    agg: &Arc<Aggregation>,
    cycles: usize,
    block_steps: usize,
) -> Result<Sap<DeviceBlocks<'a, O, T>, DeviceSpinor<T>>, CudaError> {
    Ok(Sap::with_blocks(
        DeviceBlocks::new(op, agg)?,
        cycles,
        block_steps,
    ))
}
