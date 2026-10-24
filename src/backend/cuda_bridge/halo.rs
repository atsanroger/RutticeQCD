//! Several processes, each with its own device: the hops of the domain-wall operator with the
//! faces of the local lattice exchanged around them (`cuda_bridge/include/fermion/hops.cuh`,
//! `Halo`).
//!
//! For every direction cut between processes, a kernel packs each face of the input into a
//! buffer of half spinors (12 reals a site and fibre: the leg's (1 ∓ γ_μ) w, with the link
//! already applied for a backward leg, so no process needs the links of another). The faces
//! travel while the interior of the output is computed (the sites none of whose legs comes from
//! another process); the surface is computed after, every leg of a site in the order of one
//! process, so the result is that of one process bit for bit.
//!
//! The sites of a face are numbered by their three other coordinates (`face_site` in hops.cuh):
//! the device layout keeps the sites of a face apart in memory (blocks of 32 sites in lexical
//! order, a parity numbered x/2), so the faces are gathered by a kernel, which computes the
//! number from the coordinates on both sides; no table of sites is kept or sent.
//!
//! The faces travel from device to device by NCCL (feature `nccl`: NVLink or PCIe within a node,
//! GPUDirect RDMA between nodes), on a stream of their own while the interior is computed on the
//! default one, the host waiting for nothing; or through the host (`Comm::halo_f32`,
//! `halo_f64`: MPI, or the threads of a `LocalComm`), copied off the device, exchanged while the
//! interior runs, copied back. NCCL where it can be had on every process, the host otherwise
//! (`RUTTICE_HALO=host` or `nccl` chooses).

use super::*;
use crate::comm::{HaloExchange, HaloSpec, FACES};
use std::cell::RefCell;
use std::rc::Rc;

/// What the kernels take of an exchange (`BridgeCudaHalo`).
#[repr(C)]
pub struct HaloArgs {
    recv: [*const c_void; FACES],
    bc: [f64; FACES],
    cut: [i32; ND],
    surface: *const i32,
    surface_sites: i32,
}

/// The part of the output a kernel computes (`Part` in hops.cuh).
pub(super) const INTERIOR: i32 = 1;
pub(super) const SURFACE: i32 = 2;

/// The kernels of several processes in one precision.
pub trait SplitReal: Sized + Copy + Default + Send + 'static {
    /// `bridge_cuda_hop_part`.
    ///
    /// # Safety
    /// As `DeviceReal::hop`; `halo` as `Halos::around` makes it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hop_part(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        from: *const Self,
        clover: *const Self,
        geometry: *const Geometry,
        kappa: f64,
        fifth: [f64; 3],
        dagger: i32,
        target_parity: i32,
        vectors: i32,
        halo: *const HaloArgs,
        part: i32,
    );
    /// `bridge_cuda_mdwf_dslash_part`.
    ///
    /// # Safety
    /// As `DeviceReal::mdwf_dslash`; `halo` as for `hop_part`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dslash_part(
        out: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        params: *const MdwfKernelParams,
        vectors: i32,
        halo: *const HaloArgs,
        part: i32,
    );
    /// `bridge_cuda_hop_pack`.
    ///
    /// # Safety
    /// `send` holds the face of `face` for these vectors; `inp` the sites of `in_parity`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hop_pack(
        send: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        fifth: [f64; 3],
        dagger: i32,
        in_parity: i32,
        face: i32,
        vectors: i32,
    );
    /// The buffers of an exchange through the host in this precision.
    fn host_exchange(comm: &dyn crate::Comm, spec: &HaloSpec) -> Box<dyn HaloExchange<Self>>;
    /// `ncclDataType_t` of this precision.
    const NCCL_TYPE: i32;
    /// `bridge_cuda_hop_split_halo`.
    ///
    /// # Safety
    /// As `DeviceReal::hop_split`; `halo` as for `hop_part`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn hop_split_halo(
        cross: *mut Self,
        inside: *mut Self,
        inp: *const Self,
        gauge: *const Self,
        geometry: *const Geometry,
        fifth: [f64; 4],
        leg: i32,
        block: *const i32,
        vectors: i32,
        halo: *const HaloArgs,
    );
    /// `bridge_cuda_coarse_pass_ghost`.
    ///
    /// # Safety
    /// As `DeviceReal::coarse_pass`; `ghost` a coarse field of `ghosts` sites and as many
    /// vectors, which the neighbours `>= sites` index.
    #[allow(clippy::too_many_arguments)]
    unsafe fn coarse_pass_ghost(
        out: *mut Self,
        base: *const Self,
        diag_of: *const Self,
        hops_of: *const Self,
        diag: *const Self,
        links: *const Self,
        inverse: *const Self,
        neighbours: *const u32,
        odd: *const u32,
        sizes: [i32; 6],
        ghost: *const Self,
        ghosts: i32,
    );
    /// `bridge_cuda_coarse_gather` and `bridge_cuda_coarse_place`.
    ///
    /// # Safety
    /// Coarse fields of the sizes given, `sites` holding `n` site numbers of `field`.
    unsafe fn coarse_gather(
        buf: *mut Self,
        field: *const Self,
        sites: *const u32,
        n: i32,
        n_coarse: i32,
        dim: i32,
        vectors: i32,
    );
    /// # Safety
    /// As `coarse_gather`.
    unsafe fn coarse_place(
        ghost: *mut Self,
        buf: *const Self,
        first: i32,
        n: i32,
        ghosts: i32,
        dim: i32,
        vectors: i32,
    );
}

macro_rules! split_real {
    ($t:ty: $part:ident, $dslash:ident, $pack:ident, $halo:ident, $nccl:expr,
     $split:ident, $pass:ident, $gather:ident, $place:ident) => {
        extern "C" {
            fn $split(
                cross: *mut $t,
                inside: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                leg: i32,
                block: *const i32,
                vectors: i32,
                halo: *const HaloArgs,
                stream: *mut c_void,
            );
            fn $pass(
                out: *mut $t,
                base: *const $t,
                diag_of: *const $t,
                hops_of: *const $t,
                diag: *const $t,
                links: *const $t,
                inverse: *const $t,
                neighbours: *const u32,
                odd: *const u32,
                n_coarse: i32,
                dim: i32,
                parity: i32,
                minus: i32,
                gamma5: i32,
                vectors: i32,
                ghost: *const $t,
                ghosts: i32,
                stream: *mut c_void,
            );
            fn $gather(
                buf: *mut $t,
                field: *const $t,
                sites: *const u32,
                n: i32,
                n_coarse: i32,
                dim: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $place(
                ghost: *mut $t,
                buf: *const $t,
                first: i32,
                n: i32,
                ghosts: i32,
                dim: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $part(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                target_parity: i32,
                vectors: i32,
                halo: *const HaloArgs,
                part: i32,
                stream: *mut c_void,
            );
            fn $dslash(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
                vectors: i32,
                halo: *const HaloArgs,
                part: i32,
                stream: *mut c_void,
            );
            fn $pack(
                send: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                mass: f64,
                b5: f64,
                c5: f64,
                dagger: i32,
                in_parity: i32,
                face: i32,
                vectors: i32,
                stream: *mut c_void,
            );
        }

        impl SplitReal for $t {
            const NCCL_TYPE: i32 = $nccl;
            unsafe fn hop_split_halo(
                cross: *mut $t,
                inside: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                [kappa, mass, b5, c5]: [f64; 4],
                leg: i32,
                block: *const i32,
                vectors: i32,
                halo: *const HaloArgs,
            ) {
                $split(
                    cross, inside, inp, gauge, geometry, kappa, mass, b5, c5, leg, block, vectors,
                    halo, STREAM,
                )
            }
            unsafe fn coarse_pass_ghost(
                out: *mut $t,
                base: *const $t,
                diag_of: *const $t,
                hops_of: *const $t,
                diag: *const $t,
                links: *const $t,
                inverse: *const $t,
                neighbours: *const u32,
                odd: *const u32,
                [n_coarse, dim, parity, minus, gamma5, vectors]: [i32; 6],
                ghost: *const $t,
                ghosts: i32,
            ) {
                $pass(
                    out, base, diag_of, hops_of, diag, links, inverse, neighbours, odd, n_coarse,
                    dim, parity, minus, gamma5, vectors, ghost, ghosts, STREAM,
                )
            }
            unsafe fn coarse_gather(
                buf: *mut $t,
                field: *const $t,
                sites: *const u32,
                n: i32,
                n_coarse: i32,
                dim: i32,
                vectors: i32,
            ) {
                $gather(buf, field, sites, n, n_coarse, dim, vectors, STREAM)
            }
            unsafe fn coarse_place(
                ghost: *mut $t,
                buf: *const $t,
                first: i32,
                n: i32,
                ghosts: i32,
                dim: i32,
                vectors: i32,
            ) {
                $place(ghost, buf, first, n, ghosts, dim, vectors, STREAM)
            }
            unsafe fn hop_part(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                from: *const $t,
                clover: *const $t,
                geometry: *const Geometry,
                kappa: f64,
                [mass, b5, c5]: [f64; 3],
                dagger: i32,
                target_parity: i32,
                vectors: i32,
                halo: *const HaloArgs,
                part: i32,
            ) {
                $part(
                    out,
                    inp,
                    gauge,
                    from,
                    clover,
                    geometry,
                    kappa,
                    mass,
                    b5,
                    c5,
                    dagger,
                    target_parity,
                    vectors,
                    halo,
                    part,
                    STREAM,
                )
            }
            unsafe fn dslash_part(
                out: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                params: *const MdwfKernelParams,
                vectors: i32,
                halo: *const HaloArgs,
                part: i32,
            ) {
                $dslash(
                    out, inp, gauge, geometry, params, vectors, halo, part, STREAM,
                )
            }
            unsafe fn hop_pack(
                send: *mut $t,
                inp: *const $t,
                gauge: *const $t,
                geometry: *const Geometry,
                [mass, b5, c5]: [f64; 3],
                dagger: i32,
                in_parity: i32,
                face: i32,
                vectors: i32,
            ) {
                $pack(
                    send, inp, gauge, geometry, mass, b5, c5, dagger, in_parity, face, vectors,
                    STREAM,
                )
            }
            fn host_exchange(comm: &dyn crate::Comm, spec: &HaloSpec) -> Box<dyn HaloExchange<$t>> {
                comm.$halo(spec)
            }
        }
    };
}

split_real!(f32: bridge_cuda_hop_part_f32, bridge_cuda_mdwf_dslash_part_f32,
    bridge_cuda_hop_pack_f32, halo_f32, 7, bridge_cuda_hop_split_halo_f32,
    bridge_cuda_coarse_pass_ghost_f32, bridge_cuda_coarse_gather_f32, bridge_cuda_coarse_place_f32);
split_real!(f64: bridge_cuda_hop_part_f64, bridge_cuda_mdwf_dslash_part_f64,
    bridge_cuda_hop_pack_f64, halo_f64, 8, bridge_cuda_hop_split_halo_f64,
    bridge_cuda_coarse_pass_ghost_f64, bridge_cuda_coarse_gather_f64, bridge_cuda_coarse_place_f64);

/// NCCL between the devices of the processes.
#[cfg(feature = "nccl")]
mod nccl {
    use super::*;
    use std::os::raw::c_char;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct UniqueId {
        internal: [c_char; 128],
    }

    extern "C" {
        fn ncclGetUniqueId(id: *mut UniqueId) -> i32;
        fn ncclCommInitRank(comm: *mut *mut c_void, nranks: i32, id: UniqueId, rank: i32) -> i32;
        fn ncclCommDestroy(comm: *mut c_void) -> i32;
        fn ncclGroupStart() -> i32;
        fn ncclGroupEnd() -> i32;
        fn ncclSend(
            buf: *const c_void,
            count: usize,
            datatype: i32,
            peer: i32,
            comm: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        fn ncclRecv(
            buf: *mut c_void,
            count: usize,
            datatype: i32,
            peer: i32,
            comm: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }

    fn nccl_check(code: i32, what: &str) {
        assert_eq!(code, 0, "NCCL: {what} returned {code}");
    }

    /// The NCCL communicator of the processes of a lattice, the stream the faces travel on, and
    /// the events between it and the default stream.
    pub(in super::super) struct Nccl {
        comm: *mut c_void,
        stream: *mut c_void,
        packed: *mut c_void,
        arrived: *mut c_void,
    }

    impl Drop for Nccl {
        fn drop(&mut self) {
            // SAFETY: the communicator made in `new`, destroyed once.
            unsafe { ncclCommDestroy(self.comm) };
        }
    }

    thread_local! {
        /// The communicator of this thread's process, made once for (rank, size).
        static MADE: RefCell<Option<((usize, usize), Option<Rc<Nccl>>)>> = const { RefCell::new(None) };
    }

    impl Nccl {
        /// The communicator of the processes of `comm` (every one of them calls this), none if
        /// NCCL fails on any of them (several processes on one device, say): then all of them
        /// go through the host.
        pub(in super::super) fn of(comm: &dyn crate::Comm) -> Option<Rc<Self>> {
            let key = (comm.rank(), comm.size());
            MADE.with(|made| {
                let mut made = made.borrow_mut();
                if let Some((k, nccl)) = made.as_ref() {
                    if *k == key {
                        return nccl.clone();
                    }
                }
                let nccl = Self::new(comm).map(Rc::new);
                *made = Some((key, nccl.clone()));
                nccl
            })
        }

        fn new(comm: &dyn crate::Comm) -> Option<Self> {
            // The identifier of rank 0 to every process, a byte in each real of a sum.
            let mut id = UniqueId { internal: [0; 128] };
            let mut failed = 0.0;
            if comm.rank() == 0 {
                // SAFETY: a place for the identifier.
                failed = f64::from(u8::from(unsafe { ncclGetUniqueId(&mut id) } != 0));
            }
            let mut bytes: Vec<f64> = id.internal.iter().map(|&b| f64::from(b as u8)).collect();
            bytes.push(failed);
            comm.sum(&mut bytes);
            if bytes[128] != 0.0 {
                return None;
            }
            for (b, x) in id.internal.iter_mut().zip(&bytes) {
                *b = *x as u8 as c_char;
            }
            let mut raw: *mut c_void = std::ptr::null_mut();
            // SAFETY: a place for the communicator; every process passes the same identifier.
            let code =
                unsafe { ncclCommInitRank(&mut raw, comm.size() as i32, id, comm.rank() as i32) };
            let mut failed = [f64::from(u8::from(code != 0))];
            comm.sum(&mut failed);
            if failed[0] != 0.0 {
                if code == 0 {
                    // SAFETY: made just above.
                    unsafe { ncclCommDestroy(raw) };
                }
                // SAFETY: no arguments; NCCL's failure is not an error of the kernels.
                unsafe { cudaGetLastError() };
                return None;
            }
            let (mut stream, mut packed, mut arrived) = (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            // SAFETY: places for the handles; 1 is cudaStreamNonBlocking (the stream does not
            // wait for the default one) and 2 cudaEventDisableTiming.
            unsafe {
                must(check(cudaStreamCreateWithFlags(&mut stream, 1)));
                must(check(cudaEventCreateWithFlags(&mut packed, 2)));
                must(check(cudaEventCreateWithFlags(&mut arrived, 2)));
            }
            Some(Self {
                comm: raw,
                stream,
                packed,
                arrived,
            })
        }

        /// The faces from `send` to the neighbours, into `recv` from them, on the stream of the
        /// exchange once the default stream has packed them; the default stream waits for them
        /// from `wait` on, which is called after the work that does not need them is launched.
        pub(in super::super) fn exchange<T: SplitReal>(
            &self,
            send: &[Option<DeviceBuffer<T>>],
            recv: &[Option<DeviceBuffer<T>>],
            neighbour: &[usize; FACES],
        ) {
            // SAFETY: the buffers are those of the faces, of the same length on both sides
            // (`Halos::exchange`); the handles are this communicator's.
            unsafe {
                must(check(cudaEventRecord(self.packed, STREAM)));
                must(check(cudaStreamWaitEvent(self.stream, self.packed, 0)));
                nccl_check(ncclGroupStart(), "ncclGroupStart");
                // The sends in the order of the faces, the receives in that of the faces they
                // left by (what arrives on face f left the neighbour by the opposite face), so
                // that two messages between the same two processes (two processes along a
                // direction) are matched in the same order on both sides.
                for (face, buf) in send.iter().enumerate() {
                    if let Some(b) = buf {
                        let peer = neighbour[face] as i32;
                        let p = b.ptr.cast_const().cast();
                        nccl_check(
                            ncclSend(p, b.len, T::NCCL_TYPE, peer, self.comm, self.stream),
                            "ncclSend",
                        );
                    }
                }
                for left_by in 0..FACES {
                    let face = crate::comm::opposite_face(left_by);
                    if let Some(b) = &recv[face] {
                        let peer = neighbour[face] as i32;
                        nccl_check(
                            ncclRecv(
                                b.ptr.cast(),
                                b.len,
                                T::NCCL_TYPE,
                                peer,
                                self.comm,
                                self.stream,
                            ),
                            "ncclRecv",
                        );
                    }
                }
                nccl_check(ncclGroupEnd(), "ncclGroupEnd");
                must(check(cudaEventRecord(self.arrived, self.stream)));
            }
        }

        /// The default stream waits for the faces.
        pub(in super::super) fn wait(&self) {
            // SAFETY: the handles are this communicator's.
            must(check(unsafe {
                cudaStreamWaitEvent(STREAM, self.arrived, 0)
            }));
        }
    }
}

/// How the faces travel.
enum Transport<T> {
    /// Through the host, by `Comm`.
    Host(Box<dyn HaloExchange<T>>),
    /// From device to device.
    #[cfg(feature = "nccl")]
    Nccl(Rc<nccl::Nccl>),
}

/// The transport of the exchanges `spec` describes on the processes of `lat`: NCCL where every
/// process can have it (feature `nccl`), the host otherwise; `RUTTICE_HALO=host` or `nccl`
/// chooses. Every process calls it at the same point.
fn transport_for<T: DeviceReal>(lat: &Lattice, spec: &HaloSpec) -> Transport<T> {
    let wanted = std::env::var("RUTTICE_HALO").unwrap_or_default();
    #[cfg(feature = "nccl")]
    {
        let nccl = match wanted.as_str() {
            "host" => None,
            "nccl" => Some(
                nccl::Nccl::of(lat.comm.as_ref())
                    .expect("RUTTICE_HALO=nccl, but NCCL cannot be had on every process"),
            ),
            _ => nccl::Nccl::of(lat.comm.as_ref()),
        };
        if let Some(nccl) = nccl {
            return Transport::Nccl(nccl);
        }
    }
    assert!(
        wanted != "nccl" || cfg!(feature = "nccl"),
        "RUTTICE_HALO=nccl needs a program built with the feature nccl"
    );
    Transport::Host(T::host_exchange(lat.comm.as_ref(), spec))
}

/// The faces `send` to the neighbours and `recv` from them, `between` launched while they
/// travel; when this returns, what is launched next on the default stream sees them.
fn travel<T: DeviceReal>(
    transport: &mut Transport<T>,
    send: &[Option<DeviceBuffer<T>>],
    recv: &[Option<DeviceBuffer<T>>],
    #[allow(unused_variables)] neighbour: &[usize; FACES],
    between: impl FnOnce(),
) {
    match transport {
        #[cfg(feature = "nccl")]
        Transport::Nccl(nccl) => {
            nccl.exchange(send, recv, neighbour);
            between();
            nccl.wait();
        }
        Transport::Host(host) => {
            for (face, buf) in send.iter().enumerate() {
                if let Some(send) = buf {
                    let to = host.send_buffer(face);
                    assert_eq!(to.len(), send.len);
                    // SAFETY: both ranges hold `len` reals of T; the copy waits for what was
                    // launched before (the packing).
                    must(check(unsafe {
                        cudaMemcpy(
                            to.as_mut_ptr().cast(),
                            send.ptr.cast_const().cast(),
                            send.bytes(),
                            DEVICE_TO_HOST,
                        )
                    }));
                }
            }
            host.start();
            between();
            host.wait();
            for (face, buf) in recv.iter().enumerate() {
                if let Some(recv) = buf {
                    let from = host.recv_buffer(face);
                    assert_eq!(from.len(), recv.len);
                    // SAFETY: as above, the other way; in the order of the default stream,
                    // after what `between` launched.
                    must(check(unsafe {
                        cudaMemcpy(
                            recv.ptr.cast(),
                            from.as_ptr().cast(),
                            recv.bytes(),
                            HOST_TO_DEVICE,
                        )
                    }));
                }
            }
        }
    }
}

/// The ranks on the other side of the faces of the local lattice of `lat`.
pub(super) fn neighbours(lat: &Lattice) -> [usize; FACES] {
    Stencil::nearest(lat, [Boundary::Periodic; ND])
        .halo_spec(1)
        .neighbour
}

/// One exchange of buffers of the device between the processes of `lat`, as `travel`, with
/// nothing between (for what is exchanged once in a while: the layers of an extended field of
/// links). A face with no buffer is not exchanged.
pub(super) fn exchange_now<T: DeviceReal>(
    lat: &Lattice,
    send: &[Option<DeviceBuffer<T>>],
    recv: &[Option<DeviceBuffer<T>>],
) {
    let neighbour = neighbours(lat);
    let spec = HaloSpec {
        neighbour,
        len: core::array::from_fn(|face| send[face].as_ref().map_or(0, |b| b.len)),
    };
    let mut transport = transport_for::<T>(lat, &spec);
    travel(&mut transport, send, recv, &neighbour, || {});
}

/// Buffers of the device kept for the exchanges of fields of one shape (the faces of a coarse
/// field of some number of vectors): made once, exchanged as often as the field changes.
pub(super) struct Faces<T> {
    pub(super) send: Vec<Option<DeviceBuffer<T>>>,
    pub(super) recv: Vec<Option<DeviceBuffer<T>>>,
    transport: Transport<T>,
    neighbour: [usize; FACES],
}

impl<T: DeviceReal> Faces<T> {
    /// Buffers of `len[face]` reals for each face (none for 0) between the processes of `lat`.
    pub(super) fn new(lat: &Lattice, len: [usize; FACES]) -> Self {
        let neighbour = neighbours(lat);
        let buffers = || -> Vec<Option<DeviceBuffer<T>>> {
            len.iter()
                .map(|&n| (n > 0).then(|| must(DeviceBuffer::zeros(n))))
                .collect()
        };
        Self {
            send: buffers(),
            recv: buffers(),
            transport: transport_for::<T>(lat, &HaloSpec { neighbour, len }),
            neighbour,
        }
    }

    /// What is in `send` to the neighbours, into `recv` from them.
    pub(super) fn exchange(&mut self) {
        travel(
            &mut self.transport,
            &self.send,
            &self.recv,
            &self.neighbour,
            || {},
        );
    }
}

/// Site numbers on the device: the surface of an output.
struct DeviceSites {
    ptr: *mut i32,
    len: usize,
}

impl DeviceSites {
    fn upload(sites: &[i32]) -> Result<Self, CudaError> {
        let bytes = std::mem::size_of_val(sites);
        let ptr = device_alloc(bytes)?.cast::<i32>();
        if bytes > 0 {
            // SAFETY: the device range was allocated with `bytes` bytes.
            check(unsafe { cudaMemcpy(ptr.cast(), sites.as_ptr().cast(), bytes, HOST_TO_DEVICE) })?;
        }
        Ok(Self {
            ptr,
            len: sites.len(),
        })
    }
}

impl Drop for DeviceSites {
    fn drop(&mut self) {
        // SAFETY: from `device_alloc` with this size, freed once.
        unsafe { device_free(self.ptr.cast(), self.len * std::mem::size_of::<i32>()) }
    }
}

/// The buffers of the exchanges of one output: its parity, the vectors and Ls of the fields,
/// and which of the fields exchanged together it is (the force exchanges two).
struct Exchange<T> {
    key: (Option<Parity>, usize, usize, usize),
    send: Vec<Option<DeviceBuffer<T>>>,
    recv: Vec<Option<DeviceBuffer<T>>>,
    surface: DeviceSites,
    transport: Transport<T>,
}

/// How the lattice of an operator is cut between processes, and the buffers of its exchanges.
pub(super) struct Halos<T> {
    lat: Arc<Lattice>,
    /// Directions cut between processes.
    cut: [bool; ND],
    /// The rank on the other side of each face.
    neighbour: [usize; FACES],
    /// What multiplies a leg that arrives on each face: the boundary of the whole lattice where
    /// the face is on it, 1 elsewhere.
    bc: [f64; FACES],
    exchanges: RefCell<Vec<Exchange<T>>>,
}

/// The factor of a boundary.
fn factor(b: Boundary) -> f64 {
    match b {
        Boundary::Periodic => 1.0,
        Boundary::Antiperiodic => -1.0,
        Boundary::Dirichlet => 0.0,
        Boundary::Phase(..) => {
            panic!("the CUDA kernels take periodic, antiperiodic and Dirichlet boundaries")
        }
    }
}

impl<T: DeviceReal> Halos<T> {
    /// None on one process. The local extents are even and at least 2 in every direction (so a
    /// site has the parity of its global coordinates, and a face is a layer of its own).
    pub(super) fn of(lat: &Arc<Lattice>, boundary: [Boundary; ND]) -> Option<Rc<Self>> {
        if lat.comm.size() == 1 {
            return None;
        }
        assert!(
            lat.local.iter().all(|l| l % 2 == 0),
            "the CUDA backend on several processes needs even local extents, not {:?}",
            lat.local
        );
        let cut = core::array::from_fn(|mu| lat.grid[mu] > 1);
        let bc = core::array::from_fn(|face| {
            let mu = face % ND;
            let edge = if face < ND {
                lat.origin[mu] + lat.local[mu] == lat.global[mu]
            } else {
                lat.origin[mu] == 0
            };
            if edge {
                factor(boundary[mu])
            } else {
                1.0
            }
        });
        Some(Rc::new(Self {
            lat: lat.clone(),
            cut,
            neighbour: neighbours(lat),
            bc,
            exchanges: RefCell::new(Vec::new()),
        }))
    }

    /// The sites of a face, on the whole lattice or on one parity.
    fn face_sites(&self, mu: usize, half: bool) -> usize {
        let all = self.lat.num_sites() / self.lat.local[mu];
        if half {
            all / 2
        } else {
            all
        }
    }

    /// The sites of the output (parity `to`, or the whole lattice) one of whose legs comes from
    /// another process, in the numbering of the output.
    fn surface(&self, to: Option<Parity>) -> Vec<i32> {
        let [nx, ny, nz, nt] = self.lat.local;
        let on = |x: [usize; ND]| {
            (0..ND).any(|mu| self.cut[mu] && (x[mu] == 0 || x[mu] + 1 == self.lat.local[mu]))
        };
        let mut sites = Vec::new();
        for t in 0..nt {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let number = match to {
                            None => x + nx * (y + ny * (z + nz * t)),
                            Some(p) if (x + y + z + t) % 2 == p.index() => {
                                x / 2 + (nx / 2) * (y + ny * (z + nz * t))
                            }
                            Some(_) => continue,
                        };
                        if on([x, y, z, t]) {
                            sites.push(number as i32);
                        }
                    }
                }
            }
        }
        sites
    }

    /// Index of the exchange for an output, made on first use.
    fn exchange(&self, to: Option<Parity>, vectors: usize, ls: usize, which: usize) -> usize {
        let key = (to, vectors, ls, which);
        let mut exchanges = self.exchanges.borrow_mut();
        if let Some(i) = exchanges.iter().position(|e| e.key == key) {
            return i;
        }
        let half = to.is_some();
        let len: [usize; FACES] = core::array::from_fn(|face| {
            let mu = face % ND;
            if self.cut[mu] {
                12 * ls * vectors * self.face_sites(mu, half)
            } else {
                0
            }
        });
        let buffers = || -> Vec<Option<DeviceBuffer<T>>> {
            len.iter()
                .map(|&n| (n > 0).then(|| must(DeviceBuffer::uninitialised(n))))
                .collect()
        };
        let spec = HaloSpec {
            neighbour: self.neighbour,
            len,
        };
        exchanges.push(Exchange {
            key,
            send: buffers(),
            recv: buffers(),
            surface: must(DeviceSites::upload(&self.surface(to))),
            transport: transport_for(&self.lat, &spec),
        });
        exchanges.len() - 1
    }

    /// Packs the faces of `inp` into the buffers of exchange `ex` and gives what the kernels
    /// take of it.
    #[allow(clippy::too_many_arguments)]
    fn pack(
        &self,
        ex: &Exchange<T>,
        geometry: &Geometry,
        gauge: *const T,
        inp: &DeviceSpinor<T>,
        to: Option<Parity>,
        fifth: [f64; 3],
        dagger: bool,
    ) -> HaloArgs {
        let in_parity = parity_number(to.map(Parity::flip));
        for face in 0..FACES {
            if let Some(send) = &ex.send[face] {
                // SAFETY: the buffer holds the face for these vectors and Ls; `inp` is on the
                // parity the kernel is told.
                unsafe {
                    T::hop_pack(
                        send.ptr,
                        inp.data.ptr,
                        gauge,
                        geometry,
                        fifth,
                        i32::from(dagger),
                        in_parity,
                        face as i32,
                        inp.vectors as i32,
                    );
                }
            }
        }
        let mut args = HaloArgs {
            recv: [std::ptr::null(); FACES],
            bc: self.bc,
            cut: self.cut.map(i32::from),
            surface: ex.surface.ptr,
            surface_sites: ex.surface.len as i32,
        };
        for face in 0..FACES {
            if let Some(recv) = &ex.recv[face] {
                args.recv[face] = recv.ptr.cast_const().cast();
            }
        }
        args
    }

    /// The hops `run(halo, part)` with the faces of `inp` exchanged around them: the faces packed
    /// (the legs of the hops `H (b − c Y)` of `fifth`, `dagger`), sent while the interior is
    /// computed, then the surface. `to`: the parity of the output (`inp` on the other), or the
    /// whole lattice.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn around(
        &self,
        geometry: &Geometry,
        gauge: *const T,
        inp: &DeviceSpinor<T>,
        to: Option<Parity>,
        fifth: [f64; 3],
        dagger: bool,
        mut run: impl FnMut(&HaloArgs, i32),
    ) {
        let i = self.exchange(to, inp.vectors, inp.ls, 0);
        let mut exchanges = self.exchanges.borrow_mut();
        let ex = &mut exchanges[i];
        let args = self.pack(ex, geometry, gauge, inp, to, fifth, dagger);
        travel(
            &mut ex.transport,
            &ex.send,
            &ex.recv,
            &self.neighbour,
            || run(&args, INTERIOR),
        );
        run(&args, SURFACE);
    }

    /// The faces of `inp` on the whole lattice exchanged, for a kernel that reads every leg of
    /// every site from them (the force): `which` tells apart the fields exchanged for one
    /// kernel, whose buffers it reads together.
    pub(super) fn exchanged(
        &self,
        geometry: &Geometry,
        gauge: *const T,
        inp: &DeviceSpinor<T>,
        fifth: [f64; 3],
        which: usize,
    ) -> HaloArgs {
        let i = self.exchange(None, inp.vectors, inp.ls, which);
        let mut exchanges = self.exchanges.borrow_mut();
        let ex = &mut exchanges[i];
        let args = self.pack(ex, geometry, gauge, inp, None, fifth, false);
        travel(
            &mut ex.transport,
            &ex.send,
            &ex.recv,
            &self.neighbour,
            || {},
        );
        args
    }
}
