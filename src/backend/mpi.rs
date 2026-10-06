//! `Comm` on top of an MPI library. Built with the feature `mpi`, which needs an MPI
//! installation and libclang.
//!
//! Initialisation and finalisation go through the `mpi` crate. The halo exchange and the
//! global sum call the C interface directly: the requests of an exchange stay alive from
//! `start` to `wait`, across two method calls, which the scoped requests of the `mpi`
//! crate cannot express.
//!
//! MPI is initialised for serialised calls from several threads, and every call made here
//! takes one process-wide lock, so the objects of this module can be moved between threads.

use crate::comm::{opposite_face, Comm, HaloExchange, HaloSpec, FACES};
use ::mpi::ffi;
use ::mpi::topology::Communicator;
use ::mpi::traits::AsRaw;
use ::mpi::Threading;
use std::os::raw::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

/// The world communicator, and the lock that serialises the calls made through it.
#[derive(Clone)]
struct World {
    raw: ffi::MPI_Comm,
    calls: Arc<Mutex<()>>,
}

// SAFETY: `raw` is the handle of MPI_COMM_WORLD, which stays valid from initialisation to
// finalisation and is not changed by anything here. MPI was initialised with at least
// `Threading::Serialized` (checked in `run`), and every call through the handle is made
// while holding `calls`, so calls from different threads never overlap.
unsafe impl Send for World {}
// SAFETY: as above; shared access only ever leads to calls under the lock.
unsafe impl Sync for World {}

impl World {
    fn lock(&self) -> MutexGuard<'_, ()> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn check(code: c_int, what: &str) {
    assert_eq!(code, ffi::MPI_SUCCESS as c_int, "{what} failed");
}

fn count(len: usize) -> c_int {
    c_int::try_from(len).expect("a message has more elements than MPI can count")
}

pub struct MpiComm {
    rank: usize,
    size: usize,
    world: World,
}

/// Initialise MPI, call `f` with the communicator of all processes, and finalise.
///
/// Everything that holds the communicator (lattices, fields, operators) must be dropped
/// before `f` returns. If `f` panics, all processes are aborted.
pub fn run<R>(f: impl FnOnce(Arc<MpiComm>) -> R) -> R {
    let (universe, threading) = mpi::initialize_with_threading(Threading::Serialized)
        .expect("MPI has already been initialised");
    assert!(
        matches!(threading, Threading::Serialized | Threading::Multiple),
        "the MPI library does not support calls from more than one thread"
    );
    let world = universe.world();
    let comm = Arc::new(MpiComm {
        rank: usize::try_from(world.rank()).expect("MPI ranks are not negative"),
        size: usize::try_from(world.size()).expect("MPI sizes are not negative"),
        world: World {
            raw: world.as_raw(),
            calls: Arc::new(Mutex::new(())),
        },
    });
    match catch_unwind(AssertUnwindSafe(|| f(comm.clone()))) {
        Ok(result) => {
            assert_eq!(
                Arc::strong_count(&comm),
                1,
                "something still holds the communicator at finalisation"
            );
            drop(comm);
            drop(universe);
            result
        }
        // The panic message has been printed already. The other processes may be waiting
        // for this one, so take them all down.
        Err(_) => world.abort(1),
    }
}

impl Comm for MpiComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn sum(&self, x: &mut [f64]) {
        let mine = x.to_vec();
        let _serialised = self.world.lock();
        // SAFETY: both buffers hold `x.len()` doubles and outlive the call, which blocks
        // until the reduction is complete; they do not overlap.
        let code = unsafe {
            ffi::MPI_Allreduce(
                mine.as_ptr().cast::<c_void>(),
                x.as_mut_ptr().cast::<c_void>(),
                count(x.len()),
                ffi::RSMPI_DOUBLE,
                ffi::RSMPI_SUM,
                self.world.raw,
            )
        };
        check(code, "MPI_Allreduce");
    }

    fn halo_f32(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f32>> {
        // SAFETY: reading a constant handle of the MPI library.
        Box::new(MpiExchange::new(self, spec, unsafe { ffi::RSMPI_FLOAT }))
    }

    fn halo_f64(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f64>> {
        // SAFETY: reading a constant handle of the MPI library.
        Box::new(MpiExchange::new(self, spec, unsafe { ffi::RSMPI_DOUBLE }))
    }
}

/// Send and receive buffers for the eight faces, and the requests of a transfer that is
/// under way.
struct MpiExchange<T> {
    world: World,
    datatype: ffi::MPI_Datatype,
    neighbour: [c_int; FACES],
    send: [Vec<T>; FACES],
    recv: [Vec<T>; FACES],
    /// Not empty exactly between `start` and `wait`. While it is not, MPI may read the
    /// send buffers and write the receive buffers at any time.
    requests: Vec<ffi::MPI_Request>,
}

// SAFETY: the raw handles (datatype, requests) are only passed to MPI under the lock in
// `world`; the buffers are owned vectors that are never resized.
unsafe impl<T: Send> Send for MpiExchange<T> {}

impl<T: Copy + Default> MpiExchange<T> {
    /// `datatype` must be the MPI datatype of `T`.
    fn new(comm: &MpiComm, spec: &HaloSpec, datatype: ffi::MPI_Datatype) -> Self {
        Self {
            world: comm.world.clone(),
            datatype,
            neighbour: spec
                .neighbour
                .map(|rank| c_int::try_from(rank).expect("MPI ranks fit in an int")),
            send: core::array::from_fn(|face| vec![T::default(); spec.len[face]]),
            recv: core::array::from_fn(|face| vec![T::default(); spec.len[face]]),
            requests: Vec::with_capacity(2 * FACES),
        }
    }

    fn finish(&mut self) {
        if self.requests.is_empty() {
            return;
        }
        let _serialised = self.world.lock();
        // SAFETY: `requests` holds the live requests created by `start`, and the buffers
        // they refer to are still in place. After the call all of them are complete.
        let code = unsafe {
            ffi::MPI_Waitall(
                count(self.requests.len()),
                self.requests.as_mut_ptr(),
                ffi::RSMPI_STATUSES_IGNORE,
            )
        };
        check(code, "MPI_Waitall");
        self.requests.clear();
    }
}

impl<T: Copy + Default + Send> HaloExchange<T> for MpiExchange<T> {
    fn send_buffer(&mut self, face: usize) -> &mut [T] {
        assert!(self.requests.is_empty(), "the exchange is under way");
        &mut self.send[face]
    }

    fn start(&mut self) {
        assert!(
            self.requests.is_empty(),
            "the exchange is already under way"
        );
        let _serialised = self.world.lock();
        for face in 0..FACES {
            if self.recv[face].is_empty() {
                continue;
            }
            // What arrives on a face left the neighbour through its opposite face.
            let tag = count(opposite_face(face));
            // SAFETY: a zeroed request handle is a valid value to be overwritten by MPI.
            let mut request: ffi::MPI_Request = unsafe { core::mem::zeroed() };
            // SAFETY: the buffer holds `len` elements of the datatype. It is not moved,
            // resized, read or written until `finish` has waited for the request: the
            // accessors refuse while `requests` is not empty, and `drop` waits.
            let code = unsafe {
                ffi::MPI_Irecv(
                    self.recv[face].as_mut_ptr().cast::<c_void>(),
                    count(self.recv[face].len()),
                    self.datatype,
                    self.neighbour[face],
                    tag,
                    self.world.raw,
                    &mut request,
                )
            };
            check(code, "MPI_Irecv");
            self.requests.push(request);
        }
        for face in 0..FACES {
            if self.send[face].is_empty() {
                continue;
            }
            // SAFETY: as for the receive above.
            let mut request: ffi::MPI_Request = unsafe { core::mem::zeroed() };
            // SAFETY: as for the receive above; this buffer is only read by MPI.
            let code = unsafe {
                ffi::MPI_Isend(
                    self.send[face].as_ptr().cast::<c_void>(),
                    count(self.send[face].len()),
                    self.datatype,
                    self.neighbour[face],
                    count(face),
                    self.world.raw,
                    &mut request,
                )
            };
            check(code, "MPI_Isend");
            self.requests.push(request);
        }
    }

    fn wait(&mut self) {
        self.finish();
    }

    fn recv_buffer(&self, face: usize) -> &[T] {
        assert!(self.requests.is_empty(), "the exchange is under way");
        &self.recv[face]
    }
}

impl<T> Drop for MpiExchange<T> {
    fn drop(&mut self) {
        // MPI must not be left holding pointers into buffers that are about to be freed.
        if !self.requests.is_empty() {
            let _serialised = self.world.lock();
            // SAFETY: as in `finish`.
            unsafe {
                ffi::MPI_Waitall(
                    count(self.requests.len()),
                    self.requests.as_mut_ptr(),
                    ffi::RSMPI_STATUSES_IGNORE,
                );
            }
        }
    }
}
