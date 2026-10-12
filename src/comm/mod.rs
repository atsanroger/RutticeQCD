//! Communication between the processes that share a lattice.
//!
//! Kernels see two things: a global sum, and a non-blocking exchange of the eight faces of
//! the local lattice with the neighbouring processes. Writing a file adds a third: a gather
//! to rank 0, which writes it. Which library moves the data is hidden behind `Comm`.
//!
//! Implementations here: `SingleComm` (one process) and `LocalComm` (several ranks as
//! threads of one process, for tests and for development without an MPI installation).

use crate::lattice::{Lattice, ND};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Number of faces of the local lattice. Faces are numbered like stencil legs: direction
/// `mu` forward is `mu`, backward is `ND + mu`.
pub const FACES: usize = 2 * ND;

/// The face opposite to `face`.
#[inline(always)]
pub fn opposite_face(face: usize) -> usize {
    (face + ND) % FACES
}

/// What one exchange moves: for every face, the rank on the other side and the number of
/// reals sent (and received) there. A length of zero means the face is not exchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HaloSpec {
    pub neighbour: [usize; FACES],
    pub len: [usize; FACES],
}

/// A reusable set of send and receive buffers for the faces of the local lattice.
///
/// Data written to `send_buffer(face)` travels to the neighbour on that side and appears
/// there in `recv_buffer(opposite_face(face))`. `start` begins all transfers and returns
/// without waiting; the receive buffers are valid after `wait`.
pub trait HaloExchange<T>: Send {
    fn send_buffer(&mut self, face: usize) -> &mut [T];
    fn start(&mut self);
    fn wait(&mut self);
    fn recv_buffer(&self, face: usize) -> &[T];
}

pub trait Comm: Send + Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// Global reduction: replace each element with its sum over all processes. Every
    /// process gets the same bits.
    fn sum(&self, x: &mut [f64]);
    /// Gather to rank 0: rank 0 gets what every process passed, in rank order, and the
    /// others get nothing. The processes may pass different lengths, none at all too. The
    /// values arrive with their bits unchanged.
    fn gather(&self, x: &[f64]) -> Vec<Vec<f64>>;
    /// Buffers for a halo exchange of single-precision data.
    fn halo_f32(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f32>>;
    /// Buffers for a halo exchange of double-precision data.
    fn halo_f64(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f64>>;
}

/// Buffers that are never transferred: every face has length zero.
struct NoExchange;

impl<T> HaloExchange<T> for NoExchange {
    fn send_buffer(&mut self, _face: usize) -> &mut [T] {
        &mut []
    }
    fn start(&mut self) {}
    fn wait(&mut self) {}
    fn recv_buffer(&self, _face: usize) -> &[T] {
        &[]
    }
}

/// One process. Neighbours across the lattice boundary are local sites, so nothing is
/// ever exchanged.
pub struct SingleComm;

fn no_exchange(spec: &HaloSpec) -> NoExchange {
    assert!(
        spec.len.iter().all(|l| *l == 0),
        "a single process has no faces to exchange"
    );
    NoExchange
}

impl Comm for SingleComm {
    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        1
    }
    fn sum(&self, _x: &mut [f64]) {}
    fn gather(&self, x: &[f64]) -> Vec<Vec<f64>> {
        vec![x.to_vec()]
    }
    fn halo_f32(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f32>> {
        Box::new(no_exchange(spec))
    }
    fn halo_f64(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f64>> {
        Box::new(no_exchange(spec))
    }
}

/// (from, to, tag)
type Address = (usize, usize, u32);

/// Messages in transit between the ranks of a `LocalComm`.
struct Mailbox<T> {
    queues: Mutex<HashMap<Address, VecDeque<Vec<T>>>>,
    arrived: Condvar,
}

impl<T> Mailbox<T> {
    fn new() -> Self {
        Self {
            queues: Mutex::new(HashMap::new()),
            arrived: Condvar::new(),
        }
    }

    fn send(&self, from: usize, to: usize, tag: u32, data: Vec<T>) {
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        queues.entry((from, to, tag)).or_default().push_back(data);
        self.arrived.notify_all();
    }

    /// Blocks until the message is there. Panics if another rank has failed, so that one
    /// failing rank does not leave the others waiting forever.
    fn recv(&self, from: usize, to: usize, tag: u32, failed: &AtomicBool) -> Vec<T> {
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(data) = queues.get_mut(&(from, to, tag)).and_then(|q| q.pop_front()) {
                return data;
            }
            assert!(
                !failed.load(Ordering::SeqCst),
                "rank {to}: another rank failed while this one waited for rank {from}"
            );
            queues = self
                .arrived
                .wait_timeout(queues, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

struct Shared {
    size: usize,
    f32: Mailbox<f32>,
    f64: Mailbox<f64>,
    failed: AtomicBool,
}

/// Several ranks inside one process, one thread each. Behaves like a distributed run as
/// far as the code above `Comm` can tell: every rank has its own lattice and fields, and
/// data crosses only through halo exchanges and global sums.
pub struct LocalComm {
    rank: usize,
    shared: Arc<Shared>,
}

/// Tag of the messages `LocalComm::sum` sends; halo exchanges use the face number.
const SUM_TAG: u32 = u32::MAX;
/// Tag of the messages `LocalComm::gather` sends.
const GATHER_TAG: u32 = u32::MAX - 1;

/// Marks the run as failed when a rank's thread unwinds.
struct FailureFlag(Arc<Shared>);

impl Drop for FailureFlag {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.failed.store(true, Ordering::SeqCst);
        }
    }
}

impl LocalComm {
    /// Split a lattice of size `global` over a process grid, run `f` once per rank with
    /// that rank's lattice, each on its own thread, and return the results in rank order.
    pub fn run<R, F>(global: [usize; ND], grid: [usize; ND], f: F) -> Vec<R>
    where
        R: Send,
        F: Fn(Arc<Lattice>) -> R + Sync,
    {
        let size = grid.iter().product();
        let shared = Arc::new(Shared {
            size,
            f32: Mailbox::new(),
            f64: Mailbox::new(),
            failed: AtomicBool::new(false),
        });
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..size)
                .map(|rank| {
                    let (shared, f) = (shared.clone(), &f);
                    scope.spawn(move || {
                        let _flag = FailureFlag(shared.clone());
                        let comm = Arc::new(LocalComm { rank, shared });
                        f(Lattice::new(global, grid, comm))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(r) => r,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }
}

impl Comm for LocalComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    /// Every rank adds the contributions in rank order, so all of them get the same bits.
    fn sum(&self, x: &mut [f64]) {
        let (me, shared) = (self.rank, &self.shared);
        for to in (0..shared.size).filter(|r| *r != me) {
            shared.f64.send(me, to, SUM_TAG, x.to_vec());
        }
        let mine = x.to_vec();
        x.fill(0.0);
        for from in 0..shared.size {
            let part = if from == me {
                mine.clone()
            } else {
                shared.f64.recv(from, me, SUM_TAG, &shared.failed)
            };
            assert_eq!(
                part.len(),
                x.len(),
                "ranks disagree on what is being summed"
            );
            for (t, p) in x.iter_mut().zip(part) {
                *t += p;
            }
        }
    }

    /// The other ranks do not wait for rank 0: what they send queues up until it is taken.
    fn gather(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (me, shared) = (self.rank, &self.shared);
        if me != 0 {
            shared.f64.send(me, 0, GATHER_TAG, x.to_vec());
            return Vec::new();
        }
        (0..shared.size)
            .map(|from| {
                if from == 0 {
                    x.to_vec()
                } else {
                    shared.f64.recv(from, 0, GATHER_TAG, &shared.failed)
                }
            })
            .collect()
    }

    fn halo_f32(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f32>> {
        Box::new(LocalExchange::new(self, spec, |s| &s.f32))
    }

    fn halo_f64(&self, spec: &HaloSpec) -> Box<dyn HaloExchange<f64>> {
        Box::new(LocalExchange::new(self, spec, |s| &s.f64))
    }
}

struct LocalExchange<T> {
    rank: usize,
    shared: Arc<Shared>,
    mailbox: fn(&Shared) -> &Mailbox<T>,
    neighbour: [usize; FACES],
    send: [Vec<T>; FACES],
    recv: [Vec<T>; FACES],
}

impl<T: Copy + Default> LocalExchange<T> {
    fn new(comm: &LocalComm, spec: &HaloSpec, mailbox: fn(&Shared) -> &Mailbox<T>) -> Self {
        Self {
            rank: comm.rank,
            shared: comm.shared.clone(),
            mailbox,
            neighbour: spec.neighbour,
            send: core::array::from_fn(|face| vec![T::default(); spec.len[face]]),
            recv: core::array::from_fn(|face| vec![T::default(); spec.len[face]]),
        }
    }
}

impl<T: Copy + Default + Send> HaloExchange<T> for LocalExchange<T> {
    fn send_buffer(&mut self, face: usize) -> &mut [T] {
        &mut self.send[face]
    }

    fn start(&mut self) {
        let mailbox = (self.mailbox)(&self.shared);
        for face in 0..FACES {
            if self.send[face].is_empty() {
                continue;
            }
            let tag = u32::try_from(face).expect("there are eight faces");
            mailbox.send(
                self.rank,
                self.neighbour[face],
                tag,
                self.send[face].clone(),
            );
        }
    }

    fn wait(&mut self) {
        let mailbox = (self.mailbox)(&self.shared);
        for face in 0..FACES {
            if self.recv[face].is_empty() {
                continue;
            }
            // What arrives on this face left the neighbour through its opposite face.
            let tag = u32::try_from(opposite_face(face)).expect("there are eight faces");
            let data = mailbox.recv(self.neighbour[face], self.rank, tag, &self.shared.failed);
            assert_eq!(
                data.len(),
                self.recv[face].len(),
                "neighbours disagree on a face"
            );
            self.recv[face] = data;
        }
    }

    fn recv_buffer(&self, face: usize) -> &[T] {
        &self.recv[face]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_gives_every_rank_the_same_total() {
        let totals = LocalComm::run([4, 4, 4, 8], [2, 1, 1, 2], |lat| {
            let r = lat.comm.rank() as f64;
            let mut x = [r + 0.1, 1.0, -r];
            lat.comm.sum(&mut x);
            lat.comm.sum(&mut x);
            x
        });
        assert_eq!(totals.len(), 4);
        for t in &totals {
            assert_eq!(*t, totals[0]);
            assert!((t[0] - 4.0 * 6.4).abs() < 1e-12);
            assert_eq!(t[1], 16.0);
            assert_eq!(t[2], -24.0);
        }
    }

    #[test]
    fn faces_arrive_on_the_opposite_side() {
        // Two ranks along t: each rank's forward and backward neighbour is the other one.
        let got = LocalComm::run([2, 2, 2, 4], [1, 1, 1, 2], |lat| {
            let me = lat.comm.rank();
            let mut spec = HaloSpec {
                neighbour: [me; FACES],
                len: [0; FACES],
            };
            for face in [3, ND + 3] {
                spec.neighbour[face] = 1 - me;
                spec.len[face] = 2;
            }
            let mut ex = lat.comm.halo_f64(&spec);
            for round in 0..2 {
                ex.send_buffer(3).fill((10 * me + round) as f64);
                ex.send_buffer(ND + 3).fill((10 * me + round) as f64 + 0.5);
                ex.start();
                ex.wait();
                let other = (10 * (1 - me) + round) as f64;
                assert_eq!(ex.recv_buffer(3), [other + 0.5; 2]);
                assert_eq!(ex.recv_buffer(ND + 3), [other; 2]);
            }
            me
        });
        assert_eq!(got, [0, 1]);
    }

    #[test]
    fn gather_brings_every_rank_to_rank_0_in_order() {
        let got = LocalComm::run([4, 4, 4, 8], [1, 1, 2, 2], |lat| {
            let r = lat.comm.rank();
            // Rank 2 passes nothing; -0.0 must keep its sign.
            let x: Vec<f64> = (0..r % 2 + usize::from(r != 2))
                .map(|i| if r == 3 { -0.0 } else { (10 * r + i) as f64 })
                .collect();
            (lat.comm.gather(&x), lat.comm.gather(&[r as f64]))
        });
        assert_eq!(
            got[0].0,
            [vec![0.0], vec![10.0, 11.0], vec![], vec![-0.0, -0.0]]
        );
        assert!(got[0].0[3].iter().all(|x| x.is_sign_negative()));
        assert_eq!(got[0].1, [[0.0], [1.0], [2.0], [3.0]]);
        for (first, second) in &got[1..] {
            assert!(first.is_empty() && second.is_empty());
        }
    }

    #[test]
    #[should_panic]
    fn a_failing_rank_does_not_hang_the_others() {
        LocalComm::run([2, 2, 2, 4], [1, 1, 1, 2], |lat| {
            assert_eq!(lat.comm.rank(), 0, "rank 1 fails on purpose");
            let mut x = [1.0];
            lat.comm.sum(&mut x);
        });
    }
}
