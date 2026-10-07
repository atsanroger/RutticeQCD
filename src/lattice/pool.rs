//! Threads for the loops over sites, and how a loop is dealt out to them.
//!
//! A loop over the sites of a lattice does the same work at every site. Three ways of
//! running one in parallel are here, chosen by the environment variable `RUTTICE_SCHEDULE`
//! so that they can be compared on the same program:
//!
//! - `steal`: the loop is cut into tasks that the threads of the rayon pool take from each
//!   other (the default so far).
//! - `static`: every thread of the rayon pool gets one fixed piece (`rayon::broadcast`).
//! - `pool`: the same fixed pieces on the threads of `Pool`, a small pool of its own whose
//!   threads are pinned to one processor each and, when idle, spin for a while before they
//!   sleep. This is what an OpenMP runtime does with a static schedule, bound threads and
//!   an active wait policy.
//!
//! `RUTTICE_PIN=1` also pins the threads of the rayon pool, one processor each.
//!
//! The number of threads is that of the rayon pool in every case (`RAYON_NUM_THREADS`).

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// How loops over sites are dealt out to threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Steal,
    Static,
    Pool,
}

/// The schedule in use: `RUTTICE_SCHEDULE`, read once.
pub fn schedule() -> Schedule {
    static VALUE: OnceLock<Schedule> = OnceLock::new();
    *VALUE.get_or_init(|| match std::env::var("RUTTICE_SCHEDULE").as_deref() {
        Ok("static") => Schedule::Static,
        Ok("pool") => Schedule::Pool,
        _ => Schedule::Steal,
    })
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// Prepare the threads before the first loop: with `RUTTICE_PIN=1`, start the rayon pool
/// with its threads pinned. Called when a lattice is made; later calls do nothing.
pub fn setup() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if flag("RUTTICE_PIN") {
            let cpus = allowed_cpus();
            if !cpus.is_empty() {
                // Fails if the pool exists already; the threads then stay as they are.
                let _ = rayon::ThreadPoolBuilder::new()
                    .start_handler(move |index| pin_to(cpus[index % cpus.len()]))
                    .build_global();
            }
        }
    });
}

/// The processors this thread may run on, in increasing order. Empty where that cannot be
/// asked.
pub fn allowed_cpus() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: an all-zero `cpu_set_t` is a valid, empty set. `sched_getaffinity` writes
        // at most the size it is given into the set it is given; pid 0 is the caller.
        let set = unsafe {
            let mut set: libc::cpu_set_t = core::mem::zeroed();
            if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return Vec::new();
            }
            set
        };
        let limit = usize::try_from(libc::CPU_SETSIZE).unwrap_or(0);
        // SAFETY: `CPU_ISSET` reads the set for an index below `CPU_SETSIZE`.
        (0..limit)
            .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
            .collect()
    }
    #[cfg(not(target_os = "linux"))]
    Vec::new()
}

/// Bind the calling thread to one processor. Does nothing where that cannot be done.
pub fn pin_to(cpu: usize) {
    #[cfg(target_os = "linux")]
    // SAFETY: as in `allowed_cpus`; `CPU_SET` writes the set for an index below
    // `CPU_SETSIZE` (checked), and `sched_setaffinity` only reads the set.
    unsafe {
        if cpu < usize::try_from(libc::CPU_SETSIZE).unwrap_or(0) {
            let mut set: libc::cpu_set_t = core::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// The body of a loop, as the threads of the pool see it: called with the number of the
/// thread.
type Body<'a> = &'a (dyn Fn(usize) + Sync);

/// A pointer to the body of the loop under way. The lifetime is erased; `Pool::run` keeps
/// the body alive until every thread is done with it.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync));

// SAFETY: the pointee is `Sync`, and the pointer is only ever turned back into a shared
// reference.
unsafe impl Send for Job {}

struct Shared {
    /// Counts the loops started; a change tells the threads there is a new one.
    epoch: AtomicUsize,
    job: Mutex<Option<Job>>,
    /// Threads that have not finished the loop under way.
    busy: AtomicUsize,
    panicked: AtomicBool,
    /// Held while a thread decides to sleep and while it is woken.
    sleep: Mutex<()>,
    wake: Condvar,
}

/// A pool of threads for loops with one piece per thread. The calling thread is thread 0
/// of every loop; the others wait for loops, spinning for a while before they sleep.
pub struct Pool {
    shared: Arc<Shared>,
    threads: usize,
    /// One loop at a time.
    running: Mutex<()>,
}

thread_local! {
    /// Set on the threads of the pool, which must not start loops themselves.
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

/// How long an idle thread looks for a new loop before it sleeps, in turns of a spin
/// loop: a few hundred microseconds. `RUTTICE_POOL_SPIN` overrides it.
const SPIN_TURNS: usize = 100_000;

impl Pool {
    /// The pool of this process, with as many threads as the rayon pool has.
    pub fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool::new(rayon::current_num_threads()))
    }

    fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            job: Mutex::new(None),
            busy: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
        });
        let cpus = allowed_cpus();
        let spin = std::env::var("RUTTICE_POOL_SPIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(SPIN_TURNS);
        if let Some(cpu) = cpus.first() {
            pin_to(*cpu);
        }
        for index in 1..threads {
            let (shared, cpu) = (shared.clone(), cpus.get(index % cpus.len().max(1)).copied());
            std::thread::Builder::new()
                .name(format!("ruttice-{index}"))
                .spawn(move || {
                    if let Some(cpu) = cpu {
                        pin_to(cpu);
                    }
                    IN_POOL.with(|flag| flag.set(true));
                    worker(&shared, index, spin);
                })
                .expect("a thread for the pool can be started");
        }
        Self {
            shared,
            threads,
            running: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Whether a loop started now would run on the threads of the pool: not from one of
    /// its own threads, and not while another loop is under way.
    pub fn free(&self) -> bool {
        !IN_POOL.with(Cell::get) && self.running.try_lock().is_ok()
    }

    /// Call `body(k)` once for every thread `k` of the pool, each on its own thread, and
    /// return when all are done. If the pool is not free the calls are made one after the
    /// other on the calling thread.
    pub fn run(&self, body: Body<'_>) {
        let guard = match self.running.try_lock() {
            Ok(guard) if !IN_POOL.with(Cell::get) => guard,
            _ => return (0..self.threads).for_each(body),
        };
        let shared = &self.shared;
        // SAFETY: only the lifetime changes. The threads use the pointer between the bump
        // of `epoch` below and their decrement of `busy`; `Waiting` does not let this
        // function return, or unwind, before `busy` is zero, so the body outlives every
        // use.
        let job = Job(unsafe {
            core::mem::transmute::<Body<'_>, Body<'static>>(body) as *const (dyn Fn(usize) + Sync)
        });
        *lock(&shared.job) = Some(job);
        shared.busy.store(self.threads - 1, Ordering::Release);
        shared.epoch.fetch_add(1, Ordering::Release);
        {
            let _sleepers = lock(&shared.sleep);
            shared.wake.notify_all();
        }
        let waiting = Waiting(shared);
        body(0);
        drop(waiting);
        *lock(&shared.job) = None;
        drop(guard);
        if shared.panicked.swap(false, Ordering::AcqRel) {
            panic!("the body of a loop panicked on a thread of the pool");
        }
    }
}

/// Waits, when dropped, until no thread is busy with the loop under way.
struct Waiting<'a>(&'a Shared);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let mut turns = 0usize;
        while self.0.busy.load(Ordering::Acquire) != 0 {
            turns += 1;
            if turns.is_multiple_of(1024) {
                std::thread::yield_now();
            } else {
                core::hint::spin_loop();
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What a thread of the pool does: wait for a loop, run its piece, say it is done.
fn worker(shared: &Shared, index: usize, spin: usize) {
    let mut seen = 0;
    loop {
        // Look for a new loop for a while, then sleep until one is announced.
        let mut turns = 0;
        loop {
            let now = shared.epoch.load(Ordering::Acquire);
            if now != seen {
                seen = now;
                break;
            }
            if turns < spin {
                turns += 1;
                core::hint::spin_loop();
            } else {
                let sleepers = lock(&shared.sleep);
                if shared.epoch.load(Ordering::Acquire) == seen {
                    drop(
                        shared
                            .wake
                            .wait(sleepers)
                            .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    );
                }
                turns = 0;
            }
        }
        let job = *lock(&shared.job);
        if let Some(Job(body)) = job {
            // SAFETY: see `Pool::run`: the body is alive until `busy` is decremented below.
            let body = unsafe { &*body };
            if catch_unwind(AssertUnwindSafe(|| body(index))).is_err() {
                shared.panicked.store(true, Ordering::Release);
            }
        }
        shared.busy.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Hand one piece to every thread under the static schedules: `work(k, piece)` for piece
/// `k` on thread `k`. There must be one piece per thread of the pool in use. If the threads
/// are not available (the caller is one of them, or a loop is under way) the pieces are
/// worked on one after the other on the calling thread.
pub fn each_thread<P, W>(pieces: Vec<P>, work: W)
where
    P: Send,
    W: Fn(usize, P) + Sync,
{
    let pieces: Vec<Mutex<Option<P>>> = pieces.into_iter().map(|p| Mutex::new(Some(p))).collect();
    let take = |k: usize| {
        if let Some(piece) = pieces.get(k).and_then(|p| lock(p).take()) {
            work(k, piece);
        }
    };
    match schedule() {
        Schedule::Pool => {
            assert_eq!(
                pieces.len(),
                Pool::global().threads(),
                "one piece per thread"
            );
            Pool::global().run(&take);
        }
        _ if rayon::current_thread_index().is_some() => (0..pieces.len()).for_each(take),
        _ => {
            rayon::broadcast(|context| take(context.index()));
        }
    }
}

/// Threads a loop under a static schedule is cut for, or `None` if it should run on the
/// calling thread: under `steal`, with a single thread, or from inside a parallel loop.
pub fn static_threads() -> Option<usize> {
    // The pool that will run the loop decides how many pieces there are: inside
    // `ThreadPool::install` the rayon pool at hand is not the global one.
    let (free, threads) = match schedule() {
        Schedule::Steal => (false, 0),
        Schedule::Static => (
            rayon::current_thread_index().is_none(),
            rayon::current_num_threads(),
        ),
        Schedule::Pool => (Pool::global().free(), Pool::global().threads()),
    };
    (free && threads > 1).then_some(threads)
}
