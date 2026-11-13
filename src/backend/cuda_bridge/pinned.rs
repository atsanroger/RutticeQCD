//! Page-locked (pinned) host memory, for data that crosses the bus often.
//!
//! A copy between the device and pageable memory goes through a staging buffer of the driver,
//! and into a new `Vec` it also faults in every page: 4.3 GB from the A30 take 2.5 s into a
//! new `Vec`, 1.1 s into memory written before, 0.22 s into pinned memory. Pinning itself is
//! slow (2.0 s for 4.3 GB), so a dropped buffer goes back to a cache of the process and the
//! next request of at least half its size takes it (so a small request does not hold a large
//! block); the cache is never freed.

use super::ffi::cudaMallocHost;
use super::{check, select_device, CudaError};
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

/// Pinned blocks not in use: (address, bytes).
static FREE: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// `len` values of `T` in pinned host memory, zero when first pinned and otherwise holding
/// what the last user of the block left (every user here writes the whole range first).
pub(crate) struct PinnedHost<T> {
    ptr: *mut T,
    len: usize,
    bytes: usize,
}

// SAFETY: the buffer owns its block alone (taken out of the cache, given back on drop), like a
// `Vec<T>`; `T` is a plain real.
unsafe impl<T: Send> Send for PinnedHost<T> {}
// SAFETY: shared access only reads, as for `&[T]`.
unsafe impl<T: Sync> Sync for PinnedHost<T> {}

impl<T: Copy> PinnedHost<T> {
    /// A buffer of `len` values: the smallest cached block that holds them and is at most twice
    /// as large, or a new one.
    pub(crate) fn new(len: usize) -> Result<Self, CudaError> {
        let need = len * std::mem::size_of::<T>();
        if need == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len,
                bytes: 0,
            });
        }
        let cached = {
            let mut free = FREE.lock().unwrap_or_else(|e| e.into_inner());
            let best = (0..free.len())
                .filter(|&k| (need..=2 * need).contains(&free[k].1))
                .min_by_key(|&k| free[k].1);
            best.map(|k| free.swap_remove(k))
        };
        let (addr, bytes) = match cached {
            Some(block) => block,
            None => {
                select_device();
                let mut ptr: *mut c_void = std::ptr::null_mut();
                // SAFETY: `ptr` is a valid place for the pointer.
                check(unsafe { cudaMallocHost(&mut ptr, need) })?;
                // SAFETY: the block was just allocated with `need` bytes; zero is a valid real,
                // so the slice below never shows memory nothing has written.
                unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, need) };
                (ptr as usize, need)
            }
        };
        Ok(Self {
            ptr: addr as *mut T,
            len,
            bytes,
        })
    }
}

impl<T> Drop for PinnedHost<T> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let mut free = FREE.lock().unwrap_or_else(|e| e.into_inner());
            free.push((self.ptr as usize, self.bytes));
        }
    }
}

impl<T> Deref for PinnedHost<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the block holds at least `len` values of `T`, all written (zeroed when
        // pinned), aligned for any real (cudaMallocHost aligns to pages), owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for PinnedHost<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`; `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
