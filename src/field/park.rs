//! Fields set aside while a solve needs the room for others: a field of the device goes to the
//! host and comes back; one on the CPU stays where it is. Code that owns a field it does not need
//! for a while parks it, and gets it back by `restore` (a copy, the parked one kept, for a field
//! needed again later) or `unpark` (the parked one given up).

use super::{Field, Kind, Layout, Sites, Storage};
use crate::{Cpu, Vector};
use std::error::Error;

/// What a field `V` is while parked.
#[cfg_attr(not(feature = "cuda-bridge"), allow(dead_code))]
pub(crate) type Parked<V> = <V as Park>::Parked;

#[cfg_attr(not(feature = "cuda-bridge"), allow(dead_code))]
pub(crate) trait Park: Sized {
    /// A field while it is parked: another type, so that it cannot be used as a field by mistake.
    type Parked;
    fn park(self) -> Result<Self::Parked, Box<dyn Error>>;
    fn restore(parked: &Self::Parked) -> Result<Self, Box<dyn Error>>;
    fn unpark(parked: Self::Parked) -> Result<Self, Box<dyn Error>>;
}

/// On the CPU a parked field is the field itself.
impl<T: Storage, K: Kind, S: Sites, L: Layout> Park for Field<T, K, S, L, Cpu> {
    type Parked = Self;
    fn park(self) -> Result<Self, Box<dyn Error>> {
        Ok(self)
    }
    fn restore(parked: &Self) -> Result<Self, Box<dyn Error>> {
        let mut f = parked.zeros_like();
        f.copy_from(parked);
        Ok(f)
    }
    fn unpark(parked: Self) -> Result<Self, Box<dyn Error>> {
        Ok(parked)
    }
}
