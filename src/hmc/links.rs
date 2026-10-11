//! The links of the molecular dynamics as the actions see them: the links themselves and
//! their smeared versions. Each smearing is computed when an action first asks for it and
//! then shared by all actions (several fermion terms on the same stout links smear once)
//! until the links change, when the integrator clears the cache. The idea is that of
//! IroIro's smeared configuration object.

use crate::gauge::GaugeSmearing;
use crate::{Field, Gauge};
use std::cell::RefCell;
use std::rc::Rc;

/// The fields after each step of a smearing.
pub type Levels = Rc<Vec<Field<f64, Gauge>>>;

/// The smeared levels computed so far for the current links.
#[derive(Default)]
pub struct SmearingCache {
    entries: RefCell<Vec<(GaugeSmearing, Levels)>>,
}

impl SmearingCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget everything: the links have changed.
    pub fn clear(&self) {
        self.entries.borrow_mut().clear();
    }

    /// Smearings computed since the last `clear`.
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The current links and the cache of their smearings.
#[derive(Clone, Copy)]
pub struct Links<'a> {
    pub thin: &'a Field<f64, Gauge>,
    cache: &'a SmearingCache,
}

impl<'a> Links<'a> {
    pub fn new(thin: &'a Field<f64, Gauge>, cache: &'a SmearingCache) -> Self {
        Self { thin, cache }
    }

    /// The fields after each step of `smearing` (none for no smearing), computed once.
    pub fn levels(&self, smearing: Option<&GaugeSmearing>) -> Levels {
        let Some(s) = smearing.filter(|s| s.steps > 0) else {
            return Rc::new(Vec::new());
        };
        if let Some((_, l)) = self.cache.entries.borrow().iter().find(|(k, _)| k == s) {
            return l.clone();
        }
        let l = Rc::new(s.levels(self.thin));
        self.cache.entries.borrow_mut().push((*s, l.clone()));
        l
    }
}
