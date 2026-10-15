//! The levels below the coarse lattice of the multigrid on the device: the signed multilevel
//! V-cycle of the CPU (`SignedMultilevel`, `multigrid/levels.rs`) step for step, with the
//! operator of every level a coarse operator of the device (`DeviceCoarseOp`), and its local
//! vectors, the probing of its operators and the dense inverse of the last level made there
//! (`multigrid/levels.cuh`). Only the tables of the lattices of the levels are made on the
//! host.

use super::coarse::DeviceTable;
use super::*;
use crate::solver::preconditioner::multigrid::{LevelBootstrap, MultigridError};
use crate::{Vector, C64};
use std::error::Error;

/// The kernels of `multigrid/levels.cuh` in one precision, as `bridge_cuda.h` declares them; the
/// pointers are device memory of the sizes those read and write.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub trait LevelReal: DeviceReal {
    unsafe fn level_restrict(
        out: *mut Self,
        v: *const Self,
        q: *const Self,
        members: *const u32,
        n_blocks: i32,
        nq: i32,
        bs: i32,
        n_parent: i32,
        pd: i32,
        vectors: i32,
    );
    unsafe fn level_prolong(
        out: *mut Self,
        y: *const Self,
        q: *const Self,
        owner: *const u32,
        n_blocks: i32,
        nq: i32,
        bs: i32,
        n_parent: i32,
        pd: i32,
        add: i32,
        vectors: i32,
    );
    unsafe fn level_basis(
        q: *mut f64,
        tests: *const Self,
        members: *const u32,
        n_blocks: i32,
        n_t: i32,
        bs: i32,
        n_parent: i32,
        pd: i32,
    );
    unsafe fn level_sources(e: *mut Self, colour_of: *const u32, colour: u32, sites: i32, nq: i32);
    unsafe fn level_probe(
        diag: *mut f64,
        links: *mut f64,
        r: *const Self,
        colour_of: *const u32,
        neighbours: *const u32,
        colour: u32,
        used: u32,
        sites: i32,
        nq: i32,
    );
}

extern "C" {
    fn bridge_cuda_level_dense(
        dense: *mut f64,
        diag: *const f64,
        links: *const f64,
        neighbours: *const u32,
        used: u32,
        sites: i32,
        nq: i32,
        stream: *mut c_void,
    );
}

macro_rules! level_real {
    ($t:ty: $restrict:ident, $prolong:ident, $basis:ident, $sources:ident, $probe:ident) => {
        extern "C" {
            fn $restrict(
                out: *mut $t,
                v: *const $t,
                q: *const $t,
                members: *const u32,
                n_blocks: i32,
                nq: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $prolong(
                out: *mut $t,
                y: *const $t,
                q: *const $t,
                owner: *const u32,
                n_blocks: i32,
                nq: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
                add: i32,
                vectors: i32,
                stream: *mut c_void,
            );
            fn $basis(
                q: *mut f64,
                tests: *const $t,
                members: *const u32,
                n_blocks: i32,
                n_t: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
                stream: *mut c_void,
            );
            fn $sources(
                e: *mut $t,
                colour_of: *const u32,
                colour: u32,
                sites: i32,
                nq: i32,
                stream: *mut c_void,
            );
            fn $probe(
                diag: *mut f64,
                links: *mut f64,
                r: *const $t,
                colour_of: *const u32,
                neighbours: *const u32,
                colour: u32,
                used: u32,
                sites: i32,
                nq: i32,
                stream: *mut c_void,
            );
        }

        #[allow(clippy::too_many_arguments)]
        impl LevelReal for $t {
            unsafe fn level_restrict(
                out: *mut Self,
                v: *const Self,
                q: *const Self,
                members: *const u32,
                n_blocks: i32,
                nq: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
                vectors: i32,
            ) {
                $restrict(
                    out, v, q, members, n_blocks, nq, bs, n_parent, pd, vectors, STREAM,
                )
            }
            unsafe fn level_prolong(
                out: *mut Self,
                y: *const Self,
                q: *const Self,
                owner: *const u32,
                n_blocks: i32,
                nq: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
                add: i32,
                vectors: i32,
            ) {
                $prolong(
                    out, y, q, owner, n_blocks, nq, bs, n_parent, pd, add, vectors, STREAM,
                )
            }
            unsafe fn level_basis(
                q: *mut f64,
                tests: *const Self,
                members: *const u32,
                n_blocks: i32,
                n_t: i32,
                bs: i32,
                n_parent: i32,
                pd: i32,
            ) {
                $basis(q, tests, members, n_blocks, n_t, bs, n_parent, pd, STREAM)
            }
            unsafe fn level_sources(
                e: *mut Self,
                colour_of: *const u32,
                colour: u32,
                sites: i32,
                nq: i32,
            ) {
                $sources(e, colour_of, colour, sites, nq, STREAM)
            }
            unsafe fn level_probe(
                diag: *mut f64,
                links: *mut f64,
                r: *const Self,
                colour_of: *const u32,
                neighbours: *const u32,
                colour: u32,
                used: u32,
                sites: i32,
                nq: i32,
            ) {
                $probe(
                    diag, links, r, colour_of, neighbours, colour, used, sites, nq, STREAM,
                )
            }
        }
    };
}

level_real!(f32: bridge_cuda_level_restrict_f32, bridge_cuda_level_prolong_f32,
    bridge_cuda_level_basis_f32, bridge_cuda_level_sources_f32, bridge_cuda_level_probe_f32);
level_real!(f64: bridge_cuda_level_restrict_f64, bridge_cuda_level_prolong_f64,
    bridge_cuda_level_basis_f64, bridge_cuda_level_sources_f64, bridge_cuda_level_probe_f64);

fn site_coords(s: usize, ext: &[usize; ND]) -> [usize; ND] {
    let mut c = [0; ND];
    let mut r = s;
    for mu in 0..ND {
        c[mu] = r % ext[mu];
        r /= ext[mu];
    }
    c
}

fn site_index(c: &[usize; ND], ext: &[usize; ND]) -> usize {
    (0..ND).rev().fold(0, |s, mu| s * ext[mu] + c[mu])
}

/// The super-blocks and local vectors of one level (`Level` of the CPU).
struct Level<T> {
    q: DeviceBuffer<T>,
    members: DeviceTable,
    owner: DeviceTable,
    /// Super-blocks, their local vectors, and the sites of the level above in each.
    blocks: usize,
    nq: usize,
    bs: usize,
    /// Sites of the level above, and their complex numbers.
    parent_sites: usize,
    pd: usize,
}

impl<T: LevelReal> Level<T> {
    /// The level from the sites of the level above, at `coords` (of every site, as the fields
    /// there have them) in the lattice `parent_ext`, with `pd` components, and test vectors
    /// there; and the extent of its lattice, numbered as `site_index` does.
    fn build(
        coords: &[[usize; ND]],
        parent_ext: [usize; ND],
        pd: usize,
        super_block: usize,
        tests: &DeviceCoarse<T>,
    ) -> Result<(Self, [usize; ND]), Box<dyn Error>> {
        assert!(
            parent_ext.iter().all(|e| e % super_block == 0),
            "super-blocks of {super_block} do not divide the extent {parent_ext:?}"
        );
        assert!((tests.sites, tests.dim) == (coords.len(), pd));
        let ext = parent_ext.map(|e| e / super_block);
        let blocks: usize = ext.iter().product();
        let bs = super_block.pow(ND as u32);
        // The sites of every super-block in the order of their global index above.
        let mut sites: Vec<Vec<(usize, usize)>> = vec![Vec::with_capacity(bs); blocks];
        for (x, c) in coords.iter().enumerate() {
            sites[site_index(&c.map(|v| v / super_block), &ext)]
                .push((site_index(c, &parent_ext), x));
        }
        let mut members = vec![0u32; blocks * bs];
        let mut owner = vec![0u32; coords.len()];
        for (s, list) in sites.iter_mut().enumerate() {
            list.sort_unstable();
            assert_eq!(list.len(), bs);
            for (k, &(_, x)) in list.iter().enumerate() {
                members[s * bs + k] = x as u32;
                owner[x] = (s * bs + k) as u32;
            }
        }
        let nq = 2 * tests.vectors;
        let members = DeviceTable::from_host(&members)?;
        let q64 = DeviceBuffer::<f64>::zeros(2 * blocks * nq * bs * pd)?;
        // SAFETY: the local vectors of every super-block, the test vectors on the sites above,
        // and a table of the sites of every super-block.
        unsafe {
            T::level_basis(
                q64.ptr,
                tests.data.ptr,
                members.ptr,
                blocks as i32,
                tests.vectors as i32,
                bs as i32,
                coords.len() as i32,
                pd as i32,
            );
        }
        let mut q = DeviceBuffer::<T>::uninitialised(q64.len)?;
        convert(&mut q, &q64);
        Ok((
            Self {
                q,
                members,
                owner: DeviceTable::from_host(&owner)?,
                blocks,
                nq,
                bs,
                parent_sites: coords.len(),
                pd,
            },
            ext,
        ))
    }

    /// Q† v.
    fn restrict(&self, v: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        assert!((v.sites, v.dim) == (self.parent_sites, self.pd));
        let out = must(DeviceCoarse::zeros(self.blocks, self.nq, v.vectors));
        // SAFETY: fields of the two lattices, Q and the table of this level.
        unsafe {
            T::level_restrict(
                out.data.ptr,
                v.data.ptr,
                self.q.ptr,
                self.members.ptr,
                self.blocks as i32,
                self.nq as i32,
                self.bs as i32,
                self.parent_sites as i32,
                self.pd as i32,
                v.vectors as i32,
            );
        }
        out
    }

    /// out = Q y, or out += Q y.
    fn prolong(&self, out: &mut DeviceCoarse<T>, y: &DeviceCoarse<T>, add: bool) {
        assert!((y.sites, y.dim) == (self.blocks, self.nq));
        assert!((out.sites, out.dim, out.vectors) == (self.parent_sites, self.pd, y.vectors));
        // SAFETY: fields of the two lattices, Q and the table of this level.
        unsafe {
            T::level_prolong(
                out.data.ptr,
                y.data.ptr,
                self.q.ptr,
                self.owner.ptr,
                self.blocks as i32,
                self.nq as i32,
                self.bs as i32,
                self.parent_sites as i32,
                self.pd as i32,
                i32::from(add),
                y.vectors as i32,
            );
        }
    }
}

/// What the dense matrix of the last level is made from: the links of its operator in double
/// precision, the sites they come from, and the directions that are used.
struct Probed {
    links: DeviceBuffer<f64>,
    neighbours: DeviceTable,
    used: u32,
}

/// The operator of `level` (lattice `ext`) from that of the level above (`apply`), by probing as
/// `LevelOp::probe` of the CPU: sources c_μ apart, c_μ the smallest divisor of the extent from 3
/// up (or the extent), share no neighbour. Also the applications of the operator above.
fn probe<T: LevelReal>(
    level: &Level<T>,
    ext: [usize; ND],
    apply: &mut dyn FnMut(&mut DeviceCoarse<T>, &DeviceCoarse<T>),
) -> Result<(DeviceCoarseOp<T>, Probed, usize), Box<dyn Error>> {
    let nq = level.nq;
    assert!(
        nq <= 1024,
        "a level has at most 1024 complex numbers a site on the device"
    );
    let total: usize = ext.iter().product();
    let mut neighbours = vec![0u32; 2 * ND * total];
    for t in 0..total {
        let c = site_coords(t, &ext);
        for mu in 0..ND {
            let (mut f, mut b) = (c, c);
            f[mu] = (c[mu] + 1) % ext[mu];
            b[mu] = (c[mu] + ext[mu] - 1) % ext[mu];
            neighbours[mu * total + t] = site_index(&f, &ext) as u32;
            neighbours[(ND + mu) * total + t] = site_index(&b, &ext) as u32;
        }
    }
    // A direction of extent 1 is in the block; of extent 2 its two hops are the same one.
    let used = (0..2 * ND)
        .filter(|&d| {
            let e = ext[d % ND];
            !(e == 1 || (e == 2 && d >= ND))
        })
        .fold(0u32, |m, d| m | (1 << d));
    let colours: [usize; ND] = ext.map(|e| (3..=e).find(|k| e % k == 0).unwrap_or(e).max(1));
    let n_colours: usize = colours.iter().product();
    let colour_of: Vec<u32> = (0..total)
        .map(|t| {
            let c = site_coords(t, &ext);
            (0..ND)
                .rev()
                .fold(0, |s, mu| s * colours[mu] + c[mu] % colours[mu]) as u32
        })
        .collect();
    let colour_table = DeviceTable::from_host(&colour_of)?;
    let neighbour_table = DeviceTable::from_host(&neighbours)?;
    let matrix = 2 * nq * nq;
    let diag = DeviceBuffer::<f64>::zeros(matrix * total)?;
    let links = DeviceBuffer::<f64>::zeros(2 * ND * matrix * total)?;
    let e = must(DeviceCoarse::<T>::zeros(total, nq, nq));
    let mut p = must(DeviceCoarse::<T>::zeros(level.parent_sites, level.pd, nq));
    let mut ap = p.zeros_like();
    let mut applications = 0;
    for colour in 0..n_colours as u32 {
        // SAFETY: a field of nq vectors on this level's lattice and its table of colours.
        unsafe {
            T::level_sources(
                e.data.ptr,
                colour_table.ptr,
                colour,
                total as i32,
                nq as i32,
            )
        };
        level.prolong(&mut p, &e, false);
        apply(&mut ap, &p);
        let r = level.restrict(&ap);
        applications += nq;
        // SAFETY: the blocks and links of this lattice, the response on it and its tables.
        unsafe {
            T::level_probe(
                diag.ptr,
                links.ptr,
                r.data.ptr,
                colour_table.ptr,
                neighbour_table.ptr,
                colour,
                used,
                total as i32,
                nq as i32,
            );
        }
    }
    let op = DeviceCoarseOp::from_parts(diag, &links, &neighbours, &vec![0; total], nq, false)?;
    Ok((
        op,
        Probed {
            links,
            neighbours: neighbour_table,
            used,
        },
        applications,
    ))
}

/// The levels below the coarse lattice as a V-cycle for a coarse operator A_c on the device,
/// `SignedMultilevel` of the CPU: `apply` is Q₂ M₂ Q₂†.
pub struct DeviceMultilevel<T: LevelReal> {
    levels: Vec<Level<T>>,
    ops: Vec<DeviceCoarseOp<T>>,
    exts: Vec<[usize; ND]>,
    /// Of the last of `ops`.
    last: Option<Probed>,
    coarsest_inv: Option<DeviceBuffer<T>>,
    coarsest_dim: usize,
    /// Applications of A_c to build it.
    pub setup_applications: usize,
}

/// `iterations` steps of GCR on A x = b, preconditioned by `m`, from x = 0 (`gcr` of the CPU):
/// one vector.
fn gcr<T: LevelReal>(
    a: &DeviceCoarseOp<T>,
    m: &dyn Fn(&DeviceCoarse<T>) -> DeviceCoarse<T>,
    b: &DeviceCoarse<T>,
    iterations: usize,
) -> DeviceCoarse<T> {
    assert_eq!(b.vectors, 1);
    let mut x = b.zeros_like();
    let mut r = b.zeros_like();
    r.copy_from(b);
    let (mut ps, mut aps): (Vec<DeviceCoarse<T>>, Vec<DeviceCoarse<T>>) = (Vec::new(), Vec::new());
    let mut c = [C64::ZERO];
    let mut n2 = [0.0];
    for _ in 0..iterations {
        let mut z = m(&r);
        let mut az = z.zeros_like();
        a.apply(&mut az, &z, false);
        for (p, ap) in ps.iter().zip(&aps) {
            ap.dot(&az, &mut c);
            z.axpy(&[-c[0]], p);
            az.axpy(&[-c[0]], ap);
        }
        az.norm_sqr(&mut n2);
        let n = n2[0].sqrt();
        if n == 0.0 {
            break;
        }
        z.scale(&[1.0 / n]);
        az.scale(&[1.0 / n]);
        az.dot(&r, &mut c);
        x.axpy(&c, &z);
        r.axpy(&[-c[0]], &az);
        ps.push(z);
        aps.push(az);
    }
    x
}

impl<T: LevelReal> DeviceMultilevel<T> {
    /// The levels for `a`, on the coarse lattice `lat`, from the test vectors `vectors` there
    /// (one each), with super-blocks of `super_blocks[ℓ]` sites per direction on level ℓ + 2
    /// and the test vectors of level ℓ + 2 refined by `bootstrap[ℓ]` (missing entries: none)
    /// before level ℓ + 3 is built from them; the last level is solved exactly.
    pub fn build(
        a: &DeviceCoarseOp<T>,
        lat: &Lattice,
        vectors: &DeviceCoarse<T>,
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<Self, Box<dyn Error>> {
        assert!(!super_blocks.is_empty() && vectors.vectors > 0);
        assert_eq!(vectors.sites, lat.num_sites());
        let coords: Vec<[usize; ND]> = (0..lat.num_sites()).map(|x| lat.global_coords(x)).collect();
        let (level, ext) = Level::build(&coords, lat.global, a.dim(), super_blocks[0], vectors)?;
        let (op, probed, applications) =
            probe(&level, ext, &mut |out, inp| a.apply(out, inp, false))?;
        let below = level.restrict(vectors);
        let mut me = Self {
            levels: vec![level],
            ops: vec![op],
            exts: vec![ext],
            last: Some(probed),
            coarsest_inv: None,
            coarsest_dim: 0,
            setup_applications: applications,
        };
        me.rebuild(1, &below, super_blocks)?;
        me.extend(1, below, super_blocks, bootstrap)?;
        Ok(me)
    }

    /// Complex components of the coarsest level, solved densely.
    pub fn coarsest_dim(&self) -> usize {
        self.coarsest_dim
    }

    /// `SignedMultilevel::extend` of the CPU.
    fn extend(
        &mut self,
        from: usize,
        mut tests: DeviceCoarse<T>,
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<(), Box<dyn Error>> {
        if from >= super_blocks.len() {
            return Ok(());
        }
        let boot = bootstrap.get(from - 1).copied().unwrap_or_default();
        for _ in 0..boot.rounds {
            let op = &self.ops[from - 1];
            let m = |r: &DeviceCoarse<T>| self.vcycle(from - 1, r);
            let mut refined = tests.zeros_like();
            for j in 0..tests.vectors {
                let mut x = gcr(op, &m, &tests.select(&[j]), boot.iterations.max(1));
                let mut n2 = [0.0];
                x.norm_sqr(&mut n2);
                x.scale(&[1.0 / n2[0].sqrt()]);
                refined.assign(&[j], &x);
            }
            tests = refined;
            self.rebuild(from, &tests, super_blocks)?;
        }
        let below = self.levels[from].restrict(&tests);
        self.extend(from + 1, below, super_blocks, bootstrap)
    }

    /// Levels `from`.. built again from `tests` (on the lattice of level `from` − 1's
    /// operator), without refinement; the coarsest inverse again.
    fn rebuild(
        &mut self,
        from: usize,
        tests: &DeviceCoarse<T>,
        super_blocks: &[usize],
    ) -> Result<(), Box<dyn Error>> {
        self.levels.truncate(from);
        self.ops.truncate(from);
        self.exts.truncate(from);
        let mut tests = tests.select(&(0..tests.vectors).collect::<Vec<_>>());
        for &sb in &super_blocks[from..] {
            let ext = *self.exts.last().expect("a level above");
            let above = self.ops.last().expect("a level above");
            let coords: Vec<[usize; ND]> = (0..ext.iter().product())
                .map(|x| site_coords(x, &ext))
                .collect();
            let (level, level_ext) = Level::build(&coords, ext, above.dim(), sb, &tests)?;
            let (op, probed, _) = probe(&level, level_ext, &mut |out, inp| {
                above.apply(out, inp, false)
            })?;
            tests = level.restrict(&tests);
            self.levels.push(level);
            self.ops.push(op);
            self.exts.push(level_ext);
            self.last = Some(probed);
        }
        // The coarsest level dense, inverted by Gauss-Jordan elimination in double precision.
        let last = self.ops.last().expect("at least one level");
        let probed = self.last.as_ref().expect("the links of the last level");
        let (sites, nq) = (last.sites(), last.dim());
        let n = sites * nq;
        if n > 1024 {
            return Err(format!(
                "the last level below the coarse lattice has {n} complex numbers, more than the \
                 1024 the device inverts densely: add a level or take larger super-blocks"
            )
            .into());
        }
        let dense = DeviceBuffer::<f64>::zeros(2 * n * n)?;
        // SAFETY: an n × n matrix, and the blocks, links and table of the last level.
        unsafe {
            bridge_cuda_level_dense(
                dense.ptr,
                last.diag_f64().ptr,
                probed.links.ptr,
                probed.neighbours.ptr,
                probed.used,
                sites as i32,
                nq as i32,
                STREAM,
            );
        }
        let inverse = DeviceBuffer::<f64>::uninitialised(dense.len)?;
        let scratch = DeviceBuffer::<f64>::uninitialised(2 * dense.len)?;
        let fail = DeviceTable::from_host(&[0])?;
        // SAFETY: one matrix of dim n, as many places for its inverse, the scratch the kernel
        // asks for, and a place for the failure.
        unsafe {
            bridge_cuda_blocks_inverse(
                inverse.ptr,
                dense.ptr,
                scratch.ptr,
                fail.ptr,
                1,
                n as i32,
                STREAM,
            );
        }
        finish();
        if fail.download(1)?[0] != 0 {
            return Err(Box::new(MultigridError::SingularCoarseDiagonal {
                coarse_site: 0,
            }));
        }
        let mut inv = DeviceBuffer::<T>::uninitialised(inverse.len)?;
        convert(&mut inv, &inverse);
        self.coarsest_inv = Some(inv);
        self.coarsest_dim = n;
        Ok(())
    }

    /// The dense inverse of the last level applied to `r`.
    fn coarsest(&self, r: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        let inverse = self.coarsest_inv.as_ref().expect("the coarsest inverse");
        assert_eq!(r.sites * r.dim, self.coarsest_dim);
        let out = r.zeros_like();
        // SAFETY: the fields of the last level have the layout of one site of all their
        // complex numbers (`coarse_index` of q = X dim + k), and the inverse is that site's.
        unsafe {
            T::coarse_pass(
                out.data.ptr,
                r.data.ptr,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                inverse.ptr,
                std::ptr::null(),
                std::ptr::null(),
                1,
                self.coarsest_dim as i32,
                -1,
                0,
                0,
                r.vectors as i32,
            );
        }
        out
    }

    /// The V-cycle from the lattice of level `l`'s operator down.
    fn vcycle(&self, l: usize, r: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        if l + 1 == self.levels.len() {
            return self.coarsest(r);
        }
        let nv = r.vectors;
        let below = &self.levels[l + 1];
        let zc = self.vcycle(l + 1, &below.restrict(r));
        let mut z = r.zeros_like();
        below.prolong(&mut z, &zc, false);
        let op = &self.ops[l];
        let mut az = z.zeros_like();
        op.apply(&mut az, &z, false);
        az.aypx(&vec![C64::new(-1.0, 0.0); nv], r);
        let mut t = az.zeros_like();
        op.apply_blocks(&mut t, &az, op.block_inverses());
        z.axpy(&vec![C64::new(1.0, 0.0); nv], &t);
        z
    }

    /// `out = Q₂ M₂ Q₂† inp`, all the vectors of the field at once.
    pub fn apply(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        let first = &self.levels[0];
        let z = self.vcycle(0, &first.restrict(inp));
        first.prolong(out, &z, false);
    }
}
