//! The levels below the coarse lattice of the multigrid on the device: the signed multilevel
//! V-cycle of the CPU for the quark factor (`SignedMultilevel`, `multigrid/levels.rs`) and the
//! absolute-value preconditioner for the Pauli-Villars factor (`AbsMultilevel`), step for step,
//! with the operator of every level a coarse operator of the device (`DeviceCoarseOp`), and its
//! local vectors, the probing of its operators (`multigrid/levels.cuh`) and the dense inverse
//! of the last level (`multigrid/dense.cu`, by cuSOLVER: any size) made there. Only the tables
//! of the lattices of the levels are made on the host.

use super::coarse::{CoarseGhosts, DeviceTable};
use super::*;
use crate::solver::preconditioner::multigrid::{LevelBootstrap, MultigridError};
use crate::{Vector, C64};
use std::error::Error;
use std::rc::Rc;

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
    unsafe fn dense_apply(out: *mut Self, r: *const Self, m: *const Self, n: i32, vectors: i32);
    unsafe fn coarse_adjoint(
        out: *mut Self,
        m: *const Self,
        from: *const u32,
        dirs: i32,
        n_coarse: i32,
        dim: i32,
    );
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
    fn bridge_cuda_dense_inverse(inverse: *mut f64, a: *const f64, n: i32) -> i32;
    fn bridge_cuda_dense_abs_inverse(out: *mut f64, a: *const f64, n: i32, cutoff: f64) -> i32;
    fn bridge_cuda_blocks_abs_inverse(
        out: *mut f64,
        a: *const f64,
        scratch: *mut f64,
        sites: i32,
        dim: i32,
        cutoff: f64,
        stream: *mut c_void,
    );
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
    ($t:ty: $restrict:ident, $prolong:ident, $basis:ident, $sources:ident, $probe:ident,
     $dense_apply:ident, $adjoint:ident) => {
        extern "C" {
            fn $adjoint(
                out: *mut $t,
                m: *const $t,
                from: *const u32,
                dirs: i32,
                n_coarse: i32,
                dim: i32,
                stream: *mut c_void,
            );
            fn $dense_apply(
                out: *mut $t,
                r: *const $t,
                m: *const $t,
                n: i32,
                vectors: i32,
                stream: *mut c_void,
            );
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
            unsafe fn coarse_adjoint(
                out: *mut Self,
                m: *const Self,
                from: *const u32,
                dirs: i32,
                n_coarse: i32,
                dim: i32,
            ) {
                $adjoint(out, m, from, dirs, n_coarse, dim, STREAM)
            }
            unsafe fn dense_apply(
                out: *mut Self,
                r: *const Self,
                m: *const Self,
                n: i32,
                vectors: i32,
            ) {
                $dense_apply(out, r, m, n, vectors, STREAM)
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
    bridge_cuda_level_basis_f32, bridge_cuda_level_sources_f32, bridge_cuda_level_probe_f32,
    bridge_cuda_dense_apply_f32, bridge_cuda_coarse_adjoint_f32);
level_real!(f64: bridge_cuda_level_restrict_f64, bridge_cuda_level_prolong_f64,
    bridge_cuda_level_basis_f64, bridge_cuda_level_sources_f64, bridge_cuda_level_probe_f64,
    bridge_cuda_dense_apply_f64, bridge_cuda_coarse_adjoint_f64);

/// The dense matrix of the operator of the last level (`last`, with the links `probed`, on the
/// lattice `lat`), in double precision, n × n for n = its sites times its components. On
/// several processes it is the matrix of the whole level, sites in their global order, the same
/// on every process: the blocks and links of all of them gathered.
fn dense_of<T: LevelReal>(
    last: &DeviceCoarseOp<T>,
    probed: &Probed,
    lat: &Lattice,
) -> Result<(DeviceBuffer<f64>, usize), CudaError> {
    let nq = last.dim();
    let dense_from = |sites: usize,
                      diag: &DeviceBuffer<f64>,
                      links: &DeviceBuffer<f64>,
                      neighbours: &DeviceTable|
     -> Result<(DeviceBuffer<f64>, usize), CudaError> {
        let n = sites * nq;
        let dense = DeviceBuffer::<f64>::zeros(2 * n * n)?;
        crate::solver::preconditioner::multigrid::count::copy(2 * n * n, 8);
        // SAFETY: an n × n matrix, and the blocks, links and table of `sites` sites.
        unsafe {
            bridge_cuda_level_dense(
                dense.ptr,
                diag.ptr,
                links.ptr,
                neighbours.ptr,
                probed.used,
                sites as i32,
                nq as i32,
                STREAM,
            );
        }
        Ok((dense, n))
    };
    if lat.comm.size() == 1 {
        return dense_from(
            last.sites(),
            last.diag_f64(),
            &probed.links,
            &probed.neighbours,
        );
    }
    // Every matrix of a site to its place in the whole level, summed over the processes (the
    // others add zeros, so the sum is a copy).
    let matrix = 2 * nq * nq;
    let total: usize = lat.global.iter().product();
    let (sites, global) = (lat.num_sites(), global_order(lat));
    let (diag, links) = (last.diag_f64().download()?, probed.links.download()?);
    let mut all_diag = vec![0.0; total * matrix];
    let mut all_links = vec![0.0; 2 * ND * total * matrix];
    for t in 0..sites {
        let g = global[t];
        all_diag[g * matrix..(g + 1) * matrix].copy_from_slice(&diag[t * matrix..(t + 1) * matrix]);
        for d in 0..2 * ND {
            let (from, to) = ((d * sites + t) * matrix, (d * total + g) * matrix);
            all_links[to..to + matrix].copy_from_slice(&links[from..from + matrix]);
        }
    }
    lat.comm.sum(&mut all_diag);
    lat.comm.sum(&mut all_links);
    let mut neighbours = vec![0u32; 2 * ND * total];
    for g in 0..total {
        let c = lat.global_site_coords(g);
        for mu in 0..ND {
            let (mut f, mut b) = (c, c);
            f[mu] = (c[mu] + 1) % lat.global[mu];
            b[mu] = (c[mu] + lat.global[mu] - 1) % lat.global[mu];
            neighbours[mu * total + g] = site_index(&f, &lat.global) as u32;
            neighbours[(ND + mu) * total + g] = site_index(&b, &lat.global) as u32;
        }
    }
    dense_from(
        total,
        &DeviceBuffer::from_host(&all_diag)?,
        &DeviceBuffer::from_host(&all_links)?,
        &DeviceTable::from_host(&neighbours)?,
    )
}

/// The global number (x fastest) of every local site of `lat`.
fn global_order(lat: &Lattice) -> Vec<usize> {
    (0..lat.num_sites())
        .map(|t| site_index(&lat.global_coords(t), &lat.global))
        .collect()
}

/// A field of the last level, of the local sites of `lat`, as the field of the whole level (in
/// its global order) on every process, or back.
fn gathered<T: LevelReal>(r: &DeviceCoarse<T>, lat: &Lattice) -> DeviceCoarse<T> {
    let (sites, dim, nv) = (r.sites, r.dim, r.vectors);
    let total: usize = lat.global.iter().product();
    let global = global_order(lat);
    let host = must(r.data.download());
    let (one, all_one) = (2 * padded(sites * dim), 2 * padded(total * dim));
    let mut all = vec![0.0; nv * all_one];
    for v in 0..nv {
        for (t, &g) in global.iter().enumerate() {
            for k in 0..dim {
                let (q, p) = (t * dim + k, g * dim + k);
                for part in 0..2 {
                    all[v * all_one + at(part, p, 2)] = host[v * one + at(part, q, 2)];
                }
            }
        }
    }
    lat.comm.sum(&mut all);
    let mut out = must(DeviceCoarse::<T>::zeros(total, dim, nv));
    out.data = must(DeviceBuffer::from_host(&all));
    out
}

fn scattered<T: LevelReal>(
    all: &DeviceCoarse<T>,
    like: &DeviceCoarse<T>,
    lat: &Lattice,
) -> DeviceCoarse<T> {
    let (sites, dim, nv) = (like.sites, like.dim, like.vectors);
    let total: usize = lat.global.iter().product();
    let host = must(all.data.download());
    let (one, all_one) = (2 * padded(sites * dim), 2 * padded(total * dim));
    let mut mine = vec![0.0; nv * one];
    for v in 0..nv {
        for (t, g) in global_order(lat).into_iter().enumerate() {
            for k in 0..dim {
                let (q, p) = (t * dim + k, g * dim + k);
                for part in 0..2 {
                    mine[v * one + at(part, q, 2)] = host[v * all_one + at(part, p, 2)];
                }
            }
        }
    }
    let mut out = like.zeros_like();
    out.data = must(DeviceBuffer::from_host(&mine));
    out
}

/// A dense n × n matrix of the last level in the precision `T`, made in double precision by
/// `make` (out, the dense operator, n: two n × n matrices) with its status (0: done).
fn dense_map<T: LevelReal>(
    dense: &DeviceBuffer<f64>,
    n: usize,
    what: &str,
    make: impl FnOnce(*mut f64, *const f64, i32) -> i32,
) -> Result<DeviceBuffer<T>, Box<dyn Error>> {
    let out = DeviceBuffer::<f64>::zeros(dense.len)?;
    finish();
    let status = make(out.ptr, dense.ptr, n as i32);
    finish();
    match status {
        0 => {}
        s if s > 0 => {
            return Err(Box::new(MultigridError::SingularCoarseDiagonal {
                coarse_site: 0,
            }))
        }
        s => {
            return Err(
                format!("{what} of the last level ({n} complex numbers): status {s}").into(),
            )
        }
    }
    let mut converted = DeviceBuffer::<T>::uninitialised(out.len)?;
    convert(&mut converted, &out);
    Ok(converted)
}

/// `out = M r` for the dense matrix `m` of n × n of the last level, on the lattice `lat` (on
/// several processes r of the local sites, gathered to the whole level and back).
fn dense_apply<T: LevelReal>(
    m: &DeviceBuffer<T>,
    n: usize,
    r: &DeviceCoarse<T>,
    lat: &Lattice,
) -> DeviceCoarse<T> {
    if lat.comm.size() > 1 {
        let all = gathered(r, lat);
        let out = dense_apply_here(m, n, &all);
        return scattered(&out, r, lat);
    }
    dense_apply_here(m, n, r)
}

fn dense_apply_here<T: LevelReal>(
    m: &DeviceBuffer<T>,
    n: usize,
    r: &DeviceCoarse<T>,
) -> DeviceCoarse<T> {
    assert_eq!(r.sites * r.dim, n);
    crate::solver::preconditioner::multigrid::count::dense_apply(
        n,
        r.vectors,
        std::mem::size_of::<T>(),
    );
    let out = r.zeros_like();
    // SAFETY: the fields of the last level have the layout of one site of all their complex
    // numbers (`coarse_index` of q = X dim + k), and m is n × n.
    unsafe { T::dense_apply(out.data.ptr, r.data.ptr, m.ptr, n as i32, r.vectors as i32) };
    out
}

fn site_index(c: &[usize; ND], ext: &[usize; ND]) -> usize {
    (0..ND).rev().fold(0, |s, mu| s * ext[mu] + c[mu])
}

/// The super-blocks and local vectors of one level (`Level` of the CPU).
struct Level<T> {
    /// The lattice of the level: its super-blocks, cut between the processes as the lattice
    /// above (a super-block within a process).
    lat: Arc<Lattice>,
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
    /// The level of super-blocks of `super_block` sites in every direction of the lattice above
    /// (`parent`, with `pd` components a site), and test vectors there.
    fn build(
        parent: &Arc<Lattice>,
        pd: usize,
        super_block: usize,
        tests: &DeviceCoarse<T>,
    ) -> Result<Self, Box<dyn Error>> {
        assert!(
            parent.local.iter().all(|e| e % super_block == 0),
            "super-blocks of {super_block} do not divide the extent {:?} of a process",
            parent.local
        );
        assert!((tests.sites, tests.dim) == (parent.num_sites(), pd));
        let lat = Lattice::new(
            parent.global.map(|e| e / super_block),
            parent.grid,
            parent.comm.clone(),
        );
        let blocks = lat.num_sites();
        let bs = super_block.pow(ND as u32);
        // The sites of every super-block in the order of their global index above.
        let mut sites: Vec<Vec<(usize, usize)>> = vec![Vec::with_capacity(bs); blocks];
        for x in 0..parent.num_sites() {
            let c = parent.coords(x);
            sites[lat.site(c.map(|v| v / super_block))]
                .push((site_index(&parent.global_coords(x), &parent.global), x));
        }
        let mut members = vec![0u32; blocks * bs];
        let mut owner = vec![0u32; parent.num_sites()];
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
        crate::solver::preconditioner::multigrid::count::level_basis(blocks, bs * pd, nq);
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
                parent.num_sites() as i32,
                pd as i32,
            );
        }
        let mut q = DeviceBuffer::<T>::uninitialised(q64.len)?;
        convert(&mut q, &q64);
        Ok(Self {
            lat,
            q,
            members,
            owner: DeviceTable::from_host(&owner)?,
            blocks,
            nq,
            bs,
            parent_sites: parent.num_sites(),
            pd,
        })
    }

    /// Counts Q† v or Q y (`meter`): every complex number of the level above times the `nq`
    /// columns of its block, 8 flops each; Q read once, the two fields (`reals`) once.
    fn count(&self, vectors: usize) {
        let size = std::mem::size_of::<T>();
        crate::solver::preconditioner::multigrid::count::level_transfer(
            self.parent_sites,
            self.pd,
            self.nq,
            vectors,
            size,
        );
    }

    /// Q† v.
    fn restrict(&self, v: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        assert!((v.sites, v.dim) == (self.parent_sites, self.pd));
        let out = must(DeviceCoarse::zeros(self.blocks, self.nq, v.vectors)).on(&self.lat);
        self.count(v.vectors);
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
        self.count(y.vectors);
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

/// The operator of `level` from that of the level above (`apply`), by probing as
/// `LevelOp::probe` of the CPU: sources c_μ apart, c_μ the smallest divisor of the extent from 3
/// up (or the extent), share no neighbour. Also the applications of the operator above. The
/// colours go by the global coordinates, so a ghost (a site of the neighbouring process) has
/// the colour of the site it stands for.
fn probe<T: LevelReal>(
    level: &Level<T>,
    apply: &mut dyn FnMut(&mut DeviceCoarse<T>, &DeviceCoarse<T>),
) -> Result<(DeviceCoarseOp<T>, Probed, usize), Box<dyn Error>> {
    let nq = level.nq;
    assert!(
        nq <= 1024,
        "a level has at most 1024 complex numbers a site on the device"
    );
    let lat = &level.lat;
    let ext = lat.global;
    let total = lat.num_sites();
    let (neighbours, ghosts) = CoarseGhosts::tables(lat)?;
    // A direction of extent 1 is in the block; of extent 2 its two hops are the same one.
    let used = (0..2 * ND)
        .filter(|&d| {
            let e = ext[d % ND];
            !(e == 1 || (e == 2 && d >= ND))
        })
        .fold(0u32, |m, d| m | (1 << d));
    let colours: [usize; ND] = ext.map(|e| (3..=e).find(|k| e % k == 0).unwrap_or(e).max(1));
    let n_colours: usize = colours.iter().product();
    let colour = |c: [usize; ND]| {
        (0..ND)
            .rev()
            .fold(0, |s, mu| s * colours[mu] + c[mu] % colours[mu]) as u32
    };
    let mut colour_of: Vec<u32> = (0..total).map(|t| colour(lat.global_coords(t))).collect();
    if let Some(g) = &ghosts {
        colour_of.extend(g.ghost_coords.iter().map(|&c| colour(c)));
    }
    let colour_table = DeviceTable::from_host(&colour_of)?;
    let neighbour_table = DeviceTable::from_host(&neighbours)?;
    let matrix = 2 * nq * nq;
    let diag = DeviceBuffer::<f64>::zeros(matrix * total)?;
    let links = DeviceBuffer::<f64>::zeros(2 * ND * matrix * total)?;
    let e = must(DeviceCoarse::<T>::zeros(total, nq, nq)).on(lat);
    let mut p = must(DeviceCoarse::<T>::zeros(level.parent_sites, level.pd, nq));
    let mut ap = p.zeros_like();
    let mut applications = 0;
    for colour in 0..n_colours as u32 {
        crate::solver::preconditioner::multigrid::count::copy(e.data.len, std::mem::size_of::<T>());
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
        crate::solver::preconditioner::multigrid::count::copy(r.data.len, std::mem::size_of::<T>());
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
    let op = DeviceCoarseOp::from_parts(diag, &links, &neighbours, &vec![0; total], nq, false)?
        .with_ghosts(ghosts.map(Rc::new));
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
        lat: &Arc<Lattice>,
        vectors: &DeviceCoarse<T>,
        super_blocks: &[usize],
        bootstrap: &[LevelBootstrap],
    ) -> Result<Self, Box<dyn Error>> {
        assert!(!super_blocks.is_empty() && vectors.vectors > 0);
        assert_eq!(vectors.sites, lat.num_sites());
        let level = Level::build(lat, a.dim(), super_blocks[0], vectors)?;
        let (op, probed, applications) = probe(&level, &mut |out, inp| a.apply(out, inp, false))?;
        let below = level.restrict(vectors);
        let mut me = Self {
            levels: vec![level],
            ops: vec![op],
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
        let mut tests = tests.select(&(0..tests.vectors).collect::<Vec<_>>());
        for &sb in &super_blocks[from..] {
            let parent = self.levels.last().expect("a level above").lat.clone();
            let above = self.ops.last().expect("a level above");
            let level = Level::build(&parent, above.dim(), sb, &tests)?;
            let (op, probed, _) = probe(&level, &mut |out, inp| above.apply(out, inp, false))?;
            tests = level.restrict(&tests);
            self.levels.push(level);
            self.ops.push(op);
            self.last = Some(probed);
        }
        // The coarsest level dense, inverted by LU in double precision.
        let last = self.ops.last().expect("at least one level");
        let probed = self.last.as_ref().expect("the links of the last level");
        let lat = &self.levels.last().expect("at least one level").lat;
        let (dense, n) = dense_of(last, probed, lat)?;
        crate::solver::preconditioner::multigrid::count::invert_blocks(1, n);
        self.coarsest_inv = Some(dense_map(&dense, n, "the inverse", |out, a, n| {
            // SAFETY: two n × n matrices.
            unsafe { bridge_cuda_dense_inverse(out, a, n) }
        })?);
        self.coarsest_dim = n;
        Ok(())
    }

    /// The dense inverse of the last level applied to `r`.
    fn coarsest(&self, r: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        let inverse = self.coarsest_inv.as_ref().expect("the coarsest inverse");
        let lat = &self.levels.last().expect("at least one level").lat;
        dense_apply(inverse, self.coarsest_dim, r, lat)
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

/// The cutoff of `abs_inverse` of the CPU: eigenvalues not above it times the largest in
/// magnitude come from dependent local vectors and are left out.
const ABS_CUTOFF: f64 = 1e-10;

/// The levels below the coarse lattice for the Hermitian form H = A Γ5 of the Pauli-Villars
/// factor on the device, `AbsMultilevel` of the CPU: the Hermitian positive preconditioner
///
///   M_ℓ = |B_ℓ|⁻¹ + Q_(ℓ+1) M_(ℓ+1) Q_(ℓ+1)†,    M = |H|⁻¹ exactly on the last level,
///
/// of which `add` adds Q₂ M₂ Q₂† r to the |B₁|⁻¹ r of the coarse lattice itself.
pub struct DeviceAbsMultilevel<T: LevelReal> {
    levels: Vec<Level<T>>,
    /// |B_ℓ|⁻¹ of every level but the last, and the operators of those levels.
    abs_blocks: Vec<DeviceBuffer<T>>,
    ops: Vec<DeviceCoarseOp<T>>,
    coarsest_abs_inv: DeviceBuffer<T>,
    coarsest_dim: usize,
    /// Applications of the coarse operator to build it.
    pub setup_applications: usize,
}

impl<T: LevelReal> DeviceAbsMultilevel<T> {
    /// The levels for H = A Γ5 of `a`, applied as Γ5 (Γ5 A Γ5), on the coarse lattice `lat`, from
    /// the test vectors `vectors` there, with super-blocks of `super_blocks[ℓ]` sites per
    /// direction on level ℓ + 2.
    pub fn build(
        a: &DeviceCoarseOp<T>,
        lat: &Arc<Lattice>,
        vectors: &DeviceCoarse<T>,
        super_blocks: &[usize],
    ) -> Result<Self, Box<dyn Error>> {
        assert!(!super_blocks.is_empty() && vectors.vectors > 0);
        assert_eq!(vectors.sites, lat.num_sites());
        let (mut parent, mut pd) = (lat.clone(), a.dim());
        let mut tests = vectors.select(&(0..vectors.vectors).collect::<Vec<_>>());
        let (mut levels, mut ops) = (Vec::new(), Vec::<DeviceCoarseOp<T>>::new());
        let mut last: Option<Probed> = None;
        let mut setup_applications = 0;
        for (l, &sb) in super_blocks.iter().enumerate() {
            let level = Level::build(&parent, pd, sb, &tests)?;
            let (op, probed, applications) = if l == 0 {
                probe(&level, &mut |out, inp| {
                    a.apply(out, inp, true);
                    a.gamma5(out);
                })?
            } else {
                let above = ops.last().expect("a level above");
                probe(&level, &mut |out, inp| above.apply(out, inp, false))?
            };
            if l == 0 {
                setup_applications = applications;
            }
            tests = level.restrict(&tests);
            (parent, pd) = (level.lat.clone(), op.dim());
            levels.push(level);
            ops.push(op);
            last = Some(probed);
        }
        let probed = last.expect("at least one level");
        let (dense, n) = dense_of(ops.last().expect("at least one level"), &probed, &parent)?;
        crate::solver::preconditioner::multigrid::count::abs_inverse_blocks(1, n);
        let coarsest_abs_inv = dense_map(&dense, n, "|H|⁻¹", |out, a, n| {
            // SAFETY: two n × n matrices.
            unsafe { bridge_cuda_dense_abs_inverse(out, a, n, ABS_CUTOFF) }
        })?;
        // |B_ℓ|⁻¹ of the levels but the last, by the Jacobi method on the device.
        let mut abs_blocks = Vec::new();
        for op in &ops[..ops.len() - 1] {
            let diag = op.diag_f64();
            crate::solver::preconditioner::multigrid::count::abs_inverse_blocks(
                op.sites(),
                op.dim(),
            );
            let out = DeviceBuffer::<f64>::uninitialised(diag.len)?;
            let scratch = DeviceBuffer::<f64>::uninitialised(2 * diag.len)?;
            // SAFETY: the blocks of the level's sites, as many places for the results, and the
            // scratch the kernel asks for.
            unsafe {
                bridge_cuda_blocks_abs_inverse(
                    out.ptr,
                    diag.ptr,
                    scratch.ptr,
                    op.sites() as i32,
                    op.dim() as i32,
                    ABS_CUTOFF,
                    STREAM,
                );
            }
            finish();
            let mut converted = DeviceBuffer::<T>::uninitialised(out.len)?;
            convert(&mut converted, &out);
            abs_blocks.push(converted);
        }
        Ok(Self {
            levels,
            abs_blocks,
            ops,
            coarsest_abs_inv,
            coarsest_dim: n,
            setup_applications,
        })
    }

    /// Complex components of the coarsest level, solved densely.
    pub fn coarsest_dim(&self) -> usize {
        self.coarsest_dim
    }

    /// M of the lattice the level `l` restricts to, applied to `r` there.
    fn m(&self, l: usize, r: &DeviceCoarse<T>) -> DeviceCoarse<T> {
        if l + 1 == self.levels.len() {
            let lat = &self.levels[l].lat;
            return dense_apply(&self.coarsest_abs_inv, self.coarsest_dim, r, lat);
        }
        let below = &self.levels[l + 1];
        let mut z = r.zeros_like();
        self.ops[l].apply_blocks(&mut z, r, &self.abs_blocks[l]);
        below.prolong(&mut z, &self.m(l + 1, &below.restrict(r)), true);
        z
    }

    /// `out += Q₂ M₂ Q₂† inp`, all the vectors of the field at once.
    pub fn add(&self, out: &mut DeviceCoarse<T>, inp: &DeviceCoarse<T>) {
        let first = &self.levels[0];
        let z = self.m(0, &first.restrict(inp));
        first.prolong(out, &z, true);
    }
}

impl<T: LevelReal> DeviceCoarseOp<T> {
    /// The exact adjoint A† of this operator, as an operator of its own (`CoarseOp::adjoint`
    /// of the CPU): its blocks A(X)†, and for every direction d the link B_d(Y)† from the site
    /// Y whose hop d comes from X, so (A† v)(X) = A(X)† v(X) + Σ_d B_d(Y)† v(Y). Made on the
    /// device; with the even-odd form where this has it, and the inverses of its blocks
    /// (A(X)⁻¹)†.
    pub fn adjoint(&self) -> Result<Self, Box<dyn Error>> {
        if let Some(ghosts) = self.ghosts() {
            return self.adjoint_across(ghosts);
        }
        let (sites, dim) = (self.sites(), self.dim());
        let (neighbours, odd) = self.tables();
        // Every hop is a permutation of the sites: Y = its inverse at X.
        let mut from = vec![0u32; 2 * ND * sites];
        for d in 0..2 * ND {
            for y in 0..sites {
                from[d * sites + neighbours[d * sites + y] as usize] = y as u32;
            }
        }
        let from_table = DeviceTable::from_host(&from)?;
        let diag = DeviceBuffer::<f64>::uninitialised(self.diag_f64().len)?;
        let links = DeviceBuffer::<T>::uninitialised(self.links().len)?;
        crate::solver::preconditioner::multigrid::count::copy(
            diag.len + links.len,
            std::mem::size_of::<T>(),
        );
        // SAFETY: the blocks and links of this lattice, as many places for those of the
        // adjoint, and the table of the sites they come from.
        unsafe {
            f64::coarse_adjoint(
                diag.ptr,
                self.diag_f64().ptr,
                std::ptr::null(),
                1,
                sites as i32,
                dim as i32,
            );
            T::coarse_adjoint(
                links.ptr,
                self.links().ptr,
                from_table.ptr,
                (2 * ND) as i32,
                sites as i32,
                dim as i32,
            );
        }
        Self::from_device_links(diag, links, &from, odd, dim, self.has_even_odd())
    }

    /// `adjoint` on a lattice cut between processes: Y = the site hop d of X comes from in the
    /// adjoint is X's neighbour the other way, maybe a ghost, whose links come from the
    /// neighbouring process (exchanged on the host, once); the links of the adjoint are their
    /// conjugate transposes, made on the host (a copy with signs: the same numbers as the
    /// kernel's).
    fn adjoint_across(&self, ghosts: &Rc<CoarseGhosts>) -> Result<Self, Box<dyn Error>> {
        let (sites, dim) = (self.sites(), self.dim());
        let (neighbours, odd) = self.tables();
        let matrix = 2 * dim * dim;
        let mut from = vec![0u32; 2 * ND * sites];
        for d in 0..2 * ND {
            let back = (d + ND) % (2 * ND);
            from[d * sites..(d + 1) * sites]
                .copy_from_slice(&neighbours[back * sites..(back + 1) * sites]);
        }
        // The links of every site, all directions together, and those of the ghosts.
        let links = self.links().download()?;
        let per_site = 2 * ND * matrix;
        let mut by_site = vec![0.0; sites * per_site];
        for x in 0..sites {
            for d in 0..2 * ND {
                let at = (d * sites + x) * matrix;
                by_site[x * per_site + d * matrix..x * per_site + (d + 1) * matrix]
                    .copy_from_slice(&links[at..at + matrix]);
            }
        }
        let ghost_links = ghosts.of_host(&by_site, per_site);
        let mut adjoint = vec![0.0; links.len()];
        for d in 0..2 * ND {
            for x in 0..sites {
                let y = from[d * sites + x] as usize;
                let m = if y < sites {
                    &by_site[y * per_site + d * matrix..y * per_site + (d + 1) * matrix]
                } else {
                    let g = y - sites;
                    &ghost_links[g * per_site + d * matrix..g * per_site + (d + 1) * matrix]
                };
                let out = &mut adjoint[(d * sites + x) * matrix..(d * sites + x + 1) * matrix];
                for i in 0..dim {
                    for j in 0..dim {
                        out[2 * (i * dim + j)] = m[2 * (j * dim + i)];
                        out[2 * (i * dim + j) + 1] = -m[2 * (j * dim + i) + 1];
                    }
                }
            }
        }
        let diag = DeviceBuffer::<f64>::uninitialised(self.diag_f64().len)?;
        // SAFETY: the blocks of this lattice and as many places for those of the adjoint.
        unsafe {
            f64::coarse_adjoint(
                diag.ptr,
                self.diag_f64().ptr,
                std::ptr::null(),
                1,
                sites as i32,
                dim as i32,
            );
        }
        let links = DeviceBuffer::<f64>::from_host(&adjoint)?;
        let mut converted = DeviceBuffer::<T>::uninitialised(links.len)?;
        convert(&mut converted, &links);
        Ok(
            Self::from_device_links(diag, converted, &from, odd, dim, self.has_even_odd())?
                .with_ghosts(Some(ghosts.clone())),
        )
    }
}
