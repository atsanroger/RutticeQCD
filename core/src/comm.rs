/// Communication between processes. Stage 0 has only what a single node needs; halo
/// exchange and general messaging (DESIGN.md §4) arrive with the MPI implementation.
pub trait Comm: Send + Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// Global reduction: replace each element with its sum over all processes.
    fn sum(&self, x: &mut [f64]);
}

pub struct SingleComm;

impl Comm for SingleComm {
    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        1
    }
    fn sum(&self, _x: &mut [f64]) {}
}
