/// Communication between processes. Only what a single node needs so far; halo
/// exchange and general messaging arrive with the MPI implementation.
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
