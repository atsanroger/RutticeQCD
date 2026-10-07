//! Checks: writing the output of a sweep with stores that bypass the cache gives the same
//! field as ordinary stores.
//!
//! Which stores a sweep uses depends on the size of its output, through a threshold that is
//! read once per process. So the comparison is between two runs of the benchmark program,
//! one with the threshold at zero and one with it out of reach: the norms it prints, for
//! both precisions, both ways of keeping the links and the fields in blocks, must agree to
//! every digit printed.

use std::process::Command;

/// The rows of the benchmark's table as (name, |D psi|^2 as printed).
fn norms(stream_min_bytes: &str) -> Vec<(String, String)> {
    let output = Command::new(env!("CARGO_BIN_EXE_ruttice-bench"))
        .args(["8", "8", "8", "8", "2"])
        .env("RUTTICE_STREAM_MIN_BYTES", stream_min_bytes)
        .env("RAYON_NUM_THREADS", "3")
        .output()
        .expect("the benchmark program runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("the table is text")
        .lines()
        .filter(|l| l.starts_with("double") || l.starts_with("single"))
        .map(|l| {
            let name = l[..19].trim().to_string();
            let norm = l.split_whitespace().last().unwrap().to_string();
            (name, norm)
        })
        .collect()
}

#[test]
fn streaming_stores_do_not_change_the_result() {
    let streamed = norms("0");
    let ordinary = norms(&usize::MAX.to_string());
    assert!(streamed.len() >= 8, "{streamed:?}");
    assert_eq!(streamed, ordinary);
    for (name, norm) in &streamed {
        assert!(norm.parse::<f64>().unwrap() > 0.0, "{name}: {norm}");
    }
}
