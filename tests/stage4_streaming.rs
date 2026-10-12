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

/// The pion correlator a domain-wall run prints, as text.
fn domain_wall_pion(dir: &std::path::Path, stream_min_bytes: &str) -> Vec<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_ruttice"))
        .arg(dir.join("run.yaml"))
        .env("RUTTICE_STREAM_MIN_BYTES", stream_min_bytes)
        .env("RAYON_NUM_THREADS", "3")
        .output()
        .expect("the program runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("the report is text")
        .lines()
        .skip_while(|l| l.trim() != "PS:")
        .skip(1)
        .take_while(|l| l.starts_with("    "))
        .map(str::to_string)
        .collect()
}

/// The even-odd domain-wall operator does C⁻¹ and the difference with its input on each
/// site its sweeps have just written (`After`). With stores that bypass the cache the site
/// is computed in a buffer and written once; that gives the same correlator, digit by digit.
#[test]
fn streaming_stores_do_not_change_a_domain_wall_run() {
    use ruttice::gauge::random_gauge;
    use ruttice::Lattice;
    let dir = std::env::temp_dir().join(format!("ruttice_stream_dw_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lat = Lattice::single_node([4, 4, 4, 8]);
    ruttice::io::write_bridge_text(&dir.join("conf.txt"), &random_gauge(&lat, 201)).unwrap();
    let text = "Main:\n  lattice_size : [4, 4, 4, 8]\nGauge:\n  config_filename_input : conf.txt\nFopr:\n  fermion_type : Mobius\n  quark_mass : 0.1\n  domain_wall_height : 1.4\n  extent_of_5th_dimension : 4\nSolver:\n  solver_type : BiCGStab\n  convergence_criterion_squared : 1.0e-24\n  maximum_number_of_iteration : 4000\nSource:\n  source_position : [0, 0, 0, 0]\n  number_of_rhs_per_solve : 1\n";
    std::fs::write(dir.join("run.yaml"), text).unwrap();
    let streamed = domain_wall_pion(&dir, "0");
    let ordinary = domain_wall_pion(&dir, &usize::MAX.to_string());
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(streamed.len(), 8, "{streamed:?}");
    assert_eq!(streamed, ordinary);
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
