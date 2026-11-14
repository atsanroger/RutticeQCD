//! With the feature `cuda-bridge`: builds the CUDA library of `cuda_bridge/` with cmake and
//! links it and the CUDA runtime. Without it: nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CUDA_BRIDGE").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=cuda_bridge");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=RUTTICE_CUDA_ARCH");
    // The toolkit: CUDA_HOME, or the usual place.
    let cuda = std::env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda.join("bin/nvcc");
    assert!(
        nvcc.exists(),
        "no nvcc at {}: set CUDA_HOME to the CUDA toolkit",
        nvcc.display()
    );
    let source = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("cuda_bridge");
    let build = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("cuda_bridge");
    std::fs::create_dir_all(&build).unwrap();
    let arch = architectures();
    println!("cargo:warning=CUDA kernels compiled for sm {arch} (RUTTICE_CUDA_ARCH to change)");
    let run = |command: &mut Command| {
        let status = command.status().expect("cmake runs");
        assert!(status.success(), "{command:?} failed");
    };
    run(Command::new("cmake")
        .arg(&source)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={arch}"))
        .arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.display()))
        .current_dir(&build));
    run(Command::new("cmake")
        .args(["--build", ".", "--target", "bridge_cuda_static", "-j"])
        .current_dir(&build));
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=static=bridge_cuda_static");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    // The dense matrices of the last level below the coarse lattice.
    println!("cargo:rustc-link-lib=dylib=cusolver");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // NCCL for the exchanges between the devices: NCCL_HOME (lib, include), or the system's.
    if std::env::var_os("CARGO_FEATURE_NCCL").is_some() {
        println!("cargo:rerun-if-env-changed=NCCL_HOME");
        if let Some(home) = std::env::var_os("NCCL_HOME") {
            let lib = PathBuf::from(home).join("lib");
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
        }
        println!("cargo:rustc-link-lib=dylib=nccl");
    }
}

/// The architectures of the GPUs, as a cmake list, from `RUTTICE_CUDA_ARCH` (it can be set in
/// the `[env]` table of `.cargo/config.toml`):
/// - unset or `auto`: those of the GPUs `nvidia-smi` sees, so a build compiles only what this
///   machine runs; `ALL` where it sees none (a login node);
/// - `all`: `ALL`;
/// - anything else: passed to cmake as it is, e.g. `80` or `80;90`.
fn architectures() -> String {
    // 80 the A100, 86 the RTX 30 series, 90 the H100. Without its own code a GPU
    // compiles the PTX of a lower architecture at its first start, with the register limits
    // ptxas chose for that one.
    const ALL: &str = "80;86;90";
    let asked = std::env::var("RUTTICE_CUDA_ARCH").unwrap_or_default();
    match asked.trim() {
        "" | "auto" => detected().unwrap_or_else(|| {
            println!("cargo:warning=nvidia-smi sees no GPU: compiling for all of {ALL}");
            ALL.to_string()
        }),
        "all" => ALL.to_string(),
        list => list.to_string(),
    }
}

/// The compute capabilities of the GPUs of this machine as cmake architectures ("8.0" is 80),
/// each once; `None` without `nvidia-smi` or a GPU.
fn detected() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|out| out.status.success())?;
    let mut found: Vec<u32> = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((major, minor)) = line.trim().split_once('.') else {
            continue;
        };
        let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u32>()) else {
            continue;
        };
        let sm = 10 * major + minor;
        // The kernels use the tensor-core instructions of sm_80 (mma in f64 and tf32).
        assert!(
            sm >= 80,
            "this machine has a GPU of compute capability {major}.{minor}; the kernels need 8.0 \
             or above (set RUTTICE_CUDA_ARCH to build for another machine)"
        );
        if !found.contains(&sm) {
            found.push(sm);
        }
    }
    found.sort_unstable();
    (!found.is_empty()).then(|| {
        found
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(";")
    })
}
