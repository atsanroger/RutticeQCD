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
    // The architecture of the GPU: 86 is the RTX 30 series; RUTTICE_CUDA_ARCH overrides it.
    let arch = std::env::var("RUTTICE_CUDA_ARCH").unwrap_or_else(|_| "86".to_string());
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
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
