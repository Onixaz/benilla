fn main() {
    println!("cargo::rerun-if-env-changed=DLSS_SDK");

    // Only `--features dlss` (this crate's `ngx`) links the SDK; a workspace build compiles the
    // NGX calls as failures and needs neither the SDK nor Windows.
    if std::env::var_os("CARGO_FEATURE_NGX").is_none() {
        return;
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        panic!("benilla-dlss5 is supported on Windows only");
    }

    let sdk = std::env::var("DLSS_SDK").unwrap_or_else(|_| {
        panic!(
            "DLSS_SDK is required for --features dlss; point it at an NVIDIA NGX SDK containing nvsdk_ngx_d.lib"
        )
    });
    let lib = std::path::Path::new(&sdk).join("lib/Windows_x86_64/x64");
    if !lib.join("nvsdk_ngx_d.lib").is_file() {
        panic!(
            "DLSS_SDK has no lib/Windows_x86_64/x64/nvsdk_ngx_d.lib: {}",
            lib.display()
        );
    }

    println!("cargo::rustc-link-search=native={}", lib.display());
    println!("cargo::rustc-link-lib=static=nvsdk_ngx_d");
    for lib in ["advapi32", "user32", "shell32", "ole32", "delayimp"] {
        println!("cargo::rustc-link-lib={lib}");
    }

    build_bridge();
}

/// Compiles `benilla-nvngx`'s one dependency-free source into `OUT_DIR/benilla_nvngx.dll`, which
/// `ngx.rs` embeds: a cdylib cannot be a cargo dependency, and its file name is load-bearing.
fn build_bridge() {
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("../benilla-nvngx/src/lib.rs");
    println!("cargo::rerun-if-changed={}", source.display());
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let target = std::env::var("TARGET").unwrap();
    let status = std::process::Command::new(rustc)
        .args(["--edition", "2021", "--crate-type", "cdylib"])
        .args(["--crate-name", "benilla_nvngx", "--target", &target])
        .args(["-C", "opt-level=2", "--out-dir"])
        .arg(&out)
        .arg(&source)
        .status()
        .expect("could not run rustc for benilla_nvngx.dll");
    if !status.success() {
        panic!("rustc failed to build benilla_nvngx.dll ({status})");
    }
}
