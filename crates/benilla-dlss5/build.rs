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
}
