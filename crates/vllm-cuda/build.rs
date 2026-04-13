use std::process::Command;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/paged_attention.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_file = Path::new(&out_dir).join("paged_attention.ptx");

    let output = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&out_file)
        .arg("-arch=native")
        .arg("-ccbin")
        .arg("/usr/bin/gcc-14")
        .arg("-allow-unsupported-compiler")
        .arg("src/kernels/paged_attention.cu")
        .arg("-I/usr/local/cuda/include")
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("cargo:rustc-env=PAGED_ATTENTION_PTX={}", out_file.display());
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr);
            println!("cargo:warning=nvcc failed: {}", err);
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute nvcc: {}", e);
        }
    }
}
