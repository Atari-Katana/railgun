use std::process::Command;
use std::path::Path;

macro_rules! info {
    ($($tokens:tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() {
    let kernels = vec!["paged_attention", "rope"];
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for kernel in kernels {
        let src_file = format!("src/kernels/{}.cu", kernel);
        let out_file = Path::new(&out_dir).join(format!("{}.ptx", kernel));
        
        println!("cargo:rerun-if-changed={}", src_file);

        let output = Command::new("nvcc")
            .arg("-ptx")
            .arg("-o")
            .arg(&out_file)
            .arg("-arch=native")
            .arg("-ccbin")
            .arg("/usr/bin/gcc-14")
            .arg("-allow-unsupported-compiler")
            .arg(&src_file)
            .arg("-I/usr/local/cuda/include")
            .output();

        match output {
            Ok(out) if out.status.success() => {
                info!("Compiled {} to PTX", kernel);
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                println!("cargo:warning=nvcc failed for {}: {}", kernel, err);
            }
            Err(e) => {
                println!("cargo:warning=Failed to execute nvcc for {}: {}", kernel, e);
            }
        }
    }
}
