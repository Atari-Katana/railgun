//! # vllm-core
//!
//! Core tensor types, device abstractions, and error handling for Railgun.
//!
//! This crate intentionally has no transitive dependency on GPU drivers — all GPU
//! functionality lives in `vllm-cuda`. This allows unit tests and CPU-only builds to
//! run without a CUDA installation.

pub mod device;
pub mod dtype;
pub mod error;
pub mod tensor;

pub use device::Device;
pub use dtype::DType;
pub use error::CoreError;
pub use tensor::Tensor;

/// Convenience alias for `Result<T, CoreError>`.
pub type Result<T> = std::result::Result<T, CoreError>;
