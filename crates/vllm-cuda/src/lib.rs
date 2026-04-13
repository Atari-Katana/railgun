//! # vllm-cuda
//!
//! CUDA backend for Railgun. Provides:
//!
//! - [`context`]: CUDA device context and stream management.
//! - [`allocator`]: Type-safe device memory buffers.
//! - [`kernels`]: PTX kernel loaders (populated in later phases).
//!
//! ## Feature flag
//!
//! This entire crate is gated behind the `cuda` feature. When the feature is
//! disabled, the crate compiles to an empty stub so that workspace builds
//! succeed in CPU-only environments.

#[cfg(feature = "cuda")]
pub mod allocator;
#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub use allocator::DeviceBuffer;
#[cfg(feature = "cuda")]
pub use context::RailgunCudaContext as CudaContext;
#[cfg(feature = "cuda")]
pub use kernels::PagedAttentionKernels;
