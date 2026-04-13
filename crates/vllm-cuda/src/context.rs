//! CUDA device context and stream management.
//!
//! [`CudaContext`] owns the primary CUDA context for a single GPU device.
//! It wraps `cudarc::driver::CudaDevice` and adds Railgun-specific
//! error mapping and convenience methods.
//!
//! # Thread Safety
//!
//! `CudaContext` is `Send + Sync`: `cudarc::CudaDevice` handles the
//! internal synchronisation around driver calls. You may share a
//! `Arc<CudaContext>` across threads.
//!
//! # Example
//!
//! ```no_run
//! use vllm_cuda::CudaContext;
//! let ctx = CudaContext::new(0).expect("CUDA device 0 not available");
//! ctx.synchronize().expect("sync failed");
//! ```

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use thiserror::Error;
use tracing::debug;

/// Errors specific to CUDA device operations.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The CUDA device could not be initialised.
    #[error("CUDA init failed for device {ordinal}: {source}")]
    Init {
        ordinal: u32,
        source: cudarc::driver::DriverError,
    },

    /// A CUDA runtime operation failed.
    #[error("CUDA operation failed: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}

pub type CudaResult<T> = std::result::Result<T, CudaError>;

use crate::kernels::PagedAttentionKernels;

/// Owns a CUDA device and its primary stream.
pub struct CudaContext {
    device: Arc<CudaDevice>,
    kernels: PagedAttentionKernels,
    ordinal: u32,
}

impl CudaContext {
    pub fn new(ordinal: u32) -> CudaResult<Self> {
        debug!(%ordinal, "creating CUDA context");
        let device = Arc::new(CudaDevice::new(ordinal as usize).map_err(|e| CudaError::Init {
            ordinal,
            source: e,
        })?);
        
        let kernels = PagedAttentionKernels::new(device.clone(), ordinal as usize)
            .map_err(|e| CudaError::Driver(cudarc::driver::DriverError::Unknown(0)))?; // Simplified for now

        Ok(Self {
            device,
            kernels,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn kernels(&self) -> &PagedAttentionKernels {
        &self.kernels
    }

    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        self.device.synchronize().map_err(CudaError::Driver)
    }

    pub fn memory_info(&self) -> CudaResult<(usize, usize)> {
        self.device.memory_info().map_err(CudaError::Driver)
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext")
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test is skipped automatically in CPU-only CI by checking whether
    /// `CudaContext::new(0)` returns an error.
    #[test]
    fn create_context_or_skip() {
        match CudaContext::new(0) {
            Ok(ctx) => {
                assert_eq!(ctx.ordinal(), 0);
                ctx.synchronize().expect("synchronize failed");
                let (free, total) = ctx.memory_info().expect("memory_info failed");
                assert!(total > 0, "total memory should be positive");
                assert!(free <= total, "free <= total invariant");
                println!("GPU 0: {free}/{total} bytes free");
            }
            Err(e) => {
                println!("Skipping CUDA context test (no GPU): {e}");
            }
        }
    }
}
