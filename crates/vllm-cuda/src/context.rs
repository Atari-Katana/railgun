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

use candle_core::CudaDevice;
use candle_core::backend::BackendDevice;
pub use candle_core::cuda_backend::cudarc::driver::CudaContext;
use candle_core::cuda_backend::cudarc::driver::result as cudarc_result;
use thiserror::Error;
use tracing::debug;

/// Errors specific to CUDA device operations.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The CUDA device could not be initialised.
    #[error("CUDA init failed for device {ordinal}: {source}")]
    Init {
        ordinal: u32,
        source: candle_core::Error,
    },

    /// A CUDA runtime operation failed.
    #[error("CUDA operation failed: {0}")]
    Driver(#[from] candle_core::cuda_backend::cudarc::driver::DriverError),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type CudaResult<T> = std::result::Result<T, CudaError>;

use crate::kernels::PagedAttentionKernels;

/// Owns a CUDA device and its primary stream.
pub struct RailgunCudaContext {
    candle_dev: CudaDevice,
    kernels: PagedAttentionKernels,
    ordinal: u32,
}

impl RailgunCudaContext {
    pub fn new(ordinal: u32) -> CudaResult<Self> {
        debug!(%ordinal, "creating CUDA context via candle");
        let candle_dev = CudaDevice::new(ordinal as usize)?;
        
        let stream = candle_dev.cuda_stream();
        let _context = stream.context();
        
        let kernels = PagedAttentionKernels::new(&candle_dev, ordinal as usize)
            .map_err(|e| CudaError::Init { ordinal, source: candle_core::Error::Msg(e.to_string()) })?;

        Ok(Self {
            candle_dev,
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

    pub fn candle_device(&self) -> &CudaDevice {
        &self.candle_dev
    }

    pub fn context(&self) -> Arc<CudaContext> {
        self.candle_dev.cuda_stream().context().clone()
    }

    pub fn stream(&self) -> Arc<candle_core::cuda_backend::cudarc::driver::CudaStream> {
        self.candle_dev.cuda_stream()
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        self.candle_dev.synchronize().map_err(CudaError::Candle)
    }

    pub fn memory_info(&self) -> CudaResult<(usize, usize)> {
        cudarc_result::mem_get_info().map_err(CudaError::Driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test is skipped automatically in CPU-only CI by checking whether
    /// `RailgunCudaContext::new(0)` returns an error.
    #[test]
    fn create_context_or_skip() {
        match RailgunCudaContext::new(0) {
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
