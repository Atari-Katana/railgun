//! Error types for the `vllm-core` crate.
//!
//! All public functions in Railgun's library crates return typed errors so that
//! callers can pattern-match on specific failure modes. Application code (the
//! CLI and engine) converts these to `anyhow::Error` for ergonomic propagation.
//!
//! # Design rationale
//!
//! Using `thiserror` (rather than a single `Box<dyn Error>`) keeps errors
//! zero-allocation on the happy path and allows the scheduler to make
//! policy decisions based on error kind (e.g. retry on `OutOfMemory`).

use thiserror::Error;

use crate::device::Device;
use crate::dtype::DType;

/// Errors originating in `vllm-core`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A memory allocation request could not be satisfied.
    ///
    /// This is returned when GPU or CPU memory is exhausted. The scheduler
    /// should react by preempting lower-priority requests rather than
    /// propagating this error to the end user.
    #[error("out of memory on {device}: requested {requested_bytes} bytes")]
    OutOfMemory {
        /// The device that ran out of memory.
        device: Device,
        /// How many bytes were requested.
        requested_bytes: usize,
    },

    /// A tensor shape was invalid for the requested operation.
    ///
    /// This should never occur in production — it indicates a programming bug.
    #[error("invalid shape for {op}: expected {expected:?}, got {got:?}")]
    InvalidShape {
        /// The operation that detected the mismatch.
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Two tensors that must live on the same device do not.
    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch {
        expected: Device,
        got: Device,
    },

    /// The requested dtype is not supported for this operation or backend.
    #[error("dtype {dtype} is not supported in context '{context}'")]
    UnsupportedDType {
        dtype: DType,
        context: &'static str,
    },

    /// A CUDA device could not be initialised.
    ///
    /// This wraps the underlying driver error as a string to avoid a
    /// dependency on `cudarc` types in `vllm-core`.
    #[error("failed to initialise {device}: {reason}")]
    DeviceInit {
        device: Device,
        reason: String,
    },

    /// An IO error occurred (e.g. while loading model weights).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialisation/parse error.
    #[error("JSON error: {0}")]
    Json(String),

    /// A candle-core or other tensor error (message preserved).
    #[error("tensor error: {0}")]
    Tensor(String),
}

impl From<candle_core::Error> for CoreError {
    fn from(e: candle_core::Error) -> Self {
        CoreError::Tensor(e.to_string())
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        CoreError::Json(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_out_of_memory() {
        let e = CoreError::OutOfMemory {
            device: Device::Cuda(0),
            requested_bytes: 1_073_741_824,
        };
        let msg = e.to_string();
        assert!(msg.contains("out of memory"));
        assert!(msg.contains("cuda:0"));
        assert!(msg.contains("1073741824"));
    }

    #[test]
    fn display_device_mismatch() {
        let e = CoreError::DeviceMismatch {
            expected: Device::Cuda(0),
            got: Device::Cpu,
        };
        let msg = e.to_string();
        assert!(msg.contains("device mismatch"));
        assert!(msg.contains("cuda:0"));
        assert!(msg.contains("cpu"));
    }

    #[test]
    fn display_unsupported_dtype() {
        let e = CoreError::UnsupportedDType {
            dtype: DType::Bool,
            context: "candle DType conversion",
        };
        let msg = e.to_string();
        assert!(msg.contains("bool"));
        assert!(msg.contains("candle DType conversion"));
    }

    #[test]
    fn io_error_conversion() {
        let io_e = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let core_e: CoreError = io_e.into();
        assert!(matches!(core_e, CoreError::Io(_)));
    }
}
