//! Tensor newtype wrapping `candle_core::Tensor`.
//!
//! [`Tensor`] is Railgun's primary n-dimensional array type. It wraps
//! `candle_core::Tensor` to:
//!
//! 1. Return [`CoreError`] instead of candle's internal error type.
//! 2. Provide a stable public API that insulates the rest of Railgun from
//!    upstream candle API changes.
//! 3. Expose `Device` and `DType` from `vllm-core` rather than candle's
//!    equivalents, keeping dependency surfaces minimal.
//!
//! # Device placement
//!
//! A `Tensor` always belongs to exactly one *device*. Operations that combine
//! two tensors require both to live on the same device; if they do not,
//! [`CoreError::DeviceMismatch`] is returned.
//!
//! # Memory management
//!
//! Ownership follows `candle_core::Tensor` semantics: tensors are reference-
//! counted (cheap to clone); the underlying buffer is freed when the last
//! reference is dropped.

use candle_core::{DType as CDType, Device as CDevice, Tensor as CTensor};

use crate::{CoreError, Device, DType, Result};

/// An n-dimensional array of scalar values residing on a [`Device`].
///
/// # Cloning
///
/// Cloning a `Tensor` is cheap (reference-counted). Use [`Tensor::contiguous`]
/// if you need an independent copy of the underlying data.
#[derive(Debug, Clone)]
pub struct Tensor(pub(crate) CTensor);

impl Tensor {
    // ──────────────────────────────────────────────────────────────────────────
    // Constructors
    // ──────────────────────────────────────────────────────────────────────────

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` – Dimension sizes, e.g. `&[batch, seq_len, hidden]`.
    /// * `dtype` – Scalar element type.
    /// * `device` – Target device for the allocation.
    ///
    /// # Errors
    ///
    /// * [`CoreError::UnsupportedDType`] if `dtype` cannot be mapped to a
    ///   candle dtype (currently only `Bool`).
    /// * [`CoreError::DeviceInit`] if the target CUDA device is unavailable.
    /// * [`CoreError::OutOfMemory`] if the allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::{Tensor, DType, Device};
    /// let t = Tensor::zeros(&[2, 4], DType::F32, Device::Cpu).unwrap();
    /// assert_eq!(t.shape(), &[2, 4]);
    /// ```
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let cdtype = CDType::try_from(dtype)?;
        let cdevice = CDevice::try_from(device)?;
        let inner = CTensor::zeros(shape, cdtype, &cdevice)?;
        Ok(Self(inner))
    }

    /// Create a tensor filled with ones.
    ///
    /// Arguments and errors are identical to [`Tensor::zeros`].
    pub fn ones(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let cdtype = CDType::try_from(dtype)?;
        let cdevice = CDevice::try_from(device)?;
        let inner = CTensor::ones(shape, cdtype, &cdevice)?;
        Ok(Self(inner))
    }

    /// Create a 1-D tensor from a slice of `f32` values on the CPU.
    ///
    /// This is a convenience constructor for tests and host-side preprocessing.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Tensor`] if the allocation fails.
    pub fn from_slice_f32(data: &[f32], shape: &[usize], device: Device) -> Result<Self> {
        let cdevice = CDevice::try_from(device)?;
        let inner = CTensor::from_slice(data, shape, &cdevice)?;
        Ok(Self(inner))
    }

    /// Wrap an existing `candle_core::Tensor`.
    ///
    /// This is the escape hatch for code that must call candle APIs directly.
    /// Prefer the typed constructors when possible.
    #[inline]
    pub fn from_candle(inner: CTensor) -> Self {
        Self(inner)
    }

    /// Access the inner candle tensor.
    ///
    /// Prefer the Railgun API over reaching into the inner tensor whenever
    /// possible, to keep coupling to candle minimal.
    #[inline]
    pub fn inner(&self) -> &CTensor {
        &self.0
    }

    /// Consume the wrapper and return the inner candle tensor.
    #[inline]
    pub fn into_inner(self) -> CTensor {
        self.0
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Metadata
    // ──────────────────────────────────────────────────────────────────────────

    /// Returns the shape (dimension sizes) of the tensor.
    ///
    /// The returned slice has one element per dimension. A scalar tensor
    /// returns an empty slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.0.dims()
    }

    /// Returns the rank (number of dimensions) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Returns the total number of scalar elements.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.0.elem_count()
    }

    /// Returns the element dtype of this tensor.
    #[inline]
    pub fn dtype(&self) -> DType {
        DType::from(self.0.dtype())
    }

    /// Returns the device this tensor lives on.
    #[inline]
    pub fn device(&self) -> Device {
        Device::from(self.0.device())
    }

    /// Returns the number of bytes used by this tensor's data buffer.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype().size_of()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Operations
    // ──────────────────────────────────────────────────────────────────────────

    /// Returns a logically equivalent, physically contiguous tensor.
    ///
    /// If the tensor is already contiguous this is a no-op (reference clone).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Tensor`] if the operation fails.
    pub fn contiguous(&self) -> Result<Self> {
        Ok(Self(self.0.contiguous()?))
    }

    /// Move the tensor to a different device, copying data if necessary.
    ///
    /// If the tensor is already on `target`, returns a cheap clone.
    ///
    /// # Errors
    ///
    /// * [`CoreError::DeviceInit`] if the CUDA device is not available.
    /// * [`CoreError::Tensor`] if the copy fails (e.g., CUDA error).
    pub fn to_device(&self, target: Device) -> Result<Self> {
        let cdevice = CDevice::try_from(target)?;
        Ok(Self(self.0.to_device(&cdevice)?))
    }

    /// Cast the tensor to a different dtype.
    ///
    /// # Errors
    ///
    /// * [`CoreError::UnsupportedDType`] if `dtype` is not supported.
    /// * [`CoreError::Tensor`] if the cast fails.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        let cdtype = CDType::try_from(dtype)?;
        Ok(Self(self.0.to_dtype(cdtype)?))
    }

    /// Reshape the tensor to the given shape.
    ///
    /// The total number of elements must be preserved.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidShape`] if element counts differ.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let new_elems: usize = shape.iter().product();
        if new_elems != self.num_elements() {
            return Err(CoreError::InvalidShape {
                op: "reshape",
                expected: shape.to_vec(),
                got: self.shape().to_vec(),
            });
        }
        Ok(Self(self.0.reshape(shape)?))
    }

    /// Flatten dimensions `start_dim..=end_dim` into one.
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        Ok(Self(self.0.flatten(start_dim, end_dim)?))
    }

    /// Copy tensor data to a host `Vec<f32>`.
    ///
    /// Performs a device-to-host transfer if the tensor is on GPU.
    ///
    /// # Errors
    ///
    /// * [`CoreError::UnsupportedDType`] if the tensor is not F32.
    /// * [`CoreError::Tensor`] if the copy fails.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.0.to_vec1::<f32>().or_else(|_| {
            // Try casting to f32 first
            self.0
                .to_dtype(CDType::F32)
                .and_then(|t| t.to_vec1::<f32>())
        })?)
    }

    /// Copy a 1-D tensor to a host `Vec<u32>`.
    ///
    /// Useful for extracting token IDs.
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        Ok(self.0.to_vec1::<u32>()?)
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor[{}, {:?}, {}]",
            self.dtype(),
            self.shape(),
            self.device()
        )
    }
}

// Allow transparent access to candle ops through deref
// (disabled: explicit is better here; callers should use .inner())

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape_dtype_device() {
        let t = Tensor::zeros(&[2, 4], DType::F32, Device::Cpu).unwrap();
        assert_eq!(t.shape(), &[2, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.num_elements(), 8);
        assert_eq!(t.byte_size(), 32);
    }

    #[test]
    fn zeros_bf16() {
        let t = Tensor::zeros(&[1, 8], DType::BF16, Device::Cpu).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.byte_size(), 16); // 8 elements × 2 bytes
    }

    #[test]
    fn from_slice_round_trip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice_f32(&data, &[4], Device::Cpu).unwrap();
        let back = t.to_vec_f32().unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn reshape_valid() {
        let t = Tensor::zeros(&[2, 4], DType::F32, Device::Cpu).unwrap();
        let r = t.reshape(&[8]).unwrap();
        assert_eq!(r.shape(), &[8]);
    }

    #[test]
    fn reshape_invalid_size_returns_error() {
        let t = Tensor::zeros(&[2, 4], DType::F32, Device::Cpu).unwrap();
        let err = t.reshape(&[3]).unwrap_err();
        assert!(matches!(err, CoreError::InvalidShape { .. }));
    }

    #[test]
    fn display_is_informative() {
        let t = Tensor::zeros(&[4, 8], DType::BF16, Device::Cpu).unwrap();
        let s = t.to_string();
        assert!(s.contains("bf16"));
        assert!(s.contains("4"));
        assert!(s.contains("8"));
        assert!(s.contains("cpu"));
    }

    #[test]
    fn ones_has_correct_values() {
        let t = Tensor::ones(&[3], DType::F32, Device::Cpu).unwrap();
        let vals = t.to_vec_f32().unwrap();
        assert!(vals.iter().all(|&v| (v - 1.0f32).abs() < 1e-6));
    }
}
