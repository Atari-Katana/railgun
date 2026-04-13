//! Data type enumeration for tensors.
//!
//! [`DType`] mirrors the set of scalar types supported by both the CPU and CUDA
//! backends. It does not depend on any external tensor library so that it can be
//! used in scheduling and caching code without pulling in GPU dependencies.

use serde::{Deserialize, Serialize};

/// Scalar element type of a tensor.
///
/// # Ordering
///
/// The discriminant values are stable across releases; code that serialises
/// `DType` (e.g. for IPC) may depend on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float (half precision).
    F16,
    /// 16-bit brain float (Google's format; wider dynamic range than F16).
    BF16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Unsigned 8-bit integer (used for quantised weights and images).
    U8,
    /// Boolean (stored as 1 byte per element).
    Bool,
}

impl DType {
    /// Returns the number of bytes required to store one element of this dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::DType;
    /// assert_eq!(DType::F32.size_of(), 4);
    /// assert_eq!(DType::BF16.size_of(), 2);
    /// ```
    #[inline]
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::Bool => 1,
        }
    }

    /// Returns `true` if this dtype represents a floating-point type.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::DType;
    /// assert!(DType::BF16.is_float());
    /// assert!(!DType::I32.is_float());
    /// ```
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Returns `true` if this dtype represents an integer type.
    #[inline]
    pub const fn is_int(self) -> bool {
        !self.is_float() && !matches!(self, DType::Bool)
    }

    /// Human-readable name used in log output and error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Conversion from candle's DType to ours.
impl From<candle_core::DType> for DType {
    fn from(d: candle_core::DType) -> Self {
        match d {
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::F16 => DType::F16,
            candle_core::DType::BF16 => DType::BF16,
            candle_core::DType::I32 => DType::I32,
            candle_core::DType::I64 => DType::I64,
            candle_core::DType::U8 => DType::U8,
            candle_core::DType::U32 => DType::I32, // widen loss-less for Railgun
            _ => DType::F16, // FP8 / future types fall back to F16
        }
    }
}

/// Conversion from our DType to candle's DType.
impl TryFrom<DType> for candle_core::DType {
    type Error = crate::CoreError;

    fn try_from(d: DType) -> crate::Result<candle_core::DType> {
        match d {
            DType::F32 => Ok(candle_core::DType::F32),
            DType::F16 => Ok(candle_core::DType::F16),
            DType::BF16 => Ok(candle_core::DType::BF16),
            DType::I32 => Ok(candle_core::DType::I32),
            DType::I64 => Ok(candle_core::DType::I64),
            DType::U8 => Ok(candle_core::DType::U8),
            DType::Bool => Err(crate::CoreError::UnsupportedDType {
                dtype: DType::Bool,
                context: "candle DType conversion",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_of_all_variants() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::I64.size_of(), 8);
        assert_eq!(DType::U8.size_of(), 1);
        assert_eq!(DType::Bool.size_of(), 1);
    }

    #[test]
    fn float_classification() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn display_is_stable() {
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(format!("{}", DType::F32), "f32");
    }

    #[test]
    fn roundtrip_candle_dtype() {
        let pairs = [
            (DType::F32, candle_core::DType::F32),
            (DType::F16, candle_core::DType::F16),
            (DType::BF16, candle_core::DType::BF16),
        ];
        for (ours, theirs) in pairs {
            assert_eq!(DType::from(theirs), ours);
            assert_eq!(candle_core::DType::try_from(ours).unwrap(), theirs);
        }
    }
}
