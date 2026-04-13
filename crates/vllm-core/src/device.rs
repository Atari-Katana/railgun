//! Device abstraction for Railgun.
//!
//! A [`Device`] identifies *where* a tensor lives. Railgun is designed so that
//! scheduling and memory-management code can reason about device placement without
//! importing any GPU-driver crate — only [`vllm-cuda`] ever touches cudarc.

use serde::{Deserialize, Serialize};

/// Identifies the compute device that owns a piece of memory.
///
/// # Equality
///
/// Two `Device` values are equal if and only if they refer to the same physical device.
/// `Device::Cuda(0)` and `Device::Cuda(1)` are *not* equal.
///
/// # Cloning cost
///
/// `Device` is `Copy`; cloning is free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Device {
    /// Host (CPU) memory — always available.
    Cpu,
    /// NVIDIA CUDA device with the given ordinal (0-indexed).
    ///
    /// # Invariant
    ///
    /// The ordinal must be in range `0..N` where `N` is the number of CUDA
    /// devices visible to the process. Railgun does not enforce this at
    /// construction time; the first operation that touches the device will fail
    /// with an appropriate error.
    Cuda(u32),
}

impl Device {
    /// Returns `true` if this device is a CUDA GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::Device;
    /// assert!(Device::Cuda(0).is_cuda());
    /// assert!(!Device::Cpu.is_cuda());
    /// ```
    #[inline]
    pub const fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Returns `true` if this device is the host CPU.
    #[inline]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns the CUDA ordinal if this is a `Cuda` device, or `None` for `Cpu`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::Device;
    /// assert_eq!(Device::Cuda(2).ordinal(), Some(2));
    /// assert_eq!(Device::Cpu.ordinal(), None);
    /// ```
    #[inline]
    pub const fn ordinal(self) -> Option<u32> {
        match self {
            Device::Cuda(n) => Some(n),
            Device::Cpu => None,
        }
    }

    /// Returns a human-readable string suitable for log messages.
    ///
    /// # Examples
    ///
    /// ```
    /// use vllm_core::Device;
    /// assert_eq!(Device::Cpu.label(), "cpu");
    /// assert_eq!(Device::Cuda(0).label(), "cuda:0");
    /// ```
    pub fn label(self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(n) => format!("cuda:{n}"),
        }
    }

    /// The default CUDA device (ordinal 0).
    ///
    /// Convenience constant for single-GPU setups.
    pub const CUDA0: Device = Device::Cuda(0);
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

/// Convert `Device` to a candle `Device`.
///
/// # Errors
///
/// Returns [`CoreError::CudaUnavailable`] if `vllm-cuda` feature is off and a
/// CUDA device is requested. (In practice candle will return its own error
/// first; this wraps it for type uniformity.)
impl TryFrom<Device> for candle_core::Device {
    type Error = crate::CoreError;

    fn try_from(device: Device) -> crate::Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(n) => candle_core::Device::cuda_if_available(n as usize)
                .map_err(|e| crate::CoreError::DeviceInit {
                    device,
                    reason: e.to_string(),
                }),
        }
    }
}

impl From<&candle_core::Device> for Device {
    fn from(d: &candle_core::Device) -> Self {
        match d {
            candle_core::Device::Cpu => Device::Cpu,
            candle_core::Device::Cuda(dev) => {
                // cudarc CudaDevice exposes `ordinal()` via the CudaDevice struct
                // In candle 0.10.x CudaDevice wraps cudarc; we use a best-effort
                // approach: always map to cuda:0 if we can't introspect.
                // TODO: use dev.ordinal() once candle exposes it stably.
                let _ = dev;
                Device::Cuda(0)
            }
            // Future backends (Metal, etc.) fall back to CPU for now
            _ => Device::Cpu,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_properties() {
        let dev = Device::Cpu;
        assert!(!dev.is_cuda());
        assert!(dev.is_cpu());
        assert_eq!(dev.ordinal(), None);
        assert_eq!(dev.label(), "cpu");
        assert_eq!(dev.to_string(), "cpu");
    }

    #[test]
    fn cuda_properties() {
        let dev = Device::Cuda(3);
        assert!(dev.is_cuda());
        assert!(!dev.is_cpu());
        assert_eq!(dev.ordinal(), Some(3));
        assert_eq!(dev.label(), "cuda:3");
    }

    #[test]
    fn cuda0_constant() {
        assert_eq!(Device::CUDA0, Device::Cuda(0));
    }

    #[test]
    fn equality() {
        assert_eq!(Device::Cuda(0), Device::Cuda(0));
        assert_ne!(Device::Cuda(0), Device::Cuda(1));
        assert_ne!(Device::Cpu, Device::Cuda(0));
    }

    #[test]
    fn candle_cpu_conversion() {
        let candle_dev = candle_core::Device::try_from(Device::Cpu).unwrap();
        assert!(candle_dev.is_cpu());
        assert_eq!(Device::from(&candle_dev), Device::Cpu);
    }
}
