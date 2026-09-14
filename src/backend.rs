use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    #[default]
    Default,
    Openvino,
    Cuda,
}

impl Runtime {
    pub const fn capability(self) -> &'static str {
        match self {
            Self::Default => "cpu",
            Self::Openvino => "openvino",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    #[default]
    Error,
    Cpu,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    pub kind: String,
    pub runtime: Runtime,
    pub device: String,
    pub threads: u16,
    pub fallback: Fallback,
    pub device_id: u32,
    /// Application-owned native library directories, searched before the
    /// process's ambient loader path.
    pub library_dirs: Vec<PathBuf>,
    /// Exact ONNX Runtime core library. Relative paths resolve from config.toml.
    pub onnxruntime_library: Option<PathBuf>,
    /// Exact external execution-provider library. Relative paths resolve from config.toml.
    pub provider_library: Option<PathBuf>,
    /// Exact OpenVINO C API library used by the direct OpenVINO runtime.
    pub openvino_library: Option<PathBuf>,
    /// Exact OpenVINO plugins.xml used to register CPU, GPU, and NPU devices.
    pub openvino_plugins: Option<PathBuf>,
    pub options: BTreeMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "supertonic".into(),
            runtime: Runtime::Default,
            device: "auto".into(),
            threads: 2,
            fallback: Fallback::Error,
            device_id: 0,
            library_dirs: Vec::new(),
            onnxruntime_library: None,
            provider_library: None,
            openvino_library: None,
            openvino_plugins: None,
            options: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BackendError {
    #[error("backend threads must be between 1 and 64")]
    InvalidThreads,
    #[error("backend device {device} is invalid for runtime {runtime:?}")]
    InvalidDevice { runtime: Runtime, device: String },
    #[error("backend device_id is only valid with runtime cuda")]
    InvalidDeviceId,
    #[error("backend option key {key:?} is invalid")]
    InvalidOptionKey { key: String },
    #[error("backend option {key:?} contains a newline or NUL byte")]
    InvalidOptionValue { key: String },
}

impl BackendConfig {
    pub fn canonical_device(&self) -> Result<String, BackendError> {
        canonical_device(self.runtime, &self.device)
    }

    pub fn validate_shape(&self) -> Result<(), BackendError> {
        if !(1..=64).contains(&self.threads) {
            return Err(BackendError::InvalidThreads);
        }
        if self.runtime != Runtime::Cuda && self.device_id != 0 {
            return Err(BackendError::InvalidDeviceId);
        }
        for (key, value) in &self.options {
            if !valid_option_key(key) {
                return Err(BackendError::InvalidOptionKey { key: key.clone() });
            }
            if value.contains(['\r', '\n', '\0']) {
                return Err(BackendError::InvalidOptionValue { key: key.clone() });
            }
        }
        self.canonical_device()?;
        Ok(())
    }
}

pub const fn supported_capabilities() -> &'static [&'static str] {
    &["cpu", "openvino", "cuda"]
}

fn valid_option_key(key: &str) -> bool {
    let mut characters = key.chars();
    matches!(characters.next(), Some(first) if first.is_ascii_alphanumeric())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-')
        })
}

pub fn canonical_device(runtime: Runtime, raw: &str) -> Result<String, BackendError> {
    let trimmed = raw.trim();
    let invalid = || BackendError::InvalidDevice {
        runtime,
        device: raw.to_owned(),
    };

    match runtime {
        Runtime::Default => match trimmed.to_ascii_lowercase().as_str() {
            "auto" => Ok("auto".into()),
            "cpu" => Ok("cpu".into()),
            _ => Err(invalid()),
        },
        Runtime::Cuda => match trimmed.to_ascii_lowercase().as_str() {
            "auto" => Ok("auto".into()),
            "gpu" => Ok("gpu".into()),
            _ => Err(invalid()),
        },
        Runtime::Openvino => canonical_openvino_device(trimmed).ok_or_else(invalid),
    }
}

fn canonical_openvino_device(raw: &str) -> Option<String> {
    let upper = raw.to_ascii_uppercase();
    if matches!(upper.as_str(), "AUTO" | "NPU" | "GPU" | "CPU") {
        return Some(upper.to_ascii_lowercase());
    }
    None
}

#[cfg(test)]
#[path = "../tests/unit/backend.rs"]
mod tests;
