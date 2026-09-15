use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::backend::{BackendConfig, Runtime};
use crate::runtime::LibraryPathReport;

const INTEL: u32 = 0x8086;
const NVIDIA: u32 = 0x10de;
const DISPLAY_CLASS: u32 = 0x03;
const PROCESSING_ACCELERATOR_CLASS: u32 = 0x12;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct HardwareReport {
    pub cuda_gpu: bool,
    pub intel_npu: bool,
    pub intel_gpu: bool,
    pub vulkan_candidate: bool,
    pub devices: Vec<HardwareDevice>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HardwareDevice {
    pub address: String,
    pub vendor: String,
    pub class: String,
    pub driver: Option<String>,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProviderAvailability {
    pub packaged_cpu: bool,
    pub cuda: bool,
    pub openvino: bool,
    pub vulkan: bool,
}

pub fn provider_availability(
    config: &BackendConfig,
    locations: &LibraryPathReport,
) -> ProviderAvailability {
    let selected_audio = locations.audiocpp_library.as_deref();
    let external_audio = selected_audio.is_some_and(|library| {
        (config.runtime != Runtime::Default && config.library.is_some())
            || (config.library.is_none()
                && !locations
                    .package_library_dirs
                    .iter()
                    .any(|directory| library.starts_with(directory)))
    });
    let packaged_cpu =
        crate::runtime::find_versioned_library(&locations.package_library_dirs, "libaudiocpp.so")
            .is_some()
            || (config.runtime == Runtime::Default && selected_audio.is_some() && !external_audio);
    ProviderAvailability {
        packaged_cpu,
        // The public audio.cpp ABI proves that the provider is complete, but
        // only Apply with a model proves the compiled accelerator backend.
        cuda: selected_audio.is_some() && (config.runtime == Runtime::Cuda || external_audio),
        openvino: locations.runtime_loadable.get("openvino") == Some(&true),
        vulkan: selected_audio.is_some() && (config.runtime == Runtime::Vulkan || external_audio),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Recommendation {
    pub runtime: Runtime,
    pub device: String,
    pub label: String,
    pub hardware_detected: bool,
    pub provider_detected: bool,
    pub ready: bool,
    pub detail: String,
}

pub fn detect() -> HardwareReport {
    detect_at(Path::new("/sys/bus/pci/devices"))
}

pub fn detect_at(root: &Path) -> HardwareReport {
    let mut report = HardwareReport::default();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            report.errors.push(format!(
                "inspect PCI devices at {}: {error}",
                root.display()
            ));
            return report;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                report.errors.push(format!("inspect PCI entry: {error}"));
                continue;
            }
        };
        match inspect_device(&entry.path()) {
            Ok(Some(device)) => {
                report.cuda_gpu |= device.capabilities.iter().any(|item| item == "cuda");
                report.intel_npu |= device.capabilities.iter().any(|item| item == "intel-npu");
                report.intel_gpu |= device.capabilities.iter().any(|item| item == "intel-gpu");
                report.vulkan_candidate |= device
                    .capabilities
                    .iter()
                    .any(|item| item == "vulkan-candidate");
                report.devices.push(device);
            }
            Ok(None) => {}
            Err(error) => report.errors.push(format!(
                "inspect PCI device {}: {error}",
                entry.path().display()
            )),
        }
    }
    report
        .devices
        .sort_by(|left, right| left.address.cmp(&right.address));
    report
}

fn inspect_device(path: &Path) -> std::io::Result<Option<HardwareDevice>> {
    let vendor_text = read_trimmed(path.join("vendor"))?;
    let class_text = read_trimmed(path.join("class"))?;
    let Some(vendor) = parse_hex(&vendor_text) else {
        return Ok(None);
    };
    let Some(class) = parse_hex(&class_text) else {
        return Ok(None);
    };
    let base_class = (class >> 16) & 0xff;
    let driver = driver_name(path.join("driver"));
    let mut capabilities = Vec::new();
    if vendor == NVIDIA && base_class == DISPLAY_CLASS {
        capabilities.push("cuda".into());
    }
    if vendor == INTEL
        && base_class == PROCESSING_ACCELERATOR_CLASS
        && driver.as_deref() == Some("intel_vpu")
    {
        capabilities.push("intel-npu".into());
    }
    if vendor == INTEL
        && base_class == DISPLAY_CLASS
        && matches!(driver.as_deref(), Some("xe" | "i915"))
    {
        capabilities.push("intel-gpu".into());
    }
    if base_class == DISPLAY_CLASS && driver.is_some() {
        // PCI and a bound display driver are useful setup hints, but they do
        // not prove that a Vulkan device or provider can execute this model.
        capabilities.push("vulkan-candidate".into());
    }
    if capabilities.is_empty() {
        return Ok(None);
    }
    Ok(Some(HardwareDevice {
        address: path
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        vendor: vendor_text,
        class: class_text,
        driver,
        capabilities,
    }))
}

fn read_trimmed(path: PathBuf) -> std::io::Result<String> {
    fs::read_to_string(path).map(|value| value.trim().to_ascii_lowercase())
}

fn parse_hex(value: &str) -> Option<u32> {
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

fn driver_name(path: PathBuf) -> Option<String> {
    fs::read_link(&path)
        .ok()
        .and_then(|target| {
            target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .or_else(|| {
            fs::read_to_string(path)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
}

pub fn recommend(hardware: &HardwareReport, providers: ProviderAvailability) -> Recommendation {
    let accelerators = [
        (
            hardware.cuda_gpu,
            providers.cuda,
            Runtime::Cuda,
            "gpu",
            "NVIDIA GPU through CUDA",
        ),
        (
            hardware.intel_npu,
            providers.openvino,
            Runtime::Openvino,
            "npu",
            "Intel NPU through OpenVINO",
        ),
        (
            hardware.intel_gpu,
            providers.openvino,
            Runtime::Openvino,
            "gpu",
            "Intel GPU through OpenVINO",
        ),
        (
            hardware.vulkan_candidate,
            providers.vulkan,
            Runtime::Vulkan,
            "gpu",
            "GPU candidate through Vulkan",
        ),
    ];
    let usable_accelerator = accelerators
        .iter()
        .find(|(detected, provider, ..)| *detected && *provider);
    let detected_accelerator = accelerators.iter().find(|(detected, ..)| *detected);
    let (runtime, device, label, hardware_detected, provider_detected) =
        if let Some((_, _, runtime, device, label)) = usable_accelerator {
            (*runtime, *device, *label, true, true)
        } else if providers.packaged_cpu {
            (
                Runtime::Default,
                "cpu",
                "packaged CPU provider",
                false,
                true,
            )
        } else if let Some((_, provider, runtime, device, label)) = detected_accelerator {
            (*runtime, *device, *label, true, *provider)
        } else {
            (
                Runtime::Default,
                "cpu",
                "packaged CPU provider",
                false,
                false,
            )
        };
    let detail = match (
        hardware_detected,
        provider_detected,
        detected_accelerator.is_some(),
    ) {
        (true, true, _) => format!(
            "Recommended for detected hardware: {label}; provider detected, Apply still runs the isolated and model-backed proof"
        ),
        (true, false, _) => format!(
            "Recommended for detected hardware: {label}; a complete external provider is still required"
        ),
        (false, true, true) => "Detected accelerator providers are incomplete; recommended available fallback: packaged CPU provider. Apply still runs the model-backed proof".into(),
        (false, true, false) => "Recommended portable fallback: packaged CPU provider; Apply still runs the model-backed proof".into(),
        (false, false, _) => {
            "Recommended portable fallback: CPU; install the packaged provider before Apply".into()
        }
    };
    Recommendation {
        runtime,
        device: device.into(),
        label: label.into(),
        hardware_detected,
        provider_detected,
        // Discovery is deliberately advisory. Only the existing setup probe
        // may promote a candidate to ready.
        ready: false,
        detail,
    }
}

#[cfg(test)]
#[path = "../tests/unit/hardware.rs"]
mod tests;
