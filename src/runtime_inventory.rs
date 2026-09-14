//! Runtime candidates are resolved read-only and validated in a separate process.

use crate::{
    backend::{BackendConfig, Runtime},
    config::Config,
    runtime,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    env,
    ffi::CStr,
    path::{Path, PathBuf},
};
#[cfg(not(test))]
use std::{
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Probe {
    pub loadable: bool,
    pub device_accessible: bool,
    pub ready: bool,
    pub evidence: Evidence,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evidence {
    pub versions: Vec<String>,
    pub provider_registration: bool,
    pub available_devices: Vec<String>,
    pub selected_device: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct State {
    pub runtime: &'static str,
    pub device: String,
    pub supported: bool,
    pub discovered: bool,
    pub source: &'static str,
    pub configured: bool,
    #[serde(flatten)]
    pub probe: Probe,
    pub paths: BackendConfig,
    pub remediation: Vec<String>,
}

pub fn name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

pub fn inventory(config: &BackendConfig, path: &Path) -> Vec<State> {
    [
        (Runtime::Default, &["auto", "cpu"][..]),
        (
            Runtime::Openvino,
            &["auto", "cpu", "gpu", "npu"][..],
        ),
        (Runtime::Cuda, &["auto", "gpu"][..]),
    ]
    .into_iter()
    .flat_map(|(runtime, devices)| {
        devices
            .iter()
            .map(move |device| (runtime, *device))
    })
    .map(|(runtime, device)| {
        let mut candidate = config.clone();
        if runtime != config.runtime {
            candidate.provider_library = None;
            candidate.device_id = 0;
        }
        candidate.runtime = runtime;
        candidate.device = device.into();
        let locations = runtime::discover(&candidate, path);
        let exact = resolve(&candidate, path);
        let anchor = match runtime {
            Runtime::Default => exact.onnxruntime_library.as_deref(),
            Runtime::Openvino => exact.openvino_library.as_deref(),
            Runtime::Cuda => exact.provider_library.as_deref(),
        };
        let under = |dirs: &[PathBuf]| {
            anchor.is_some_and(|path| dirs.iter().any(|directory| path.starts_with(directory)))
        };
        let explicitly_configured = match runtime {
            Runtime::Default => config.onnxruntime_library.is_some(),
            Runtime::Openvino => config.openvino_library.is_some(),
            Runtime::Cuda => config.provider_library.is_some(),
        };
        let explicit_environment = [
            runtime::ONNXRUNTIME_LIBRARY_ENV,
            runtime::PROVIDER_LIBRARY_ENV,
            runtime::OPENVINO_LIBRARY_ENV,
        ]
        .iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .any(|configured| {
            anchor.is_some_and(|actual| {
                actual == configured
                    || configured
                        .canonicalize()
                        .is_ok_and(|configured| actual == configured)
            })
        });
        let loader_environment = env::var_os("LD_LIBRARY_PATH")
            .is_some_and(|value| under(&env::split_paths(&value).collect::<Vec<_>>()));
        let source = candidate_source(
            under(&locations.configured_library_dirs) || explicitly_configured,
            under(&locations.environment_library_dirs)
                || loader_environment
                || explicit_environment,
            under(&locations.package_library_dirs),
            anchor.is_some(),
        );
        let required = match runtime {
            Runtime::Default => "ONNX Runtime 1.29.0 libonnxruntime.so",
            Runtime::Openvino => {
                "Intel OpenVINO libopenvino_c.so, plugins.xml and device plugins"
            }
            Runtime::Cuda => {
                "ONNX Runtime 1.29.0 core, libonnxruntime_providers_cuda.so and NVIDIA vendor libraries"
            }
        };
        State {
            runtime: name(runtime),
            device: device.into(),
            supported: true,
            discovered: anchor.is_some_and(Path::is_file),
            source,
            configured: path.is_file()
                && config.runtime == runtime
                && config.device.eq_ignore_ascii_case(device),
            probe: probe(&candidate, path),
            paths: exact,
            remediation: vec![format!(
                "Supply {required}; preview with `omaspeak setup runtime --runtime {} --device {device} --dir /absolute/runtime`, then add --apply to save after a successful probe.",
                name(runtime)
            )],
        }
    })
    .collect()
}

fn candidate_source(
    configured: bool,
    environment: bool,
    package: bool,
    discovered: bool,
) -> &'static str {
    if configured {
        "configured"
    } else if environment {
        "environment"
    } else if package {
        "package"
    } else if discovered {
        "system"
    } else {
        "candidate"
    }
}

pub fn resolve(config: &BackendConfig, path: &Path) -> BackendConfig {
    let locations = runtime::discover(config, path);
    resolve_with_locations(config, locations)
}

fn resolve_with_locations(
    config: &BackendConfig,
    locations: runtime::LibraryPathReport,
) -> BackendConfig {
    let mut exact = config.clone();
    exact.onnxruntime_library = locations.onnxruntime_library;
    exact.provider_library = locations.provider_library;
    exact.openvino_library = locations.openvino_library;
    exact.openvino_plugins = locations.openvino_plugins;
    match config.runtime {
        Runtime::Default => {
            exact.provider_library = None;
            exact.openvino_library = None;
            exact.openvino_plugins = None;
        }
        Runtime::Cuda => {
            exact.openvino_library = None;
            exact.openvino_plugins = None;
        }
        Runtime::Openvino => {
            exact.onnxruntime_library = None;
            exact.provider_library = None;
        }
    }
    // Carry the application-specific runtime overlay into the staged candidate.
    // The isolated child replaces LD_LIBRARY_PATH, so omitting these directories
    // can make a valid split runtime (for example ORT/provider in one directory
    // and CUDA vendor dependencies in another) fail only during setup. Ambient
    // LD_LIBRARY_PATH is deliberately absent from both of these collections.
    exact.library_dirs = locations.configured_library_dirs;
    for directory in locations.environment_library_dirs {
        if !exact.library_dirs.contains(&directory) {
            exact.library_dirs.push(directory);
        }
    }
    let parents = required(&exact)
        .into_iter()
        .flatten()
        .filter_map(|path| path.parent().map(Path::to_owned))
        .collect::<Vec<_>>();
    for parent in parents {
        if !exact.library_dirs.contains(&parent) {
            exact.library_dirs.push(parent);
        }
    }
    exact
}

fn required(config: &BackendConfig) -> Vec<Option<&Path>> {
    match config.runtime {
        Runtime::Default => vec![config.onnxruntime_library.as_deref()],
        Runtime::Cuda => vec![
            config.onnxruntime_library.as_deref(),
            config.provider_library.as_deref(),
        ],
        Runtime::Openvino => vec![
            config.openvino_library.as_deref(),
            config.openvino_plugins.as_deref(),
        ],
    }
}

pub fn probe(config: &BackendConfig, path: &Path) -> Probe {
    let exact = resolve(config, path);
    let attempt = (|| -> Result<Probe> {
        exact.validate_shape()?;
        for library in required(&exact) {
            if !library.is_some_and(Path::is_file) {
                bail!(
                    "required {} library or plugins.xml missing; inspect paths and supply --dir /absolute/runtime",
                    name(exact.runtime)
                );
            }
        }
        for directory in &exact.library_dirs {
            if !directory.is_absolute() || !directory.is_dir() {
                bail!("invalid dependency directory: {}", directory.display());
            }
        }
        isolated(&exact)
    })();
    attempt.unwrap_or_else(|error| Probe {
        errors: vec![format!("{error:#}")],
        ..Default::default()
    })
}

fn isolated(config: &BackendConfig) -> Result<Probe> {
    #[cfg(test)]
    return Ok(child(config));

    #[cfg(not(test))]
    {
        let executable = env::current_exe()?;
        let mut child = Command::new(executable)
            .arg("__inventory-probe")
            .arg(serde_json::to_string(config)?)
            .env("LD_LIBRARY_PATH", env::join_paths(&config.library_dirs)?)
            .env("ORT_DISABLE_TELEMETRY", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("start isolated native probe")?;
        let start = Instant::now();
        while child.try_wait()?.is_none() {
            if start.elapsed() > Duration::from_secs(20) {
                child.kill()?;
                child.wait()?;
                bail!("native probe timed out after 20 seconds");
            }
            thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            bail!("native probe terminated: {}", output.status);
        }
        serde_json::from_slice(&output.stdout).context("read isolated probe evidence")
    }
}

/// Verify the exact core before the ORT crate initializes its process-global loader.
pub(crate) fn initialize_ort(path: &Path) -> Result<()> {
    static CORE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);
    let exact = path.canonicalize().context("resolve ONNX Runtime core")?;
    let mut loaded = CORE
        .lock()
        .map_err(|_| anyhow::anyhow!("ORT initialization lock poisoned"))?;
    if let Some(previous) = loaded.as_ref() {
        if previous != &exact {
            bail!(
                "ONNX Runtime core changed from {} to {}; restart the process before changing native runtimes",
                previous.display(),
                exact.display()
            );
        }
        return Ok(());
    }
    verify_ort_version(&exact)?;
    ort::init_from(&exact)?.with_name("omaspeak").commit();
    *loaded = Some(exact);
    Ok(())
}

pub(crate) fn verify_ort_version(path: &Path) -> Result<()> {
    #[repr(C)]
    struct ApiBase {
        get_api: *const std::ffi::c_void,
        version: unsafe extern "C" fn() -> *const std::ffi::c_char,
    }
    unsafe {
        let library = libloading::Library::new(path)
            .with_context(|| format!("load ONNX Runtime {}", path.display()))?;
        let get: libloading::Symbol<unsafe extern "C" fn() -> *const ApiBase> =
            library.get(b"OrtGetApiBase\0")?;
        let base = get();
        if base.is_null() {
            bail!("ORT API base is null");
        }
        let version = ((*base).version)();
        if version.is_null() {
            bail!("ORT version is null");
        }
        let version = CStr::from_ptr(version).to_str()?;
        if version != "1.29.0" {
            bail!("ONNX Runtime version mismatch: expected 1.29.0, loaded {version}");
        }
    }
    Ok(())
}

pub fn child(config: &BackendConfig) -> Probe {
    child_with(config, native_openvino_probe, native_ort_probe)
}

fn native_openvino_probe(paths: runtime::OpenvinoRuntimePaths) -> Result<(String, Vec<String>)> {
    let devices = crate::supertonic::probe_runtime(paths, "auto")?;
    let version = openvino::version();
    Ok((
        format!(
            "OpenVINO {} ({})",
            version.build_number, version.description
        ),
        devices,
    ))
}

fn native_ort_probe(ort_path: &Path, provider: Option<&Path>) -> Result<Vec<(u32, u32, String)>> {
    verify_ort_version(ort_path)?;
    ort::init_from(ort_path)?
        .with_name("omaspeak-probe")
        .commit();
    let Some(provider) = provider else {
        return Ok(Vec::new());
    };
    let environment = ort::environment::Environment::current()?;
    let _registration = environment
        .register_ep_library("omaspeak-cuda", provider)
        .context("register selected CUDA provider library")?;
    Ok(environment
        .devices()
        .filter(|device| device.ep().ok() == Some("CUDAExecutionProvider"))
        .enumerate()
        .map(|(ordinal, device)| {
            let hardware = device.hardware_device();
            (
                ordinal as u32,
                hardware.id(),
                format!(
                    "CUDA ordinal {ordinal}, hardware {} ({:?})",
                    hardware.id(),
                    hardware.ty()
                ),
            )
        })
        .collect())
}

fn child_with(
    config: &BackendConfig,
    mut openvino_probe: impl FnMut(runtime::OpenvinoRuntimePaths) -> Result<(String, Vec<String>)>,
    mut ort_probe: impl FnMut(&Path, Option<&Path>) -> Result<Vec<(u32, u32, String)>>,
) -> Probe {
    let mut result = Probe::default();
    let attempt = (|| -> Result<()> {
        if config.runtime == Runtime::Openvino {
            let paths = runtime::OpenvinoRuntimePaths {
                library: config
                    .openvino_library
                    .clone()
                    .context("missing OpenVINO C library")?,
                plugins: config
                    .openvino_plugins
                    .clone()
                    .context("missing plugins.xml")?,
            };
            let (version, devices) = openvino_probe(paths)?;
            result.evidence.versions.push(version);
            result.loadable = true;
            result.evidence.provider_registration = true;
            result.evidence.available_devices = devices;
            let selected = config.canonical_device()?.to_ascii_uppercase();
            if selected == "AUTO" && result.evidence.available_devices.is_empty() {
                bail!("OpenVINO reports no accessible devices");
            }
            if selected != "AUTO"
                && !result.evidence.available_devices.iter().any(|device| {
                    device == &selected || device.starts_with(&format!("{selected}."))
                })
            {
                bail!(
                    "requested device {selected} is unavailable; available: {:?}",
                    result.evidence.available_devices
                );
            }
            if selected != "AUTO" {
                result.evidence.selected_device = Some(selected);
            }
        } else {
            let ort_path = config
                .onnxruntime_library
                .as_deref()
                .context("missing ORT library")?;
            let provider = if config.runtime == Runtime::Cuda {
                Some(
                    config
                        .provider_library
                        .as_deref()
                        .context("missing CUDA provider")?,
                )
            } else {
                None
            };
            let devices = ort_probe(ort_path, provider)?;
            result.evidence.versions.push("ONNX Runtime 1.29.0".into());
            if config.runtime == Runtime::Cuda {
                result.loadable = true;
                result.evidence.provider_registration = true;
                let (available, selected) = cuda_device_evidence(devices, config.device_id)?;
                result.evidence.available_devices = available;
                result.evidence.selected_device = Some(selected);
            } else {
                result.loadable = true;
                result.evidence.available_devices = vec!["cpu".into()];
                result.evidence.selected_device = Some("cpu".into());
            }
        }
        result.device_accessible = true;
        result.ready = true;
        Ok(())
    })();
    if let Err(error) = attempt {
        result.errors.push(format!("{error:#}"));
    }
    result
}

fn cuda_device_evidence(
    devices: Vec<(u32, u32, String)>,
    selected_ordinal: u32,
) -> Result<(Vec<String>, String)> {
    let selected = devices
        .iter()
        .find(|(ordinal, _, _)| *ordinal == selected_ordinal)
        .map(|(_, _, description)| description.clone())
        .context("requested CUDA device is unavailable")?;
    Ok((
        devices
            .into_iter()
            .map(|(_, _, description)| description)
            .collect(),
        selected,
    ))
}

pub fn apply_with(
    config: &Config,
    path: &Path,
    apply: bool,
    probe: impl FnOnce(&BackendConfig, &Path) -> Probe,
) -> Result<Probe> {
    let result = probe(&config.backend, path);
    if !result.ready {
        bail!(
            "runtime candidate rejected; config unchanged: {}",
            result.errors.join("; ")
        );
    }
    if apply {
        config.save(path)?;
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/unit/runtime_inventory.rs"]
mod tests;
