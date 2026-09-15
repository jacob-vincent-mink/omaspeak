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
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpuPreparationRequest {
    pub config: Config,
    pub paths: crate::paths::AppPaths,
    pub cache_dir: PathBuf,
    pub require_cache_hits: bool,
}

pub fn npu_child(
    request: NpuPreparationRequest,
) -> Result<crate::supertonic::NpuNativePreparation> {
    npu_child_with(
        request,
        |config, path| {
            let locations = runtime::discover(config, path);
            runtime::resolve_openvino_runtime(config, path, &locations)
        },
        crate::supertonic::prepare_npu_cache_native,
    )
}

fn npu_child_with(
    request: NpuPreparationRequest,
    resolve_runtime: impl FnOnce(&BackendConfig, &Path) -> Result<runtime::OpenvinoRuntimePaths>,
    prepare: impl FnOnce(
        &Config,
        &crate::paths::AppPaths,
        runtime::OpenvinoRuntimePaths,
        &Path,
        bool,
    ) -> Result<crate::supertonic::NpuNativePreparation>,
) -> Result<crate::supertonic::NpuNativePreparation> {
    let runtime = resolve_runtime(&request.config.backend, &request.paths.config_file)?;
    prepare(
        &request.config,
        &request.paths,
        runtime,
        &request.cache_dir,
        request.require_cache_hits,
    )
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
            Runtime::Default => "ONNX Runtime 1.30.0 libonnxruntime.so",
            Runtime::Openvino => {
                "Intel OpenVINO libopenvino_c.so, plugins.xml and device plugins"
            }
            Runtime::Cuda => {
                "the CUDA Plugin EP libonnxruntime_providers_cuda.so and its NVIDIA vendor libraries; Omaspeak supplies the ONNX Runtime 1.30.0 core"
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
        let missing = match exact.runtime {
            Runtime::Default
                if !exact
                    .onnxruntime_library
                    .as_deref()
                    .is_some_and(Path::is_file) =>
            {
                Some("ONNX Runtime 1.30.0 core library")
            }
            Runtime::Cuda
                if !exact
                    .onnxruntime_library
                    .as_deref()
                    .is_some_and(Path::is_file) =>
            {
                Some("packaged ONNX Runtime 1.30.0 core library")
            }
            Runtime::Cuda if !exact.provider_library.as_deref().is_some_and(Path::is_file) => {
                Some("CUDA Plugin EP libonnxruntime_providers_cuda.so")
            }
            Runtime::Openvino if !exact.openvino_library.as_deref().is_some_and(Path::is_file) => {
                Some("OpenVINO C library")
            }
            Runtime::Openvino if !exact.openvino_plugins.as_deref().is_some_and(Path::is_file) => {
                Some("OpenVINO plugins.xml")
            }
            _ => None,
        };
        if let Some(missing) = missing {
            bail!("required {missing} missing; inspect paths and supply --dir /absolute/runtime");
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

pub fn prepare_npu_cache(
    config: &mut Config,
    paths: &crate::paths::AppPaths,
) -> Result<Option<crate::supertonic::NpuCacheState>> {
    if !crate::supertonic::uses_static_npu_shapes(config) {
        return Ok(None);
    }
    config.backend = resolve(&config.backend, &paths.config_file);
    prepare_npu_cache_with(config, paths, run_npu_preparation_child).map(Some)
}

fn prepare_npu_cache_with(
    config: &Config,
    paths: &crate::paths::AppPaths,
    mut invoke: impl FnMut(&NpuPreparationRequest) -> Result<crate::supertonic::NpuNativePreparation>,
) -> Result<crate::supertonic::NpuCacheState> {
    let validate_report = |report: crate::supertonic::NpuNativePreparation,
                           exact_blobs: bool|
     -> Result<()> {
        if report.compiled_models != crate::supertonic::NPU_COMPILED_MODELS {
            bail!(
                "NPU preparation covered {} compiled graph-shape keys; expected {}",
                report.compiled_models,
                crate::supertonic::NPU_COMPILED_MODELS
            );
        }
        if report.cache_blobs.is_empty()
            || (exact_blobs && report.cache_blobs.len() != crate::supertonic::NPU_COMPILED_MODELS)
        {
            bail!(
                "NPU preparation returned {} compiled-model cache blobs; expected {}",
                report.cache_blobs.len(),
                crate::supertonic::NPU_COMPILED_MODELS
            );
        }
        Ok(())
    };
    let target = crate::supertonic::npu_cache_directory(config, paths)?;
    if crate::supertonic::npu_cache_state(config, paths).ready {
        validate_report(
            invoke(&NpuPreparationRequest {
                config: config.clone(),
                paths: paths.clone(),
                cache_dir: target.clone(),
                require_cache_hits: true,
            })?,
            true,
        )
        .context("verify prepared OpenVINO NPU cache in an isolated process")?;
        return Ok(crate::supertonic::npu_cache_state(config, paths));
    }

    let parent = target.parent().context("NPU cache target has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create NPU cache directory {}", parent.display()))?;
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let staging = parent.join(format!(".prepare-{nonce}"));
    let backup = parent.join(format!(".backup-{nonce}"));
    fs::create_dir(&staging)
        .with_context(|| format!("create staging NPU cache {}", staging.display()))?;

    let prepared = (|| -> Result<()> {
        // Some OpenVINO NPU releases persist only a subset of newly compiled
        // blobs before a compiler process exits. Re-entering the same complete
        // static plan imports the blobs already present and fills the missing
        // keys. Keep this bounded and require the exact plan before publishing.
        for attempt in 1..=5 {
            validate_report(
                invoke(&NpuPreparationRequest {
                    config: config.clone(),
                    paths: paths.clone(),
                    cache_dir: staging.clone(),
                    require_cache_hits: false,
                })?,
                false,
            )
            .with_context(|| {
                format!("compile OpenVINO models for Intel NPU (pass {attempt} of 5)")
            })?;
            let count = crate::supertonic::cache_blobs(&staging)?.len();
            if count == crate::supertonic::NPU_COMPILED_MODELS {
                break;
            }
            if attempt == 5 {
                bail!(
                    "OpenVINO persisted {count} of {} required NPU cache blobs after 5 complete static-plan passes",
                    crate::supertonic::NPU_COMPILED_MODELS
                );
            }
        }
        crate::supertonic::write_npu_cache_manifest(config, paths, &staging)?;

        let had_previous = target.exists();
        if had_previous {
            fs::rename(&target, &backup).with_context(|| {
                format!(
                    "move previous NPU cache {} to {}",
                    target.display(),
                    backup.display()
                )
            })?;
        }
        if let Err(error) = fs::rename(&staging, &target) {
            if had_previous {
                fs::rename(&backup, &target).with_context(|| {
                    format!(
                        "restore previous NPU cache {} after install failed: {error}",
                        target.display()
                    )
                })?;
            }
            return Err(error)
                .with_context(|| format!("install prepared NPU cache at {}", target.display()));
        }

        let verification = invoke(&NpuPreparationRequest {
            config: config.clone(),
            paths: paths.clone(),
            cache_dir: target.clone(),
            require_cache_hits: true,
        })
        .and_then(|report| validate_report(report, true))
        .context("verify OpenVINO loads every prepared NPU model from cache");
        if let Err(error) = verification {
            let rollback = (|| -> Result<()> {
                fs::remove_dir_all(&target)
                    .with_context(|| format!("remove rejected NPU cache {}", target.display()))?;
                if had_previous {
                    fs::rename(&backup, &target).with_context(|| {
                        format!("restore previous NPU cache {}", target.display())
                    })?;
                }
                Ok(())
            })();
            if let Err(rollback_error) = rollback {
                return Err(error.context(format!(
                    "NPU cache rollback also failed: {rollback_error:#}"
                )));
            }
            return Err(error);
        }
        if had_previous {
            fs::remove_dir_all(&backup)
                .with_context(|| format!("remove superseded NPU cache {}", backup.display()))?;
        }
        Ok(())
    })();
    if let Err(error) = prepared {
        if staging.exists()
            && let Err(cleanup_error) = fs::remove_dir_all(&staging)
        {
            return Err(error.context(format!(
                "partial NPU cache cleanup also failed: {cleanup_error}"
            )));
        }
        return Err(error);
    }
    let state = crate::supertonic::npu_cache_state(config, paths);
    if !state.ready {
        bail!("prepared NPU cache failed validation: {}", state.detail);
    }
    Ok(state)
}

fn run_npu_preparation_child(
    request: &NpuPreparationRequest,
) -> Result<crate::supertonic::NpuNativePreparation> {
    #[cfg(test)]
    return npu_child(request.clone());

    #[cfg(not(test))]
    {
        let executable = env::current_exe().context("locate Omaspeak executable")?;
        let loader_path = env::join_paths(&request.config.backend.library_dirs)?;
        let child = Command::new(executable)
            .arg("__npu-precompile")
            .arg(serde_json::to_string(request)?)
            .env("LD_LIBRARY_PATH", loader_path)
            .env("ORT_DISABLE_TELEMETRY", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("start isolated NPU cache preparation")?;
        let output = collect_child_output(child, Duration::from_secs(30 * 60))?;
        decode_npu_preparation(output)
    }
}

struct ChildOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn collect_child_output(mut child: Child, timeout: Duration) -> Result<ChildOutput> {
    let mut child_stdout = child
        .stdout
        .take()
        .context("capture NPU preparation stdout")?;
    let mut child_stderr = child
        .stderr
        .take()
        .context("capture NPU preparation stderr")?;
    let stdout_reader = thread::spawn(move || {
        let mut output = Vec::new();
        child_stdout.read_to_end(&mut output).map(|_| output)
    });
    let stderr_reader = thread::spawn(move || {
        let mut output = Vec::new();
        child_stderr.read_to_end(&mut output).map(|_| output)
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > timeout {
            child.kill()?;
            child.wait()?;
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!(
                "NPU cache preparation timed out after {} seconds",
                timeout.as_secs_f64()
            );
        }
        thread::sleep(Duration::from_millis(50));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("read NPU preparation stdout"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("read NPU preparation stderr"))??;
    Ok(ChildOutput {
        status,
        stdout,
        stderr,
    })
}

fn decode_npu_preparation(output: ChildOutput) -> Result<crate::supertonic::NpuNativePreparation> {
    let detail = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        bail!(
            "NPU cache preparation terminated: {}{}",
            output.status,
            if detail.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", detail.trim())
            }
        );
    }
    serde_json::from_slice(&output.stdout).with_context(|| {
        if detail.trim().is_empty() {
            "read isolated NPU preparation evidence".into()
        } else {
            format!("read isolated NPU preparation evidence: {}", detail.trim())
        }
    })
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
        if version != "1.30.0" {
            bail!("ONNX Runtime version mismatch: expected 1.30.0, loaded {version}");
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
        .register_ep_library(crate::supertonic::CUDA_PLUGIN_EP, provider)
        .context("register selected CUDA provider library")?;
    Ok(environment
        .devices()
        .filter(|device| device.ep().ok() == Some(crate::supertonic::CUDA_PLUGIN_EP))
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
            result.evidence.versions.push("ONNX Runtime 1.30.0".into());
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
