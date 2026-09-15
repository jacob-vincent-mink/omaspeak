//! Runtime candidates are resolved read-only and validated in a separate process.

use crate::{
    backend::{BackendConfig, Runtime},
    config::Config,
    runtime,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::process::{Child, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};
use std::{
    env, fs,
    path::{Path, PathBuf},
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
    pub provider_path: Option<PathBuf>,
    pub model_inference_verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NpuPreparationRequest {
    pub config: Config,
    pub paths: crate::paths::AppPaths,
    pub cache_dir: PathBuf,
    pub require_cache_hits: bool,
}

const NPU_RESULT_BEGIN: &[u8] = b"OMASPEAK_NPU_PREPARATION_V1_BEGIN\n";
const NPU_RESULT_END: &[u8] = b"\nOMASPEAK_NPU_PREPARATION_V1_END\n";

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
    runtime.name()
}

pub fn inventory(config: &BackendConfig, path: &Path) -> Vec<State> {
    [
        (Runtime::Default, &["auto", "cpu"][..]),
        (Runtime::Cuda, &["auto", "gpu"][..]),
        (Runtime::Vulkan, &["auto", "gpu"][..]),
        (Runtime::Hip, &["auto", "gpu"][..]),
        (Runtime::Openvino, &["auto", "cpu", "gpu", "npu"][..]),
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
            candidate.device_id = 0;
        }
        candidate.runtime = runtime;
        candidate.device = device.into();
        candidate.kind = if runtime.uses_audiocpp() {
            "audiocpp".into()
        } else {
            "supertonic".into()
        };
        let locations = runtime::discover(&candidate, path);
        let exact = resolve(&candidate, path);
        let anchor = match runtime {
            Runtime::Openvino => exact.openvino_library.as_deref(),
            Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => {
                exact.library.as_deref()
            }
        };
        let under = |dirs: &[PathBuf]| {
            anchor.is_some_and(|path| dirs.iter().any(|directory| path.starts_with(directory)))
        };
        let explicitly_configured = if runtime == Runtime::Openvino {
            config.openvino_library.is_some()
        } else {
            config.library.is_some()
        };
        let explicit_environment = [
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
            Runtime::Default => "the packaged or an external complete audio.cpp CPU provider",
            Runtime::Openvino => {
                "Intel OpenVINO libopenvino_c.so, plugins.xml and device plugins"
            }
            Runtime::Cuda => "a complete CUDA-enabled audio.cpp provider and CUDA libraries",
            Runtime::Vulkan => "a complete Vulkan-enabled audio.cpp provider and Vulkan loader",
            Runtime::Hip => "a complete HIP-enabled audio.cpp provider and ROCm libraries",
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
    exact.library = locations.audiocpp_library;
    exact.openvino_library = locations.openvino_library;
    exact.openvino_plugins = locations.openvino_plugins;
    match config.runtime {
        Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => {
            exact.openvino_library = None;
            exact.openvino_plugins = None;
        }
        Runtime::Openvino => {
            exact.library = None;
        }
    }
    // Carry the application-specific runtime overlay into the staged candidate.
    // The isolated child replaces LD_LIBRARY_PATH, so omitting these directories
    // can make a complete provider plus adjacent vendor dependency directory
    // fail only during setup. Ambient
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
        Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => {
            vec![config.library.as_deref()]
        }
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
            Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
                if !exact.library.as_deref().is_some_and(Path::is_file) =>
            {
                Some("complete audio.cpp provider library")
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
        let mut child = std::process::Command::new(executable)
            .arg("__inventory-probe")
            .arg(serde_json::to_string(config)?)
            .env("LD_LIBRARY_PATH", env::join_paths(&config.library_dirs)?)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
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
        let child = std::process::Command::new(executable)
            .arg("__npu-precompile")
            .arg(serde_json::to_string(request)?)
            .env("LD_LIBRARY_PATH", loader_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("start isolated NPU cache preparation")?;
        let output = collect_child_output(child, Duration::from_secs(30 * 60))?;
        decode_npu_preparation(output)
    }
}

#[derive(Debug)]
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
    let diagnostics = child_diagnostics(&output);
    if !output.status.success() {
        bail!(
            "NPU cache preparation terminated: {}{}",
            output.status,
            if diagnostics.is_empty() {
                String::new()
            } else {
                format!(": {diagnostics}")
            }
        );
    }
    let start = output
        .stdout
        .windows(NPU_RESULT_BEGIN.len())
        .rposition(|candidate| candidate == NPU_RESULT_BEGIN)
        .map(|offset| offset + NPU_RESULT_BEGIN.len())
        .with_context(|| format!("NPU cache preparation returned no result frame{diagnostics}"))?;
    let remainder = &output.stdout[start..];
    let end = remainder
        .windows(NPU_RESULT_END.len())
        .position(|candidate| candidate == NPU_RESULT_END)
        .with_context(|| {
            format!("NPU cache preparation returned an incomplete result frame{diagnostics}")
        })?;
    serde_json::from_slice(&remainder[..end])
        .with_context(|| format!("read isolated NPU preparation evidence{diagnostics}"))
}

fn child_diagnostics(output: &ChildOutput) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let before_frame = stdout
        .rfind(std::str::from_utf8(NPU_RESULT_BEGIN).expect("result marker is UTF-8"))
        .map_or(stdout.as_ref(), |offset| &stdout[..offset]);
    let stdout = before_frame.trim();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (false, true) => format!("; native stdout: {stdout}"),
        (true, false) => format!("; native stderr: {stderr}"),
        (false, false) => format!("; native stdout: {stdout}; native stderr: {stderr}"),
    }
}

/// Write the hidden-child result as a framed record after native libraries have
/// finished their work. OpenVINO and its plugins may also write to stdout, so
/// stdout cannot itself be treated as a JSON transport.
pub fn write_npu_preparation_result(
    result: &crate::supertonic::NpuNativePreparation,
) -> Result<()> {
    let json = serde_json::to_vec(result)?;
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(NPU_RESULT_BEGIN)?;
    stdout.write_all(&json)?;
    stdout.write_all(NPU_RESULT_END)?;
    stdout.flush().context("flush NPU preparation result")
}

pub fn child(config: &BackendConfig) -> Probe {
    child_with(config, native_openvino_probe, |config| {
        let full = Config {
            backend: config.clone(),
            ..Default::default()
        };
        crate::audio_cpp::probe_provider(&full, Path::new("/nonexistent/config.toml"))
    })
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

fn child_with(
    config: &BackendConfig,
    mut openvino_probe: impl FnMut(runtime::OpenvinoRuntimePaths) -> Result<(String, Vec<String>)>,
    mut audiocpp_probe: impl FnMut(&BackendConfig) -> Result<PathBuf>,
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
            let provider = audiocpp_probe(config)?;
            result.evidence.versions.push("audio.cpp C ABI 0.1".into());
            result.evidence.provider_path = Some(provider);
            result.evidence.provider_registration = true;
            result.loadable = true;
            // The public ABI cannot enumerate compiled backends without a
            // model/session. Setup performs that proof with the selected GGUF.
            result.evidence.selected_device = Some(config.canonical_device()?);
            result.evidence.available_devices = vec![config.runtime.capability().into()];
            result.device_accessible = false;
            result.ready = true;
            return Ok(());
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
