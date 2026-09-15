use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{BackendConfig, Runtime, supported_capabilities};

pub const LIBRARY_PATH_ENV: &str = "OMASPEAK_LIBRARY_PATH";
pub const ONNXRUNTIME_LIBRARY_ENV: &str = "OMASPEAK_ONNXRUNTIME_LIBRARY";
pub const PROVIDER_LIBRARY_ENV: &str = "OMASPEAK_PROVIDER_LIBRARY";
pub const OPENVINO_LIBRARY_ENV: &str = "OMASPEAK_OPENVINO_LIBRARY";
pub const OPENVINO_PLUGINS_ENV: &str = "OMASPEAK_OPENVINO_PLUGINS";
pub const REEXEC_SENTINEL: &str = "OMASPEAK_LIBRARY_PATH_READY";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LibraryPathReport {
    pub configured_library_dirs: Vec<PathBuf>,
    pub environment_library_dirs: Vec<PathBuf>,
    pub package_library_dirs: Vec<PathBuf>,
    pub effective_library_dirs: Vec<PathBuf>,
    pub missing_library_dirs: Vec<PathBuf>,
    pub onnxruntime_library: Option<PathBuf>,
    pub provider_library: Option<PathBuf>,
    pub openvino_library: Option<PathBuf>,
    pub openvino_plugins: Option<PathBuf>,
    pub runtime_loadable: BTreeMap<&'static str, bool>,
    pub runtime_probe_errors: BTreeMap<&'static str, String>,
    pub runtime_device_accessible: BTreeMap<&'static str, bool>,
    pub device_probe_errors: BTreeMap<&'static str, String>,
    pub remediation: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnnxRuntimePaths {
    pub onnxruntime: PathBuf,
    pub provider: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpenvinoRuntimePaths {
    pub library: PathBuf,
    pub plugins: PathBuf,
}

impl LibraryPathReport {
    pub fn remediation(&self, runtime: Runtime) -> Option<String> {
        let runtime_name = match runtime {
            Runtime::Default => "default",
            Runtime::Openvino => "openvino",
            Runtime::Cuda => "cuda",
        };
        if self.runtime_loadable.get(runtime_name) == Some(&true) {
            return None;
        }
        if !self.missing_library_dirs.is_empty() {
            return Some(format!(
                "create or correct missing/non-absolute directories in backend.library_dirs or {LIBRARY_PATH_ENV}: {}",
                display_paths(&self.missing_library_dirs)
            ));
        }
        if let Some(error) = self.runtime_probe_errors.get(runtime_name) {
            return Some(format!("runtime validation failed: {error}"));
        }
        Some(match runtime {
            Runtime::Default => format!(
                "set backend.onnxruntime_library, or use {LIBRARY_PATH_ENV}, for an ONNX Runtime library"
            ),
            Runtime::Openvino => format!(
                "set backend.openvino_library and backend.openvino_plugins (or {OPENVINO_LIBRARY_ENV} and {OPENVINO_PLUGINS_ENV}) to an installed OpenVINO C runtime"
            ),
            Runtime::Cuda => "configure the official CUDA Plugin EP with its vendor dependencies; Omaspeak supplies ONNX Runtime 1.30.0 in release packages".to_owned(),
        })
    }
}

pub fn inspect(config: &BackendConfig, config_file: &Path) -> LibraryPathReport {
    let mut report = discover(config, config_file);
    probe_installed_runtimes(config, &mut report);
    report
}

/// Resolve candidate runtime files without loading them into this process.
///
/// Interactive setup uses this for its choice list, then probes only the
/// selected runtime immediately before persisting the staged configuration.
pub fn discover(config: &BackendConfig, config_file: &Path) -> LibraryPathReport {
    inspect_with(
        config,
        config_file,
        env::var_os(LIBRARY_PATH_ENV),
        env::var_os("LD_LIBRARY_PATH"),
        env::current_exe().ok().as_deref(),
    )
}

pub fn inspect_with(
    config: &BackendConfig,
    config_file: &Path,
    environment: Option<OsString>,
    loader_environment: Option<OsString>,
    executable: Option<&Path>,
) -> LibraryPathReport {
    let config_base = config_file.parent().unwrap_or_else(|| Path::new("."));
    let configured_library_dirs = config
        .library_dirs
        .iter()
        .map(|path| resolve(path, config_base))
        .collect::<Vec<_>>();
    let environment_library_dirs = environment
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .map(|path| {
            if path.is_absolute() {
                canonical_or(path)
            } else {
                path
            }
        })
        .collect::<Vec<_>>();
    let package_library_dirs = package_library_dirs(executable);
    let exact_library_dirs = [
        config
            .onnxruntime_library
            .as_deref()
            .map(|path| resolve(path, config_base)),
        config
            .provider_library
            .as_deref()
            .map(|path| resolve(path, config_base)),
        config
            .openvino_library
            .as_deref()
            .map(|path| resolve(path, config_base)),
        config
            .openvino_plugins
            .as_deref()
            .map(|path| resolve(path, config_base)),
        env::var_os(ONNXRUNTIME_LIBRARY_ENV).map(PathBuf::from),
        env::var_os(PROVIDER_LIBRARY_ENV).map(PathBuf::from),
        env::var_os(OPENVINO_LIBRARY_ENV).map(PathBuf::from),
        env::var_os(OPENVINO_PLUGINS_ENV).map(PathBuf::from),
    ]
    .into_iter()
    .flatten()
    .filter(|path| path.is_absolute() && path.is_file())
    .filter_map(|path| path.parent().map(Path::to_path_buf));

    let candidates = configured_library_dirs
        .iter()
        .chain(&environment_library_dirs)
        .cloned()
        .collect::<Vec<_>>();
    let missing_library_dirs = stable_unique(
        candidates
            .iter()
            .filter(|directory| !directory.is_absolute() || !directory.is_dir())
            .cloned(),
    );
    let effective_library_dirs = stable_unique(
        candidates
            .into_iter()
            .filter(|directory| directory.is_absolute() && directory.is_dir())
            .chain(exact_library_dirs)
            .chain(package_library_dirs.iter().cloned()),
    );
    let ambient = loader_environment
        .as_deref()
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .filter(|path| path.is_dir())
        .map(canonical_or);
    let search_dirs = stable_unique(
        effective_library_dirs
            .iter()
            .cloned()
            .chain(ambient)
            .chain(system_library_dirs()),
    );
    let exact = |configured: Option<&Path>, environment: &str| {
        configured
            .map(|path| resolve(path, config_base))
            .or_else(|| env::var_os(environment).map(PathBuf::from))
    };
    let selected_onnxruntime = exact(
        config.onnxruntime_library.as_deref(),
        ONNXRUNTIME_LIBRARY_ENV,
    );
    let selected_provider = exact(config.provider_library.as_deref(), PROVIDER_LIBRARY_ENV);
    let selected_openvino = exact(config.openvino_library.as_deref(), OPENVINO_LIBRARY_ENV);
    let selected_plugins = exact(config.openvino_plugins.as_deref(), OPENVINO_PLUGINS_ENV);
    let onnxruntime_library = locate_runtime_library(
        selected_onnxruntime.as_ref(),
        &search_dirs,
        "libonnxruntime.so",
    );
    let provider_name = match config.runtime {
        Runtime::Default => None,
        Runtime::Openvino => None,
        Runtime::Cuda => Some("libonnxruntime_providers_cuda.so"),
    };
    let provider_library = provider_name
        .and_then(|name| locate_runtime_library(selected_provider.as_ref(), &search_dirs, name));
    let openvino_library =
        locate_runtime_library(selected_openvino.as_ref(), &search_dirs, "libopenvino_c.so");
    let openvino_plugins = locate_openvino_plugins(selected_plugins.as_ref(), &search_dirs);
    let runtime_loadable = runtime_loadability(RuntimeLoadabilityInput {
        search_dirs: &search_dirs,
        effective_dirs: &effective_library_dirs,
        missing_dirs: &missing_library_dirs,
        exact_onnxruntime: selected_onnxruntime,
        exact_provider: selected_provider,
        exact_openvino: selected_openvino,
        exact_plugins: selected_plugins,
        selected_runtime: config.runtime,
    });
    let mut remediation = Vec::new();
    if !missing_library_dirs.is_empty() {
        remediation.push(format!(
            "create or correct missing/non-absolute directories in backend.library_dirs or {LIBRARY_PATH_ENV}: {}",
            display_paths(&missing_library_dirs)
        ));
    }
    if runtime_loadable.get("default") != Some(&true) {
        remediation.push(format!(
            "set backend.onnxruntime_library, or use {LIBRARY_PATH_ENV}, for an ONNX Runtime library"
        ));
    }
    if runtime_loadable.get("openvino") != Some(&true) {
        remediation.push(format!(
            "set backend.openvino_library and backend.openvino_plugins (or {OPENVINO_LIBRARY_ENV} and {OPENVINO_PLUGINS_ENV}) to an installed OpenVINO C runtime"
        ));
    }
    let accelerated_runtime = Runtime::Cuda;
    let name = accelerated_runtime.capability();
    if supported_capabilities().contains(&name) && runtime_loadable.get(name) != Some(&true) {
        remediation.push(format!(
            "set backend.library_dirs or {LIBRARY_PATH_ENV} to directories containing the official CUDA Plugin EP and its vendor runtime libraries"
        ));
    }

    LibraryPathReport {
        configured_library_dirs,
        environment_library_dirs,
        package_library_dirs,
        effective_library_dirs,
        missing_library_dirs,
        onnxruntime_library,
        provider_library,
        openvino_library,
        openvino_plugins,
        runtime_loadable,
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation,
    }
}

fn probe_installed_runtimes(config: &BackendConfig, report: &mut LibraryPathReport) {
    probe_installed_runtimes_with(
        config,
        report,
        crate::supertonic::probe_onnx_runtime,
        crate::supertonic::probe_runtime,
    );
}

fn probe_installed_runtimes_with(
    config: &BackendConfig,
    report: &mut LibraryPathReport,
    mut probe_onnx: impl FnMut(&OnnxRuntimePaths, Runtime) -> Result<()>,
    mut probe_openvino: impl FnMut(OpenvinoRuntimePaths, &str) -> Result<Vec<String>>,
) {
    if report.runtime_loadable.get("default") == Some(&true)
        && let Some(onnxruntime) = report.onnxruntime_library.clone()
    {
        let paths = OnnxRuntimePaths {
            onnxruntime,
            provider: None,
        };
        if let Err(error) = probe_onnx(&paths, Runtime::Default) {
            report.runtime_loadable.insert("default", false);
            report
                .runtime_probe_errors
                .insert("default", format!("{error:#}"));
        }
    }
    if report.runtime_loadable.get("openvino") == Some(&true)
        && let (Some(library), Some(plugins)) = (
            report.openvino_library.clone(),
            report.openvino_plugins.clone(),
        )
    {
        match probe_openvino(OpenvinoRuntimePaths { library, plugins }, "auto") {
            Ok(available) if config.runtime == Runtime::Openvino => {
                let device = config
                    .canonical_device()
                    .unwrap_or_else(|_| "invalid".to_owned())
                    .to_ascii_uppercase();
                let accessible = device == "AUTO" || available.iter().any(|item| item == &device);
                report
                    .runtime_device_accessible
                    .insert("openvino", accessible);
                if !accessible {
                    report.device_probe_errors.insert(
                        "openvino",
                        format!(
                            "device {device} is not accessible; available devices: {}",
                            available.join(", ")
                        ),
                    );
                }
            }
            Ok(_) => {}
            Err(error) => {
                report.runtime_loadable.insert("openvino", false);
                report
                    .runtime_probe_errors
                    .insert("openvino", format!("{error:#}"));
            }
        }
    }
    if config.runtime == Runtime::Cuda
        && report.runtime_loadable.get("cuda") == Some(&true)
        && let (Some(onnxruntime), Some(provider)) = (
            report.onnxruntime_library.clone(),
            report.provider_library.clone(),
        )
    {
        let paths = OnnxRuntimePaths {
            onnxruntime,
            provider: Some(provider),
        };
        if let Err(error) = probe_onnx(&paths, Runtime::Cuda) {
            report.runtime_loadable.insert("cuda", false);
            report
                .runtime_probe_errors
                .insert("cuda", format!("{error:#}"));
        }
    }
}

fn system_library_dirs() -> impl Iterator<Item = PathBuf> {
    let architecture = match std::env::consts::ARCH {
        "x86_64" => Some("x86_64-linux-gnu"),
        "aarch64" => Some("aarch64-linux-gnu"),
        _ => None,
    };
    ["/lib", "/usr/lib", "/lib64", "/usr/lib64", "/usr/local/lib"]
        .into_iter()
        .map(PathBuf::from)
        .chain(
            architecture
                .into_iter()
                .flat_map(|arch| [format!("/lib/{arch}"), format!("/usr/lib/{arch}")])
                .map(PathBuf::from),
        )
        .filter(|path| path.is_dir())
}

struct RuntimeLoadabilityInput<'a> {
    search_dirs: &'a [PathBuf],
    effective_dirs: &'a [PathBuf],
    missing_dirs: &'a [PathBuf],
    exact_onnxruntime: Option<PathBuf>,
    exact_provider: Option<PathBuf>,
    exact_openvino: Option<PathBuf>,
    exact_plugins: Option<PathBuf>,
    selected_runtime: Runtime,
}

fn runtime_loadability(input: RuntimeLoadabilityInput<'_>) -> BTreeMap<&'static str, bool> {
    let RuntimeLoadabilityInput {
        search_dirs,
        effective_dirs,
        missing_dirs,
        exact_onnxruntime,
        exact_provider,
        exact_openvino,
        exact_plugins,
        selected_runtime,
    } = input;
    let supported = supported_capabilities();
    let locate =
        |exact: Option<&PathBuf>, name: &str| locate_runtime_library(exact, search_dirs, name);
    let core = locate(exact_onnxruntime.as_ref(), "libonnxruntime.so");
    let base_loadable = missing_dirs.is_empty()
        && core
            .as_deref()
            .is_some_and(|path| dependencies_resolve(path, effective_dirs));
    let openvino = locate(exact_openvino.as_ref(), "libopenvino_c.so");
    let openvino_plugins = locate_openvino_plugins(exact_plugins.as_ref(), search_dirs);
    let openvino_loadable = missing_dirs.is_empty()
        && openvino
            .as_deref()
            .is_some_and(|path| dependencies_resolve(path, effective_dirs))
        && openvino_plugins.is_some();
    let mut result = BTreeMap::from([
        ("default", base_loadable),
        (
            "openvino",
            supported.contains(&"openvino") && openvino_loadable,
        ),
    ]);
    let capability = "cuda";
    let selected = (selected_runtime.capability() == capability)
        .then_some(exact_provider.as_ref())
        .flatten();
    let provider = locate(selected, "libonnxruntime_providers_cuda.so");
    let loadable = supported.contains(&capability)
        && base_loadable
        && provider
            .as_deref()
            .is_some_and(|path| dependencies_resolve(path, effective_dirs));
    result.insert(capability, loadable);
    result
}

fn locate_runtime_library(
    exact: Option<&PathBuf>,
    search_dirs: &[PathBuf],
    name: &str,
) -> Option<PathBuf> {
    if let Some(path) = exact {
        return (path.is_absolute() && path.is_file()).then(|| canonical_or(path.clone()));
    }
    find_versioned_library(search_dirs, name).or_else(|| find_loader_cached_library(name))
}

pub(crate) fn resolve_onnx_runtime(
    config: &BackendConfig,
    config_file: &Path,
    report: &LibraryPathReport,
    runtime: Runtime,
) -> Result<OnnxRuntimePaths> {
    if runtime == Runtime::Openvino {
        bail!("runtime=openvino uses the direct OpenVINO runner, not ONNX Runtime");
    }
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "configured native library directories do not exist: {}",
            display_paths(&report.missing_library_dirs)
        );
    }
    let config_base = config_file.parent().unwrap_or_else(|| Path::new("."));
    let selected = |configured: Option<&Path>, environment: &str| {
        configured
            .map(|path| resolve(path, config_base))
            .or_else(|| env::var_os(environment).map(PathBuf::from))
    };
    let validate = |path: PathBuf, description: &str| -> Result<PathBuf> {
        if !path.is_absolute() || !path.is_file() {
            bail!(
                "{description} must name an absolute existing library: {}",
                path.display()
            );
        }
        let path = canonical_or(path);
        if !dependencies_resolve(&path, &report.effective_library_dirs) {
            bail!(
                "native library dependencies do not resolve: {}",
                path.display()
            );
        }
        Ok(path)
    };
    let onnxruntime = selected(
        config.onnxruntime_library.as_deref(),
        ONNXRUNTIME_LIBRARY_ENV,
    )
    .or_else(|| report.onnxruntime_library.clone())
    .with_context(|| {
        format!(
            "ONNX Runtime was not found; set backend.onnxruntime_library or {ONNXRUNTIME_LIBRARY_ENV}"
        )
    })?;
    let onnxruntime = validate(
        onnxruntime,
        "OMASPEAK_ONNXRUNTIME_LIBRARY or backend.onnxruntime_library",
    )?;
    let provider = if runtime == Runtime::Cuda {
        Some(
            selected(config.provider_library.as_deref(), PROVIDER_LIBRARY_ENV)
                .or_else(|| report.provider_library.clone())
                .with_context(|| {
                    format!(
                        "CUDA provider was not found; set backend.provider_library or {PROVIDER_LIBRARY_ENV}"
                    )
                })
                .and_then(|path| {
                    validate(
                        path,
                        "OMASPEAK_PROVIDER_LIBRARY or backend.provider_library",
                    )
                })?,
        )
    } else {
        None
    };
    Ok(OnnxRuntimePaths {
        onnxruntime,
        provider,
    })
}

pub(crate) fn resolve_openvino_runtime(
    config: &BackendConfig,
    config_file: &Path,
    report: &LibraryPathReport,
) -> Result<OpenvinoRuntimePaths> {
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "configured native library directories do not exist: {}",
            display_paths(&report.missing_library_dirs)
        );
    }
    let config_base = config_file.parent().unwrap_or_else(|| Path::new("."));
    let selected_library = config
        .openvino_library
        .as_deref()
        .map(|path| resolve(path, config_base))
        .or_else(|| env::var_os(OPENVINO_LIBRARY_ENV).map(PathBuf::from));
    let library = selected_library
        .map(|path| {
            if !path.is_absolute() || !path.is_file() {
                bail!(
                    "{OPENVINO_LIBRARY_ENV} or backend.openvino_library must name an absolute existing library: {}",
                    path.display()
                );
            }
            Ok(canonical_or(path))
        })
        .transpose()?
        .or_else(|| report.openvino_library.clone())
        .with_context(|| {
            format!(
                "OpenVINO C library was not found; set backend.openvino_library or {OPENVINO_LIBRARY_ENV}"
            )
        })?;
    if !dependencies_resolve(&library, &report.effective_library_dirs) {
        bail!(
            "OpenVINO C library dependencies do not resolve: {}",
            library.display()
        );
    }
    let selected_plugins = config
        .openvino_plugins
        .as_deref()
        .map(|path| resolve(path, config_base))
        .or_else(|| env::var_os(OPENVINO_PLUGINS_ENV).map(PathBuf::from));
    let plugins = selected_plugins
        .map(|path| {
            if !path.is_absolute() || !path.is_file() {
                bail!(
                    "{OPENVINO_PLUGINS_ENV} or backend.openvino_plugins must name an absolute existing plugins.xml: {}",
                    path.display()
                );
            }
            Ok(canonical_or(path))
        })
        .transpose()?
        .or_else(|| report.openvino_plugins.clone())
        .with_context(|| {
            format!(
                "OpenVINO plugins.xml was not found; set backend.openvino_plugins or {OPENVINO_PLUGINS_ENV}"
            )
        })?;
    Ok(OpenvinoRuntimePaths { library, plugins })
}

fn package_library_dirs(executable: Option<&Path>) -> Vec<PathBuf> {
    let Some(binary_dir) = executable.and_then(Path::parent) else {
        return Vec::new();
    };
    // Keep package discovery to layouts owned by Omaspeak. Cargo and local
    // validation commonly place transient native artifacts beside the binary
    // in target/{debug,release}; treating that directory as a package makes
    // those artifacts visible to setup long after the test that created them.
    let mut candidates = vec![binary_dir.join("lib")];
    if let Some(prefix) = binary_dir.parent() {
        candidates.push(prefix.join("lib/omaspeak"));
    }
    stable_unique(
        candidates
            .into_iter()
            .filter(|directory| directory.is_dir() && contains_runtime_anchor(directory))
            .map(canonical_or),
    )
}

fn contains_runtime_anchor(directory: &Path) -> bool {
    [
        "libonnxruntime.so",
        "libonnxruntime_providers_cuda.so",
        "libopenvino_c.so",
        "libaudiocpp.so",
    ]
    .iter()
    .any(|name| {
        find_versioned_library(std::slice::from_ref(&directory.to_path_buf()), name).is_some()
    })
}

fn locate_openvino_plugins(exact: Option<&PathBuf>, search_dirs: &[PathBuf]) -> Option<PathBuf> {
    if let Some(path) = exact {
        return (path.is_absolute() && path.is_file()).then(|| canonical_or(path.clone()));
    }
    search_dirs
        .iter()
        .flat_map(|directory| {
            [
                directory.join("plugins.xml"),
                directory.join("openvino/plugins.xml"),
            ]
        })
        .find(|path| path.is_file())
        .map(canonical_or)
}

fn find_loader_cached_library(name: &str) -> Option<PathBuf> {
    let output = Command::new("ldconfig").arg("-p").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = format!("{name} ");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().strip_prefix(&prefix))
        .filter_map(|line| {
            line.rsplit_once("=>")
                .map(|(_, path)| PathBuf::from(path.trim()))
        })
        .find(|path| path.is_file())
}

fn find_versioned_library(directories: &[PathBuf], name: &str) -> Option<PathBuf> {
    for directory in directories {
        let direct = directory.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        let prefix = format!("{name}.");
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        let mut matches = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|file| file.starts_with(&prefix))
                    && path.is_file()
            })
            .collect::<Vec<_>>();
        matches.sort();
        if let Some(path) = matches.pop() {
            return Some(path);
        }
    }
    None
}

fn dependencies_resolve(provider: &Path, effective_dirs: &[PathBuf]) -> bool {
    let Some(loader) = native_dynamic_loader() else {
        return false;
    };
    let mut command = Command::new(loader);
    command.arg("--list").arg(provider);
    let mut loader_dirs = effective_dirs.to_vec();
    if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
        loader_dirs.extend(env::split_paths(&existing));
    }
    if let Ok(joined) = env::join_paths(stable_unique(loader_dirs)) {
        command.env("LD_LIBRARY_PATH", joined);
    }
    command.output().is_ok_and(|output| {
        output.status.success()
            && !String::from_utf8_lossy(&output.stdout).contains("not found")
            && !String::from_utf8_lossy(&output.stderr).contains("not found")
    })
}

fn native_dynamic_loader() -> Option<&'static Path> {
    #[cfg(target_arch = "x86_64")]
    const CANDIDATES: &[&str] = &[
        "/lib64/ld-linux-x86-64.so.2",
        "/usr/lib64/ld-linux-x86-64.so.2",
    ];
    #[cfg(target_arch = "aarch64")]
    const CANDIDATES: &[&str] = &[
        "/lib/ld-linux-aarch64.so.1",
        "/usr/lib/ld-linux-aarch64.so.1",
    ];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const CANDIDATES: &[&str] = &[];

    CANDIDATES.iter().map(Path::new).find(|path| path.is_file())
}

pub fn augmented_loader_path(report: &LibraryPathReport) -> Result<Option<OsString>> {
    augmented_loader_path_with(report, env::var_os("LD_LIBRARY_PATH"))
}

pub fn reexec_loader_path(report: &LibraryPathReport) -> Result<Option<OsString>> {
    reexec_loader_path_with(
        report,
        env::var_os(REEXEC_SENTINEL),
        env::var_os("LD_LIBRARY_PATH"),
    )
}

fn reexec_loader_path_with(
    report: &LibraryPathReport,
    sentinel: Option<OsString>,
    loader_environment: Option<OsString>,
) -> Result<Option<OsString>> {
    let augmented = augmented_loader_path_with(report, loader_environment)?;
    if sentinel.is_some() {
        if augmented.is_some() {
            bail!(
                "{REEXEC_SENTINEL} is set but the effective native library directories are absent from LD_LIBRARY_PATH"
            );
        }
        Ok(None)
    } else {
        Ok(augmented)
    }
}

fn augmented_loader_path_with(
    report: &LibraryPathReport,
    loader_environment: Option<OsString>,
) -> Result<Option<OsString>> {
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "configured native library directories do not exist: {}",
            display_paths(&report.missing_library_dirs)
        );
    }
    if report.effective_library_dirs.is_empty() {
        return Ok(None);
    }
    let ambient = loader_environment
        .as_deref()
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .map(canonical_or)
        .collect::<Vec<_>>();
    let missing = report
        .effective_library_dirs
        .iter()
        .any(|directory| !ambient.contains(directory));
    if !missing {
        return Ok(None);
    }
    let paths = stable_unique(report.effective_library_dirs.iter().cloned().chain(ambient));
    env::join_paths(paths).map(Some).context(
        "native library path contains a value that cannot be represented by LD_LIBRARY_PATH",
    )
}

pub fn effective_library_path(report: &LibraryPathReport) -> Result<Option<OsString>> {
    if !report.missing_library_dirs.is_empty() {
        bail!(
            "configured native library directories do not exist: {}",
            display_paths(&report.missing_library_dirs)
        );
    }
    if report.effective_library_dirs.is_empty() {
        Ok(None)
    } else {
        env::join_paths(&report.effective_library_dirs)
            .map(Some)
            .context("native library path contains a value that cannot be represented by LD_LIBRARY_PATH")
    }
}

fn resolve(path: &Path, base: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        return PathBuf::new();
    }
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    };
    canonical_or(absolute)
}

fn canonical_or(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

fn stable_unique(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for path in paths {
        if !result.contains(&path) {
            result.push(path);
        }
    }
    result
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
#[path = "../tests/unit/runtime.rs"]
mod tests;
