//! Native provider discovery and loader-path construction.
//!
//! Omaspeak loads one complete audio.cpp provider or one complete OpenVINO
//! installation. It never assembles providers from separate core/plugin files.

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::backend::{BackendConfig, Runtime};

pub const LIBRARY_PATH_ENV: &str = "OMASPEAK_LIBRARY_PATH";
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
    pub audiocpp_library: Option<PathBuf>,
    pub openvino_library: Option<PathBuf>,
    pub openvino_plugins: Option<PathBuf>,
    pub runtime_loadable: BTreeMap<&'static str, bool>,
    pub runtime_probe_errors: BTreeMap<&'static str, String>,
    pub runtime_device_accessible: BTreeMap<&'static str, bool>,
    pub device_probe_errors: BTreeMap<&'static str, String>,
    pub remediation: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpenvinoRuntimePaths {
    pub library: PathBuf,
    pub plugins: PathBuf,
}

impl Runtime {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Openvino => "openvino",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Hip => "hip",
        }
    }

    pub const fn uses_audiocpp(self) -> bool {
        !matches!(self, Self::Openvino)
    }
}

impl LibraryPathReport {
    pub fn remediation(&self, runtime: Runtime) -> Option<String> {
        if self.runtime_loadable.get(runtime.name()) == Some(&true) {
            return None;
        }
        if !self.missing_library_dirs.is_empty() {
            return Some(format!(
                "correct missing native-provider directories in backend.library_dirs or {LIBRARY_PATH_ENV}: {}",
                display_paths(&self.missing_library_dirs)
            ));
        }
        if let Some(error) = self.runtime_probe_errors.get(runtime.name()) {
            return Some(format!("provider validation failed: {error}"));
        }
        Some(match runtime {
            Runtime::Openvino => format!(
                "select a complete OpenVINO installation with backend.openvino_library and backend.openvino_plugins, or {OPENVINO_LIBRARY_ENV} and {OPENVINO_PLUGINS_ENV}"
            ),
            Runtime::Default => format!(
                "install the packaged audio.cpp CPU provider or select a complete audio.cpp installation with backend.library or {LIBRARY_PATH_ENV}"
            ),
            Runtime::Cuda | Runtime::Vulkan | Runtime::Hip => format!(
                "select a complete {}-enabled audio.cpp installation with backend.library or {LIBRARY_PATH_ENV}",
                runtime.name()
            ),
        })
    }
}

pub fn inspect(config: &BackendConfig, config_file: &Path) -> LibraryPathReport {
    discover(config, config_file)
}

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
        .map(canonical_or)
        .collect::<Vec<_>>();
    let package_library_dirs = package_library_dirs(executable);
    let exact_files = [
        config
            .library
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
        env::var_os(OPENVINO_LIBRARY_ENV).map(PathBuf::from),
        env::var_os(OPENVINO_PLUGINS_ENV).map(PathBuf::from),
    ];
    let exact_dirs = exact_files
        .iter()
        .flatten()
        .filter_map(|path| path.parent().map(Path::to_path_buf));
    let requested_dirs = configured_library_dirs
        .iter()
        .chain(&environment_library_dirs)
        .cloned()
        .collect::<Vec<_>>();
    let missing_library_dirs = stable_unique(
        requested_dirs
            .iter()
            .filter(|path| !path.is_absolute() || !path.is_dir())
            .cloned(),
    );
    let effective_library_dirs = stable_unique(
        requested_dirs
            .into_iter()
            .filter(|path| path.is_absolute() && path.is_dir())
            .chain(exact_dirs)
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
    let audiocpp_library = config
        .library
        .as_deref()
        .map(|path| resolve(path, config_base))
        .filter(|path| path.is_file())
        .map(canonical_or)
        .or_else(|| find_versioned_library(&search_dirs, "libaudiocpp.so"));
    let selected_openvino = config
        .openvino_library
        .as_deref()
        .map(|path| resolve(path, config_base))
        .or_else(|| env::var_os(OPENVINO_LIBRARY_ENV).map(PathBuf::from));
    let selected_plugins = config
        .openvino_plugins
        .as_deref()
        .map(|path| resolve(path, config_base))
        .or_else(|| env::var_os(OPENVINO_PLUGINS_ENV).map(PathBuf::from));
    let openvino_library =
        locate_library(selected_openvino.as_ref(), &search_dirs, "libopenvino_c.so");
    let openvino_plugins = locate_openvino_plugins(selected_plugins.as_ref(), &search_dirs);
    let audio_present = missing_library_dirs.is_empty() && audiocpp_library.is_some();
    let openvino_present =
        missing_library_dirs.is_empty() && openvino_library.is_some() && openvino_plugins.is_some();
    let runtime_loadable = BTreeMap::from([
        ("default", audio_present),
        ("cuda", audio_present),
        ("vulkan", audio_present),
        ("hip", audio_present),
        ("openvino", openvino_present),
    ]);
    let mut remediation = Vec::new();
    if !missing_library_dirs.is_empty() {
        remediation.push(format!(
            "correct missing native-provider directories: {}",
            display_paths(&missing_library_dirs)
        ));
    }
    if !audio_present {
        remediation.push(
            "select a complete audio.cpp provider; Omaspeak does not install optional vendor runtimes"
                .into(),
        );
    }
    if !openvino_present {
        remediation.push(
            "select a complete OpenVINO installation; Omaspeak does not install OpenVINO".into(),
        );
    }
    LibraryPathReport {
        configured_library_dirs,
        environment_library_dirs,
        package_library_dirs,
        effective_library_dirs,
        missing_library_dirs,
        audiocpp_library,
        openvino_library,
        openvino_plugins,
        runtime_loadable,
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation,
    }
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
    let base = config_file.parent().unwrap_or_else(|| Path::new("."));
    let library = config
        .openvino_library
        .as_deref()
        .map(|path| resolve(path, base))
        .or_else(|| env::var_os(OPENVINO_LIBRARY_ENV).map(PathBuf::from))
        .or_else(|| report.openvino_library.clone())
        .with_context(|| {
            format!(
                "OpenVINO C library was not found; set backend.openvino_library or {OPENVINO_LIBRARY_ENV}"
            )
        })?;
    if !library.is_absolute() || !library.is_file() {
        bail!(
            "OpenVINO C library must be an absolute existing file: {}",
            library.display()
        );
    }
    let plugins = config
        .openvino_plugins
        .as_deref()
        .map(|path| resolve(path, base))
        .or_else(|| env::var_os(OPENVINO_PLUGINS_ENV).map(PathBuf::from))
        .or_else(|| report.openvino_plugins.clone())
        .with_context(|| {
            format!(
                "OpenVINO plugins.xml was not found; set backend.openvino_plugins or {OPENVINO_PLUGINS_ENV}"
            )
        })?;
    if !plugins.is_absolute() || !plugins.is_file() {
        bail!(
            "OpenVINO plugins.xml must be an absolute existing file: {}",
            plugins.display()
        );
    }
    Ok(OpenvinoRuntimePaths {
        library: canonical_or(library),
        plugins: canonical_or(plugins),
    })
}

fn package_library_dirs(executable: Option<&Path>) -> Vec<PathBuf> {
    let Some(binary_dir) = executable.and_then(Path::parent) else {
        return Vec::new();
    };
    let mut candidates = vec![binary_dir.join("lib")];
    if let Some(prefix) = binary_dir.parent() {
        candidates.push(prefix.join("lib/omaspeak"));
    }
    stable_unique(
        candidates
            .into_iter()
            .filter(|directory| {
                directory.is_dir()
                    && (find_versioned_library(std::slice::from_ref(directory), "libaudiocpp.so")
                        .is_some()
                        || find_versioned_library(
                            std::slice::from_ref(directory),
                            "libopenvino_c.so",
                        )
                        .is_some())
            })
            .map(canonical_or),
    )
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

fn locate_library(exact: Option<&PathBuf>, search_dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    if let Some(path) = exact {
        return (path.is_absolute() && path.is_file()).then(|| canonical_or(path.clone()));
    }
    find_versioned_library(search_dirs, name)
}

fn locate_openvino_plugins(exact: Option<&PathBuf>, search_dirs: &[PathBuf]) -> Option<PathBuf> {
    if let Some(path) = exact {
        return (path.is_absolute() && path.is_file()).then(|| canonical_or(path.clone()));
    }
    find_openvino_plugins(search_dirs)
}

pub fn find_openvino_plugins(search_dirs: &[PathBuf]) -> Option<PathBuf> {
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

pub fn find_versioned_library(directories: &[PathBuf], name: &str) -> Option<PathBuf> {
    for directory in directories {
        let direct = directory.join(name);
        if direct.is_file() {
            return Some(canonical_or(direct));
        }
        let prefix = format!("{name}.");
        let mut matches = std::fs::read_dir(directory)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter_map(|path| {
                let version = path
                    .is_file()
                    .then(|| path.file_name()?.to_str()?.strip_prefix(&prefix))
                    .flatten()?
                    .split('.')
                    .map(str::parse::<u64>)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .ok()?;
                (!version.is_empty()).then_some((version, path))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|(left, _), (right, _)| left.cmp(right));
        if let Some((_, path)) = matches.pop() {
            return Some(canonical_or(path));
        }
    }
    None
}

pub fn augmented_loader_path(report: &LibraryPathReport) -> Result<Option<OsString>> {
    augmented_loader_path_with(report, env::var_os("LD_LIBRARY_PATH"))
}

pub fn reexec_loader_path(report: &LibraryPathReport) -> Result<Option<OsString>> {
    let augmented = augmented_loader_path(report)?;
    if env::var_os(REEXEC_SENTINEL).is_some() {
        if augmented.is_some() {
            bail!(
                "{REEXEC_SENTINEL} is set but native provider directories are absent from LD_LIBRARY_PATH"
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
    let ambient = loader_environment
        .as_deref()
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .map(canonical_or)
        .collect::<Vec<_>>();
    if report
        .effective_library_dirs
        .iter()
        .all(|path| ambient.contains(path))
    {
        return Ok(None);
    }
    let paths = stable_unique(report.effective_library_dirs.iter().cloned().chain(ambient));
    env::join_paths(paths)
        .map(Some)
        .context("native library path cannot be represented by LD_LIBRARY_PATH")
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
            .context("native library path cannot be represented by LD_LIBRARY_PATH")
    }
}

fn resolve(path: &Path, base: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        return PathBuf::new();
    }
    canonical_or(if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    })
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
