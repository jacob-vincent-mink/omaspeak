pub mod menu;
pub mod model;
pub mod systemd;
pub mod wizard;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::catalog;
use crate::config::Config;
use crate::paths::AppPaths;

#[derive(Clone, Debug, Serialize)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
    pub remediation: Option<String>,
}

struct EngineSummary<'a> {
    backend_kind: &'a str,
    model_name: &'a str,
    sample_rate: i32,
    effective_runtime: crate::backend::Runtime,
    fallback_used: bool,
}

/// Load the current configuration for a setup flow. Pre-release configuration
/// files are intentionally not migrated field by field: an invalid file is
/// represented by current defaults until the user applies a setup change.
/// Read-only setup actions therefore leave the invalid bytes untouched, while
/// successful setup saves replace the whole file with the current schema.
pub fn load_config(path: &Path) -> Result<Config> {
    setup_config(path).map(|(config, _)| config)
}

/// Describe an invalid configuration without preventing the setup UI and
/// repair commands from starting. File-system errors remain fatal so setup
/// never mistakes an unreadable file for a replaceable pre-release config.
pub fn config_recovery(path: &Path) -> Result<Option<String>> {
    setup_config(path).map(|(_, error)| error)
}

fn setup_config(path: &Path) -> Result<(Config, Option<String>)> {
    let input = match fs::read_to_string(path) {
        Ok(input) => input,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Config::default(), None));
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read config {}", path.display()));
        }
    };
    match toml::from_str(&input) {
        Ok(config) => Ok((config, None)),
        Err(error) => Ok((
            Config::default(),
            Some(format!("parse config {}: {error}", path.display())),
        )),
    }
}

pub fn ensure_config(path: &Path) -> Result<Config> {
    let exists = path.exists();
    let config = load_config(path)?;
    if !exists {
        config.save(path)?;
    }
    Ok(config)
}

pub fn checks(path: &Path, paths: &AppPaths) -> Vec<Check> {
    let mut result = checks_with(
        path,
        paths,
        crate::runtime_inventory::probe,
        |config, paths| {
            let engine = crate::engine::Engine::load(config, paths)?;
            validate_engine_summary(
                config,
                EngineSummary {
                    backend_kind: engine.backend_kind,
                    model_name: &engine.model_name,
                    sample_rate: engine.sample_rate,
                    effective_runtime: engine.effective_runtime,
                    fallback_used: engine.fallback_used,
                },
            )
        },
    );
    if let Ok(config) = Config::load(path) {
        let inventory = crate::audio_devices::inventory("output", &config.audio.device);
        if crate::audio_devices::is_default(&config.audio.device) {
            result.push(ok(
                "audio_device",
                "System default (resolved when audio opens)",
            ));
        } else if inventory["devices"].as_array().is_some_and(|devices| {
            devices
                .iter()
                .any(|d| d["selector"] == config.audio.device && d["available"] == true)
        }) {
            result.push(ok("audio_device", config.audio.device));
        } else {
            result.push(fail(
                "audio_device",
                format!("{} is unavailable", config.audio.device),
                "connect the device or run `omaspeak setup audio`",
            ));
        }
    }
    result
}

fn validate_engine_summary(config: &Config, engine: EngineSummary<'_>) -> Result<String> {
    if engine.fallback_used || engine.effective_runtime != config.backend.runtime {
        bail!(
            "configured {} runtime initialized as {}{}",
            config.backend.runtime.name(),
            engine.effective_runtime.name(),
            if engine.fallback_used {
                " through fallback"
            } else {
                ""
            }
        );
    }
    Ok(format!(
        "{} initialized {} on {} at {} Hz",
        engine.backend_kind,
        engine.model_name,
        config.backend.canonical_device()?,
        engine.sample_rate
    ))
}

fn checks_with(
    path: &Path,
    paths: &AppPaths,
    probe_runtime: impl FnOnce(&crate::backend::BackendConfig, &Path) -> crate::runtime_inventory::Probe,
    probe_engine: impl FnOnce(&Config, &AppPaths) -> Result<String>,
) -> Vec<Check> {
    let mut result = Vec::new();
    let config = match Config::load(path) {
        Ok(config) => {
            result.push(ok("config", path.display().to_string()));
            config
        }
        Err(error) => {
            result.push(fail(
                "config",
                format!("{error:#}"),
                "run `omaspeak setup all --accept-license OpenRAIL-M`",
            ));
            return result;
        }
    };

    match catalog::backends()
        .iter()
        .find(|backend| backend.kind == config.backend.kind)
    {
        Some(backend) => result.push(ok(
            "backend",
            format!("{} runtime adapter is available", backend.kind),
        )),
        None => result.push(fail(
            "backend",
            format!("{} is not registered", config.backend.kind),
            "choose a backend shown by `omaspeak setup runtime`",
        )),
    }

    let model_ready = match catalog::model(&config.model.name) {
        Some(spec) => match model::verify(paths, spec) {
            Ok(()) => {
                result.push(ok(
                    "model",
                    model::model_directory(paths, spec).display().to_string(),
                ));
                true
            }
            Err(error) => {
                result.push(fail(
                    "model",
                    format!("{error:#}"),
                    format!("run `omaspeak setup model --download {}`", spec.id),
                ));
                false
            }
        },
        None => {
            let directory = config.model_directory(paths);
            if directory.exists() {
                result.push(ok(
                    "model",
                    format!("custom model at {}", directory.display()),
                ));
                true
            } else {
                result.push(fail(
                    "model",
                    format!("custom model directory is missing: {}", directory.display()),
                    "set model.directory or install a catalog model",
                ));
                false
            }
        }
    };
    let voice_ready = match crate::voices::installed(&config, paths)
        .and_then(|voices| crate::voices::validate_selected(&config, &voices).map(|()| voices))
    {
        Ok(voices) => {
            let selected = voices
                .iter()
                .find(|voice| voice.id == config.model.voice)
                .expect("validated voice must exist");
            result.push(ok(
                "voice",
                format!("{} (speaker ID {})", selected.name, selected.id),
            ));
            true
        }
        Err(error) => {
            result.push(fail(
                "voice",
                format!("{error:#}"),
                "run `omaspeak setup model` to choose an available voice",
            ));
            false
        }
    };
    let runtime_probe = probe_runtime(&config.backend, path);
    let runtime_name = config.backend.runtime.name();
    if runtime_probe.loadable {
        result.push(ok(
            "runtime",
            format!("{runtime_name} runtime is installed and its libraries resolve"),
        ));
    } else {
        result.push(fail(
            "runtime",
            format!("{runtime_name} runtime is not installed or its libraries do not resolve"),
            runtime_probe.errors.join("; "),
        ));
    }
    if config.backend.runtime == crate::backend::Runtime::Openvino {
        if runtime_probe.device_accessible == Some(true) {
            result.push(ok(
                "device",
                format!(
                    "OpenVINO device {} is accessible",
                    config
                        .backend
                        .canonical_device()
                        .unwrap_or_else(|_| config.backend.device.clone())
                ),
            ));
        } else {
            result.push(fail(
                "device",
                runtime_probe.errors.join("; "),
                "install the device plugin and driver, or select an accessible OpenVINO device",
            ));
        }
    }
    let npu_cache = crate::supertonic::npu_cache_state(&config, paths);
    let mut npu_ready = true;
    if npu_cache.required {
        if npu_cache.ready {
            result.push(ok(
                "npu-cache",
                format!(
                    "{} at {}",
                    npu_cache.detail,
                    npu_cache
                        .directory
                        .as_deref()
                        .map_or_else(|| "(unknown)".into(), |path| path.display().to_string())
                ),
            ));
        } else {
            npu_ready = false;
            result.push(fail(
                "npu-cache",
                npu_cache.detail,
                "rerun model or full setup to compile every supported Intel NPU shape before synthesis",
            ));
        }
    }
    if runtime_probe.ready && model_ready && voice_ready && npu_ready {
        match probe_engine(&config, paths) {
            Ok(detail) => result.push(ok("engine", detail)),
            Err(error) => result.push(fail(
                "engine",
                format!("{error:#}"),
                "rerun runtime or model setup to repeat the model-backed provider proof",
            )),
        }
    } else {
        result.push(fail(
            "engine",
            "model-backed provider check skipped because a prerequisite above failed",
            "fix the backend, runtime, model, voice, or NPU cache settings shown above",
        ));
    }
    append_environment_checks(&mut result, paths, inspect_environment(paths));
    result
}

#[derive(Clone, Copy, Debug)]
struct EnvironmentStatus {
    audio_available: bool,
    launcher_installed: bool,
    systemctl_available: bool,
    service_active: bool,
}

fn inspect_environment(paths: &AppPaths) -> EnvironmentStatus {
    let systemctl_available = command_exists("systemctl");
    EnvironmentStatus {
        audio_available: command_exists("pw-play") || command_exists("aplay"),
        launcher_installed: menu::launcher_path(paths).is_file(),
        systemctl_available,
        service_active: systemctl_available && systemd::is_active(),
    }
}

fn append_environment_checks(
    result: &mut Vec<Check>,
    paths: &AppPaths,
    environment: EnvironmentStatus,
) {
    result.push(if environment.audio_available {
        ok("audio", "pw-play or aplay is available")
    } else {
        ok(
            "audio",
            "no playback command found (optional; WAV output and --no-play remain available)",
        )
    });
    let launcher = menu::launcher_path(paths);
    result.push(if environment.launcher_installed {
        ok("launcher", launcher.display().to_string())
    } else {
        ok(
            "launcher",
            format!(
                "not installed (optional; run `omaspeak setup menu`): {}",
                launcher.display()
            ),
        )
    });
    let service = systemd::service_path(paths);
    result.push(if environment.service_active {
        if service.is_file() {
            ok("systemd", format!("active: {}", service.display()))
        } else {
            ok(
                "systemd",
                "active (unit is managed outside Omaspeak's user config)",
            )
        }
    } else if service.is_file() {
        if !environment.systemctl_available {
            ok(
                "systemd",
                format!(
                    "unit installed at {}; systemctl is unavailable",
                    service.display()
                ),
            )
        } else {
            ok(
                "systemd",
                format!("installed but inactive (optional): {}", service.display()),
            )
        }
    } else {
        ok(
            "systemd",
            format!("not installed (optional): {}", service.display()),
        )
    });
}

pub fn print_checks(path: &Path, paths: &AppPaths, json: bool) -> Result<()> {
    print_check_results(checks(path, paths), json)
}

fn print_check_results(checks: Vec<Check>, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        for check in &checks {
            println!(
                "{} {:<9} {}",
                if check.ok { "ok" } else { "error" },
                check.name,
                check.detail
            );
            if let Some(remediation) = &check.remediation {
                println!("  fix: {remediation}");
            }
        }
    }
    if checks.iter().any(|check| !check.ok) {
        bail!("setup checks failed")
    }
    Ok(())
}

pub fn print_checks_event(path: &Path, paths: &AppPaths) -> Result<()> {
    print_check_results_event(checks(path, paths))
}

fn print_check_results_event(checks: Vec<Check>) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "event": "checks",
            "checks": checks,
        }))?
    );
    if checks.iter().any(|check| !check.ok) {
        bail!("setup checks failed")
    }
    Ok(())
}

pub fn print_runtime(config_path: &Path, json: bool) -> Result<()> {
    let config = load_config(config_path)?;
    let locations = crate::runtime::discover(&config.backend, config_path);
    let inventory = crate::runtime_inventory::inventory(&config.backend, config_path);
    let hardware = crate::hardware::detect();
    let recommendation = crate::hardware::recommend(
        &hardware,
        crate::hardware::provider_availability(&config.backend, &locations),
    );
    let value = serde_json::json!({
        "backends": catalog::backends(),
        "supported_capabilities": crate::backend::supported_capabilities(),
        "libraries": locations,
        "runtime_device_matrix": {
            "default": ["auto", "cpu"],
            "cuda": ["auto", "gpu"],
            "vulkan": ["auto", "gpu"],
            "hip": ["auto", "gpu"],
            "openvino": ["auto", "cpu", "gpu", "npu"]
        },
        "inventory": inventory,
        "hardware": hardware,
        "recommendation": recommendation,
        "loader_environment": std::env::var_os("LD_LIBRARY_PATH")
            .map(|value| value.to_string_lossy().into_owned()),
        "models": catalog::models(),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("Omaspeak runtime catalog\n");
        println!("Recommendation: {}", recommendation.detail);
        if hardware.devices.is_empty() {
            println!("Detected accelerator hardware: none");
        } else {
            for device in &hardware.devices {
                println!(
                    "Detected hardware: {} vendor={} class={} driver={} capabilities={}",
                    device.address,
                    device.vendor,
                    device.class,
                    device.driver.as_deref().unwrap_or("unbound"),
                    device.capabilities.join(",")
                );
            }
        }
        for error in &hardware.errors {
            println!("Hardware discovery note: {error}");
        }
        println!();
        for state in &inventory {
            println!(
                "{} / {}: supported={} discovered={} configured={} loadable={} device_accessible={} ready={} source={}",
                state.runtime,
                state.device,
                state.supported,
                state.discovered,
                state.configured,
                state.probe.loadable,
                match state.probe.device_accessible {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "unverified",
                },
                state.probe.ready,
                state.source
            );
            for error in &state.probe.errors {
                println!("  error: {error}");
            }
            if !state.probe.ready {
                for action in &state.remediation {
                    println!("  fix: {action}");
                }
            }
        }
        println!("Backends:");
        for backend in catalog::backends() {
            println!("  {}\t{}", backend.kind, backend.description);
        }
        println!(
            "Supported runtime capabilities: {}",
            crate::backend::supported_capabilities().join(", ")
        );
        println!(
            "Configured native library directories: {}",
            display_paths(&locations.configured_library_dirs)
        );
        println!(
            "{} native library directories: {}",
            crate::runtime::LIBRARY_PATH_ENV,
            display_paths(&locations.environment_library_dirs)
        );
        println!(
            "Package native library directories: {}",
            display_paths(&locations.package_library_dirs)
        );
        println!(
            "Effective app-owned native library path: {}",
            display_paths(&locations.effective_library_dirs)
        );
        println!(
            "Selected audio.cpp provider: {}",
            locations
                .audiocpp_library
                .as_deref()
                .map_or_else(|| "not found".to_owned(), |path| path.display().to_string())
        );
        println!(
            "Selected OpenVINO C library: {}",
            locations
                .openvino_library
                .as_deref()
                .map_or_else(|| "not found".to_owned(), |path| path.display().to_string())
        );
        println!(
            "Selected OpenVINO plugins.xml: {}",
            locations
                .openvino_plugins
                .as_deref()
                .map_or_else(|| "not found".to_owned(), |path| path.display().to_string())
        );
        if !locations.missing_library_dirs.is_empty() {
            println!(
                "Missing native library directories: {}",
                display_paths(&locations.missing_library_dirs)
            );
        }
        println!("\nRuntime and device choices:");
        println!(
            "  default    auto, cpu                 {}",
            if locations.runtime_loadable.get("default") == Some(&true) {
                "runtime installed"
            } else {
                "runtime not loadable"
            }
        );
        println!(
            "  openvino   auto, cpu, gpu, npu       {}",
            if locations.runtime_loadable.get("openvino") == Some(&true) {
                "runtime installed (device accessibility checked by `omaspeak setup check`)"
            } else {
                "runtime not loadable"
            }
        );
        println!(
            "  cuda       auto, gpu                 {}",
            if locations.runtime_loadable.get("cuda") == Some(&true) {
                "loadable"
            } else {
                "runtime not loadable"
            }
        );
        for runtime in ["vulkan", "hip"] {
            println!(
                "  {runtime:<10} auto, gpu                 {}",
                if locations.runtime_loadable.get(runtime) == Some(&true) {
                    "provider found; model-backed setup proves capability"
                } else {
                    "provider not found"
                }
            );
        }
        println!("\nCatalog models:");
        for model in catalog::models() {
            let download = model.download_size();
            println!(
                "  {:<24} {:>4} MiB  {}{}",
                model.id,
                download.div_ceil(1024 * 1024),
                model.description,
                if model.npu_capable {
                    " [Intel NPU setup available]"
                } else {
                    ""
                }
            );
        }
        println!("\nRun `omaspeak setup` for guided setup with arrow-key selection.");
        println!("Run `omaspeak setup model --list` to see local installation status.");
        for remediation in &locations.remediation {
            println!("  fix: {remediation}");
        }
    }
    Ok(())
}

fn display_paths(paths: &[std::path::PathBuf]) -> String {
    if paths.is_empty() {
        "(none)".into()
    } else {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    }
}

fn command_exists(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join(name).is_file())
    })
}

fn ok(name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        name: name.into(),
        ok: true,
        detail: detail.into(),
        remediation: None,
    }
}

fn fail(
    name: impl Into<String>,
    detail: impl Into<String>,
    remediation: impl Into<String>,
) -> Check {
    Check {
        name: name.into(),
        ok: false,
        detail: detail.into(),
        remediation: Some(remediation.into()),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_mod.rs"]
mod tests;

pub mod audio;
