pub mod menu;
pub mod model;
pub mod systemd;
pub mod wizard;

use std::path::Path;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::catalog;
use crate::config::Config;
use crate::paths::AppPaths;

#[derive(Serialize)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
    pub remediation: Option<String>,
}

pub fn ensure_config(path: &Path) -> Result<Config> {
    if path.exists() {
        Config::load(path)
    } else {
        let config = Config::default();
        config.save(path)?;
        Ok(config)
    }
}

pub fn checks(path: &Path, paths: &AppPaths) -> Vec<Check> {
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
                "run `omaspeak setup all`",
            ));
            return result;
        }
    };

    match catalog::backends()
        .iter()
        .find(|backend| backend.kind == config.backend.kind)
    {
        Some(backend) if backend.built => {
            result.push(ok("backend", format!("{} is compiled", backend.kind)))
        }
        Some(backend) => result.push(fail(
            "backend",
            format!("{} is registered but unavailable", backend.kind),
            "install a build containing that backend",
        )),
        None => result.push(fail(
            "backend",
            format!("{} is not registered", config.backend.kind),
            "choose a backend shown by `omaspeak setup runtime`",
        )),
    }

    match catalog::model(&config.model.name) {
        Some(spec) => match model::verify(paths, spec) {
            Ok(()) => result.push(ok(
                "model",
                model::model_directory(paths, spec).display().to_string(),
            )),
            Err(error) => result.push(fail(
                "model",
                format!("{error:#}"),
                format!("run `omaspeak setup model --download {}`", spec.id),
            )),
        },
        None => {
            let directory = config.model_directory(paths);
            if directory.exists() {
                result.push(ok(
                    "model",
                    format!("custom model at {}", directory.display()),
                ));
            } else {
                result.push(fail(
                    "model",
                    format!("custom model directory is missing: {}", directory.display()),
                    "set model.directory or install a catalog model",
                ));
            }
        }
    }
    match crate::engine::Engine::load(&config, paths) {
        Ok(engine) => result.push(ok(
            "engine",
            format!(
                "{} initialized {} at {} Hz",
                engine.backend_kind, engine.model_name, engine.sample_rate
            ),
        )),
        Err(error) => result.push(fail(
            "engine",
            format!("{error:#}"),
            "fix the backend/runtime/model settings shown above",
        )),
    }
    result.push(if command_exists("pw-play") || command_exists("aplay") {
        ok("audio", "pw-play or aplay is available")
    } else {
        fail(
            "audio",
            "no playback command found",
            "install PipeWire tools or ALSA utilities; --no-play still works",
        )
    });
    let launcher = menu::launcher_path(paths);
    result.push(if launcher.is_file() {
        ok("launcher", launcher.display().to_string())
    } else {
        fail(
            "launcher",
            format!("desktop launcher is missing: {}", launcher.display()),
            "run `omaspeak setup menu`",
        )
    });
    let service = systemd::service_path(paths);
    let systemctl_available = command_exists("systemctl");
    let service_active = systemctl_available && systemd::is_active();
    result.push(if service_active {
        if service.is_file() {
            ok("systemd", format!("active: {}", service.display()))
        } else {
            ok(
                "systemd",
                "active (unit is managed outside Omaspeak's user config)",
            )
        }
    } else if service.is_file() {
        if !systemctl_available {
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
    result
}

pub fn print_checks(path: &Path, paths: &AppPaths, json: bool) -> Result<()> {
    let checks = checks(path, paths);
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
    let checks = checks(path, paths);
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

pub fn print_runtime(json: bool) -> Result<()> {
    let value = serde_json::json!({
        "backends": catalog::backends(),
        "compiled_capabilities": crate::backend::compiled_capabilities(),
        "runtime_device_matrix": {
            "default": ["auto", "cpu"],
            "cuda": ["auto", "gpu"],
            "openvino": ["auto", "cpu", "gpu", "npu", "AUTO:<devices>", "HETERO:<devices>", "MULTI:<devices>"]
        },
        "models": catalog::models(),
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("Omaspeak runtime catalog\n");
        println!("Backends:");
        for backend in catalog::backends() {
            println!(
                "  {}\t{}\t{}",
                backend.kind,
                if backend.built { "built" } else { "not built" },
                backend.description
            );
        }
        println!(
            "Compiled runtime capabilities: {}",
            crate::backend::compiled_capabilities().join(", ")
        );
        println!("\nRuntime and device choices:");
        println!("  default    auto, cpu                 built in");
        println!(
            "  openvino   auto, cpu, gpu, npu       {}",
            if crate::backend::compiled_capabilities().contains(&"openvino") {
                "available"
            } else {
                "unavailable in this build"
            }
        );
        println!("  cuda       auto, gpu                 unavailable in this build");
        println!("  OpenVINO also accepts AUTO:<devices>, HETERO:<devices>, and MULTI:<devices>.");
        println!("\nCatalog models:");
        for model in catalog::models() {
            let download = model.archive_size
                + model
                    .supplemental_files
                    .iter()
                    .map(|file| file.size)
                    .sum::<u64>();
            println!(
                "  {:<24} {:>4} MiB  {}{}",
                model.id,
                download.div_ceil(1024 * 1024),
                model.description,
                if model.npu_capable {
                    " [Intel NPU validated]"
                } else {
                    ""
                }
            );
        }
        println!("\nRun `omaspeak setup` for guided setup with arrow-key selection.");
        println!("Run `omaspeak setup model --list` to see local installation status.");
    }
    Ok(())
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
