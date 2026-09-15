use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::paths::AppPaths;

const UNIT: &str = "omaspeak.service";

pub fn service_path(paths: &AppPaths) -> PathBuf {
    paths
        .config_file
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join("systemd/user/omaspeak.service")
}

pub fn unit_uses_config(unit_path: &Path, config: &Path) -> bool {
    let Ok(unit) = fs::read_to_string(unit_path) else {
        return false;
    };
    quote_systemd_path(config, "Omaspeak configuration").is_ok_and(|quoted| {
        let argument = format!(" --config {quoted} daemon");
        unit.lines()
            .any(|line| line.starts_with("ExecStart=") && line.contains(&argument))
    })
}

pub fn generate(binary: &Path, config: &Path) -> Result<String> {
    Ok(format!(
        "[Unit]\nDescription=Omaspeak local text-to-speech daemon\nPartOf=graphical-session.target\nAfter=graphical-session.target pipewire.service\n\n[Service]\nType=simple\nExecStart={} --config {} daemon\nRestart=on-failure\nRestartSec=1\nEnvironment=XDG_RUNTIME_DIR=%t\n\n[Install]\nWantedBy=graphical-session.target\n",
        quote_systemd_path(binary, "Omaspeak executable")?,
        quote_systemd_path(config, "Omaspeak configuration")?,
    ))
}

pub fn install(paths: &AppPaths, config: &Path, start: bool) -> Result<PathBuf> {
    let path = service_path(paths);
    let binary = std::env::current_exe()?.canonicalize()?;
    let unit = generate(&binary, config)?;
    write_atomic(&path, unit.as_bytes())?;
    apply_service_lifecycle(start, systemctl, is_active)?;
    Ok(path)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    let _ = systemctl(&["disable", "--now", UNIT]);
    let path = service_path(paths);
    if path.exists() {
        fs::remove_file(&path)?;
    }
    systemctl(&["daemon-reload"])
}

pub fn status(paths: &AppPaths) -> Result<()> {
    let path = service_path(paths);
    println!("unit: {}", path.display());
    if !path.exists() {
        bail!("omaspeak systemd user service is not installed");
    }
    let status = Command::new("systemctl")
        .args(["--user", "status", UNIT, "--no-pager"])
        .status()
        .context("run systemctl --user status")?;
    if !status.success() {
        bail!("omaspeak.service is not running");
    }
    Ok(())
}

pub fn is_active() -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", UNIT])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn reload_if_was_active(was_active: bool) -> Result<bool> {
    reload_if_was_active_with(was_active, || systemctl(&["try-restart", UNIT]), is_active)
}

pub fn restart() -> Result<bool> {
    restart_with(|| systemctl(&["restart", UNIT]), is_active)
}

fn restart_with(
    restart_service: impl FnOnce() -> Result<()>,
    active_after: impl FnOnce() -> bool,
) -> Result<bool> {
    restart_service()?;
    if !active_after() {
        bail!("{UNIT} did not remain active after restart");
    }
    Ok(true)
}

fn reload_if_was_active_with(
    was_active: bool,
    try_restart: impl FnOnce() -> Result<()>,
    active_after: impl FnOnce() -> bool,
) -> Result<bool> {
    if !was_active {
        return Ok(false);
    }
    try_restart()?;
    if !active_after() {
        bail!("{UNIT} stopped while applying setup changes");
    }
    Ok(true)
}

fn apply_service_lifecycle(
    start: bool,
    mut run_systemctl: impl FnMut(&[&str]) -> Result<()>,
    active_after: impl FnOnce() -> bool,
) -> Result<()> {
    run_systemctl(&["daemon-reload"])?;
    run_systemctl(&["enable", UNIT])?;
    if start {
        run_systemctl(&["restart", UNIT])?;
        if !active_after() {
            bail!("{UNIT} did not remain active after restart");
        }
    }
    Ok(())
}

fn systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .context("run systemctl --user")?;
    if !status.success() {
        bail!("systemctl --user command failed");
    }
    Ok(())
}

fn quote_systemd_path(path: &Path, description: &str) -> Result<String> {
    let value = escape_systemd_value(path.as_os_str(), description)?;
    if value.is_empty() {
        bail!("{description} path is empty");
    }
    Ok(format!("\"{value}\""))
}

fn escape_systemd_value(value: &OsStr, description: &str) -> Result<String> {
    let value = value
        .to_str()
        .with_context(|| format!("{description} is not valid UTF-8"))?;
    if value.chars().any(char::is_control) {
        bail!("{description} contains a control character unsupported by systemd units");
    }
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '%' => escaped.push_str("%%"),
            _ => escaped.push(character),
        }
    }
    Ok(escaped)
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/setup_systemd.rs"]
mod tests;
