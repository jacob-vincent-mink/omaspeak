use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::systemd::write_atomic;
use crate::paths::AppPaths;

pub fn launcher_path(paths: &AppPaths) -> PathBuf {
    paths
        .data_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("applications/omaspeak-settings.desktop")
}

pub fn install(paths: &AppPaths) -> Result<PathBuf> {
    let binary = std::env::current_exe()?.canonicalize()?;
    let binary = quote_exec_path(&binary)?;
    let contents = format!(
        "[Desktop Entry]\nType=Application\nName=Omaspeak Setup\nComment=Configure Omaspeak runtimes, models, and voices\nExec=\"{}\" setup\nTerminal=true\nCategories=Settings;\nKeywords=voice;speech;tts;\n",
        binary
    );
    let path = launcher_path(paths);
    write_atomic(&path, contents.as_bytes())?;
    Ok(path)
}

/// Quote an executable path according to the Desktop Entry `Exec` grammar.
/// That grammar applies desktop-entry string escapes before command quoting,
/// so the four command-quoted characters require two leading backslashes in
/// the file. Percent signs are doubled to prevent field-code expansion.
fn quote_exec_path(path: &Path) -> Result<String> {
    let value = path
        .to_str()
        .context("Omaspeak executable path is not valid UTF-8")?;
    if value.is_empty() {
        bail!("Omaspeak executable path is empty");
    }
    if value.contains('=') {
        bail!("Omaspeak executable path contains '=' which desktop Exec entries prohibit");
    }
    if value.chars().any(char::is_control) {
        bail!("Omaspeak executable path contains a control character");
    }
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' | '`' | '$' => {
                escaped.push_str("\\\\");
                escaped.push(character);
            }
            '\\' => escaped.push_str("\\\\\\\\"),
            '%' => escaped.push_str("%%"),
            _ => escaped.push(character),
        }
    }
    Ok(escaped)
}

pub fn uninstall(paths: &AppPaths) -> Result<()> {
    let path = launcher_path(paths);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn status(paths: &AppPaths) -> Result<()> {
    let path = launcher_path(paths);
    if path.exists() {
        println!("installed: {}", path.display());
        Ok(())
    } else {
        bail!("Omaspeak setup launcher is not installed")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_menu.rs"]
mod tests;
