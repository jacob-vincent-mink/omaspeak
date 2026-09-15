use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::catalog::ModelSpec;
use crate::config::Config;
use crate::paths::AppPaths;

pub use crate::catalog::SUPERTONIC_VOICE_NAMES as SUPERTONIC_PRESET_NAMES;

/// One selectable voice exposed by the active model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Voice {
    pub id: i32,
    pub name: String,
}

/// Inspect the installed active model and return its actual speaker inventory.
pub fn installed(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
    match config.model.family.as_str() {
        "supertonic" if config.backend.kind == "audiocpp" => Ok(supertonic_presets()),
        "supertonic" => supertonic_voices(config, paths),
        family => bail!("cannot enumerate voices for model family {family:?}"),
    }
}

/// Return installed metadata when present, otherwise the pinned catalog inventory.
pub fn available(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
    if config.model.family == "supertonic" && config.backend.kind == "audiocpp" {
        return Ok(supertonic_presets());
    }
    let directory = config.model_directory(paths);
    if directory.exists() {
        return installed(config, paths);
    }
    let spec = crate::catalog::model(&config.model.name).with_context(|| {
        format!(
            "model directory is missing and model {:?} has no catalog voice metadata",
            config.model.name
        )
    })?;
    Ok(from_catalog(spec))
}

pub fn supertonic_presets() -> Vec<Voice> {
    SUPERTONIC_PRESET_NAMES
        .iter()
        .enumerate()
        .map(|(id, name)| Voice {
            id: id as i32,
            name: (*name).into(),
        })
        .collect()
}

pub fn from_catalog(spec: &ModelSpec) -> Vec<Voice> {
    spec.voices
        .iter()
        .map(|voice| Voice {
            id: voice.id,
            name: voice.name.to_owned(),
        })
        .collect()
}

pub fn validate_selected(config: &Config, voices: &[Voice]) -> Result<()> {
    if voices.iter().any(|voice| voice.id == config.model.voice) {
        Ok(())
    } else {
        let valid = voices
            .iter()
            .map(|voice| voice.id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "voice {} is unavailable for model {}; choose one of: {}",
            config.model.voice,
            config.model.name,
            valid
        )
    }
}

fn supertonic_voices(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
    if config.model.voice_style.trim().is_empty() {
        bail!("Supertonic voice style file is not configured");
    }
    let directory = config
        .model_directory(paths)
        .join(&config.model.voice_style);
    if !directory.is_dir() {
        bail!(
            "Supertonic voice style directory is missing: {}",
            directory.display()
        );
    }
    for name in SUPERTONIC_PRESET_NAMES {
        let path = directory.join(format!("{name}.json"));
        if !path.is_file() {
            bail!("Supertonic voice style is missing: {}", path.display());
        }
    }
    Ok(supertonic_presets())
}

#[cfg(test)]
#[path = "../tests/unit/voices.rs"]
mod tests;
