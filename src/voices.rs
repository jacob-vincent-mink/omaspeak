use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::catalog::ModelSpec;
use crate::config::Config;
use crate::paths::AppPaths;

pub use crate::catalog::SUPERTONIC_VOICE_NAMES as SUPERTONIC_PRESET_NAMES;

/// Stable preset name, or a legacy numeric ID scoped to the selected model.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum VoiceSelection {
    Legacy(i32),
    Name(String),
}

impl From<i32> for VoiceSelection {
    fn from(id: i32) -> Self {
        Self::Legacy(id)
    }
}

impl std::str::FromStr for VoiceSelection {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        anyhow::ensure!(!value.is_empty(), "voice must not be empty");
        Ok(match value.parse::<i32>() {
            Ok(id) => Self::Legacy(id),
            Err(_) => Self::Name(value.to_owned()),
        })
    }
}

impl std::fmt::Display for VoiceSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy(id) => id.fmt(f),
            Self::Name(name) => name.fmt(f),
        }
    }
}

impl VoiceSelection {
    pub fn resolve(&self, voices: &[Voice]) -> Result<i32> {
        voices
            .iter()
            .find(|voice| self.matches(voice))
            .map(|voice| voice.id)
            .with_context(|| {
                format!(
                    "voice {self} is unavailable; choose {}",
                    voices
                        .iter()
                        .map(|v| format!("{} ({})", v.name, v.id))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    pub fn matches(&self, voice: &Voice) -> bool {
        match self {
            Self::Legacy(id) => *id == voice.id,
            Self::Name(name) => name.eq_ignore_ascii_case(&voice.name),
        }
    }
}

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
        "kokoro" => {
            // The catalog is the trusted inventory. GGUF voices are embedded;
            // OpenVINO GenAI voices are separate pinned embedding files.
            let directory = config.model_directory(paths);
            if !directory.join(&config.model.file).is_file() {
                bail!(
                    "Kokoro model file is not installed: {}",
                    directory.join(&config.model.file).display()
                );
            }
            crate::catalog::model(&config.model.name)
                .map(|spec| {
                    if config.backend.kind == "kokoro-genai" {
                        for voice in spec.voices {
                            let file = directory.join(format!("voices/{}.bin", voice.name));
                            if !file.is_file() {
                                bail!(
                                    "Kokoro voice embedding is not installed: {}",
                                    file.display()
                                );
                            }
                        }
                    }
                    Ok(from_catalog(spec))
                })
                .with_context(|| {
                    format!(
                        "Kokoro model {:?} has no catalog voice metadata",
                        config.model.name
                    )
                })?
        }
        family => bail!("cannot enumerate voices for model family {family:?}"),
    }
}

/// Return installed metadata when present, otherwise the pinned catalog inventory.
pub fn available(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
    if config.model.family == "supertonic" && config.backend.kind == "audiocpp" {
        return Ok(supertonic_presets());
    }
    if config.model.family == "kokoro" {
        let spec = crate::catalog::model(&config.model.name).with_context(|| {
            format!(
                "Kokoro model {:?} has no catalog voice metadata",
                config.model.name
            )
        })?;
        return Ok(from_catalog(spec));
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
    if voices.iter().any(|voice| config.model.voice.matches(voice)) {
        Ok(())
    } else {
        let valid = voices
            .iter()
            .map(|voice| voice.name.to_string())
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
