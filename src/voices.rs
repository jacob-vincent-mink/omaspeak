use std::fs::File;
use std::io::Read;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::catalog::ModelSpec;
use crate::config::Config;
use crate::paths::AppPaths;

/// One selectable voice exposed by the active model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Voice {
    pub id: i32,
    pub name: String,
}

/// Inspect the installed active model and return its actual speaker inventory.
pub fn installed(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
    match config.model.family.as_str() {
        "supertonic" => supertonic_voices(config, paths),
        family => bail!("cannot enumerate voices for model family {family:?}"),
    }
}

/// Return installed metadata when present, otherwise the pinned catalog inventory.
pub fn available(config: &Config, paths: &AppPaths) -> Result<Vec<Voice>> {
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
    let path = config
        .model_directory(paths)
        .join(&config.model.voice_style);
    let mut header = [0_u8; 48];
    File::open(&path)
        .with_context(|| format!("read Supertonic voice styles {}", path.display()))?
        .read_exact(&mut header)
        .with_context(|| format!("read Supertonic voice header {}", path.display()))?;
    let mut dimensions = [0_i64; 6];
    for (index, bytes) in header.as_chunks::<8>().0.iter().enumerate() {
        dimensions[index] = i64::from_le_bytes(*bytes);
    }
    if dimensions.iter().any(|dimension| *dimension <= 0) || dimensions[0] != dimensions[3] {
        bail!(
            "{} has invalid Supertonic voice dimensions {dimensions:?}",
            path.display()
        );
    }
    let count = i32::try_from(dimensions[0]).context("Supertonic voice count exceeds i32")?;
    Ok((0..count)
        .map(|id| Voice {
            id,
            name: fallback_name(config, id),
        })
        .collect())
}

fn fallback_name(config: &Config, id: i32) -> String {
    crate::catalog::model(&config.model.name)
        .and_then(|spec| spec.voices.iter().find(|voice| voice.id == id))
        .map(|voice| voice.name.to_owned())
        .unwrap_or_else(|| format!("Voice {}", id + 1))
}

#[cfg(test)]
#[path = "../tests/unit/voices.rs"]
mod tests;
