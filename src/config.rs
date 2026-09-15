use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::backend::BackendConfig;
use crate::paths::AppPaths;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub backend: BackendConfig,
    pub model: ModelConfig,
    pub audio: AudioConfig,
    pub daemon: DaemonConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let input =
            fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
        let mut document: toml::Value =
            toml::from_str(&input).with_context(|| format!("parse config {}", path.display()))?;
        migrate_legacy_config(&mut document)
            .with_context(|| format!("upgrade config {}", path.display()))?;
        document
            .try_into()
            .with_context(|| format!("parse config {}", path.display()))
    }

    pub fn model_directory(&self, paths: &AppPaths) -> PathBuf {
        if self.model.directory.trim().is_empty() {
            paths.data_dir.join("models").join(&self.model.name)
        } else {
            PathBuf::from(&self.model.directory)
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create config directory {}", parent.display()))?;
        }
        let temporary = path.with_extension("toml.tmp");
        fs::write(&temporary, toml::to_string_pretty(self)?)
            .with_context(|| format!("write temporary config {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("install config {}", path.display()))?;
        Ok(())
    }
}

/// Accept only fields emitted by earlier Omaspeak builds. The final typed
/// deserialization still uses `deny_unknown_fields`, so misspellings and
/// unrelated keys remain errors instead of disappearing during an upgrade.
fn migrate_legacy_config(document: &mut toml::Value) -> Result<()> {
    if let Some(backend) = document
        .get_mut("backend")
        .and_then(toml::Value::as_table_mut)
    {
        if let Some(provider_config) = backend.get("provider_config") {
            let provider_config = provider_config
                .as_str()
                .context("legacy backend.provider_config must be a string")?;
            if !provider_config.is_empty() {
                bail!(
                    "backend.provider_config is no longer supported; remove it, then configure backend library paths and backend.options with `omaspeak setup runtime`"
                );
            }
            backend.remove("provider_config");
        }
        if backend.get("kind").and_then(toml::Value::as_str) == Some("sherpa-onnx") {
            backend.insert("kind".into(), toml::Value::String("supertonic".into()));
        }
    }

    if let Some(model) = document
        .get_mut("model")
        .and_then(toml::Value::as_table_mut)
    {
        for field in ["model_file", "tokens_file", "data_directory"] {
            if let Some(value) = model.get(field) {
                value
                    .as_str()
                    .with_context(|| format!("legacy model.{field} must be a string"))?;
            }
            model.remove(field);
        }
        for field in ["noise_scale", "noise_scale_w", "length_scale"] {
            if let Some(value) = model.get(field) {
                value
                    .as_float()
                    .with_context(|| format!("legacy model.{field} must be a number"))?;
            }
            model.remove(field);
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub family: String,
    pub name: String,
    pub directory: String,
    pub duration_predictor: String,
    pub text_encoder: String,
    pub vector_estimator: String,
    pub vocoder: String,
    pub tts_json: String,
    pub unicode_indexer: String,
    pub voice_style: String,
    pub language: String,
    pub steps: i32,
    pub voice: i32,
    pub options: BTreeMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            family: "supertonic".into(),
            name: "supertonic-3-int8".into(),
            directory: String::new(),
            duration_predictor: "duration_predictor.int8.onnx".into(),
            text_encoder: "text_encoder.int8.onnx".into(),
            vector_estimator: "vector_estimator.int8.onnx".into(),
            vocoder: "vocoder.int8.onnx".into(),
            tts_json: "tts.json".into(),
            unicode_indexer: "unicode_indexer.bin".into(),
            voice_style: "voice.bin".into(),
            language: "en".into(),
            steps: 5,
            voice: 0,
            options: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    pub device: String,
    pub volume: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: "default".into(),
            volume: 1.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    pub queue_capacity: usize,
    pub max_text_bytes: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 8,
            max_text_bytes: 65_536,
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/config.rs"]
mod tests;
