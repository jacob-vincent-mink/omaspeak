use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
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
        toml::from_str(&input).with_context(|| format!("parse config {}", path.display()))
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub family: String,
    pub name: String,
    pub directory: String,
    pub model_file: String,
    pub tokens_file: String,
    pub data_directory: String,
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
    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub length_scale: f32,
    pub options: BTreeMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            family: "piper".into(),
            name: "en_US-lessac-medium".into(),
            directory: String::new(),
            model_file: "en_US-lessac-medium.onnx".into(),
            tokens_file: "tokens.txt".into(),
            data_directory: "espeak-ng-data".into(),
            duration_predictor: String::new(),
            text_encoder: String::new(),
            vector_estimator: String::new(),
            vocoder: String::new(),
            tts_json: String::new(),
            unicode_indexer: String::new(),
            voice_style: String::new(),
            language: "en".into(),
            steps: 5,
            voice: 0,
            noise_scale: 0.667,
            noise_scale_w: 0.8,
            length_scale: 1.0,
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
