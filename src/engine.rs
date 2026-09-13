use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};

use crate::backend::{Fallback, Runtime, compiled_capabilities};
use crate::config::Config;
use crate::paths::AppPaths;

pub struct Engine {
    tts: OfflineTts,
    pub model_name: String,
    pub sample_rate: i32,
    pub load_time: Duration,
    pub effective_runtime: Runtime,
    pub fallback_used: bool,
}

pub struct Synthesis {
    pub output: PathBuf,
    pub sample_rate: i32,
    pub samples: usize,
    pub synthesis_time: Duration,
}

impl Engine {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        config.backend.validate_shape()?;
        let (effective_runtime, fallback_used) = match config
            .backend
            .validate_capabilities(compiled_capabilities())
        {
            Ok(()) => (config.backend.runtime, false),
            Err(error) if config.backend.fallback == Fallback::Cpu => {
                eprintln!("omaspeak: warning: {error}; falling back to cpu");
                (Runtime::Default, true)
            }
            Err(error) => return Err(error.into()),
        };
        if config.model.family != "piper" && config.model.family != "vits" {
            bail!(
                "model family {} is not implemented yet",
                config.model.family
            );
        }

        let directory = config.model_directory(paths);
        let required = |name: &str| -> Result<String> {
            let path = directory.join(name);
            if !path.exists() {
                bail!("required model asset is missing: {}", path.display());
            }
            Ok(path.to_string_lossy().into_owned())
        };
        let provider = match effective_runtime {
            Runtime::Default => "cpu".to_owned(),
            Runtime::Cuda => "cuda".to_owned(),
            Runtime::Openvino => {
                bail!("OpenVINO provider file generation is unavailable in the CPU build")
            }
        };

        let sherpa_config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(required(&config.model.model_file)?),
                    tokens: Some(required(&config.model.tokens_file)?),
                    data_dir: Some(required(&config.model.data_directory)?),
                    noise_scale: config.model.noise_scale,
                    noise_scale_w: config.model.noise_scale_w,
                    length_scale: config.model.length_scale,
                    ..Default::default()
                },
                num_threads: config.backend.threads.into(),
                provider: Some(provider),
                ..Default::default()
            },
            ..Default::default()
        };

        let started = Instant::now();
        let tts = OfflineTts::create(&sherpa_config)
            .context("sherpa-onnx could not create the TTS engine")?;
        let load_time = started.elapsed();
        let sample_rate = tts.sample_rate();
        if sample_rate <= 0 {
            bail!("model reported invalid sample rate {sample_rate}");
        }

        Ok(Self {
            tts,
            model_name: config.model.name.clone(),
            sample_rate,
            load_time,
            effective_runtime,
            fallback_used,
        })
    }

    pub fn synthesize(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        output: &Path,
    ) -> Result<Synthesis> {
        if text.trim().is_empty() {
            bail!("text must not be empty");
        }
        if !(0.25..=4.0).contains(&speed) || !speed.is_finite() {
            bail!("speed must be finite and between 0.25 and 4.0");
        }
        if voice < 0 || voice >= self.tts.num_speakers().max(1) {
            bail!("voice {voice} is outside the model speaker range");
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }

        let started = Instant::now();
        let audio = self
            .tts
            .generate_with_config(
                text,
                &GenerationConfig {
                    speed,
                    sid: voice,
                    ..Default::default()
                },
                None::<fn(&[f32], f32) -> bool>,
            )
            .context("sherpa-onnx synthesis failed")?;
        let synthesis_time = started.elapsed();
        let samples = audio.samples().len();
        if samples == 0 || !audio.samples().iter().all(|sample| sample.is_finite()) {
            bail!("synthesis returned empty or non-finite audio");
        }
        if !audio.save(output.to_string_lossy().as_ref()) {
            bail!("failed to save WAV to {}", output.display());
        }

        Ok(Synthesis {
            output: output.to_owned(),
            sample_rate: self.sample_rate,
            samples,
            synthesis_time,
        })
    }
}
