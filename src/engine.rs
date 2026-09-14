use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sherpa_onnx::{OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig};

use crate::backend::{Fallback, Runtime, compiled_capabilities};
use crate::config::Config;
use crate::paths::AppPaths;

pub trait TtsBackend {
    fn kind(&self) -> &'static str;
    fn sample_rate(&self) -> i32;
    fn num_voices(&self) -> i32;
    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>>;
}

pub struct Engine {
    backend: Box<dyn TtsBackend>,
    pub backend_kind: &'static str,
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
        Self::load_with(config, paths, |config, paths, runtime| {
            let backend: Box<dyn TtsBackend> = match config.backend.kind.as_str() {
                "sherpa-onnx" => {
                    let native_config = build_sherpa_config(config, paths, runtime)?;
                    Box::new(crate::native::sherpa::SherpaOnnxBackend::create(
                        &native_config,
                    )?)
                }
                kind => bail!(
                    "TTS backend {kind} is not available in this build; run `omaspeak setup runtime`"
                ),
            };
            Ok(backend)
        })
    }

    /// Construct an engine with a caller-provided backend factory.
    ///
    /// This is also the extension point for backends that live outside this crate.
    pub fn load_with(
        config: &Config,
        paths: &AppPaths,
        create_backend: impl FnOnce(&Config, &AppPaths, Runtime) -> Result<Box<dyn TtsBackend>>,
    ) -> Result<Self> {
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

        let started = Instant::now();
        let backend = create_backend(config, paths, effective_runtime)?;
        let load_time = started.elapsed();
        let sample_rate = backend.sample_rate();
        let backend_kind = backend.kind();

        Ok(Self {
            backend,
            backend_kind,
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
        if voice < 0 || voice >= self.backend.num_voices().max(1) {
            bail!("voice {voice} is outside the model speaker range");
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }

        let started = Instant::now();
        let samples = self.backend.generate(text, speed, voice)?;
        let synthesis_time = started.elapsed();
        if samples.is_empty() || !samples.iter().all(|sample| sample.is_finite()) {
            bail!("synthesis returned empty or non-finite audio");
        }
        save_wav(output, self.sample_rate, &samples)?;

        Ok(Synthesis {
            output: output.to_owned(),
            sample_rate: self.sample_rate,
            samples: samples.len(),
            synthesis_time,
        })
    }
}

fn build_sherpa_config(
    config: &Config,
    paths: &AppPaths,
    runtime: Runtime,
) -> Result<OfflineTtsConfig> {
    if config.model.family != "piper" && config.model.family != "vits" {
        bail!(
            "sherpa-onnx TTS model family {} is not implemented",
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
    let provider = match runtime {
        Runtime::Default => "cpu".to_owned(),
        Runtime::Cuda => "cuda".to_owned(),
        Runtime::Openvino => {
            bail!("OpenVINO provider file generation is unavailable in this build")
        }
    };
    Ok(OfflineTtsConfig {
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
    })
}

fn save_wav(path: &Path, sample_rate: i32, samples: &[f32]) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate.try_into().context("invalid WAV sample rate")?,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .with_context(|| format!("create WAV {}", path.display()))?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
