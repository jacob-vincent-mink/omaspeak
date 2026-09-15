use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::backend::{Fallback, Runtime};
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
                "audiocpp" => Box::new(crate::audio_cpp::AudioCppBackend::create(
                    config, paths, runtime,
                )?),
                "supertonic" => {
                    let locations = crate::runtime::inspect(&config.backend, &paths.config_file);
                    if runtime == Runtime::Openvino {
                        let runtime = crate::runtime::resolve_openvino_runtime(
                            &config.backend,
                            &paths.config_file,
                            &locations,
                        )?;
                        Box::new(crate::supertonic::DirectOpenvinoBackend::create(
                            config, paths, runtime,
                        )?)
                    } else {
                        if config.model.family != "supertonic" {
                            bail!(
                                "the direct ONNX Runtime backend supports Supertonic models; configured family is {:?}",
                                config.model.family
                            );
                        }
                        let libraries = crate::runtime::resolve_onnx_runtime(
                            &config.backend,
                            &paths.config_file,
                            &locations,
                            runtime,
                        )?;
                        Box::new(crate::supertonic::DirectOrtBackend::create(
                            config, paths, libraries, runtime,
                        )?)
                    }
                }
                kind => bail!("unsupported TTS backend {kind:?}"),
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
        mut create_backend: impl FnMut(&Config, &AppPaths, Runtime) -> Result<Box<dyn TtsBackend>>,
    ) -> Result<Self> {
        config.backend.validate_shape()?;
        let mut effective_runtime = config.backend.runtime;
        let mut fallback_used = false;

        let started = Instant::now();
        let backend = match create_backend(config, paths, effective_runtime) {
            Ok(backend) => backend,
            Err(accelerator_error)
                if effective_runtime != Runtime::Default
                    && config.backend.fallback == Fallback::Cpu =>
            {
                eprintln!(
                    "omaspeak: warning: accelerated backend initialization failed: {accelerator_error:#}; falling back to cpu"
                );
                effective_runtime = Runtime::Default;
                fallback_used = true;
                create_backend(config, paths, Runtime::Default).with_context(|| {
                    format!(
                        "accelerated backend initialization failed ({accelerator_error:#}); CPU fallback also failed"
                    )
                })?
            }
            Err(error) => return Err(error),
        };
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
        validate_voice(voice, self.backend.num_voices(), &self.model_name)?;
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

fn validate_voice(voice: i32, voices: i32, model: &str) -> Result<()> {
    if voices <= 0 {
        bail!("model {model} reported no voices");
    }
    if voice < 0 || voice >= voices {
        bail!(
            "voice {voice} is outside the speaker range 0..{} for model {model}",
            voices - 1
        );
    }
    Ok(())
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
