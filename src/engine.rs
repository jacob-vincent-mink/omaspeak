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
    /// Emit ordered mono PCM. Returning an error from the sink stops generation.
    /// Providers without incremental inference retain the complete-buffer path.
    fn generate_stream(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        sink(&self.generate(text, speed, voice)?)
    }
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
        Self::load_configured_with(config, paths, |config, paths, runtime| {
            if crate::cloud::is_cloud(&config.backend.kind) {
                return Ok(Box::new(crate::cloud::CloudBackend::create(config)?));
            }
            let backend: Box<dyn TtsBackend> = match config.backend.kind.as_str() {
                "audiocpp" => Box::new(crate::audio_cpp::AudioCppBackend::create(
                    config, paths, runtime,
                )?),
                "supertonic" => {
                    let locations = crate::runtime::inspect(&config.backend, &paths.config_file);
                    if runtime != Runtime::Openvino {
                        bail!(
                            "backend.kind=\"supertonic\" is the direct OpenVINO provider; select runtime=openvino"
                        )
                    }
                    let runtime = crate::runtime::resolve_openvino_runtime(
                        &config.backend,
                        &paths.config_file,
                        &locations,
                    )?;
                    Box::new(crate::supertonic::DirectOpenvinoBackend::create(
                        config, paths, runtime,
                    )?)
                }
                "paradee-openvino" => Box::new(crate::paradee::ParadeeOpenvinoBackend::create(
                    config, paths,
                )?),
                "kokoro-genai" => {
                    if runtime != Runtime::Openvino {
                        bail!("backend.kind=\"kokoro-genai\" requires runtime=openvino")
                    }
                    Box::new(crate::kokoro_genai::KokoroGenAiBackend::create(
                        config, paths,
                    )?)
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
        Self::load_configured_with(config, paths, |config, paths, runtime| {
            create_backend(config, paths, runtime)
        })
    }

    fn load_configured_with(
        config: &Config,
        paths: &AppPaths,
        mut create_backend: impl FnMut(&Config, &AppPaths, Runtime) -> Result<Box<dyn TtsBackend>>,
    ) -> Result<Self> {
        config.backend.validate_shape()?;
        let mut effective = config.clone();
        let mut fallback_used = false;

        let started = Instant::now();
        let backend = match create_backend(&effective, paths, effective.backend.runtime) {
            Ok(backend) => backend,
            Err(accelerator_error)
                if (effective.backend.runtime != Runtime::Default
                    || (matches!(
                        effective.backend.kind.as_str(),
                        "supertonic" | "kokoro-genai"
                    ) && !effective.backend.device.eq_ignore_ascii_case("cpu")))
                    && config.backend.fallback == Fallback::Cpu =>
            {
                eprintln!(
                    "omaspeak: warning: accelerated backend initialization failed: {accelerator_error:#}; falling back to cpu"
                );
                if effective.backend.kind == "audiocpp" {
                    effective.backend.runtime = Runtime::Default;
                } else if matches!(
                    effective.backend.kind.as_str(),
                    "supertonic" | "kokoro-genai"
                ) && effective.backend.runtime == Runtime::Openvino
                {
                    effective.backend.runtime = Runtime::Openvino;
                } else {
                    return Err(accelerator_error);
                }
                effective.backend.device = "cpu".into();
                effective.backend.device_id = 0;
                fallback_used = true;
                create_backend(&effective, paths, effective.backend.runtime).with_context(|| {
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
            model_name: if crate::cloud::is_cloud(&config.backend.kind) {
                crate::cloud::model_name(config)
            } else {
                config.model.name.clone()
            },
            sample_rate,
            load_time,
            effective_runtime: effective.backend.runtime,
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
        self.synthesize_to(text, speed, voice, output, false, &mut |_| Ok(()))
    }

    /// Write a complete WAV while delivering each generated chunk to playback.
    /// The caller must stage the destination and publish it only after success.
    pub fn synthesize_stream(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        output: &Path,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<Synthesis> {
        self.synthesize_to(text, speed, voice, output, true, sink)
    }

    fn synthesize_to(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        output: &Path,
        streaming: bool,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
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
        // Offline exports must not truncate an existing file if generation
        // fails. Streaming callers stage their destination before calling us.
        let buffered = if streaming {
            None
        } else {
            let audio = self.backend.generate(text, speed, voice)?;
            if audio.is_empty() || !audio.iter().all(|sample| sample.is_finite()) {
                bail!("synthesis returned empty or non-finite audio");
            }
            ensure_wav_length(audio.len())?;
            Some(audio)
        };
        let mut writer = hound::WavWriter::create(
            output,
            hound::WavSpec {
                channels: 1,
                sample_rate: self
                    .sample_rate
                    .try_into()
                    .context("invalid WAV sample rate")?,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .with_context(|| format!("create WAV {}", output.display()))?;
        let mut samples = 0usize;
        let mut emit = |chunk: &[f32]| -> Result<()> {
            if chunk.is_empty() || !chunk.iter().all(|sample| sample.is_finite()) {
                bail!("synthesis returned empty or non-finite audio");
            }
            samples = samples
                .checked_add(chunk.len())
                .context("audio length overflow")?;
            // Keep the WAV within RIFF's 32-bit size limit.
            ensure_wav_length(samples)?;
            for sample in chunk {
                writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
            }
            // Small writes bound playback buffering; the pipe applies backpressure.
            for block in chunk.chunks(4096) {
                sink(block)?;
            }
            Ok(())
        };
        if streaming {
            self.backend
                .generate_stream(text, speed, voice, &mut emit)?;
        } else {
            emit(
                buffered
                    .as_ref()
                    .expect("offline synthesis has buffered audio"),
            )?;
        }
        if samples == 0 {
            bail!("synthesis returned empty audio");
        }
        writer.finalize()?;
        Ok(Synthesis {
            output: output.to_owned(),
            sample_rate: self.sample_rate,
            samples,
            synthesis_time: started.elapsed(),
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

fn ensure_wav_length(samples: usize) -> Result<()> {
    anyhow::ensure!(
        samples <= (u32::MAX as usize - 36) / 2,
        "audio exceeds WAV size limit"
    );
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
