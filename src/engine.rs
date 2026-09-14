use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use anyhow::{Context, Result, bail};
use sherpa_onnx::{
    OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsSupertonicModelConfig,
    OfflineTtsVitsModelConfig,
};

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

#[derive(Clone, Debug, Default)]
pub(crate) struct SherpaGenerationSettings {
    pub language: Option<String>,
    pub num_steps: Option<i32>,
}

impl Engine {
    pub fn load(config: &Config, paths: &AppPaths) -> Result<Self> {
        Self::load_with(config, paths, |config, paths, runtime| {
            let backend: Box<dyn TtsBackend> = match config.backend.kind.as_str() {
                "sherpa-onnx" => {
                    let native_config = build_sherpa_config(config, paths, runtime)?;
                    let generation = sherpa_generation_settings(config)?;
                    Box::new(crate::native::sherpa::SherpaOnnxBackend::create(
                        &native_config,
                        generation,
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
        create_backend: impl FnMut(&Config, &AppPaths, Runtime) -> Result<Box<dyn TtsBackend>>,
    ) -> Result<Self> {
        Self::load_with_capabilities(config, paths, compiled_capabilities(), create_backend)
    }

    fn load_with_capabilities(
        config: &Config,
        paths: &AppPaths,
        capabilities: &[&str],
        mut create_backend: impl FnMut(&Config, &AppPaths, Runtime) -> Result<Box<dyn TtsBackend>>,
    ) -> Result<Self> {
        config.backend.validate_shape()?;
        let (mut effective_runtime, mut fallback_used) =
            match config.backend.validate_capabilities(capabilities) {
                Ok(()) => (config.backend.runtime, false),
                Err(error) if config.backend.fallback == Fallback::Cpu => {
                    eprintln!("omaspeak: warning: {error}; falling back to cpu");
                    (Runtime::Default, true)
                }
                Err(error) => return Err(error.into()),
            };

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
    sherpa_generation_settings(config)?;
    let directory = config.model_directory(paths);
    let required_file = |name: &str| -> Result<String> {
        if name.trim().is_empty() {
            bail!("required model asset path is not configured");
        }
        let path = directory.join(name);
        if !path.is_file() {
            bail!("required model asset is missing: {}", path.display());
        }
        Ok(path.to_string_lossy().into_owned())
    };
    let required_directory = |name: &str| -> Result<String> {
        if name.trim().is_empty() {
            bail!("required model asset directory is not configured");
        }
        let path = directory.join(name);
        if !path.is_dir() {
            bail!(
                "required model asset directory is missing: {}",
                path.display()
            );
        }
        Ok(path.to_string_lossy().into_owned())
    };
    let provider = match runtime {
        Runtime::Default => "cpu".to_owned(),
        Runtime::Cuda => cuda_provider(config, paths)?,
        Runtime::Openvino => openvino_provider(config, paths)?,
    };
    let mut model = OfflineTtsModelConfig {
        num_threads: config.backend.threads.into(),
        provider: Some(provider),
        ..Default::default()
    };
    match config.model.family.as_str() {
        "piper" | "vits" => {
            model.vits = OfflineTtsVitsModelConfig {
                model: Some(required_file(&config.model.model_file)?),
                tokens: Some(required_file(&config.model.tokens_file)?),
                data_dir: Some(required_directory(&config.model.data_directory)?),
                noise_scale: config.model.noise_scale,
                noise_scale_w: config.model.noise_scale_w,
                length_scale: config.model.length_scale,
                ..Default::default()
            };
        }
        "supertonic" => {
            model.supertonic = OfflineTtsSupertonicModelConfig {
                duration_predictor: Some(required_file(&config.model.duration_predictor)?),
                text_encoder: Some(required_file(&config.model.text_encoder)?),
                vector_estimator: Some(required_file(&config.model.vector_estimator)?),
                vocoder: Some(required_file(&config.model.vocoder)?),
                tts_json: Some(required_file(&config.model.tts_json)?),
                unicode_indexer: Some(required_file(&config.model.unicode_indexer)?),
                voice_style: Some(required_file(&config.model.voice_style)?),
            };
        }
        _ => unreachable!("model family was validated above"),
    }
    Ok(OfflineTtsConfig {
        model,
        ..Default::default()
    })
}

fn sherpa_generation_settings(config: &Config) -> Result<SherpaGenerationSettings> {
    const SUPERTONIC_LANGUAGES: &[&str] = &[
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr",
        "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
        "vi",
    ];
    match config.model.family.as_str() {
        "piper" | "vits" => Ok(SherpaGenerationSettings::default()),
        "supertonic" => {
            if !SUPERTONIC_LANGUAGES.contains(&config.model.language.as_str()) {
                bail!(
                    "Supertonic language {:?} is unsupported; use one of {}",
                    config.model.language,
                    SUPERTONIC_LANGUAGES.join(", ")
                );
            }
            if config.model.steps <= 0 {
                bail!("Supertonic generation steps must be greater than zero");
            }
            Ok(SherpaGenerationSettings {
                language: Some(config.model.language.clone()),
                num_steps: Some(config.model.steps),
            })
        }
        family => bail!("sherpa-onnx TTS model family {family} is not implemented"),
    }
}

fn openvino_provider(config: &Config, paths: &AppPaths) -> Result<String> {
    config.backend.validate_shape()?;
    let supplied = config.backend.provider_config.trim();
    let provider_config = if supplied.is_empty() {
        generate_openvino_provider_config(config, paths)?
    } else {
        resolve_provider_config(supplied, paths, "OpenVINO")?
    };
    Ok(format!("openvino:{}", provider_config.display()))
}

fn cuda_provider(config: &Config, paths: &AppPaths) -> Result<String> {
    config.backend.validate_shape()?;
    let supplied = config.backend.provider_config.trim();
    let provider_config = if supplied.is_empty() {
        generate_cuda_provider_config(config, paths)?
    } else {
        resolve_provider_config(supplied, paths, "CUDA")?
    };
    Ok(format!("cuda:{}", provider_config.display()))
}

fn resolve_provider_config(supplied: &str, paths: &AppPaths, runtime: &str) -> Result<PathBuf> {
    let supplied = PathBuf::from(supplied);
    let supplied = if supplied.is_absolute() {
        supplied
    } else {
        paths
            .config_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(supplied)
    };
    if !supplied.is_file() {
        bail!(
            "{runtime} provider config is not an existing file: {}",
            supplied.display()
        );
    }
    supplied
        .canonicalize()
        .with_context(|| format!("resolve {runtime} provider config {}", supplied.display()))
}

fn generate_cuda_provider_config(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    let directory = paths
        .state_dir
        .join("cache")
        .join("cuda")
        .join(format!("device-{}", config.backend.device_id));
    create_private_directory(&directory)?;
    let directory = directory
        .canonicalize()
        .with_context(|| format!("resolve CUDA cache directory {}", directory.display()))?;

    let mut options = config.backend.options.clone();
    if options.contains_key("device_id") {
        bail!("backend.options.device_id is managed by backend.device_id");
    }
    options
        .entry("cudnn_conv_algo_search".to_owned())
        .or_insert_with(|| "HEURISTIC".to_owned());
    options.insert("device_id".to_owned(), config.backend.device_id.to_string());
    let mut contents = String::new();
    for (key, value) in options {
        contents.push_str(&key);
        contents.push('=');
        contents.push_str(&value);
        contents.push('\n');
    }
    let provider_config = directory.join("provider.config");
    write_private_atomic(&provider_config, contents.as_bytes())?;
    Ok(provider_config)
}

fn generate_openvino_provider_config(config: &Config, paths: &AppPaths) -> Result<PathBuf> {
    let device = config.backend.canonical_device()?.to_ascii_uppercase();
    let slug: String = device
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let directory = paths.state_dir.join("cache").join("openvino").join(slug);
    create_private_directory(&directory)?;
    let directory = directory
        .canonicalize()
        .with_context(|| format!("resolve OpenVINO cache directory {}", directory.display()))?;
    let compiled_cache = directory.join("compiled");
    create_private_directory(&compiled_cache)?;

    let mut options = BTreeMap::from([
        ("cache_dir".to_owned(), compiled_cache.display().to_string()),
        ("device_type".to_owned(), device.clone()),
    ]);
    if device == "NPU" {
        options.insert("enable_qdq_optimizer".to_owned(), "True".to_owned());
        options.insert("disable_dynamic_shapes".to_owned(), "True".to_owned());
        if config.model.family == "supertonic" {
            let components = crate::catalog::model(&config.model.name)
                .filter(|spec| {
                    spec.npu_capable
                        && spec.family == config.model.family
                        && spec.duration_predictor == config.model.duration_predictor
                        && spec.text_encoder == config.model.text_encoder
                        && spec.vector_estimator == config.model.vector_estimator
                        && spec.vocoder == config.model.vocoder
                        && config.model_directory(paths)
                            == crate::setup::model::model_directory(paths, spec)
                })
                .map_or("duration_predictor,text_encoder,vocoder", |_| "all");
            options.insert(
                "SherpaOnnx.SupertonicComponents".to_owned(),
                components.to_owned(),
            );
        }
    } else if config.model.family == "supertonic" && device == "GPU" {
        options.insert("disable_dynamic_shapes".to_owned(), "True".to_owned());
        options.insert("enable_qdq_optimizer".to_owned(), "False".to_owned());
        options.insert("precision".to_owned(), "FP32".to_owned());
        options.insert(
            "SherpaOnnx.SupertonicComponents".to_owned(),
            "vector_estimator".to_owned(),
        );
    } else if config.model.family == "supertonic" && device == "CPU" {
        options.insert("disable_dynamic_shapes".to_owned(), "True".to_owned());
    }
    for (key, value) in &config.backend.options {
        match key.as_str() {
            "device_type" if value != &device => {
                bail!("backend.options.device_type must match canonical device {device:?}")
            }
            "cache_dir" => {
                bail!("backend.options.cache_dir is managed by Omaspeak for each OpenVINO device")
            }
            _ => {
                options.insert(key.clone(), value.clone());
            }
        }
    }

    let mut contents = String::new();
    for (key, value) in options {
        contents.push_str(&key);
        contents.push('=');
        contents.push_str(&value);
        contents.push('\n');
    }
    let provider_config = directory.join("provider.config");
    write_private_atomic(&provider_config, contents.as_bytes())?;
    Ok(provider_config)
}

fn create_private_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("create private runtime directory {}", path.display()))?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("protect runtime directory {}", path.display()))?;
    Ok(())
}

fn write_private_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .context("provider config has no parent directory")?;
    let unique = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".provider.config.{}.{}.tmp",
        std::process::id(),
        unique
    ));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temporary)
            .with_context(|| format!("create temporary provider config {}", temporary.display()))?;
        file.write_all(contents)
            .with_context(|| format!("write temporary provider config {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("sync temporary provider config {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("install provider config {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
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
