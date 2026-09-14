use anyhow::{Context, Result, bail};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig};

use crate::engine::{SherpaGenerationSettings, TtsBackend};

pub(crate) struct SherpaOnnxBackend {
    tts: OfflineTts,
    sample_rate: i32,
    generation: SherpaGenerationSettings,
}

impl SherpaOnnxBackend {
    pub(crate) fn create(
        config: &OfflineTtsConfig,
        generation: SherpaGenerationSettings,
    ) -> Result<Self> {
        let tts =
            OfflineTts::create(config).context("sherpa-onnx could not create the TTS engine")?;
        let sample_rate = tts.sample_rate();
        if sample_rate <= 0 {
            bail!("model reported invalid sample rate {sample_rate}");
        }
        Ok(Self {
            tts,
            sample_rate,
            generation,
        })
    }
}

impl TtsBackend for SherpaOnnxBackend {
    fn kind(&self) -> &'static str {
        "sherpa-onnx"
    }

    fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    fn num_voices(&self) -> i32 {
        self.tts.num_speakers()
    }

    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut extra = std::collections::HashMap::new();
        if let Some(language) = &self.generation.language {
            extra.insert("lang".to_owned(), serde_json::json!(language));
        }
        let audio = self
            .tts
            .generate_with_config(
                text,
                &GenerationConfig {
                    speed,
                    sid: voice,
                    num_steps: self.generation.num_steps.unwrap_or(5),
                    extra: (!extra.is_empty()).then_some(extra),
                    ..Default::default()
                },
                None::<fn(&[f32], f32) -> bool>,
            )
            .context("sherpa-onnx synthesis failed")?;
        Ok(audio.samples().to_vec())
    }
}
