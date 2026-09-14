use anyhow::{Context, Result, bail};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig};

use crate::engine::TtsBackend;

pub(crate) struct SherpaOnnxBackend {
    tts: OfflineTts,
    sample_rate: i32,
}

impl SherpaOnnxBackend {
    pub(crate) fn create(config: &OfflineTtsConfig) -> Result<Self> {
        let tts =
            OfflineTts::create(config).context("sherpa-onnx could not create the TTS engine")?;
        let sample_rate = tts.sample_rate();
        if sample_rate <= 0 {
            bail!("model reported invalid sample rate {sample_rate}");
        }
        Ok(Self { tts, sample_rate })
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
        Ok(audio.samples().to_vec())
    }
}
