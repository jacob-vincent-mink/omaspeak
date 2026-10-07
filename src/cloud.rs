//! HTTP TTS providers with incremental signed-16-bit little-endian mono PCM.
use crate::{
    cloud_http::{self, CloudConfig},
    config::Config,
    engine::TtsBackend,
    voices::Voice,
};
use anyhow::{Result, bail, ensure};
use serde_json::json;
use std::io::Read;

pub fn is_cloud(kind: &str) -> bool {
    matches!(
        kind,
        "elevenlabs" | "openai-compatible" | "cartesia" | "deepgram"
    )
}
fn defaults(kind: &str) -> Result<(&'static str, &'static str, &'static str, &'static str)> {
    Ok(match kind {
        "elevenlabs" => (
            "https://api.elevenlabs.io",
            "ELEVENLABS_API_KEY",
            "eleven_flash_v2_5",
            "",
        ),
        "openai-compatible" => (
            "https://api.openai.com/v1",
            "OPENAI_API_KEY",
            "gpt-4o-mini-tts",
            "alloy",
        ),
        "cartesia" => (
            "https://api.cartesia.ai",
            "CARTESIA_API_KEY",
            "sonic-3.6",
            "",
        ),
        "deepgram" => (
            "https://api.deepgram.com/v1",
            "DEEPGRAM_API_KEY",
            "aura-2-thalia-en",
            "aura-2-thalia-en",
        ),
        _ => bail!("unknown cloud TTS provider"),
    })
}
pub fn model_name(config: &Config) -> String {
    if config.backend.cloud.model.is_empty() {
        defaults(&config.backend.kind)
            .map(|d| d.2.to_owned())
            .unwrap_or_default()
    } else {
        config.backend.cloud.model.clone()
    }
}
pub fn voices(config: &Config) -> Result<Vec<Voice>> {
    let (_, _, _, default_voice) = defaults(&config.backend.kind)?;
    let c = &config.backend.cloud;
    ensure!(
        c.voices.len() <= 64,
        "at most 64 cloud voice aliases are supported"
    );
    if !c.voices.is_empty() {
        ensure!(
            c.voice.is_empty(),
            "choose cloud.voice or cloud.voices, not both"
        );
        return c
            .voices
            .iter()
            .enumerate()
            .map(|(i, (name, id))| {
                ensure!(
                    !name.trim().is_empty()
                        && !id.trim().is_empty()
                        && name.len() <= 256
                        && id.len() <= 256
                        && !id.contains(['\r', '\n', '\0']),
                    "invalid cloud voice alias"
                );
                Ok(Voice {
                    id: i as i32,
                    name: name.clone(),
                })
            })
            .collect();
    }
    let voice = if c.voice.is_empty() {
        if config.backend.kind == "deepgram" && !c.model.is_empty() {
            &c.model
        } else {
            default_voice
        }
    } else {
        &c.voice
    };
    ensure!(
        !voice.trim().is_empty() && voice.len() <= 256 && !voice.contains(['\r', '\n', '\0']),
        "set backend.cloud.voice to a provider voice ID"
    );
    Ok(vec![Voice {
        id: 0,
        name: voice.to_owned(),
    }])
}
pub struct CloudBackend {
    kind: &'static str,
    config: CloudConfig,
    base: url::Url,
    model: String,
    language: String,
    voices: Vec<String>,
}
impl CloudBackend {
    pub fn create(config: &Config) -> Result<Self> {
        let (base, env, _, _) = defaults(&config.backend.kind)?;
        ensure!(
            config.backend.runtime == crate::backend::Runtime::Default
                && config.backend.device_id == 0
                && config.backend.fallback == crate::backend::Fallback::Error,
            "cloud providers require runtime=default, device_id=0, fallback=error; placement is remote"
        );
        ensure!(
            matches!(
                config.backend.device.to_ascii_lowercase().as_str(),
                "auto" | "cpu" | "remote"
            ),
            "cloud device must be remote or the default auto/cpu"
        );
        ensure!(
            config.backend.options.is_empty() && config.model.options.is_empty(),
            "cloud providers do not accept native backend/model options"
        );
        let inventory = voices(config)?;
        crate::voices::validate_selected(config, &inventory)?;
        let c = config.backend.cloud.clone();
        let base = cloud_http::base_url(&c, base)?;
        cloud_http::credential(
            &c,
            env,
            config.backend.kind != "openai-compatible" || c.base_url.is_empty(),
        )?;
        let voices = inventory
            .iter()
            .map(|v| {
                c.voices
                    .get(&v.name)
                    .cloned()
                    .unwrap_or_else(|| v.name.clone())
            })
            .collect();
        let kind = match config.backend.kind.as_str() {
            "elevenlabs" => "elevenlabs",
            "cartesia" => "cartesia",
            "deepgram" => "deepgram",
            _ => "openai-compatible",
        };
        let model = model_name(config);
        ensure!(
            !model.is_empty() && model.len() <= 256 && !model.contains(['\r', '\n', '\0']),
            "invalid cloud model"
        );
        ensure!(
            config.model.language.len() <= 32
                && config
                    .model
                    .language
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-'),
            "invalid cloud language"
        );
        if kind == "deepgram" {
            ensure!(
                config.model.language == "en",
                "initial Deepgram TTS adapter supports English presets; select language=en"
            );
        }
        Ok(Self {
            kind,
            config: c,
            base,
            model,
            language: config.model.language.clone(),
            voices,
        })
    }
    fn request(&self, text: &str, speed: f32, voice: i32) -> Result<ureq::Response> {
        let voice = self
            .voices
            .get(usize::try_from(voice).unwrap_or(usize::MAX))
            .ok_or_else(|| anyhow::anyhow!("invalid cloud voice"))?;
        let (_, env, _, _) = defaults(self.kind)?;
        let key = cloud_http::credential(
            &self.config,
            env,
            self.kind != "openai-compatible" || self.config.base_url.is_empty(),
        )?;
        let mut url = self.base.clone();
        let prefix = url.path().trim_end_matches('/').to_owned();
        let body = match self.kind {
            "elevenlabs" => {
                ensure!(
                    (0.7..=1.2).contains(&speed),
                    "ElevenLabs speed must be 0.7..1.2"
                );
                // Path-segment API escapes provider IDs rather than interpolating URLs.
                url.path_segments_mut()
                    .map_err(|_| anyhow::anyhow!("invalid cloud URL"))?
                    .pop_if_empty()
                    .extend(["v1", "text-to-speech", voice, "stream"]);
                url.query_pairs_mut()
                    .append_pair("output_format", "pcm_24000");
                json!({"text":text,"model_id":self.model,"language_code":self.language,"voice_settings":{"speed":speed}})
            }
            "openai-compatible" => {
                ensure!(
                    (0.25..=4.0).contains(&speed),
                    "OpenAI-compatible speed must be 0.25..4"
                );
                url.set_path(&format!("{prefix}/audio/speech"));
                json!({"input":text,"model":self.model,"voice":voice,"response_format":"pcm","stream_format":"audio","speed":speed})
            }
            "cartesia" => {
                ensure!(
                    (0.6..=1.5).contains(&speed),
                    "Cartesia speed must be 0.6..1.5"
                );
                url.set_path(&format!("{prefix}/tts/bytes"));
                json!({"transcript":text,"model_id":self.model,"voice":voice,"language":self.language,
                    "output_format":{"container":"raw","encoding":"pcm_s16le","sample_rate":24000},"generation_config":{"speed":speed}})
            }
            "deepgram" => {
                ensure!(
                    (0.7..=1.5).contains(&speed),
                    "Deepgram speed must be 0.7..1.5"
                );
                url.set_path(&format!("{prefix}/speak"));
                url.query_pairs_mut()
                    .append_pair("model", voice)
                    .append_pair("encoding", "linear16")
                    .append_pair("sample_rate", "24000")
                    .append_pair("container", "none")
                    .append_pair("speed", &speed.to_string());
                json!({"text":text})
            }
            _ => unreachable!(),
        };
        let agent = cloud_http::agent(&self.config);
        let mut req = agent
            .post(url.as_str())
            .set("Content-Type", "application/json")
            .set("Accept", "application/octet-stream");
        if let Some(key) = key {
            req = if self.kind == "elevenlabs" {
                req.set("xi-api-key", &key)
            } else {
                req.set(
                    "Authorization",
                    &format!(
                        "{} {key}",
                        if self.kind == "deepgram" {
                            "Token"
                        } else {
                            "Bearer"
                        }
                    ),
                )
            };
        }
        if self.kind == "cartesia" {
            req = req.set("Cartesia-Version", "2026-08-14");
        }
        cloud_http::response(req.send_string(&body.to_string()))
    }
}
impl TtsBackend for CloudBackend {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn sample_rate(&self) -> i32 {
        24000
    }
    fn num_voices(&self) -> i32 {
        self.voices.len() as i32
    }
    fn generate(&self, text: &str, speed: f32, voice: i32) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.generate_stream(text, speed, voice, &mut |pcm| {
            out.extend_from_slice(pcm);
            Ok(())
        })?;
        Ok(out)
    }
    fn generate_stream(
        &self,
        text: &str,
        speed: f32,
        voice: i32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let response = self.request(text, speed, voice)?;
        let content_type = response.header("Content-Type").unwrap_or("");
        for parameter in content_type.split(';').skip(1) {
            if let Some((name, value)) = parameter.trim().split_once('=') {
                if matches!(name, "rate" | "sample_rate") {
                    ensure!(
                        value.trim_matches('"') == "24000",
                        "cloud PCM sample rate differs from requested 24000 Hz"
                    );
                }
                if name == "channels" {
                    ensure!(value.trim_matches('"') == "1", "cloud PCM must be mono");
                }
            }
        }
        let mime = response
            .header("Content-Type")
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        ensure!(
            matches!(
                mime,
                "application/octet-stream"
                    | "audio/pcm"
                    | "audio/raw"
                    | "audio/l16"
                    | "audio/linear16"
                    | "audio/x-pcm"
            ),
            "cloud response must be raw PCM (unexpected content type)"
        );
        decode_pcm(
            response.into_reader(),
            24000 * self.config.max_audio_seconds as usize,
            sink,
        )
    }
}
fn decode_pcm(
    mut reader: impl Read,
    max_samples: usize,
    sink: &mut dyn FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    let mut bytes = [0u8; 8192];
    let mut pending = None;
    let mut total = 0;
    loop {
        let n = reader
            .read(&mut bytes)
            .map_err(|_| anyhow::anyhow!("cloud audio read failed or timed out"))?;
        if n == 0 {
            break;
        }
        let mut pcm = Vec::with_capacity(n / 2 + 1);
        let mut offset = 0;
        if let Some(low) = pending.take() {
            pcm.push(i16::from_le_bytes([low, bytes[0]]) as f32 / 32768.0);
            offset = 1;
        }
        for pair in bytes[offset..n].as_chunks::<2>().0 {
            pcm.push(i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0);
        }
        if (n - offset) % 2 == 1 {
            pending = Some(bytes[n - 1]);
        }
        total += pcm.len();
        ensure!(
            total <= max_samples,
            "cloud audio exceeds configured duration limit"
        );
        if !pcm.is_empty() {
            sink(&pcm)?;
        }
    }
    ensure!(pending.is_none(), "cloud PCM ended with a truncated sample");
    ensure!(total > 0, "cloud provider returned empty audio");
    Ok(())
}
#[cfg(test)]
#[path = "../tests/unit/cloud.rs"]
mod tests;
