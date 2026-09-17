use serde::Serialize;

use crate::backend::Runtime;
use crate::config::Config;

pub const DEFAULT_MODEL_ID: &str = "supertonic-3-gguf";
pub const OPENVINO_MODEL_ID: &str = "supertonic-3-openvino";
pub const KOKORO_MODEL_ID: &str = "kokoro-82m-gguf";

pub const SUPERTONIC_VOICE_NAMES: [&str; 10] =
    ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"];

/// Engine voice IDs for the pinned Kokoro 82M GGUF, in the exact order of its
/// embedded `voices.json` (54 preset voice packs, all languages).
pub const KOKORO_VOICE_IDS: [&str; 54] = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
];

const OFFICIAL_REVISION: &str = "aafc6e32416a594460b32413efc49d7fe4ce6d46";
const OFFICIAL_SOURCE: &str = "https://huggingface.co/supertone-oss-archive/supertonic-3";

const KOKORO_ARTIFACT_REVISION: &str = "1b13cd58245c74e3ff4ca06925766c5ef7991bd4";
const KOKORO_ARTIFACT_SOURCE: &str = "https://huggingface.co/audio-cpp/audio.cpp-gguf";
const KOKORO_MODEL_SOURCE: &str = "https://huggingface.co/hexgrad/Kokoro-82M";

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BackendSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub description: &'static str,
}

/// One independently downloadable, content-addressed model file.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    pub path: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct VoiceSpec {
    pub id: i32,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub backend: &'static str,
    pub family: &'static str,
    pub name: &'static str,
    #[serde(skip)] // Presentation is not part of the schema-1 install identity.
    pub display_name: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub license_status: &'static str,
    pub downloadable: bool,
    pub requires_acceptance: bool,
    pub source_revision: &'static str,
    pub artifact_source: &'static str,
    pub artifact_revision: &'static str,
    pub original_model_source: &'static str,
    pub original_model_revision: &'static str,
    pub license_file: &'static str,
    pub license_sha256: &'static str,
    /// The single model path passed to providers such as audio.cpp.
    pub model_file: &'static str,
    pub duration_predictor: &'static str,
    pub text_encoder: &'static str,
    pub vector_estimator: &'static str,
    pub vocoder: &'static str,
    pub tts_json: &'static str,
    pub unicode_indexer: &'static str,
    /// A single style file or a directory containing named style files.
    pub voice_style: &'static str,
    pub language: &'static str,
    pub steps: i32,
    pub voices: &'static [VoiceSpec],
    pub openvino_capable: bool,
    pub npu_capable: bool,
    pub files: &'static [ModelFile],
}

const SUPERTONIC_VOICES: &[VoiceSpec] = &[
    VoiceSpec { id: 0, name: "M1" },
    VoiceSpec { id: 1, name: "M2" },
    VoiceSpec { id: 2, name: "M3" },
    VoiceSpec { id: 3, name: "M4" },
    VoiceSpec { id: 4, name: "M5" },
    VoiceSpec { id: 5, name: "F1" },
    VoiceSpec { id: 6, name: "F2" },
    VoiceSpec { id: 7, name: "F3" },
    VoiceSpec { id: 8, name: "F4" },
    VoiceSpec { id: 9, name: "F5" },
];

const KOKORO_VOICES: &[VoiceSpec] = &[
    VoiceSpec {
        id: 0,
        name: "af_alloy",
    },
    VoiceSpec {
        id: 1,
        name: "af_aoede",
    },
    VoiceSpec {
        id: 2,
        name: "af_bella",
    },
    VoiceSpec {
        id: 3,
        name: "af_heart",
    },
    VoiceSpec {
        id: 4,
        name: "af_jessica",
    },
    VoiceSpec {
        id: 5,
        name: "af_kore",
    },
    VoiceSpec {
        id: 6,
        name: "af_nicole",
    },
    VoiceSpec {
        id: 7,
        name: "af_nova",
    },
    VoiceSpec {
        id: 8,
        name: "af_river",
    },
    VoiceSpec {
        id: 9,
        name: "af_sarah",
    },
    VoiceSpec {
        id: 10,
        name: "af_sky",
    },
    VoiceSpec {
        id: 11,
        name: "am_adam",
    },
    VoiceSpec {
        id: 12,
        name: "am_echo",
    },
    VoiceSpec {
        id: 13,
        name: "am_eric",
    },
    VoiceSpec {
        id: 14,
        name: "am_fenrir",
    },
    VoiceSpec {
        id: 15,
        name: "am_liam",
    },
    VoiceSpec {
        id: 16,
        name: "am_michael",
    },
    VoiceSpec {
        id: 17,
        name: "am_onyx",
    },
    VoiceSpec {
        id: 18,
        name: "am_puck",
    },
    VoiceSpec {
        id: 19,
        name: "am_santa",
    },
    VoiceSpec {
        id: 20,
        name: "bf_alice",
    },
    VoiceSpec {
        id: 21,
        name: "bf_emma",
    },
    VoiceSpec {
        id: 22,
        name: "bf_isabella",
    },
    VoiceSpec {
        id: 23,
        name: "bf_lily",
    },
    VoiceSpec {
        id: 24,
        name: "bm_daniel",
    },
    VoiceSpec {
        id: 25,
        name: "bm_fable",
    },
    VoiceSpec {
        id: 26,
        name: "bm_george",
    },
    VoiceSpec {
        id: 27,
        name: "bm_lewis",
    },
    VoiceSpec {
        id: 28,
        name: "ef_dora",
    },
    VoiceSpec {
        id: 29,
        name: "em_alex",
    },
    VoiceSpec {
        id: 30,
        name: "em_santa",
    },
    VoiceSpec {
        id: 31,
        name: "ff_siwis",
    },
    VoiceSpec {
        id: 32,
        name: "hf_alpha",
    },
    VoiceSpec {
        id: 33,
        name: "hf_beta",
    },
    VoiceSpec {
        id: 34,
        name: "hm_omega",
    },
    VoiceSpec {
        id: 35,
        name: "hm_psi",
    },
    VoiceSpec {
        id: 36,
        name: "if_sara",
    },
    VoiceSpec {
        id: 37,
        name: "im_nicola",
    },
    VoiceSpec {
        id: 38,
        name: "jf_alpha",
    },
    VoiceSpec {
        id: 39,
        name: "jf_gongitsune",
    },
    VoiceSpec {
        id: 40,
        name: "jf_nezumi",
    },
    VoiceSpec {
        id: 41,
        name: "jf_tebukuro",
    },
    VoiceSpec {
        id: 42,
        name: "jm_kumo",
    },
    VoiceSpec {
        id: 43,
        name: "pf_dora",
    },
    VoiceSpec {
        id: 44,
        name: "pm_alex",
    },
    VoiceSpec {
        id: 45,
        name: "pm_santa",
    },
    VoiceSpec {
        id: 46,
        name: "zf_xiaobei",
    },
    VoiceSpec {
        id: 47,
        name: "zf_xiaoni",
    },
    VoiceSpec {
        id: 48,
        name: "zf_xiaoxiao",
    },
    VoiceSpec {
        id: 49,
        name: "zf_xiaoyi",
    },
    VoiceSpec {
        id: 50,
        name: "zm_yunjian",
    },
    VoiceSpec {
        id: 51,
        name: "zm_yunxi",
    },
    VoiceSpec {
        id: 52,
        name: "zm_yunxia",
    },
    VoiceSpec {
        id: 53,
        name: "zm_yunyang",
    },
];

const BACKENDS: &[BackendSpec] = &[
    BackendSpec {
        kind: "audiocpp",
        name: "audio.cpp",
        description: "Portable GGUF synthesis through one complete native audio.cpp provider",
    },
    BackendSpec {
        kind: "supertonic",
        name: "Direct OpenVINO",
        description: "Direct execution of official Supertonic ONNX graphs on Intel hardware",
    },
];

const GGUF_FILES: &[ModelFile] = &[ModelFile {
    path: "supertonic-3-orig.gguf",
    url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/09fe073ba154561f4474162e8bd4ab233a848eca/Supertonic-3-GGUF/supertonic-3-orig.gguf?download=true",
    size: 454_072_836,
    sha256: "af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662",
}];

const OPENVINO_FILES: &[ModelFile] = &[
    ModelFile {
        path: "onnx/duration_predictor.onnx",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/duration_predictor.onnx?download=true",
        size: 3_700_147,
        sha256: "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    },
    ModelFile {
        path: "onnx/text_encoder.onnx",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/text_encoder.onnx?download=true",
        size: 36_416_150,
        sha256: "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    },
    ModelFile {
        path: "onnx/vector_estimator.onnx",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/vector_estimator.onnx?download=true",
        size: 256_534_781,
        sha256: "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
    },
    ModelFile {
        path: "onnx/vocoder.onnx",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/vocoder.onnx?download=true",
        size: 101_424_195,
        sha256: "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
    },
    ModelFile {
        path: "onnx/tts.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/tts.json?download=true",
        size: 8_253,
        sha256: "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    },
    ModelFile {
        path: "onnx/unicode_indexer.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/onnx/unicode_indexer.json?download=true",
        size: 277_676,
        sha256: "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
    },
    ModelFile {
        path: "voice_styles/M1.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/M1.json?download=true",
        size: 291_748,
        sha256: "e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b",
    },
    ModelFile {
        path: "voice_styles/M2.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/M2.json?download=true",
        size: 292_055,
        sha256: "b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50",
    },
    ModelFile {
        path: "voice_styles/M3.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/M3.json?download=true",
        size: 290_198,
        sha256: "ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b",
    },
    ModelFile {
        path: "voice_styles/M4.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/M4.json?download=true",
        size: 291_522,
        sha256: "ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303",
    },
    ModelFile {
        path: "voice_styles/M5.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/M5.json?download=true",
        size: 291_469,
        sha256: "dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2",
    },
    ModelFile {
        path: "voice_styles/F1.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/F1.json?download=true",
        size: 292_046,
        sha256: "bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2",
    },
    ModelFile {
        path: "voice_styles/F2.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/F2.json?download=true",
        size: 292_423,
        sha256: "7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6",
    },
    ModelFile {
        path: "voice_styles/F3.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/F3.json?download=true",
        size: 290_794,
        sha256: "12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab",
    },
    ModelFile {
        path: "voice_styles/F4.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/F4.json?download=true",
        size: 291_808,
        sha256: "c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b",
    },
    ModelFile {
        path: "voice_styles/F5.json",
        url: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/voice_styles/F5.json?download=true",
        size: 291_479,
        sha256: "45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d",
    },
];

const KOKORO_FILES: &[ModelFile] = &[ModelFile {
    path: "kokoro-82m-q8_0.gguf",
    url: "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/1b13cd58245c74e3ff4ca06925766c5ef7991bd4/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf?download=true",
    size: 189_611_360,
    sha256: "378abf37a0d086774f88e341165a51af631d09452ae37caed5e7cdc34c9889e6",
}];

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: DEFAULT_MODEL_ID,
        backend: "audiocpp",
        family: "supertonic",
        name: "supertonic-3-gguf",
        display_name: "Supertonic 3 · audio.cpp",
        description: "Supertonic 3 original-precision GGUF for audio.cpp (31 languages)",
        license: "OpenRAIL-M",
        license_url: "https://huggingface.co/supertone-oss-archive/supertonic-3/blob/aafc6e32416a594460b32413efc49d7fe4ce6d46/LICENSE",
        license_status: "verified model license; conversion supplied by audio.cpp",
        downloadable: true,
        requires_acceptance: true,
        source_revision: "09fe073ba154561f4474162e8bd4ab233a848eca",
        artifact_source: "https://huggingface.co/audio-cpp/audio.cpp-gguf",
        artifact_revision: "09fe073ba154561f4474162e8bd4ab233a848eca",
        original_model_source: OFFICIAL_SOURCE,
        original_model_revision: OFFICIAL_REVISION,
        license_file: "MODEL-LICENSE",
        license_sha256: "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
        model_file: "supertonic-3-orig.gguf",
        duration_predictor: "",
        text_encoder: "",
        vector_estimator: "",
        vocoder: "",
        tts_json: "",
        unicode_indexer: "",
        voice_style: "",
        language: "en",
        steps: 8,
        voices: SUPERTONIC_VOICES,
        openvino_capable: false,
        npu_capable: false,
        files: GGUF_FILES,
    },
    ModelSpec {
        id: OPENVINO_MODEL_ID,
        backend: "supertonic",
        family: "supertonic",
        name: "supertonic-3-openvino",
        display_name: "Supertonic 3 · OpenVINO",
        description: "Official Supertonic 3 ONNX graphs for direct OpenVINO (31 languages)",
        license: "OpenRAIL-M",
        license_url: "https://huggingface.co/supertone-oss-archive/supertonic-3/blob/aafc6e32416a594460b32413efc49d7fe4ce6d46/LICENSE",
        license_status: "official archived model files; explicit acceptance required",
        downloadable: true,
        requires_acceptance: true,
        source_revision: OFFICIAL_REVISION,
        artifact_source: OFFICIAL_SOURCE,
        artifact_revision: OFFICIAL_REVISION,
        original_model_source: OFFICIAL_SOURCE,
        original_model_revision: OFFICIAL_REVISION,
        license_file: "MODEL-LICENSE",
        license_sha256: "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
        model_file: "",
        duration_predictor: "onnx/duration_predictor.onnx",
        text_encoder: "onnx/text_encoder.onnx",
        vector_estimator: "onnx/vector_estimator.onnx",
        vocoder: "onnx/vocoder.onnx",
        tts_json: "onnx/tts.json",
        unicode_indexer: "onnx/unicode_indexer.json",
        voice_style: "voice_styles",
        language: "en",
        steps: 5,
        voices: SUPERTONIC_VOICES,
        openvino_capable: true,
        npu_capable: true,
        files: OPENVINO_FILES,
    },
    ModelSpec {
        id: KOKORO_MODEL_ID,
        backend: "audiocpp",
        family: "kokoro",
        name: "kokoro-82m-gguf",
        display_name: "Kokoro 82M · audio.cpp",
        description: "Kokoro 82M multilingual GGUF (54 preset voices) for audio.cpp; Apache-2.0, 24 kHz output",
        license: "Apache-2.0",
        license_url: "https://www.apache.org/licenses/LICENSE-2.0",
        license_status: "Apache-2.0 (hexgrad/Kokoro-82M weights; GGUF packaged by audio.cpp)",
        downloadable: true,
        requires_acceptance: false,
        source_revision: "",
        artifact_source: KOKORO_ARTIFACT_SOURCE,
        artifact_revision: KOKORO_ARTIFACT_REVISION,
        original_model_source: KOKORO_MODEL_SOURCE,
        original_model_revision: "",
        license_file: "MODEL-LICENSE",
        license_sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        model_file: "kokoro-82m-q8_0.gguf",
        duration_predictor: "",
        text_encoder: "",
        vector_estimator: "",
        vocoder: "",
        tts_json: "",
        unicode_indexer: "",
        voice_style: "",
        language: "en",
        steps: 0,
        voices: KOKORO_VOICES,
        openvino_capable: false,
        npu_capable: false,
        files: KOKORO_FILES,
    },
];

pub fn backends() -> &'static [BackendSpec] {
    BACKENDS
}
pub fn models() -> &'static [ModelSpec] {
    MODELS
}
pub fn model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|item| item.id == id)
}

pub fn model_license_text(spec: &ModelSpec) -> Option<&'static str> {
    match spec.license {
        "OpenRAIL-M" => Some(include_str!("../licenses/SUPERTONIC-3-MODEL-LICENSE")),
        "Apache-2.0" => Some(include_str!("../licenses/KOKORO-82M-MODEL-LICENSE")),
        _ => None,
    }
}

/// Format/adapter compatibility does not establish hardware qualification.
pub fn default_model(
    backend: &str,
    runtime: Runtime,
    device: &str,
) -> anyhow::Result<&'static ModelSpec> {
    let id = match backend {
        "audiocpp" => DEFAULT_MODEL_ID,
        "supertonic" => OPENVINO_MODEL_ID,
        _ => anyhow::bail!("no catalog default for backend {backend:?}"),
    };
    let spec = model(id).expect("catalog default exists");
    anyhow::ensure!(
        spec.compatible_with(backend, runtime, device),
        "backend {backend:?} has no compatible default for {runtime:?} / {device}"
    );
    Ok(spec)
}

pub fn setup_model(config: &Config) -> anyhow::Result<&'static ModelSpec> {
    if let Some(spec) = model(&config.model.name)
        && spec.compatible_with(
            &config.backend.kind,
            config.backend.runtime,
            &config.backend.device,
        )
    {
        return Ok(spec);
    }
    default_model(
        &config.backend.kind,
        config.backend.runtime,
        &config.backend.device,
    )
}

impl ModelSpec {
    pub fn compatible_with(self, backend: &str, runtime: Runtime, device: &str) -> bool {
        if self.backend != backend || crate::backend::canonical_device(runtime, device).is_err() {
            return false;
        }
        match backend {
            "audiocpp" => matches!(
                runtime,
                Runtime::Default | Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
            ),
            "supertonic" => {
                runtime == Runtime::Openvino
                    && self.openvino_capable
                    && (!device.trim().eq_ignore_ascii_case("npu") || self.npu_capable)
            }
            _ => false,
        }
    }

    pub fn download_size(self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }

    pub fn activate(self, config: &mut Config) {
        let device = config.backend.canonical_device().unwrap_or_else(|_| {
            if matches!(
                config.backend.runtime,
                Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
            ) {
                "gpu".into()
            } else {
                "cpu".into()
            }
        });
        config.backend.device = device;
        if !self.compatible_with(
            &config.backend.kind,
            config.backend.runtime,
            &config.backend.device,
        ) {
            config.backend.library = None;
            config.backend.openvino_library = None;
            config.backend.openvino_plugins = None;
            config.backend.library_dirs.clear();
            config.backend.options.clear();
            config.backend.device_id = 0;
            config.backend.fallback = Default::default();
            config.backend.runtime = if self.backend == "supertonic" {
                Runtime::Openvino
            } else {
                Runtime::Default
            };
            config.backend.device = "cpu".into();
        }
        config.backend.kind = self.backend.into();
        config.model.family = self.family.into();
        config.model.name = self.name.into();
        config.model.directory.clear();
        config.model.file = self.model_file.into();
        config.model.duration_predictor = self.duration_predictor.into();
        config.model.text_encoder = self.text_encoder.into();
        config.model.vector_estimator = self.vector_estimator.into();
        config.model.vocoder = self.vocoder.into();
        config.model.tts_json = self.tts_json.into();
        config.model.unicode_indexer = self.unicode_indexer.into();
        config.model.voice_style = self.voice_style.into();
        config.model.language = self.language.into();
        config.model.steps = self.steps;
        config.model.voice = 0.into();
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;
