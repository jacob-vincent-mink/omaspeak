use serde::Serialize;

use crate::config::Config;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BackendSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RequiredFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SupplementalFile {
    pub path: &'static str,
    pub supersedes: &'static str,
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
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub license_status: &'static str,
    pub downloadable: bool,
    pub requires_acceptance: bool,
    pub source_revision: &'static str,
    pub license_file: &'static str,
    pub license_sha256: &'static str,
    pub archive_url: &'static str,
    pub archive_size: u64,
    pub archive_sha256: &'static str,
    pub archive_root: &'static str,
    pub duration_predictor: &'static str,
    pub text_encoder: &'static str,
    pub vector_estimator: &'static str,
    pub vocoder: &'static str,
    pub tts_json: &'static str,
    pub unicode_indexer: &'static str,
    pub voice_style: &'static str,
    pub language: &'static str,
    pub steps: i32,
    pub voices: &'static [VoiceSpec],
    pub openvino_capable: bool,
    pub npu_capable: bool,
    pub required_files: &'static [RequiredFile],
    pub supplemental_files: &'static [SupplementalFile],
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

const BACKENDS: &[BackendSpec] = &[BackendSpec {
    kind: "supertonic",
    name: "Supertonic",
    description: "Local Supertonic synthesis through runtime-loaded inference engines",
}];

const SUPERTONIC_FILES: &[RequiredFile] = &[
    RequiredFile {
        path: "duration_predictor.int8.onnx",
        size: 3_700_147,
        sha256: "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    },
    RequiredFile {
        path: "text_encoder.int8.onnx",
        size: 36_416_150,
        sha256: "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    },
    RequiredFile {
        path: "vector_estimator.int8.onnx",
        size: 78_400_833,
        sha256: "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd",
    },
    RequiredFile {
        path: "vocoder.int8.onnx",
        size: 25_991_073,
        sha256: "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
    },
    RequiredFile {
        path: "tts.json",
        size: 8_253,
        sha256: "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    },
    RequiredFile {
        path: "unicode_indexer.bin",
        size: 262_144,
        sha256: "8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30",
    },
    RequiredFile {
        path: "voice.bin",
        size: 517_168,
        sha256: "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8",
    },
];

const SUPERTONIC_NPU_FILES: &[RequiredFile] = &[
    RequiredFile {
        path: "duration_predictor.int8.onnx",
        size: 3_700_147,
        sha256: "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
    },
    RequiredFile {
        path: "text_encoder.int8.onnx",
        size: 36_416_150,
        sha256: "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
    },
    RequiredFile {
        path: "vector_estimator.onnx",
        size: 256_534_781,
        sha256: "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
    },
    RequiredFile {
        path: "vocoder.int8.onnx",
        size: 25_991_073,
        sha256: "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
    },
    RequiredFile {
        path: "tts.json",
        size: 8_253,
        sha256: "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
    },
    RequiredFile {
        path: "unicode_indexer.bin",
        size: 262_144,
        sha256: "8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30",
    },
    RequiredFile {
        path: "voice.bin",
        size: 517_168,
        sha256: "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8",
    },
];

const SUPERTONIC_NPU_SUPPLEMENTS: &[SupplementalFile] = &[SupplementalFile {
    path: "vector_estimator.onnx",
    supersedes: "vector_estimator.int8.onnx",
    url: "https://huggingface.co/Supertone/supertonic-3/resolve/724fb5abbf5502583fb520898d45929e62f02c0b/onnx/vector_estimator.onnx?download=true",
    size: 256_534_781,
    sha256: "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
}];

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "supertonic-3-int8",
        backend: "supertonic",
        family: "supertonic",
        name: "supertonic-3-int8",
        description: "Supertonic 3 multilingual int8 (31 languages; OpenVINO evaluation model)",
        license: "OpenRAIL-M",
        license_url: "https://huggingface.co/Supertone/supertonic-3/blob/724fb5abbf5502583fb520898d45929e62f02c0b/LICENSE",
        license_status: "verified",
        downloadable: true,
        requires_acceptance: true,
        source_revision: "724fb5abbf5502583fb520898d45929e62f02c0b",
        license_file: "MODEL-LICENSE",
        license_sha256: "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
        archive_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        archive_size: 128_774_318,
        archive_sha256: "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427",
        archive_root: "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        duration_predictor: "duration_predictor.int8.onnx",
        text_encoder: "text_encoder.int8.onnx",
        vector_estimator: "vector_estimator.int8.onnx",
        vocoder: "vocoder.int8.onnx",
        tts_json: "tts.json",
        unicode_indexer: "unicode_indexer.bin",
        voice_style: "voice.bin",
        language: "en",
        steps: 5,
        voices: SUPERTONIC_VOICES,
        openvino_capable: true,
        npu_capable: false,
        required_files: SUPERTONIC_FILES,
        supplemental_files: &[],
    },
    ModelSpec {
        id: "supertonic-3-npu",
        backend: "supertonic",
        family: "supertonic",
        name: "supertonic-3-npu",
        description: "Supertonic 3 for Intel NPU (FP32 vector estimator; 31 languages)",
        license: "OpenRAIL-M",
        license_url: "https://huggingface.co/Supertone/supertonic-3/blob/724fb5abbf5502583fb520898d45929e62f02c0b/LICENSE",
        license_status: "verified-modified",
        downloadable: true,
        requires_acceptance: true,
        source_revision: "724fb5abbf5502583fb520898d45929e62f02c0b",
        license_file: "MODEL-LICENSE",
        license_sha256: "0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f",
        archive_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        archive_size: 128_774_318,
        archive_sha256: "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427",
        archive_root: "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        duration_predictor: "duration_predictor.int8.onnx",
        text_encoder: "text_encoder.int8.onnx",
        vector_estimator: "vector_estimator.onnx",
        vocoder: "vocoder.int8.onnx",
        tts_json: "tts.json",
        unicode_indexer: "unicode_indexer.bin",
        voice_style: "voice.bin",
        language: "en",
        steps: 5,
        voices: SUPERTONIC_VOICES,
        openvino_capable: true,
        npu_capable: true,
        required_files: SUPERTONIC_NPU_FILES,
        supplemental_files: SUPERTONIC_NPU_SUPPLEMENTS,
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
    (spec.license == "OpenRAIL-M").then_some(include_str!("../licenses/SUPERTONIC-3-MODEL-LICENSE"))
}

impl ModelSpec {
    pub fn activate(self, config: &mut Config) {
        config.backend.kind = self.backend.into();
        config.model.family = self.family.into();
        config.model.name = self.name.into();
        config.model.directory.clear();
        config.model.duration_predictor = self.duration_predictor.into();
        config.model.text_encoder = self.text_encoder.into();
        config.model.vector_estimator = self.vector_estimator.into();
        config.model.vocoder = self.vocoder.into();
        config.model.tts_json = self.tts_json.into();
        config.model.unicode_indexer = self.unicode_indexer.into();
        config.model.voice_style = self.voice_style.into();
        config.model.language = self.language.into();
        config.model.steps = self.steps;
        config.model.voice = 0;
    }
}

#[cfg(test)]
#[path = "../tests/unit/catalog.rs"]
mod tests;
