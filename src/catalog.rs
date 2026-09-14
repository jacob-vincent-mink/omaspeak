use serde::Serialize;

use crate::config::Config;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BackendSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub built: bool,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RequiredFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub backend: &'static str,
    pub family: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub archive_url: &'static str,
    pub archive_size: u64,
    pub archive_sha256: &'static str,
    pub archive_root: &'static str,
    pub model_file: &'static str,
    pub tokens_file: &'static str,
    pub data_directory: &'static str,
    pub duration_predictor: &'static str,
    pub text_encoder: &'static str,
    pub vector_estimator: &'static str,
    pub vocoder: &'static str,
    pub tts_json: &'static str,
    pub unicode_indexer: &'static str,
    pub voice_style: &'static str,
    pub language: &'static str,
    pub steps: i32,
    pub npu_capable: bool,
    pub required_files: &'static [RequiredFile],
}

const BACKENDS: &[BackendSpec] = &[BackendSpec {
    kind: "sherpa-onnx",
    name: "sherpa-onnx",
    built: true,
    description: "In-process ONNX speech synthesis through the official Rust crate",
}];

const LESSAC_FILES: &[RequiredFile] = &[
    RequiredFile {
        path: "en_US-lessac-medium.onnx",
        size: 63_149_198,
        sha256: "4ba07d8549906668ee855fd9abf9faf66c5db74742712ff026a159f7277fca9f",
    },
    RequiredFile {
        path: "en_US-lessac-medium.onnx.json",
        size: 4_885,
        sha256: "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    },
    RequiredFile {
        path: "tokens.txt",
        size: 921,
        sha256: "87c8ef66eae5473ed0cc0366b3964c736ca6c5f676c979522ea31234e47430b9",
    },
    RequiredFile {
        path: "espeak-ng-data/phontab",
        size: 55_796,
        sha256: "886f3fa402cb0ba73d483aa8ad000af47a6b7cc06293c75a97913fba68a530f6",
    },
    RequiredFile {
        path: "espeak-ng-data/phondata",
        size: 550_424,
        sha256: "4e0288957874029a8c3c9f41a8f517ad4bf18127046decbdd4b9d1d6807ce3a3",
    },
];

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

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "en_US-lessac-medium",
        backend: "sherpa-onnx",
        family: "piper",
        name: "en_US-lessac-medium",
        description: "Piper US English, medium quality (about 79 MiB installed)",
        archive_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
        archive_size: 67_230_653,
        archive_sha256: "9e3febfacf0abf4270172d2958bcec246032b7e88efc2720840cc80c93de334e",
        archive_root: "vits-piper-en_US-lessac-medium",
        model_file: "en_US-lessac-medium.onnx",
        tokens_file: "tokens.txt",
        data_directory: "espeak-ng-data",
        duration_predictor: "",
        text_encoder: "",
        vector_estimator: "",
        vocoder: "",
        tts_json: "",
        unicode_indexer: "",
        voice_style: "",
        language: "en",
        steps: 5,
        npu_capable: false,
        required_files: LESSAC_FILES,
    },
    ModelSpec {
        id: "supertonic-3-int8",
        backend: "sherpa-onnx",
        family: "supertonic",
        name: "supertonic-3-int8",
        description: "Supertonic 3 multilingual int8 (31 languages; OpenVINO evaluation model)",
        archive_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        archive_size: 128_774_318,
        archive_sha256: "82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427",
        archive_root: "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        model_file: "",
        tokens_file: "",
        data_directory: "",
        duration_predictor: "duration_predictor.int8.onnx",
        text_encoder: "text_encoder.int8.onnx",
        vector_estimator: "vector_estimator.int8.onnx",
        vocoder: "vocoder.int8.onnx",
        tts_json: "tts.json",
        unicode_indexer: "unicode_indexer.bin",
        voice_style: "voice.bin",
        language: "en",
        steps: 5,
        npu_capable: false,
        required_files: SUPERTONIC_FILES,
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

impl ModelSpec {
    pub fn activate(self, config: &mut Config) {
        config.backend.kind = self.backend.into();
        config.model.family = self.family.into();
        config.model.name = self.name.into();
        config.model.directory.clear();
        config.model.model_file = self.model_file.into();
        config.model.tokens_file = self.tokens_file.into();
        config.model.data_directory = self.data_directory.into();
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
