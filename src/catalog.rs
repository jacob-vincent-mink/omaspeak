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

const MODELS: &[ModelSpec] = &[ModelSpec {
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
    required_files: LESSAC_FILES,
}];

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
        config.model.voice = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_references_a_backend() {
        for model in models() {
            assert!(
                backends()
                    .iter()
                    .any(|backend| backend.kind == model.backend)
            );
            assert_eq!(model.archive_sha256.len(), 64);
            assert!(!model.required_files.is_empty());
        }
    }
}
