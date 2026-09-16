use std::collections::BTreeSet;

use super::*;

#[test]
fn every_catalog_file_is_pinned_and_every_model_references_a_backend() {
    for model in models() {
        assert!(
            backends()
                .iter()
                .any(|backend| backend.kind == model.backend)
        );
        assert!(!model.files.is_empty());
        assert_eq!(
            model.download_size(),
            model.files.iter().map(|file| file.size).sum::<u64>()
        );
        assert!(!model.license.is_empty());
        assert!(model.license_url.starts_with("https://"));
        assert!(!model.license_status.is_empty());
        assert!(!model.voices.is_empty());
        assert!(
            model
                .voices
                .windows(2)
                .all(|pair| pair[1].id == pair[0].id + 1)
        );
        let mut paths = BTreeSet::new();
        for file in model.files {
            assert!(paths.insert(file.path));
            assert!(file.url.starts_with("https://"));
            assert!(file.size > 0);
            assert_eq!(file.sha256.len(), 64);
        }
        if model.requires_acceptance {
            assert_eq!(model.license_sha256.len(), 64);
            assert_eq!(model_license_text(model).unwrap().len(), 15_007);
        }
    }
}

#[test]
fn direct_openvino_uses_one_complete_official_archive_snapshot() {
    let spec = model("supertonic-3-openvino").unwrap();
    assert!(spec.downloadable && spec.openvino_capable && spec.npu_capable);
    assert_eq!(spec.files.len(), 16);
    assert_eq!(spec.artifact_source, OFFICIAL_SOURCE);
    assert_eq!(spec.artifact_revision, OFFICIAL_REVISION);
    assert_eq!(spec.original_model_source, OFFICIAL_SOURCE);
    assert_eq!(spec.original_model_revision, OFFICIAL_REVISION);
    assert!(
        spec.files
            .iter()
            .all(|file| file.url.contains(OFFICIAL_REVISION))
    );
    assert!(
        spec.files
            .iter()
            .all(|file| file.url.contains("supertone-oss-archive"))
    );
    let paths = spec
        .files
        .iter()
        .map(|file| file.path)
        .collect::<BTreeSet<_>>();
    for path in [
        "onnx/duration_predictor.onnx",
        "onnx/text_encoder.onnx",
        "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx",
        "onnx/tts.json",
        "onnx/unicode_indexer.json",
        "voice_styles/M1.json",
        "voice_styles/M5.json",
        "voice_styles/F1.json",
        "voice_styles/F5.json",
    ] {
        assert!(paths.contains(path));
    }
    assert!(
        !paths
            .iter()
            .any(|path| path.contains("int8") || path.ends_with(".bin"))
    );
    assert!(model("supertonic-3-int8").is_none());
    assert!(model("supertonic-3-npu").is_none());
}

#[test]
fn converted_default_keeps_distinct_artifact_provenance() {
    let gguf = model("supertonic-3-gguf").unwrap();
    assert_eq!(gguf.backend, "audiocpp");
    assert_eq!(gguf.files.len(), 1);
    assert_eq!(gguf.model_file, "supertonic-3-orig.gguf");
    assert_eq!(gguf.files[0].size, 454_072_836);
    assert_eq!(
        gguf.artifact_source,
        "https://huggingface.co/audio-cpp/audio.cpp-gguf"
    );
    assert_eq!(gguf.original_model_source, OFFICIAL_SOURCE);
    assert!(
        gguf.license_status
            .contains("conversion supplied by audio.cpp")
    );
}

#[test]
fn lookup_and_activation_populate_official_paths() {
    assert!(model("missing").is_none());
    let spec = model("supertonic-3-openvino").unwrap();
    let mut config = Config::default();
    config.model.directory = "/custom".into();
    config.model.voice = crate::voices::VoiceSelection::Legacy(9);
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "supertonic");
    assert_eq!(config.model.name, "supertonic-3-openvino");
    assert!(config.model.directory.is_empty());
    assert!(config.model.file.is_empty());
    assert_eq!(
        config.model.duration_predictor,
        "onnx/duration_predictor.onnx"
    );
    assert_eq!(config.model.unicode_indexer, "onnx/unicode_indexer.json");
    assert_eq!(config.model.voice_style, "voice_styles");
    assert_eq!(config.model.voice, crate::voices::VoiceSelection::Legacy(0));
}

#[test]
fn defaults_cover_all_supported_runtime_devices_without_crossing_formats() {
    for (backend, runtime, devices) in [
        ("audiocpp", Runtime::Default, vec!["cpu", "auto"]),
        ("audiocpp", Runtime::Cuda, vec!["gpu"]),
        ("audiocpp", Runtime::Vulkan, vec!["gpu"]),
        ("audiocpp", Runtime::Hip, vec!["gpu"]),
        (
            "supertonic",
            Runtime::Openvino,
            vec!["cpu", "gpu", "npu", "auto"],
        ),
    ] {
        for device in devices {
            let spec = default_model(backend, runtime, device).unwrap();
            assert!(spec.downloadable);
            assert!(spec.voices.iter().any(|voice| voice.id == 0));
            for other in models().iter().filter(|other| other.backend != backend) {
                assert!(!other.compatible_with(backend, runtime, device));
            }
        }
    }
    for (backend, runtime, device) in [
        ("missing", Runtime::Default, "cpu"),
        ("audiocpp", Runtime::Openvino, "cpu"),
        ("supertonic", Runtime::Default, "cpu"),
        ("supertonic", Runtime::Openvino, "bogus"),
    ] {
        assert!(default_model(backend, runtime, device).is_err());
    }
    let mut cpu_only = *model(OPENVINO_MODEL_ID).unwrap();
    cpu_only.npu_capable = false;
    assert!(!cpu_only.compatible_with("supertonic", Runtime::Openvino, " NPU "));
}

#[test]
fn model_activation_resets_foreign_runtime_state_but_preserves_compatible_provider() {
    let mut config = Config::default();
    config.backend.library = Some("/old/libaudiocpp.so".into());
    config.backend.device_id = 2;
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.options.insert("old".into(), "option".into());
    model(OPENVINO_MODEL_ID).unwrap().activate(&mut config);
    assert_eq!(config.backend.runtime, Runtime::Openvino);
    assert_eq!(config.backend.device, "cpu");
    assert!(config.backend.library.is_none());
    assert!(config.backend.options.is_empty());
    assert_eq!(config.backend.device_id, 0);
    config.backend.openvino_library = Some("/opt/libopenvino_c.so".into());
    config.backend.device = "npu".into();
    model(OPENVINO_MODEL_ID).unwrap().activate(&mut config);
    assert_eq!(config.backend.device, "npu");
    assert!(config.backend.openvino_library.is_some());
    model(DEFAULT_MODEL_ID).unwrap().activate(&mut config);
    assert!(config.backend.openvino_library.is_none());
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(setup_model(&config).unwrap().id, DEFAULT_MODEL_ID);
    assert!(
        serde_json::to_value(model(DEFAULT_MODEL_ID).unwrap())
            .unwrap()
            .get("display_name")
            .is_none()
    );
}
