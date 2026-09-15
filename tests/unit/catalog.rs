use super::*;

#[test]
fn every_model_references_a_backend() {
    for model in models() {
        assert!(
            backends()
                .iter()
                .any(|backend| backend.kind == model.backend)
        );
        if let Some(file) = model.single_file {
            assert_eq!(file.sha256.len(), 64);
            assert!(file.url.starts_with("https://"));
            assert!(file.size > 0);
            assert!(!file.path.is_empty());
        } else if model.downloadable {
            assert_eq!(model.archive_sha256.len(), 64);
            assert!(model.archive_url.starts_with("https://"));
            assert!(model.archive_size > 0);
        }
        assert!(!model.license.is_empty());
        assert!(model.license_url.starts_with("https://"));
        assert!(!model.license_status.is_empty());
        if model.requires_acceptance {
            assert!(!model.license_file.is_empty());
            assert_eq!(model.license_sha256.len(), 64);
            assert!(model_license_text(model).is_some());
        }
        assert!(!model.required_files.is_empty());
        assert!(!model.voices.is_empty());
        assert_eq!(model.voices[0].id, 0);
        assert!(
            model
                .voices
                .windows(2)
                .all(|pair| pair[1].id == pair[0].id + 1)
        );
        for required in model.required_files {
            assert_eq!(required.sha256.len(), 64);
            assert!(required.size > 0);
        }
        for supplemental in model.supplemental_files {
            assert_eq!(supplemental.sha256.len(), 64);
            assert!(supplemental.size > 0);
            assert!(supplemental.url.starts_with("https://"));
        }
    }
}

#[test]
fn catalog_enforces_current_model_license_policy() {
    let gguf = model("supertonic-3-gguf").unwrap();
    assert!(gguf.downloadable);
    assert_eq!(gguf.backend, "audiocpp");
    assert_eq!(
        gguf.source_revision,
        "09fe073ba154561f4474162e8bd4ab233a848eca"
    );
    let file = gguf.single_file.unwrap();
    assert_eq!(file.size, 454_072_836);
    assert_eq!(
        file.sha256,
        "af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662"
    );

    for id in ["supertonic-3-int8", "supertonic-3-npu"] {
        let supertonic = model(id).unwrap();
        assert!(!supertonic.downloadable);
        assert!(supertonic.archive_url.is_empty());
        assert!(supertonic.requires_acceptance);
        assert_eq!(supertonic.license, "OpenRAIL-M");
        assert_eq!(supertonic.source_revision.len(), 40);
        assert_eq!(supertonic.license_file, "MODEL-LICENSE");
        assert_eq!(model_license_text(supertonic).unwrap().len(), 15_007);
    }
}

#[test]
fn lookup_and_activation_populate_config() {
    assert!(model("missing").is_none());
    let spec = model("supertonic-3-int8").unwrap();
    let mut config = Config::default();
    config.backend.kind = "other".into();
    config.model.directory = "/custom".into();
    config.model.voice = 9;
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "supertonic");
    assert_eq!(config.model.family, "supertonic");
    assert_eq!(config.model.name, spec.name);
    assert!(config.model.directory.is_empty());
    assert_eq!(config.model.voice, 0);

    let supertonic = model("supertonic-3-int8").unwrap();
    assert!(!supertonic.npu_capable);
    supertonic.activate(&mut config);
    assert_eq!(config.model.family, "supertonic");
    assert_eq!(
        config.model.duration_predictor,
        "duration_predictor.int8.onnx"
    );
    assert_eq!(config.model.language, "en");
    assert_eq!(config.model.steps, 5);
    assert_eq!(config.model.voice_style, "voice.bin");

    let npu = model("supertonic-3-npu").unwrap();
    assert!(npu.npu_capable);
    assert_eq!(npu.supplemental_files.len(), 1);
    npu.activate(&mut config);
    assert_eq!(config.model.name, "supertonic-3-npu");
    assert_eq!(config.model.vector_estimator, "vector_estimator.onnx");
}
