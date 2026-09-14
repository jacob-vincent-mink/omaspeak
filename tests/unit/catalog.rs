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
        for required in model.required_files {
            assert_eq!(required.sha256.len(), 64);
            assert!(required.size > 0);
        }
    }
}

#[test]
fn lookup_and_activation_populate_config() {
    assert!(model("missing").is_none());
    let spec = model("en_US-lessac-medium").unwrap();
    let mut config = Config::default();
    config.backend.kind = "other".into();
    config.model.directory = "/custom".into();
    config.model.voice = 9;
    spec.activate(&mut config);
    assert_eq!(config.backend.kind, "sherpa-onnx");
    assert_eq!(config.model.family, "piper");
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
    assert!(config.model.model_file.is_empty());
}
