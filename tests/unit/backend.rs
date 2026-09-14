use super::*;

#[test]
fn defaults_to_cpu_runtime() {
    let config = BackendConfig::default();
    assert_eq!(config.runtime, Runtime::Default);
    assert_eq!(config.canonical_device().unwrap(), "auto");
    assert!(
        config
            .validate_capabilities(compiled_capabilities())
            .is_ok()
    );
    let expected: &[&str] = match (cfg!(feature = "openvino"), cfg!(feature = "cuda")) {
        (true, true) => &["cpu", "openvino", "cuda"],
        (true, false) => &["cpu", "openvino"],
        (false, true) => &["cpu", "cuda"],
        (false, false) => &["cpu"],
    };
    assert_eq!(compiled_capabilities(), expected);
}

#[test]
fn validates_runtime_device_matrix() {
    for (runtime, accepted) in [
        (Runtime::Default, &["auto", "CPU"][..]),
        (Runtime::Cuda, &["auto", "GPU"][..]),
        (
            Runtime::Openvino,
            &[
                "auto",
                "npu",
                "GPU",
                "cpu",
                "auto:GPU,NPU,CPU",
                "hetero:GPU,CPU",
                "multi:NPU,CPU",
            ][..],
        ),
    ] {
        for device in accepted {
            assert!(
                canonical_device(runtime, device).is_ok(),
                "{runtime:?} {device}"
            );
        }
    }
    assert!(canonical_device(Runtime::Default, "gpu").is_err());
    assert!(canonical_device(Runtime::Cuda, "cpu").is_err());
    assert!(canonical_device(Runtime::Openvino, "hetero:GPU").is_err());
    assert!(canonical_device(Runtime::Openvino, "multi:").is_err());
    assert!(canonical_device(Runtime::Openvino, "auto:TPU").is_err());
    assert!(canonical_device(Runtime::Openvino, "unknown:GPU,CPU").is_err());

    let invalid_device_id = BackendConfig {
        device_id: 1,
        ..Default::default()
    };
    assert_eq!(
        invalid_device_id.validate_shape(),
        Err(BackendError::InvalidDeviceId)
    );
}

#[test]
fn canonicalizes_openvino_provider_syntax() {
    assert_eq!(
        canonical_device(Runtime::Openvino, " hetero:gpu, cpu ").unwrap(),
        "HETERO:GPU,CPU"
    );
    assert_eq!(
        canonical_device(Runtime::Openvino, "auto:npu,gpu").unwrap(),
        "AUTO:NPU,GPU"
    );
}

#[test]
fn acceleration_requires_a_compiled_capability() {
    let config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "npu".into(),
        ..Default::default()
    };
    assert!(matches!(
        config.validate_capabilities(&["cpu"]),
        Err(BackendError::CapabilityUnavailable { .. })
    ));
    assert!(config.validate_capabilities(&["cpu", "openvino"]).is_ok());
}

#[test]
fn validates_provider_option_file_syntax() {
    let mut config = BackendConfig::default();
    config
        .options
        .insert("SessionConfig.mlas.disable_kleidiai".into(), "1".into());
    config
        .options
        .insert("ProfilingFilePrefix".into(), "/tmp/omaspeak profile".into());
    assert!(config.validate_shape().is_ok());

    config.options.insert("bad key".into(), "value".into());
    assert!(matches!(
        config.validate_shape(),
        Err(BackendError::InvalidOptionKey { .. })
    ));
    config.options.remove("bad key");
    config.options.insert("also=bad".into(), "value".into());
    assert!(matches!(
        config.validate_shape(),
        Err(BackendError::InvalidOptionKey { .. })
    ));
    config.options.remove("also=bad");
    config
        .options
        .insert("device_type".into(), "NPU\ncache_dir=/tmp".into());
    assert!(matches!(
        config.validate_shape(),
        Err(BackendError::InvalidOptionValue { .. })
    ));
}

#[test]
fn validates_supertonic_component_allowlists() {
    let mut config = BackendConfig::default();
    for value in [
        "",
        "all",
        "duration_predictor",
        "duration_predictor,text_encoder,vocoder",
    ] {
        config
            .options
            .insert("SherpaOnnx.SupertonicComponents".into(), value.into());
        config.validate_shape().unwrap();
    }
    for value in [
        "duration_predictor,unknown",
        "duration_predictor, duration_predictor",
        "duration_predictor,duration_predictor",
        "all,vocoder",
    ] {
        config
            .options
            .insert("SherpaOnnx.SupertonicComponents".into(), value.into());
        assert_eq!(
            config.validate_shape(),
            Err(BackendError::InvalidSupertonicComponents)
        );
    }
}
