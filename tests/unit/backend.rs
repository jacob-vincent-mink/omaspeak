use super::*;

#[test]
fn defaults_to_cpu_runtime() {
    let config = BackendConfig::default();
    assert_eq!(config.runtime, Runtime::Default);
    assert_eq!(config.canonical_device().unwrap(), "auto");
    assert!(config.validate_shape().is_ok());
    assert_eq!(config.kind, "audiocpp");
    assert_eq!(
        supported_capabilities(),
        &["cpu", "cuda", "vulkan", "hip", "openvino"]
    );
}

#[test]
fn validates_runtime_device_matrix() {
    for (runtime, accepted) in [
        (Runtime::Default, &["auto", "CPU"][..]),
        (Runtime::Cuda, &["auto", "GPU"][..]),
        (Runtime::Vulkan, &["auto", "GPU"][..]),
        (Runtime::Hip, &["auto", "GPU"][..]),
        (Runtime::Openvino, &["auto", "npu", "GPU", "cpu"][..]),
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
    assert!(canonical_device(Runtime::Vulkan, "cpu").is_err());
    assert!(canonical_device(Runtime::Openvino, "tpu").is_err());

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
