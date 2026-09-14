use super::*;
use std::fs;

#[test]
fn initialized_ort_core_cannot_silently_change_during_reload() {
    let core = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/libonnxruntime.so");
    if !core.is_file() {
        return;
    }
    initialize_ort(&core).unwrap();
    initialize_ort(&core.canonicalize().unwrap()).unwrap();
    let other = env::temp_dir().join(format!("omaspeak-other-core-{}", std::process::id()));
    fs::write(&other, b"different candidate").unwrap();
    let error = initialize_ort(&other).unwrap_err().to_string();
    assert!(error.contains("restart the process"), "{error}");
    fs::remove_file(other).unwrap();
}

#[test]
fn candidate_transactions_preserve_bytes_until_successful_apply() {
    let root = env::temp_dir().join(format!("omaspeak-runtime-apply-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let original = b"# Keep this comment\n[backend]\ndevice = 'cpu'\n";
    fs::write(&path, original).unwrap();
    let mut candidate = Config::load(&path).unwrap();
    candidate.backend.device = "auto".into();
    for error in [
        "missing libraries",
        "wrong ABI",
        "provider registration failed",
        "device unavailable",
        "native probe terminated",
    ] {
        assert!(
            apply_with(&candidate, &path, true, |_, _| Probe {
                errors: vec![error.into()],
                ..Default::default()
            })
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }
    let success = |_: &BackendConfig, _: &Path| Probe {
        ready: true,
        loadable: true,
        device_accessible: true,
        ..Default::default()
    };
    apply_with(&candidate, &path, false, success).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    apply_with(&candidate, &path, true, success).unwrap();
    assert_eq!(Config::load(&path).unwrap().backend.device, "auto");
    assert!(!path.with_extension("toml.tmp").exists());
    let missing = BackendConfig {
        onnxruntime_library: Some(root.join("missing")),
        ..Default::default()
    };
    assert!(!probe(&missing, &root.join("absent/config.toml")).ready);
    assert!(!root.join("absent").exists());
}

#[test]
fn inventory_names_and_absent_optional_adapters_are_stable() {
    let path = env::temp_dir().join(format!("omaspeak-inventory-absent-{}", std::process::id()));
    let config = BackendConfig {
        onnxruntime_library: Some(path.join("missing-ort")),
        openvino_library: Some(path.join("missing-openvino")),
        provider_library: Some(path.join("missing-provider")),
        ..Default::default()
    };
    let states = inventory(&config, &path.join("config.toml"));
    assert_eq!(states.len(), 8);
    for state in states {
        assert!(state.supported && !state.probe.ready && !state.configured);
        let value = serde_json::to_value(state).unwrap();
        for key in [
            "supported",
            "discovered",
            "source",
            "configured",
            "loadable",
            "device_accessible",
            "ready",
            "paths",
            "evidence",
            "errors",
            "remediation",
        ] {
            assert!(value.get(key).is_some(), "{key}");
        }
    }
}

#[test]
fn resolved_candidates_keep_runtime_overlay_dirs_without_ambient_loader_paths() {
    let root = env::temp_dir().join(format!("omaspeak-runtime-overlay-{}", std::process::id()));
    let configured = root.join("configured");
    let overlay = root.join("overlay");
    let ambient = root.join("ambient");
    for directory in [&configured, &overlay, &ambient] {
        fs::create_dir_all(directory).unwrap();
    }
    let core = configured.join("libonnxruntime.so.1.29.0");
    fs::write(&core, b"fixture").unwrap();

    let config = BackendConfig {
        library_dirs: vec![configured.clone()],
        onnxruntime_library: Some(core),
        ..Default::default()
    };
    let report = runtime::inspect_with(
        &config,
        &root.join("config.toml"),
        Some(env::join_paths([&overlay]).unwrap()),
        Some(env::join_paths([&ambient]).unwrap()),
        None,
    );
    let exact = resolve_with_locations(&config, report);
    assert!(
        exact
            .library_dirs
            .contains(&configured.canonicalize().unwrap())
    );
    assert!(
        exact
            .library_dirs
            .contains(&overlay.canonicalize().unwrap())
    );
    assert!(
        !exact
            .library_dirs
            .contains(&ambient.canonicalize().unwrap())
    );
}

#[test]
fn cuda_selection_uses_logical_ordinal_and_retains_hardware_identity() {
    let devices = vec![
        (0, 11794, "CUDA ordinal 0, hardware 11794 (GPU)".to_owned()),
        (1, 42, "CUDA ordinal 1, hardware 42 (GPU)".to_owned()),
    ];
    let (available, selected) = cuda_device_evidence(devices, 0).unwrap();
    assert_eq!(available.len(), 2);
    assert_eq!(selected, "CUDA ordinal 0, hardware 11794 (GPU)");
    assert!(cuda_device_evidence(vec![(0, 11794, "GPU 0".into())], 1).is_err());
}

#[test]
fn invalid_external_cuda_provider_fails_at_the_registration_boundary() {
    let core = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/libonnxruntime.so");
    if !core.is_file() {
        return;
    }

    let root = env::temp_dir().join(format!(
        "omaspeak-invalid-cuda-provider-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider = root.join("libonnxruntime_providers_cuda.so");
    fs::write(&provider, b"not a shared library").unwrap();
    let result = child(&BackendConfig {
        runtime: Runtime::Cuda,
        device: "gpu".into(),
        onnxruntime_library: Some(core),
        provider_library: Some(provider),
        library_dirs: vec![root],
        ..Default::default()
    });

    assert!(!result.ready);
    assert!(!result.evidence.provider_registration);
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.contains("register selected CUDA provider library")),
        "{:?}",
        result.errors
    );
}

#[test]
fn child_evidence_and_device_validation_are_testable_without_native_runtimes() {
    let root = env::temp_dir().join(format!("omaspeak-child-evidence-{}", std::process::id()));
    let ort = root.join("libonnxruntime.so");
    let provider = root.join("libonnxruntime_providers_cuda.so");
    let openvino = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");

    let mut config = BackendConfig {
        runtime: Runtime::Openvino,
        device: "gpu".into(),
        openvino_library: Some(openvino.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |paths| {
            assert_eq!(paths.library, openvino);
            assert_eq!(paths.plugins, plugins);
            Ok((
                "OpenVINO fixture".into(),
                vec!["CPU".into(), "GPU.0".into()],
            ))
        },
        |_, _| unreachable!(),
    );
    assert!(result.ready && result.loadable && result.device_accessible);
    assert!(result.evidence.provider_registration);
    assert_eq!(result.evidence.versions, ["OpenVINO fixture"]);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("GPU"));

    config.device = "auto".into();
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), Vec::new())),
        |_, _| unreachable!(),
    );
    assert!(result.errors[0].contains("no accessible devices"));
    config.device = "npu".into();
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), vec!["CPU".into()])),
        |_, _| unreachable!(),
    );
    assert!(result.errors[0].contains("requested device NPU is unavailable"));

    config.runtime = Runtime::Default;
    config.device = "cpu".into();
    config.onnxruntime_library = Some(ort.clone());
    let result = child_with(
        &config,
        |_| unreachable!(),
        |actual, provider| {
            assert_eq!(actual, ort);
            assert!(provider.is_none());
            Ok(Vec::new())
        },
    );
    assert!(result.ready && !result.evidence.provider_registration);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("cpu"));

    config.runtime = Runtime::Cuda;
    config.device = "gpu".into();
    config.provider_library = Some(provider.clone());
    config.device_id = 1;
    let result = child_with(
        &config,
        |_| unreachable!(),
        |actual, actual_provider| {
            assert_eq!(actual, ort);
            assert_eq!(actual_provider, Some(provider.as_path()));
            Ok(vec![
                (0, 11794, "CUDA ordinal 0, hardware 11794 (GPU)".into()),
                (1, 42, "CUDA ordinal 1, hardware 42 (GPU)".into()),
            ])
        },
    );
    assert!(result.ready && result.evidence.provider_registration);
    assert_eq!(
        result.evidence.selected_device.as_deref(),
        Some("CUDA ordinal 1, hardware 42 (GPU)")
    );

    config.provider_library = None;
    let result = child_with(&config, |_| unreachable!(), |_, _| unreachable!());
    assert!(result.errors[0].contains("missing CUDA provider"));
}
