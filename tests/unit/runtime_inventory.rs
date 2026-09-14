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
fn cuda_selection_matches_hardware_id_instead_of_inventory_position() {
    let devices = vec![
        (7, "CUDA hardware 7 (GPU)".to_owned()),
        (2, "CUDA hardware 2 (GPU)".to_owned()),
    ];
    let (available, selected) = cuda_device_evidence(devices, 2).unwrap();
    assert_eq!(available.len(), 2);
    assert_eq!(selected, "CUDA hardware 2 (GPU)");
    assert!(cuda_device_evidence(vec![(7, "GPU 7".into())], 0).is_err());
}
