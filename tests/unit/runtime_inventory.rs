use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

fn temp(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "omaspeak-inventory-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn resolves_one_complete_audio_provider() {
    let root = temp("resolve-audio");
    let library = root.join("libaudiocpp.so");
    fs::write(&library, b"provider").unwrap();
    let config = BackendConfig {
        runtime: Runtime::Vulkan,
        device: "gpu".into(),
        library: Some(library.clone()),
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let exact = resolve(&config, &root.join("config.toml"));
    assert_eq!(exact.library, Some(library.canonicalize().unwrap()));
    assert!(exact.openvino_library.is_none());
    assert!(exact.openvino_plugins.is_none());
    assert!(exact.library_dirs.contains(&root.canonicalize().unwrap()));
}

#[test]
fn resolves_openvino_without_audio_provider() {
    let root = temp("resolve-openvino");
    let library = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&plugins, b"plugins").unwrap();
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        library: Some(root.join("libaudiocpp.so")),
        openvino_library: Some(library.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let exact = resolve(&config, &root.join("config.toml"));
    assert!(exact.library.is_none());
    assert_eq!(
        exact.openvino_library,
        Some(library.canonicalize().unwrap())
    );
    assert_eq!(
        exact.openvino_plugins,
        Some(plugins.canonicalize().unwrap())
    );
}

#[test]
fn audio_child_reports_abi_only_until_model_proof() {
    let root = temp("audio-child");
    let library = root.join("libaudiocpp.so");
    fs::write(&library, b"fixture").unwrap();
    let config = BackendConfig {
        runtime: Runtime::Cuda,
        device: "gpu".into(),
        library: Some(library.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |_| unreachable!(),
        |candidate| {
            assert_eq!(candidate.runtime, Runtime::Cuda);
            Ok(library.clone())
        },
    );
    assert!(result.loadable);
    assert!(result.ready);
    assert!(!result.device_accessible);
    assert!(!result.evidence.model_inference_verified);
    assert_eq!(result.evidence.provider_path, Some(library));
    assert_eq!(result.evidence.selected_device.as_deref(), Some("gpu"));
}

#[test]
fn openvino_child_reports_selected_accessible_device() {
    let root = temp("openvino-child");
    let library = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");
    fs::write(&library, b"runtime").unwrap();
    fs::write(&plugins, b"plugins").unwrap();
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        openvino_library: Some(library.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |paths| {
            assert_eq!(paths.library, library);
            assert_eq!(paths.plugins, plugins);
            Ok(("OpenVINO fixture".into(), vec!["CPU".into(), "NPU".into()]))
        },
        |_| unreachable!(),
    );
    assert!(result.ready && result.loadable && result.device_accessible);
    assert_eq!(result.evidence.selected_device.as_deref(), Some("NPU"));
    assert_eq!(result.evidence.available_devices, ["CPU", "NPU"]);
}

#[test]
fn openvino_child_rejects_unavailable_device() {
    let config = BackendConfig {
        kind: "supertonic".into(),
        runtime: Runtime::Openvino,
        device: "npu".into(),
        openvino_library: Some("/runtime/libopenvino_c.so".into()),
        openvino_plugins: Some("/runtime/plugins.xml".into()),
        ..Default::default()
    };
    let result = child_with(
        &config,
        |_| Ok(("OpenVINO fixture".into(), vec!["CPU".into()])),
        |_| unreachable!(),
    );
    assert!(!result.ready);
    assert!(result.errors[0].contains("unavailable"));
}

#[test]
fn apply_only_saves_ready_candidates() {
    let root = temp("apply");
    let path = root.join("config.toml");
    let config = Config::default();
    let ready = Probe {
        loadable: true,
        ready: true,
        ..Default::default()
    };
    apply_with(&config, &path, true, |_, _| ready.clone()).unwrap();
    assert!(path.is_file());

    fs::remove_file(&path).unwrap();
    let rejected = Probe {
        errors: vec!["no provider".into()],
        ..Default::default()
    };
    assert!(apply_with(&config, &path, true, |_, _| rejected).is_err());
    assert!(!path.exists());
}

#[test]
fn inventory_lists_every_native_runtime_device_pair() {
    let root = temp("matrix");
    let config = BackendConfig {
        library: Some(root.join("missing-libaudiocpp.so")),
        ..Default::default()
    };
    let states = inventory(&config, &root.join("config.toml"));
    let pairs = states
        .iter()
        .map(|state| (state.runtime, state.device.as_str()))
        .collect::<Vec<_>>();
    assert!(pairs.contains(&("default", "cpu")));
    assert!(pairs.contains(&("cuda", "gpu")));
    assert!(pairs.contains(&("vulkan", "gpu")));
    assert!(pairs.contains(&("hip", "gpu")));
    assert!(pairs.contains(&("openvino", "npu")));
}

#[test]
fn candidate_source_precedence_is_deterministic() {
    assert_eq!(candidate_source(true, true, true, true), "configured");
    assert_eq!(candidate_source(false, true, true, true), "environment");
    assert_eq!(candidate_source(false, false, true, true), "package");
    assert_eq!(candidate_source(false, false, false, true), "system");
    assert_eq!(candidate_source(false, false, false, false), "candidate");
}

#[test]
fn probe_rejects_invalid_dependency_directories_before_native_load() {
    let root = temp("invalid-dirs");
    let provider = root.join("libaudiocpp.so");
    fs::write(&provider, b"fixture").unwrap();
    let config = BackendConfig {
        library: Some(provider),
        library_dirs: vec![root.join("missing")],
        ..Default::default()
    };
    let result = probe(&config, &root.join("config.toml"));
    assert!(!result.ready);
    assert!(result.errors[0].contains("invalid dependency directory"));
}

#[test]
fn evidence_defaults_do_not_claim_inference_or_devices() {
    let evidence = Evidence::default();
    assert!(!evidence.model_inference_verified);
    assert!(evidence.provider_path.is_none());
    assert_eq!(BTreeMap::<String, String>::new().len(), 0);
}
