use super::*;
use crate::backend::BackendConfig;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "omaspeak-runtime-test-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn shared_library(directory: &Path, name: &str) -> PathBuf {
    let source = directory.join(format!("{name}.c"));
    let library = directory.join(name);
    std::fs::write(&source, "int oma_runtime_fixture(void) { return 1; }\n").unwrap();
    let output = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    library
}

#[test]
fn paths_merge_in_stable_precedence_and_report_missing_entries() {
    let root = fixture("merge");
    let config_dir = root.join("config");
    let configured = config_dir.join("configured");
    let from_env = root.join("from-env");
    let package = root.join("package/lib");
    for directory in [&configured, &from_env, &package] {
        std::fs::create_dir_all(directory).unwrap();
    }
    std::fs::write(package.join("libopenvino_c.so"), b"anchor").unwrap();
    let config = BackendConfig {
        library_dirs: vec![PathBuf::from("configured"), root.join("missing")],
        ..Default::default()
    };
    let environment = env::join_paths([&from_env, &configured]).unwrap();
    let report = inspect_with(
        &config,
        &config_dir.join("config.toml"),
        Some(environment),
        None,
        Some(&root.join("package/omaspeak")),
    );
    assert_eq!(
        report.effective_library_dirs,
        vec![configured, from_env, package]
    );
    assert_eq!(report.missing_library_dirs, vec![root.join("missing")]);
}

#[test]
fn effective_path_excludes_the_ambient_loader_path() {
    let root = fixture("effective");
    let owned = root.join("owned");
    std::fs::create_dir_all(&owned).unwrap();
    let config = BackendConfig {
        library_dirs: vec![owned.clone()],
        ..Default::default()
    };
    let report = inspect_with(
        &config,
        &root.join("config.toml"),
        None,
        Some(OsString::from("/ambient/only")),
        None,
    );
    assert_eq!(
        env::split_paths(&effective_library_path(&report).unwrap().unwrap()).collect::<Vec<_>>(),
        vec![owned]
    );
}

#[test]
fn invalid_explicit_paths_prevent_loader_path_generation() {
    let root = fixture("missing");
    let config = BackendConfig {
        library_dirs: vec![root.join("absent")],
        ..Default::default()
    };
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert!(augmented_loader_path(&report).is_err());
    assert!(effective_library_path(&report).is_err());
}

#[test]
fn empty_configured_path_is_invalid_instead_of_resolving_to_the_config_directory() {
    let root = fixture("empty-config-path");
    let config = BackendConfig {
        library_dirs: vec![PathBuf::new()],
        ..Default::default()
    };
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert_eq!(report.missing_library_dirs, vec![PathBuf::new()]);
    assert!(report.effective_library_dirs.is_empty());
}

#[test]
fn relative_environment_path_is_rejected_for_stable_service_use() {
    let root = fixture("relative-env");
    std::fs::create_dir_all(root.join("relative")).unwrap();
    let report = inspect_with(
        &BackendConfig::default(),
        &root.join("config.toml"),
        Some(OsString::from("relative")),
        None,
        None,
    );
    assert_eq!(report.missing_library_dirs, vec![PathBuf::from("relative")]);
    assert!(report.effective_library_dirs.is_empty());
}

#[test]
fn loader_path_planning_only_adds_missing_owned_directories() {
    let root = fixture("augment");
    let owned = root.join("owned");
    let ambient = root.join("ambient");
    std::fs::create_dir_all(&owned).unwrap();
    std::fs::create_dir_all(&ambient).unwrap();
    let report = LibraryPathReport {
        configured_library_dirs: vec![owned.clone()],
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: vec![owned.clone()],
        missing_library_dirs: Vec::new(),
        onnxruntime_library: None,
        provider_library: None,
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: BTreeMap::new(),
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation: Vec::new(),
    };
    let joined = augmented_loader_path_with(&report, Some(env::join_paths([&ambient]).unwrap()))
        .unwrap()
        .unwrap();
    assert_eq!(
        env::split_paths(&joined).collect::<Vec<_>>(),
        vec![owned.clone(), ambient]
    );
    assert!(
        augmented_loader_path_with(&report, Some(env::join_paths([owned]).unwrap()))
            .unwrap()
            .is_none()
    );
}

#[test]
fn reexec_sentinel_prevents_a_loop_and_rejects_an_incomplete_environment() {
    let root = fixture("sentinel");
    let owned = root.join("owned");
    std::fs::create_dir_all(&owned).unwrap();
    let report = LibraryPathReport {
        configured_library_dirs: vec![owned.clone()],
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: vec![owned.clone()],
        missing_library_dirs: Vec::new(),
        onnxruntime_library: None,
        provider_library: None,
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: BTreeMap::new(),
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation: Vec::new(),
    };
    assert!(
        reexec_loader_path_with(&report, Some("1".into()), None)
            .unwrap_err()
            .to_string()
            .contains(REEXEC_SENTINEL)
    );
    assert!(
        reexec_loader_path_with(
            &report,
            Some("1".into()),
            Some(env::join_paths([owned]).unwrap()),
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn package_paths_cover_owned_layouts_and_ignore_executable_directory_debris() {
    let archive = fixture("archive-layout");
    std::fs::create_dir_all(archive.join("lib")).unwrap();
    std::fs::write(archive.join("lib/libopenvino_c.so.1"), b"anchor").unwrap();
    assert_eq!(
        package_library_dirs(Some(&archive.join("omaspeak"))),
        vec![archive.join("lib")]
    );

    let developer = fixture("developer-layout");
    std::fs::write(developer.join("libonnxruntime.so"), b"test core").unwrap();
    std::fs::write(developer.join("libonnxruntime_providers_cuda.so"), b"").unwrap();
    assert!(package_library_dirs(Some(&developer.join("omaspeak"))).is_empty());
    let report = inspect_with(
        &BackendConfig {
            runtime: Runtime::Cuda,
            ..Default::default()
        },
        &developer.join("config.toml"),
        None,
        None,
        Some(&developer.join("omaspeak")),
    );
    assert!(report.package_library_dirs.is_empty());
    assert_ne!(
        report.provider_library.as_deref(),
        Some(developer.join("libonnxruntime_providers_cuda.so").as_path())
    );

    let prefix = fixture("system-layout");
    std::fs::create_dir_all(prefix.join("bin")).unwrap();
    std::fs::create_dir_all(prefix.join("lib/omaspeak")).unwrap();
    std::fs::write(prefix.join("lib/omaspeak/libopenvino_c.so"), b"anchor").unwrap();
    assert_eq!(
        package_library_dirs(Some(&prefix.join("bin/omaspeak"))),
        vec![prefix.join("lib/omaspeak")]
    );

    let empty = fixture("empty-layout");
    std::fs::create_dir_all(empty.join("lib")).unwrap();
    assert!(package_library_dirs(Some(&empty.join("omaspeak"))).is_empty());
    assert!(package_library_dirs(None).is_empty());
}

#[test]
fn direct_openvino_stack_is_independent_and_resolves_exact_files() {
    let root = fixture("external-stack");
    let native = root.join("native");
    std::fs::create_dir_all(&native).unwrap();
    let ort = shared_library(&native, "libonnxruntime.so");
    let openvino = shared_library(&native, "libopenvino_c.so");
    let plugins = native.join("plugins.xml");
    std::fs::write(&plugins, "<ie><plugins/></ie>").unwrap();
    let config_file = root.join("config/config.toml");
    std::fs::create_dir_all(config_file.parent().unwrap()).unwrap();
    let config = BackendConfig {
        runtime: Runtime::Openvino,
        library_dirs: vec![PathBuf::from("../native")],
        onnxruntime_library: Some(PathBuf::from("../native/libonnxruntime.so")),
        openvino_library: Some(PathBuf::from("../native/libopenvino_c.so")),
        openvino_plugins: Some(PathBuf::from("../native/plugins.xml")),
        ..Default::default()
    };
    let report = inspect_with(&config, &config_file, None, None, None);
    assert_eq!(report.onnxruntime_library.as_deref(), Some(ort.as_path()));
    assert_eq!(report.provider_library, None);
    assert_eq!(report.openvino_library.as_deref(), Some(openvino.as_path()));
    assert_eq!(report.openvino_plugins.as_deref(), Some(plugins.as_path()));
    assert_eq!(report.runtime_loadable.get("default"), Some(&true));
    assert_eq!(report.runtime_loadable.get("openvino"), Some(&true));
    assert_eq!(report.runtime_loadable.get("cuda"), Some(&false));
    assert!(report.remediation(Runtime::Openvino).is_none());
    assert!(report.remediation(Runtime::Cuda).unwrap().contains("cuda"));

    let resolved_ort =
        resolve_onnx_runtime(&config, &config_file, &report, Runtime::Default).unwrap();
    assert_eq!(resolved_ort.onnxruntime, ort);
    assert_eq!(resolved_ort.provider, None);

    let resolved = resolve_openvino_runtime(&config, &config_file, &report).unwrap();
    assert_eq!(resolved.library, openvino);
    assert_eq!(resolved.plugins, plugins);
}

#[test]
fn versioned_library_search_is_deterministic() {
    let root = fixture("versions");
    std::fs::write(root.join("libexample.so.1"), b"one").unwrap();
    std::fs::write(root.join("libexample.so.20"), b"twenty").unwrap();
    assert_eq!(
        find_versioned_library(std::slice::from_ref(&root), "libexample.so"),
        Some(root.join("libexample.so.20"))
    );
}

#[test]
fn direct_runtime_resolution_covers_success_and_actionable_failures() {
    let root = fixture("direct-resolution");
    let native = root.join("native");
    std::fs::create_dir_all(&native).unwrap();
    let ort = shared_library(&native, "libonnxruntime.so");
    let cuda = shared_library(&native, "libonnxruntime_providers_cuda.so");
    let openvino = shared_library(&native, "libopenvino_c.so");
    let plugins = native.join("plugins.xml");
    std::fs::write(&plugins, "<ie><plugins/></ie>").unwrap();
    let config_file = root.join("config/config.toml");
    std::fs::create_dir_all(config_file.parent().unwrap()).unwrap();
    let mut config = BackendConfig {
        library_dirs: vec![native.clone()],
        onnxruntime_library: Some(ort.clone()),
        provider_library: Some(cuda.clone()),
        openvino_library: Some(openvino.clone()),
        openvino_plugins: Some(plugins.clone()),
        ..Default::default()
    };
    let report = inspect_with(&config, &config_file, None, None, None);

    assert!(resolve_onnx_runtime(&config, &config_file, &report, Runtime::Openvino).is_err());
    let cpu = resolve_onnx_runtime(&config, &config_file, &report, Runtime::Default).unwrap();
    assert_eq!(cpu.onnxruntime, ort);
    assert!(cpu.provider.is_none());
    let gpu = resolve_onnx_runtime(&config, &config_file, &report, Runtime::Cuda).unwrap();
    assert_eq!(gpu.provider.as_deref(), Some(cuda.as_path()));
    let ov = resolve_openvino_runtime(&config, &config_file, &report).unwrap();
    assert_eq!(ov.library, openvino);
    assert_eq!(ov.plugins, plugins);

    let mut missing_report = report.clone();
    missing_report
        .missing_library_dirs
        .push(root.join("missing"));
    assert!(
        resolve_onnx_runtime(&config, &config_file, &missing_report, Runtime::Default).is_err()
    );
    assert!(resolve_openvino_runtime(&config, &config_file, &missing_report).is_err());

    config.onnxruntime_library = Some(PathBuf::from("missing.so"));
    assert!(resolve_onnx_runtime(&config, &config_file, &report, Runtime::Default).is_err());
    config.onnxruntime_library = None;
    let mut empty_report = report.clone();
    empty_report.onnxruntime_library = None;
    assert!(resolve_onnx_runtime(&config, &config_file, &empty_report, Runtime::Default).is_err());

    config.provider_library = None;
    empty_report.onnxruntime_library = report.onnxruntime_library.clone();
    empty_report.provider_library = None;
    assert!(resolve_onnx_runtime(&config, &config_file, &empty_report, Runtime::Cuda).is_err());

    config.openvino_library = Some(PathBuf::from("missing-openvino.so"));
    assert!(resolve_openvino_runtime(&config, &config_file, &report).is_err());
    config.openvino_library = None;
    config.openvino_plugins = Some(PathBuf::from("missing-plugins.xml"));
    assert!(resolve_openvino_runtime(&config, &config_file, &report).is_err());
    config.openvino_plugins = None;
    let mut missing_openvino = report.clone();
    missing_openvino.openvino_library = None;
    assert!(resolve_openvino_runtime(&config, &config_file, &missing_openvino).is_err());
    missing_openvino.openvino_library = report.openvino_library.clone();
    missing_openvino.openvino_plugins = None;
    assert!(resolve_openvino_runtime(&config, &config_file, &missing_openvino).is_err());
}

#[test]
fn remediation_covers_runtime_specific_probe_and_missing_path_messages() {
    let mut report = LibraryPathReport {
        configured_library_dirs: Vec::new(),
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: Vec::new(),
        missing_library_dirs: Vec::new(),
        onnxruntime_library: None,
        provider_library: None,
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: BTreeMap::from([("default", true), ("openvino", false), ("cuda", false)]),
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation: Vec::new(),
    };
    assert!(report.remediation(Runtime::Default).is_none());
    assert!(
        report
            .remediation(Runtime::Openvino)
            .unwrap()
            .contains("plugins")
    );
    assert!(
        report
            .remediation(Runtime::Cuda)
            .unwrap()
            .contains("matching external")
    );
    report.runtime_probe_errors.insert("cuda", "bad ABI".into());
    assert!(
        report
            .remediation(Runtime::Cuda)
            .unwrap()
            .contains("bad ABI")
    );
    report.missing_library_dirs.push(PathBuf::from("relative"));
    assert!(
        report
            .remediation(Runtime::Cuda)
            .unwrap()
            .contains("non-absolute")
    );
}

#[test]
fn runtime_probes_record_load_and_device_results_without_native_dependencies() {
    let root = fixture("probe-injection");
    let ort = root.join("libonnxruntime.so");
    let provider = root.join("libonnxruntime_providers_cuda.so");
    let openvino = root.join("libopenvino_c.so");
    let plugins = root.join("plugins.xml");
    for path in [&ort, &provider, &openvino, &plugins] {
        std::fs::write(path, b"fixture").unwrap();
    }
    let base = || LibraryPathReport {
        configured_library_dirs: vec![root.clone()],
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: vec![root.clone()],
        missing_library_dirs: Vec::new(),
        onnxruntime_library: Some(ort.clone()),
        provider_library: Some(provider.clone()),
        openvino_library: Some(openvino.clone()),
        openvino_plugins: Some(plugins.clone()),
        runtime_loadable: BTreeMap::from([("default", true), ("openvino", true), ("cuda", true)]),
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation: Vec::new(),
    };

    let mut cpu = base();
    probe_installed_runtimes_with(
        &BackendConfig::default(),
        &mut cpu,
        |paths, runtime| {
            assert_eq!(paths.onnxruntime, ort);
            assert_eq!(runtime, Runtime::Default);
            Ok(())
        },
        |_, _| Ok(vec!["CPU".into()]),
    );
    assert_eq!(cpu.runtime_loadable.get("default"), Some(&true));

    let mut cuda_config = BackendConfig {
        runtime: Runtime::Cuda,
        device: "gpu".into(),
        ..Default::default()
    };
    let mut failed = base();
    probe_installed_runtimes_with(
        &cuda_config,
        &mut failed,
        |_, runtime| anyhow::bail!("{runtime:?} probe failed"),
        |_, _| anyhow::bail!("OpenVINO probe failed"),
    );
    assert_eq!(failed.runtime_loadable.get("default"), Some(&false));
    assert_eq!(failed.runtime_loadable.get("cuda"), Some(&false));
    assert_eq!(failed.runtime_loadable.get("openvino"), Some(&false));
    assert!(failed.runtime_probe_errors["cuda"].contains("probe failed"));

    cuda_config.runtime = Runtime::Openvino;
    cuda_config.device = "npu".into();
    let mut inaccessible = base();
    probe_installed_runtimes_with(
        &cuda_config,
        &mut inaccessible,
        |_, _| Ok(()),
        |paths, device| {
            assert_eq!(paths.library, openvino);
            assert_eq!(paths.plugins, plugins);
            assert_eq!(device, "auto");
            Ok(vec!["CPU".into(), "GPU".into()])
        },
    );
    assert_eq!(
        inaccessible.runtime_device_accessible.get("openvino"),
        Some(&false)
    );
    assert!(inaccessible.device_probe_errors["openvino"].contains("NPU"));

    cuda_config.device = "auto".into();
    let mut accessible = base();
    probe_installed_runtimes_with(
        &cuda_config,
        &mut accessible,
        |_, _| Ok(()),
        |_, _| Ok(vec!["CPU".into()]),
    );
    assert_eq!(
        accessible.runtime_device_accessible.get("openvino"),
        Some(&true)
    );
}
