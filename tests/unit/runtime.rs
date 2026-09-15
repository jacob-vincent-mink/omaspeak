use super::*;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

fn temp(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "omaspeak-runtime-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn finds_complete_audio_provider_and_natural_latest_version() {
    let root = temp("audio");
    fs::write(root.join("libaudiocpp.so.9"), b"nine").unwrap();
    fs::write(root.join("libaudiocpp.so.10"), b"ten").unwrap();
    let config = BackendConfig {
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert_eq!(
        report.audiocpp_library.as_deref(),
        Some(root.join("libaudiocpp.so.10").as_path())
    );
    for runtime in ["default", "cuda", "vulkan", "hip"] {
        assert_eq!(report.runtime_loadable.get(runtime), Some(&true));
    }
}

#[test]
fn exact_audio_provider_and_missing_directories_are_reported() {
    let root = temp("exact");
    let library = root.join("libaudiocpp.so.0.1.0");
    fs::write(&library, b"provider").unwrap();
    let config = BackendConfig {
        library: Some(library.clone()),
        library_dirs: vec![root.join("missing")],
        ..Default::default()
    };
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert_eq!(
        report.audiocpp_library,
        Some(library.canonicalize().unwrap())
    );
    assert_eq!(report.missing_library_dirs, [root.join("missing")]);
    assert_eq!(report.runtime_loadable.get("default"), Some(&false));
    assert!(
        report
            .remediation(Runtime::Default)
            .unwrap()
            .contains("missing")
    );
}

#[test]
fn direct_openvino_requires_library_and_plugins() {
    let root = temp("openvino");
    let library = root.join("libopenvino_c.so.2600");
    let plugins = root.join("plugins.xml");
    fs::write(&library, b"openvino").unwrap();
    fs::write(&plugins, b"plugins").unwrap();
    let mut config = BackendConfig {
        runtime: Runtime::Openvino,
        library_dirs: vec![root.clone()],
        ..Default::default()
    };
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert_eq!(report.runtime_loadable.get("openvino"), Some(&true));
    let resolved = resolve_openvino_runtime(&config, &root.join("config.toml"), &report).unwrap();
    assert_eq!(resolved.library, library.canonicalize().unwrap());
    assert_eq!(resolved.plugins, plugins.canonicalize().unwrap());

    fs::remove_file(&plugins).unwrap();
    config.openvino_plugins = Some(plugins);
    let report = inspect_with(&config, &root.join("config.toml"), None, None, None);
    assert!(resolve_openvino_runtime(&config, &root.join("config.toml"), &report).is_err());
}

#[test]
fn package_and_environment_directories_are_visible() {
    let root = temp("layouts");
    let bin = root.join("bin");
    let lib = bin.join("lib");
    let env = root.join("external");
    fs::create_dir_all(&lib).unwrap();
    fs::create_dir_all(&env).unwrap();
    fs::write(lib.join("libaudiocpp.so"), b"package").unwrap();
    let report = inspect_with(
        &BackendConfig::default(),
        &root.join("config.toml"),
        Some(env.as_os_str().to_owned()),
        None,
        Some(&bin.join("omaspeak")),
    );
    assert_eq!(report.package_library_dirs, [lib.canonicalize().unwrap()]);
    assert_eq!(
        report.environment_library_dirs,
        [env.canonicalize().unwrap()]
    );
    assert!(report.audiocpp_library.is_some());
}

#[test]
fn loader_path_adds_app_owned_directories_once() {
    let root = temp("loader").canonicalize().unwrap();
    let report = LibraryPathReport {
        configured_library_dirs: vec![root.clone()],
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: vec![root.clone()],
        missing_library_dirs: Vec::new(),
        audiocpp_library: None,
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: BTreeMap::new(),
        runtime_probe_errors: BTreeMap::new(),
        runtime_device_accessible: BTreeMap::new(),
        device_probe_errors: BTreeMap::new(),
        remediation: Vec::new(),
    };
    assert_eq!(
        augmented_loader_path_with(&report, None).unwrap(),
        Some(root.as_os_str().to_owned())
    );
    assert!(
        augmented_loader_path_with(&report, Some(root.as_os_str().to_owned()))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        effective_library_path(&report).unwrap(),
        Some(root.into_os_string())
    );
}

#[test]
fn runtime_names_and_provider_families_are_stable() {
    assert_eq!(Runtime::Default.name(), "default");
    assert_eq!(Runtime::Cuda.name(), "cuda");
    assert_eq!(Runtime::Vulkan.name(), "vulkan");
    assert_eq!(Runtime::Hip.name(), "hip");
    assert_eq!(Runtime::Openvino.name(), "openvino");
    assert!(Runtime::Vulkan.uses_audiocpp());
    assert!(!Runtime::Openvino.uses_audiocpp());
}
