use super::*;
use std::fs;

fn fixture(name: &str) -> (std::path::PathBuf, AppPaths) {
    let root =
        std::env::temp_dir().join(format!("omaspeak-setup-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let paths = AppPaths {
        config_file: root.join("config/omaspeak/config.toml"),
        data_dir: root.join("data/omaspeak"),
        state_dir: root.join("state/omaspeak"),
        runtime_dir: root.join("run/omaspeak"),
    };
    (root, paths)
}

#[test]
fn ensure_config_creates_and_reloads_defaults() {
    let (_, paths) = fixture("ensure");
    let created = ensure_config(&paths.config_file).unwrap();
    assert_eq!(created.model.name, "supertonic-3-int8");
    let loaded = ensure_config(&paths.config_file).unwrap();
    assert_eq!(loaded.backend.kind, "supertonic");
}

#[test]
fn checks_report_malformed_missing_and_custom_states() {
    let (root, paths) = fixture("checks");
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, "bad = [toml").unwrap();
    let malformed = checks(&paths.config_file, &paths);
    assert_eq!(malformed.len(), 1);
    assert!(!malformed[0].ok);

    let mut config = Config::default();
    config.backend.kind = "future".into();
    config.model.name = "custom".into();
    config.model.directory = root.join("custom-model").display().to_string();
    config.save(&paths.config_file).unwrap();
    let missing = checks(&paths.config_file, &paths);
    assert!(
        missing
            .iter()
            .any(|check| check.name == "backend" && !check.ok)
    );
    assert!(
        missing
            .iter()
            .any(|check| check.name == "model" && !check.ok)
    );
    fs::create_dir_all(&config.model.directory).unwrap();
    config.model.voice_style = "voice.bin".into();
    config.model.voice = 0;
    fs::write(
        std::path::Path::new(&config.model.directory).join("voice.bin"),
        [1_i64, 1, 1, 1, 1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    config.save(&paths.config_file).unwrap();
    let present = checks(&paths.config_file, &paths);
    assert!(
        present
            .iter()
            .any(|check| check.name == "model" && check.ok)
    );
    assert!(
        present
            .iter()
            .any(|check| check.name == "voice" && check.ok)
    );
    assert!(present.iter().any(|check| check.name == "engine"));
    let systemd = present
        .iter()
        .find(|check| check.name == "systemd")
        .unwrap();
    assert!(systemd.ok);
    assert!(systemd.detail.contains("optional") || systemd.detail.contains("active"));
    assert!(systemd.remediation.is_none());
}

#[test]
fn check_printers_fail_when_remediation_is_required() {
    let (_, paths) = fixture("print");
    ensure_config(&paths.config_file).unwrap();
    assert!(print_checks(&paths.config_file, &paths, false).is_err());
    assert!(print_checks(&paths.config_file, &paths, true).is_err());
    assert!(print_checks_event(&paths.config_file, &paths).is_err());
    print_runtime(&paths.config_file, false).unwrap();
    print_runtime(&paths.config_file, true).unwrap();
}

#[test]
fn checks_accept_a_verified_catalog_model_and_initialized_openvino_engine() {
    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        return;
    };
    let source = home.join(".local/share/omaspeak/models/supertonic-3-int8");
    let openvino = std::path::PathBuf::from("/usr/lib/libopenvino_c.so");
    let plugins = std::path::PathBuf::from("/usr/lib/openvino/plugins.xml");
    if !source.join("voice.bin").is_file() || !openvino.is_file() || !plugins.is_file() {
        return;
    }

    let (_, mut paths) = fixture("healthy-openvino");
    paths.data_dir = std::env::current_dir()
        .unwrap()
        .join(format!("target/setup-model-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&paths.data_dir);
    let spec = crate::catalog::model("supertonic-3-int8").unwrap();
    let target = model::model_directory(&paths, spec);
    fs::create_dir_all(&target).unwrap();
    for required in spec.required_files {
        fs::copy(source.join(required.path), target.join(required.path)).unwrap();
    }
    fs::write(
        target.join(spec.license_file),
        crate::catalog::model_license_text(spec).unwrap(),
    )
    .unwrap();

    let mut config = Config::default();
    config.backend.runtime = crate::backend::Runtime::Openvino;
    config.backend.device = "cpu".into();
    config.backend.openvino_library = Some(openvino);
    config.backend.openvino_plugins = Some(plugins);
    config.save(&paths.config_file).unwrap();
    let result = checks(&paths.config_file, &paths);
    for name in [
        "config", "backend", "model", "voice", "runtime", "device", "engine",
    ] {
        let check = result.iter().find(|check| check.name == name).unwrap();
        assert!(check.ok, "{name}: {}", check.detail);
    }
    fs::remove_dir_all(&paths.data_dir).unwrap();
}

#[test]
fn injected_checks_cover_healthy_runtime_device_and_engine_boundaries() {
    let (root, paths) = fixture("injected-healthy");
    let model = root.join("custom-model");
    fs::create_dir_all(&model).unwrap();
    fs::write(
        model.join("voice.bin"),
        [1_i64, 1, 1, 1, 1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut config = Config::default();
    config.backend.runtime = crate::backend::Runtime::Openvino;
    config.backend.device = "cpu".into();
    config.model.name = "custom".into();
    config.model.directory = model.display().to_string();
    config.save(&paths.config_file).unwrap();

    let result = checks_with(&paths.config_file, &paths, |_, _| {
        crate::runtime_inventory::Probe {
            loadable: true,
            device_accessible: true,
            ready: true,
            ..Default::default()
        }
    });
    for name in ["backend", "model", "voice", "runtime", "device", "engine"] {
        let check = result.iter().find(|check| check.name == name).unwrap();
        assert!(check.ok, "{name}: {}", check.detail);
    }

    let failed = checks_with(&paths.config_file, &paths, |_, _| {
        crate::runtime_inventory::Probe {
            errors: vec!["injected device failure; injected engine failure".into()],
            ..Default::default()
        }
    });
    assert!(failed.iter().any(|check| {
        check.name == "device" && !check.ok && check.detail.contains("injected device failure")
    }));
    assert!(failed.iter().any(|check| {
        check.name == "engine" && !check.ok && check.detail.contains("injected engine failure")
    }));
}

#[test]
fn environment_checks_cover_audio_launcher_and_optional_service_states() {
    let (_, paths) = fixture("environment-states");
    let service = systemd::service_path(&paths);
    fs::create_dir_all(service.parent().unwrap()).unwrap();
    fs::write(&service, "[Service]").unwrap();

    let cases = [
        (
            EnvironmentStatus {
                audio_available: true,
                launcher_installed: true,
                systemctl_available: true,
                service_active: true,
            },
            "active:",
        ),
        (
            EnvironmentStatus {
                audio_available: false,
                launcher_installed: false,
                systemctl_available: false,
                service_active: false,
            },
            "systemctl is unavailable",
        ),
        (
            EnvironmentStatus {
                audio_available: true,
                launcher_installed: false,
                systemctl_available: true,
                service_active: false,
            },
            "installed but inactive",
        ),
    ];
    for (environment, expected) in cases {
        let mut result = Vec::new();
        append_environment_checks(&mut result, &paths, environment);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].ok, environment.audio_available);
        assert_eq!(result[1].ok, environment.launcher_installed);
        assert!(result[2].detail.contains(expected));
    }

    fs::remove_file(&service).unwrap();
    let mut result = Vec::new();
    append_environment_checks(
        &mut result,
        &paths,
        EnvironmentStatus {
            audio_available: true,
            launcher_installed: true,
            systemctl_available: true,
            service_active: true,
        },
    );
    assert!(result[2].detail.contains("managed outside"));
    let mut result = Vec::new();
    append_environment_checks(
        &mut result,
        &paths,
        EnvironmentStatus {
            audio_available: false,
            launcher_installed: false,
            systemctl_available: true,
            service_active: false,
        },
    );
    assert!(result[2].detail.contains("not installed"));
}

#[test]
fn check_renderers_and_path_display_handle_successful_results() {
    let successful = vec![ok("runtime", "ready")];
    print_check_results(successful.clone(), false).unwrap();
    print_check_results(successful.clone(), true).unwrap();
    print_check_results_event(successful).unwrap();
    assert_eq!(display_paths(&[]), "(none)");
    assert_eq!(display_paths(&["/one".into(), "/two".into()]), "/one:/two");
}
