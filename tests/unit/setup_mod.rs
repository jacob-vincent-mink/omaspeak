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
    assert_eq!(created.model.name, "en_US-lessac-medium");
    let loaded = ensure_config(&paths.config_file).unwrap();
    assert_eq!(loaded.backend.kind, "sherpa-onnx");
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
    let present = checks(&paths.config_file, &paths);
    assert!(
        present
            .iter()
            .any(|check| check.name == "model" && check.ok)
    );
    assert!(
        present
            .iter()
            .any(|check| check.name == "engine" && !check.ok)
    );
}

#[test]
fn check_printers_fail_when_remediation_is_required() {
    let (_, paths) = fixture("print");
    ensure_config(&paths.config_file).unwrap();
    assert!(print_checks(&paths.config_file, &paths, false).is_err());
    assert!(print_checks(&paths.config_file, &paths, true).is_err());
    assert!(print_checks_event(&paths.config_file, &paths).is_err());
    print_runtime(false).unwrap();
    print_runtime(true).unwrap();
}
