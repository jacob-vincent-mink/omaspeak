use super::*;

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("omaspeak-config-{}-{name}", std::process::id()))
}

#[test]
fn partial_config_uses_defaults() {
    let cfg: Config = toml::from_str(
        r#"
                [model]
                voice = 2
                [backend]
                runtime = "default"
            "#,
    )
    .unwrap();
    assert_eq!(cfg.model.voice, 2);
    assert_eq!(cfg.model.family, "supertonic");
    assert_eq!(cfg.model.language, "en");
    assert_eq!(cfg.model.steps, 5);
    assert_eq!(cfg.model.duration_predictor, "duration_predictor.int8.onnx");
    assert!(cfg.model.file.is_empty());
    assert_eq!(cfg.backend.threads, 2);
    assert!(cfg.backend.library.is_none());
}

#[test]
fn missing_save_load_and_model_paths_round_trip() {
    let root = temp("round-trip");
    let path = root.join("nested/config.toml");
    let _ = fs::remove_dir_all(&root);
    let defaults = Config::load(&path).unwrap();
    assert_eq!(defaults.audio.device, "default");
    assert_eq!(defaults.audio.volume, 1.0);
    assert_eq!(defaults.daemon.queue_capacity, 8);
    assert_eq!(defaults.daemon.max_text_bytes, 65_536);
    defaults.save(&path).unwrap();
    assert_eq!(Config::load(&path).unwrap().model.name, defaults.model.name);

    let paths = AppPaths {
        config_file: path.clone(),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    assert_eq!(
        defaults.model_directory(&paths),
        root.join("data/models/supertonic-3-int8")
    );
    let mut custom = defaults;
    custom.model.directory = root.join("custom").display().to_string();
    assert_eq!(custom.model_directory(&paths), root.join("custom"));
}

#[test]
fn malformed_and_unknown_config_is_rejected() {
    let root = temp("invalid");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    fs::write(&path, "not = [valid").unwrap();
    assert!(Config::load(&path).is_err());
    fs::write(&path, "unknown = true").unwrap();
    assert!(Config::load(&path).is_err());

    let target = root.join("directory-as-config");
    fs::create_dir_all(&target).unwrap();
    assert!(Config::default().save(&target).is_err());
}

#[test]
fn config_io_failures_identify_the_failed_operation() {
    let root = temp("io-errors");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    let directory = root.join("directory");
    fs::create_dir(&directory).unwrap();
    let error = Config::load(&directory).unwrap_err();
    assert!(error.to_string().contains("read config"));

    let blocked_parent = root.join("blocked-parent");
    fs::write(&blocked_parent, "not a directory").unwrap();
    let error = Config::default()
        .save(&blocked_parent.join("config.toml"))
        .unwrap_err();
    assert!(error.to_string().contains("create config directory"));

    let config = root.join("config.toml");
    fs::create_dir(config.with_extension("toml.tmp")).unwrap();
    let error = Config::default().save(&config).unwrap_err();
    assert!(error.to_string().contains("write temporary config"));
}
