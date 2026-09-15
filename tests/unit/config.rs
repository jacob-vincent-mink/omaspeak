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
    assert_eq!(cfg.backend.threads, 2);
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
fn generated_legacy_config_loads_without_weakening_unknown_field_checks() {
    let root = temp("legacy-generated");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let path = root.join("config.toml");
    let previous = r#"
[backend]
kind = "sherpa-onnx"
runtime = "default"
device = "cpu"
threads = 4
fallback = "error"
device_id = 0
provider_config = ""

[backend.options]

[model]
family = "supertonic"
name = "supertonic-3-int8"
directory = ""
model_file = ""
tokens_file = ""
data_directory = ""
duration_predictor = "duration_predictor.int8.onnx"
text_encoder = "text_encoder.int8.onnx"
vector_estimator = "vector_estimator.int8.onnx"
vocoder = "vocoder.int8.onnx"
tts_json = "tts.json"
unicode_indexer = "unicode_indexer.bin"
voice_style = "voice.bin"
language = "en"
steps = 5
voice = 2
noise_scale = 0.667
noise_scale_w = 0.8
length_scale = 1.0

[model.options]

[audio]
device = "default"
volume = 0.9

[daemon]
queue_capacity = 8
max_text_bytes = 65536
"#;
    fs::write(&path, previous).unwrap();
    let migrated = Config::load(&path).unwrap();
    assert_eq!(migrated.backend.kind, "supertonic");
    assert_eq!(migrated.backend.threads, 4);
    assert_eq!(migrated.model.name, "supertonic-3-int8");
    assert_eq!(migrated.model.voice, 2);
    assert_eq!(migrated.audio.volume, 0.9);

    migrated.save(&path).unwrap();
    let saved = fs::read_to_string(&path).unwrap();
    for stale in [
        "provider_config",
        "model_file",
        "tokens_file",
        "data_directory",
        "noise_scale",
        "noise_scale_w",
        "length_scale",
        "sherpa-onnx",
    ] {
        assert!(!saved.contains(stale), "saved config retained {stale}");
    }

    fs::write(
        &path,
        previous.replace("threads = 4", "threads = 4\nthreadz = 4"),
    )
    .unwrap();
    let unknown = Config::load(&path).unwrap_err();
    assert!(unknown.to_string().contains("parse config"));
    assert!(format!("{unknown:#}").contains("unknown field `threadz`"));

    fs::write(
        &path,
        previous.replace(
            "provider_config = \"\"",
            "provider_config = \"legacy.conf\"",
        ),
    )
    .unwrap();
    let error = Config::load(&path).unwrap_err().to_string();
    assert!(error.contains("upgrade config"));
    assert!(format!("{:#}", Config::load(&path).unwrap_err()).contains("no longer supported"));
}
