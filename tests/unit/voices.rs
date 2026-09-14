use super::*;
use std::fs;

fn fixture(name: &str) -> (Config, AppPaths) {
    let root = std::env::temp_dir().join(format!(
        "omaspeak-voices-test-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    (Config::default(), paths)
}

#[test]
fn supertonic_header_provides_stable_voice_ids() {
    let (mut config, paths) = fixture("supertonic");
    config.model.name = "custom-supertonic".into();
    config.model.family = "supertonic".into();
    config.model.voice_style = "voice.bin".into();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(&directory).unwrap();
    let dimensions = [3_i64, 50, 256, 3, 8, 16];
    fs::write(
        directory.join("voice.bin"),
        dimensions
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let voices = installed(&config, &paths).unwrap();
    assert_eq!(voices.len(), 3);
    assert_eq!(
        voices[2],
        Voice {
            id: 2,
            name: "Voice 3".into()
        }
    );
    config.model.voice = 2;
    validate_selected(&config, &voices).unwrap();
    config.model.voice = 3;
    assert!(validate_selected(&config, &voices).is_err());
}

#[test]
fn catalog_inventory_is_available_before_download() {
    let (mut config, paths) = fixture("catalog");
    assert_eq!(
        available(&config, &paths).unwrap(),
        from_catalog(crate::catalog::model("supertonic-3-int8").unwrap())
    );
    config.model.name = "missing".into();
    assert!(available(&config, &paths).is_err());
}

#[test]
fn installed_voice_metadata_rejects_unsupported_empty_and_corrupt_models() {
    let (mut config, paths) = fixture("invalid");
    config.model.family = "other".into();
    assert!(
        installed(&config, &paths)
            .unwrap_err()
            .to_string()
            .contains("family")
    );

    config.model.family = "supertonic".into();
    config.model.voice_style.clear();
    assert!(
        installed(&config, &paths)
            .unwrap_err()
            .to_string()
            .contains("not configured")
    );

    config.model.voice_style = "voice.bin".into();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(&directory).unwrap();
    let invalid_dimensions = [2_i64, 50, 256, 3, 8, 16];
    fs::write(
        directory.join("voice.bin"),
        invalid_dimensions
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(
        installed(&config, &paths)
            .unwrap_err()
            .to_string()
            .contains("invalid Supertonic voice dimensions")
    );
}
