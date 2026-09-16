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
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    (Config::default(), paths)
}

#[test]
fn official_style_directory_provides_stable_voice_ids() {
    let (mut config, paths) = fixture("supertonic");
    config.backend.kind = "supertonic".into();
    config.model.name = "custom-supertonic".into();
    config.model.family = "supertonic".into();
    config.model.voice_style = "voice_styles".into();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(directory.join("voice_styles")).unwrap();
    for name in SUPERTONIC_PRESET_NAMES {
        fs::write(directory.join(format!("voice_styles/{name}.json")), b"{}").unwrap();
    }

    let voices = installed(&config, &paths).unwrap();
    assert_eq!(voices.len(), 10);
    assert_eq!(
        voices[2],
        Voice {
            id: 2,
            name: "M3".into()
        }
    );
    config.model.voice = crate::voices::VoiceSelection::Legacy(9);
    validate_selected(&config, &voices).unwrap();
    config.model.voice = crate::voices::VoiceSelection::Legacy(10);
    assert!(validate_selected(&config, &voices).is_err());
}

#[test]
fn catalog_inventory_is_available_before_download() {
    let (mut config, paths) = fixture("catalog");
    assert_eq!(
        available(&config, &paths).unwrap(),
        from_catalog(crate::catalog::model("supertonic-3-gguf").unwrap())
    );
    config.model.name = "missing".into();
    config.backend.kind = "supertonic".into();
    assert!(available(&config, &paths).is_err());
}

#[test]
fn audiocpp_exposes_the_stable_supertonic_presets_without_sidecar_files() {
    let (mut config, paths) = fixture("audiocpp");
    config.backend.kind = "audiocpp".into();
    config.model.name = "custom-gguf".into();
    let voices = available(&config, &paths).unwrap();
    assert_eq!(voices.len(), 10);
    assert_eq!(voices.first().unwrap().name, "M1");
    assert_eq!(voices.last().unwrap().name, "F5");
    assert_eq!(installed(&config, &paths).unwrap(), voices);
}

#[test]
fn installed_voice_metadata_rejects_unsupported_empty_and_corrupt_models() {
    let (mut config, paths) = fixture("invalid");
    config.backend.kind = "supertonic".into();
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

    config.model.voice_style = "voice_styles".into();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(&directory).unwrap();
    fs::create_dir_all(directory.join("voice_styles")).unwrap();
    assert!(
        installed(&config, &paths)
            .unwrap_err()
            .to_string()
            .contains("voice style is missing")
    );
    for name in SUPERTONIC_PRESET_NAMES {
        fs::write(directory.join(format!("voice_styles/{name}.json")), b"{}").unwrap();
    }
    fs::remove_file(directory.join("voice_styles/F5.json")).unwrap();
    assert!(
        installed(&config, &paths)
            .unwrap_err()
            .to_string()
            .contains("F5.json")
    );
}

#[test]
fn legacy_and_named_presets_preserve_all_supertonic_identities() {
    let voices = supertonic_presets();
    for (index, name) in SUPERTONIC_PRESET_NAMES.iter().enumerate() {
        let legacy = VoiceSelection::Legacy(index as i32);
        let named = VoiceSelection::Name(name.to_lowercase());
        assert_eq!(legacy.resolve(&voices).unwrap(), index as i32);
        assert_eq!(named.resolve(&voices).unwrap(), index as i32);
        assert_eq!(
            serde_json::to_value(&legacy).unwrap(),
            serde_json::json!(index)
        );
        assert_eq!(
            serde_json::to_value(&named).unwrap(),
            serde_json::json!(name.to_lowercase())
        );
        assert_eq!(
            legacy.to_string().parse::<VoiceSelection>().unwrap(),
            legacy
        );
        assert_eq!(named.to_string().parse::<VoiceSelection>().unwrap(), named);
    }
    for selection in [
        VoiceSelection::Legacy(-1),
        VoiceSelection::Legacy(10),
        VoiceSelection::Name("unknown".into()),
        VoiceSelection::Name("".into()),
    ] {
        assert!(selection.resolve(&voices).is_err());
    }
    assert!(" ".parse::<VoiceSelection>().is_err());
    for invalid in ["true", "1.5", "{}", "null"] {
        assert!(serde_json::from_str::<VoiceSelection>(invalid).is_err());
    }
}
