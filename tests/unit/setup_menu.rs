use super::*;

#[test]
fn launcher_path_falls_back_and_uninstall_is_idempotent() {
    let paths = AppPaths {
        config_file: "config.toml".into(),
        data_dir: "/".into(),
        cache_dir: "/".into(),
        state_dir: "state".into(),
        runtime_dir: "run".into(),
    };
    assert_eq!(
        launcher_path(&paths),
        PathBuf::from("./applications/omaspeak-settings.desktop")
    );
    let mut isolated = paths;
    isolated.data_dir = std::env::temp_dir()
        .join(format!("omaspeak-menu-{}", std::process::id()))
        .join("data");
    uninstall(&isolated).unwrap();

    let launcher = install(&isolated).unwrap();
    let contents = fs::read_to_string(&launcher).unwrap();
    assert!(contents.contains(" setup\nTerminal=true"));
    assert!(!contents.contains(" setup model"));

    uninstall(&isolated).unwrap();
    assert!(!launcher.exists());
}

#[test]
fn desktop_exec_path_escapes_field_codes_and_quoted_characters() {
    assert_eq!(
        quote_exec_path(Path::new("/opt/Oma Speak/$bin`x`%20\\omaspeak")).unwrap(),
        "/opt/Oma Speak/\\\\$bin\\\\`x\\\\`%%20\\\\\\\\omaspeak"
    );
    assert!(quote_exec_path(Path::new("/opt/omaspeak=bad")).is_err());
    assert!(quote_exec_path(Path::new("")).is_err());
    assert!(quote_exec_path(Path::new("/opt/oma\nspeak")).is_err());
    assert_eq!(
        quote_exec_path(Path::new("/opt/ómaspeak")).unwrap(),
        "/opt/ómaspeak"
    );
}
