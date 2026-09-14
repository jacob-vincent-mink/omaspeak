use super::*;

#[test]
fn derives_runtime_files() {
    let paths = AppPaths {
        config_file: "/cfg/omaspeak/config.toml".into(),
        data_dir: "/data/omaspeak".into(),
        state_dir: "/state/omaspeak".into(),
        runtime_dir: "/run/omaspeak".into(),
    };
    assert_eq!(paths.socket(), PathBuf::from("/run/omaspeak/control.sock"));
    assert_eq!(
        paths.status_file(),
        PathBuf::from("/state/omaspeak/status.json")
    );
}

#[test]
fn discovery_honors_xdg_and_has_stable_fallbacks() {
    let custom = AppPaths::discover_with(
        |key| match key {
            "HOME" => Some("/home/test".into()),
            "XDG_CONFIG_HOME" => Some("/cfg".into()),
            "XDG_DATA_HOME" => Some("/data".into()),
            "XDG_STATE_HOME" => Some("/state".into()),
            "XDG_RUNTIME_DIR" => Some("/run".into()),
            _ => None,
        },
        Path::new("/tmp"),
    );
    assert_eq!(
        custom.config_file,
        PathBuf::from("/cfg/omaspeak/config.toml")
    );
    assert_eq!(custom.data_dir, PathBuf::from("/data/omaspeak"));
    assert_eq!(custom.state_dir, PathBuf::from("/state/omaspeak"));
    assert_eq!(custom.runtime_dir, PathBuf::from("/run/omaspeak"));

    let home_defaults = AppPaths::discover_with(
        |key| (key == "HOME").then(|| OsString::from("/home/test")),
        Path::new("/tmp"),
    );
    assert_eq!(
        home_defaults.config_file,
        PathBuf::from("/home/test/.config/omaspeak/config.toml")
    );
    assert_eq!(
        home_defaults.data_dir,
        PathBuf::from("/home/test/.local/share/omaspeak")
    );
    assert_eq!(
        home_defaults.state_dir,
        PathBuf::from("/home/test/.local/state/omaspeak")
    );
    assert_eq!(
        home_defaults.runtime_dir,
        PathBuf::from("/tmp/omavoice-unknown/omaspeak")
    );

    let no_home = AppPaths::discover_with(|_| None, Path::new("/tmp"));
    assert_eq!(
        no_home.config_file,
        PathBuf::from("./.config/omaspeak/config.toml")
    );
}
