use super::*;

#[test]
fn launcher_path_falls_back_and_uninstall_is_idempotent() {
    let paths = AppPaths {
        config_file: "config.toml".into(),
        data_dir: "/".into(),
        state_dir: "state".into(),
        runtime_dir: "run".into(),
    };
    assert_eq!(
        launcher_path(&paths),
        PathBuf::from("./applications/omaspeak-settings.desktop")
    );
    let mut isolated = paths;
    isolated.data_dir = std::env::temp_dir().join("omaspeak-menu-missing/data");
    uninstall(&isolated).unwrap();
}
