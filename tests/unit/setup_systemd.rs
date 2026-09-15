use super::*;

#[test]
fn unit_uses_absolute_binary_and_config() {
    let unit = generate(Path::new("/opt/oma speak"), Path::new("/tmp/config.toml")).unwrap();
    assert!(unit.contains("ExecStart=\"/opt/oma speak\" --config \"/tmp/config.toml\" daemon"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("XDG_RUNTIME_DIR=%t"));
}

#[test]
fn unit_preserves_and_escapes_the_native_library_path() {
    let unit = generate_with_library_path(
        Path::new("/opt/omaspeak"),
        Path::new("/tmp/config.toml"),
        Some(OsStr::new("/opt/oma lib:/opt/%t/openvino\\runtime\"quoted")),
    )
    .unwrap();
    assert!(unit.contains(
        "Environment=\"LD_LIBRARY_PATH=/opt/oma lib:/opt/%%t/openvino\\\\runtime\\\"quoted\""
    ));
}

#[test]
fn generated_unit_does_not_capture_the_ambient_loader_path() {
    let unit = generate(Path::new("/opt/omaspeak"), Path::new("/tmp/config.toml")).unwrap();
    assert!(!unit.contains("LD_LIBRARY_PATH"));
}

#[test]
fn unit_escapes_every_systemd_exec_specifier_and_rejects_control_characters() {
    let unit = generate(
        Path::new("/opt/oma \\\"speak%t"),
        Path::new("/tmp/config %h/quoted\"file.toml"),
    )
    .unwrap();
    assert!(unit.contains(
        "ExecStart=\"/opt/oma \\\\\\\"speak%%t\" --config \"/tmp/config %%h/quoted\\\"file.toml\" daemon"
    ));

    let error = generate(
        Path::new("/opt/omaspeak\nunsafe"),
        Path::new("/tmp/config.toml"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("control character"));
    let error = generate(
        Path::new("/opt/omaspeak"),
        Path::new("/tmp/config\tunsafe.toml"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("control character"));
}

#[test]
fn no_start_still_enables_the_unit_without_starting_it() {
    use std::cell::{Cell, RefCell};

    let calls = RefCell::new(Vec::<Vec<String>>::new());
    let active_checked = Cell::new(false);
    apply_service_lifecycle(
        false,
        |arguments| {
            calls
                .borrow_mut()
                .push(arguments.iter().map(ToString::to_string).collect());
            Ok(())
        },
        || {
            active_checked.set(true);
            true
        },
    )
    .unwrap();
    assert_eq!(
        &*calls.borrow(),
        &[
            vec![String::from("daemon-reload")],
            vec![String::from("enable"), String::from(UNIT)]
        ]
    );
    assert!(!active_checked.get());
}

#[test]
fn normal_install_enables_restarts_and_verifies_the_unit() {
    use std::cell::RefCell;

    let calls = RefCell::new(Vec::<Vec<String>>::new());
    apply_service_lifecycle(
        true,
        |arguments| {
            calls
                .borrow_mut()
                .push(arguments.iter().map(ToString::to_string).collect());
            Ok(())
        },
        || true,
    )
    .unwrap();
    assert_eq!(calls.borrow().len(), 3);
    assert_eq!(calls.borrow()[2], ["restart", UNIT]);
    assert!(apply_service_lifecycle(true, |_| Ok(()), || false).is_err());
}

#[test]
fn setup_reload_only_restarts_a_service_that_was_active() {
    use std::cell::Cell;

    let restarted = Cell::new(false);
    assert!(
        !reload_if_was_active_with(
            false,
            || {
                restarted.set(true);
                Ok(())
            },
            || true
        )
        .unwrap()
    );
    assert!(!restarted.get());

    assert!(reload_if_was_active_with(true, || Ok(()), || true).unwrap());
    assert!(reload_if_was_active_with(true, || Ok(()), || false).is_err());
}

#[test]
fn atomic_unit_write_creates_parent_and_replaces_the_target() {
    let root = std::env::temp_dir().join(format!("omaspeak-unit-write-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let path = root.join("nested/omaspeak.service");
    write_atomic(&path, b"first").unwrap();
    write_atomic(&path, b"second").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"second");
    assert!(!path.with_extension("tmp").exists());
}
