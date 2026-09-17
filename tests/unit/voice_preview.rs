use super::*;

fn preview(installed: bool) -> VoicePreview {
    let root = std::env::temp_dir().join(format!("omaspeak-preview-test-{}", request_id()));
    fs::create_dir_all(&root).unwrap();
    let paths = AppPaths {
        config_file: root.join("config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    };
    VoicePreview::new(Config::default(), paths, vec![0, 5], installed)
}

fn process(script: &str) -> Child {
    use std::os::unix::process::CommandExt;
    ProcessCommand::new("sh")
        .args(["-c", script])
        .process_group(0)
        .spawn()
        .unwrap()
}

fn completed(success: bool) -> Child {
    let mut child = process(if success { "exit 0" } else { "exit 1" });
    child.wait().unwrap();
    child
}

#[test]
fn unavailable_preview_never_creates_files_or_changes_configuration() {
    let mut preview = preview(false);
    let before = toml::to_string(&preview.config).unwrap();
    assert!(
        preview
            .toggle(0)
            .unwrap_err()
            .to_string()
            .contains("Install")
    );
    assert!(!preview.paths.runtime_dir.exists());
    preview.installed = true;
    assert!(preview.toggle(99).is_err());
    assert_eq!(before, toml::to_string(&preview.config).unwrap());
    assert!(preview.poll().unwrap().is_none());
    assert!(worker("invalid JSON").is_err());
}

#[test]
fn sample_start_and_stop_are_temporary_and_reap_the_owned_child() {
    let mut preview = preview(true);
    let before = toml::to_string(&preview.config).unwrap();
    assert!(preview.toggle(1).unwrap().contains("Preparing"));
    let directory = preview.directory.clone().unwrap();
    assert_eq!(
        fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert!(preview.toggle(1).unwrap().contains("stopped"));
    assert!(!directory.exists());
    assert!(preview.child.is_none());
    assert_eq!(before, toml::to_string(&preview.config).unwrap());
    assert!(!preview.paths.config_file.exists());
}

#[test]
fn timeout_failure_and_drop_stop_only_the_audition() {
    let mut unrelated = process("sleep 30");
    let mut preview = preview(true);
    preview.child = Some(process("sleep 30 & wait"));
    assert!(preview.poll().unwrap().is_none());
    preview.started = Instant::now() - Duration::from_secs(121);
    assert!(
        preview
            .poll()
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert!(preview.child.is_none());
    assert!(unrelated.try_wait().unwrap().is_none());
    preview.child = Some(completed(false));
    assert!(preview.poll().is_err());
    preview.child = Some(process("sleep 30"));
    let pid = preview.child.as_ref().unwrap().id();
    drop(preview);
    assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    unrelated.kill().unwrap();
    unrelated.wait().unwrap();
}

#[test]
fn synthesis_completion_starts_one_player_then_cleans_up() {
    let mut preview = preview(true);
    let directory = preview.paths.runtime_dir.join("sample");
    fs::create_dir_all(&directory).unwrap();
    preview.directory = Some(directory.clone());
    preview.child = Some(completed(true));
    let message = preview
        .poll_with(|path| {
            assert_eq!(path, directory.join("sample.wav"));
            Ok(completed(true))
        })
        .unwrap()
        .unwrap();
    assert!(message.contains("playback"));
    assert!(preview.poll().unwrap().unwrap().contains("finished"));
    assert!(!directory.exists());

    preview.child = Some(completed(true));
    preview.directory = Some(preview.paths.runtime_dir.join("missing"));
    assert!(
        preview
            .poll_with(|_| Err(std::io::ErrorKind::NotFound.into()))
            .unwrap_err()
            .to_string()
            .contains("playback worker")
    );
}
