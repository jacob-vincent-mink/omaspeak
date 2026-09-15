use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;

use omaspeak::config::Config;
use omaspeak::protocol::{Request, Response, ResultPayload};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_os = "linux")]
fn build_audio_cpp_stub(root: &Path) -> PathBuf {
    let output = root.join("native/libaudiocpp.so.0.1.0");
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-Wl,-soname,libaudiocpp.so.0"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audiocpp_stub.c"))
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap();
    assert!(status.success());
    output
}

#[cfg(target_os = "linux")]
fn audio_cpp_stub_config(root: &Path, library: PathBuf, model_file: &str) -> Config {
    let model_dir = root.join("model");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join(model_file), b"stub model").unwrap();
    let mut config = Config::default();
    config.backend.device = "cpu".into();
    config.backend.library_dirs = vec![library.parent().unwrap().to_owned()];
    config.backend.library = Some(library);
    config.model.name = "stub-supertonic".into();
    config.model.directory = model_dir.to_string_lossy().into_owned();
    config.model.file = model_file.into();
    config
}

#[cfg(target_os = "linux")]
#[test]
fn process_isolated_audio_cpp_provider_synthesizes_without_its_cli() {
    let root = sandbox();
    let library = build_audio_cpp_stub(&root);
    let mut config = audio_cpp_stub_config(&root, library, "supertonic.gguf");
    config
        .backend
        .options
        .insert("load.stub".into(), "1".into());
    config
        .backend
        .options
        .insert("session.stub".into(), "1".into());
    config
        .backend
        .options
        .insert("request.stub".into(), "1".into());
    config
        .save(&root.join("config/omaspeak/config.toml"))
        .unwrap();

    let wav = root.join("spoken.wav");
    let output = run(
        &root,
        &[
            "say",
            "coverage proof",
            "--no-play",
            "--out",
            wav.to_str().unwrap(),
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    let bytes = fs::read(wav).unwrap();
    assert!(bytes.starts_with(b"RIFF"));
    assert_eq!(bytes.len(), 44 + 441 * 2);
}

#[cfg(target_os = "linux")]
#[test]
fn environment_provider_path_is_used_for_real_engine_loading() {
    let root = sandbox();
    let library = build_audio_cpp_stub(&root);
    let provider_directory = library.parent().unwrap().to_owned();
    let mut config = audio_cpp_stub_config(&root, library, "supertonic.gguf");
    config.backend.library = None;
    config.backend.library_dirs.clear();
    config
        .save(&root.join("config/omaspeak/config.toml"))
        .unwrap();

    let wav = root.join("environment-provider.wav");
    let output = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args([
            "say",
            "environment provider proof",
            "--no-play",
            "--out",
            wav.to_str().unwrap(),
        ])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_LIBRARY_PATH", provider_directory)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(fs::read(wav).unwrap().starts_with(b"RIFF"));
}

#[cfg(target_os = "linux")]
#[test]
fn process_isolated_audio_cpp_provider_reports_native_failures_without_crashing() {
    for model_file in [
        "fail-load.gguf",
        "null-model.gguf",
        "unsupported.gguf",
        "fail-session.gguf",
        "null-session.gguf",
    ] {
        let root = sandbox();
        let library = build_audio_cpp_stub(&root);
        audio_cpp_stub_config(&root, library, model_file)
            .save(&root.join("config/omaspeak/config.toml"))
            .unwrap();
        let output = run(&root, &["say", "native failure", "--no-play"]);
        assert!(!output.status.success(), "{model_file}");
        assert!(stderr(&output).contains("worker initialization failed"));
    }

    for (text, extra) in [
        ("fail-text", None),
        ("fail-run", None),
        ("bad-rate", None),
        ("bad-channels", None),
        ("empty-audio", None),
        ("nan-audio", None),
        ("normal text", Some("3.75")),
    ] {
        let root = sandbox();
        let library = build_audio_cpp_stub(&root);
        audio_cpp_stub_config(&root, library, "supertonic.gguf")
            .save(&root.join("config/omaspeak/config.toml"))
            .unwrap();
        let mut arguments = vec!["say", text, "--no-play"];
        if let Some(speed) = extra {
            arguments.extend(["--speed", speed]);
        }
        let output = run(&root, &arguments);
        assert!(!output.status.success(), "{text}");
        assert!(stderr(&output).contains("audio.cpp synthesis failed"));
    }

    for model_file in [
        "null-request.gguf",
        "fail-voice.gguf",
        "fail-steps.gguf",
        "null-result.gguf",
        "fail-result.gguf",
        "null-samples.gguf",
        "huge-audio.gguf",
    ] {
        let root = sandbox();
        let library = build_audio_cpp_stub(&root);
        audio_cpp_stub_config(&root, library, model_file)
            .save(&root.join("config/omaspeak/config.toml"))
            .unwrap();
        let output = run(&root, &["say", "native result failure", "--no-play"]);
        assert!(!output.status.success(), "{model_file}");
        assert!(stderr(&output).contains("audio.cpp synthesis failed"));
    }

    for option in ["load.fail", "session.fail"] {
        let root = sandbox();
        let library = build_audio_cpp_stub(&root);
        let mut config = audio_cpp_stub_config(&root, library, "supertonic.gguf");
        config.backend.options.insert(option.into(), "1".into());
        config
            .save(&root.join("config/omaspeak/config.toml"))
            .unwrap();
        let output = run(&root, &["say", "option failure", "--no-play"]);
        assert!(!output.status.success(), "{option}");
        assert!(stderr(&output).contains("worker initialization failed"));
    }

    let root = sandbox();
    let library = build_audio_cpp_stub(&root);
    let mut config = audio_cpp_stub_config(&root, library, "supertonic.gguf");
    config
        .backend
        .options
        .insert("request.fail".into(), "1".into());
    config
        .save(&root.join("config/omaspeak/config.toml"))
        .unwrap();
    let output = run(&root, &["say", "request option failure", "--no-play"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("audio.cpp synthesis failed"));

    let root = sandbox();
    let library = build_audio_cpp_stub(&root);
    audio_cpp_stub_config(&root, library, "exit-load.gguf")
        .save(&root.join("config/omaspeak/config.toml"))
        .unwrap();
    let output = run(&root, &["say", "bounded stderr", "--no-play"]);
    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(
        error.contains("worker exited with exit status: 70"),
        "{error}"
    );
    assert!(
        error.contains("audio.cpp startup diagnostic from provider"),
        "{error}"
    );
    assert!(error.contains("[earlier output truncated]"), "{error}");
    assert!(
        error.len() < 24 * 1024,
        "native diagnostics escaped their configured bound"
    );
}

#[test]
fn runtime_apply_rejects_a_non_audiocpp_library_without_writes() {
    let root = sandbox();
    let directory = root.join("native");
    fs::create_dir_all(&directory).unwrap();
    let config_path = root.join("config/omaspeak/config.toml");
    Config::default().save(&config_path).unwrap();
    let before = fs::read(&config_path).unwrap();
    let library = [
        "/usr/lib/libm.so.6",
        "/usr/lib64/libm.so.6",
        "/lib/x86_64-linux-gnu/libm.so.6",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file());
    let Some(library) = library else {
        return;
    };
    fs::copy(library, directory.join("libaudiocpp.so")).unwrap();
    let output = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--dir",
            directory.to_str().unwrap(),
            "--apply",
        ],
    );
    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(
        error.contains("runtime candidate rejected; config unchanged"),
        "{error}"
    );
    assert!(error.contains("audiocpp_abi_version"), "{error}");
    assert_eq!(fs::read(&config_path).unwrap(), before);
}

#[test]
fn real_cpu_preview_then_apply_pins_paths() {
    let library = std::env::var_os("OMASPEAK_TEST_AUDIOCPP_LIBRARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join(".scratch-ortfree/build-release-omaspeak/bin/libaudiocpp.so.0.1.0")
        });
    if !library.is_file() {
        return;
    }
    let root = sandbox();
    let config_path = root.join("config/omaspeak/config.toml");
    let directory = library.parent().unwrap().to_str().unwrap();
    let preview = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--dir",
            directory,
        ],
    );
    assert!(preview.status.success(), "{}", stderr(&preview));
    assert!(!config_path.exists());
    let applied = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--dir",
            directory,
            "--apply",
        ],
    );
    assert!(applied.status.success(), "{}", stderr(&applied));
    let config = Config::load(&config_path).unwrap();
    assert_eq!(config.backend.kind, "audiocpp");
    assert_eq!(
        config.backend.library,
        Some(library.canonicalize().unwrap())
    );
}

fn sandbox() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-cli-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn run_with_input(root: &Path, args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[cfg(unix)]
#[test]
fn guided_setup_accepts_arrow_keys_and_enter_in_a_real_pty() {
    assert!(
        Command::new("script").arg("--version").output().is_ok(),
        "the real-PTY setup regression test requires util-linux script(1)"
    );
    let root = sandbox();
    let binary = env!("CARGO_BIN_EXE_omaspeak");
    assert!(!binary.contains(['\'', '"', ' ']));
    let mut child = Command::new("script")
        .args(["-qec", &format!("{binary} setup"), "/dev/null"])
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("TERM", "xterm-256color")
        .env("OMASPEAK_LIBRARY_PATH", root.join("missing-runtime"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    // Wait until the child enables raw mode, choose Runtime from the setup
    // screen, accept the preselected runtime, choose CPU, then leave the
    // optional external-stack directory empty when setup asks for one.
    thread::sleep(Duration::from_millis(750));
    input.write_all(b"\x1b[B\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    input.write_all(b"\x1b[B\r").unwrap();
    input.flush().unwrap();
    thread::sleep(Duration::from_millis(150));
    // The intentionally missing CPU runtime makes this answer the directory
    // prompt and then exercises the rejected-candidate path.
    let _ = input.write_all(b"\n");
    drop(input);
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            panic!("guided setup did not finish after PTY input");
        }
        thread::sleep(Duration::from_millis(25));
    }
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let terminal = stdout(&output);
    assert!(terminal.contains("Omaspeak setup"));
    assert!(terminal.contains("Omaspeak runtime"));
    assert!(terminal.contains("Omaspeak device"));
    assert!(terminal.contains("audio.cpp provider was not found"));
    assert!(!root.join("config/omaspeak/config.toml").exists());
}

fn serve_once(root: &Path, result: ResultPayload) -> Option<thread::JoinHandle<Request>> {
    let directory = root.join("run/omaspeak");
    fs::create_dir_all(&directory).unwrap();
    let socket = directory.join("control.sock");
    if socket.exists() {
        fs::remove_file(&socket).unwrap();
    }
    let listener = match UnixListener::bind(socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return None,
        Err(error) => panic!("bind fake daemon: {error}"),
    };
    Some(thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(&mut stream).read_line(&mut line).unwrap();
        let request: Request = serde_json::from_str(&line).unwrap();
        serde_json::to_writer(
            &mut stream,
            &Response {
                protocol: 1,
                id: request.id.clone(),
                result,
            },
        )
        .unwrap();
        stream.write_all(b"\n").unwrap();
        request
    }))
}

fn fake_systemctl(root: &Path, exit: i32) -> PathBuf {
    let bin = root.join(format!("bin-{exit}"));
    fs::create_dir_all(&bin).unwrap();
    let program = bin.join("systemctl");
    fs::write(
        &program,
        format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$OMASPEAK_SYSTEMCTL_LOG\"\nexit {exit}\n"),
    )
    .unwrap();
    #[cfg(unix)]
    fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).unwrap();
    bin
}

fn run_with_path(root: &Path, args: &[&str], path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(args)
        .env("HOME", root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_SYSTEMCTL_LOG", root.join("systemctl.log"))
        .env("PATH", path)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn setup_can_inspect_an_invalid_config_without_weakening_runtime_parsing() {
    let root = sandbox();
    let config_path = root.join("config/omaspeak/config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    let invalid = b"[backend]\nkind = \"supertonic\"\nremoved_pre_release_field = true\n";
    fs::write(&config_path, invalid).unwrap();

    for args in [
        &["setup", "runtime", "--json"][..],
        &["setup", "model", "--json"],
    ] {
        let output = run(&root, args);
        assert!(output.status.success(), "{args:?}: {}", stderr(&output));
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap();
        assert!(stderr(&output).contains("successful setup apply will replace it"));
        assert_eq!(fs::read(&config_path).unwrap(), invalid);
    }

    for args in [&["config", "get", "--json"][..], &["voices", "--json"]] {
        let normal = run(&root, args);
        assert!(!normal.status.success());
        assert!(stderr(&normal).contains("unknown field `removed_pre_release_field`"));
        assert_eq!(fs::read(&config_path).unwrap(), invalid);
    }

    let check = run(&root, &["setup", "check", "--json"]);
    assert!(!check.status.success());
    let checks: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(checks[0]["name"], "config");
    assert_eq!(checks[0]["ok"], false);
    assert_eq!(fs::read(&config_path).unwrap(), invalid);

    let Some(library) = std::env::var_os("OMASPEAK_TEST_AUDIOCPP_LIBRARY")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    else {
        return;
    };
    let applied = run(
        &root,
        &[
            "setup",
            "runtime",
            "--runtime",
            "default",
            "--device",
            "cpu",
            "--dir",
            library.parent().unwrap().to_str().unwrap(),
            "--apply",
        ],
    );
    assert!(applied.status.success(), "{}", stderr(&applied));
    assert!(stderr(&applied).contains("successful setup apply will replace it"));
    let repaired = Config::load(&config_path).unwrap();
    assert_eq!(repaired.backend.kind, "audiocpp");
    assert_eq!(repaired.backend.device, "cpu");
    assert!(
        !fs::read_to_string(&config_path)
            .unwrap()
            .contains("removed_pre_release_field")
    );
}

#[test]
fn successful_setup_replaces_an_invalid_config_and_failed_setup_restores_it() {
    let invalid = b"[audio\nvolume = 0.5\n";

    let failed_root = sandbox();
    let failed_config = failed_root.join("config/omaspeak/config.toml");
    fs::create_dir_all(failed_config.parent().unwrap()).unwrap();
    fs::write(&failed_config, invalid).unwrap();
    let failed_bin = fake_systemctl(&failed_root, 1);
    let failed = run_with_path(
        &failed_root,
        &["setup", "systemd", "--no-start"],
        &failed_bin,
    );
    assert!(!failed.status.success());
    assert_eq!(fs::read(&failed_config).unwrap(), invalid);

    let root = sandbox();
    let config_path = root.join("config/omaspeak/config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, invalid).unwrap();
    let bin = fake_systemctl(&root, 0);
    let service = run_with_path(&root, &["setup", "systemd", "--no-start"], &bin);
    assert!(service.status.success(), "{}", stderr(&service));
    assert!(stderr(&service).contains("successful setup apply will replace it"));

    let repaired = Config::load(&config_path).unwrap();
    assert_eq!(repaired.backend.kind, "audiocpp");
    assert!(
        !fs::read_to_string(&config_path)
            .unwrap()
            .contains("volume = 0.5")
    );
}

#[test]
fn setup_discovery_and_remediation_commands() {
    let root = sandbox();
    for args in [
        &["setup", "runtime"][..],
        &["setup", "runtime", "--json"],
        &["setup", "model", "--list"],
        &["setup", "model", "--json"],
        &["setup", "model"],
    ] {
        let output = run(&root, args);
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(!stdout(&output).is_empty());
    }

    for args in [
        &["setup"][..],
        &["setup", "check", "--json"],
        &["setup", "model", "--verify", "missing"],
        &["setup", "all", "--model", "missing"],
        &["setup", "systemd", "--status"],
    ] {
        let output = run(&root, args);
        assert!(!output.status.success());
    }

    let install = run(&root, &["setup", "menu"]);
    assert!(install.status.success(), "{}", stderr(&install));
    assert!(stdout(&install).contains("installed:"));
    assert!(run(&root, &["setup", "menu", "--status"]).status.success());
    assert!(
        run(&root, &["setup", "menu", "--uninstall"])
            .status
            .success()
    );
    assert!(!run(&root, &["setup", "menu", "--status"]).status.success());
}

#[test]
fn runtime_discovery_reports_invalid_paths_without_reexecing() {
    let root = sandbox();
    let missing = root.join("missing-vendor-runtime");
    let output = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(["setup", "runtime", "--json"])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_LIBRARY_PATH", &missing)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["libraries"]["environment_library_dirs"],
        serde_json::json!([missing])
    );
    assert_eq!(
        report["libraries"]["missing_library_dirs"],
        report["libraries"]["environment_library_dirs"]
    );
    assert!(report["libraries"]["runtime_loadable"].is_object());

    let explicit_dir = root.join("explicit-provider");
    fs::create_dir_all(&explicit_dir).unwrap();
    let explicit_provider = explicit_dir.join("libaudiocpp.so.0.1.0");
    fs::write(&explicit_provider, b"invalid audio.cpp fixture").unwrap();
    let explicit = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(["setup", "runtime", "--json"])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("explicit-config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_LIBRARY_PATH", &explicit_dir)
        .output()
        .unwrap();
    assert!(explicit.status.success(), "{}", stderr(&explicit));
    let report: serde_json::Value = serde_json::from_slice(&explicit.stdout).unwrap();
    let cpu = report["inventory"]
        .as_array()
        .unwrap()
        .iter()
        .find(|state| state["runtime"] == "default" && state["device"] == "cpu")
        .unwrap();
    assert_eq!(cpu["source"], "environment");
    assert_eq!(cpu["discovered"], true);
    assert_eq!(cpu["ready"], false);

    let mutation = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(["config", "set", "daemon.max_text_bytes", "4096"])
        .env("HOME", &root)
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_LIBRARY_PATH", root.join("still-missing"))
        .output()
        .unwrap();
    assert!(mutation.status.success(), "{}", stderr(&mutation));
}

#[test]
fn config_commands_round_trip_and_reject_invalid_values() {
    let root = sandbox();
    let get = run(&root, &["config", "get", "backend.kind"]);
    assert!(get.status.success());
    assert_eq!(stdout(&get).trim(), "audiocpp");
    assert!(run(&root, &["config", "get", "--json"]).status.success());
    assert!(run(&root, &["config", "schema"]).status.success());
    assert!(run(&root, &["config", "schema", "--json"]).status.success());

    for (key, value) in [
        ("backend.kind", "future-backend"),
        ("backend.runtime", "default"),
        ("backend.device", "cpu"),
        ("backend.threads", "3"),
        ("backend.fallback", "cpu"),
        ("backend.device_id", "0"),
        ("backend.library_dirs", "/opt/openvino:/opt/cuda"),
        ("model.family", "future-family"),
        ("model.name", "custom"),
        ("model.directory", "/tmp/model"),
        ("model.voice", "0"),
    ] {
        let output = run(&root, &["config", "set", key, value]);
        assert!(output.status.success(), "{key}: {}", stderr(&output));
        assert!(
            run(&root, &["config", "unset", key]).status.success(),
            "{key}"
        );
    }

    for args in [
        &["config", "get", "missing.key"][..],
        &["config", "set", "missing.key", "x"],
        &["config", "unset", "missing.key"],
        &["config", "set", "backend.runtime", "bogus"],
        &["config", "set", "backend.fallback", "bogus"],
        &["config", "set", "backend.threads", "0"],
    ] {
        assert!(!run(&root, args).status.success());
    }
}

#[test]
fn runtime_commands_report_expected_failures_without_a_model_or_daemon() {
    let root = sandbox();
    let voices = run(&root, &["voices"]);
    assert!(voices.status.success());
    assert!(stdout(&voices).contains("*\t0\tM1"));
    assert!(stdout(&voices).contains(" \t9\tF5"));
    let voices = run(&root, &["voices", "--json"]);
    assert!(voices.status.success());
    let voices: serde_json::Value = serde_json::from_slice(&voices.stdout).unwrap();
    assert_eq!(voices.as_array().unwrap().len(), 10);
    assert_eq!(
        voices[0],
        serde_json::json!({"id":0,"name":"M1","active":true})
    );
    assert_eq!(
        voices[9],
        serde_json::json!({"id":9,"name":"F5","active":false})
    );
    assert!(!run(&root, &["say", ""]).status.success());
    assert!(!run(&root, &["say", "hello", "--no-play"]).status.success());
    assert!(
        !run(
            &root,
            &["benchmark", "--text", "hello", "--out-dir", "benchmark",],
        )
        .status
        .success()
    );
    assert!(!run(&root, &["stop"]).status.success());
    assert!(run(&root, &["status"]).status.success());
    assert!(run(&root, &["status", "--json"]).status.success());
    assert!(!run(&root, &["daemon"]).status.success());

    let mut config = Config::default();
    config.model.family = "unknown".into();
    config.daemon.max_text_bytes = 4;
    let config_path = root.join("small.toml");
    config.save(&config_path).unwrap();
    assert!(
        !run(
            &root,
            &[
                "--config",
                config_path.to_str().unwrap(),
                "say",
                "too long",
                "--no-play",
            ],
        )
        .status
        .success()
    );

    for args in [
        &["setup", "model", "--verify", "unknown-model"][..],
        &["setup", "model", "--set", "unknown-model"],
        &[
            "setup",
            "model",
            "--download",
            "unknown-model",
            "--source",
            "/definitely/missing/model-directory",
        ],
        &[
            "setup",
            "all",
            "--source",
            "/definitely/missing/model-directory",
        ],
    ] {
        assert!(!run(&root, args).status.success());
    }
}

#[test]
fn voices_enumerates_installed_supertonic_speakers_and_marks_the_active_one() {
    let root = sandbox();
    let config_path = root.join("config/omaspeak/config.toml");
    let model_dir = root.join("data/omaspeak/models/custom-supertonic");
    fs::create_dir_all(model_dir.join("voice_styles")).unwrap();
    for name in omaspeak::catalog::SUPERTONIC_VOICE_NAMES {
        fs::write(model_dir.join(format!("voice_styles/{name}.json")), b"{}").unwrap();
    }
    let mut config = Config::default();
    config.backend.kind = "supertonic".into();
    config.model.family = "supertonic".into();
    config.model.name = "custom-supertonic".into();
    config.model.voice_style = "voice_styles".into();
    config.model.voice = 1;
    config.save(&config_path).unwrap();

    let output = run(&root, &["voices", "--json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let voices: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        voices,
        serde_json::json!([
            {"id":0,"name":"M1","active":false}, {"id":1,"name":"M2","active":true},
            {"id":2,"name":"M3","active":false}, {"id":3,"name":"M4","active":false},
            {"id":4,"name":"M5","active":false}, {"id":5,"name":"F1","active":false},
            {"id":6,"name":"F2","active":false}, {"id":7,"name":"F3","active":false},
            {"id":8,"name":"F4","active":false}, {"id":9,"name":"F5","active":false}
        ])
    );
}

#[test]
fn runtime_commands_use_the_daemon_protocol_when_socket_is_present() {
    let root = sandbox();
    let Some(server) = serve_once(
        &root,
        ResultPayload::Synthesis {
            output: "/tmp/spoken.wav".into(),
            sample_rate: 22_050,
            samples: 22_050,
            audio_seconds: 1.0,
            load_milliseconds: 2,
            synthesis_milliseconds: 3,
        },
    ) else {
        return;
    };
    let output = run(
        &root,
        &[
            "say",
            "hello",
            "--voice",
            "2",
            "--speed",
            "1.25",
            "--out",
            "/tmp/spoken.wav",
            "--no-play",
        ],
    );
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("spoken.wav"));
    let request = server.join().unwrap();
    let encoded = serde_json::to_value(request).unwrap();
    assert_eq!(encoded["type"], "say");
    assert_eq!(encoded["text"], "hello");
    assert_eq!(encoded["voice"], 2);

    let server = serve_once(
        &root,
        ResultPayload::Status {
            running: true,
            pid: 42,
            model: "test-model".into(),
            sample_rate: 16_000,
            backend: serde_json::json!({"kind":"test"}),
        },
    )
    .unwrap();
    let output = run(&root, &["status"]);
    assert!(output.status.success());
    assert_eq!(stdout(&output).trim(), "running");
    server.join().unwrap();

    let server = serve_once(
        &root,
        ResultPayload::Status {
            running: true,
            pid: 42,
            model: "test-model".into(),
            sample_rate: 16_000,
            backend: serde_json::json!({}),
        },
    )
    .unwrap();
    let output = run(&root, &["status", "--json"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("test-model"));
    server.join().unwrap();

    let server = serve_once(&root, ResultPayload::Shutdown).unwrap();
    let output = run(&root, &["stop"]);
    assert!(output.status.success());
    assert!(stdout(&output).contains("shutdown"));
    server.join().unwrap();

    let server = serve_once(&root, ResultPayload::Shutdown).unwrap();
    let output = run_with_input(&root, &["say", "--no-play"], "spoken over stdin\n");
    assert!(output.status.success(), "{}", stderr(&output));
    let request = server.join().unwrap();
    let encoded = serde_json::to_value(request).unwrap();
    assert_eq!(encoded["text"], "spoken over stdin\n");

    let server = serve_once(
        &root,
        ResultPayload::Error {
            code: "busy".into(),
            message: "try again".into(),
        },
    )
    .unwrap();
    let output = run(&root, &["stop"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("busy: try again"));
    server.join().unwrap();
}

#[test]
fn systemd_lifecycle_uses_user_manager_and_propagates_failures() {
    let root = sandbox();
    let success = fake_systemctl(&root, 0);
    assert!(
        run_with_path(&root, &["setup", "systemd", "--no-start"], &success)
            .status
            .success()
    );
    let active_check = run_with_path(&root, &["setup", "check", "--json"], &success);
    assert!(stdout(&active_check).contains("active:"));
    fs::remove_file(root.join("config/systemd/user/omaspeak.service")).unwrap();
    let external_active_check = run_with_path(&root, &["setup", "check", "--json"], &success);
    assert!(
        stdout(&external_active_check)
            .contains("active (unit is managed outside Omaspeak's user config)")
    );
    assert!(
        run_with_path(&root, &["setup", "systemd"], &success)
            .status
            .success()
    );
    assert!(
        run_with_path(&root, &["setup", "systemd", "--status"], &success)
            .status
            .success()
    );
    assert!(
        run_with_path(&root, &["setup", "systemd", "--uninstall"], &success)
            .status
            .success()
    );
    assert!(!root.join("config/systemd/user/omaspeak.service").exists());
    let calls = fs::read_to_string(root.join("systemctl.log")).unwrap();
    for expected in [
        "--user daemon-reload",
        "--user enable omaspeak.service",
        "--user restart omaspeak.service",
        "--user is-active --quiet omaspeak.service",
        "--user status omaspeak.service --no-pager",
        "--user disable --now omaspeak.service",
    ] {
        assert!(
            calls.lines().any(|call| call == expected),
            "missing {expected:?} in {calls:?}"
        );
    }

    let failure = fake_systemctl(&root, 1);
    let reinstall = fake_systemctl(&root, 0);
    assert!(
        run_with_path(&root, &["setup", "systemd", "--no-start"], &reinstall)
            .status
            .success()
    );
    let inactive_check = run_with_path(&root, &["setup", "check", "--json"], &failure);
    assert!(stdout(&inactive_check).contains("installed but inactive (optional)"));
    assert!(
        !run_with_path(&root, &["setup", "systemd", "--status"], &failure)
            .status
            .success()
    );
    assert!(
        !run_with_path(&root, &["setup", "systemd"], &failure)
            .status
            .success()
    );
}
