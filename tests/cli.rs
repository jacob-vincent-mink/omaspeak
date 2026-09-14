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
    if Command::new("script").arg("--version").output().is_err() {
        return;
    }
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    // Wait until the child enables raw mode, choose Runtime from the setup
    // screen, accept the preselected runtime, choose CPU, then leave the
    // optional external-stack directory empty when setup asks for one, and
    // confirm the final Apply screen.
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
    // A packaged or ambient CPU runtime skips the directory prompt, so either
    // this Enter or the next one confirms Apply.
    let _ = input.write_all(b"\n");
    let _ = input.flush();
    thread::sleep(Duration::from_millis(150));
    let _ = input.write_all(b"\r");
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
    assert!(output.status.success(), "{}", stderr(&output));
    let terminal = stdout(&output);
    assert!(terminal.contains("Omaspeak setup"));
    assert!(terminal.contains("Omaspeak runtime"));
    assert!(terminal.contains("Omaspeak device"));
    assert!(terminal.contains("Runtime configured: default on cpu"));
    let config = Config::load(&root.join("config/omaspeak/config.toml")).unwrap();
    assert_eq!(config.backend.device, "cpu");
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

    let mutation = Command::new(env!("CARGO_BIN_EXE_omaspeak"))
        .args(["config", "set", "audio.volume", "0.7"])
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
    assert_eq!(stdout(&get).trim(), "supertonic");
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
        ("model.family", "vits"),
        ("model.name", "custom"),
        ("model.directory", "/tmp/model"),
        ("model.voice", "0"),
        ("audio.device", "test"),
        ("audio.volume", "0.8"),
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
            "--archive",
            "/definitely/missing/archive.tar.bz2",
        ],
        &[
            "setup",
            "all",
            "--archive",
            "/definitely/missing/archive.tar.bz2",
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
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("voice.bin"),
        [2_i64, 1, 1, 2, 1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut config = Config::default();
    config.model.family = "supertonic".into();
    config.model.name = "custom-supertonic".into();
    config.model.voice_style = "voice.bin".into();
    config.model.voice = 1;
    config.save(&config_path).unwrap();

    let output = run(&root, &["voices", "--json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let voices: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        voices,
        serde_json::json!([
            {"id":0,"name":"Voice 1","active":false},
            {"id":1,"name":"Voice 2","active":true}
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
