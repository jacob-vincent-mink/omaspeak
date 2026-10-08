use omaspeak::config::Config;
use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("omaspeak-polish-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}
fn cli(root: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_omaspeak"));
    c.arg("--config")
        .arg(root.join("config.toml"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env_remove("DEEPGRAM_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ELEVENLABS_API_KEY")
        .env_remove("CARTESIA_API_KEY")
        .stdin(Stdio::null());
    c
}
#[test]
fn cloud_setup_key_handoff_and_offline_checks() {
    let root = root("credentials");
    let out = cli(&root)
        .args(["setup", "cloud", "--provider", "deepgram"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let config = Config::load(&root.join("config.toml")).unwrap();
    assert_eq!(config.backend.device, "remote");
    assert_eq!(config.backend.kind, "deepgram");
    let out = cli(&root)
        .args(["cloud", "credential", "check"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":false"));
    let mut child = cli(&root)
        .args(["cloud", "credential", "install", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"private-fixture-secret\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!String::from_utf8_lossy(&out.stderr).contains("private-fixture-secret"));
    let config = Config::load(&root.join("config.toml")).unwrap();
    let key = PathBuf::from(&config.backend.cloud.api_key_file);
    assert_eq!(
        fs::metadata(&key).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(
        !fs::read_to_string(root.join("config.toml"))
            .unwrap()
            .contains("private-fixture-secret")
    );
    let out = cli(&root)
        .args(["cloud", "credential", "check"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":true"));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("private-fixture-secret"));
    let out = cli(&root)
        .args(["setup", "check", "--json"])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&out.stdout).contains("private-fixture-secret"));
    fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        !cli(&root)
            .args(["cloud", "credential", "check"])
            .status()
            .unwrap()
            .success()
    );
    fs::remove_file(&key).unwrap();
    std::os::unix::fs::symlink("config.toml", &key).unwrap();
    assert!(
        !cli(&root)
            .args(["cloud", "credential", "check"])
            .status()
            .unwrap()
            .success()
    );
    let out = cli(&root)
        .args(["config", "schema", "--json"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("backend.cloud.api_key_file"));
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn invalid_cloud_setup_never_overwrites_config_or_accepts_secret_arguments() {
    let root = root("invalid");
    let path = root.join("config.toml");
    Config::default().save(&path).unwrap();
    let before = fs::read(&path).unwrap();
    for args in [
        vec!["setup", "cloud", "--provider", "bad"],
        vec![
            "setup",
            "cloud",
            "--provider",
            "deepgram",
            "--base-url",
            "http://example.com",
        ],
        vec![
            "setup",
            "cloud",
            "--provider",
            "deepgram",
            "--api-key-env",
            "BAD\nKEY",
        ],
        vec!["setup", "cloud"],
        vec!["cloud", "credential", "install"],
    ] {
        assert!(!cli(&root).args(args).output().unwrap().status.success());
        assert_eq!(fs::read(&path).unwrap(), before);
    }
    fs::remove_dir_all(root).unwrap();
}
fn response_server(
    listener: std::net::TcpListener,
    responses: Vec<serde_json::Value>,
) -> std::thread::JoinHandle<Vec<String>> {
    use std::io::{BufRead, BufReader};
    std::thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut headers = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                headers.push_str(&line);
            }
            requests.push(headers);
            let response = response.to_string();
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).unwrap();
        }
        requests
    })
}
#[test]
fn discovery_paginates_account_voices_and_saves_the_selected_alias() {
    use serde_json::json;
    for provider in ["elevenlabs", "cartesia"] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let root = root(provider);
        let responses = if provider == "elevenlabs" {
            vec![
                json!({"voices":[{"voice_id":"a","name":"Alice"}],"has_more":true,"next_page_token":"page two"}),
                json!({"voices":[{"voice_id":"b","name":"Bob"}],"has_more":false}),
            ]
        } else {
            vec![
                json!({"data":[{"id":"a","name":"Alice"}],"has_more":true,"next_page":"a"}),
                json!({"data":[{"id":"b","name":"Bob"}],"has_more":false}),
            ]
        };
        let mut c = Config::default();
        c.backend.kind = provider.into();
        c.backend.device = "remote".into();
        c.backend.cloud.base_url =
            format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        c.backend.cloud.api_key_env = "OMASPEAK_POLISH_KEY".into();
        c.save(&root.join("config.toml")).unwrap();
        let server = response_server(listener, responses);
        let output = cli(&root)
            .args([
                "cloud", "voices", "--json", "--save", "b", "--alias", "friendly",
            ])
            .env("OMASPEAK_POLISH_KEY", "fixture-secret")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let inventory: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(inventory.as_array().unwrap().len(), 2);
        assert!(!String::from_utf8_lossy(&output.stdout).contains("fixture-secret"));
        let c = Config::load(&root.join("config.toml")).unwrap();
        assert_eq!(c.backend.cloud.voices["friendly"], "b");
        assert_eq!(c.model.voice.to_string(), "friendly");
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with(if provider == "elevenlabs" {
            "GET /v2/voices?"
        } else {
            "GET /voices?"
        }));
        assert!(requests[1].contains(if provider == "elevenlabs" {
            "next_page_token=page+two"
        } else {
            "starting_after=a"
        }));
        assert!(requests[0].contains("fixture-secret"));
        if provider == "cartesia" {
            assert!(
                requests[0]
                    .to_lowercase()
                    .contains("cartesia-version: 2026-08-14")
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn discovery_rejects_bad_inventory_cursors_and_unlisted_selection_without_saving() {
    use serde_json::json;
    for (name, responses, id) in [
        ("missing", vec![json!({})], "b"),
        ("invalid", vec![json!({"voices":[{"voice_id":"a"}]})], "b"),
        ("cursor", vec![json!({"voices":[],"has_more":true})], "b"),
        (
            "repeat",
            vec![
                json!({"voices":[],"has_more":true,"next_page_token":"same"}),
                json!({"voices":[],"has_more":true,"next_page_token":"same"}),
            ],
            "b",
        ),
        (
            "unlisted",
            vec![json!({"voices":[{"voice_id":"a","name":"Alice"}]})],
            "b",
        ),
        (
            "too-many-pages",
            (0..20)
                .map(|i| json!({"voices":[],"has_more":true,"next_page_token":i.to_string()}))
                .collect(),
            "b",
        ),
    ] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let root = root(name);
        let mut c = Config::default();
        c.backend.kind = "elevenlabs".into();
        c.backend.cloud.base_url =
            format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        c.backend.cloud.api_key_env = "OMASPEAK_POLISH_KEY".into();
        c.save(&root.join("config.toml")).unwrap();
        let before = fs::read(root.join("config.toml")).unwrap();
        let server = response_server(listener, responses);
        let output = cli(&root)
            .args(["cloud", "voices", "--save", id])
            .env("OMASPEAK_POLISH_KEY", "fixture-secret")
            .output()
            .unwrap();
        assert!(!output.status.success(), "{name}");
        assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("fixture-secret"));
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn smoke_measures_first_pcm_exports_wav_and_preserves_existing_output() {
    use std::io::{BufRead, BufReader, Read};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let root = root("smoke");
    let mut c = Config::default();
    c.backend.kind = "openai-compatible".into();
    c.backend.device = "remote".into();
    c.backend.cloud.base_url = format!(
        "http://127.0.0.1:{}/v1",
        listener.local_addr().unwrap().port()
    );
    c.backend.cloud.api_key_env = "OMASPEAK_POLISH_KEY".into();
    c.save(&root.join("config.toml")).unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut size = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if line.to_lowercase().starts_with("content-length:") {
                size = line
                    .split(':')
                    .nth(1)
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                    .unwrap();
            }
        }
        reader.read_exact(&mut vec![0; size]).unwrap();
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n4\r\n\x01\x00\x02\x00\r\n").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
        stream
            .write_all(b"4\r\n\x03\x00\x04\x00\r\n0\r\n\r\n")
            .unwrap();
    });
    let out = root.join("smoke.wav");
    let output = cli(&root)
        .args(["cloud", "smoke", "--no-play", "--out"])
        .arg(&out)
        .env("OMASPEAK_POLISH_KEY", "fixture-secret")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(data["first_audio_ms"].as_f64().unwrap() < data["total_ms"].as_f64().unwrap());
    assert_eq!(hound::WavReader::open(&out).unwrap().duration(), 4);
    let before = fs::read(&out).unwrap();
    assert!(
        !cli(&root)
            .args(["cloud", "smoke", "--no-play", "--out"])
            .arg(&out)
            .output()
            .unwrap()
            .status
            .success()
    );
    assert_eq!(fs::read(&out).unwrap(), before);
    server.join().unwrap();
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn account_provider_can_be_configured_before_discovering_its_voice() {
    let root = root("onboarding");
    for provider in ["elevenlabs", "cartesia", "openai-compatible"] {
        let output = cli(&root)
            .args(["setup", "cloud", "--provider", provider])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn daemon_credential_check_handles_stopped_and_wrong_process_services() {
    let root = root("daemon-check");
    let bin = root.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let script = bin.join("systemctl");
    assert!(
        cli(&root)
            .args(["setup", "cloud", "--provider", "deepgram"])
            .output()
            .unwrap()
            .status
            .success()
    );
    let path = std::env::join_paths(
        std::iter::once(bin.clone())
            .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
    )
    .unwrap();
    for body in [
        "#!/bin/sh\necho 0\n".to_owned(),
        format!("#!/bin/sh\necho {}\n", std::process::id()),
    ] {
        fs::write(&script, body).unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let out = cli(&root)
            .args(["cloud", "credential", "check", "--daemon"])
            .env("PATH", &path)
            .output()
            .unwrap();
        assert!(!out.status.success());
    }
    let mut daemon = Command::new("python3")
        .args(["-c", "import time; time.sleep(20)", "--config"])
        .arg(root.join("config.toml"))
        .arg("daemon")
        .env("DEEPGRAM_API_KEY", "daemon-fixture-secret")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fs::write(&script, format!("#!/bin/sh\necho {}\n", daemon.id())).unwrap();
    let out = cli(&root)
        .args(["cloud", "credential", "check", "--daemon"])
        .env("PATH", &path)
        .output()
        .unwrap();
    daemon.kill().unwrap();
    daemon.wait().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"credential_available\":true"));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("daemon-fixture-secret"));
    fs::remove_dir_all(root).unwrap();
}
