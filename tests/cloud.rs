#![cfg(target_os = "linux")]
use omaspeak::config::Config;
use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
fn read_request(stream: &mut TcpStream) -> (String, serde_json::Value) {
    stream
        .set_read_timeout(Some(Duration::from_secs(8)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut headers = String::new();
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        if line.to_ascii_lowercase().starts_with("content-length:") {
            length = line.split(':').nth(1).unwrap().trim().parse().unwrap();
        }
        headers.push_str(&line);
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    (headers, serde_json::from_slice(&body).unwrap())
}
fn config_root(name: &str, provider: &str, port: u16) -> PathBuf {
    let root = std::env::temp_dir().join(format!("omaspeak-cloud-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let mut c = Config::default();
    c.backend.kind = provider.into();
    c.backend.device = "remote".into();
    c.backend.cloud.base_url = format!(
        "http://127.0.0.1:{port}{}",
        if provider == "openai-compatible" || provider == "deepgram" {
            "/v1"
        } else {
            ""
        }
    );
    c.backend.cloud.api_key_env = "OMASPEAK_CLOUD_TEST_KEY".into();
    c.backend.cloud.voice = "test-voice".into();
    c.save(&root.join("config/omaspeak/config.toml")).unwrap();
    root
}
fn command(root: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_omaspeak"));
    c.env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_STATE_HOME", root.join("state"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_RUNTIME_DIR", root.join("run"))
        .env("OMASPEAK_CLOUD_TEST_KEY", "test-secret")
        .stdin(Stdio::null());
    c
}
#[test]
fn four_providers_send_documented_requests_and_decode_fragmented_pcm_without_models() {
    for provider in ["elevenlabs", "openai-compatible", "cartesia", "deepgram"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let root = config_root(provider, provider, listener.local_addr().unwrap().port());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let req = read_request(&mut stream);
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").unwrap();
            for b in [1000i16, -1000, 2000, -2000]
                .into_iter()
                .flat_map(i16::to_le_bytes)
            {
                stream
                    .write_all(&[b'1', b'\r', b'\n', b, b'\r', b'\n'])
                    .unwrap();
            }
            stream.write_all(b"0\r\n\r\n").unwrap();
            req
        });
        let out = command(&root)
            .args(["say", "Hello cloud", "--no-play", "--out"])
            .arg(root.join("out.wav"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{provider}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let (headers, body) = server.join().unwrap();
        assert!(headers.contains("test-secret"));
        let headers = headers.to_ascii_lowercase();
        match provider {
            "elevenlabs" => {
                assert!(headers.starts_with(
                    "post /v1/text-to-speech/test-voice/stream?output_format=pcm_24000 "
                ));
                assert!(headers.contains("xi-api-key:"));
                assert_eq!(body["model_id"], "eleven_flash_v2_5");
            }
            "openai-compatible" => {
                assert!(headers.starts_with("post /v1/audio/speech "));
                assert_eq!(body["response_format"], "pcm");
                assert_eq!(body["voice"], "test-voice");
            }
            "cartesia" => {
                assert!(headers.starts_with("post /tts/bytes "));
                assert!(headers.contains("cartesia-version: 2026-08-14"));
                assert_eq!(body["voice"], "test-voice");
                assert_eq!(body["output_format"]["container"], "raw");
            }
            "deepgram" => {
                assert!(headers.starts_with("post /v1/speak?"));
                assert!(headers.contains("container=none"));
                assert!(headers.contains("authorization: token test-secret"));
                assert_eq!(body["text"], "Hello cloud");
            }
            _ => unreachable!(),
        }
        let mut wav = hound::WavReader::open(root.join("out.wav")).unwrap();
        assert_eq!(wav.spec().sample_rate, 24000);
        assert_eq!(wav.samples::<i16>().count(), 4);
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn streamed_http_audio_reaches_player_before_final_chunk() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = config_root(
        "early",
        "openai-compatible",
        listener.local_addr().unwrap().port(),
    );
    fs::create_dir_all(root.join("bin")).unwrap();
    fs::write(root.join("bin/pw-play"),"#!/usr/bin/python3\nimport sys,os\nb=sys.stdin.buffer.read(2)\nopen(os.environ['CLOUD_PLAYED'],'wb').write(b)\nsys.stdin.buffer.read()\n").unwrap();
    fs::set_permissions(root.join("bin/pw-play"), fs::Permissions::from_mode(0o755)).unwrap();
    let marker = root.join("played");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        read_request(&mut stream);
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n\x01\x00\r\n").unwrap();
        stream.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !marker.exists() {
            assert!(
                Instant::now() < deadline,
                "first chunk buffered instead of played"
            );
            thread::sleep(Duration::from_millis(10));
        }
        stream.write_all(b"2\r\n\x02\x00\r\n0\r\n\r\n").unwrap();
    });
    let out = command(&root)
        .env("PATH", root.join("bin"))
        .env("CLOUD_PLAYED", root.join("played"))
        .args(["say", "stream this", "--out"])
        .arg(root.join("out.wav"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    server.join().unwrap();
    assert!(root.join("played").exists());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn cloud_errors_and_redirects_are_redacted_and_preserve_file() {
    for (name, status, body, mime) in [
        (
            "auth",
            "401 Unauthorized",
            "test-secret echoed text",
            "application/json",
        ),
        ("redirect", "302 Found", "", "application/octet-stream"),
        ("bad-mime", "200 OK", "not PCM", "application/json"),
        ("truncated", "200 OK", "x", "audio/pcm"),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let root = config_root(
            name,
            "openai-compatible",
            listener.local_addr().unwrap().port(),
        );
        fs::write(root.join("out.wav"), b"original").unwrap();
        let server = thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            read_request(&mut s);
            write!(s,"HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nLocation: https://example.com/secret\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
        });
        let out = command(&root)
            .args(["say", "text must stay private", "--no-play", "--out"])
            .arg(root.join("out.wav"))
            .output()
            .unwrap();
        server.join().unwrap();
        assert!(!out.status.success());
        let msg = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!msg.contains("test-secret"));
        assert!(!msg.contains("text must stay private"));
        assert_eq!(fs::read(root.join("out.wav")).unwrap(), b"original");
        fs::remove_dir_all(root).unwrap();
    }
}
#[test]
fn cancelling_a_stalled_cloud_stream_closes_http_preserves_export_and_keeps_daemon_responsive() {
    use std::os::unix::net::UnixStream;
    struct ProcessGuard(std::process::Child);
    impl Drop for ProcessGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let root = config_root(
        "cancel",
        "openai-compatible",
        listener.local_addr().unwrap().port(),
    );
    fs::create_dir_all(root.join("bin")).unwrap();
    fs::write(root.join("out.wav"), b"original").unwrap();
    fs::write(root.join("bin/pw-play"),"#!/usr/bin/python3\nimport os,sys\nb=sys.stdin.buffer.read(2)\nopen(os.environ['CLOUD_PLAYED'],'wb').write(b)\nsys.stdin.buffer.read()\n").unwrap();
    fs::set_permissions(root.join("bin/pw-play"), fs::Permissions::from_mode(0o755)).unwrap();
    let server = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        read_request(&mut s);
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n\x01\x00\r\n").unwrap();
        s.flush().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
        let mut byte = [0u8];
        assert_eq!(
            s.read(&mut byte).unwrap(),
            0,
            "cancel did not close cloud HTTP connection"
        );
    });
    let mut daemon = ProcessGuard(
        command(&root)
            .env_remove("DBUS_SESSION_BUS_ADDRESS")
            .env("PATH", root.join("bin"))
            .env("CLOUD_PLAYED", root.join("played"))
            .arg("daemon")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let socket = root.join("run/omaspeak/control.sock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while UnixStream::connect(&socket).is_err() {
        assert!(Instant::now() < deadline);
        assert!(daemon.0.try_wait().unwrap().is_none());
        thread::sleep(Duration::from_millis(10));
    }
    let mut say = ProcessGuard(
        command(&root)
            .args(["say", "stalled cloud", "--out"])
            .arg(root.join("out.wav"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while !root.join("played").exists() {
        assert!(Instant::now() < deadline);
        assert!(say.0.try_wait().unwrap().is_none());
        thread::sleep(Duration::from_millis(10));
    }
    let start = Instant::now();
    let out = command(&root).arg("cancel").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(!say.0.wait().unwrap().success());
    server.join().unwrap();
    assert_eq!(fs::read(root.join("out.wav")).unwrap(), b"original");
    assert!(
        command(&root)
            .args(["status", "--json"])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        command(&root)
            .arg("stop")
            .output()
            .unwrap()
            .status
            .success()
    );
    daemon.0.wait().unwrap();
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn cloud_settings_and_checks_do_not_require_native_models_or_make_paid_requests() {
    let root = config_root("settings", "openai-compatible", 1);
    let out = command(&root)
        .args(["config", "schema", "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let data: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let runtime = data["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["key"] == "backend.runtime")
        .unwrap();
    assert_eq!(runtime["choices"][0]["capability"], "remote");
    assert_eq!(runtime["choices"][0]["available"], true);
    let out = command(&root)
        .args(["setup", "check", "--json"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8_lossy(&out.stdout);
    assert!(!json.contains("test-secret"));
    assert!(json.contains("no paid request"));
    fs::remove_dir_all(root).unwrap();
}
