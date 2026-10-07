#![cfg(target_os = "linux")]
use omaspeak::config::Config;
use omaspeak::protocol::{Request, Response, ResultPayload};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Harness {
    root: PathBuf,
    daemon: Option<Child>,
}
impl Harness {
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_omaspeak"));
        command
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env("XDG_STATE_HOME", self.root.join("state"))
            .env("XDG_CACHE_HOME", self.root.join("cache"))
            .env("XDG_RUNTIME_DIR", self.root.join("run"))
            .env("PATH", self.root.join("bin"))
            .env("OMASPEAK_STREAM_NEXT", self.root.join("next"))
            .env("STREAM_PLAYER_BYTES", self.root.join("pcm"))
            .env("STREAM_PLAYER_STARTED", self.root.join("played"))
            .env("STREAM_PLAYER_PID", self.root.join("player-pid"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
    fn new(name: &str, daemon: bool) -> Self {
        let root = std::env::temp_dir().join(format!(
            "omaspeak-stream-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("model")).unwrap();
        fs::write(root.join("model/model.gguf"), b"stub").unwrap();
        let library = root.join("libaudiocpp.so");
        assert!(
            Command::new("cc")
                .args(["-shared", "-fPIC", "-DOMASPEAK_TEST_STREAMING"])
                .arg(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/fixtures/audiocpp_stub.c")
                )
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        let mut config = Config::default();
        config.backend.library = Some(library);
        config.model.directory = root.join("model").to_string_lossy().into_owned();
        config.model.file = "model.gguf".into();
        config.model.name = "stub-supertonic".into();
        config
            .save(&root.join("config/omaspeak/config.toml"))
            .unwrap();
        let player = root.join("bin/pw-play");
        fs::write(
            &player,
            r#"#!/usr/bin/python3
import os, sys
open(os.environ['STREAM_PLAYER_PID'], 'w').write(str(os.getpid()))
first = sys.stdin.buffer.read(2)
open(os.environ['STREAM_PLAYER_STARTED'], 'wb').write(first)
with open(os.environ['STREAM_PLAYER_BYTES'], 'wb') as output:
    output.write(first)
    while True:
        chunk = sys.stdin.buffer.read(4096)
        if not chunk: break
        output.write(chunk)
"#,
        )
        .unwrap();
        fs::set_permissions(player, fs::Permissions::from_mode(0o755)).unwrap();
        let mut harness = Self { root, daemon: None };
        if daemon {
            harness.daemon = Some(harness.command().arg("daemon").spawn().unwrap());
            harness.wait_for("run/omaspeak/control.sock");
        }
        harness
    }
    fn wait_for(&mut self, path: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !self.root.join(path).exists() {
            if let Some(daemon) = &mut self.daemon {
                assert!(daemon.try_wait().unwrap().is_none());
            }
            assert!(Instant::now() < deadline, "missing {path}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn send(&self, id: &str, text: &str, no_play: bool) -> UnixStream {
        let mut socket = UnixStream::connect(self.root.join("run/omaspeak/control.sock")).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        serde_json::to_writer(
            &mut socket,
            &Request {
                protocol: 1,
                id: id.into(),
                command: omaspeak::protocol::Command::Say {
                    text: text.into(),
                    speed: 1.0,
                    voice: 0.into(),
                    output: Some(self.root.join("out.wav").to_string_lossy().into_owned()),
                    no_play,
                },
            },
        )
        .unwrap();
        socket.write_all(b"\n").unwrap();
        socket
    }
    fn reply(socket: &mut UnixStream) -> Response {
        let mut line = String::new();
        BufReader::new(socket).read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(child) = &mut self.daemon {
            unsafe {
                libc::kill(child.id() as i32, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            while child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn native_stream_plays_before_final_chunk_and_publishes_exact_pcm_after_drain() {
    let mut h = Harness::new("early", true);
    fs::write(h.root.join("out.wav"), b"old output").unwrap();
    let mut socket = h.send("early", "stream-test", false);
    h.wait_for("next");
    h.wait_for("played");
    assert_eq!(fs::read(h.root.join("out.wav")).unwrap(), b"old output");
    let status = h.command().arg("status").output().unwrap();
    assert!(status.status.success());
    let response = Harness::reply(&mut socket);
    assert!(matches!(
        response.result,
        ResultPayload::Synthesis { samples: 882, .. }
    ));
    let mut expected = Vec::new();
    for sample in hound::WavReader::open(h.root.join("out.wav"))
        .unwrap()
        .into_samples::<i16>()
    {
        expected.extend_from_slice(&sample.unwrap().to_le_bytes());
    }
    assert_eq!(fs::read(h.root.join("pcm")).unwrap(), expected);
    // The provider's final merged result must not be played a second time.
    assert_eq!(expected.len(), 882 * 2);
}

#[test]
fn cancelling_a_stream_kills_playback_and_preserves_destination_then_recovers() {
    let mut h = Harness::new("cancel", true);
    fs::write(h.root.join("out.wav"), b"old output").unwrap();
    let mut socket = h.send("cancel", "stream-stall", false);
    h.wait_for("next");
    h.wait_for("played");
    let pid: i32 = fs::read_to_string(h.root.join("player-pid"))
        .unwrap()
        .parse()
        .unwrap();
    let start = Instant::now();
    assert!(
        h.command()
            .args(["cancel", "cancel"])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(
        matches!(Harness::reply(&mut socket).result, ResultPayload::Error { code, .. } if code == "cancelled")
    );
    assert_eq!(fs::read(h.root.join("out.wav")).unwrap(), b"old output");
    let deadline = Instant::now() + Duration::from_secs(2);
    while unsafe { libc::kill(pid, 0) } == 0 {
        assert!(
            Instant::now() < deadline,
            "stream player survived cancellation"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut socket = h.send("recovery", "hello again", true);
    assert!(matches!(
        Harness::reply(&mut socket).result,
        ResultPayload::Synthesis { .. }
    ));
}

#[test]
fn failed_late_chunk_keeps_original_output_and_stops_player() {
    let mut h = Harness::new("failure", true);
    fs::write(h.root.join("out.wav"), b"old output").unwrap();
    let mut socket = h.send("fail", "stream-fail", false);
    assert!(matches!(
        Harness::reply(&mut socket).result,
        ResultPayload::Error { .. }
    ));
    assert_eq!(fs::read(h.root.join("out.wav")).unwrap(), b"old output");
    let mut socket = h.send("recovery", "hello again", true);
    assert!(matches!(
        Harness::reply(&mut socket).result,
        ResultPayload::Synthesis { .. }
    ));
    h.wait_for("player-pid");
    let pid: i32 = fs::read_to_string(h.root.join("player-pid"))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
}

#[test]
fn on_demand_speech_uses_streaming_and_matches_saved_audio() {
    let h = Harness::new("local", false);
    let output = h
        .command()
        .args(["say", "hello", "--out"])
        .arg(h.root.join("out.wav"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut wav = Vec::new();
    fs::File::open(h.root.join("out.wav"))
        .unwrap()
        .read_to_end(&mut wav)
        .unwrap();
    assert_eq!(fs::read(h.root.join("pcm")).unwrap(), wav[44..]);
}
