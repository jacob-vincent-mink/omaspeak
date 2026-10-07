use super::*;
fn request() -> Request {
    Request {
        protocol: 1,
        id: "test".into(),
        command: Command::Status,
    }
}
struct Deferred;
impl Read for Deferred {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::WouldBlock.into())
    }
}
#[test]
fn fragmented_frames_are_bounded_and_single_message_only() {
    let mut pending = Vec::new();
    assert!(
        read_frame(&mut Deferred, &mut pending, 8)
            .unwrap()
            .is_none()
    );
    assert!(
        read_frame(&mut &b"one"[..], &mut pending, 8)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        read_frame(&mut &b"\n"[..], &mut pending, 8)
            .unwrap()
            .unwrap(),
        b"one\n"
    );
    assert!(pending.is_empty());
    assert!(read_frame(&mut &b"123456789"[..], &mut pending, 8).is_err());
    pending.clear();
    assert!(read_frame(&mut &b"a\nb\n"[..], &mut pending, 8).is_err());
    assert!(read_frame(&mut &b""[..], &mut Vec::new(), 8).is_err());
    assert!(nonblocking(-1).is_err());
}
#[test]
fn staged_output_and_job_errors_preserve_destination() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("omaspeak-staging-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let destination = root.join("nested/output.wav");
    let stage = StagedOutput::create(&destination).unwrap();
    assert_eq!(
        fs::metadata(&stage.0).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let path = stage.0.clone();
    drop(stage);
    assert!(!path.exists());
    fs::write(&destination, b"original").unwrap();
    let (client, mut server) = UnixStream::pair().unwrap();
    let mut job = Job::new(client, request());
    job.stage = Some(root.join("staged.wav"));
    fs::write(job.stage.as_ref().unwrap(), b"partial").unwrap();
    job.error("cancelled", "stopped");
    let mut line = String::new();
    BufReader::new(&mut server).read_line(&mut line).unwrap();
    let response: Response = serde_json::from_str(&line).unwrap();
    assert_eq!(response.id, "test");
    drop(job);
    assert!(!root.join("staged.wav").exists());
    assert_eq!(fs::read(&destination).unwrap(), b"original");
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn worker_messages_roundtrip_without_exposing_partial_results() {
    let mut bytes = Vec::new();
    write_worker_message(
        &mut bytes,
        &WorkerMessage::Playing {
            id: "request".into(),
            first_audio_milliseconds: 12,
        },
    )
    .unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert!(matches!(
        serde_json::from_slice::<WorkerMessage>(&bytes).unwrap(),
        WorkerMessage::Playing {
            first_audio_milliseconds: 12,
            ..
        }
    ));
    let (client, mut server) = UnixStream::pair().unwrap();
    let mut job = Job::new(client, request());
    let response = Response::error("test", "runtime", "failure");
    complete_synthesis(&mut job, response).unwrap();
    let mut line = String::new();
    BufReader::new(&mut server).read_line(&mut line).unwrap();
    assert!(line.contains("failure"));
}

#[test]
fn streaming_worker_writes_pcm_reports_first_chunk_and_drains() {
    use std::net::TcpListener;
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("omaspeak-stream-direct-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let player = root.join("pw-play");
    fs::write(
        &player,
        format!("#!/bin/sh\ncat > '{}'\n", root.join("pcm").display()),
    )
    .unwrap();
    fs::set_permissions(&player, fs::Permissions::from_mode(0o755)).unwrap();
    let _env = SavedEnv::set(&[
        ("PATH", format!("{}:/usr/bin:/bin", root.display()).into()),
        ("XDG_RUNTIME_DIR", root.join("run").into_os_string()),
    ]);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut config = Config::default();
    config.backend.kind = "openai-compatible".into();
    config.backend.cloud.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let paths = AppPaths {
        config_file: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.clone(),
        runtime_dir: root.join("run"),
    };
    let engine = Engine::load(&config, &paths).unwrap();
    let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut buf = [0; 8192];
        let count = socket.read(&mut buf).unwrap();
        assert!(count > 0);
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 6\r\nConnection: close\r\n\r\n\x00\x00\xff\x7f\x00\x80").unwrap();
    });
    let mut output = Vec::new();
    let speech = Request {
        protocol: 1,
        id: "stream".into(),
        command: Command::Say {
            text: "hello".into(),
            speed: 1.0,
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
            output: Some(root.join("out.wav").to_string_lossy().into_owned()),
            no_play: false,
        },
    };
    let response = stream_request(&engine, &config, &paths, speech, &mut output);
    server.join().unwrap();
    assert!(
        matches!(response.result, ResultPayload::Synthesis { .. }),
        "{}",
        serde_json::to_string(&response).unwrap()
    );
    assert_eq!(fs::read(root.join("pcm")).unwrap().len(), 6);
    assert!(matches!(
        serde_json::from_slice::<WorkerMessage>(&output).unwrap(),
        WorkerMessage::Playing { .. }
    ));
    assert!(matches!(
        stream_request(&engine, &config, &paths, request(), &mut Vec::new()).result,
        ResultPayload::Error { .. }
    ));
    let mut closed = StreamingPlayer {
        child: ProcessCommand::new("true")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap(),
        input: None,
        pause: None,
    };
    assert!(closed.push(&[0.0]).is_err());
    drop(closed);
    fs::remove_dir_all(root).unwrap();
}

struct SavedEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl SavedEnv {
    fn set(values: &[(&'static str, std::ffi::OsString)]) -> Self {
        let saved = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in values {
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self(saved)
    }
}
impl Drop for SavedEnv {
    fn drop(&mut self) {
        for (key, value) in self.0.drain(..) {
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}
#[test]
fn missing_pipewire_player_uses_alsa_and_drain_failure_is_reported() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("omaspeak-alsa-direct-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let _env = SavedEnv::set(&[
        ("PATH", root.clone().into_os_string()),
        ("XDG_RUNTIME_DIR", root.join("run").into_os_string()),
    ]);
    let aplay = root.join("aplay");
    fs::write(
        &aplay,
        format!(
            "#!/bin/sh\n/usr/bin/cat > '{}'\n",
            root.join("pcm").display()
        ),
    )
    .unwrap();
    fs::set_permissions(&aplay, fs::Permissions::from_mode(0o755)).unwrap();
    let mut player = StreamingPlayer::start(24000, "default").unwrap();
    player.push(&[-2.0, 0.0, 2.0]).unwrap();
    player.finish().unwrap();
    assert_eq!(
        fs::read(root.join("pcm")).unwrap(),
        [-32767_i16, 0, 32767]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    fs::write(&aplay, "#!/bin/sh\n/usr/bin/cat >/dev/null\nexit 1\n").unwrap();
    let mut player = StreamingPlayer::start(24000, "default").unwrap();
    player.push(&[0.1]).unwrap();
    assert!(player.finish().is_err());
    fs::remove_file(&aplay).unwrap();
    assert!(StreamingPlayer::start(24000, "default").is_err());
    assert!(StreamingPlayer::start(24000, "invalid").is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn supervised_worker_rejects_pending_requests_bad_json_and_startup_timeout() {
    fn fixture_worker(script: &str) -> ProcessWorker {
        let mut child = ProcessCommand::new("sh")
            .args(["-c", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        nonblocking(input.as_raw_fd()).unwrap();
        nonblocking(output.as_raw_fd()).unwrap();
        ProcessWorker {
            child,
            input,
            output,
            incoming: vec![],
            outgoing: vec![],
            written: 0,
            ready: false,
            started: Instant::now(),
        }
    }
    let mut worker = fixture_worker("cat >/dev/null");
    worker.submit(&request()).unwrap();
    assert!(worker.submit(&request()).is_err());
    assert!(worker.poll().unwrap().is_none());
    assert!(worker.outgoing.is_empty());
    worker.started = Instant::now() - WORK_TIMEOUT - Duration::from_secs(1);
    assert!(
        worker
            .poll()
            .err()
            .unwrap()
            .to_string()
            .contains("startup timed out")
    );
    drop(worker);
    let mut worker = fixture_worker("printf 'bad json\\n'; cat >/dev/null");
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match worker.poll() {
            Err(error) => {
                assert!(error.to_string().contains("parse"));
                break;
            }
            Ok(None) => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(10));
            }
            Ok(Some(_)) => panic!("bad JSON accepted"),
        }
    }
    drop(worker);
    let mut worker = fixture_worker("exit 1");
    thread::sleep(Duration::from_millis(30));
    assert!(worker.poll().is_err());
}

#[test]
fn worker_handshake_rejects_control_commands_and_exits_on_shutdown() {
    let root = std::env::temp_dir().join(format!("omaspeak-worker-io-{}", std::process::id()));
    let paths = AppPaths {
        config_file: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.clone(),
        runtime_dir: root.join("run"),
    };
    let mut config = Config::default();
    config.backend.kind = "openai-compatible".into();
    config.backend.cloud.base_url = "http://localhost:1234/v1".into();
    let spec = serde_json::to_string(&WorkerConfig { config, paths }).unwrap();
    let mut input = serde_json::to_vec(&request()).unwrap();
    input.push(b'\n');
    input.extend(
        serde_json::to_vec(&Request {
            protocol: 1,
            id: "stop".into(),
            command: Command::Shutdown,
        })
        .unwrap(),
    );
    input.push(b'\n');
    let mut output = Vec::new();
    worker_io(&spec, input.as_slice(), &mut output).unwrap();
    let mut messages = output
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<WorkerMessage>(line).unwrap());
    assert!(
        matches!(messages.next().unwrap(),WorkerMessage::Response{response:Response{id,result:ResultPayload::Status{..},..}} if id=="ready")
    );
    assert!(matches!(
        messages.next().unwrap(),
        WorkerMessage::Response {
            response: Response {
                result: ResultPayload::Error { .. },
                ..
            }
        }
    ));
    assert!(messages.next().is_none());
    assert!(worker_io("invalid", &b""[..], Vec::new()).is_err());
}
