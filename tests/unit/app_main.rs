use super::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn sandbox() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omaspeak-main-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn request(command: Command) -> Request {
    Request {
        protocol: 1,
        id: "request-id".into(),
        command,
    }
}

struct FakeEngine {
    fail: bool,
    runtime: Runtime,
}

struct LibraryBackend;

struct FakeModelOperations {
    installed: bool,
}

#[derive(Default)]
struct ScriptedSelector {
    selections: VecDeque<Option<usize>>,
    calls: Vec<(String, Vec<MenuItem>, usize)>,
}

impl ScriptedSelector {
    fn new(selections: impl IntoIterator<Item = Option<usize>>) -> Self {
        Self {
            selections: selections.into_iter().collect(),
            calls: Vec::new(),
        }
    }
}

impl SetupSelector for ScriptedSelector {
    fn select(
        &mut self,
        title: &str,
        _: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>> {
        self.calls.push((title.into(), items.to_vec(), preferred));
        Ok(self.selections.pop_front().flatten())
    }
}

impl ModelSetupOperations for FakeModelOperations {
    fn models(&self) -> &'static [omaspeak::catalog::ModelSpec] {
        omaspeak::catalog::models()
    }

    fn resolve(&self, id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
        model_spec(id)
    }

    fn verify(&self, _: &AppPaths, _: &omaspeak::catalog::ModelSpec) -> Result<()> {
        if self.installed {
            Ok(())
        } else {
            bail!("not installed")
        }
    }

    fn install(
        &self,
        paths: &AppPaths,
        spec: &omaspeak::catalog::ModelSpec,
        _: Option<&Path>,
        _: ProgressFormat,
    ) -> Result<PathBuf> {
        Ok(paths.data_dir.join("models").join(spec.id))
    }
}

impl omaspeak::engine::TtsBackend for LibraryBackend {
    fn kind(&self) -> &'static str {
        "library-fake"
    }

    fn sample_rate(&self) -> i32 {
        24_000
    }

    fn num_voices(&self) -> i32 {
        1
    }

    fn generate(&self, _: &str, _: f32, _: i32) -> Result<Vec<f32>> {
        Ok(vec![0.0])
    }
}

struct MemoryStream {
    input: std::io::Cursor<Vec<u8>>,
    output: Arc<Mutex<Vec<u8>>>,
    fail_write: bool,
}

impl MemoryStream {
    fn request(command: Command, output: Arc<Mutex<Vec<u8>>>) -> Self {
        let mut input = serde_json::to_vec(&request(command)).unwrap();
        input.push(b'\n');
        Self {
            input: std::io::Cursor::new(input),
            output,
            fail_write: false,
        }
    }
}

impl Read for MemoryStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.input.read(buffer)
    }
}

impl Write for MemoryStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if self.fail_write {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "client disconnected",
            ));
        }
        self.output.lock().unwrap().extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl SpeechEngine for FakeEngine {
    fn synthesize(
        &self,
        _text: &str,
        _speed: f32,
        _voice: i32,
        output: &Path,
    ) -> Result<Synthesis> {
        if self.fail {
            bail!("synthesis failed")
        }
        Ok(Synthesis {
            output: output.into(),
            sample_rate: 16_000,
            samples: 8_000,
            synthesis_time: Duration::from_millis(7),
        })
    }

    fn backend_kind(&self) -> &'static str {
        "fake"
    }

    fn model_name(&self) -> &str {
        "fixture"
    }

    fn sample_rate(&self) -> i32 {
        16_000
    }

    fn load_milliseconds(&self) -> u64 {
        5
    }

    fn effective_runtime(&self) -> Runtime {
        self.runtime
    }

    fn fallback_used(&self) -> bool {
        true
    }
}

#[test]
fn request_framing_accepts_valid_messages_and_rejects_bad_input() {
    let cases = [
        (
            serde_json::to_vec(&request(Command::Status)).unwrap(),
            4096,
            None,
        ),
        (
            br#"{"protocol":2,"id":"old","type":"status"}"#.to_vec(),
            4096,
            Some("protocol 2 is unsupported"),
        ),
        (b"not-json".to_vec(), 4096, Some("expected ident")),
        (
            serde_json::to_vec(&request(Command::Status)).unwrap(),
            4,
            Some("message exceeds 4 bytes"),
        ),
    ];

    for (mut bytes, limit, expected_error) in cases {
        bytes.push(b'\n');
        let mut reader = std::io::Cursor::new(bytes);
        match expected_error {
            None => {
                let decoded = read_request(&mut reader, limit).unwrap();
                assert_eq!(decoded.id, "request-id");
                assert!(matches!(decoded.command, Command::Status));
            }
            Some(message) => assert!(
                read_request(&mut reader, limit)
                    .unwrap_err()
                    .to_string()
                    .contains(message)
            ),
        }
    }
}

#[test]
fn socket_client_sends_newline_delimited_request_and_decodes_response() {
    let root = sandbox();
    let socket = root.join("control.sock");
    let listener = match UnixListener::bind(&socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("bind test socket: {error}"),
    };
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let incoming = read_request(&mut stream, 4096).unwrap();
        assert_eq!(incoming.id, "request-id");
        assert!(matches!(incoming.command, Command::Shutdown));
        serde_json::to_writer(
            &mut stream,
            &Response {
                protocol: 1,
                id: incoming.id,
                result: ResultPayload::Shutdown,
            },
        )
        .unwrap();
        stream.write_all(b"\n").unwrap();
    });

    let response = send_request(&socket, &request(Command::Shutdown)).unwrap();
    assert_eq!(response.id, "request-id");
    assert!(matches!(response.result, ResultPayload::Shutdown));
    server.join().unwrap();

    assert!(send_request(&root.join("missing.sock"), &request(Command::Status)).is_err());

    let invalid_socket = root.join("invalid.sock");
    let listener = UnixListener::bind(&invalid_socket).unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(&mut stream).read_line(&mut line).unwrap();
        stream.write_all(b"invalid\n").unwrap();
    });
    assert!(send_request(&invalid_socket, &request(Command::Status)).is_err());
    server.join().unwrap();
}

#[test]
fn response_printing_propagates_daemon_errors() {
    print_response(Response {
        protocol: 1,
        id: "ok".into(),
        result: ResultPayload::Shutdown,
    })
    .unwrap();
    let error = print_response(Response::error("bad", "invalid", "bad request")).unwrap_err();
    assert_eq!(error.to_string(), "invalid: bad request");
}

#[test]
fn daemon_request_handler_validates_and_dispatches_all_commands() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.daemon.max_text_bytes = 8;
    let engine = FakeEngine {
        fail: false,
        runtime: Runtime::Default,
    };

    let response = handle_request(
        &engine,
        &config,
        &paths,
        request(Command::Say {
            text: "hello".into(),
            speed: 1.0,
            voice: 0,
            output: None,
            no_play: true,
        }),
    );
    match response.result {
        ResultPayload::Synthesis {
            output,
            sample_rate,
            samples,
            audio_seconds,
            load_milliseconds,
            synthesis_milliseconds,
        } => {
            assert_eq!(output, paths.state_dir.join("last.wav").to_string_lossy());
            assert_eq!(sample_rate, 16_000);
            assert_eq!(samples, 8_000);
            assert_eq!(audio_seconds, 0.5);
            assert_eq!(load_milliseconds, 5);
            assert_eq!(synthesis_milliseconds, 7);
        }
        _ => panic!("expected synthesis"),
    }

    let too_long = handle_request(
        &engine,
        &config,
        &paths,
        request(Command::Say {
            text: "ninebytes".into(),
            speed: 1.0,
            voice: 0,
            output: None,
            no_play: true,
        }),
    );
    assert!(matches!(too_long.result, ResultPayload::Error { .. }));

    let failing = FakeEngine {
        fail: true,
        runtime: Runtime::Cuda,
    };
    let failed = handle_request(
        &failing,
        &config,
        &paths,
        request(Command::Say {
            text: "hello".into(),
            speed: 1.0,
            voice: 0,
            output: Some("explicit.wav".into()),
            no_play: true,
        }),
    );
    assert!(matches!(failed.result, ResultPayload::Error { .. }));

    let status = handle_request(&engine, &config, &paths, request(Command::Status));
    let status = serde_json::to_value(status).unwrap();
    assert_eq!(status["type"], "status");
    assert_eq!(status["running"], true);
    assert_eq!(status["model"], "fixture");
    assert_eq!(status["backend"]["effective"]["device"], "cpu");
    assert_eq!(status["backend"]["fallback_used"], true);

    config.backend.device = "gpu".into();
    let status = status_payload(&failing, &config);
    let status = serde_json::to_value(status).unwrap();
    assert_eq!(status["backend"]["effective"]["device"], "gpu");
    assert_eq!(status["backend"]["placement_verified"], false);

    let shutdown = handle_request(&engine, &config, &paths, request(Command::Shutdown));
    assert!(matches!(shutdown.result, ResultPayload::Shutdown));
}

#[test]
fn config_helpers_cover_supported_values_defaults_and_schema() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();

    let assignments = [
        ("backend.kind", "custom"),
        ("backend.runtime", "OPENVINO"),
        ("backend.device", "npu"),
        ("backend.threads", "7"),
        ("backend.fallback", "CPU"),
        ("backend.device_id", "2"),
        ("backend.provider_config", "provider.json"),
        ("model.family", "vits"),
        ("model.name", "custom-model"),
        ("model.directory", "/models/custom"),
        ("model.model_file", "model.onnx"),
        ("model.tokens_file", "tokens.txt"),
        ("model.data_directory", "data"),
        ("model.duration_predictor", "duration.onnx"),
        ("model.text_encoder", "text.onnx"),
        ("model.vector_estimator", "vector.onnx"),
        ("model.vocoder", "vocoder.onnx"),
        ("model.tts_json", "tts.json"),
        ("model.unicode_indexer", "unicode.bin"),
        ("model.voice_style", "voice.bin"),
        ("model.language", "fr"),
        ("model.steps", "8"),
        ("model.voice", "3"),
        ("audio.device", "speakers"),
        ("audio.volume", "0.5"),
    ];
    for (key, value) in assignments {
        set_config(&mut config, key, value).unwrap();
    }
    assert_eq!(config.backend.runtime, Runtime::Openvino);
    assert_eq!(config.backend.fallback, Fallback::Cpu);
    assert_eq!(config.backend.threads, 7);
    assert_eq!(config.model.voice, 3);
    assert_eq!(config.model.language, "fr");
    assert_eq!(config.model.steps, 8);
    assert_eq!(config.audio.volume, 0.5);
    assert!(set_config(&mut config, "unknown", "x").is_err());
    assert!(set_config(&mut config, "backend.threads", "many").is_err());

    save_config(&paths.config_file, &config).unwrap();
    let value = serde_json::to_value(Config::load(&paths.config_file).unwrap()).unwrap();
    assert_eq!(dotted_get(&value, "backend.threads"), Some(&json!(7)));
    assert!(dotted_get(&value, "backend.missing").is_none());
    let description = schema(&paths.config_file, &paths).unwrap();
    assert_eq!(description["app"], "omaspeak");
    assert_eq!(description["schema_version"], 1);
    assert!(
        description["keys"]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key["key"] == "model.duration_predictor")
    );

    for (key, _) in assignments {
        unset_config(&mut config, key).unwrap();
    }
    assert!(unset_config(&mut config, "unknown").is_err());
    let defaults = Config::default();
    assert_eq!(config.backend.kind, defaults.backend.kind);
    assert_eq!(config.model.name, defaults.model.name);
    assert_eq!(config.audio.volume, defaults.audio.volume);

    assert_eq!(parse_runtime("DEFAULT").unwrap(), Runtime::Default);
    assert_eq!(parse_runtime("cuda").unwrap(), Runtime::Cuda);
    assert!(parse_runtime("metal").is_err());
    assert_eq!(parse_fallback("error").unwrap(), Fallback::Error);
    assert!(parse_fallback("maybe").is_err());
}

#[test]
fn cli_parser_and_catalog_helpers_cover_command_surface() {
    let commands = [
        vec!["omaspeak", "daemon"],
        vec!["omaspeak", "status", "--json"],
        vec![
            "omaspeak",
            "say",
            "hello",
            "--voice",
            "2",
            "--speed",
            "1.2",
            "--out",
            "a.wav",
            "--no-play",
        ],
        vec![
            "omaspeak",
            "benchmark",
            "--text",
            "hello",
            "--out-dir",
            "benchmark",
            "--warmup",
            "2",
            "--iterations",
            "4",
        ],
        vec!["omaspeak", "stop"],
        vec!["omaspeak", "voices", "--json"],
        vec!["omaspeak", "config", "schema", "--json"],
        vec!["omaspeak", "config", "get", "model.name", "--json"],
        vec!["omaspeak", "config", "set", "model.voice", "1"],
        vec!["omaspeak", "config", "unset", "model.voice"],
        vec!["omaspeak", "setup"],
        vec!["omaspeak", "setup", "check", "--json"],
        vec!["omaspeak", "setup", "runtime", "--json"],
        vec!["omaspeak", "setup", "model", "--list"],
        vec![
            "omaspeak",
            "setup",
            "model",
            "--download",
            "en_US-lessac-medium",
            "--no-activate",
            "--progress-format",
            "json",
        ],
        vec!["omaspeak", "setup", "systemd", "--no-start"],
        vec!["omaspeak", "setup", "menu", "--status"],
        vec![
            "omaspeak",
            "setup",
            "all",
            "--no-start",
            "--progress-format",
            "json",
        ],
    ];
    for args in commands {
        assert!(Cli::try_parse_from(args).is_ok());
    }
    assert!(Cli::try_parse_from(["omaspeak", "unknown"]).is_err());
    for args in [
        ["omaspeak", "setup", "model", "--list", "--json"],
        ["omaspeak", "setup", "model", "--json", "--set"],
        ["omaspeak", "setup", "model", "--list", "--verify"],
    ] {
        assert!(Cli::try_parse_from(args).is_err());
    }
    assert!(
        Cli::try_parse_from([
            "omaspeak",
            "setup",
            "model",
            "--json",
            "--download",
            "en_US-lessac-medium",
        ])
        .is_err()
    );
    assert!(Cli::try_parse_from(["omaspeak", "benchmark", "--text", "hello"]).is_err());
    assert!(
        Cli::try_parse_from([
            "omaspeak",
            "benchmark",
            "--text",
            "hello",
            "--out-dir",
            "out",
            "--iterations",
            "0",
        ])
        .is_err()
    );
    assert!(model_spec("en_US-lessac-medium").is_ok());
    assert!(
        model_spec("missing")
            .unwrap_err()
            .to_string()
            .contains("unknown model")
    );
    let first = request_id();
    let second = request_id();
    assert!(first.starts_with(&format!("{}-", std::process::id())));
    assert_ne!(first, second);
}

#[test]
fn benchmark_writes_deterministic_outputs_and_reports_timings() {
    let root = sandbox();
    let output = root.join("benchmark");
    let base = Instant::now();
    let mut clock = [
        base,
        base + Duration::from_millis(5),
        base + Duration::from_millis(5),
        base + Duration::from_millis(20),
        base + Duration::from_millis(20),
        base + Duration::from_millis(45),
    ]
    .into_iter();
    let mut outputs = Vec::new();
    let iterations = benchmark_syntheses(
        "hello",
        &output,
        2,
        3,
        |path| {
            outputs.push(path.to_owned());
            let synthesis_time = match path.file_stem().unwrap().to_str().unwrap() {
                "iteration-0001" => Duration::from_millis(10),
                "iteration-0002" => Duration::from_millis(20),
                "iteration-0003" => Duration::from_millis(30),
                _ => Duration::from_millis(1),
            };
            Ok(Synthesis {
                output: path.to_owned(),
                sample_rate: 1_000,
                samples: 500,
                synthesis_time,
            })
        },
        || clock.next().unwrap(),
    )
    .unwrap();

    assert_eq!(outputs.len(), 5);
    assert!(outputs[0].ends_with("benchmark/warmup-0001.wav"));
    assert!(outputs[1].ends_with("benchmark/warmup-0002.wav"));
    assert!(outputs[4].ends_with("benchmark/iteration-0003.wav"));
    assert_eq!(iterations[0].iteration, 1);
    assert_eq!(iterations[0].elapsed_milliseconds, 5.0);
    assert_eq!(iterations[1].elapsed_milliseconds, 15.0);
    assert_eq!(iterations[2].elapsed_milliseconds, 25.0);
    assert_eq!(iterations[0].synthesis_milliseconds, 10.0);
    assert_eq!(iterations[0].audio_duration_milliseconds, 500.0);
    assert!((iterations[0].real_time_factor - 0.02).abs() < f64::EPSILON);

    let summary = benchmark_summary(&iterations);
    assert_eq!(summary.samples, 3);
    assert_eq!(summary.p50_elapsed_milliseconds, Some(15.0));
    assert_eq!(summary.p95_elapsed_milliseconds, Some(25.0));
    assert_eq!(summary.p50_synthesis_milliseconds, Some(20.0));
    assert_eq!(summary.p95_synthesis_milliseconds, Some(30.0));
    assert!((summary.p50_real_time_factor.unwrap() - 0.04).abs() < f64::EPSILON);
    assert!((summary.p95_real_time_factor.unwrap() - 0.06).abs() < f64::EPSILON);

    let args = BenchmarkArgs {
        text: "hello".into(),
        out_dir: output,
        warmup: 2,
        iterations: 3,
    };
    let engine = FakeEngine {
        fail: false,
        runtime: Runtime::Cuda,
    };
    let report = benchmark_report(&Config::default(), &engine, &args, iterations).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["benchmark"], "omaspeak-file-synthesis");
    assert_eq!(report["model_load_milliseconds"], 5);
    assert_eq!(report["backend"]["kind"], "fake");
    assert_eq!(report["backend"]["effective_runtime"], "cuda");
    assert_eq!(report["backend"]["placement_verified"], false);
    assert_eq!(
        report["iterations"][2]["output"],
        outputs[4].to_string_lossy().as_ref()
    );
    assert_eq!(report["summary"]["p50_synthesis_milliseconds"], 20.0);
}

#[test]
fn benchmark_rejects_invalid_text_iterations_and_synthesis_metadata() {
    fn synthesis(path: &Path) -> Result<Synthesis> {
        Ok(Synthesis {
            output: path.to_owned(),
            sample_rate: 1_000,
            samples: 500,
            synthesis_time: Duration::from_millis(1),
        })
    }
    assert!(validate_benchmark_text(" ", 100).is_err());
    assert!(validate_benchmark_text("long", 3).is_err());
    assert!(validate_benchmark_text("okay", 4).is_ok());
    assert!(benchmark_syntheses(" ", Path::new("out"), 0, 1, synthesis, Instant::now).is_err());
    assert!(benchmark_syntheses("x", Path::new("out"), 0, 0, synthesis, Instant::now).is_err());
    assert!(
        benchmark_syntheses(
            "x",
            Path::new("out"),
            1,
            1,
            |_| bail!("warmup failed"),
            Instant::now,
        )
        .is_err()
    );
    assert!(
        benchmark_syntheses(
            "x",
            Path::new("out"),
            0,
            1,
            |_| bail!("synthesis failed"),
            Instant::now,
        )
        .is_err()
    );
    for (sample_rate, samples) in [(0, 1), (1_000, 0)] {
        let base = Instant::now();
        let mut clock = [base, base].into_iter();
        assert!(
            benchmark_syntheses(
                "x",
                Path::new("out"),
                0,
                1,
                |path| {
                    Ok(Synthesis {
                        output: path.to_owned(),
                        sample_rate,
                        samples,
                        synthesis_time: Duration::from_millis(1),
                    })
                },
                || clock.next().unwrap(),
            )
            .is_err()
        );
    }
    assert_eq!(benchmark_summary(&[]).p50_elapsed_milliseconds, None);
    assert_eq!(percentile(&[30.0, 10.0, 20.0], 0.5), Some(20.0));
    assert_eq!(milliseconds(Duration::from_micros(1_500)), 1.5);
}

#[test]
fn playback_tries_players_in_order_and_reports_failure() {
    use std::os::unix::process::ExitStatusExt;

    let path = Path::new("voice.wav");
    let mut attempts = Vec::new();
    assert!(
        play_with(path, |program, received_path| {
            assert_eq!(received_path, path);
            attempts.push(program.to_owned());
            Ok(std::process::ExitStatus::from_raw(1 << 8))
        })
        .is_err()
    );
    assert_eq!(attempts, ["pw-play", "aplay"]);

    let mut attempts = Vec::new();
    play_with(path, |program, _| {
        attempts.push(program.to_owned());
        Ok(std::process::ExitStatus::from_raw(0))
    })
    .unwrap();
    assert_eq!(attempts, ["pw-play"]);
}

#[test]
fn in_memory_daemon_loop_replies_to_bad_status_and_shutdown_requests() {
    let root = sandbox();
    let paths = paths(&root);
    let config = Config::default();
    let engine = FakeEngine {
        fail: false,
        runtime: Runtime::Default,
    };
    let outputs: Vec<_> = (0..3).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
    let mut invalid = MemoryStream::request(Command::Status, outputs[0].clone());
    invalid.input = std::io::Cursor::new(b"invalid json\n".to_vec());
    let mut streams = std::collections::VecDeque::from([
        invalid,
        MemoryStream::request(Command::Status, outputs[1].clone()),
        MemoryStream::request(Command::Shutdown, outputs[2].clone()),
    ]);

    serve_requests(&engine, &config, &paths, || {
        streams.pop_front().ok_or_else(|| anyhow!("no client"))
    })
    .unwrap();

    for output in &outputs {
        let response: Response = serde_json::from_slice(&output.lock().unwrap()).unwrap();
        assert_eq!(response.protocol, 1);
    }
    let invalid: Response = serde_json::from_slice(&outputs[0].lock().unwrap()).unwrap();
    assert!(matches!(invalid.result, ResultPayload::Error { .. }));
    let shutdown: Response = serde_json::from_slice(&outputs[2].lock().unwrap()).unwrap();
    assert!(matches!(shutdown.result, ResultPayload::Shutdown));

    let failed_output = Arc::new(Mutex::new(Vec::new()));
    let mut failed = MemoryStream::request(Command::Status, failed_output.clone());
    failed.fail_write = true;
    let shutdown_output = Arc::new(Mutex::new(Vec::new()));
    let mut streams = std::collections::VecDeque::from([
        failed,
        MemoryStream::request(Command::Shutdown, shutdown_output.clone()),
    ]);
    serve_requests(&engine, &config, &paths, || {
        streams.pop_front().ok_or_else(|| anyhow!("done"))
    })
    .unwrap();
    assert!(failed_output.lock().unwrap().is_empty());
    let shutdown: Response = serde_json::from_slice(&shutdown_output.lock().unwrap()).unwrap();
    assert!(matches!(shutdown.result, ResultPayload::Shutdown));

    let mut failed_shutdown =
        MemoryStream::request(Command::Shutdown, Arc::new(Mutex::new(Vec::new())));
    failed_shutdown.fail_write = true;
    let mut failed_shutdown = Some(failed_shutdown);
    serve_requests(&engine, &config, &paths, || {
        failed_shutdown.take().ok_or_else(|| anyhow!("done"))
    })
    .unwrap();

    assert!(
        serve_requests::<MemoryStream>(&engine, &config, &paths, || bail!("accept failed"))
            .is_err()
    );
}

#[test]
fn daemon_cleanup_runs_after_success_and_failure() {
    for (name, result) in [
        ("success", Ok(())),
        ("failure", Err(anyhow!("serve failed"))),
    ] {
        let root = sandbox().join(name);
        fs::create_dir_all(&root).unwrap();
        let socket = root.join("control.sock");
        fs::write(&socket, b"socket placeholder").unwrap();

        let finished = finish_daemon(&socket, result);
        assert!(!socket.exists());
        assert_eq!(finished.is_err(), name == "failure");
    }

    finish_daemon(&sandbox().join("already-gone.sock"), Ok(())).unwrap();

    let directory_instead_of_socket = sandbox().join("socket-directory");
    fs::create_dir_all(&directory_instead_of_socket).unwrap();
    finish_daemon(&directory_instead_of_socket, Ok(())).unwrap();
    assert!(directory_instead_of_socket.is_dir());
}

#[test]
fn production_engine_adapter_exposes_loaded_backend_metadata() {
    let root = sandbox();
    let paths = paths(&root);
    let config = Config::default();
    let engine =
        Engine::load_with(&config, &paths, |_, _, _| Ok(Box::new(LibraryBackend))).unwrap();
    let adapter: &dyn SpeechEngine = &engine;
    assert_eq!(adapter.backend_kind(), "library-fake");
    assert_eq!(adapter.model_name(), config.model.name);
    assert_eq!(adapter.sample_rate(), 24_000);
    assert_eq!(adapter.effective_runtime(), Runtime::Default);
    assert!(!adapter.fallback_used());
    assert!(adapter.load_milliseconds() < 1000);
    let result = adapter
        .synthesize("hello", 1.0, 0, &root.join("adapter.wav"))
        .unwrap();
    assert_eq!(result.samples, 1);
}

#[test]
fn config_override_becomes_the_effective_app_paths_config_file() {
    let root = sandbox();
    let mut paths = paths(&root);
    let discovered = paths.config_file.clone();
    assert_eq!(select_config_path(None, &mut paths), discovered);
    assert_eq!(paths.config_file, discovered);

    let custom = root.join("custom/config.toml");
    assert_eq!(select_config_path(Some(custom.clone()), &mut paths), custom);
    assert_eq!(paths.config_file, custom);
}

#[test]
fn socket_presence_routes_commands_through_connection_errors() {
    let root = sandbox();
    let paths = paths(&root);
    Config::default().save(&paths.config_file).unwrap();
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    fs::write(paths.socket(), b"not a socket").unwrap();

    assert!(
        say(
            &paths.config_file,
            &paths,
            SayArgs {
                text: Some("hello".into()),
                voice: None,
                speed: None,
                out: None,
                no_play: true,
            },
        )
        .is_err()
    );
    assert!(print_status(&paths.config_file, &paths, false).is_err());
    assert!(stop(&paths).is_err());
}

#[test]
fn model_setup_dispatches_list_verify_set_and_install_actions() {
    let root = sandbox();
    let paths = paths(&root);
    let available = FakeModelOperations { installed: false };
    let installed = FakeModelOperations { installed: true };

    setup_model(
        &paths.config_file,
        &paths,
        &available,
        true,
        false,
        None,
        None,
        None,
        None,
        false,
        ProgressFormat::Human,
    )
    .unwrap();
    print_models_with(&paths, &installed);
    setup_model(
        &paths.config_file,
        &paths,
        &installed,
        false,
        true,
        None,
        None,
        None,
        None,
        false,
        ProgressFormat::Json,
    )
    .unwrap();
    setup_model(
        &paths.config_file,
        &paths,
        &installed,
        false,
        false,
        None,
        None,
        Some("en_US-lessac-medium".into()),
        None,
        false,
        ProgressFormat::Human,
    )
    .unwrap();
    setup_model(
        &paths.config_file,
        &paths,
        &installed,
        false,
        false,
        None,
        Some("en_US-lessac-medium".into()),
        None,
        None,
        false,
        ProgressFormat::Human,
    )
    .unwrap();
    assert_eq!(Config::load(&paths.config_file).unwrap().model.voice, 0);

    setup_model(
        &paths.config_file,
        &paths,
        &available,
        false,
        false,
        Some("en_US-lessac-medium".into()),
        None,
        None,
        Some(root.join("archive")),
        false,
        ProgressFormat::Human,
    )
    .unwrap();
    setup_model(
        &paths.config_file,
        &paths,
        &available,
        false,
        false,
        Some("en_US-lessac-medium".into()),
        None,
        None,
        None,
        true,
        ProgressFormat::Json,
    )
    .unwrap();
    assert!(
        setup_model(
            &paths.config_file,
            &paths,
            &available,
            false,
            false,
            None,
            None,
            Some("missing".into()),
            None,
            false,
            ProgressFormat::Human,
        )
        .is_err()
    );
}

#[test]
fn guided_runtime_preselects_current_values_and_saves_selection() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.device = "cpu".into();
    config.save(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(0), Some(1)]);

    let selected = guided_runtime(&paths.config_file, &mut selector)
        .unwrap()
        .unwrap();
    assert_eq!(selected, (Runtime::Default, "cpu".into()));
    assert_eq!(selector.calls[0].0, "Omaspeak runtime");
    assert_eq!(selector.calls[0].2, 0);
    assert!(selector.calls[0].1[0].enabled);
    assert!(selector.calls[0].1.iter().any(|item| {
        item.label == "CUDA" && !item.enabled && item.detail.contains("Unavailable")
    }));
    assert_eq!(selector.calls[1].0, "Omaspeak device");
    assert_eq!(selector.calls[1].2, 1);
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.device,
        "cpu"
    );
}

#[test]
fn guided_runtime_rejects_npu_when_active_catalog_model_is_incompatible() {
    let root = sandbox();
    let paths = paths(&root);
    let config = Config::default();
    config.save(&paths.config_file).unwrap();
    let original = fs::read_to_string(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(1), Some(3)]);

    let error = guided_runtime(&paths.config_file, &mut selector).unwrap_err();

    assert!(error.to_string().contains("not validated for Intel NPU"));
    assert!(error.to_string().contains("choose Full setup"));
    assert_eq!(fs::read_to_string(&paths.config_file).unwrap(), original);
}

#[test]
fn guided_model_marks_active_installed_and_downloadable_models() {
    let root = sandbox();
    let paths = paths(&root);
    Config::default().save(&paths.config_file).unwrap();
    let installed = FakeModelOperations { installed: true };
    let mut selector = ScriptedSelector::new([Some(0)]);
    let selected =
        choose_model(&paths.config_file, &paths, &installed, None, &mut selector).unwrap();
    assert_eq!(selected.as_deref(), Some("en_US-lessac-medium"));
    assert!(selector.calls[0].1[0].label.contains("● active"));
    assert!(selector.calls[0].1[1].label.contains("○ installed"));
    assert!(
        selector.calls[0].1[2]
            .detail
            .contains("Intel NPU validated")
    );

    let available = FakeModelOperations { installed: false };
    let mut selector = ScriptedSelector::new([Some(1)]);
    guided_model(&paths.config_file, &paths, &available, &mut selector).unwrap();
    assert!(selector.calls[0].1[1].label.contains("· download"));
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        "supertonic-3-int8"
    );
}

#[test]
fn model_only_flow_constrains_catalog_for_active_npu_runtime() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "NPU".into();
    config.model.name = "supertonic-3-npu".into();
    config.save(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(2)]);
    let selected = choose_model(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: true },
        Some((config.backend.runtime, &config.backend.device)),
        &mut selector,
    )
    .unwrap();

    assert_eq!(selected.as_deref(), Some("supertonic-3-npu"));
    assert!(!selector.calls[0].1[0].enabled);
    assert!(!selector.calls[0].1[1].enabled);
    assert!(selector.calls[0].1[2].enabled);
    assert_eq!(selector.calls[0].2, 2);
    assert!(selector.calls[0].1[0].detail.contains("Incompatible"));
}

#[test]
fn full_setup_confirmation_cancel_leaves_configuration_untouched() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([Some(0), Some(0), Some(0), Some(1)]);
    guided_full_setup(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        &mut selector,
    )
    .unwrap();

    assert_eq!(selector.calls[3].0, "Apply Omaspeak setup");
    assert_eq!(selector.calls[3].1[0].label, "Apply setup");
    assert!(!paths.config_file.exists());
}

#[test]
fn guided_flows_handle_back_without_mutating_configuration() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([None]);
    assert!(
        guided_runtime(&paths.config_file, &mut selector)
            .unwrap()
            .is_none()
    );
    assert!(!paths.config_file.exists());

    let mut selector = ScriptedSelector::new([None]);
    assert!(
        guided_model(
            &paths.config_file,
            &paths,
            &FakeModelOperations { installed: false },
            &mut selector,
        )
        .unwrap()
        .is_none()
    );
    assert!(!paths.config_file.exists());
}

#[test]
fn top_level_guide_routes_every_choice_and_rejects_invalid_selection() {
    let root = sandbox();
    let paths = paths(&root);
    let operations = FakeModelOperations { installed: false };

    let mut selector = ScriptedSelector::new([None]);
    guided_setup(&paths.config_file, &paths, &operations, &mut selector).unwrap();
    assert_eq!(
        selector.calls[0]
            .1
            .iter()
            .map(|item| item.label.as_str())
            .collect::<Vec<_>>(),
        ["Full setup", "Runtime", "Model", "Check"]
    );

    for selections in [
        vec![Some(0), None],
        vec![Some(1), None],
        vec![Some(2), None],
    ] {
        let mut selector = ScriptedSelector::new(selections);
        guided_setup(&paths.config_file, &paths, &operations, &mut selector).unwrap();
    }

    let mut selector = ScriptedSelector::new([Some(3)]);
    assert!(guided_setup(&paths.config_file, &paths, &operations, &mut selector).is_err());
    let mut selector = ScriptedSelector::new([Some(99)]);
    assert!(
        guided_setup(&paths.config_file, &paths, &operations, &mut selector)
            .unwrap_err()
            .to_string()
            .contains("invalid choice")
    );
}

#[test]
fn runtime_catalog_covers_each_device_matrix_and_back_at_device_picker() {
    assert_eq!(runtime_name(Runtime::Default), "default");
    assert_eq!(runtime_name(Runtime::Openvino), "openvino");
    assert_eq!(runtime_name(Runtime::Cuda), "cuda");
    assert_eq!(
        device_items(Runtime::Default)
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
        ["auto", "cpu"]
    );
    assert_eq!(
        device_items(Runtime::Cuda)
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
        ["auto", "gpu"]
    );
    assert_eq!(
        device_items(Runtime::Openvino)
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
        ["auto", "cpu", "gpu", "npu"]
    );
    assert!(runtime_item("x", "y", true, "z").enabled);
    assert!(!runtime_item("x", "y", false, "z").enabled);

    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([Some(0), None]);
    assert!(
        choose_runtime(&paths.config_file, &mut selector)
            .unwrap()
            .is_none()
    );
}

#[test]
fn confirmed_full_setup_applies_runtime_model_and_install_steps() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([Some(0), Some(1), Some(1), Some(0)]);
    let result = guided_full_setup_with(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: true },
        &mut selector,
        |paths| {
            let path = paths.data_dir.join("applications/omaspeak.desktop");
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, "launcher")?;
            Ok(path)
        },
        |paths, _, start| {
            assert!(start);
            let path = paths.data_dir.join("systemd/omaspeak.service");
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, "service")?;
            Ok(path)
        },
    );
    // Fake model operations do not install bytes, so the final health check
    // fails after all intended mutations have been exercised.
    assert!(result.is_err());
    let config = Config::load(&paths.config_file).unwrap();
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");
    assert_eq!(config.model.name, "supertonic-3-int8");
    assert!(
        paths
            .data_dir
            .join("applications/omaspeak.desktop")
            .is_file()
    );
    assert!(paths.data_dir.join("systemd/omaspeak.service").is_file());
}

#[test]
fn daemon_socket_preparation_creates_private_dirs_and_removes_stale_files() {
    use std::os::unix::fs::PermissionsExt;

    let root = sandbox();
    let paths = paths(&root);
    let socket = prepare_daemon_socket(&paths).unwrap();
    assert_eq!(socket, paths.socket());
    assert!(paths.state_dir.is_dir());
    assert_eq!(
        fs::metadata(&paths.runtime_dir)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );

    fs::write(&socket, b"stale").unwrap();
    assert_eq!(prepare_daemon_socket(&paths).unwrap(), socket);
    assert!(!socket.exists());
}

#[test]
fn setup_all_completes_each_install_step_and_both_output_formats() {
    for (name, format) in [
        ("all-human", ProgressFormat::Human),
        ("all-json", ProgressFormat::Json),
    ] {
        let root = sandbox().join(name);
        fs::create_dir_all(&root).unwrap();
        let paths = paths(&root);
        let operations = FakeModelOperations { installed: true };
        let result = setup_all(
            &paths.config_file,
            &paths,
            &operations,
            "en_US-lessac-medium",
            None,
            true,
            format,
            |paths| {
                let path = paths.data_dir.join("applications/omaspeak.desktop");
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(&path, b"launcher")?;
                Ok(path)
            },
            |paths, _, start| {
                assert!(!start);
                let path = paths.data_dir.join("systemd/omaspeak.service");
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(&path, b"service")?;
                Ok(path)
            },
        );
        // The final health check correctly reports that the fake install did not
        // place real model bytes, after every setup action has completed.
        assert!(result.is_err());
        assert!(paths.config_file.is_file());
    }
}
