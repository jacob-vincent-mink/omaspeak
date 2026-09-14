use super::*;
use std::collections::{BTreeMap, VecDeque};
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
    let _ = fs::remove_dir_all(&path);
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

fn create_stale_stream_socket(path: &Path) -> bool {
    let listener = match UnixListener::bind(path) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return false,
        Err(error) => panic!("bind stale test socket: {error}"),
    };
    drop(listener);

    for _ in 0..100 {
        match UnixStream::connect(path) {
            Err(error) if indicates_stale_socket(error.kind()) => return true,
            Err(error) => panic!("unexpected stale socket error: {error}"),
            Ok(stream) => drop(stream),
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("closed Unix listener continued accepting connections")
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

struct MatrixModelOperations {
    models: &'static [omaspeak::catalog::ModelSpec],
    installed: &'static [&'static str],
    install_fails: bool,
}

struct ErrorSelector;

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
    fn probe_runtime(
        &mut self,
        _: &Config,
        _: &Path,
    ) -> Result<omaspeak::runtime_inventory::Probe> {
        Ok(omaspeak::runtime_inventory::Probe {
            ready: true,
            loadable: true,
            device_accessible: true,
            ..Default::default()
        })
    }

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

fn accept_runtime(_: &Config, _: &Path, _: bool) -> Result<()> {
    Ok(())
}

impl SetupSelector for ErrorSelector {
    fn select(&mut self, _: &str, _: &str, _: &[MenuItem], _: usize) -> Result<Option<usize>> {
        bail!("injected selection failure")
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
        _: Option<&str>,
    ) -> Result<PathBuf> {
        Ok(paths.data_dir.join("models").join(spec.id))
    }
}

impl ModelSetupOperations for MatrixModelOperations {
    fn models(&self) -> &'static [omaspeak::catalog::ModelSpec] {
        self.models
    }

    fn resolve(&self, id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| anyhow!("unknown matrix model {id}"))
    }

    fn verify(&self, _: &AppPaths, spec: &omaspeak::catalog::ModelSpec) -> Result<()> {
        if self.installed.contains(&spec.id) {
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
        _: Option<&str>,
    ) -> Result<PathBuf> {
        if self.install_fails {
            bail!("injected installation failure")
        }
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

    let response = try_send_request(&socket, &request(Command::Shutdown))
        .unwrap()
        .unwrap();
    assert_eq!(response.id, "request-id");
    assert!(matches!(response.result, ResultPayload::Shutdown));
    server.join().unwrap();

    assert!(
        try_send_request(&root.join("missing.sock"), &request(Command::Status))
            .unwrap()
            .is_none()
    );

    let invalid_socket = root.join("invalid.sock");
    let listener = UnixListener::bind(&invalid_socket).unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(&mut stream).read_line(&mut line).unwrap();
        stream.write_all(b"invalid\n").unwrap();
    });
    assert!(try_send_request(&invalid_socket, &request(Command::Status)).is_err());
    server.join().unwrap();

    let status_socket = root.join("status.sock");
    let listener = UnixListener::bind(&status_socket).unwrap();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let incoming = read_request(&mut stream, 4096).unwrap();
            assert!(matches!(incoming.command, Command::Status));
            serde_json::to_writer(
                &mut stream,
                &Response {
                    protocol: 1,
                    id: incoming.id,
                    result: ResultPayload::Status {
                        running: true,
                        pid: 42,
                        model: "test-model".into(),
                        sample_rate: 24_000,
                        backend: json!({"kind": "test"}),
                    },
                },
            )
            .unwrap();
            stream.write_all(b"\n").unwrap();
        }
    });
    let mut status_paths = paths(&root);
    status_paths.runtime_dir = root.clone();
    fs::rename(&status_socket, status_paths.socket()).unwrap();
    print_status(Path::new("unused.toml"), &status_paths, false).unwrap();
    print_status(Path::new("unused.toml"), &status_paths, true).unwrap();
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
        ("backend.library_dirs", "/opt/openvino:/opt/cuda"),
        ("backend.onnxruntime_library", "/opt/ort/libonnxruntime.so"),
        (
            "backend.provider_library",
            "/opt/ort/libonnxruntime_providers_cuda.so",
        ),
        ("backend.openvino_library", "/opt/openvino/libopenvino_c.so"),
        ("backend.openvino_plugins", "/opt/openvino/plugins.xml"),
        ("backend.options.ProfilingFilePrefix", "/tmp/ort-profile"),
        ("model.family", "future-family"),
        ("model.name", "custom-model"),
        ("model.directory", "/models/custom"),
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
        ("model.options.custom_key", "custom-value"),
        ("audio.device", "speakers"),
        ("audio.volume", "0.5"),
        ("daemon.queue_capacity", "12"),
        ("daemon.max_text_bytes", "4096"),
    ];
    for (key, value) in assignments {
        set_config(&mut config, key, value).unwrap();
    }
    assert_eq!(config.backend.runtime, Runtime::Openvino);
    assert_eq!(config.backend.fallback, Fallback::Cpu);
    assert_eq!(config.backend.threads, 7);
    assert_eq!(
        config.backend.library_dirs,
        [PathBuf::from("/opt/openvino"), PathBuf::from("/opt/cuda")]
    );
    assert_eq!(
        config.backend.options.get("ProfilingFilePrefix"),
        Some(&"/tmp/ort-profile".to_string())
    );
    assert_eq!(config.model.voice, 3);
    assert_eq!(config.model.language, "fr");
    assert_eq!(config.model.steps, 8);
    assert_eq!(config.daemon.queue_capacity, 12);
    assert_eq!(config.audio.volume, 0.5);
    assert!(set_config(&mut config, "unknown", "x").is_err());
    assert!(set_config(&mut config, "backend.options.", "x").is_err());
    assert!(set_config(&mut config, "backend.threads", "many").is_err());

    save_config(&paths.config_file, &config).unwrap();
    let value = serde_json::to_value(Config::load(&paths.config_file).unwrap()).unwrap();
    assert_eq!(dotted_get(&value, "backend.threads"), Some(&json!(7)));
    assert!(dotted_get(&value, "backend.missing").is_none());
    let description = schema(&paths.config_file, &paths).unwrap();
    assert_eq!(description["app"], "omaspeak");
    assert_eq!(description["schema_version"], 1);
    let voice = description["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["key"] == "model.voice")
        .unwrap();
    assert_eq!(voice["value"], 3);
    assert!(voice["choices"].is_array());
    let family = description["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["key"] == "model.family")
        .unwrap();
    assert_eq!(family["choices"], json!(["supertonic"]));
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
    assert!(unset_config(&mut config, "model.options.").is_err());
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
fn runtime_config_mutations_reconcile_provider_specific_state() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.device_id = 1;
    config
        .backend
        .options
        .insert("cudnn_conv_algo_search".into(), "HEURISTIC".into());
    config.save(&paths.config_file).unwrap();

    config_command(
        ConfigCommand::Unset {
            key: "backend.runtime".into(),
        },
        &paths.config_file,
        &paths,
    )
    .unwrap();
    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.runtime, Runtime::Default);
    assert_eq!(saved.backend.device, "auto");
    assert_eq!(saved.backend.device_id, 0);
    assert!(saved.backend.options.is_empty());
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
        vec!["omaspeak", "setup", "runtime", "--dir", "/opt/oma-sdk"],
        vec!["omaspeak", "setup", "model", "--list"],
        vec![
            "omaspeak",
            "setup",
            "model",
            "--download",
            "supertonic-3-int8",
            "--no-activate",
            "--progress-format",
            "json",
        ],
        vec!["omaspeak", "setup", "systemd", "--no-start"],
        vec!["omaspeak", "setup", "menu", "--status"],
        vec!["omaspeak", "setup", "all", "--progress-format", "json"],
    ];
    for args in commands {
        assert!(Cli::try_parse_from(args).is_ok());
    }
    use clap::CommandFactory as _;
    let mut command = Cli::command();
    let setup = command.find_subcommand_mut("setup").unwrap();
    let all = setup.find_subcommand_mut("all").unwrap();
    let help = all.render_long_help().to_string();
    assert!(help.contains("leaves the systemd unit unchanged"));
    assert!(help.contains("omaspeak setup systemd"));
    assert!(help.contains("already-active daemon is safely restarted"));
    assert!(!help.contains("--no-start"));
    assert!(
        Cli::try_parse_from(["omaspeak", "setup", "all", "--no-start"]).is_err(),
        "service lifecycle flags belong to the explicit setup systemd command"
    );
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
            "supertonic-3-int8",
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
    assert!(model_spec("supertonic-3-int8").is_ok());
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
        voice: None,
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
        let listener = match UnixListener::bind(&socket) {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind test socket: {error}"),
        };
        let metadata = fs::symlink_metadata(&socket).unwrap();
        drop(listener);

        let finished = finish_daemon(&socket, &metadata, result);
        assert!(!socket.exists());
        assert_eq!(finished.is_err(), name == "failure");
    }

    let absent_root = sandbox();
    let absent_socket = absent_root.join("already-gone.sock");
    let temporary = absent_root.join("temporary.sock");
    let listener = UnixListener::bind(&temporary).unwrap();
    let metadata = fs::symlink_metadata(&temporary).unwrap();
    drop(listener);
    fs::remove_file(&temporary).unwrap();
    finish_daemon(&absent_socket, &metadata, Ok(())).unwrap();

    let directory_instead_of_socket = sandbox().join("socket-directory");
    fs::create_dir_all(&directory_instead_of_socket).unwrap();
    let metadata = fs::symlink_metadata(&directory_instead_of_socket).unwrap();
    finish_daemon(&directory_instead_of_socket, &metadata, Ok(())).unwrap();
    assert!(directory_instead_of_socket.is_dir());
}

#[test]
fn interrupted_daemon_accept_exits_for_graceful_socket_cleanup() {
    let root = sandbox();
    let socket = root.join("control.sock");
    let listener = match UnixListener::bind(socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("bind test socket: {error}"),
    };
    listener.set_nonblocking(true).unwrap();
    let interrupted = AtomicBool::new(true);
    let error = accept_daemon_connection(&listener, &interrupted).unwrap_err();
    assert!(error.downcast_ref::<DaemonInterrupted>().is_some());
}

#[test]
fn daemon_native_boundaries_are_injected_for_deterministic_error_coverage() {
    let interrupted = AtomicBool::new(false);
    let calls = AtomicUsize::new(0);
    let waits = AtomicUsize::new(0);
    let accepted = accept_daemon_connection_with(
        &interrupted,
        || {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
            } else {
                Ok(7)
            }
        },
        || {
            waits.fetch_add(1, Ordering::Relaxed);
        },
    )
    .unwrap();
    assert_eq!(accepted, 7);
    assert_eq!(waits.load(Ordering::Relaxed), 1);

    let error = accept_daemon_connection_with::<()>(
        &interrupted,
        || Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
        || unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("accept daemon request"));
    interrupted.store(true, Ordering::Relaxed);
    assert!(
        accept_daemon_connection_with::<()>(&interrupted, || unreachable!(), || unreachable!())
            .unwrap_err()
            .downcast_ref::<DaemonInterrupted>()
            .is_some()
    );

    finish_daemon_with(Ok(()), || Ok(())).unwrap();
    finish_daemon_with(Ok(()), || bail!("remove failed")).unwrap();
    assert!(finish_daemon_with(Err(anyhow!("serve failed")), || Ok(())).is_err());
}

#[test]
fn daemon_exchange_serializes_requests_and_decodes_responses() {
    let response = Response {
        protocol: 1,
        id: "response".into(),
        result: ResultPayload::Shutdown,
    };
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut stream = MemoryStream {
        input: std::io::Cursor::new({
            let mut bytes = serde_json::to_vec(&response).unwrap();
            bytes.push(b'\n');
            bytes
        }),
        output: output.clone(),
        fail_write: false,
    };
    let decoded = exchange_request(&mut stream, &request(Command::Status)).unwrap();
    assert!(matches!(decoded.result, ResultPayload::Shutdown));
    assert!(output.lock().unwrap().ends_with(b"\n"));

    stream.input = std::io::Cursor::new(b"not-json\n".to_vec());
    assert!(exchange_request(&mut stream, &request(Command::Status)).is_err());
    stream.fail_write = true;
    assert!(exchange_request(&mut stream, &request(Command::Status)).is_err());
}

#[test]
fn say_request_supports_explicit_text_and_piped_stdin_defaults() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.model.voice = 2;
    let expected_output = root.join("spoken.wav");
    let received = build_say_request(
        &config,
        &paths,
        SayArgs {
            text: Some("hello from the client".into()),
            voice: Some(4),
            speed: Some(1.25),
            out: Some(expected_output),
            no_play: true,
        },
        std::io::empty(),
    )
    .unwrap();
    assert_eq!(received.protocol, 1);
    match received.command {
        Command::Say {
            text,
            speed,
            voice,
            output,
            no_play,
        } => {
            assert_eq!(text, "hello from the client");
            assert_eq!(speed, 1.25);
            assert_eq!(voice, 4);
            assert_eq!(
                output.as_deref(),
                Some(root.join("spoken.wav").to_str().unwrap())
            );
            assert!(no_play);
        }
        command => panic!("unexpected command: {command:?}"),
    }

    let received = build_say_request(
        &config,
        &paths,
        SayArgs {
            text: None,
            voice: None,
            speed: None,
            out: None,
            no_play: false,
        },
        std::io::Cursor::new("piped text\n"),
    )
    .unwrap();
    match received.command {
        Command::Say {
            text,
            speed,
            voice,
            output,
            no_play,
        } => {
            assert_eq!(text, "piped text\n");
            assert_eq!(speed, 1.0);
            assert_eq!(voice, 2);
            assert_eq!(
                output.as_deref(),
                Some(paths.state_dir.join("last.wav").to_str().unwrap())
            );
            assert!(!no_play);
        }
        command => panic!("unexpected command: {command:?}"),
    }

    assert!(
        build_say_request(
            &config,
            &paths,
            SayArgs {
                text: Some("  ".into()),
                voice: None,
                speed: None,
                out: None,
                no_play: true,
            },
            std::io::empty(),
        )
        .is_err()
    );
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
fn stale_socket_falls_back_locally_and_offline_commands_clean_it_up() {
    let root = sandbox();
    let paths = paths(&root);
    Config::default().save(&paths.config_file).unwrap();
    fs::create_dir_all(&paths.runtime_dir).unwrap();
    let socket = paths.socket();

    if !create_stale_stream_socket(&socket) {
        return;
    }
    let local_called = Arc::new(AtomicUsize::new(0));
    let called = local_called.clone();
    let response = send_or_handle_locally(&socket, request(Command::Status), move |_| {
        called.fetch_add(1, Ordering::Relaxed);
        Ok(Response {
            protocol: 1,
            id: "local".into(),
            result: ResultPayload::Shutdown,
        })
    })
    .unwrap();
    assert!(matches!(response.result, ResultPayload::Shutdown));
    assert_eq!(local_called.load(Ordering::Relaxed), 1);
    assert!(!socket.exists());

    assert!(create_stale_stream_socket(&socket));
    print_status(&paths.config_file, &paths, false).unwrap();
    assert!(!socket.exists());

    assert!(create_stale_stream_socket(&socket));
    assert_eq!(
        stop(&paths).unwrap_err().to_string(),
        "daemon is not running"
    );
    assert!(!socket.exists());
}

#[test]
fn refused_non_socket_path_is_never_removed() {
    let root = sandbox();
    let socket = root.join("control.sock");
    fs::write(&socket, b"not a socket").unwrap();
    assert!(try_send_request(&socket, &request(Command::Status)).is_err());
    assert_eq!(fs::read(&socket).unwrap(), b"not a socket");
}

#[test]
fn stale_socket_connection_errors_cover_kernel_variants() {
    for kind in [
        std::io::ErrorKind::ConnectionRefused,
        std::io::ErrorKind::ConnectionReset,
        std::io::ErrorKind::ConnectionAborted,
        std::io::ErrorKind::NotFound,
    ] {
        assert!(indicates_stale_socket(kind));
    }
    assert!(!indicates_stale_socket(
        std::io::ErrorKind::PermissionDenied
    ));
}

#[test]
fn offline_socket_paths_cover_local_dispatch_and_safe_errors() {
    use std::os::unix::ffi::OsStringExt;

    let root = sandbox();
    let paths = paths(&root);
    Config::default().save(&paths.config_file).unwrap();
    let socket = paths.socket();

    voices(&paths.config_file, &paths, false).unwrap();
    voices(&paths.config_file, &paths, true).unwrap();

    let response = send_or_handle_locally(&socket, request(Command::Status), |request| {
        Ok(Response {
            protocol: request.protocol,
            id: request.id,
            result: ResultPayload::Shutdown,
        })
    })
    .unwrap();
    assert!(matches!(response.result, ResultPayload::Shutdown));
    print_status(&paths.config_file, &paths, false).unwrap();
    assert_eq!(
        stop(&paths).unwrap_err().to_string(),
        "daemon is not running"
    );

    let regular = root.join("not-a-socket");
    fs::write(&regular, "keep").unwrap();
    let metadata = fs::symlink_metadata(&regular).unwrap();
    assert!(remove_stale_socket(&regular, &metadata).is_err());
    assert_eq!(fs::read_to_string(&regular).unwrap(), "keep");

    let invalid = PathBuf::from(std::ffi::OsString::from_vec(b"bad\0socket".to_vec()));
    assert!(
        connect_daemon(&invalid)
            .unwrap_err()
            .to_string()
            .contains("inspect daemon socket")
    );
    assert_eq!(DaemonInterrupted.to_string(), "daemon interrupted");
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
        Some("supertonic-3-int8".into()),
        None,
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
        Some("supertonic-3-int8".into()),
        None,
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
        Some("supertonic-3-int8".into()),
        None,
        None,
        Some(root.join("archive")),
        None,
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
        Some("supertonic-3-int8".into()),
        None,
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
    let mut selector = ScriptedSelector::new([Some(0), Some(1), Some(0)]);

    let selected = guided_runtime(&paths.config_file, &mut selector)
        .unwrap()
        .unwrap();
    assert_eq!(selected, (Runtime::Default, "cpu".into()));
    assert_eq!(selector.calls[0].0, "Omaspeak runtime");
    assert_eq!(selector.calls[0].2, 0);
    assert!(selector.calls[0].1[0].enabled);
    assert!(selector.calls[0].1.iter().any(|item| {
        item.label == "CUDA" && item.enabled && item.detail.contains("Needs setup")
    }));
    assert_eq!(selector.calls[1].0, "Omaspeak device");
    assert_eq!(selector.calls[1].2, 1);
    assert_eq!(selector.calls[2].0, "Apply Omaspeak runtime");
    assert_eq!(selector.calls[2].1[0].label, "Apply runtime");
    assert_eq!(
        Config::load(&paths.config_file).unwrap().backend.device,
        "cpu"
    );
}

#[test]
fn guided_runtime_change_clears_provider_specific_configuration() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    config
        .backend
        .options
        .insert("device_type".into(), "GPU".into());
    config.save(&paths.config_file).unwrap();

    save_runtime_with(
        &paths.config_file,
        Runtime::Default,
        "cpu",
        None,
        accept_runtime,
    )
    .unwrap();
    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.runtime, Runtime::Default);
    assert_eq!(saved.backend.device, "cpu");
    assert!(saved.backend.options.is_empty());
}

#[test]
fn runtime_directory_setup_discovers_flat_and_sdk_library_layouts() {
    let root = sandbox();
    let paths = paths(&root);
    let bundle = root.join("native-sdk");
    let lib = bundle.join("lib");
    let lib64 = bundle.join("lib64");
    let vendor = bundle.join("runtime/lib/intel64");
    let release = vendor.join("Release");
    for directory in [&lib, &lib64, &vendor, &release] {
        fs::create_dir_all(directory).unwrap();
    }
    fs::write(lib.join("libonnxruntime.so.1.29.0"), b"ort").unwrap();
    fs::write(release.join("libopenvino_c.so.2600"), b"openvino").unwrap();
    fs::create_dir_all(vendor.join("openvino")).unwrap();
    fs::write(vendor.join("openvino/plugins.xml"), b"<ie/>").unwrap();
    fs::write(vendor.join("libopenvino.so.2600"), b"dependency").unwrap();

    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.save(&paths.config_file).unwrap();
    configure_runtime_directory_with(&paths.config_file, &bundle, |_, _, _| Ok(())).unwrap();

    let configured = Config::load(&paths.config_file).unwrap().backend;
    assert!(configured.onnxruntime_library.is_none());
    assert!(configured.provider_library.is_none());
    assert_eq!(
        configured.openvino_library.unwrap(),
        release
            .join("libopenvino_c.so.2600")
            .canonicalize()
            .unwrap()
    );
    assert_eq!(
        configured.openvino_plugins.unwrap(),
        vendor.join("openvino/plugins.xml").canonicalize().unwrap()
    );
    assert!(
        !configured
            .library_dirs
            .contains(&lib64.canonicalize().unwrap())
    );
    for directory in [lib, vendor, release] {
        assert!(
            configured
                .library_dirs
                .contains(&directory.canonicalize().unwrap())
        );
    }

    fs::remove_file(bundle.join("runtime/lib/intel64/Release/libopenvino_c.so.2600")).unwrap();
    assert!(configure_runtime_directory(&paths.config_file, &bundle).is_err());
}

#[test]
fn runtime_directory_setup_accepts_cpu_ort() {
    let root = sandbox();
    let paths = paths(&root);
    let bundle = root.join("cpu-runtime");
    let lib = bundle.join("lib");
    fs::create_dir_all(&lib).unwrap();
    let ort = lib.join("libonnxruntime.so.1.29.0");
    fs::write(&ort, b"ort").unwrap();

    let mut config = Config::default();
    config.backend.runtime = Runtime::Default;
    config.save(&paths.config_file).unwrap();
    configure_runtime_directory_with(&paths.config_file, &bundle, |_, _, _| Ok(())).unwrap();

    let configured = Config::load(&paths.config_file).unwrap().backend;
    assert_eq!(
        configured.onnxruntime_library.unwrap(),
        ort.canonicalize().unwrap()
    );
    assert!(configured.provider_library.is_none());
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

    assert!(
        error
            .to_string()
            .contains("not compatible with direct OpenVINO")
    );
    assert!(error.to_string().contains("choose Full setup"));
    assert_eq!(fs::read_to_string(&paths.config_file).unwrap(), original);
}

#[test]
fn runtime_validation_failure_never_persists_the_staged_selection() {
    let root = sandbox();
    let app_paths = paths(&root);
    let config = Config::default();
    config.save(&app_paths.config_file).unwrap();
    let original = fs::read_to_string(&app_paths.config_file).unwrap();

    let error = save_runtime_with(
        &app_paths.config_file,
        Runtime::Cuda,
        "gpu",
        None,
        |staged, path, explicit_directory| {
            assert_eq!(path, app_paths.config_file);
            assert_eq!(staged.backend.runtime, Runtime::Cuda);
            assert_eq!(staged.backend.device, "gpu");
            assert!(!explicit_directory);
            bail!("injected ABI probe failure")
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("injected ABI probe failure"));
    assert_eq!(
        fs::read_to_string(&app_paths.config_file).unwrap(),
        original
    );

    let new_paths = paths(&sandbox());
    assert!(
        save_runtime_with(
            &new_paths.config_file,
            Runtime::Cuda,
            "gpu",
            None,
            |_, _, _| bail!("new config probe failure"),
        )
        .is_err()
    );
    assert!(!new_paths.config_file.exists());
}

#[test]
fn guided_runtime_cancel_at_apply_leaves_configuration_untouched() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.device = "cpu".into();
    config.save(&paths.config_file).unwrap();
    let original = fs::read_to_string(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(0), Some(0), Some(1)]);

    assert!(
        guided_runtime(&paths.config_file, &mut selector)
            .unwrap()
            .is_none()
    );
    assert_eq!(selector.calls[2].0, "Apply Omaspeak runtime");
    assert_eq!(fs::read_to_string(&paths.config_file).unwrap(), original);
}

fn runtime_report(
    runtime: Runtime,
    loadable: bool,
    device_accessible: Option<bool>,
) -> omaspeak::runtime::LibraryPathReport {
    let name = runtime_name(runtime);
    omaspeak::runtime::LibraryPathReport {
        configured_library_dirs: Vec::new(),
        environment_library_dirs: Vec::new(),
        package_library_dirs: Vec::new(),
        effective_library_dirs: Vec::new(),
        missing_library_dirs: Vec::new(),
        onnxruntime_library: None,
        provider_library: None,
        openvino_library: None,
        openvino_plugins: None,
        runtime_loadable: BTreeMap::from([(name, loadable)]),
        runtime_probe_errors: if loadable {
            BTreeMap::new()
        } else {
            BTreeMap::from([(name, "injected ABI failure".into())])
        },
        runtime_device_accessible: match device_accessible {
            Some(accessible) => BTreeMap::from([(name, accessible)]),
            None => BTreeMap::new(),
        },
        device_probe_errors: if device_accessible == Some(false) {
            BTreeMap::from([(name, "injected device failure".into())])
        } else {
            BTreeMap::new()
        },
        remediation: Vec::new(),
    }
}

#[test]
fn runtime_report_validation_covers_abi_device_and_success_paths() {
    let abi = validate_runtime_report(Runtime::Cuda, &runtime_report(Runtime::Cuda, false, None))
        .unwrap_err();
    assert!(abi.to_string().contains("injected ABI failure"));
    assert!(abi.to_string().contains("configuration was not changed"));

    let device = validate_runtime_report(
        Runtime::Openvino,
        &runtime_report(Runtime::Openvino, true, Some(false)),
    )
    .unwrap_err();
    assert!(device.to_string().contains("injected device failure"));

    validate_runtime_report(
        Runtime::Openvino,
        &runtime_report(Runtime::Openvino, true, Some(true)),
    )
    .unwrap();
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
    assert_eq!(selected.as_deref(), Some("supertonic-3-int8"));
    assert!(selector.calls[0].1[0].label.contains("● active"));
    assert!(
        selector.calls[0].1[1]
            .detail
            .contains("Intel NPU validated")
    );

    let available = FakeModelOperations { installed: false };
    let mut selector = ScriptedSelector::new([Some(0), Some(0), Some(0)]);
    guided_model(&paths.config_file, &paths, &available, &mut selector).unwrap();
    assert!(
        selector.calls[0].1[0]
            .label
            .contains("license acceptance required")
    );
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.name,
        "supertonic-3-int8"
    );
    assert_eq!(selector.calls[1].0, "Omaspeak voice");
    assert_eq!(selector.calls[1].1.len(), 10);
    assert_eq!(selector.calls[2].0, "Model license");
    assert!(selector.calls[2].1[0].label.contains("Accept OpenRAIL-M"));
}

#[test]
fn model_license_prompt_requires_an_explicit_accept_choice() {
    let spec = omaspeak::catalog::model("supertonic-3-int8").unwrap();
    let mut cancelled = ScriptedSelector::new([Some(1)]);
    assert!(!confirm_model_license(spec, false, &mut cancelled).unwrap());
    assert!(cancelled.calls[0].1[0].detail.contains("use restrictions"));

    let mut accepted = ScriptedSelector::new([Some(0)]);
    assert!(confirm_model_license(spec, false, &mut accepted).unwrap());
    assert!(accepted.calls[0].1[0].label.contains(spec.license));
}

#[test]
fn voice_picker_uses_installed_metadata_and_preselects_active_voice() {
    let root = sandbox();
    let paths = paths(&root);
    let spec = omaspeak::catalog::model("supertonic-3-int8").unwrap();
    let mut config = Config::default();
    spec.activate(&mut config);
    config.model.voice = 1;
    config.save(&paths.config_file).unwrap();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(&directory).unwrap();
    let dimensions = [2_i64, 1, 1, 2, 1, 1];
    fs::write(
        directory.join("voice.bin"),
        dimensions
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut selector = ScriptedSelector::new([Some(0)]);

    let selected = choose_voice(&paths.config_file, &paths, spec, true, &mut selector)
        .unwrap()
        .unwrap();

    assert_eq!(selected.id, 0);
    assert_eq!(selector.calls[0].0, "Omaspeak voice");
    assert_eq!(selector.calls[0].1.len(), 2);
    assert_eq!(selector.calls[0].2, 1);
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
    let mut selector = ScriptedSelector::new([Some(1)]);
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
    assert!(selector.calls[0].1[1].enabled);
    assert_eq!(selector.calls[0].2, 1);
    assert!(selector.calls[0].1[0].detail.contains("Incompatible"));
}

#[test]
fn full_setup_confirmation_cancel_leaves_configuration_untouched() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector =
        ScriptedSelector::new([Some(0), Some(0), Some(0), Some(0), Some(0), Some(1)]);
    guided_full_setup_with_validator(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        &mut selector,
        accept_runtime,
        |_| unreachable!(),
        || unreachable!(),
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap();

    assert_eq!(selector.calls[3].0, "Omaspeak voice");
    assert_eq!(selector.calls[4].0, "Model license");
    assert_eq!(selector.calls[5].0, "Apply Omaspeak setup");
    assert_eq!(selector.calls[5].1[0].label, "Apply setup");
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
    assert_eq!(setup_path_list(&[]), "none");
    assert_eq!(
        setup_path_list(&[PathBuf::from("/one"), PathBuf::from("/two")]),
        "/one:/two"
    );
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
    assert!(runtime_item("x", "y", false, "z").enabled);

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
fn only_engine_loading_commands_require_runtime_path_preparation() {
    assert!(command_loads_engine(&TopCommand::Daemon));
    assert!(command_loads_engine(&TopCommand::Say(SayArgs {
        text: Some("test".into()),
        voice: None,
        speed: None,
        out: None,
        no_play: true,
    })));
    assert!(!command_loads_engine(&TopCommand::Setup {
        command: Some(SetupCommand::Check { json: true }),
    }));
    assert!(!command_loads_engine(&TopCommand::Setup {
        command: Some(SetupCommand::Runtime {
            json: true,
            runtime: None,
            device: None,
            dir: None,
            apply: false,
        }),
    }));
    assert!(!command_loads_engine(&TopCommand::Config {
        command: ConfigCommand::Get {
            key: None,
            json: true,
        },
    }));
    assert!(!command_loads_engine(&TopCommand::Config {
        command: ConfigCommand::Set {
            key: "audio.volume".into(),
            value: "0.8".into(),
        },
    }));
    assert!(!command_loads_engine(&TopCommand::Setup {
        command: Some(SetupCommand::All {
            model: "supertonic-3-int8".into(),
            archive: None,
            accept_license: None,
            progress_format: ProgressFormat::Human,
        }),
    }));
}

#[test]
fn confirmed_full_setup_applies_runtime_model_and_launcher_but_leaves_service_untouched() {
    use std::cell::Cell;

    let root = sandbox();
    let paths = paths(&root);
    let service = app_setup::systemd::service_path(&paths);
    fs::create_dir_all(service.parent().unwrap()).unwrap();
    fs::write(&service, "existing service").unwrap();
    let reload_called = Cell::new(false);
    let mut selector =
        ScriptedSelector::new([Some(0), Some(1), Some(0), Some(1), Some(0), Some(0)]);
    let result = guided_full_setup_with_validator(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        &mut selector,
        accept_runtime,
        |paths| {
            let path = paths.data_dir.join("applications/omaspeak.desktop");
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, "launcher")?;
            Ok(path)
        },
        || true,
        |was_active| {
            assert!(was_active);
            reload_called.set(true);
            Ok(true)
        },
        |config, paths| app_setup::print_checks(config, paths, false),
        app_setup::print_checks_event,
    );
    // Fake model operations do not install bytes, so the pre-restart health
    // check fails and the transaction restores the absent config.
    assert!(result.is_err());
    assert!(!paths.config_file.exists());
    assert!(
        paths
            .data_dir
            .join("applications/omaspeak.desktop")
            .is_file()
    );
    assert_eq!(fs::read_to_string(service).unwrap(), "existing service");
    assert!(!reload_called.get());
}

#[test]
fn guided_full_validates_runtime_before_model_launcher_or_service_callbacks() {
    let root = sandbox();
    let paths = paths(&root);
    let original = b"# preserve exact config bytes\n";
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, original).unwrap();
    let mut selector = ScriptedSelector::new([Some(0), Some(1)]);

    let error = guided_full_setup_with_validator(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: true },
        &mut selector,
        |candidate, path, explicit_directory| {
            assert_eq!(candidate.backend.runtime, Runtime::Default);
            assert_eq!(candidate.backend.device, "cpu");
            assert_eq!(path, paths.config_file);
            assert!(!explicit_directory);
            bail!("staged runtime rejected")
        },
        |_| unreachable!("launcher must follow runtime validation"),
        || unreachable!("service probe must follow runtime validation"),
        |_| unreachable!("restart must follow runtime validation"),
        |_, _| unreachable!("checks must follow runtime validation"),
        |_, _| unreachable!("checks must follow runtime validation"),
    )
    .unwrap_err();

    assert!(error.to_string().contains("staged runtime rejected"));
    assert_eq!(selector.calls.len(), 2);
    assert_eq!(fs::read(&paths.config_file).unwrap(), original);
}

#[test]
fn daemon_socket_preparation_creates_private_dirs_and_preserves_non_socket_paths() {
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
    assert!(prepare_daemon_socket(&paths).is_err());
    assert_eq!(fs::read(&socket).unwrap(), b"stale");
}

#[test]
fn setup_all_installs_model_and_launcher_but_leaves_service_untouched() {
    use std::cell::Cell;

    for (name, format) in [
        ("all-human", ProgressFormat::Human),
        ("all-json", ProgressFormat::Json),
    ] {
        let root = sandbox().join(name);
        fs::create_dir_all(&root).unwrap();
        let paths = paths(&root);
        let service = app_setup::systemd::service_path(&paths);
        if matches!(format, ProgressFormat::Json) {
            fs::create_dir_all(service.parent().unwrap()).unwrap();
            fs::write(&service, "existing service").unwrap();
        }
        let operations = FakeModelOperations { installed: true };
        let service_was_active = matches!(format, ProgressFormat::Json);
        let reload_called = Cell::new(false);
        let result = setup_all(
            &paths.config_file,
            &paths,
            &operations,
            "supertonic-3-int8",
            None,
            None,
            Some("OpenRAIL-M"),
            format,
            |paths| {
                let path = paths.data_dir.join("applications/omaspeak.desktop");
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(&path, b"launcher")?;
                Ok(path)
            },
            || service_was_active,
            |was_active| {
                assert_eq!(was_active, service_was_active);
                reload_called.set(true);
                Ok(was_active)
            },
        );
        // The health check runs before restart and rolls back the config when
        // the fake install did not place real model bytes.
        assert!(result.is_err());
        assert!(!paths.config_file.exists());
        assert!(!reload_called.get());
        if matches!(format, ProgressFormat::Json) {
            assert_eq!(fs::read_to_string(service).unwrap(), "existing service");
        } else {
            assert!(!service.exists());
        }
    }
}

#[test]
fn setup_transaction_restores_existing_and_new_configs_on_late_failures() {
    let operations = FakeModelOperations { installed: true };

    let existing_root = sandbox();
    let existing = paths(&existing_root);
    let original = b"# byte-for-byte rollback\n[model]\nvoice = 3\n";
    fs::create_dir_all(existing.config_file.parent().unwrap()).unwrap();
    fs::write(&existing.config_file, original).unwrap();
    let error = setup_all_with_config(
        Config::load(&existing.config_file).unwrap(),
        &existing.config_file,
        &existing,
        &operations,
        "supertonic-3-int8",
        Some(2),
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |_| bail!("launcher failed after config save"),
        || false,
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("launcher failed"));
    assert_eq!(fs::read(&existing.config_file).unwrap(), original);

    let new = paths(&sandbox());
    let restarts = std::cell::Cell::new(0);
    let error = setup_all_with_config(
        Config::default(),
        &new.config_file,
        &new,
        &operations,
        "supertonic-3-int8",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Json,
        |paths| Ok(paths.data_dir.join("omaspeak.desktop")),
        || true,
        |_| {
            restarts.set(restarts.get() + 1);
            Ok(true)
        },
        |_, _| unreachable!(),
        |_, _| bail!("final setup check failed"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("final setup check failed"));
    assert_eq!(restarts.get(), 0);
    assert!(!new.config_file.exists());

    let restart_root = sandbox();
    let restart = paths(&restart_root);
    let original = b"# restore after restart failure\n";
    fs::create_dir_all(restart.config_file.parent().unwrap()).unwrap();
    fs::write(&restart.config_file, original).unwrap();
    let error = setup_all_with_config(
        Config::load(&restart.config_file).unwrap(),
        &restart.config_file,
        &restart,
        &operations,
        "supertonic-3-int8",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |paths| Ok(paths.data_dir.join("omaspeak.desktop")),
        || true,
        |_| bail!("active service restart failed"),
        |_, _| Ok(()),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("restart failed"));
    assert_eq!(fs::read(&restart.config_file).unwrap(), original);
}

#[test]
fn setup_transaction_checks_then_restarts_and_prints_both_formats() {
    let operations = FakeModelOperations { installed: true };
    for (format, active) in [(ProgressFormat::Human, false), (ProgressFormat::Json, true)] {
        let root = sandbox();
        let paths = paths(&root);
        if active {
            let service = app_setup::systemd::service_path(&paths);
            fs::create_dir_all(service.parent().unwrap()).unwrap();
            fs::write(service, "existing").unwrap();
        }
        let order = std::cell::RefCell::new(Vec::new());
        setup_all_with_config(
            Config::default(),
            &paths.config_file,
            &paths,
            &operations,
            "supertonic-3-int8",
            Some(1),
            None,
            Some("OpenRAIL-M"),
            format,
            |paths| Ok(paths.data_dir.join("omaspeak.desktop")),
            || active,
            |was_active| {
                assert_eq!(was_active, active);
                order.borrow_mut().push("restart");
                Ok(was_active)
            },
            |_, _| {
                order.borrow_mut().push("check");
                Ok(())
            },
            |_, _| {
                order.borrow_mut().push("check");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(&*order.borrow(), &["check", "restart"]);
        assert_eq!(Config::load(&paths.config_file).unwrap().model.voice, 1);
    }
}

#[test]
fn builtin_model_boundaries_and_offline_command_validation_are_actionable() {
    let root = sandbox();
    let paths = paths(&root);
    let spec = BuiltinModels.resolve("supertonic-3-int8").unwrap();
    assert_eq!(BuiltinModels.models().len(), 2);
    assert!(BuiltinModels.verify(&paths, spec).is_err());
    assert!(
        BuiltinModels
            .install(
                &paths,
                spec,
                Some(&root.join("missing-archive.tar.bz2")),
                ProgressFormat::Human,
                Some("OpenRAIL-M"),
            )
            .is_err()
    );

    let mut config = Config::default();
    config.daemon.max_text_bytes = 3;
    config.save(&paths.config_file).unwrap();
    assert!(
        say(
            &paths.config_file,
            &paths,
            SayArgs {
                text: Some("four".into()),
                voice: None,
                speed: None,
                out: None,
                no_play: true,
            },
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds")
    );
    assert!(
        benchmark(
            &paths.config_file,
            &paths,
            BenchmarkArgs {
                text: "four".into(),
                out_dir: root.join("out"),
                warmup: 0,
                iterations: 1,
                voice: None,
            },
        )
        .is_err()
    );
    assert!(play(&root.join("missing.wav")).is_err());
}

#[test]
fn guided_setup_cancellation_paths_preserve_configuration() {
    let operations = FakeModelOperations { installed: false };
    for selections in [
        vec![Some(0), Some(0), None],
        vec![Some(0), Some(0), Some(0), None],
        vec![Some(0), Some(0), Some(0), Some(0), Some(1)],
    ] {
        let root = sandbox();
        let paths = paths(&root);
        let mut selector = ScriptedSelector::new(selections);
        guided_full_setup_with_validator(
            &paths.config_file,
            &paths,
            &operations,
            &mut selector,
            accept_runtime,
            |_| unreachable!(),
            || unreachable!(),
            |_| unreachable!(),
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap();
        assert!(!paths.config_file.exists());
    }

    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([None]);
    assert!(
        guided_model(&paths.config_file, &paths, &operations, &mut selector)
            .unwrap()
            .is_none()
    );
    let mut selector = ScriptedSelector::new([None]);
    assert!(
        choose_model(&paths.config_file, &paths, &operations, None, &mut selector)
            .unwrap()
            .is_none()
    );
}

#[test]
fn installed_guided_model_and_voice_validation_cover_local_only_paths() {
    let root = sandbox();
    let paths = paths(&root);
    let config = Config::default();
    let model = config.model_directory(&paths);
    fs::create_dir_all(&model).unwrap();
    fs::write(
        model.join("voice.bin"),
        [10_i64, 1, 1, 10, 1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    config.save(&paths.config_file).unwrap();
    let operations = FakeModelOperations { installed: true };
    let mut selector = ScriptedSelector::new([Some(0), Some(2)]);
    assert_eq!(
        guided_model(&paths.config_file, &paths, &operations, &mut selector)
            .unwrap()
            .as_deref(),
        Some("supertonic-3-int8")
    );
    assert_eq!(Config::load(&paths.config_file).unwrap().model.voice, 2);

    let mut empty = *omaspeak::catalog::model("supertonic-3-int8").unwrap();
    empty.id = "empty-voices";
    empty.name = "empty-voices";
    empty.voices = &[];
    let empty = Box::leak(Box::new(empty));
    let mut selector = ScriptedSelector::new([]);
    assert!(choose_voice(&paths.config_file, &paths, empty, false, &mut selector).is_err());

    let error = setup_all_with_config(
        Config::load(&paths.config_file).unwrap(),
        &paths.config_file,
        &paths,
        &operations,
        "supertonic-3-int8",
        Some(99),
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |_| unreachable!(),
        || false,
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("voice 99"));
}

#[test]
fn unattended_setup_rejects_a_missing_runtime_before_mutating_config() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.onnxruntime_library = Some(root.join("missing-libonnxruntime.so"));
    config.save(&paths.config_file).unwrap();
    let original = fs::read(&paths.config_file).unwrap();

    let error = setup_all(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        "supertonic-3-int8",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |_| unreachable!(),
        || unreachable!(),
        |_| unreachable!(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("runtime candidate rejected"));
    assert_eq!(fs::read(&paths.config_file).unwrap(), original);
}

#[test]
fn catalog_status_matrix_and_guided_model_cancellations_use_existing_paths() {
    let root = sandbox();
    let paths = paths(&root);
    let base = *omaspeak::catalog::model("supertonic-3-int8").unwrap();
    let model = |id, downloadable, requires_acceptance| {
        let mut model = base;
        model.id = id;
        model.name = id;
        model.downloadable = downloadable;
        model.requires_acceptance = requires_acceptance;
        model.npu_capable = false;
        model
    };
    let models = Box::leak(
        vec![
            model("active-user", false, false),
            model("installed", true, false),
            model("user-only", false, false),
            model("acceptance", true, true),
            model("plain-download", true, false),
        ]
        .into_boxed_slice(),
    );
    let operations = MatrixModelOperations {
        models,
        installed: &["installed"],
        install_fails: false,
    };
    let mut config = Config::default();
    config.model.name = "active-user".into();
    config.save(&paths.config_file).unwrap();

    let mut selector = ScriptedSelector::new([Some(4)]);
    assert_eq!(
        choose_model(&paths.config_file, &paths, &operations, None, &mut selector)
            .unwrap()
            .as_deref(),
        Some("plain-download")
    );
    let items = &selector.calls[0].1;
    assert!(items[0].label.contains("active · user-supplied"));
    assert!(!items[0].enabled);
    assert!(items[0].detail.contains("--archive PATH"));
    assert!(items[1].label.contains("installed"));
    assert!(items[2].label.contains("user-supplied only"));
    assert!(items[2].detail.contains("OpenRAIL-M"));
    assert!(items[3].label.contains("license acceptance required"));
    assert!(items[4].label.contains("download"));
    assert!(!items[4].detail.contains("acceptance required"));
    print_models_with(&paths, &operations);

    config.model.name = "plain-download".into();
    config.save(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(4)]);
    choose_model(&paths.config_file, &paths, &operations, None, &mut selector).unwrap();
    assert!(
        selector.calls[0].1[4]
            .label
            .contains("active · download required")
    );

    setup_model(
        &paths.config_file,
        &paths,
        &operations,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        false,
        ProgressFormat::Human,
    )
    .unwrap();

    let mut selector = ScriptedSelector::new([Some(4), None]);
    assert!(
        guided_model(&paths.config_file, &paths, &operations, &mut selector)
            .unwrap()
            .is_none()
    );
    let mut selector = ScriptedSelector::new([Some(3), Some(0), Some(1)]);
    assert!(
        guided_model(&paths.config_file, &paths, &operations, &mut selector)
            .unwrap()
            .is_none()
    );

    assert!(guided_setup(&paths.config_file, &paths, &operations, &mut ErrorSelector).is_err());
    assert!(
        choose_model(
            &paths.config_file,
            &paths,
            &operations,
            None,
            &mut ErrorSelector
        )
        .is_err()
    );
    assert!(
        choose_voice(
            &paths.config_file,
            &paths,
            &models[4],
            false,
            &mut ErrorSelector
        )
        .is_err()
    );

    let failing = MatrixModelOperations {
        models,
        installed: &[],
        install_fails: true,
    };
    let mut selector = ScriptedSelector::new([Some(4), Some(0)]);
    assert!(guided_model(&paths.config_file, &paths, &failing, &mut selector).is_err());
    assert!(
        setup_model(
            &paths.config_file,
            &paths,
            &failing,
            false,
            false,
            Some("plain-download".into()),
            None,
            None,
            None,
            None,
            false,
            ProgressFormat::Human,
        )
        .is_err()
    );
}

#[test]
fn filesystem_daemon_and_report_branches_need_no_native_runtime() {
    let root = sandbox();
    let paths = paths(&root);
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    let runtime = root.join("runtime");
    fs::create_dir_all(&runtime).unwrap();
    fs::write(runtime.join("libonnxruntime.so"), b"fixture").unwrap();
    fs::write(runtime.join("libonnxruntime_providers_cuda.so"), b"fixture").unwrap();

    let mut config = Config::default();
    apply_runtime_directory(&mut config, &paths.config_file, Path::new("../runtime")).unwrap();
    assert_eq!(
        config.backend.onnxruntime_library.as_deref(),
        Some(runtime.join("libonnxruntime.so").as_path())
    );
    config.backend.runtime = Runtime::Cuda;
    apply_runtime_directory(&mut config, &paths.config_file, &runtime).unwrap();
    assert_eq!(
        config.backend.provider_library.as_deref(),
        Some(runtime.join("libonnxruntime_providers_cuda.so").as_path())
    );

    let regular = root.join("not-a-directory");
    fs::write(&regular, b"file").unwrap();
    assert!(apply_runtime_directory(&mut config, &paths.config_file, &regular).is_err());
    let metadata = fs::symlink_metadata(&regular).unwrap();
    finish_daemon(&regular, &metadata, Ok(())).unwrap();
    assert!(regular.is_file());

    let engine = FakeEngine {
        fail: false,
        runtime: Runtime::Default,
    };
    let args = BenchmarkArgs {
        text: "coverage".into(),
        out_dir: root.join("bench"),
        warmup: 0,
        iterations: 0,
        voice: None,
    };
    let report = benchmark_report(&Config::default(), &engine, &args, Vec::new()).unwrap();
    assert_eq!(report["backend"]["placement_verified"], true);
    assert!(
        report["backend"]["placement_evidence"]
            .as_str()
            .unwrap()
            .contains("CPU engine")
    );

    let response = handle_request(
        &engine,
        &Config::default(),
        &paths,
        request(Command::Say {
            text: "play".into(),
            speed: 1.0,
            voice: 0,
            output: Some(root.join("missing.wav").to_string_lossy().into_owned()),
            no_play: false,
        }),
    );
    assert!(matches!(response.result, ResultPayload::Error { .. }));
}

#[test]
fn noninteractive_runtime_setup_persists_selection_and_rejects_an_invalid_library() {
    let root = sandbox();
    let paths = paths(&root);
    apply_runtime_selection(
        &paths.config_file,
        Runtime::Default,
        "cpu",
        None,
        true,
        |_, _| omaspeak::runtime_inventory::Probe {
            loadable: true,
            device_accessible: true,
            ready: true,
            evidence: omaspeak::runtime_inventory::Evidence {
                versions: vec!["injected test runtime".into()],
                available_devices: vec!["cpu".into()],
                selected_device: Some("cpu".into()),
                ..Default::default()
            },
            errors: Vec::new(),
        },
    )
    .unwrap();

    let config = Config::load(&paths.config_file).unwrap();
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");

    let runtime = root.join("runtime-sdk/lib");
    fs::create_dir_all(&runtime).unwrap();
    fs::write(runtime.join("libonnxruntime.so.1"), b"fixture").unwrap();
    let original = fs::read_to_string(&paths.config_file).unwrap();
    let error = setup(
        Some(SetupCommand::Runtime {
            json: false,
            runtime: None,
            device: None,
            dir: Some(root.join("runtime-sdk")),
            apply: true,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("runtime candidate rejected; config unchanged")
    );
    assert_eq!(fs::read_to_string(&paths.config_file).unwrap(), original);
    setup(
        Some(SetupCommand::Runtime {
            json: true,
            runtime: None,
            device: None,
            dir: None,
            apply: false,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap();

    for json in [false, true] {
        setup(
            Some(SetupCommand::Model {
                list: !json,
                json,
                download: None,
                set: None,
                verify: None,
                archive: None,
                accept_license: None,
                no_activate: false,
                progress_format: ProgressFormat::Human,
            }),
            &paths.config_file,
            &paths,
        )
        .unwrap();
    }

    assert!(
        setup(
            Some(SetupCommand::Menu {
                uninstall: false,
                status: true,
            }),
            &paths.config_file,
            &paths,
        )
        .is_err()
    );
    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: false,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: false,
            status: true,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Menu {
            uninstall: true,
            status: false,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap();
    assert!(
        setup(
            Some(SetupCommand::Systemd {
                uninstall: false,
                status: true,
                no_start: false,
            }),
            &paths.config_file,
            &paths,
        )
        .is_err()
    );
}
