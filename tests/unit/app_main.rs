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
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn paths(root: &Path) -> AppPaths {
    AppPaths {
        config_file: root.join("config/config.toml"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
        state_dir: root.join("state"),
        runtime_dir: root.join("run"),
    }
}

fn write_npu_fingerprint_assets(config: &Config, paths: &AppPaths) {
    let model = config.model_directory(paths);
    fs::create_dir_all(&model).unwrap();
    for name in [
        &config.model.duration_predictor,
        &config.model.text_encoder,
        &config.model.vector_estimator,
        &config.model.vocoder,
        &config.model.tts_json,
        &config.model.unicode_indexer,
    ] {
        let path = model.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, name.as_bytes()).unwrap();
    }
    let voices = model.join(&config.model.voice_style);
    fs::create_dir_all(&voices).unwrap();
    for name in omaspeak::catalog::SUPERTONIC_VOICE_NAMES {
        fs::write(voices.join(format!("{name}.json")), name.as_bytes()).unwrap();
    }
    let runtime = paths.data_dir.join("openvino");
    fs::create_dir_all(&runtime).unwrap();
    fs::write(runtime.join("libopenvino_c.so"), b"runtime").unwrap();
    fs::write(runtime.join("plugins.xml"), b"<ie/>").unwrap();
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

struct FailingProofOperations;

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
            device_accessible: Some(true),
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

impl ModelSetupOperations for FailingProofOperations {
    fn models(&self) -> &'static [omaspeak::catalog::ModelSpec] {
        omaspeak::catalog::models()
    }

    fn resolve(&self, id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
        model_spec(id)
    }

    fn verify(&self, _: &AppPaths, _: &omaspeak::catalog::ModelSpec) -> Result<()> {
        Ok(())
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

    fn prove(&self, _: &mut Config, _: &AppPaths) -> Result<()> {
        bail!("injected provider proof failure")
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
        match self.runtime {
            Runtime::Openvino => "openvino",
            Runtime::Hip => "audiocpp",
            _ => "fake",
        }
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
                        audio: serde_json::Value::Null,
                        running: true,
                        pid: 42,
                        language: String::new(),
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
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
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
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
            output: None,
            no_play: true,
        }),
    );
    assert!(matches!(too_long.result, ResultPayload::Error { .. }));

    let cancelled = handle_request_with_cancellation(
        &engine,
        &config,
        &paths,
        request(Command::Say {
            text: "hello".into(),
            speed: 1.0,
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
            output: None,
            no_play: true,
        }),
        || true,
    );
    assert!(matches!(
        cancelled.result,
        ResultPayload::Error { ref code, .. } if code == "cancelled"
    ));

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
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
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
        ("backend.library", "/opt/audiocpp/libaudiocpp.so"),
        ("backend.openvino_library", "/opt/openvino/libopenvino_c.so"),
        ("backend.openvino_plugins", "/opt/openvino/plugins.xml"),
        ("backend.options.session.profile", "/tmp/audiocpp-profile"),
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
        config.backend.options.get("session.profile"),
        Some(&"/tmp/audiocpp-profile".to_string())
    );
    assert_eq!(
        config.model.voice,
        omaspeak::voices::VoiceSelection::Legacy(3)
    );
    assert_eq!(config.model.language, "fr");
    assert_eq!(config.model.steps, 8);
    assert!(set_config(&mut config, "unknown", "x").is_err());
    assert!(set_config(&mut config, "backend.options.", "x").is_err());
    assert!(set_config(&mut config, "backend.threads", "many").is_err());

    config.save(&paths.config_file).unwrap();
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
    assert_eq!(family["choices"], json!(["supertonic", "kokoro"]));
    assert!(
        description["keys"]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key["key"] == "model.duration_predictor")
    );
    let schema_keys = description["keys"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["key"].as_str())
        .collect::<Vec<_>>();
    for required in [
        "backend.fallback",
        "backend.device_id",
        "model.name",
        "daemon.max_text_bytes",
    ] {
        assert!(schema_keys.contains(&required), "schema omitted {required}");
    }
    assert_eq!(
        description["collections"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|entry| entry["prefix"].as_str())
            .collect::<Vec<_>>(),
        ["backend.options.", "model.options."]
    );

    let external_provider = root.join("libaudiocpp-external.so");
    fs::write(&external_provider, b"provider fixture").unwrap();
    config.backend.kind = "audiocpp".into();
    config.backend.runtime = Runtime::Cuda;
    config.backend.library = Some(external_provider);
    config.save(&paths.config_file).unwrap();
    let external_schema = schema(&paths.config_file, &paths).unwrap();
    let runtime = external_schema["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["key"] == "backend.runtime")
        .unwrap();
    assert!(
        runtime["choices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|choice| choice["value"] == "cuda" && choice["available"] == true)
    );

    let human = human_schema_lines(&description).unwrap();
    assert!(
        human
            .iter()
            .any(|line| line.starts_with("backend.device_id\t"))
    );
    assert!(
        human
            .iter()
            .any(|line| line.starts_with("model.options.\t"))
    );

    for (key, _) in assignments {
        unset_config(&mut config, key).unwrap();
    }
    assert!(unset_config(&mut config, "unknown").is_err());
    assert!(unset_config(&mut config, "model.options.").is_err());
    let defaults = Config::default();
    assert_eq!(config.backend.kind, defaults.backend.kind);
    assert_eq!(config.model.name, defaults.model.name);

    assert_eq!(parse_runtime("DEFAULT").unwrap(), Runtime::Default);
    assert_eq!(parse_runtime("cuda").unwrap(), Runtime::Cuda);
    assert!(parse_runtime("metal").is_err());
    assert_eq!(parse_fallback("error").unwrap(), Fallback::Error);
    assert!(parse_fallback("maybe").is_err());
}

#[test]
fn runtime_schema_availability_requires_a_matching_detected_provider() {
    let mut config = Config::default();
    let choices = runtime_schema_choices(&config, false, false, false);
    assert!(choices.iter().all(|choice| choice["available"] == false));

    let choices = runtime_schema_choices(&config, true, false, false);
    assert_eq!(choices[0]["value"], "default");
    assert_eq!(choices[0]["available"], true);
    assert!(
        choices[1..]
            .iter()
            .all(|choice| choice["available"] == false)
    );

    config.backend.runtime = Runtime::Cuda;
    let choices = runtime_schema_choices(&config, true, true, true);
    assert_eq!(choices[1]["value"], "cuda");
    assert_eq!(choices[1]["available"], true);
    assert_eq!(choices[4]["value"], "openvino");
    assert_eq!(choices[4]["available"], true);
    assert_eq!(choices[2]["available"], false);
    assert_eq!(choices[3]["available"], false);
}

#[test]
fn runtime_config_mutations_reconcile_provider_specific_state() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.backend.device_id = 1;
    config.backend.library = Some(root.join("cuda.so"));
    config.backend.library_dirs.push(root.join("cuda"));
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
    assert!(saved.backend.library.is_none());
    assert!(saved.backend.library_dirs.is_empty());
}

#[test]
fn config_mutation_reload_gate_skips_inactive_services() {
    let root = sandbox();
    let paths = paths(&root);
    let reloads = std::cell::Cell::new(0);
    let mut config = Config::default();
    config.model.language = "ja".into();
    let restarted = save_and_reload_active_with(
        config,
        &paths.config_file,
        || false,
        |_| {
            reloads.set(reloads.get() + 1);
            Ok(true)
        },
        || unreachable!("inactive services are not restarted"),
    )
    .unwrap();
    assert!(!restarted);
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.language,
        "ja"
    );
    assert_eq!(reloads.get(), 0);
}

#[test]
fn config_mutation_restarts_active_service_once_and_rolls_back_restart_failure() {
    let root = sandbox();
    let paths = paths(&root);
    let mut original = Config::default();
    original.model.language = "en".into();
    original.save(&paths.config_file).unwrap();

    let mut updated = original.clone();
    updated.model.language = "ja".into();
    let reloads = std::cell::Cell::new(0);
    assert!(
        save_and_reload_active_with(
            updated,
            &paths.config_file,
            || true,
            |_| {
                reloads.set(reloads.get() + 1);
                assert_eq!(
                    Config::load(&paths.config_file).unwrap().model.language,
                    "ja"
                );
                Ok(true)
            },
            || unreachable!("successful reload does not need recovery"),
        )
        .unwrap()
    );
    assert_eq!(reloads.get(), 1);

    let before_failed_update = fs::read(&paths.config_file).unwrap();
    let mut rejected = Config::load(&paths.config_file).unwrap();
    rejected.model.language = "ko".into();
    let reloads = std::cell::Cell::new(0);
    let recovery_restarts = std::cell::Cell::new(0);
    let error = save_and_reload_active_with(
        rejected,
        &paths.config_file,
        || true,
        |_| {
            reloads.set(reloads.get() + 1);
            assert_eq!(
                Config::load(&paths.config_file).unwrap().model.language,
                "ko"
            );
            // Model a daemon that exited while reading the new config.
            Ok(false)
        },
        || {
            recovery_restarts.set(recovery_restarts.get() + 1);
            assert_eq!(fs::read(&paths.config_file).unwrap(), before_failed_update);
            Ok(true)
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("reload active daemon"));
    assert_eq!(reloads.get(), 1);
    assert_eq!(recovery_restarts.get(), 1);
    assert_eq!(fs::read(&paths.config_file).unwrap(), before_failed_update);
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
        vec![
            "omaspeak",
            "setup",
            "runtime",
            "--runtime",
            "cuda",
            "--device",
            "gpu",
            "--device-id",
            "2",
            "--dir",
            "/opt/audiocpp-cuda",
            "--apply",
        ],
        vec!["omaspeak", "setup", "cache", "--prepare", "--json"],
        vec!["omaspeak", "setup", "model", "--list"],
        vec![
            "omaspeak",
            "setup",
            "model",
            "--download",
            "supertonic-3-openvino",
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
    let systemd_help = setup
        .find_subcommand_mut("systemd")
        .unwrap()
        .render_long_help()
        .to_string();
    assert!(systemd_help.contains("Install and enable the unit without starting or restarting it"));
    assert!(
        Cli::try_parse_from(["omaspeak", "setup", "all", "--no-start"]).is_err(),
        "service lifecycle flags belong to the explicit setup systemd command"
    );
    assert!(Cli::try_parse_from(["omaspeak", "unknown"]).is_err());
    assert!(Cli::try_parse_from(["omaspeak", "setup", "runtime", "--device-id", "2"]).is_err());
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
            "supertonic-3-openvino",
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
    assert!(model_spec("supertonic-3-openvino").is_ok());
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
    let report = benchmark_report(&Config::default(), &engine, &args, 0, iterations).unwrap();
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
    let path = Path::new("voice.wav");
    let mut attempts = Vec::new();
    assert!(
        play_with(
            path,
            |program, received_path| {
                assert_eq!(received_path, path);
                attempts.push(program.to_owned());
                ProcessCommand::new("sh").args(["-c", "exit 1"]).spawn()
            },
            || false
        )
        .is_err()
    );
    assert_eq!(attempts, ["pw-play", "aplay"]);

    let mut attempts = Vec::new();
    play_with(
        path,
        |program, _| {
            attempts.push(program.to_owned());
            ProcessCommand::new("sh").args(["-c", "exit 0"]).spawn()
        },
        || false,
    )
    .unwrap();
    assert_eq!(attempts, ["pw-play"]);
}

#[test]
fn daemon_peer_disconnect_terminates_and_reaps_playback_without_fallback() {
    let (server, client) = UnixStream::pair().unwrap();
    let fd = server.as_raw_fd();
    assert!(!socket_peer_disconnected(fd));

    let disconnect = thread::spawn(move || {
        thread::sleep(Duration::from_millis(75));
        drop(client);
    });
    let mut attempts = Vec::new();
    let mut player_pid = None;
    let started = Instant::now();
    let error = play_with(
        Path::new("silent.wav"),
        |program, _| {
            attempts.push(program.to_owned());
            let child = ProcessCommand::new("sleep").arg("30").spawn()?;
            player_pid = Some(child.id());
            Ok(child)
        },
        || socket_peer_disconnected(fd),
    )
    .unwrap_err();
    disconnect.join().unwrap();

    assert!(error.downcast_ref::<PlaybackCancelled>().is_some());
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(attempts, ["pw-play"]);
    let pid = player_pid.unwrap().try_into().unwrap();
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "player {pid} was not reaped"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
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
    config.model.voice = omaspeak::voices::VoiceSelection::Legacy(2);
    assert_eq!(resolve_voice(&config, None).unwrap(), 2);
    assert_eq!(resolve_voice(&config, Some("F3")).unwrap(), 7);
    assert_eq!(resolve_voice(&config, Some("f3")).unwrap(), 7);
    assert_eq!(resolve_voice(&config, Some("4")).unwrap(), 4);
    assert!(resolve_voice(&config, Some("unknown")).is_err());
    assert!(resolve_voice(&config, Some("99")).is_err());
    config.model.name = "custom".into();
    let supertonic_error = resolve_voice(&config, Some("unknown"))
        .unwrap_err()
        .to_string();
    assert!(supertonic_error.contains("M1 (0), M2 (1)"));
    config.model.family = "custom".into();
    let custom_error = resolve_voice(&config, Some("unknown"))
        .unwrap_err()
        .to_string();
    assert!(custom_error.contains("no voice inventory for family custom"));
    config = Config::default();
    config.model.voice = omaspeak::voices::VoiceSelection::Legacy(2);
    let expected_output = root.join("spoken.wav");
    let received = build_say_request_with_terminal(
        &config,
        &paths,
        SayArgs {
            text: Some("hello from the client".into()),
            voice: Some("4".into()),
            speed: Some(1.25),
            out: Some(expected_output),
            no_play: true,
        },
        std::io::empty(),
        false,
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
            assert_eq!(voice, omaspeak::voices::VoiceSelection::Legacy(4));
            assert_eq!(
                output.as_deref(),
                Some(root.join("spoken.wav").to_str().unwrap())
            );
            assert!(no_play);
        }
        command => panic!("unexpected command: {command:?}"),
    }

    let received = build_say_request_with_terminal(
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
        false,
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
            assert_eq!(voice, omaspeak::voices::VoiceSelection::Legacy(2));
            assert_eq!(
                output.as_deref(),
                Some(paths.state_dir.join("last.wav").to_str().unwrap())
            );
            assert!(!no_play);
        }
        command => panic!("unexpected command: {command:?}"),
    }

    assert!(
        build_say_request_with_terminal(
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
            false,
        )
        .is_err()
    );

    struct UnreadableTerminal;
    impl Read for UnreadableTerminal {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("interactive stdin must not be read")
        }
    }
    let error = build_say_request_with_terminal(
        &config,
        &paths,
        SayArgs {
            text: None,
            voice: None,
            speed: None,
            out: None,
            no_play: true,
        },
        UnreadableTerminal,
        true,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("provide text as an argument or pipe text to stdin")
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
fn config_snapshot_restore_handles_missing_existing_and_unreadable_paths() {
    let root = sandbox();
    let path = root.join("nested/config.toml");

    assert!(config_snapshot(&path).unwrap().is_none());
    restore_snapshot(&path, Some(b"original"), "toml.tmp").unwrap();
    assert_eq!(config_snapshot(&path).unwrap().unwrap(), b"original");

    fs::write(path.with_extension("toml.tmp"), b"partial").unwrap();
    restore_snapshot(&path, None, "toml.tmp").unwrap();
    assert!(!path.exists());
    assert!(!path.with_extension("toml.tmp").exists());

    let directory = root.join("directory");
    fs::create_dir(&directory).unwrap();
    assert!(config_snapshot(&directory).is_err());
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
        Some("supertonic-3-openvino".into()),
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
        Some("supertonic-3-openvino".into()),
        None,
        None,
        None,
        false,
        ProgressFormat::Human,
    )
    .unwrap();
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.voice,
        omaspeak::voices::VoiceSelection::Legacy(0)
    );

    setup_model(
        &paths.config_file,
        &paths,
        &available,
        false,
        false,
        Some("supertonic-3-openvino".into()),
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
        Some("supertonic-3-openvino".into()),
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
fn installed_model_with_failed_provider_proof_keeps_active_config_unchanged() {
    let root = sandbox();
    let paths = paths(&root);
    let mut original = Config::default();
    original.model.name = "previous-model".into();
    original.save(&paths.config_file).unwrap();
    let before = fs::read(&paths.config_file).unwrap();
    let error = setup_model(
        &paths.config_file,
        &paths,
        &FailingProofOperations,
        false,
        false,
        Some("supertonic-3-gguf".into()),
        None,
        None,
        None,
        Some("OpenRAIL-M".into()),
        false,
        ProgressFormat::Human,
    )
    .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("model installation succeeded"));
    assert!(message.contains("activation was not saved"));
    assert!(message.contains("setup model --set supertonic-3-gguf"));
    assert_eq!(fs::read(&paths.config_file).unwrap(), before);
}

#[test]
fn native_provider_helpers_cover_setup_defaults_and_custom_models() {
    let root = sandbox();
    let paths = paths(&root);
    let provider = root.join("libaudiocpp.so");
    fs::write(&provider, b"fixture").unwrap();
    let mut config = Config::default();
    config.backend.library = Some(provider.canonicalize().unwrap());

    // A file with the right basename is not evidence of a runnable provider.
    assert!(
        TerminalSetupSelector
            .probe_runtime(&config, &paths.config_file)
            .is_err()
    );

    pin_audio_cpp_library(&mut config, &paths.config_file).unwrap();
    assert!(config.backend.library.as_deref().unwrap().is_file());
    config.backend.kind = "supertonic".into();
    config.backend.library = None;
    pin_audio_cpp_library(&mut config, &paths.config_file).unwrap();
    assert!(config.backend.library.is_none());

    let audio = omaspeak::catalog::model("supertonic-3-gguf").unwrap();
    config.backend.runtime = Runtime::Openvino;
    activate_model_for_setup(audio, &mut config).unwrap();
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");
    config.backend.runtime = Runtime::Cuda;
    activate_model_for_setup(audio, &mut config).unwrap();
    assert_eq!(config.backend.device, "gpu");

    config.model.name = "custom-gguf".into();
    config.model.directory = root.to_string_lossy().into_owned();
    config.model.file = "custom.gguf".into();
    config.backend.kind = "audiocpp".into();
    assert!(!active_model_is_installed(&config, &paths));
    fs::write(root.join("custom.gguf"), b"model").unwrap();
    assert!(active_model_is_installed(&config, &paths));

    config.backend.kind = "supertonic".into();
    assert!(!active_model_is_installed(&config, &paths));

    config.backend.runtime = Runtime::Openvino;
    let openvino = FakeEngine {
        fail: false,
        runtime: Runtime::Openvino,
    };
    let status = serde_json::to_value(status_payload(&openvino, &config)).unwrap();
    assert_eq!(status["backend"]["effective"]["provider"], "openvino");
    assert_eq!(status["backend"]["placement_verified"], true);

    let audiocpp = FakeEngine {
        fail: false,
        runtime: Runtime::Hip,
    };
    let status = serde_json::to_value(status_payload(&audiocpp, &config)).unwrap();
    assert_eq!(status["backend"]["effective"]["provider"], "hip");
    assert_eq!(status["backend"]["placement_verified"], true);
}

#[test]
fn guided_runtime_preselects_current_values_and_saves_selection() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.device = "cpu".into();
    config.save(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(0), Some(1), Some(0)]);

    let selected = guided_runtime(&paths.config_file, &paths, &mut selector)
        .unwrap()
        .unwrap();
    assert_eq!(selected, (Runtime::Default, "cpu".into()));
    assert_eq!(selector.calls[0].0, "Omaspeak runtime");
    assert_eq!(selector.calls[0].2, 0);
    assert!(selector.calls[0].1[0].enabled);
    assert!(selector.calls[0].1.iter().any(|item| {
        item.label == "audio.cpp · CUDA" && item.enabled && item.detail.contains("Needs setup")
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
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "gpu".into();
    config.backend.library_dirs.push(root.join("old-runtime"));
    config.backend.openvino_library = Some(root.join("old-openvino.so"));
    config.backend.openvino_plugins = Some(root.join("old-plugins.xml"));
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
    assert!(saved.backend.library_dirs.is_empty());
    assert!(saved.backend.openvino_library.is_none());
    assert!(saved.backend.openvino_plugins.is_none());
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
    let overlay = root.join("vendor-overlay");
    for directory in [&lib, &lib64, &vendor, &release] {
        fs::create_dir_all(directory).unwrap();
    }
    fs::create_dir_all(&overlay).unwrap();
    fs::write(overlay.join("libvendor.so"), b"vendor").unwrap();
    fs::write(lib.join("libaudiocpp.so.0.1.0"), b"audio.cpp").unwrap();
    fs::write(release.join("libopenvino_c.so.2600"), b"openvino").unwrap();
    fs::create_dir_all(vendor.join("openvino")).unwrap();
    fs::write(vendor.join("openvino/plugins.xml"), b"<ie/>").unwrap();
    fs::write(vendor.join("libopenvino.so.2600"), b"dependency").unwrap();

    let mut config = Config::default();
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.library_dirs.push(overlay.clone());
    config.save(&paths.config_file).unwrap();
    configure_runtime_directory_with(&paths.config_file, &bundle, |_, _, _| Ok(())).unwrap();

    let configured = Config::load(&paths.config_file).unwrap().backend;
    assert!(configured.library.is_none());
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
    assert!(configured.library_dirs.contains(&overlay));

    fs::remove_file(bundle.join("runtime/lib/intel64/Release/libopenvino_c.so.2600")).unwrap();
    assert!(configure_runtime_directory(&paths.config_file, &bundle).is_err());
}

#[test]
fn runtime_directory_setup_accepts_complete_audio_cpp_provider() {
    let root = sandbox();
    let paths = paths(&root);
    let bundle = root.join("cpu-runtime");
    let lib = bundle.join("lib");
    fs::create_dir_all(&lib).unwrap();
    let provider = lib.join("libaudiocpp.so.0.1.0");
    fs::write(&provider, b"audio.cpp").unwrap();

    let mut config = Config::default();
    config.backend.runtime = Runtime::Default;
    config.save(&paths.config_file).unwrap();
    configure_runtime_directory_with(&paths.config_file, &bundle, |_, _, _| Ok(())).unwrap();

    let configured = Config::load(&paths.config_file).unwrap().backend;
    assert_eq!(
        configured.library.unwrap(),
        provider.canonicalize().unwrap()
    );
}

#[test]
fn runtime_directory_setup_accepts_complete_cuda_audio_cpp_provider() {
    let root = sandbox();
    let paths = paths(&root);
    let bundle = root.join("cuda-plugin");
    fs::create_dir_all(&bundle).unwrap();
    let provider = bundle.join("libaudiocpp.so");
    fs::write(&provider, b"provider").unwrap();

    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.device = "gpu".into();
    config.save(&paths.config_file).unwrap();

    configure_runtime_directory_with(&paths.config_file, &bundle, |_, _, _| Ok(())).unwrap();

    let configured = Config::load(&paths.config_file).unwrap().backend;
    assert_eq!(configured.library, Some(provider.canonicalize().unwrap()));
    assert!(
        configured
            .library_dirs
            .contains(&bundle.canonicalize().unwrap())
    );
}

#[test]
fn guided_runtime_allows_npu_selection_before_a_compatible_model_is_installed() {
    let root = sandbox();
    let paths = paths(&root);
    let config = Config::default();
    config.save(&paths.config_file).unwrap();
    let mut selector = ScriptedSelector::new([Some(1), Some(3), Some(0)]);

    let selected = guided_runtime(&paths.config_file, &paths, &mut selector)
        .unwrap()
        .unwrap();

    assert_eq!(selected, (Runtime::Openvino, "npu".into()));
    let saved = Config::load(&paths.config_file).unwrap();
    assert_eq!(saved.backend.runtime, Runtime::Openvino);
    assert_eq!(saved.backend.device, "npu");
    assert!(!omaspeak::supertonic::npu_cache_state(&saved, &paths).ready);
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
        guided_runtime(&paths.config_file, &paths, &mut selector)
            .unwrap()
            .is_none()
    );
    assert_eq!(selector.calls[2].0, "Apply Omaspeak runtime");
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
    assert_eq!(selected.as_deref(), Some("supertonic-3-gguf"));
    assert!(selector.calls[0].1[0].label.contains("● active"));
    assert!(
        selector.calls[0].1[1]
            .detail
            .contains("Intel NPU compatible")
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
        "supertonic-3-gguf"
    );
    assert_eq!(selector.calls[1].0, "Omaspeak voice");
    assert_eq!(selector.calls[1].1.len(), 10);
    assert_eq!(selector.calls[2].0, "Model license");
    assert!(selector.calls[2].1[0].label.contains("Accept OpenRAIL-M"));
}

#[test]
fn model_license_prompt_requires_an_explicit_accept_choice() {
    let spec = omaspeak::catalog::model("supertonic-3-openvino").unwrap();
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
    let spec = omaspeak::catalog::model("supertonic-3-openvino").unwrap();
    let mut config = Config::default();
    spec.activate(&mut config);
    config.model.voice = omaspeak::voices::VoiceSelection::Legacy(1);
    config.save(&paths.config_file).unwrap();
    let directory = config.model_directory(&paths);
    fs::create_dir_all(&directory).unwrap();
    let voices = directory.join("voice_styles");
    fs::create_dir_all(&voices).unwrap();
    for name in omaspeak::catalog::SUPERTONIC_VOICE_NAMES {
        fs::write(voices.join(format!("{name}.json")), b"{}").unwrap();
    }
    let mut selector = ScriptedSelector::new([Some(0)]);

    let selected = choose_voice(&paths.config_file, &paths, spec, true, &mut selector)
        .unwrap()
        .unwrap();

    assert_eq!(selected.id, 0);
    assert_eq!(selector.calls[0].0, "Omaspeak voice");
    assert_eq!(selector.calls[0].1.len(), 10);
    assert_eq!(selector.calls[0].2, 1);
}

#[test]
fn model_only_flow_constrains_catalog_for_active_npu_runtime() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "NPU".into();
    config.model.name = "supertonic-3-openvino".into();
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

    assert_eq!(selected.as_deref(), Some("supertonic-3-openvino"));
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
fn fresh_full_setup_stages_audio_cpp_cpu_and_gguf_together() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector =
        ScriptedSelector::new([Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)]);

    guided_full_setup_with_validator(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        &mut selector,
        |candidate, _, _| {
            assert_eq!(candidate.backend.kind, "audiocpp");
            assert_eq!(candidate.backend.runtime, Runtime::Default);
            assert_eq!(candidate.backend.device, "auto");
            Ok(())
        },
        |paths| Ok(app_setup::menu::launcher_path(paths)),
        || false,
        |was_active| {
            assert!(!was_active);
            Ok(false)
        },
        |_, _| Ok(()),
        |_, _| Ok(()),
    )
    .unwrap();

    let configured = Config::load(&paths.config_file).unwrap();
    assert_eq!(configured.backend.kind, "audiocpp");
    assert_eq!(configured.backend.runtime, Runtime::Default);
    assert_eq!(configured.backend.device, "cpu");
    assert_eq!(configured.model.name, "supertonic-3-gguf");
    assert_eq!(configured.model.file, "supertonic-3-orig.gguf");
    assert!(selector.calls[2].1[0].enabled);
    assert!(!selector.calls[2].1[1].enabled);
}

#[test]
fn guided_flows_handle_back_without_mutating_configuration() {
    let root = sandbox();
    let paths = paths(&root);
    let mut selector = ScriptedSelector::new([None]);
    assert!(
        guided_runtime(&paths.config_file, &paths, &mut selector)
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
        ["Full setup", "Runtime", "Model", "Check", "Audio"]
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
        device_items(Runtime::Vulkan)
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
        ["auto", "gpu"]
    );
    assert_eq!(
        device_items(Runtime::Hip)
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
    assert!(!command_loads_engine(&TopCommand::Daemon));
    assert!(!command_loads_engine(&TopCommand::RequestWorker {
        spec: "{}".into()
    }));
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
            device_id: None,
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
            key: "daemon.max_text_bytes".into(),
            value: "4096".into(),
        },
    }));
    assert!(!command_loads_engine(&TopCommand::Setup {
        command: Some(SetupCommand::All {
            model: Some("supertonic-3-openvino".into()),
            source: None,
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
            let path = app_setup::menu::launcher_path(paths);
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
    assert!(!app_setup::menu::launcher_path(&paths).exists());
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
    let (socket, startup_lock) = prepare_daemon_socket(&paths).unwrap();
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
    assert_eq!(
        fs::metadata(paths.runtime_dir.join("daemon.lock"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    let contention = prepare_daemon_socket(&paths).unwrap_err();
    assert!(
        contention
            .to_string()
            .contains("already running or starting")
    );
    drop(startup_lock);

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
            "supertonic-3-openvino",
            None,
            None,
            Some("OpenRAIL-M"),
            format,
            |paths| {
                let path = app_setup::menu::launcher_path(paths);
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
        assert!(!app_setup::menu::launcher_path(&paths).exists());
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
        "supertonic-3-openvino",
        Some(2),
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |paths| {
            let launcher = app_setup::menu::launcher_path(paths);
            fs::create_dir_all(launcher.parent().unwrap())?;
            fs::write(launcher.with_extension("tmp"), "partial launcher")?;
            bail!("launcher failed after config save")
        },
        || false,
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("launcher failed"));
    assert_eq!(fs::read(&existing.config_file).unwrap(), original);
    assert!(!app_setup::menu::launcher_path(&existing).exists());
    assert!(
        !app_setup::menu::launcher_path(&existing)
            .with_extension("tmp")
            .exists()
    );

    let new = paths(&sandbox());
    let new_launcher = app_setup::menu::launcher_path(&new);
    let restarts = std::cell::Cell::new(0);
    let error = setup_all_with_config(
        Config::default(),
        &new.config_file,
        &new,
        &operations,
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Json,
        |paths| {
            let launcher = app_setup::menu::launcher_path(paths);
            fs::create_dir_all(launcher.parent().unwrap())?;
            fs::write(&launcher, "new launcher")?;
            Ok(launcher)
        },
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
    assert!(!new_launcher.exists());

    let restart_root = sandbox();
    let restart = paths(&restart_root);
    let original = b"# restore after restart failure\n";
    fs::create_dir_all(restart.config_file.parent().unwrap()).unwrap();
    fs::write(&restart.config_file, original).unwrap();
    let launcher = app_setup::menu::launcher_path(&restart);
    fs::create_dir_all(launcher.parent().unwrap()).unwrap();
    fs::write(&launcher, "prior launcher").unwrap();
    let restart_attempts = std::cell::Cell::new(0);
    let error = setup_all_with_config(
        Config::load(&restart.config_file).unwrap(),
        &restart.config_file,
        &restart,
        &operations,
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |paths| {
            let launcher = app_setup::menu::launcher_path(paths);
            fs::write(&launcher, "replacement launcher")?;
            Ok(launcher)
        },
        || true,
        |_| {
            restart_attempts.set(restart_attempts.get() + 1);
            if restart_attempts.get() == 1 {
                bail!("active service restart failed")
            }
            assert_eq!(fs::read(&restart.config_file).unwrap(), original);
            Ok(true)
        },
        |_, _| Ok(()),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("restart failed"));
    assert_eq!(restart_attempts.get(), 2);
    assert_eq!(fs::read(&restart.config_file).unwrap(), original);
    assert_eq!(fs::read_to_string(launcher).unwrap(), "prior launcher");
}

#[test]
fn setup_transaction_rejects_a_launcher_installed_outside_the_expected_path() {
    let root = sandbox();
    let paths = paths(&root);
    let unexpected_launcher = root.join("unexpected.desktop");
    let error = setup_all_with_config(
        Config::default(),
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: true },
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |_| Ok(unexpected_launcher.clone()),
        || false,
        |_| unreachable!(),
        |_, _| unreachable!(),
        |_, _| unreachable!(),
    )
    .unwrap_err();

    assert!(error.to_string().contains("expected"));
    assert!(error.to_string().contains("unexpected.desktop"));
    assert!(!paths.config_file.exists());
    assert!(!app_setup::menu::launcher_path(&paths).exists());
}

#[test]
fn setup_transaction_reports_concurrent_config_and_launcher_rollback_failures() {
    for obstruct_config in [true, false] {
        let root = sandbox();
        let paths = paths(&root);
        let launcher = app_setup::menu::launcher_path(&paths);
        let error = setup_all_with_config(
            Config::default(),
            &paths.config_file,
            &paths,
            &FakeModelOperations { installed: true },
            "supertonic-3-openvino",
            None,
            None,
            Some("OpenRAIL-M"),
            ProgressFormat::Human,
            |paths| {
                let path = app_setup::menu::launcher_path(paths);
                fs::create_dir_all(path.parent().unwrap())?;
                fs::write(&path, "new launcher")?;
                Ok(path)
            },
            || false,
            |_| unreachable!(),
            |config, _| {
                let obstructed = if obstruct_config { config } else { &launcher };
                fs::remove_file(obstructed)?;
                fs::create_dir(obstructed)?;
                bail!("concurrent filesystem change")
            },
            |_, _| unreachable!(),
        )
        .unwrap_err();

        let detail = format!("{error:#}");
        assert!(detail.contains("concurrent filesystem change"));
        assert!(detail.contains("setup rollback was incomplete"));
        assert!(detail.contains(if obstruct_config {
            "restore prior config"
        } else {
            "restore prior launcher"
        }));
    }
}

#[test]
fn setup_transaction_reports_each_failed_service_restore_outcome() {
    for restored_service_errors in [false, true] {
        let root = sandbox();
        let paths = paths(&root);
        fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
        fs::write(&paths.config_file, "# original config\n").unwrap();
        let restart_attempts = std::cell::Cell::new(0);
        let error = setup_all_with_config(
            Config::load(&paths.config_file).unwrap(),
            &paths.config_file,
            &paths,
            &FakeModelOperations { installed: true },
            "supertonic-3-openvino",
            None,
            None,
            Some("OpenRAIL-M"),
            ProgressFormat::Human,
            |paths| Ok(app_setup::menu::launcher_path(paths)),
            || true,
            |_| {
                restart_attempts.set(restart_attempts.get() + 1);
                if restart_attempts.get() == 1 {
                    bail!("initial restart failed")
                } else if restored_service_errors {
                    bail!("restored service restart failed")
                } else {
                    Ok(false)
                }
            },
            |_, _| Ok(()),
            |_, _| unreachable!(),
        )
        .unwrap_err();

        let detail = format!("{error:#}");
        assert!(detail.contains("initial restart failed"));
        assert!(detail.contains("setup rollback was incomplete"));
        assert!(detail.contains(if restored_service_errors {
            "restored service restart failed"
        } else {
            "service remained inactive"
        }));
        assert_eq!(restart_attempts.get(), 2);
    }
}

#[test]
fn setup_precompile_failure_leaves_config_launcher_and_service_untouched() {
    let root = sandbox();
    let paths = paths(&root);
    let original = b"# unchanged while NPU compilation fails\n";
    fs::create_dir_all(paths.config_file.parent().unwrap()).unwrap();
    fs::write(&paths.config_file, original).unwrap();
    let launcher_called = std::cell::Cell::new(false);
    let restart_called = std::cell::Cell::new(false);
    let error = setup_all_with_config_and_preparer(
        Config::load(&paths.config_file).unwrap(),
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: true },
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |_| {
            launcher_called.set(true);
            unreachable!()
        },
        || true,
        |_| {
            restart_called.set(true);
            unreachable!()
        },
        |_, _| unreachable!(),
        |_, _| unreachable!(),
        |_, _, _| bail!("injected NPU compilation failure"),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("NPU compilation failure"));
    assert_eq!(fs::read(&paths.config_file).unwrap(), original);
    assert!(!launcher_called.get());
    assert!(!restart_called.get());
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
            "supertonic-3-openvino",
            Some(1),
            None,
            Some("OpenRAIL-M"),
            format,
            |paths| Ok(app_setup::menu::launcher_path(paths)),
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
        assert_eq!(
            Config::load(&paths.config_file).unwrap().model.voice,
            omaspeak::voices::VoiceSelection::Legacy(1)
        );
    }
}

#[test]
fn builtin_model_boundaries_and_offline_command_validation_are_actionable() {
    let root = sandbox();
    let paths = paths(&root);
    let spec = BuiltinModels.resolve("supertonic-3-openvino").unwrap();
    assert_eq!(BuiltinModels.models().len(), 3);
    assert!(BuiltinModels.verify(&paths, spec).is_err());
    assert!(
        BuiltinModels
            .install(
                &paths,
                spec,
                Some(&root.join("missing-model-directory")),
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
        Some("supertonic-3-gguf")
    );
    assert_eq!(
        Config::load(&paths.config_file).unwrap().model.voice,
        omaspeak::voices::VoiceSelection::Name("M3".into())
    );

    let mut empty = *omaspeak::catalog::model("supertonic-3-openvino").unwrap();
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
        "supertonic-3-openvino",
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
    config.backend.library = Some(root.join("missing-libaudiocpp.so"));
    config.save(&paths.config_file).unwrap();
    let original = fs::read(&paths.config_file).unwrap();

    let error = setup_all(
        &paths.config_file,
        &paths,
        &FakeModelOperations { installed: false },
        "supertonic-3-openvino",
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
    let base = *omaspeak::catalog::model("supertonic-3-openvino").unwrap();
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
    assert!(items[0].detail.contains("--source PATH"));
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

    // Cancellation paths must select a model that the active runtime can run.
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "cpu".into();
    config.save(&paths.config_file).unwrap();
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
    fs::write(runtime.join("libaudiocpp.so"), b"fixture").unwrap();

    let mut config = Config::default();
    apply_runtime_directory(&mut config, &paths.config_file, Path::new("../runtime")).unwrap();
    assert_eq!(
        config.backend.library.as_deref(),
        Some(runtime.join("libaudiocpp.so").as_path())
    );
    config.backend.runtime = Runtime::Cuda;
    apply_runtime_directory(&mut config, &paths.config_file, &runtime).unwrap();
    assert_eq!(
        config.backend.library.as_deref(),
        Some(runtime.join("libaudiocpp.so").as_path())
    );

    let openvino = root.join("openvino");
    fs::create_dir_all(&openvino).unwrap();
    fs::write(openvino.join("libopenvino_c.so"), b"fixture").unwrap();
    let mut direct = Config::default();
    direct.backend.kind = "supertonic".into();
    direct.backend.runtime = Runtime::Openvino;
    let missing_plugins = apply_runtime_directory(&mut direct, &paths.config_file, &openvino)
        .unwrap_err()
        .to_string();
    assert!(missing_plugins.contains("plugins.xml was not found"));
    fs::write(openvino.join("plugins.xml"), b"<ie/>").unwrap();
    apply_runtime_directory(&mut direct, &paths.config_file, &openvino).unwrap();
    assert_eq!(
        direct.backend.openvino_library.as_deref(),
        Some(openvino.join("libopenvino_c.so").as_path())
    );

    direct.backend.runtime = Runtime::Cuda;
    apply_runtime_directory(&mut direct, &paths.config_file, &runtime).unwrap();
    assert_eq!(
        direct.backend.library.as_deref(),
        Some(runtime.join("libaudiocpp.so").as_path())
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
    let report = benchmark_report(&Config::default(), &engine, &args, 0, Vec::new()).unwrap();
    assert_eq!(report["backend"]["placement_verified"], false);
    assert!(
        report["backend"]["placement_evidence"]
            .as_str()
            .unwrap()
            .contains("audio.cpp")
    );
    let openvino = FakeEngine {
        fail: false,
        runtime: Runtime::Openvino,
    };
    let mut openvino_config = Config::default();
    openvino_config.backend.runtime = Runtime::Openvino;
    openvino_config.backend.device = "gpu".into();
    let report = benchmark_report(&openvino_config, &openvino, &args, 0, Vec::new()).unwrap();
    assert!(
        report["backend"]["placement_evidence"]
            .as_str()
            .unwrap()
            .contains("EXECUTION_DEVICES")
    );

    let response = handle_request(
        &engine,
        &Config::default(),
        &paths,
        request(Command::Say {
            text: "play".into(),
            speed: 1.0,
            voice: omaspeak::voices::VoiceSelection::Legacy(0),
            output: Some(root.join("missing.wav").to_string_lossy().into_owned()),
            no_play: true,
        }),
    );
    assert!(matches!(response.result, ResultPayload::Synthesis { .. }));
}

#[test]
fn hidden_inventory_probe_dispatches_and_rejects_invalid_candidates() {
    let root = sandbox();
    let config = Config::default();
    run(Cli {
        config: Some(root.join("config.toml")),
        command: TopCommand::InventoryProbe {
            candidate: serde_json::to_string(&config.backend).unwrap(),
        },
    })
    .unwrap();
    assert!(
        run(Cli {
            config: Some(root.join("config.toml")),
            command: TopCommand::InventoryProbe {
                candidate: "not json".into(),
            },
        })
        .is_err()
    );
}

#[test]
fn noninteractive_runtime_setup_persists_selection_and_rejects_an_invalid_library() {
    let make_paths = paths;
    let root = sandbox();
    let paths = make_paths(&root);
    apply_runtime_selection_with_provider_probe(
        &paths.config_file,
        &paths,
        RuntimeSelection {
            runtime: Runtime::Default,
            device: "cpu".into(),
            device_id: None,
            directory: None,
        },
        true,
        |_, _| omaspeak::runtime_inventory::Probe {
            loadable: true,
            device_accessible: Some(true),
            ready: true,
            evidence: omaspeak::runtime_inventory::Evidence {
                versions: vec!["injected test runtime".into()],
                available_devices: vec!["cpu".into()],
                selected_device: Some("cpu".into()),
                ..Default::default()
            },
            errors: Vec::new(),
        },
        |_, _| Ok(PathBuf::from("/test/libaudiocpp.so")),
    )
    .unwrap();

    let config = Config::load(&paths.config_file).unwrap();
    assert_eq!(config.backend.runtime, Runtime::Default);
    assert_eq!(config.backend.device, "cpu");

    let gpu_root = sandbox();
    let gpu_paths = make_paths(&gpu_root);
    apply_runtime_selection_with_provider_probe(
        &gpu_paths.config_file,
        &gpu_paths,
        RuntimeSelection {
            runtime: Runtime::Cuda,
            device: "gpu".into(),
            device_id: Some(2),
            directory: None,
        },
        true,
        |_, _| unreachable!("audio.cpp uses its provider probe"),
        |candidate, _| {
            assert_eq!(candidate.backend.device_id, 2);
            Ok(PathBuf::from("/test/libaudiocpp-cuda.so"))
        },
    )
    .unwrap();
    assert_eq!(
        Config::load(&gpu_paths.config_file)
            .unwrap()
            .backend
            .device_id,
        2
    );
    assert!(
        runtime_configuration_candidate(
            &gpu_paths.config_file,
            Runtime::Default,
            "cpu",
            Some(1),
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("--device-id is only valid")
    );

    let runtime = root.join("runtime-sdk/lib");
    fs::create_dir_all(&runtime).unwrap();
    fs::write(runtime.join("libaudiocpp.so.1"), b"fixture").unwrap();
    let original = fs::read_to_string(&paths.config_file).unwrap();
    let error = setup(
        Some(SetupCommand::Runtime {
            json: false,
            runtime: None,
            device: None,
            device_id: None,
            dir: Some(root.join("runtime-sdk")),
            apply: true,
        }),
        &paths.config_file,
        &paths,
    )
    .unwrap_err();
    assert!(error.to_string().contains("runtime candidate rejected"));
    assert_eq!(fs::read_to_string(&paths.config_file).unwrap(), original);
    setup(
        Some(SetupCommand::Runtime {
            json: true,
            runtime: None,
            device: None,
            device_id: None,
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
                source: None,
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

#[test]
fn noninteractive_npu_runtime_apply_defers_cache_on_a_clean_install() {
    let root = sandbox();
    let paths = paths(&root);
    apply_runtime_selection(
        &paths.config_file,
        &paths,
        RuntimeSelection {
            runtime: Runtime::Openvino,
            device: "npu".into(),
            device_id: None,
            directory: None,
        },
        true,
        |_, _| omaspeak::runtime_inventory::Probe {
            loadable: true,
            device_accessible: Some(true),
            ready: true,
            evidence: omaspeak::runtime_inventory::Evidence {
                available_devices: vec!["NPU".into()],
                selected_device: Some("NPU".into()),
                ..Default::default()
            },
            errors: Vec::new(),
        },
    )
    .unwrap();
    let config = Config::load(&paths.config_file).unwrap();
    assert_eq!(config.backend.runtime, Runtime::Openvino);
    assert_eq!(config.backend.device, "npu");
    assert!(!omaspeak::supertonic::npu_cache_state(&config, &paths).ready);
}

#[test]
fn npu_cache_progress_orchestration_handles_formats_skip_and_missing_state() {
    let root = sandbox();
    let app_paths = paths(&root);
    let mut config = Config::default();
    let calls = std::cell::Cell::new(0);
    prepare_npu_cache_with_progress_with(&mut config, &app_paths, ProgressFormat::Human, |_, _| {
        calls.set(calls.get() + 1);
        unreachable!("CPU setup must skip NPU preparation")
    })
    .unwrap();
    assert_eq!(calls.get(), 0);

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    for format in [ProgressFormat::Human, ProgressFormat::Json] {
        let expected = app_paths.cache_dir.join("prepared");
        prepare_npu_cache_with_progress_with(&mut config, &app_paths, format, |_, _| {
            Ok(Some(omaspeak::supertonic::NpuCacheState {
                required: true,
                ready: true,
                fingerprint: Some("fixture".into()),
                directory: Some(expected.clone()),
                blobs: Vec::new(),
                detail: format!(
                    "{} prepared OpenVINO cache blobs",
                    omaspeak::supertonic::NPU_COMPILED_MODELS
                ),
            }))
        })
        .unwrap();
    }
    let error = prepare_npu_cache_with_progress_with(
        &mut config,
        &app_paths,
        ProgressFormat::Json,
        |_, _| Ok(None),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unexpectedly skipped"));
}

#[test]
fn npu_model_readiness_distinguishes_catalog_custom_and_device_states() {
    let root = sandbox();
    let app_paths = paths(&root);
    let mut config = Config::default();
    assert!(!npu_model_ready_with(&config, &app_paths, |_, _| Ok(())).unwrap());

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    let error = npu_model_ready_with(&config, &app_paths, |_, _| Ok(())).unwrap_err();
    assert!(error.to_string().contains("not validated for Intel NPU"));

    config.model.name = "supertonic-3-openvino".into();
    assert!(!npu_model_ready_with(&config, &app_paths, |_, _| bail!("missing")).unwrap());
    assert!(npu_model_ready_with(&config, &app_paths, |_, _| Ok(())).unwrap());

    config.model.name = "custom-npu".into();
    config.model.directory = root.join("custom-model").display().to_string();
    assert!(!npu_model_ready_with(&config, &app_paths, |_, _| unreachable!()).unwrap());
    fs::create_dir_all(config.model_directory(&app_paths)).unwrap();
    let error = npu_model_ready_with(&config, &app_paths, |_, _| unreachable!()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot be assumed Intel NPU compatible")
    );
}

#[test]
fn explicit_cache_setup_reports_optional_unready_and_ready_states() {
    let root = sandbox();
    let app_paths = paths(&root);
    Config::default().save(&app_paths.config_file).unwrap();
    setup(
        Some(SetupCommand::Cache {
            prepare: false,
            json: false,
        }),
        &app_paths.config_file,
        &app_paths,
    )
    .unwrap();
    setup(
        Some(SetupCommand::Cache {
            prepare: false,
            json: true,
        }),
        &app_paths.config_file,
        &app_paths,
    )
    .unwrap();
    let error = setup(
        Some(SetupCommand::Cache {
            prepare: true,
            json: false,
        }),
        &app_paths.config_file,
        &app_paths,
    )
    .unwrap_err();
    assert!(error.to_string().contains("runtime/device"));

    let mut unready = Config::default();
    omaspeak::catalog::model("supertonic-3-openvino")
        .unwrap()
        .activate(&mut unready);
    unready.backend.runtime = Runtime::Openvino;
    unready.backend.device = "npu".into();
    unready.model.name = "custom-npu".into();
    unready.model.directory = root.join("missing-model").display().to_string();
    unready.save(&app_paths.config_file).unwrap();
    let error = setup(
        Some(SetupCommand::Cache {
            prepare: false,
            json: false,
        }),
        &app_paths.config_file,
        &app_paths,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cache is not ready"));

    let mut ready = unready;
    ready.model.directory = root.join("ready-model").display().to_string();
    ready.backend.openvino_library = Some(app_paths.data_dir.join("openvino/libopenvino_c.so"));
    ready.backend.openvino_plugins = Some(app_paths.data_dir.join("openvino/plugins.xml"));
    write_npu_fingerprint_assets(&ready, &app_paths);
    ready.save(&app_paths.config_file).unwrap();
    let directory = omaspeak::supertonic::npu_cache_directory(&ready, &app_paths).unwrap();
    fs::create_dir_all(&directory).unwrap();
    for index in 0..omaspeak::supertonic::NPU_COMPILED_MODELS {
        fs::write(directory.join(format!("model-{index}.blob")), b"compiled").unwrap();
    }
    omaspeak::supertonic::write_npu_cache_manifest(&ready, &app_paths, &directory).unwrap();
    setup(
        Some(SetupCommand::Cache {
            prepare: false,
            json: false,
        }),
        &app_paths.config_file,
        &app_paths,
    )
    .unwrap();
}

#[test]
fn top_level_dispatch_uses_injected_paths_for_safe_offline_commands() {
    let root = sandbox();
    let app_paths = paths(&root);
    Config::default().save(&app_paths.config_file).unwrap();

    for json in [false, true] {
        run_with_paths(
            Cli {
                config: Some(app_paths.config_file.clone()),
                command: TopCommand::Status { json },
            },
            app_paths.clone(),
        )
        .unwrap();
        run_with_paths(
            Cli {
                config: Some(app_paths.config_file.clone()),
                command: TopCommand::Config {
                    command: ConfigCommand::Schema { json },
                },
            },
            app_paths.clone(),
        )
        .unwrap();
        run_with_paths(
            Cli {
                config: Some(app_paths.config_file.clone()),
                command: TopCommand::Config {
                    command: ConfigCommand::Get {
                        key: json.then(|| "model.name".into()),
                        json,
                    },
                },
            },
            app_paths.clone(),
        )
        .unwrap();
        run_with_paths(
            Cli {
                config: Some(app_paths.config_file.clone()),
                command: TopCommand::Setup {
                    command: Some(SetupCommand::Model {
                        list: !json,
                        json,
                        download: None,
                        set: None,
                        verify: None,
                        source: None,
                        accept_license: None,
                        no_activate: false,
                        progress_format: ProgressFormat::Human,
                    }),
                },
            },
            app_paths.clone(),
        )
        .unwrap();
    }

    run_with_paths(
        Cli {
            config: Some(app_paths.config_file.clone()),
            command: TopCommand::Config {
                command: ConfigCommand::Set {
                    key: "model.language".into(),
                    value: "ja".into(),
                },
            },
        },
        app_paths.clone(),
    )
    .unwrap();
    run_with_paths(
        Cli {
            config: Some(app_paths.config_file.clone()),
            command: TopCommand::Config {
                command: ConfigCommand::Unset {
                    key: "model.language".into(),
                },
            },
        },
        app_paths.clone(),
    )
    .unwrap();
    let stop = run_with_paths(
        Cli {
            config: Some(app_paths.config_file.clone()),
            command: TopCommand::Stop,
        },
        app_paths,
    )
    .unwrap_err();
    assert!(stop.to_string().contains("daemon is not running"));

    fn invoke(injected: &mut AppPaths, command: TopCommand) -> Result<()> {
        let config_file = injected.config_file.clone();
        run_with_paths_and_prepare(
            Cli {
                config: Some(config_file),
                command,
            },
            injected,
            |_, _| Ok(()),
        )
    }
    let mut injected = paths(&root);
    assert!(
        invoke(
            &mut injected,
            TopCommand::NpuPrecompile {
                request: "not json".into()
            }
        )
        .is_err()
    );
    assert!(
        invoke(
            &mut injected,
            TopCommand::Say(SayArgs {
                text: Some(" ".into()),
                voice: None,
                speed: None,
                out: None,
                no_play: true,
            })
        )
        .is_err()
    );
    assert!(
        invoke(
            &mut injected,
            TopCommand::Benchmark(BenchmarkArgs {
                text: String::new(),
                out_dir: root.join("benchmark"),
                warmup: 0,
                iterations: 1,
                voice: None,
            })
        )
        .is_err()
    );
    assert!(
        invoke(
            &mut injected,
            TopCommand::Setup {
                command: Some(SetupCommand::Check { json: true })
            }
        )
        .is_err()
    );

    let mut unsupported = Config::default();
    unsupported.backend.kind = "unsupported".into();
    unsupported.save(&injected.config_file).unwrap();
    assert!(invoke(&mut injected, TopCommand::Daemon).is_err());
}

#[test]
fn production_npu_setup_guards_are_safe_before_native_preparation() {
    let root = sandbox();
    let app_paths = paths(&root);
    let mut config = Config::default();

    assert!(!npu_model_ready(&config, &app_paths).unwrap());
    prepare_npu_for_setup(&mut config, &app_paths, ProgressFormat::Human).unwrap();
    prepare_npu_for_runtime_selection(&mut config, &app_paths, ProgressFormat::Human).unwrap();
    prepare_npu_cache_with_progress(&mut config, &app_paths, ProgressFormat::Json).unwrap();

    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    config.model.name = "custom-npu".into();
    config.model.directory = root.join("missing-model").display().to_string();
    assert!(!npu_model_ready(&config, &app_paths).unwrap());
    for format in [ProgressFormat::Human, ProgressFormat::Json] {
        let error = prepare_npu_for_setup(&mut config, &app_paths, format).unwrap_err();
        assert!(error.to_string().contains("NPU-capable model"));
        prepare_npu_for_runtime_selection(&mut config, &app_paths, format).unwrap();
    }

    config.save(&app_paths.config_file).unwrap();
    for json in [false, true] {
        let error = setup(
            Some(SetupCommand::Cache {
                prepare: true,
                json,
            }),
            &app_paths.config_file,
            &app_paths,
        )
        .unwrap_err();
        assert!(error.to_string().contains("NPU-capable model"));
    }
}

#[test]
fn daemon_orchestration_serves_shutdown_and_cleans_its_socket_without_signals() {
    let root = sandbox();
    let app_paths = paths(&root);
    fs::create_dir_all(&app_paths.runtime_dir).unwrap();
    let probe = app_paths.runtime_dir.join("probe.sock");
    match UnixListener::bind(&probe) {
        Ok(listener) => {
            drop(listener);
            fs::remove_file(probe).unwrap();
        }
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("probe daemon socket support: {error}"),
    }
    let thread_paths = app_paths.clone();
    Config::default().save(&thread_paths.config_file).unwrap();
    let server = thread::spawn(move || {
        run_daemon_with(
            &thread_paths.config_file,
            &thread_paths,
            |_, _| {
                Ok(FakeEngine {
                    fail: false,
                    runtime: Runtime::Default,
                })
            },
            |_| Ok(()),
        )
        .unwrap();
    });

    let socket = app_paths.socket();
    let mut response = None;
    for _ in 0..100 {
        if socket.exists()
            && let Ok(Some(value)) = try_send_request(&socket, &request(Command::Shutdown))
        {
            response = Some(value);
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(matches!(response.unwrap().result, ResultPayload::Shutdown));
    server.join().unwrap();
    assert!(!socket.exists());
}

#[test]
fn benchmark_orchestration_uses_loaded_engine_and_emits_a_complete_report() {
    let root = sandbox();
    let mut config = Config::default();
    config.model.voice = omaspeak::voices::VoiceSelection::Legacy(3);
    assert!(
        benchmark_with_engine(
            &config,
            &FakeEngine {
                fail: false,
                runtime: Runtime::Default,
            },
            BenchmarkArgs {
                text: "safe injected benchmark".into(),
                out_dir: root.join("benchmark"),
                warmup: 1,
                iterations: 2,
                voice: None,
            },
        )
        .is_ok()
    );
}

#[test]
fn setup_completion_and_snapshot_cleanup_cover_service_and_path_states() {
    let root = sandbox();
    let app_paths = paths(&root);
    let service = app_setup::systemd::service_path(&app_paths);
    fs::create_dir_all(service.parent().unwrap()).unwrap();
    fs::write(&service, "existing unit").unwrap();
    for restarted in [false, true] {
        print_setup_complete(
            &root.join("model"),
            &app_paths.config_file,
            &app_paths,
            &root.join("launcher.desktop"),
            true,
            restarted,
            ProgressFormat::Human,
        )
        .unwrap();
    }

    let nested = root.join("new/config.toml");
    restore_snapshot(&nested, Some(b"[model]\nvoice = 2\n"), "toml.tmp").unwrap();
    assert!(nested.is_file());
    restore_snapshot(&nested, None, "toml.tmp").unwrap();
    assert!(!nested.exists());
}

#[test]
fn runtime_picker_handles_explicit_directories_and_cancelled_input_without_probing() {
    struct InputSelector {
        choices: VecDeque<Option<usize>>,
        inputs: VecDeque<Option<String>>,
        input_calls: usize,
    }
    impl SetupSelector for InputSelector {
        fn select(&mut self, _: &str, _: &str, _: &[MenuItem], _: usize) -> Result<Option<usize>> {
            Ok(self.choices.pop_front().flatten())
        }
        fn input(&mut self, _: &str, _: &str) -> Result<Option<String>> {
            self.input_calls += 1;
            Ok(self.inputs.pop_front().flatten())
        }
    }

    let root = sandbox();
    let app_paths = paths(&root);
    let mut config = Config::default();
    config.backend.runtime = Runtime::Cuda;
    config.backend.library = Some(root.join("missing-audiocpp.so"));
    config.save(&app_paths.config_file).unwrap();
    let mut cancelled = InputSelector {
        choices: [Some(2), Some(0)].into(),
        inputs: [None].into(),
        input_calls: 0,
    };
    assert!(
        choose_runtime(&app_paths.config_file, &mut cancelled)
            .unwrap()
            .is_none()
    );
    assert_eq!(cancelled.input_calls, 1);

    let cpu_provider = root.join("libaudiocpp-cpu.so");
    fs::write(&cpu_provider, b"packaged CPU fixture").unwrap();
    config.backend.runtime = Runtime::Default;
    config.backend.device = "cpu".into();
    config.backend.library = Some(cpu_provider.canonicalize().unwrap());
    config.save(&app_paths.config_file).unwrap();
    let accelerator = root.join("cuda-provider");
    let mut selected = InputSelector {
        choices: [Some(2), Some(1)].into(),
        inputs: [Some("3".into()), Some(accelerator.display().to_string())].into(),
        input_calls: 0,
    };
    let choice = choose_runtime(&app_paths.config_file, &mut selected)
        .unwrap()
        .unwrap();
    assert_eq!(choice.runtime, Runtime::Cuda);
    assert_eq!(choice.device, "gpu");
    assert_eq!(choice.device_id, Some(3));
    assert_eq!(choice.directory.as_deref(), Some(accelerator.as_path()));
    assert_eq!(selected.input_calls, 2);

    let mut defaults = InputSelector {
        choices: [Some(2), Some(1)].into(),
        inputs: [Some(String::new()), Some(String::new())].into(),
        input_calls: 0,
    };
    let choice = choose_runtime(&app_paths.config_file, &mut defaults)
        .unwrap()
        .unwrap();
    assert_eq!(choice.runtime, Runtime::Cuda);
    assert_eq!(choice.device_id, Some(0));
    assert!(choice.directory.is_none());

    config.backend.runtime = Runtime::Openvino;
    config.backend.openvino_library = Some(root.join("missing-openvino.so"));
    config.backend.openvino_plugins = Some(root.join("missing-plugins.xml"));
    config.save(&app_paths.config_file).unwrap();
    let directory = root.join("sdk");
    let mut selected = InputSelector {
        choices: [Some(1), Some(3)].into(),
        inputs: [Some(directory.display().to_string())].into(),
        input_calls: 0,
    };
    let choice = choose_runtime(&app_paths.config_file, &mut selected)
        .unwrap()
        .unwrap();
    assert_eq!(choice.runtime, Runtime::Openvino);
    assert_eq!(choice.device, "npu");
    assert_eq!(choice.device_id, Some(0));
    assert_eq!(choice.directory.as_deref(), Some(directory.as_path()));
    assert_eq!(selected.input_calls, 1);
}

#[test]
fn runtime_picker_uses_packaged_cpu_after_openvino_without_reusing_its_directory() {
    struct InputSelector {
        choices: VecDeque<Option<usize>>,
        inputs: VecDeque<Option<String>>,
        input_calls: usize,
    }
    impl SetupSelector for InputSelector {
        fn select(&mut self, _: &str, _: &str, _: &[MenuItem], _: usize) -> Result<Option<usize>> {
            Ok(self.choices.pop_front().flatten())
        }
        fn input(&mut self, _: &str, _: &str) -> Result<Option<String>> {
            self.input_calls += 1;
            Ok(self.inputs.pop_front().flatten())
        }
    }

    let root = sandbox();
    let app_paths = paths(&root);
    let openvino = root.join("openvino");
    fs::create_dir_all(&openvino).unwrap();
    fs::write(openvino.join("libopenvino_c.so"), b"openvino").unwrap();
    fs::write(openvino.join("plugins.xml"), b"<ie/>").unwrap();
    let packaged = root.join("package/lib/libaudiocpp.so");
    fs::create_dir_all(packaged.parent().unwrap()).unwrap();
    fs::write(&packaged, b"packaged audio.cpp").unwrap();

    let mut config = Config::default();
    config.backend.kind = "supertonic".into();
    config.backend.runtime = Runtime::Openvino;
    config.backend.device = "npu".into();
    config.backend.library_dirs = vec![openvino.clone()];
    config.backend.openvino_library = Some(openvino.join("libopenvino_c.so"));
    config.backend.openvino_plugins = Some(openvino.join("plugins.xml"));
    config.save(&app_paths.config_file).unwrap();
    let locations = omaspeak::runtime::inspect(&config.backend, &app_paths.config_file);

    let mut cpu = InputSelector {
        choices: [Some(0), Some(1)].into(),
        inputs: [].into(),
        input_calls: 0,
    };
    let choice = choose_runtime_with_discovery(
        config.clone(),
        locations.clone(),
        Some(packaged.clone()),
        Some(packaged.clone()),
        &mut cpu,
    )
    .unwrap()
    .unwrap();
    assert_eq!(choice.runtime, Runtime::Default);
    assert_eq!(choice.device, "cpu");
    assert!(choice.directory.is_none());
    assert_eq!(cpu.input_calls, 0);

    let candidate = runtime_configuration_candidate(
        &app_paths.config_file,
        choice.runtime,
        &choice.device,
        choice.device_id,
        choice.directory.as_deref(),
    )
    .unwrap();
    assert!(candidate.backend.library_dirs.is_empty());
    assert!(candidate.backend.openvino_library.is_none());
    assert!(candidate.backend.openvino_plugins.is_none());

    let accelerator = root.join("audiocpp-cuda");
    let mut cuda = InputSelector {
        choices: [Some(2), Some(1)].into(),
        inputs: [Some("0".into()), Some(accelerator.display().to_string())].into(),
        input_calls: 0,
    };
    let choice = choose_runtime_with_discovery(
        config,
        locations,
        Some(packaged.clone()),
        Some(packaged),
        &mut cuda,
    )
    .unwrap()
    .unwrap();
    assert_eq!(choice.runtime, Runtime::Cuda);
    assert_eq!(choice.directory.as_deref(), Some(accelerator.as_path()));
    assert_eq!(cuda.input_calls, 2);
}

#[test]
fn runtime_picker_visibly_preselects_detected_npu_without_claiming_readiness() {
    struct CapturingSelector {
        screens: Vec<(String, Vec<MenuItem>, usize)>,
    }
    impl SetupSelector for CapturingSelector {
        fn select(
            &mut self,
            title: &str,
            _: &str,
            items: &[MenuItem],
            preferred: usize,
        ) -> Result<Option<usize>> {
            self.screens.push((title.into(), items.to_vec(), preferred));
            Ok(Some(preferred))
        }
        fn input(&mut self, _: &str, _: &str) -> Result<Option<String>> {
            unreachable!("detected OpenVINO should not request another directory")
        }
    }

    let root = sandbox();
    let app_paths = paths(&root);
    let config = Config::default();
    let mut locations = omaspeak::runtime::inspect(&config.backend, &app_paths.config_file);
    locations.runtime_loadable.insert("openvino", true);
    locations.openvino_library = Some(root.join("libopenvino_c.so"));
    locations.openvino_plugins = Some(root.join("plugins.xml"));
    let packaged = root.join("libaudiocpp.so.0");
    let hardware = omaspeak::hardware::HardwareReport {
        intel_npu: true,
        intel_gpu: true,
        vulkan_candidate: true,
        ..Default::default()
    };
    let mut selector = CapturingSelector {
        screens: Vec::new(),
    };
    let selected = choose_runtime_with_hardware(
        config,
        locations,
        Some(packaged.clone()),
        Some(packaged),
        hardware,
        &mut selector,
    )
    .unwrap()
    .unwrap();

    assert_eq!(selected.runtime, Runtime::Openvino);
    assert_eq!(selected.device, "npu");
    assert!(selected.directory.is_none());
    assert_eq!(selector.screens.len(), 2);
    let (_, runtime_items, runtime_preferred) = &selector.screens[0];
    assert_eq!(*runtime_preferred, 1);
    assert!(runtime_items[1].label.contains("Recommended"));
    assert!(runtime_items[1].detail.contains("Intel NPU"));
    assert!(runtime_items[1].detail.contains("Apply still runs"));
    assert!(!runtime_items[1].detail.contains("ready"));
    let (_, device_items, device_preferred) = &selector.screens[1];
    assert_eq!(*device_preferred, 3);
    assert!(device_items[3].label.contains("Recommended"));
}

#[test]
fn fresh_runtime_picker_recommends_a_discoverable_external_cuda_provider() {
    struct Selector {
        input_calls: usize,
    }
    impl SetupSelector for Selector {
        fn select(
            &mut self,
            _: &str,
            _: &str,
            _: &[MenuItem],
            preferred: usize,
        ) -> Result<Option<usize>> {
            Ok(Some(preferred))
        }
        fn input(&mut self, _: &str, _: &str) -> Result<Option<String>> {
            self.input_calls += 1;
            Ok(Some(String::new()))
        }
    }

    let root = sandbox();
    let app_paths = paths(&root);
    let config = Config::default();
    let external = root.join("cuda/libaudiocpp.so.0");
    fs::create_dir_all(external.parent().unwrap()).unwrap();
    fs::write(&external, b"fixture").unwrap();
    let mut locations = omaspeak::runtime::inspect(&config.backend, &app_paths.config_file);
    locations.audiocpp_library = Some(external.clone());
    locations.environment_library_dirs = vec![external.parent().unwrap().to_path_buf()];
    locations.effective_library_dirs = locations.environment_library_dirs.clone();
    locations.package_library_dirs.clear();
    let mut selector = Selector { input_calls: 0 };
    let selected = choose_runtime_with_hardware(
        config,
        locations,
        Some(external),
        None,
        omaspeak::hardware::HardwareReport {
            cuda_gpu: true,
            ..Default::default()
        },
        &mut selector,
    )
    .unwrap()
    .unwrap();

    assert_eq!(selected.runtime, Runtime::Cuda);
    assert_eq!(selected.device, "gpu");
    assert_eq!(selected.directory, None);
    assert_eq!(selector.input_calls, 1, "only the GPU index is requested");
}

#[test]
fn top_level_online_commands_exchange_protocol_without_loading_an_engine() {
    let root = sandbox();
    let app_paths = paths(&root);
    fs::create_dir_all(&app_paths.runtime_dir).unwrap();
    Config::default().save(&app_paths.config_file).unwrap();
    let socket = app_paths.socket();
    let listener = match UnixListener::bind(&socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("bind command-dispatch socket: {error}"),
    };
    let server = thread::spawn(move || {
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let incoming = read_request(&mut stream, 4096).unwrap();
            let result = match incoming.command {
                Command::Say { .. } => ResultPayload::Synthesis {
                    output: "/tmp/injected.wav".into(),
                    sample_rate: 16_000,
                    samples: 16_000,
                    audio_seconds: 1.0,
                    load_milliseconds: 1,
                    synthesis_milliseconds: 2,
                },
                Command::Status => ResultPayload::Status {
                    audio: serde_json::Value::Null,
                    running: true,
                    pid: 42,
                    language: String::new(),
                    model: "fixture".into(),
                    sample_rate: 16_000,
                    backend: json!({"kind": "injected"}),
                },
                Command::Shutdown => ResultPayload::Shutdown,
                Command::Cancel { request_id } => ResultPayload::Cancelled {
                    request_id,
                    count: 0,
                },
            };
            write_response(
                &mut stream,
                &Response {
                    protocol: 1,
                    id: incoming.id,
                    result,
                },
            )
            .unwrap();
        }
    });

    let invoke = |command| {
        let mut paths = app_paths.clone();
        run_with_paths_and_prepare(
            Cli {
                config: Some(paths.config_file.clone()),
                command,
            },
            &mut paths,
            |_, _| Ok(()),
        )
    };
    invoke(TopCommand::Say(SayArgs {
        text: Some("protocol only".into()),
        voice: Some("0".into()),
        speed: Some(1.0),
        out: Some(root.join("unused.wav")),
        no_play: true,
    }))
    .unwrap();
    invoke(TopCommand::Status { json: false }).unwrap();
    invoke(TopCommand::Stop).unwrap();
    server.join().unwrap();
}

#[test]
fn daemon_socket_identity_and_already_running_guards_are_race_safe() {
    let root = sandbox();
    let app_paths = paths(&root);
    fs::create_dir_all(&app_paths.runtime_dir).unwrap();
    let socket = app_paths.socket();
    let first = match UnixListener::bind(&socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("bind first identity socket: {error}"),
    };
    let old = fs::symlink_metadata(&socket).unwrap();
    assert!(
        prepare_daemon_socket(&app_paths)
            .unwrap_err()
            .to_string()
            .contains("already running")
    );
    fs::remove_file(&socket).unwrap();
    let second = UnixListener::bind(&socket).unwrap();
    let changed = remove_stale_socket(&socket, &old).unwrap_err();
    assert!(changed.to_string().contains("changed while checking"));
    drop(first);
    drop(second);
    fs::remove_file(&socket).unwrap();

    let interrupted_paths = paths(&sandbox());
    fs::create_dir_all(&interrupted_paths.runtime_dir).unwrap();
    let probe = interrupted_paths.runtime_dir.join("probe.sock");
    match UnixListener::bind(&probe) {
        Ok(listener) => drop(listener),
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("probe interrupted daemon socket: {error}"),
    }
    fs::remove_file(probe).unwrap();
    serve_daemon(
        &FakeEngine {
            fail: false,
            runtime: Runtime::Default,
        },
        &Config::default(),
        &interrupted_paths,
        Arc::new(AtomicBool::new(true)),
    )
    .unwrap();
    assert!(!interrupted_paths.socket().exists());
}

#[test]
fn unattended_setup_validation_boundary_preserves_transaction_semantics() {
    let root = sandbox();
    let app_paths = paths(&root);
    Config::default().save(&app_paths.config_file).unwrap();
    let validated = std::cell::Cell::new(false);
    let error = setup_all_with_validator(
        &app_paths.config_file,
        &app_paths,
        &FakeModelOperations { installed: true },
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Human,
        |paths| Ok(app_setup::menu::launcher_path(paths)),
        || false,
        |_| unreachable!("failed health check must prevent restart"),
        |config, path, explicit| {
            assert_eq!(config.backend.runtime, Runtime::Openvino);
            assert_eq!(config.model.name, omaspeak::catalog::OPENVINO_MODEL_ID);
            assert_eq!(path, app_paths.config_file);
            assert!(!explicit);
            validated.set(true);
            Ok(())
        },
    )
    .unwrap_err();
    assert!(validated.get());
    assert!(format!("{error:#}").contains("setup checks failed"));

    let before = fs::read(&app_paths.config_file).unwrap();
    let error = setup_all_with_validator(
        &app_paths.config_file,
        &app_paths,
        &FakeModelOperations { installed: true },
        "supertonic-3-openvino",
        None,
        None,
        Some("OpenRAIL-M"),
        ProgressFormat::Json,
        |_| unreachable!(),
        || unreachable!(),
        |_| unreachable!(),
        |_, _, _| bail!("injected runtime rejection"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("runtime rejection"));
    assert_eq!(fs::read(&app_paths.config_file).unwrap(), before);
}

#[test]
fn default_setup_selector_and_accept_wrapper_cover_production_boundaries() {
    let root = sandbox();
    let app_paths = paths(&root);
    let mut config = Config::default();
    config.backend.library = Some(root.join("missing-libaudiocpp.so"));

    let mut selector = ErrorSelector;
    assert_eq!(
        selector.input("ignored", "ignored").unwrap(),
        Some(String::new())
    );
    let error = selector
        .probe_runtime(&config, &app_paths.config_file)
        .unwrap_err();
    assert!(error.to_string().contains("runtime candidate rejected"));

    fs::create_dir_all(&app_paths.runtime_dir).unwrap();
    let socket = app_paths.runtime_dir.join("accept.sock");
    let listener = match UnixListener::bind(&socket) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(error) => panic!("bind accept-wrapper socket: {error}"),
    };
    let connector = thread::spawn(move || UnixStream::connect(socket).unwrap());
    let interrupted = AtomicBool::new(false);
    let accepted = accept_daemon_connection(&listener, &interrupted).unwrap();
    drop(accepted);
    drop(connector.join().unwrap());

    let _ = is_interactive_terminal();
}

#[test]
fn runtime_switch_resolves_model_format_and_retains_voice_on_same_backend() {
    let root = sandbox();
    let paths = paths(&root);
    let mut config = Config::default();
    config.model.voice = omaspeak::voices::VoiceSelection::Legacy(9);
    config.save(&paths.config_file).unwrap();
    let cuda =
        runtime_configuration_candidate(&paths.config_file, Runtime::Cuda, "gpu", Some(2), None)
            .unwrap();
    assert_eq!(
        cuda.model.voice,
        omaspeak::voices::VoiceSelection::Legacy(9)
    );
    assert_eq!(cuda.model.name, omaspeak::catalog::DEFAULT_MODEL_ID);
    let ov =
        runtime_configuration_candidate(&paths.config_file, Runtime::Openvino, "npu", None, None)
            .unwrap();
    assert_eq!(ov.model.name, omaspeak::catalog::OPENVINO_MODEL_ID);
    assert!(ov.model.file.is_empty());
    assert_eq!(ov.backend.device, "npu");
    ov.save(&paths.config_file).unwrap();
    assert_eq!(
        setup_model_id(&paths.config_file, None).unwrap(),
        omaspeak::catalog::OPENVINO_MODEL_ID
    );
    let cpu =
        runtime_configuration_candidate(&paths.config_file, Runtime::Default, "cpu", None, None)
            .unwrap();
    assert_eq!(cpu.model.name, omaspeak::catalog::DEFAULT_MODEL_ID);
    assert!(!cpu.model.file.is_empty());
    assert_eq!(
        setup_model_id(&paths.config_file, Some("explicit-model")).unwrap(),
        "explicit-model"
    );
    let cli = Cli::try_parse_from(["omaspeak", "setup", "all"]).unwrap();
    assert!(matches!(
        cli.command,
        TopCommand::Setup {
            command: Some(SetupCommand::All { model: None, .. })
        }
    ));
}

#[test]
fn model_picker_prefers_compatible_row_and_rejects_disabled_selection() {
    let root = sandbox();
    let paths = paths(&root);
    Config::default().save(&paths.config_file).unwrap();
    let operations = FakeModelOperations { installed: true };
    let mut selector = ScriptedSelector::new([Some(1)]);
    let chosen = choose_model(
        &paths.config_file,
        &paths,
        &operations,
        Some((Runtime::Openvino, "npu")),
        &mut selector,
    )
    .unwrap();
    assert_eq!(chosen.as_deref(), Some("supertonic-3-openvino"));
    assert_eq!(selector.calls[0].2, 1);
    for index in [0, usize::MAX] {
        let mut selector = ScriptedSelector::new([Some(index)]);
        assert!(
            choose_model(
                &paths.config_file,
                &paths,
                &operations,
                Some((Runtime::Openvino, "npu")),
                &mut selector
            )
            .is_err()
        );
    }
}
