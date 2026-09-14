use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use omaspeak::backend::{Fallback, Runtime, compiled_capabilities};
use omaspeak::config::Config;
use omaspeak::engine::{Engine, Synthesis};
use omaspeak::paths::AppPaths;
use omaspeak::protocol::{Command, Request, Response, ResultPayload};
use omaspeak::setup as app_setup;
use omaspeak::setup::model::ProgressFormat;
use omaspeak::setup::wizard::MenuItem;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "omaspeak",
    version,
    about = "Local-first text-to-speech daemon"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: TopCommand,
}

#[derive(Subcommand)]
enum TopCommand {
    Daemon,
    Status {
        #[arg(long)]
        json: bool,
    },
    Say(SayArgs),
    /// Benchmark one loaded TTS engine, write WAVs, and print JSON.
    Benchmark(BenchmarkArgs),
    Stop,
    Voices {
        #[arg(long)]
        json: bool,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Configure runtimes and models through a guided terminal or scriptable commands.
    Setup {
        #[command(subcommand)]
        command: Option<SetupCommand>,
    },
}

#[derive(Args)]
struct SayArgs {
    text: Option<String>,
    #[arg(long)]
    voice: Option<i32>,
    #[arg(long)]
    speed: Option<f32>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    no_play: bool,
}

#[derive(Args)]
struct BenchmarkArgs {
    #[arg(long)]
    text: String,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    warmup: u32,
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    iterations: u32,
}

#[derive(Subcommand)]
enum ConfigCommand {
    Schema {
        #[arg(long)]
        json: bool,
    },
    Get {
        key: Option<String>,
        #[arg(long)]
        json: bool,
    },
    Set {
        key: String,
        value: String,
    },
    Unset {
        key: String,
    },
}

#[derive(Subcommand)]
enum SetupCommand {
    /// Check the active model, runtime, audio, launcher, and optional service.
    Check {
        /// Print machine-readable check results.
        #[arg(long)]
        json: bool,
    },
    /// Install the model and launcher in one scriptable command.
    ///
    /// This leaves the systemd unit unchanged; use `omaspeak setup systemd`
    /// explicitly. An already-active daemon is safely restarted after setup.
    All {
        #[arg(long, default_value = "en_US-lessac-medium")]
        model: String,
        #[arg(long)]
        archive: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    /// Select a model interactively or manage catalog models with flags.
    Model {
        /// List catalog models and their local installation status.
        #[arg(
            long,
            conflicts_with_all = ["json", "download", "set", "verify", "archive", "no_activate"]
        )]
        list: bool,
        /// Print the complete catalog as JSON.
        #[arg(
            long,
            conflicts_with_all = ["list", "download", "set", "verify", "archive", "no_activate"]
        )]
        json: bool,
        /// Download, verify, install, and activate a catalog model.
        #[arg(
            long,
            value_name = "MODEL",
            conflicts_with_all = ["list", "json", "set", "verify"]
        )]
        download: Option<String>,
        /// Activate a catalog model that is already installed and verified.
        #[arg(
            long,
            value_name = "MODEL",
            conflicts_with_all = ["list", "json", "download", "verify"]
        )]
        set: Option<String>,
        /// Verify every pinned asset of an installed catalog model.
        #[arg(
            long,
            value_name = "MODEL",
            conflicts_with_all = ["list", "json", "download", "set"]
        )]
        verify: Option<String>,
        /// Install from a local pinned archive instead of downloading it.
        #[arg(long, requires = "download")]
        archive: Option<PathBuf>,
        /// Install the downloaded model without making it active.
        #[arg(long, requires = "download")]
        no_activate: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    /// Select a runtime and device, or print discovery data outside a terminal.
    Runtime {
        /// Print runtimes, devices, capabilities, and models as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Install, inspect, or remove the systemd user service.
    Systemd {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
        #[arg(long, conflicts_with_all = ["uninstall", "status"])]
        no_start: bool,
    },
    /// Install, inspect, or remove the desktop setup launcher.
    Menu {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
    },
}

trait SetupSelector {
    fn select(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>>;
}

struct TerminalSetupSelector;

impl SetupSelector for TerminalSetupSelector {
    fn select(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>> {
        app_setup::wizard::select(title, help, items, preferred)
    }
}

trait ModelSetupOperations {
    fn models(&self) -> &'static [omaspeak::catalog::ModelSpec];
    fn resolve(&self, id: &str) -> Result<&'static omaspeak::catalog::ModelSpec>;
    fn verify(&self, paths: &AppPaths, spec: &omaspeak::catalog::ModelSpec) -> Result<()>;
    fn install(
        &self,
        paths: &AppPaths,
        spec: &omaspeak::catalog::ModelSpec,
        archive: Option<&Path>,
        progress: ProgressFormat,
    ) -> Result<PathBuf>;
}

struct BuiltinModels;

impl ModelSetupOperations for BuiltinModels {
    fn models(&self) -> &'static [omaspeak::catalog::ModelSpec] {
        omaspeak::catalog::models()
    }

    fn resolve(&self, id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
        model_spec(id)
    }

    fn verify(&self, paths: &AppPaths, spec: &omaspeak::catalog::ModelSpec) -> Result<()> {
        app_setup::model::verify(paths, spec)
    }

    fn install(
        &self,
        paths: &AppPaths,
        spec: &omaspeak::catalog::ModelSpec,
        archive: Option<&Path>,
        progress: ProgressFormat,
    ) -> Result<PathBuf> {
        app_setup::model::install(paths, spec, archive, progress)
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omaspeak: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let mut paths = AppPaths::discover();
    let config_path = select_config_path(cli.config, &mut paths);
    match cli.command {
        TopCommand::Daemon => run_daemon(&config_path, &paths),
        TopCommand::Status { json } => print_status(&config_path, &paths, json),
        TopCommand::Say(args) => say(&config_path, &paths, args),
        TopCommand::Benchmark(args) => benchmark(&config_path, &paths, args),
        TopCommand::Stop => stop(&paths),
        TopCommand::Voices { json } => voices(&config_path, json),
        TopCommand::Config { command } => config_command(command, &config_path, &paths),
        TopCommand::Setup { command } => setup(command, &config_path, &paths),
    }
}

fn select_config_path(config: Option<PathBuf>, paths: &mut AppPaths) -> PathBuf {
    let config_path = config.unwrap_or_else(|| paths.config_file.clone());
    paths.config_file = config_path.clone();
    config_path
}

fn say(config_path: &Path, paths: &AppPaths, args: SayArgs) -> Result<()> {
    let config = Config::load(config_path)?;
    let text = match args.text {
        Some(text) => text,
        None => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .context("read stdin")?;
            text
        }
    };
    if text.trim().is_empty() {
        bail!("text must not be empty");
    }
    if text.len() > config.daemon.max_text_bytes {
        bail!("text exceeds {} bytes", config.daemon.max_text_bytes);
    }
    let output = args.out.unwrap_or_else(|| paths.state_dir.join("last.wav"));
    let request = Request {
        protocol: 1,
        id: request_id(),
        command: Command::Say {
            text,
            speed: args.speed.unwrap_or(1.0),
            voice: args.voice.unwrap_or(config.model.voice),
            output: Some(output.to_string_lossy().into_owned()),
            no_play: args.no_play,
        },
    };
    let response = send_or_handle_locally(&paths.socket(), request, |request| {
        let engine = Engine::load(&config, paths)?;
        Ok(handle_request(&engine, &config, paths, request))
    })?;
    print_response(response)
}

#[derive(Debug, Serialize)]
struct BenchmarkIteration {
    iteration: u32,
    output: PathBuf,
    elapsed_milliseconds: f64,
    synthesis_milliseconds: f64,
    sample_rate: i32,
    samples: usize,
    audio_duration_milliseconds: f64,
    real_time_factor: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    samples: usize,
    p50_elapsed_milliseconds: Option<f64>,
    p95_elapsed_milliseconds: Option<f64>,
    p50_synthesis_milliseconds: Option<f64>,
    p95_synthesis_milliseconds: Option<f64>,
    p50_real_time_factor: Option<f64>,
    p95_real_time_factor: Option<f64>,
}

fn benchmark(config_path: &Path, paths: &AppPaths, args: BenchmarkArgs) -> Result<()> {
    let config = Config::load(config_path)?;
    validate_benchmark_text(&args.text, config.daemon.max_text_bytes)?;
    let engine = Engine::load(&config, paths)?;
    let iterations = benchmark_syntheses(
        &args.text,
        &args.out_dir,
        args.warmup,
        args.iterations,
        |output| engine.synthesize(&args.text, 1.0, config.model.voice, output),
        Instant::now,
    )?;
    let report = benchmark_report(&config, &engine, &args, iterations)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_benchmark_text(text: &str, max_bytes: usize) -> Result<()> {
    if text.trim().is_empty() {
        bail!("text must not be empty");
    }
    if text.len() > max_bytes {
        bail!("text exceeds {max_bytes} bytes");
    }
    Ok(())
}

fn benchmark_syntheses<S, N>(
    text: &str,
    out_dir: &Path,
    warmup: u32,
    iterations: u32,
    mut synthesize: S,
    mut now: N,
) -> Result<Vec<BenchmarkIteration>>
where
    S: FnMut(&Path) -> Result<Synthesis>,
    N: FnMut() -> Instant,
{
    if text.trim().is_empty() {
        bail!("benchmark text must not be empty");
    }
    if iterations == 0 {
        bail!("benchmark iterations must be at least one");
    }
    for iteration in 1..=warmup {
        let output = out_dir.join(format!("warmup-{iteration:04}.wav"));
        synthesize(&output).with_context(|| format!("warm up synthesis {}", output.display()))?;
    }

    (1..=iterations)
        .map(|iteration| {
            let output = out_dir.join(format!("iteration-{iteration:04}.wav"));
            let started = now();
            let synthesis = synthesize(&output)
                .with_context(|| format!("benchmark synthesis {}", output.display()))?;
            let elapsed = now().saturating_duration_since(started);
            if synthesis.sample_rate <= 0 {
                bail!("synthesis sample rate must be positive");
            }
            let audio_duration =
                Duration::from_secs_f64(synthesis.samples as f64 / synthesis.sample_rate as f64);
            if audio_duration == Duration::ZERO {
                bail!("synthesis produced zero-duration audio");
            }
            Ok(BenchmarkIteration {
                iteration,
                output: synthesis.output,
                elapsed_milliseconds: milliseconds(elapsed),
                synthesis_milliseconds: milliseconds(synthesis.synthesis_time),
                sample_rate: synthesis.sample_rate,
                samples: synthesis.samples,
                audio_duration_milliseconds: milliseconds(audio_duration),
                real_time_factor: synthesis.synthesis_time.as_secs_f64()
                    / audio_duration.as_secs_f64(),
            })
        })
        .collect()
}

fn benchmark_report(
    config: &Config,
    engine: &impl SpeechEngine,
    args: &BenchmarkArgs,
    iterations: Vec<BenchmarkIteration>,
) -> Result<Value> {
    let summary = benchmark_summary(&iterations);
    Ok(json!({
        "schema_version": 1,
        "benchmark": "omaspeak-file-synthesis",
        "text": args.text,
        "out_dir": args.out_dir,
        "model_load_milliseconds": engine.load_milliseconds(),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "backend": {
            "kind": engine.backend_kind(),
            "model": engine.model_name(),
            "requested_runtime": config.backend.runtime,
            "requested_device": config.backend.canonical_device()?,
            "effective_runtime": engine.effective_runtime(),
            "fallback_used": engine.fallback_used(),
            "placement_verified": engine.effective_runtime() == Runtime::Default,
        },
        "iterations": iterations,
        "summary": summary,
    }))
}

fn benchmark_summary(iterations: &[BenchmarkIteration]) -> BenchmarkSummary {
    let elapsed = iterations
        .iter()
        .map(|iteration| iteration.elapsed_milliseconds)
        .collect::<Vec<_>>();
    let synthesis = iterations
        .iter()
        .map(|iteration| iteration.synthesis_milliseconds)
        .collect::<Vec<_>>();
    let real_time_factors = iterations
        .iter()
        .map(|iteration| iteration.real_time_factor)
        .collect::<Vec<_>>();
    BenchmarkSummary {
        samples: iterations.len(),
        p50_elapsed_milliseconds: percentile(&elapsed, 0.50),
        p95_elapsed_milliseconds: percentile(&elapsed, 0.95),
        p50_synthesis_milliseconds: percentile(&synthesis, 0.50),
        p95_synthesis_milliseconds: percentile(&synthesis, 0.95),
        p50_real_time_factor: percentile(&real_time_factors, 0.50),
        p95_real_time_factor: percentile(&real_time_factors, 0.95),
    }
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[rank])
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn run_daemon(config_path: &Path, paths: &AppPaths) -> Result<()> {
    let config = Config::load(config_path)?;
    let engine = Engine::load(&config, paths)?;
    let interrupted = Arc::new(AtomicBool::new(false));
    for signal in [
        signal_hook::consts::signal::SIGINT,
        signal_hook::consts::signal::SIGTERM,
    ] {
        signal_hook::flag::register(signal, interrupted.clone())
            .context("install daemon shutdown handler")?;
    }
    let socket = prepare_daemon_socket(paths)?;
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("bind daemon socket {}", socket.display()))?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let socket_metadata = fs::symlink_metadata(&socket)
        .with_context(|| format!("inspect bound daemon socket {}", socket.display()))?;
    listener
        .set_nonblocking(true)
        .context("make daemon socket interruptible")?;
    eprintln!(
        "omaspeak: ready model={} sample_rate={} load_ms={} socket={}",
        engine.model_name,
        engine.sample_rate,
        engine.load_time.as_millis(),
        socket.display()
    );

    let serve_result = serve_requests(&engine, &config, paths, || {
        accept_daemon_connection(&listener, &interrupted)
    });
    let serve_result = match serve_result {
        Err(error) if error.downcast_ref::<DaemonInterrupted>().is_some() => Ok(()),
        result => result,
    };
    drop(listener);
    finish_daemon(&socket, &socket_metadata, serve_result)
}

#[derive(Debug)]
struct DaemonInterrupted;

impl std::fmt::Display for DaemonInterrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("daemon interrupted")
    }
}

impl std::error::Error for DaemonInterrupted {}

fn accept_daemon_connection(
    listener: &UnixListener,
    interrupted: &AtomicBool,
) -> Result<UnixStream> {
    loop {
        if interrupted.load(Ordering::Relaxed) {
            return Err(DaemonInterrupted.into());
        }
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return Err(error).context("accept daemon request"),
        }
    }
}

fn finish_daemon(
    socket: &Path,
    socket_metadata: &fs::Metadata,
    serve_result: Result<()>,
) -> Result<()> {
    if let Err(error) = remove_stale_socket(socket, socket_metadata) {
        eprintln!(
            "omaspeak: failed to remove daemon socket {}: {error}",
            socket.display()
        );
    }
    serve_result
}

fn prepare_daemon_socket(paths: &AppPaths) -> Result<PathBuf> {
    fs::create_dir_all(&paths.runtime_dir)?;
    fs::set_permissions(&paths.runtime_dir, fs::Permissions::from_mode(0o700))?;
    fs::create_dir_all(&paths.state_dir)?;
    let socket = paths.socket();
    if connect_daemon(&socket)?.is_some() {
        bail!("daemon is already running at {}", socket.display());
    }
    Ok(socket)
}

fn serve_requests<S: Read + Write>(
    engine: &impl SpeechEngine,
    config: &Config,
    paths: &AppPaths,
    mut accept: impl FnMut() -> Result<S>,
) -> Result<()> {
    loop {
        let mut stream = accept()?;
        let response = match read_request(&mut stream, config.daemon.max_text_bytes + 16_384) {
            Ok(request) => {
                let should_stop = matches!(request.command, Command::Shutdown);
                let response = handle_request(engine, config, paths, request);
                write_response_best_effort(&mut stream, &response);
                if should_stop {
                    return Ok(());
                }
                continue;
            }
            Err(error) => Response::error("unknown", "invalid_request", error),
        };
        write_response_best_effort(&mut stream, &response);
    }
}

fn write_response_best_effort(stream: &mut impl Write, response: &Response) {
    if let Err(error) = write_response(stream, response) {
        eprintln!("omaspeak: client disconnected before response: {error}");
    }
}

fn write_response(stream: &mut impl Write, response: &Response) -> Result<()> {
    let mut bytes = serde_json::to_vec(response).context("encode daemon response")?;
    bytes.push(b'\n');
    stream.write_all(&bytes).context("write daemon response")
}

fn read_request(stream: &mut impl Read, limit: usize) -> Result<Request> {
    let mut bytes = Vec::new();
    BufReader::new(stream)
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut bytes)?;
    if bytes.len() > limit {
        bail!("message exceeds {limit} bytes");
    }
    let request: Request = serde_json::from_slice(&bytes)?;
    if request.protocol != 1 {
        bail!("protocol {} is unsupported", request.protocol);
    }
    Ok(request)
}

trait SpeechEngine {
    fn synthesize(&self, text: &str, speed: f32, voice: i32, output: &Path) -> Result<Synthesis>;
    fn backend_kind(&self) -> &'static str;
    fn model_name(&self) -> &str;
    fn sample_rate(&self) -> i32;
    fn load_milliseconds(&self) -> u64;
    fn effective_runtime(&self) -> Runtime;
    fn fallback_used(&self) -> bool;
}

impl SpeechEngine for Engine {
    fn synthesize(&self, text: &str, speed: f32, voice: i32, output: &Path) -> Result<Synthesis> {
        Engine::synthesize(self, text, speed, voice, output)
    }

    fn backend_kind(&self) -> &'static str {
        self.backend_kind
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    fn load_milliseconds(&self) -> u64 {
        self.load_time.as_millis() as u64
    }

    fn effective_runtime(&self) -> Runtime {
        self.effective_runtime
    }

    fn fallback_used(&self) -> bool {
        self.fallback_used
    }
}

fn handle_request(
    engine: &impl SpeechEngine,
    config: &Config,
    paths: &AppPaths,
    request: Request,
) -> Response {
    let id = request.id;
    let result = match request.command {
        Command::Say {
            text,
            speed,
            voice,
            output,
            no_play,
        } => {
            if text.len() > config.daemon.max_text_bytes {
                Err(anyhow!(
                    "text exceeds {} bytes",
                    config.daemon.max_text_bytes
                ))
            } else {
                let output = output
                    .map(PathBuf::from)
                    .unwrap_or_else(|| paths.state_dir.join("last.wav"));
                engine
                    .synthesize(&text, speed, voice, &output)
                    .and_then(|synthesis| {
                        if !no_play {
                            play(&synthesis.output)?;
                        }
                        Ok(synthesis_payload(engine, synthesis))
                    })
            }
        }
        Command::Status => Ok(status_payload(engine, config)),
        Command::Shutdown => Ok(ResultPayload::Shutdown),
    };
    match result {
        Ok(result) => Response {
            protocol: 1,
            id,
            result,
        },
        Err(error) => Response::error(id, "runtime", error),
    }
}

fn synthesis_payload(engine: &impl SpeechEngine, synthesis: Synthesis) -> ResultPayload {
    ResultPayload::Synthesis {
        output: synthesis.output.to_string_lossy().into_owned(),
        sample_rate: synthesis.sample_rate,
        samples: synthesis.samples,
        audio_seconds: synthesis.samples as f64 / synthesis.sample_rate as f64,
        load_milliseconds: engine.load_milliseconds(),
        synthesis_milliseconds: synthesis.synthesis_time.as_millis() as u64,
    }
}

fn status_payload(engine: &impl SpeechEngine, config: &Config) -> ResultPayload {
    let effective_runtime = engine.effective_runtime();
    ResultPayload::Status {
        running: true,
        pid: std::process::id(),
        model: engine.model_name().into(),
        sample_rate: engine.sample_rate(),
        backend: json!({
            "kind": engine.backend_kind(),
            "requested": {"runtime": config.backend.runtime, "device": config.backend.canonical_device().unwrap_or_else(|_| config.backend.device.clone())},
            "effective": {"runtime": effective_runtime, "device": if effective_runtime == Runtime::Default { "cpu" } else { config.backend.device.as_str() }, "provider": if effective_runtime == Runtime::Default { "CPUExecutionProvider" } else { "unverified" }},
            "compiled_capabilities": compiled_capabilities(),
            "fallback_policy": config.backend.fallback,
            "fallback_used": engine.fallback_used(),
            "placement_verified": effective_runtime == Runtime::Default,
            "evidence": if effective_runtime == Runtime::Default { vec!["static CPU build"] } else { Vec::<&str>::new() }
        }),
    }
}

fn send_or_handle_locally(
    socket: &Path,
    request: Request,
    local: impl FnOnce(Request) -> Result<Response>,
) -> Result<Response> {
    match try_send_request(socket, &request)? {
        Some(response) => Ok(response),
        None => local(request),
    }
}

fn try_send_request(socket: &Path, request: &Request) -> Result<Option<Response>> {
    let Some(mut stream) = connect_daemon(socket)? else {
        return Ok(None);
    };
    serde_json::to_writer(&mut stream, request)?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    serde_json::from_str(&line)
        .context("decode daemon response")
        .map(Some)
}

fn connect_daemon(socket: &Path) -> Result<Option<UnixStream>> {
    let metadata_before = match fs::symlink_metadata(socket) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect daemon socket {}", socket.display()));
        }
    };
    match UnixStream::connect(socket) {
        Ok(stream) => Ok(Some(stream)),
        Err(error) if indicates_stale_socket(error.kind()) => {
            remove_stale_socket(socket, &metadata_before)?;
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| format!("connect to daemon {}", socket.display())),
    }
}

fn indicates_stale_socket(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::NotFound
    )
}

fn remove_stale_socket(socket: &Path, metadata_before: &fs::Metadata) -> Result<()> {
    if !metadata_before.file_type().is_socket() {
        bail!(
            "refusing to remove non-socket daemon path {}",
            socket.display()
        );
    }
    let metadata_now = match fs::symlink_metadata(socket) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect stale daemon socket {}", socket.display()));
        }
    };
    if !metadata_now.file_type().is_socket()
        || metadata_now.dev() != metadata_before.dev()
        || metadata_now.ino() != metadata_before.ino()
    {
        bail!(
            "daemon socket {} changed while checking it; refusing to remove it",
            socket.display()
        );
    }
    fs::remove_file(socket)
        .with_context(|| format!("remove stale daemon socket {}", socket.display()))
}

fn print_response(response: Response) -> Result<()> {
    match response.result {
        ResultPayload::Error { code, message } => bail!("{code}: {message}"),
        result => {
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
    }
}

fn print_status(config_path: &Path, paths: &AppPaths, as_json: bool) -> Result<()> {
    let response = try_send_request(
        &paths.socket(),
        &Request {
            protocol: 1,
            id: request_id(),
            command: Command::Status,
        },
    )?;
    let status = if let Some(response) = response {
        serde_json::to_value(response.result)?
    } else {
        let config = Config::load(config_path)?;
        json!({"type":"status","running":false,"pid":null,"model":config.model.name,"sample_rate":null,
            "backend":{"kind":config.backend.kind,"requested":{"runtime":config.backend.runtime,"device":config.backend.device},"effective":null,"compiled_capabilities":compiled_capabilities(),"fallback_policy":config.backend.fallback,"fallback_used":false,"placement_verified":false,"evidence":[]}})
    };
    if as_json {
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else if status.get("running") == Some(&Value::Bool(true)) {
        println!("running");
    } else {
        println!("stopped");
    }
    Ok(())
}

fn stop(paths: &AppPaths) -> Result<()> {
    let response = try_send_request(
        &paths.socket(),
        &Request {
            protocol: 1,
            id: request_id(),
            command: Command::Shutdown,
        },
    )?
    .ok_or_else(|| anyhow!("daemon is not running"))?;
    print_response(response)
}

fn voices(config_path: &Path, as_json: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    if as_json {
        println!(
            "{}",
            json!([{"id":config.model.voice,"name":config.model.name}])
        );
    } else {
        println!("{}\t{}", config.model.voice, config.model.name);
    }
    Ok(())
}

fn config_command(command: ConfigCommand, path: &Path, paths: &AppPaths) -> Result<()> {
    match command {
        ConfigCommand::Schema { json } => {
            let value = schema(path, paths)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                println!(
                    "backend.runtime\tdefault|openvino|cuda\nbackend.device\truntime-dependent\nbackend.options.<name>\truntime provider option\nmodel.family\tpiper|vits|supertonic\nmodel.options.<name>\tmodel-specific option"
                );
            }
        }
        ConfigCommand::Get { key, json } => {
            let value = serde_json::to_value(Config::load(path)?)?;
            let selected = key
                .as_deref()
                .map_or(Some(&value), |key| dotted_get(&value, key))
                .ok_or_else(|| anyhow!("unknown config key"))?;
            if json || !selected.is_string() {
                println!("{}", serde_json::to_string_pretty(selected)?);
            } else {
                println!("{}", selected.as_str().unwrap_or_default());
            }
        }
        ConfigCommand::Set { key, value } => {
            let mut config = Config::load(path)?;
            let previous_runtime = config.backend.runtime;
            set_config(&mut config, &key, &value)?;
            save_config_mutation(path, config, &key, previous_runtime)?;
        }
        ConfigCommand::Unset { key } => {
            let mut config = Config::load(path)?;
            let previous_runtime = config.backend.runtime;
            unset_config(&mut config, &key)?;
            save_config_mutation(path, config, &key, previous_runtime)?;
        }
    }
    Ok(())
}

fn save_config_mutation(
    path: &Path,
    mut config: Config,
    key: &str,
    previous_runtime: Runtime,
) -> Result<()> {
    if key == "backend.runtime" && config.backend.runtime != previous_runtime {
        config.backend.device = "auto".into();
        config.backend.provider_config.clear();
        config.backend.options.clear();
        if config.backend.runtime != Runtime::Cuda {
            config.backend.device_id = 0;
        }
    }
    config.backend.validate_shape()?;
    save_config(path, &config)
}

fn set_config(config: &mut Config, key: &str, value: &str) -> Result<()> {
    if let Some(option) = dynamic_option_name(key, "backend.options.")? {
        config.backend.options.insert(option.into(), value.into());
        return Ok(());
    }
    if let Some(option) = dynamic_option_name(key, "model.options.")? {
        config.model.options.insert(option.into(), value.into());
        return Ok(());
    }
    match key {
        "backend.kind" => config.backend.kind = value.into(),
        "backend.runtime" => config.backend.runtime = parse_runtime(value)?,
        "backend.device" => config.backend.device = value.into(),
        "backend.threads" => config.backend.threads = value.parse()?,
        "backend.fallback" => config.backend.fallback = parse_fallback(value)?,
        "backend.device_id" => config.backend.device_id = value.parse()?,
        "backend.provider_config" => config.backend.provider_config = value.into(),
        "model.family" => config.model.family = value.into(),
        "model.name" => config.model.name = value.into(),
        "model.directory" => config.model.directory = value.into(),
        "model.model_file" => config.model.model_file = value.into(),
        "model.tokens_file" => config.model.tokens_file = value.into(),
        "model.data_directory" => config.model.data_directory = value.into(),
        "model.duration_predictor" => config.model.duration_predictor = value.into(),
        "model.text_encoder" => config.model.text_encoder = value.into(),
        "model.vector_estimator" => config.model.vector_estimator = value.into(),
        "model.vocoder" => config.model.vocoder = value.into(),
        "model.tts_json" => config.model.tts_json = value.into(),
        "model.unicode_indexer" => config.model.unicode_indexer = value.into(),
        "model.voice_style" => config.model.voice_style = value.into(),
        "model.language" => config.model.language = value.into(),
        "model.steps" => config.model.steps = value.parse()?,
        "model.voice" => config.model.voice = value.parse()?,
        "model.noise_scale" => config.model.noise_scale = value.parse()?,
        "model.noise_scale_w" => config.model.noise_scale_w = value.parse()?,
        "model.length_scale" => config.model.length_scale = value.parse()?,
        "audio.device" => config.audio.device = value.into(),
        "audio.volume" => config.audio.volume = value.parse()?,
        "daemon.queue_capacity" => config.daemon.queue_capacity = value.parse()?,
        "daemon.max_text_bytes" => config.daemon.max_text_bytes = value.parse()?,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
}

fn unset_config(config: &mut Config, key: &str) -> Result<()> {
    if let Some(option) = dynamic_option_name(key, "backend.options.")? {
        config.backend.options.remove(option);
        return Ok(());
    }
    if let Some(option) = dynamic_option_name(key, "model.options.")? {
        config.model.options.remove(option);
        return Ok(());
    }
    let defaults = Config::default();
    match key {
        "backend.kind" => config.backend.kind = defaults.backend.kind,
        "backend.runtime" => config.backend.runtime = defaults.backend.runtime,
        "backend.device" => config.backend.device = defaults.backend.device,
        "backend.threads" => config.backend.threads = defaults.backend.threads,
        "backend.fallback" => config.backend.fallback = defaults.backend.fallback,
        "backend.device_id" => config.backend.device_id = defaults.backend.device_id,
        "backend.provider_config" => {
            config.backend.provider_config = defaults.backend.provider_config
        }
        "model.family" => config.model.family = defaults.model.family,
        "model.name" => config.model.name = defaults.model.name,
        "model.directory" => config.model.directory = defaults.model.directory,
        "model.model_file" => config.model.model_file = defaults.model.model_file,
        "model.tokens_file" => config.model.tokens_file = defaults.model.tokens_file,
        "model.data_directory" => config.model.data_directory = defaults.model.data_directory,
        "model.duration_predictor" => {
            config.model.duration_predictor = defaults.model.duration_predictor
        }
        "model.text_encoder" => config.model.text_encoder = defaults.model.text_encoder,
        "model.vector_estimator" => config.model.vector_estimator = defaults.model.vector_estimator,
        "model.vocoder" => config.model.vocoder = defaults.model.vocoder,
        "model.tts_json" => config.model.tts_json = defaults.model.tts_json,
        "model.unicode_indexer" => config.model.unicode_indexer = defaults.model.unicode_indexer,
        "model.voice_style" => config.model.voice_style = defaults.model.voice_style,
        "model.language" => config.model.language = defaults.model.language,
        "model.steps" => config.model.steps = defaults.model.steps,
        "model.voice" => config.model.voice = defaults.model.voice,
        "model.noise_scale" => config.model.noise_scale = defaults.model.noise_scale,
        "model.noise_scale_w" => config.model.noise_scale_w = defaults.model.noise_scale_w,
        "model.length_scale" => config.model.length_scale = defaults.model.length_scale,
        "audio.device" => config.audio.device = defaults.audio.device,
        "audio.volume" => config.audio.volume = defaults.audio.volume,
        "daemon.queue_capacity" => config.daemon.queue_capacity = defaults.daemon.queue_capacity,
        "daemon.max_text_bytes" => config.daemon.max_text_bytes = defaults.daemon.max_text_bytes,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
}

fn dynamic_option_name<'a>(key: &'a str, prefix: &str) -> Result<Option<&'a str>> {
    let Some(option) = key.strip_prefix(prefix) else {
        return Ok(None);
    };
    if option.is_empty() || option.trim() != option {
        bail!(
            "config option name after {prefix} must be non-empty and have no surrounding whitespace"
        );
    }
    Ok(Some(option))
}

fn save_config(path: &Path, config: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, toml::to_string_pretty(config)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn dotted_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    key.split('.')
        .try_fold(value, |value, part| value.get(part))
}

fn parse_runtime(value: &str) -> Result<Runtime> {
    match value.to_ascii_lowercase().as_str() {
        "default" => Ok(Runtime::Default),
        "openvino" => Ok(Runtime::Openvino),
        "cuda" => Ok(Runtime::Cuda),
        _ => bail!("backend runtime must be default, openvino, or cuda"),
    }
}

fn parse_fallback(value: &str) -> Result<Fallback> {
    match value.to_ascii_lowercase().as_str() {
        "error" => Ok(Fallback::Error),
        "cpu" => Ok(Fallback::Cpu),
        _ => bail!("backend fallback must be error or cpu"),
    }
}

fn schema(path: &Path, paths: &AppPaths) -> Result<Value> {
    let config = Config::load(path)?;
    Ok(
        json!({"schema_version":1,"app":"omaspeak","app_version":env!("CARGO_PKG_VERSION"),"daemon_version":env!("CARGO_PKG_VERSION"),"config_path":path,
        "keys":[
            {"key":"backend.kind","type":"enum","section":"Backend","label":"Backend","description":"Inference engine","value":config.backend.kind,"file_value":null,"compiled":true,"restart_required":true,"choices":["sherpa-onnx"]},
            {"key":"backend.runtime","type":"enum","section":"Backend","label":"Runtime","description":"ONNX Runtime provider","value":config.backend.runtime,"file_value":null,"compiled":true,"restart_required":true,"choices":[{"value":"default","available":true,"capability":"cpu"},{"value":"openvino","available":compiled_capabilities().contains(&"openvino"),"capability":"openvino"},{"value":"cuda","available":compiled_capabilities().contains(&"cuda"),"capability":"cuda"}]},
            {"key":"backend.device","type":"string","section":"Backend","label":"Device","description":"Runtime-specific device","value":config.backend.device,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.threads","type":"integer","section":"Backend","label":"Threads","description":"Inference threads","value":config.backend.threads,"file_value":null,"compiled":true,"restart_required":true,"min":1,"max":64},
            {"key":"model.family","type":"enum","section":"Model","label":"Family","description":"sherpa TTS model family","value":config.model.family,"file_value":null,"compiled":true,"restart_required":true,"choices":["piper","vits","supertonic"]},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"compiled":true,"restart_required":true},
            {"key":"model.duration_predictor","type":"string","section":"Model","label":"Duration predictor","description":"Supertonic duration predictor filename","value":config.model.duration_predictor,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.text_encoder","type":"string","section":"Model","label":"Text encoder","description":"Supertonic text encoder filename","value":config.model.text_encoder,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.vector_estimator","type":"string","section":"Model","label":"Vector estimator","description":"Supertonic vector estimator filename","value":config.model.vector_estimator,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.vocoder","type":"string","section":"Model","label":"Vocoder","description":"Supertonic vocoder filename","value":config.model.vocoder,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.tts_json","type":"string","section":"Model","label":"TTS metadata","description":"Supertonic tts.json filename","value":config.model.tts_json,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.unicode_indexer","type":"string","section":"Model","label":"Unicode indexer","description":"Supertonic unicode indexer filename","value":config.model.unicode_indexer,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.voice_style","type":"string","section":"Model","label":"Voice styles","description":"Supertonic voice style filename","value":config.model.voice_style,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.language","type":"enum","section":"Model","label":"Language","description":"Supertonic generation language","value":config.model.language,"file_value":null,"compiled":true,"restart_required":true,"choices":["en","ko","ja","ar","bg","cs","da","de","el","es","et","fi","fr","hi","hr","hu","id","it","lt","lv","nl","pl","pt","ro","ru","sk","sl","sv","tr","uk","vi"]},
            {"key":"model.steps","type":"integer","section":"Model","label":"Generation steps","description":"Supertonic denoising steps","value":config.model.steps,"file_value":null,"compiled":true,"restart_required":true,"min":1}],
        "collections":[],"constraints":[{"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["auto","npu","gpu","cpu","auto:<devices>","hetero:<2+ devices>","multi:<2+ devices>"]}]}]}),
    )
}

fn setup(command: Option<SetupCommand>, config_path: &Path, paths: &AppPaths) -> Result<()> {
    let Some(command) = command else {
        if is_interactive_terminal() {
            return guided_setup(
                config_path,
                paths,
                &BuiltinModels,
                &mut TerminalSetupSelector,
            );
        }
        eprintln!(
            "Run `omaspeak setup` in a terminal for guided setup, or use `omaspeak setup all` for an unattended install."
        );
        return app_setup::print_checks(config_path, paths, false);
    };
    match command {
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Runtime { json } => {
            if !json && is_interactive_terminal() {
                guided_runtime(config_path, &mut TerminalSetupSelector).map(|_| ())
            } else {
                app_setup::print_runtime(json)
            }
        }
        SetupCommand::Model {
            list,
            json,
            download,
            set,
            verify,
            archive,
            no_activate,
            progress_format,
        } => {
            if !list
                && !json
                && download.is_none()
                && set.is_none()
                && verify.is_none()
                && archive.is_none()
                && is_interactive_terminal()
            {
                guided_model(
                    config_path,
                    paths,
                    &BuiltinModels,
                    &mut TerminalSetupSelector,
                )
                .map(|_| ())
            } else {
                setup_model(
                    config_path,
                    paths,
                    &BuiltinModels,
                    list,
                    json,
                    download,
                    set,
                    verify,
                    archive,
                    no_activate,
                    progress_format,
                )
            }
        }
        SetupCommand::Systemd {
            uninstall,
            status,
            no_start,
        } => {
            if status {
                app_setup::systemd::status(paths)
            } else if uninstall {
                app_setup::systemd::uninstall(paths)
            } else {
                app_setup::ensure_config(config_path)?;
                let path = app_setup::systemd::install(paths, config_path, !no_start)?;
                println!("installed: {}", path.display());
                Ok(())
            }
        }
        SetupCommand::Menu { uninstall, status } => {
            if status {
                app_setup::menu::status(paths)
            } else if uninstall {
                app_setup::menu::uninstall(paths)
            } else {
                let path = app_setup::menu::install(paths)?;
                println!("installed: {}", path.display());
                Ok(())
            }
        }
        SetupCommand::All {
            model,
            archive,
            progress_format,
        } => setup_all(
            config_path,
            paths,
            &BuiltinModels,
            &model,
            archive.as_deref(),
            progress_format,
            app_setup::menu::install,
            app_setup::systemd::is_active,
            app_setup::systemd::reload_if_was_active,
        ),
    }
}

fn is_interactive_terminal() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn guided_setup(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
) -> Result<()> {
    let actions = [
        MenuItem::available(
            "Full setup",
            "Choose a runtime, device, and model, then install the desktop launcher. This never installs or starts a service; an active daemon restarts after Apply.",
        ),
        MenuItem::available(
            "Runtime",
            "Choose and save an inference runtime and device.",
        ),
        MenuItem::available(
            "Model",
            "Browse, download, verify, and activate a catalog model.",
        ),
        MenuItem::available(
            "Check",
            "Check the current model, runtime, audio, launcher, and optional service.",
        ),
    ];
    let Some(selected) = selector.select(
        "Omaspeak setup",
        "Choose the part of Omaspeak you want to configure.",
        &actions,
        0,
    )?
    else {
        println!("Setup cancelled.");
        return Ok(());
    };
    match selected {
        0 => guided_full_setup(config_path, paths, operations, selector),
        1 => guided_runtime(config_path, selector).map(|_| ()),
        2 => guided_model(config_path, paths, operations, selector).map(|_| ()),
        3 => app_setup::print_checks(config_path, paths, false),
        _ => bail!("interactive setup returned an invalid choice"),
    }
}

fn guided_full_setup(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
) -> Result<()> {
    guided_full_setup_with(
        config_path,
        paths,
        operations,
        selector,
        app_setup::menu::install,
        app_setup::systemd::is_active,
        app_setup::systemd::reload_if_was_active,
    )
}

fn guided_full_setup_with(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
) -> Result<()> {
    let service_was_active = service_is_active();
    let Some((runtime, device)) = choose_runtime(config_path, selector)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let Some(model) = choose_model(
        config_path,
        paths,
        operations,
        Some((runtime, &device)),
        selector,
    )?
    else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let confirmation = [
        MenuItem::available(
            "Apply setup",
            "Download and activate the model, then install the desktop launcher. An active daemon restarts; an inactive service remains inactive.",
        ),
        MenuItem::available("Cancel", "Leave the current configuration unchanged."),
    ];
    let summary = format!(
        "Runtime: {} · Device: {} · Model: {} · Service unit: unchanged (`omaspeak setup systemd` installs it)",
        runtime_name(runtime),
        device,
        model
    );
    if selector.select("Apply Omaspeak setup", &summary, &confirmation, 0)? != Some(0) {
        println!("Setup cancelled; no changes were made.");
        return Ok(());
    }
    save_runtime(config_path, runtime, &device)?;
    setup_all(
        config_path,
        paths,
        operations,
        &model,
        None,
        ProgressFormat::Human,
        install_launcher,
        || service_was_active,
        reload_service,
    )
}

fn guided_runtime(
    config_path: &Path,
    selector: &mut impl SetupSelector,
) -> Result<Option<(Runtime, String)>> {
    let Some((runtime, device)) = choose_runtime(config_path, selector)? else {
        println!("Runtime setup cancelled.");
        return Ok(None);
    };
    if runtime == Runtime::Openvino && device.eq_ignore_ascii_case("npu") {
        let config = Config::load(config_path)?;
        if omaspeak::catalog::model(&config.model.name).is_some_and(|model| !model.npu_capable) {
            bail!(
                "model {} is not validated for Intel NPU; run `omaspeak setup` and choose Full setup to select a compatible model",
                config.model.name
            );
        }
    }
    save_runtime(config_path, runtime, &device)?;
    println!("Runtime configured: {} on {device}", runtime_name(runtime));
    Ok(Some((runtime, device)))
}

fn choose_runtime(
    config_path: &Path,
    selector: &mut impl SetupSelector,
) -> Result<Option<(Runtime, String)>> {
    let config = Config::load(config_path)?;
    let compiled = compiled_capabilities();
    let runtimes = [
        (
            Runtime::Default,
            MenuItem::available(
                "Default CPU",
                "Built in · portable ONNX Runtime CPU execution",
            ),
        ),
        (
            Runtime::Openvino,
            runtime_item(
                "OpenVINO",
                "Intel CPU, GPU, and NPU execution",
                compiled.contains(&"openvino"),
                "rebuild Omaspeak with --features openvino and the native OpenVINO stack",
            ),
        ),
        (
            Runtime::Cuda,
            runtime_item(
                "CUDA",
                "NVIDIA GPU execution",
                compiled.contains(&"cuda"),
                "rebuild Omaspeak with --features cuda and a CUDA-enabled native runtime",
            ),
        ),
    ];
    let items: Vec<_> = runtimes.iter().map(|(_, item)| item.clone()).collect();
    let preferred = runtimes
        .iter()
        .position(|(runtime, _)| *runtime == config.backend.runtime)
        .unwrap_or_default();
    let Some(selected) = selector.select(
        "Omaspeak runtime",
        "Choose an inference runtime. Unavailable runtimes show how to enable them.",
        &items,
        preferred,
    )?
    else {
        return Ok(None);
    };
    let runtime = runtimes[selected].0;
    let devices = device_items(runtime);
    let preferred = devices
        .iter()
        .position(|(device, _)| device.eq_ignore_ascii_case(&config.backend.device))
        .unwrap_or_default();
    let items: Vec<_> = devices.iter().map(|(_, item)| item.clone()).collect();
    let Some(selected) = selector.select(
        "Omaspeak device",
        &format!(
            "Choose the device for {}. Composite OpenVINO device strings remain available through `omaspeak config set backend.device ...`.",
            runtime_name(runtime)
        ),
        &items,
        preferred,
    )?
    else {
        return Ok(None);
    };
    Ok(Some((runtime, devices[selected].0.into())))
}

fn runtime_item(label: &str, detail: &str, available: bool, remediation: &str) -> MenuItem {
    if available {
        MenuItem::available(label, format!("Available · {detail}"))
    } else {
        MenuItem::unavailable(label, format!("Unavailable · {detail}; {remediation}"))
    }
}

fn device_items(runtime: Runtime) -> Vec<(&'static str, MenuItem)> {
    let specs: &[(&str, &str)] = match runtime {
        Runtime::Default => &[
            ("auto", "Recommended · let the CPU runtime choose"),
            ("cpu", "Use CPU execution explicitly"),
        ],
        Runtime::Cuda => &[
            (
                "auto",
                "Recommended · select the first available NVIDIA GPU",
            ),
            ("gpu", "Use NVIDIA GPU execution explicitly"),
        ],
        Runtime::Openvino => &[
            (
                "auto",
                "Recommended · let OpenVINO select an available device",
            ),
            ("cpu", "Use Intel CPU through OpenVINO"),
            (
                "gpu",
                "Use Intel integrated or discrete GPU through OpenVINO",
            ),
            (
                "npu",
                "Use Intel NPU; requires a compatible model and driver",
            ),
        ],
    };
    specs
        .iter()
        .map(|(device, detail)| (*device, MenuItem::available(device.to_uppercase(), *detail)))
        .collect()
}

fn save_runtime(config_path: &Path, runtime: Runtime, device: &str) -> Result<()> {
    let mut config = app_setup::ensure_config(config_path)?;
    let runtime_changed = config.backend.runtime != runtime;
    config.backend.runtime = runtime;
    config.backend.device = device.into();
    if runtime_changed {
        config.backend.provider_config.clear();
        config.backend.options.clear();
    }
    if runtime != Runtime::Cuda {
        config.backend.device_id = 0;
    }
    config
        .backend
        .validate_capabilities(compiled_capabilities())?;
    config.save(config_path)
}

fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Default => "default",
        Runtime::Openvino => "openvino",
        Runtime::Cuda => "cuda",
    }
}

fn guided_model(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
) -> Result<Option<String>> {
    let current = Config::load(config_path)?;
    let Some(id) = choose_model(
        config_path,
        paths,
        operations,
        Some((current.backend.runtime, current.backend.device.as_str())),
        selector,
    )?
    else {
        println!("Model setup cancelled.");
        return Ok(None);
    };
    let spec = operations.resolve(&id)?;
    let installed = operations.verify(paths, spec).is_ok();
    let directory = if installed {
        app_setup::model::model_directory(paths, spec)
    } else {
        operations.install(paths, spec, None, ProgressFormat::Human)?
    };
    let mut config = app_setup::ensure_config(config_path)?;
    spec.activate(&mut config);
    config.save(config_path)?;
    println!(
        "Active model: {} ({})",
        spec.id,
        if installed {
            "already installed"
        } else {
            "downloaded and verified"
        }
    );
    println!("Model directory: {}", directory.display());
    Ok(Some(id))
}

fn choose_model(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    runtime: Option<(Runtime, &str)>,
    selector: &mut impl SetupSelector,
) -> Result<Option<String>> {
    let config = Config::load(config_path)?;
    let models = operations.models();
    let items: Vec<_> = models
        .iter()
        .map(|model| {
            let installed = operations.verify(paths, model).is_ok();
            let exact_npu = runtime.is_some_and(|(runtime, device)| {
                runtime == Runtime::Openvino && device.eq_ignore_ascii_case("npu")
            });
            let compatible = !exact_npu || model.npu_capable;
            let active = model.id == config.model.name;
            let status = if active && installed {
                "● active"
            } else if active {
                "● active · download required"
            } else if installed {
                "○ installed"
            } else {
                "· download"
            };
            let bytes = model.archive_size
                + model
                    .supplemental_files
                    .iter()
                    .map(|file| file.size)
                    .sum::<u64>();
            let npu = if model.npu_capable {
                " · Intel NPU validated"
            } else {
                ""
            };
            let detail = format!(
                "{} · {} / {} · ~{} MiB download{}",
                model.description,
                model.backend,
                model.family,
                bytes.div_ceil(1024 * 1024),
                npu
            );
            if compatible {
                MenuItem::available(format!("{}  {status}", model.id), detail)
            } else {
                MenuItem::unavailable(
                    format!("{}  {status}", model.id),
                    format!("Incompatible with selected NPU · {detail}"),
                )
            }
        })
        .collect();
    let preferred = models
        .iter()
        .position(|model| model.id == config.model.name)
        .unwrap_or_default();
    let Some(selected) = selector.select(
        "Omaspeak model",
        "Choose a catalog model. Missing assets will download after you select it.",
        &items,
        preferred,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(models[selected].id.into()))
}

#[allow(clippy::too_many_arguments)]
fn setup_all(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    model: &str,
    archive: Option<&Path>,
    progress_format: ProgressFormat,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
) -> Result<()> {
    let service_was_active = service_is_active();
    let spec = operations.resolve(model)?;
    let mut config = app_setup::ensure_config(config_path)?;
    let directory = operations.install(paths, spec, archive, progress_format)?;
    spec.activate(&mut config);
    config.save(config_path)?;
    let launcher = install_launcher(paths)?;
    let service_restarted = reload_service(service_was_active)?;
    let service = app_setup::systemd::service_path(paths);
    let service_installed = service.is_file();
    match progress_format {
        ProgressFormat::Human => {
            let service_detail = if service_restarted {
                "active daemon restarted; unit unchanged".into()
            } else if service_installed {
                format!("unchanged (installed at {})", service.display())
            } else {
                "unchanged (optional; run `omaspeak setup systemd` to install)".into()
            };
            println!(
                "model: {}\nconfig: {}\nlauncher: {}\nservice: {}",
                directory.display(),
                config_path.display(),
                launcher.display(),
                service_detail,
            );
            app_setup::print_checks(config_path, paths, false)
        }
        ProgressFormat::Json => {
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "event": "setup-complete",
                    "model": directory,
                    "config": config_path,
                    "launcher": launcher,
                    "service": {
                        "installed": service_installed,
                        "active_before": service_was_active,
                        "restarted": service_restarted,
                        "unit_modified": false,
                        "optional": true,
                        "path": service,
                        "install_command": "omaspeak setup systemd"
                    },
                }))?
            );
            app_setup::print_checks_event(config_path, paths)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn setup_model(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    list: bool,
    json: bool,
    download: Option<String>,
    set: Option<String>,
    verify: Option<String>,
    archive: Option<PathBuf>,
    no_activate: bool,
    progress_format: ProgressFormat,
) -> Result<()> {
    if list || json {
        if json {
            println!("{}", serde_json::to_string_pretty(operations.models())?);
        } else {
            print_models_with(paths, operations);
        }
        if download.is_none() && set.is_none() && verify.is_none() {
            return Ok(());
        }
    }
    if let Some(id) = verify {
        let spec = operations.resolve(&id)?;
        operations.verify(paths, spec)?;
        println!(
            "verified: {}",
            app_setup::model::model_directory(paths, spec).display()
        );
        return Ok(());
    }
    if let Some(id) = set {
        let spec = operations.resolve(&id)?;
        operations.verify(paths, spec)?;
        let mut config = app_setup::ensure_config(config_path)?;
        spec.activate(&mut config);
        config.save(config_path)?;
        println!("active model: {}", spec.id);
        return Ok(());
    }
    let selected = match download {
        Some(id) => Some(id),
        None => {
            print_models_with(paths, operations);
            println!(
                "Run `omaspeak setup model --download en_US-lessac-medium` to install the default model."
            );
            None
        }
    };
    if let Some(id) = selected {
        let spec = operations.resolve(&id)?;
        let directory = operations.install(paths, spec, archive.as_deref(), progress_format)?;
        if !no_activate {
            let mut config = app_setup::ensure_config(config_path)?;
            spec.activate(&mut config);
            config.save(config_path)?;
        }
        match progress_format {
            ProgressFormat::Human => println!("model ready: {}", directory.display()),
            ProgressFormat::Json => println!(
                "{}",
                serde_json::to_string(&json!({
                    "event": "model-ready",
                    "model": spec.id,
                    "path": directory,
                    "active": !no_activate,
                }))?
            ),
        }
    }
    Ok(())
}

fn print_models_with(paths: &AppPaths, operations: &impl ModelSetupOperations) {
    for model in operations.models() {
        let status = if operations.verify(paths, model).is_ok() {
            "installed"
        } else {
            "available"
        };
        println!(
            "{}\t{}\t{}\t{}",
            model.id, model.backend, status, model.description
        );
    }
}

fn model_spec(id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
    omaspeak::catalog::model(id)
        .ok_or_else(|| anyhow!("unknown model {id}; run `omaspeak setup model --list`"))
}

fn play(path: &Path) -> Result<()> {
    play_with(path, |program, path| {
        ProcessCommand::new(program).arg(path).status()
    })
}

fn play_with(
    path: &Path,
    mut run: impl FnMut(&str, &Path) -> std::io::Result<std::process::ExitStatus>,
) -> Result<()> {
    for program in ["pw-play", "aplay"] {
        if run(program, path).is_ok_and(|status| status.success()) {
            return Ok(());
        }
    }
    bail!("no working WAV player found (tried pw-play and aplay)")
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

#[cfg(test)]
#[path = "../tests/unit/app_main.rs"]
mod tests;
