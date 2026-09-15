#![recursion_limit = "256"]

use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use omaspeak::backend::{Fallback, Runtime, canonical_device, supported_capabilities};
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
    #[command(name = "__audiocpp-probe", hide = true)]
    AudioCppProbe {
        #[arg(long)]
        spec: String,
    },
    #[command(name = "__audiocpp-worker", hide = true)]
    AudioCppWorker {
        #[arg(long)]
        spec: String,
    },
    #[command(name = "__inventory-probe", hide = true)]
    InventoryProbe {
        candidate: String,
    },
    #[command(name = "__npu-precompile", hide = true)]
    NpuPrecompile {
        request: String,
    },
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
    /// Speaker name (for example F3) or numeric ID.
    #[arg(long)]
    voice: Option<String>,
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
    /// Override the configured model voice for this benchmark.
    #[arg(long)]
    voice: Option<String>,
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
    /// Inspect or explicitly prepare the compiled OpenVINO Intel NPU cache.
    Cache {
        /// Compile every supported static shape and verify a second process loads it from cache.
        #[arg(long)]
        prepare: bool,
        /// Print machine-readable cache state and progress events.
        #[arg(long)]
        json: bool,
    },
    /// Install the model and launcher in one scriptable command.
    ///
    /// This leaves the systemd unit unchanged; use `omaspeak setup systemd`
    /// explicitly. An already-active daemon is safely restarted after setup.
    All {
        #[arg(long, default_value = "supertonic-3-gguf")]
        model: String,
        #[arg(long)]
        archive: Option<PathBuf>,
        /// Confirm acceptance of the model license required for catalog installation.
        #[arg(long, value_name = "LICENSE")]
        accept_license: Option<String>,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    /// Select a model interactively or manage catalog models with flags.
    Model {
        /// List catalog models and their local installation status.
        #[arg(
            long,
            conflicts_with_all = ["json", "download", "set", "verify", "archive", "no_activate", "accept_license"]
        )]
        list: bool,
        /// Print the complete catalog as JSON.
        #[arg(
            long,
            conflicts_with_all = ["list", "download", "set", "verify", "archive", "no_activate", "accept_license"]
        )]
        json: bool,
        /// Download, verify, install, and activate a catalog model.
        #[arg(
            long,
            visible_alias = "model",
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
        /// Install from a local pinned model file, directory, or archive instead of downloading it.
        #[arg(long, requires = "download")]
        archive: Option<PathBuf>,
        /// Confirm acceptance of the model license required for catalog installation.
        #[arg(long, value_name = "LICENSE", requires = "download")]
        accept_license: Option<String>,
        /// Install the downloaded model without making it active.
        #[arg(long, requires = "download")]
        no_activate: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    /// Select a runtime and device, or print discovery data outside a terminal.
    Runtime {
        /// Print runtimes, devices, capabilities, and models as JSON.
        #[arg(long, conflicts_with_all = ["dir", "runtime", "device"])]
        json: bool,
        /// Select a runtime without opening the terminal UI.
        #[arg(long, value_name = "RUNTIME", requires = "device")]
        runtime: Option<String>,
        /// Select a device without opening the terminal UI.
        #[arg(long, value_name = "DEVICE", requires = "runtime")]
        device: Option<String>,
        /// Configure the current runtime from a directory containing matching native libraries.
        #[arg(long, value_name = "DIRECTORY")]
        dir: Option<PathBuf>,
        /// Save only after the candidate passes its isolated native probe.
        #[arg(long, conflicts_with = "json")]
        apply: bool,
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
    fn probe_runtime(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<omaspeak::runtime_inventory::Probe> {
        omaspeak::runtime_inventory::apply_with(
            config,
            path,
            false,
            omaspeak::runtime_inventory::probe,
        )
    }

    fn select(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>>;

    fn input(&mut self, _title: &str, _help: &str) -> Result<Option<String>> {
        Ok(Some(String::new()))
    }
}

struct TerminalSetupSelector;

impl SetupSelector for TerminalSetupSelector {
    fn probe_runtime(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<omaspeak::runtime_inventory::Probe> {
        if config.backend.kind == "audiocpp" {
            let library = omaspeak::audio_cpp::discover_provider_library(config, path)?
                .context("audio.cpp provider was not found")?;
            return Ok(omaspeak::runtime_inventory::Probe {
                ready: true,
                loadable: true,
                device_accessible: true,
                evidence: omaspeak::runtime_inventory::Evidence {
                    versions: vec![format!("audio.cpp provider {}", library.display())],
                    ..Default::default()
                },
                errors: Vec::new(),
            });
        }
        omaspeak::runtime_inventory::apply_with(
            config,
            path,
            false,
            omaspeak::runtime_inventory::probe,
        )
    }

    fn select(
        &mut self,
        title: &str,
        help: &str,
        items: &[MenuItem],
        preferred: usize,
    ) -> Result<Option<usize>> {
        app_setup::wizard::select(title, help, items, preferred)
    }

    fn input(&mut self, title: &str, help: &str) -> Result<Option<String>> {
        println!("\n{title}\n{help}");
        print!("> ");
        std::io::stdout().flush()?;
        let mut value = String::new();
        if std::io::stdin().read_line(&mut value)? == 0 {
            return Ok(None);
        }
        Ok(Some(value.trim().to_owned()))
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
        accepted_license: Option<&str>,
    ) -> Result<PathBuf>;

    fn prove(&self, _config: &mut Config, _paths: &AppPaths) -> Result<()> {
        Ok(())
    }
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
        accepted_license: Option<&str>,
    ) -> Result<PathBuf> {
        app_setup::model::install(paths, spec, archive, progress, accepted_license)
    }

    fn prove(&self, config: &mut Config, paths: &AppPaths) -> Result<()> {
        pin_audio_cpp_library(config, &paths.config_file)?;
        prove_setup_synthesis(config, paths)
    }
}

fn prove_setup_synthesis(config: &Config, paths: &AppPaths) -> Result<()> {
    let output = paths
        .cache_dir
        .join(format!("setup-probe-{}.wav", std::process::id()));
    let result = (|| {
        let engine = Engine::load(config, paths).context("initialize selected setup provider")?;
        let synthesis = engine
            .synthesize("Omaspeak setup test.", 1.0, config.model.voice, &output)
            .context("run file-only setup synthesis")?;
        if synthesis.sample_rate <= 0 || synthesis.samples == 0 {
            bail!("file-only setup synthesis returned invalid audio");
        }
        Ok(())
    })();
    let _ = fs::remove_file(&output);
    result
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
    run_with_paths(cli, AppPaths::discover())
}

fn run_with_paths(cli: Cli, mut paths: AppPaths) -> Result<()> {
    run_with_paths_and_prepare(cli, &mut paths, prepare_native_library_path)
}

fn run_with_paths_and_prepare(
    cli: Cli,
    paths: &mut AppPaths,
    prepare: impl FnOnce(&TopCommand, &Path) -> Result<()>,
) -> Result<()> {
    let config_path = select_config_path(cli.config, paths);
    prepare(&cli.command, &config_path)?;
    match cli.command {
        TopCommand::AudioCppProbe { spec } => omaspeak::audio_cpp::run_provider_probe(&spec),
        TopCommand::AudioCppWorker { spec } => omaspeak::audio_cpp::run_worker(&spec),
        TopCommand::InventoryProbe { candidate } => {
            #[cfg(not(test))]
            omaspeak::audio_cpp::disable_core_dumps()?;
            println!(
                "{}",
                serde_json::to_string(&omaspeak::runtime_inventory::child(&serde_json::from_str(
                    &candidate
                )?))?
            );
            Ok(())
        }
        TopCommand::NpuPrecompile { request } => {
            #[cfg(not(test))]
            omaspeak::audio_cpp::disable_core_dumps()?;
            let result = omaspeak::runtime_inventory::npu_child(serde_json::from_str(&request)?)?;
            omaspeak::runtime_inventory::write_npu_preparation_result(&result)
        }
        TopCommand::Daemon => run_daemon(&config_path, paths),
        TopCommand::Status { json } => print_status(&config_path, paths, json),
        TopCommand::Say(args) => say(&config_path, paths, args),
        TopCommand::Benchmark(args) => benchmark(&config_path, paths, args),
        TopCommand::Stop => stop(paths),
        TopCommand::Voices { json } => voices(&config_path, paths, json),
        TopCommand::Config { command } => config_command(command, &config_path, paths),
        TopCommand::Setup { command } => setup(command, &config_path, paths),
    }
}

fn command_loads_engine(command: &TopCommand) -> bool {
    matches!(
        command,
        TopCommand::Daemon | TopCommand::Say(_) | TopCommand::Benchmark(_)
    )
}

#[cfg(target_os = "linux")]
fn prepare_native_library_path(command: &TopCommand, config_path: &Path) -> Result<()> {
    if !command_loads_engine(command) {
        return Ok(());
    }
    let config = Config::load(config_path)?;
    // audio.cpp and its transitive libraries belong to the supervised worker.
    // Keep optional native code out of the CLI/daemon process and scope its
    // loader path to that worker's exec environment.
    if config.backend.kind == "audiocpp" {
        return Ok(());
    }
    let report = omaspeak::runtime::discover(&config.backend, config_path);
    let Some(loader_path) = omaspeak::runtime::reexec_loader_path(&report)? else {
        return Ok(());
    };
    let executable = std::env::current_exe().context("locate Omaspeak executable for re-exec")?;
    let error = ProcessCommand::new(executable)
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", loader_path)
        .env(omaspeak::runtime::REEXEC_SENTINEL, "1")
        .exec();
    Err(error).context("re-exec Omaspeak with its configured native library path")
}

#[cfg(not(target_os = "linux"))]
fn prepare_native_library_path(_command: &TopCommand, _config_path: &Path) -> Result<()> {
    Ok(())
}

fn select_config_path(config: Option<PathBuf>, paths: &mut AppPaths) -> PathBuf {
    let config_path = config.unwrap_or_else(|| paths.config_file.clone());
    paths.config_file = config_path.clone();
    config_path
}

fn say(config_path: &Path, paths: &AppPaths, args: SayArgs) -> Result<()> {
    let config = Config::load(config_path)?;
    let request = build_say_request(&config, paths, args, std::io::stdin())?;
    let response = send_or_handle_locally(&paths.socket(), request, |request| {
        let engine = Engine::load(&config, paths)?;
        Ok(handle_request(&engine, &config, paths, request))
    })?;
    print_response(response)
}

fn resolve_voice(config: &Config, requested: Option<&str>) -> Result<i32> {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(config.model.voice);
    };
    let spec = omaspeak::catalog::model(&config.model.name);
    let voice = requested.parse::<i32>().ok().or_else(|| {
        spec.and_then(|model| {
            model
                .voices
                .iter()
                .find(|voice| voice.name.eq_ignore_ascii_case(requested))
                .map(|voice| voice.id)
        })
        .or_else(|| {
            (config.model.family == "supertonic")
                .then_some(omaspeak::voices::SUPERTONIC_PRESET_NAMES)
                .and_then(|names| {
                    names
                        .iter()
                        .position(|name| name.eq_ignore_ascii_case(requested))
                        .map(|id| id as i32)
                })
        })
    });
    let Some(voice) = voice else {
        let choices = spec.map_or_else(
            || {
                if config.model.family == "supertonic" {
                    omaspeak::voices::SUPERTONIC_PRESET_NAMES.join(", ")
                } else {
                    "a numeric ID".to_owned()
                }
            },
            |model| {
                model
                    .voices
                    .iter()
                    .map(|voice| format!("{} ({})", voice.name, voice.id))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        );
        bail!(
            "unknown voice {requested:?} for model {}; choose {choices}",
            config.model.name
        );
    };
    if let Some(model) = spec
        && !model.voices.iter().any(|candidate| candidate.id == voice)
    {
        bail!(
            "voice {voice} is unavailable for model {}; choose {}",
            model.id,
            model
                .voices
                .iter()
                .map(|candidate| format!("{} ({})", candidate.name, candidate.id))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(voice)
}

fn build_say_request(
    config: &Config,
    paths: &AppPaths,
    args: SayArgs,
    mut input: impl Read,
) -> Result<Request> {
    let text = match args.text {
        Some(text) => text,
        None => {
            let mut text = String::new();
            input.read_to_string(&mut text).context("read stdin")?;
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
    let voice = resolve_voice(config, args.voice.as_deref())?;
    Ok(Request {
        protocol: 1,
        id: request_id(),
        command: Command::Say {
            text,
            speed: args.speed.unwrap_or(1.0),
            voice,
            output: Some(output.to_string_lossy().into_owned()),
            no_play: args.no_play,
        },
    })
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
    benchmark_with_engine(&config, &engine, args)
}

fn benchmark_with_engine(
    config: &Config,
    engine: &impl SpeechEngine,
    args: BenchmarkArgs,
) -> Result<()> {
    let voice = resolve_voice(config, args.voice.as_deref())?;
    let iterations = benchmark_syntheses(
        &args.text,
        &args.out_dir,
        args.warmup,
        args.iterations,
        |output| engine.synthesize(&args.text, 1.0, voice, output),
        Instant::now,
    )?;
    let report = benchmark_report(config, engine, &args, voice, iterations)?;
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
    voice: i32,
    iterations: Vec<BenchmarkIteration>,
) -> Result<Value> {
    let summary = benchmark_summary(&iterations);
    Ok(json!({
        "schema_version": 1,
        "benchmark": "omaspeak-file-synthesis",
        "text": args.text,
        "voice": voice,
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
            "placement_verified": matches!(engine.backend_kind(), "audiocpp" | "openvino"),
            "placement_evidence": if engine.backend_kind() == "openvino" {
                "OpenVINO EXECUTION_DEVICES matched the requested device for every compiled graph"
            } else if engine.backend_kind() == "audiocpp" && engine.effective_runtime() == Runtime::Default {
                "audio.cpp CPU backend initialized in a supervised worker"
            } else {
                "audio.cpp accepted the requested backend while creating the model session"
            },
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
    run_daemon_with(config_path, paths, Engine::load, |interrupted| {
        for signal in [
            signal_hook::consts::signal::SIGINT,
            signal_hook::consts::signal::SIGTERM,
        ] {
            signal_hook::flag::register(signal, interrupted.clone())
                .context("install daemon shutdown handler")?;
        }
        Ok(())
    })
}

fn run_daemon_with<E: SpeechEngine>(
    config_path: &Path,
    paths: &AppPaths,
    load: impl FnOnce(&Config, &AppPaths) -> Result<E>,
    register_shutdown: impl FnOnce(&Arc<AtomicBool>) -> Result<()>,
) -> Result<()> {
    let config = Config::load(config_path)?;
    let engine = load(&config, paths)?;
    let interrupted = Arc::new(AtomicBool::new(false));
    register_shutdown(&interrupted)?;
    serve_daemon(&engine, &config, paths, interrupted)
}

fn serve_daemon(
    engine: &impl SpeechEngine,
    config: &Config,
    paths: &AppPaths,
    interrupted: Arc<AtomicBool>,
) -> Result<()> {
    let (socket, _startup_lock) = prepare_daemon_socket(paths)?;
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
        engine.model_name(),
        engine.sample_rate(),
        engine.load_milliseconds(),
        socket.display()
    );

    let serve_result = serve_requests(engine, config, paths, || {
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
    accept_daemon_connection_with(
        interrupted,
        || listener.accept().map(|(stream, _)| stream),
        || std::thread::sleep(Duration::from_millis(25)),
    )
}

fn accept_daemon_connection_with<T>(
    interrupted: &AtomicBool,
    mut accept: impl FnMut() -> std::io::Result<T>,
    mut wait: impl FnMut(),
) -> Result<T> {
    loop {
        if interrupted.load(Ordering::Relaxed) {
            return Err(DaemonInterrupted.into());
        }
        match accept() {
            Ok(stream) => return Ok(stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                wait();
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
    finish_daemon_with(serve_result, || {
        remove_stale_socket(socket, socket_metadata)
    })
}

fn finish_daemon_with(
    serve_result: Result<()>,
    remove_socket: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if let Err(error) = remove_socket() {
        eprintln!("omaspeak: failed to remove daemon socket after shutdown: {error}");
    }
    serve_result
}

fn prepare_daemon_socket(paths: &AppPaths) -> Result<(PathBuf, fs::File)> {
    fs::create_dir_all(&paths.runtime_dir)?;
    fs::set_permissions(&paths.runtime_dir, fs::Permissions::from_mode(0o700))?;
    fs::create_dir_all(&paths.state_dir)?;
    let lock_path = paths.runtime_dir.join("daemon.lock");
    let startup_lock = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("open daemon lock {}", lock_path.display()))?;
    fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600))?;
    // Keep this advisory lock for the daemon's lifetime. It serializes stale
    // socket cleanup with binding, so an inode that is unlinked and then reused
    // cannot make one daemon remove another daemon's socket.
    if unsafe { libc::flock(startup_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            bail!(
                "daemon is already running or starting at {}",
                paths.socket().display()
            );
        }
        return Err(error).with_context(|| format!("lock daemon startup {}", lock_path.display()));
    }
    let socket = paths.socket();
    if connect_daemon(&socket)?.is_some() {
        bail!("daemon is already running at {}", socket.display());
    }
    Ok((socket, startup_lock))
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
    let (effective_provider, placement_verified, evidence) = match engine.backend_kind() {
        "audiocpp" => (
            effective_runtime.capability(),
            true,
            vec!["audio.cpp accepted the requested backend while creating the model session"],
        ),
        "openvino" => (
            "openvino",
            true,
            vec!["OpenVINO execution devices were validated during graph compilation"],
        ),
        _ => (
            effective_runtime.capability(),
            false,
            vec!["backend did not provide placement evidence"],
        ),
    };
    ResultPayload::Status {
        running: true,
        pid: std::process::id(),
        model: engine.model_name().into(),
        sample_rate: engine.sample_rate(),
        backend: json!({
            "kind": engine.backend_kind(),
            "requested": {"runtime": config.backend.runtime, "device": config.backend.canonical_device().unwrap_or_else(|_| config.backend.device.clone())},
            "effective": {"runtime": effective_runtime, "device": if effective_runtime == Runtime::Default { "cpu" } else { config.backend.device.as_str() }, "provider": effective_provider},
            "supported_capabilities": supported_capabilities(),
            "fallback_policy": config.backend.fallback,
            "fallback_used": engine.fallback_used(),
            "placement_verified": placement_verified,
            "evidence": evidence
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
    exchange_request(&mut stream, request).map(Some)
}

fn exchange_request(stream: &mut (impl Read + Write), request: &Request) -> Result<Response> {
    serde_json::to_writer(&mut *stream, request)?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    serde_json::from_str(&line).context("decode daemon response")
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
            "backend":{"kind":config.backend.kind,"requested":{"runtime":config.backend.runtime,"device":config.backend.device},"effective":null,"supported_capabilities":supported_capabilities(),"fallback_policy":config.backend.fallback,"fallback_used":false,"placement_verified":false,"evidence":[]}})
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

fn voices(config_path: &Path, paths: &AppPaths, as_json: bool) -> Result<()> {
    let config = Config::load(config_path)?;
    let voices = omaspeak::voices::available(&config, paths)?;
    if as_json {
        let output = voices
            .iter()
            .map(|voice| {
                json!({
                    "id": voice.id,
                    "name": voice.name,
                    "active": voice.id == config.model.voice,
                })
            })
            .collect::<Vec<_>>();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        for voice in voices {
            println!(
                "{}\t{}\t{}",
                if voice.id == config.model.voice {
                    "*"
                } else {
                    " "
                },
                voice.id,
                voice.name
            );
        }
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
                    "backend.runtime\tdefault|cuda|vulkan|hip|openvino\nbackend.device\truntime-dependent\nbackend.library\texact complete audio.cpp provider library\nbackend.library_dirs\tprovider dependency directories\nbackend.openvino_library\texact OpenVINO C API library\nbackend.openvino_plugins\texact OpenVINO plugins.xml\nbackend.options.<scope>.<name>\taudio.cpp load/session/request option or direct OpenVINO property\nmodel.family\tsupertonic\nmodel.file\tsingle-file GGUF model\nmodel.options.<name>\tmodel-specific option"
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
        config.backend.options.clear();
        if !matches!(
            config.backend.runtime,
            Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
        ) {
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
        "backend.library_dirs" => {
            config.backend.library_dirs = std::env::split_paths(value).collect()
        }
        "backend.library" => config.backend.library = Some(PathBuf::from(value)),
        "backend.openvino_library" => config.backend.openvino_library = Some(PathBuf::from(value)),
        "backend.openvino_plugins" => config.backend.openvino_plugins = Some(PathBuf::from(value)),
        "model.family" => config.model.family = value.into(),
        "model.name" => config.model.name = value.into(),
        "model.directory" => config.model.directory = value.into(),
        "model.file" => config.model.file = value.into(),
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
        "backend.library_dirs" => config.backend.library_dirs = defaults.backend.library_dirs,
        "backend.library" => config.backend.library = defaults.backend.library,
        "backend.openvino_library" => {
            config.backend.openvino_library = defaults.backend.openvino_library
        }
        "backend.openvino_plugins" => {
            config.backend.openvino_plugins = defaults.backend.openvino_plugins
        }
        "model.family" => config.model.family = defaults.model.family,
        "model.name" => config.model.name = defaults.model.name,
        "model.directory" => config.model.directory = defaults.model.directory,
        "model.file" => config.model.file = defaults.model.file,
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
        "vulkan" => Ok(Runtime::Vulkan),
        "hip" | "rocm" => Ok(Runtime::Hip),
        _ => bail!("backend runtime must be default, cuda, vulkan, hip, or openvino"),
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
    let voice_choices = omaspeak::voices::available(&config, paths)
        .unwrap_or_else(|_| {
            omaspeak::catalog::model(&config.model.name)
                .map(omaspeak::voices::from_catalog)
                .unwrap_or_default()
        })
        .into_iter()
        .map(|voice| json!({"value":voice.id,"label":voice.name}))
        .collect::<Vec<_>>();
    Ok(
        json!({"schema_version":1,"app":"omaspeak","app_version":env!("CARGO_PKG_VERSION"),"daemon_version":env!("CARGO_PKG_VERSION"),"config_path":path,
        "keys":[
            {"key":"backend.kind","type":"enum","section":"Backend","label":"Backend","description":"Inference engine","value":config.backend.kind,"file_value":null,"compiled":true,"restart_required":true,"choices":["audiocpp","supertonic"]},
            {"key":"backend.runtime","type":"enum","section":"Backend","label":"Runtime","description":"Inference runtime","value":config.backend.runtime,"file_value":null,"compiled":true,"restart_required":true,"choices":[{"value":"default","available":true,"capability":"cpu"},{"value":"cuda","available":true,"capability":"cuda"},{"value":"vulkan","available":true,"capability":"vulkan"},{"value":"hip","available":true,"capability":"hip"},{"value":"openvino","available":true,"capability":"openvino"}]},
            {"key":"backend.device","type":"string","section":"Backend","label":"Device","description":"Runtime-specific device","value":config.backend.device,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.library_dirs","type":"path-list","section":"Backend","label":"Native library directories","description":"Application-owned provider/vendor runtime search path","value":config.backend.library_dirs,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.library","type":"path","section":"Backend","label":"Provider library","description":"Exact complete native provider library","value":config.backend.library,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.openvino_library","type":"path","section":"Backend","label":"OpenVINO library","description":"Exact OpenVINO C API library","value":config.backend.openvino_library,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.openvino_plugins","type":"path","section":"Backend","label":"OpenVINO plugins","description":"Exact OpenVINO plugins.xml","value":config.backend.openvino_plugins,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"backend.threads","type":"integer","section":"Backend","label":"Threads","description":"Inference threads","value":config.backend.threads,"file_value":null,"compiled":true,"restart_required":true,"min":1,"max":64},
            {"key":"model.family","type":"enum","section":"Model","label":"Family","description":"TTS model family","value":config.model.family,"file_value":null,"compiled":true,"restart_required":true,"choices":["supertonic"]},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"compiled":true,"restart_required":true},
            {"key":"model.file","type":"string","section":"Model","label":"Model file","description":"Single-file native model inside the model directory","value":config.model.file,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.duration_predictor","type":"string","section":"Model","label":"Duration predictor","description":"Supertonic duration predictor filename","value":config.model.duration_predictor,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.text_encoder","type":"string","section":"Model","label":"Text encoder","description":"Supertonic text encoder filename","value":config.model.text_encoder,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.vector_estimator","type":"string","section":"Model","label":"Vector estimator","description":"Supertonic vector estimator filename","value":config.model.vector_estimator,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.vocoder","type":"string","section":"Model","label":"Vocoder","description":"Supertonic vocoder filename","value":config.model.vocoder,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.tts_json","type":"string","section":"Model","label":"TTS metadata","description":"Supertonic tts.json filename","value":config.model.tts_json,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.unicode_indexer","type":"string","section":"Model","label":"Unicode indexer","description":"Supertonic unicode indexer filename","value":config.model.unicode_indexer,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.voice_style","type":"string","section":"Model","label":"Voice styles","description":"Supertonic voice style filename","value":config.model.voice_style,"file_value":null,"compiled":true,"restart_required":true},
            {"key":"model.language","type":"enum","section":"Model","label":"Language","description":"Supertonic generation language","value":config.model.language,"file_value":null,"compiled":true,"restart_required":true,"choices":["en","ko","ja","ar","bg","cs","da","de","el","es","et","fi","fr","hi","hr","hu","id","it","lt","lv","nl","pl","pt","ro","ru","sk","sl","sv","tr","uk","vi"]},
            {"key":"model.steps","type":"integer","section":"Model","label":"Generation steps","description":"Supertonic denoising steps","value":config.model.steps,"file_value":null,"compiled":true,"restart_required":true,"min":1},
            {"key":"model.voice","type":"enum","section":"Model","label":"Voice","description":"Default TTS speaker","value":config.model.voice,"file_value":null,"compiled":true,"restart_required":true,"choices":voice_choices}],
        "collections":[],"constraints":[{"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"vulkan","backend.device":["auto","gpu"]},{"backend.runtime":"hip","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["auto","npu","gpu","cpu"]}]}]}),
    )
}

fn setup(command: Option<SetupCommand>, config_path: &Path, paths: &AppPaths) -> Result<()> {
    if let Some(error) = app_setup::config_recovery(config_path)? {
        eprintln!(
            "omaspeak setup: the existing configuration is invalid; a successful setup apply will replace it with the current schema\n  {error}"
        );
    }
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
            "Run `omaspeak setup` in a terminal for guided setup, or use `omaspeak setup all --accept-license OpenRAIL-M` for an unattended install."
        );
        return app_setup::print_checks(config_path, paths, false);
    };
    match command {
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Cache { prepare, json } => {
            let mut config = app_setup::load_config(config_path)?;
            if prepare {
                if !omaspeak::supertonic::uses_static_npu_shapes(&config) {
                    bail!(
                        "NPU cache preparation requires the active runtime/device to be openvino/npu"
                    );
                }
                prepare_npu_for_setup(
                    &mut config,
                    paths,
                    if json {
                        ProgressFormat::Json
                    } else {
                        ProgressFormat::Human
                    },
                )?;
            }
            let state = omaspeak::supertonic::npu_cache_state(&config, paths);
            if json {
                println!("{}", serde_json::to_string_pretty(&state)?);
            } else {
                println!("{}", state.detail);
                if let Some(directory) = &state.directory {
                    println!("cache: {}", directory.display());
                }
            }
            if state.required && !state.ready {
                bail!("Intel NPU cache is not ready; run `omaspeak setup cache --prepare`");
            }
            Ok(())
        }
        SetupCommand::Runtime {
            json,
            runtime,
            device,
            dir,
            apply,
        } => {
            if let (Some(runtime), Some(device)) = (runtime, device) {
                let runtime = parse_runtime(&runtime)?;
                canonical_device(runtime, &device)?;
                apply_runtime_selection(
                    config_path,
                    paths,
                    runtime,
                    &device,
                    dir.as_deref(),
                    apply,
                    omaspeak::runtime_inventory::probe,
                )
            } else if let Some(dir) = dir {
                let current = app_setup::load_config(config_path)?;
                apply_runtime_selection(
                    config_path,
                    paths,
                    current.backend.runtime,
                    &current.backend.device,
                    Some(&dir),
                    apply,
                    omaspeak::runtime_inventory::probe,
                )
            } else if !json && is_interactive_terminal() {
                guided_runtime(config_path, paths, &mut TerminalSetupSelector).map(|_| ())
            } else {
                app_setup::print_runtime(config_path, json)
            }
        }
        SetupCommand::Model {
            list,
            json,
            download,
            set,
            verify,
            archive,
            accept_license,
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
                    accept_license,
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
                let original = config_snapshot(config_path)?;
                let result = (|| {
                    let mut config = app_setup::ensure_config(config_path)?;
                    prepare_npu_for_setup(&mut config, paths, ProgressFormat::Human)?;
                    config.save(config_path)?;
                    let path = app_setup::systemd::install(paths, config_path, !no_start)?;
                    println!("installed: {}", path.display());
                    Ok(())
                })();
                if let Err(error) = result {
                    restore_config_snapshot(config_path, original.as_deref())?;
                    return Err(error);
                }
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
            accept_license,
            progress_format,
        } => setup_all(
            config_path,
            paths,
            &BuiltinModels,
            &model,
            None,
            archive.as_deref(),
            accept_license.as_deref(),
            progress_format,
            app_setup::menu::install,
            app_setup::systemd::is_active,
            app_setup::systemd::reload_if_was_active,
        ),
    }
}

fn prepare_npu_cache_with_progress(
    config: &mut Config,
    paths: &AppPaths,
    progress: ProgressFormat,
) -> Result<()> {
    prepare_npu_cache_with_progress_with(config, paths, progress, |config, paths| {
        omaspeak::runtime_inventory::prepare_npu_cache(config, paths)
    })
}

fn prepare_npu_cache_with_progress_with(
    config: &mut Config,
    paths: &AppPaths,
    progress: ProgressFormat,
    prepare: impl FnOnce(&mut Config, &AppPaths) -> Result<Option<omaspeak::supertonic::NpuCacheState>>,
) -> Result<()> {
    if !omaspeak::supertonic::uses_static_npu_shapes(config) {
        return Ok(());
    }
    match progress {
        ProgressFormat::Human => {
            println!(
                "Compiling the fixed Supertonic shape set for Intel NPU. This can take several minutes; setup will wait."
            );
        }
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "npu-cache-compile-start",
                "model": config.model.name,
                "text_bucket": omaspeak::supertonic::NPU_TEXT_BUCKET,
                "latent_buckets": omaspeak::supertonic::NPU_LATENT_BUCKETS,
            }))?
        ),
    }
    let state =
        prepare(config, paths)?.context("Intel NPU cache preparation was unexpectedly skipped")?;
    match progress {
        ProgressFormat::Human => println!(
            "Intel NPU cache ready: {} ({})",
            state
                .directory
                .as_deref()
                .map_or_else(|| "(unknown)".into(), |path| path.display().to_string()),
            state.detail
        ),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "npu-cache-compile-complete",
                "model": config.model.name,
                "cache": state,
            }))?
        ),
    }
    Ok(())
}

fn npu_model_ready(config: &Config, paths: &AppPaths) -> Result<bool> {
    npu_model_ready_with(config, paths, app_setup::model::verify_at)
}

fn npu_model_ready_with(
    config: &Config,
    paths: &AppPaths,
    verify: impl FnOnce(&Path, &omaspeak::catalog::ModelSpec) -> Result<()>,
) -> Result<bool> {
    if !omaspeak::supertonic::uses_static_npu_shapes(config) {
        return Ok(false);
    }
    if let Some(spec) = omaspeak::catalog::model(&config.model.name) {
        if verify(&config.model_directory(paths), spec).is_err() {
            return Ok(false);
        }
        if !spec.npu_capable {
            bail!(
                "installed model {} is not validated for Intel NPU; select an NPU-capable model during model or full setup",
                spec.id
            );
        }
        return Ok(true);
    }
    if !config.model_directory(paths).is_dir() {
        return Ok(false);
    }
    bail!(
        "custom model {} cannot be assumed Intel NPU compatible; select an NPU-capable catalog model",
        config.model.name
    )
}

fn prepare_npu_for_setup(
    config: &mut Config,
    paths: &AppPaths,
    progress: ProgressFormat,
) -> Result<()> {
    if !omaspeak::supertonic::uses_static_npu_shapes(config) {
        return Ok(());
    }
    if !npu_model_ready(config, paths)? {
        bail!(
            "an installed, verified NPU-capable model is required before Intel NPU cache preparation"
        );
    }
    prepare_npu_cache_with_progress(config, paths, progress)
}

fn prepare_npu_for_runtime_selection(
    config: &mut Config,
    paths: &AppPaths,
    progress: ProgressFormat,
) -> Result<()> {
    if !omaspeak::supertonic::uses_static_npu_shapes(config) {
        return Ok(());
    }
    if npu_model_ready(config, paths)? {
        return prepare_npu_cache_with_progress(config, paths, progress);
    }
    match progress {
        ProgressFormat::Human => println!(
            "Intel NPU runtime configured; cache preparation is deferred until an NPU-capable model is installed."
        ),
        ProgressFormat::Json => println!(
            "{}",
            serde_json::to_string(&json!({
                "event": "npu-cache-deferred",
                "reason": "an NPU-capable model is not installed",
                "next": "omaspeak setup model",
            }))?
        ),
    }
    Ok(())
}

fn apply_runtime_selection(
    config_path: &Path,
    paths: &AppPaths,
    runtime: Runtime,
    device: &str,
    directory: Option<&Path>,
    apply: bool,
    probe: impl FnOnce(&omaspeak::backend::BackendConfig, &Path) -> omaspeak::runtime_inventory::Probe,
) -> Result<()> {
    apply_runtime_selection_with_provider_probe(
        config_path,
        paths,
        runtime,
        device,
        directory,
        apply,
        probe,
        omaspeak::audio_cpp::probe_provider,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_runtime_selection_with_provider_probe(
    config_path: &Path,
    paths: &AppPaths,
    runtime: Runtime,
    device: &str,
    directory: Option<&Path>,
    apply: bool,
    probe: impl FnOnce(&omaspeak::backend::BackendConfig, &Path) -> omaspeak::runtime_inventory::Probe,
    probe_audio_cpp: impl FnOnce(&Config, &Path) -> Result<PathBuf>,
) -> Result<()> {
    let mut candidate = runtime_configuration_candidate(config_path, runtime, device, directory)?;
    let mut evidence = if candidate.backend.kind == "audiocpp" {
        let library = probe_audio_cpp(&candidate, config_path)
            .context("runtime candidate rejected; config unchanged")?;
        candidate.backend.library = Some(library.clone());
        omaspeak::runtime_inventory::Probe {
            ready: true,
            loadable: true,
            device_accessible: false,
            evidence: omaspeak::runtime_inventory::Evidence {
                versions: vec![format!(
                    "audio.cpp provider ABI ready: {}",
                    library.display()
                )],
                provider_registration: true,
                available_devices: vec![candidate.backend.runtime.capability().into()],
                selected_device: Some(candidate.backend.device.clone()),
                provider_path: Some(library),
                ..Default::default()
            },
            errors: Vec::new(),
        }
    } else {
        omaspeak::runtime_inventory::apply_with(&candidate, config_path, false, probe)?
    };
    let model_installed = active_model_is_installed(&candidate, paths);
    if apply {
        prepare_npu_for_runtime_selection(&mut candidate, paths, ProgressFormat::Json)?;
        if model_installed {
            prove_setup_synthesis(&candidate, paths).context(
                "runtime candidate rejected by model-backed synthesis; config unchanged",
            )?;
            evidence.device_accessible = true;
            evidence.ready = true;
            evidence.evidence.model_inference_verified = true;
        }
        candidate.save(config_path)?;
    }
    let next = (!model_installed).then_some(
        "provider ABI is valid; run `omaspeak setup` and choose Full setup, or install a compatible model to complete the model-backed provider proof",
    );
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"candidate": candidate.backend, "probe": evidence, "applied": apply, "next": next})
        )?
    );
    Ok(())
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
        1 => guided_runtime(config_path, paths, selector).map(|_| ()),
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
    guided_full_setup_with_validator(
        config_path,
        paths,
        operations,
        selector,
        validate_runtime_configuration,
        install_launcher,
        service_is_active,
        reload_service,
        |config, paths| app_setup::print_checks(config, paths, false),
        app_setup::print_checks_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn guided_full_setup_with_validator(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
    validate_runtime: impl FnOnce(&Config, &Path, bool) -> Result<()>,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
    check_human: impl FnOnce(&Path, &AppPaths) -> Result<()>,
    check_json: impl FnOnce(&Path, &AppPaths) -> Result<()>,
) -> Result<()> {
    let Some((runtime, device, library_dir)) = choose_runtime(config_path, selector)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    let candidate =
        runtime_configuration_candidate(config_path, runtime, &device, library_dir.as_deref())?;
    validate_runtime(&candidate, config_path, library_dir.as_deref().is_some())?;
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
    let spec = operations.resolve(&model)?;
    let installed = operations.verify(paths, spec).is_ok();
    let Some(voice) = choose_voice(config_path, paths, spec, installed, selector)? else {
        println!("Setup cancelled.");
        return Ok(());
    };
    if !confirm_model_license(spec, installed, selector)? {
        println!("Setup cancelled; the model license was not accepted.");
        return Ok(());
    }
    let confirmation = [
        MenuItem::available(
            "Apply setup",
            "Download and activate the model, then install the desktop launcher. An active daemon restarts; an inactive service remains inactive.",
        ),
        MenuItem::available("Cancel", "Leave the current configuration unchanged."),
    ];
    let summary = format!(
        "Runtime: {} · Device: {} · Model: {} · Voice: {} (ID {}) · Service unit: unchanged (`omaspeak setup systemd` installs it)",
        runtime_name(runtime),
        device,
        model,
        voice.name,
        voice.id,
    );
    if selector.select("Apply Omaspeak setup", &summary, &confirmation, 0)? != Some(0) {
        println!("Setup cancelled; no changes were made.");
        return Ok(());
    }
    setup_all_with_config(
        candidate,
        config_path,
        paths,
        operations,
        &model,
        Some(voice.id),
        None,
        spec.requires_acceptance.then_some(spec.license),
        ProgressFormat::Human,
        install_launcher,
        service_is_active,
        reload_service,
        check_human,
        check_json,
    )
}

fn guided_runtime(
    config_path: &Path,
    paths: &AppPaths,
    selector: &mut impl SetupSelector,
) -> Result<Option<(Runtime, String)>> {
    let Some((runtime, device, library_dir)) = choose_runtime(config_path, selector)? else {
        println!("Runtime setup cancelled.");
        return Ok(None);
    };
    if runtime == Runtime::Openvino {
        let config = app_setup::load_config(config_path)?;
        let installed = omaspeak::catalog::model(&config.model.name)
            .is_some_and(|model| app_setup::model::verify(paths, model).is_ok());
        let compatible = !installed
            || omaspeak::catalog::model(&config.model.name).is_some_and(|model| {
                model.openvino_capable && (!device.eq_ignore_ascii_case("npu") || model.npu_capable)
            });
        if !compatible {
            bail!(
                "model {} is not compatible with direct OpenVINO on {device}; run `omaspeak setup` and choose Full setup to select a compatible Supertonic model",
                config.model.name
            );
        }
    }
    let confirmation = [
        MenuItem::available(
            "Apply runtime",
            "Save the runtime and device after their isolated probe passes.",
        ),
        MenuItem::available("Cancel", "Leave the current configuration unchanged."),
    ];
    let mut candidate =
        runtime_configuration_candidate(config_path, runtime, &device, library_dir.as_deref())?;
    let evidence = selector.probe_runtime(&candidate, config_path)?;
    let summary = format!(
        "Runtime: {} · Device: {device}\r\n{}",
        runtime_name(runtime),
        serde_json::to_string_pretty(&json!({
            "candidate": candidate.backend,
            "probe": evidence,
        }))?
    );
    if selector.select("Apply Omaspeak runtime", &summary, &confirmation, 0)? != Some(0) {
        println!("Runtime setup cancelled; no changes were made.");
        return Ok(None);
    }
    prepare_npu_for_runtime_selection(&mut candidate, paths, ProgressFormat::Human)?;
    let model_installed = active_model_is_installed(&candidate, paths);
    if model_installed {
        pin_audio_cpp_library(&mut candidate, config_path)?;
        prove_setup_synthesis(&candidate, paths)
            .context("runtime candidate rejected by model-backed synthesis; config unchanged")?;
    }
    candidate.save(config_path)?;
    if model_installed {
        println!(
            "Runtime provider configured: {} on {device}; file-only model proof passed.",
            runtime_name(runtime)
        );
    } else {
        println!(
            "Runtime provider configured: {} on {device}; provider ABI passed. Model setup must still run the file-only synthesis proof.",
            runtime_name(runtime)
        );
    }
    Ok(Some((runtime, device)))
}

fn choose_runtime(
    config_path: &Path,
    selector: &mut impl SetupSelector,
) -> Result<Option<(Runtime, String, Option<PathBuf>)>> {
    let config = app_setup::load_config(config_path)?;
    let locations = omaspeak::runtime::discover(&config.backend, config_path);
    // A stale configured provider path is reported by setup checks, but must
    // not prevent the user from opening the picker to replace it.
    let audio_cpp_library =
        omaspeak::audio_cpp::discover_provider_library(&config, config_path).unwrap_or(None);
    let runtimes = [
        (
            Runtime::Default,
            runtime_item(
                "audio.cpp · CPU",
                "Default packaged GGUF provider on CPU",
                audio_cpp_library.is_some(),
                "install the packaged provider or choose its installation directory",
            ),
        ),
        (
            Runtime::Openvino,
            runtime_item(
                "Direct OpenVINO · Intel",
                "Direct OpenVINO execution for compatible Supertonic models",
                locations.runtime_loadable.get("openvino") == Some(&true),
                &locations
                    .remediation(Runtime::Openvino)
                    .unwrap_or_else(|| "configure libopenvino_c and plugins.xml".into()),
            ),
        ),
        (
            Runtime::Cuda,
            runtime_item(
                "audio.cpp · CUDA",
                "External complete audio.cpp provider on NVIDIA GPU",
                audio_cpp_library.is_some() && config.backend.runtime == Runtime::Cuda,
                "choose a complete CUDA-enabled audio.cpp installation directory",
            ),
        ),
        (
            Runtime::Vulkan,
            runtime_item(
                "audio.cpp · Vulkan",
                "External complete audio.cpp provider on a Vulkan GPU",
                audio_cpp_library.is_some() && config.backend.runtime == Runtime::Vulkan,
                "choose a complete Vulkan-enabled audio.cpp installation directory",
            ),
        ),
        (
            Runtime::Hip,
            runtime_item(
                "audio.cpp · HIP/ROCm",
                "External complete audio.cpp provider on an AMD GPU",
                audio_cpp_library.is_some() && config.backend.runtime == Runtime::Hip,
                "choose a complete HIP-enabled audio.cpp installation directory",
            ),
        ),
    ];
    let items: Vec<_> = runtimes.iter().map(|(_, item)| item.clone()).collect();
    let preferred = runtimes
        .iter()
        .position(|(runtime, _)| *runtime == config.backend.runtime)
        .unwrap_or_default();
    let runtime_help = format!(
        "Choose one complete inference provider. Setup never installs optional vendor runtimes.\r\nConfigured paths: {}\r\nPackage paths: {}\r\nResolved audio.cpp: {}\r\nResolved OpenVINO: {}\r\nOpenVINO plugins: {}",
        setup_path_list(&locations.configured_library_dirs),
        setup_path_list(&locations.package_library_dirs),
        audio_cpp_library
            .as_deref()
            .map_or_else(|| "not found".to_owned(), |path| path.display().to_string()),
        locations
            .openvino_library
            .as_deref()
            .map_or_else(|| "not found".to_owned(), |path| path.display().to_string()),
        locations
            .openvino_plugins
            .as_deref()
            .map_or_else(|| "not found".to_owned(), |path| path.display().to_string())
    );
    let Some(selected) = selector.select("Omaspeak runtime", &runtime_help, &items, preferred)?
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
        &format!("Choose the device for {}.", runtime_name(runtime)),
        &items,
        preferred,
    )?
    else {
        return Ok(None);
    };
    let device = devices[selected].0.to_owned();
    let loadable = if runtime == Runtime::Openvino {
        locations.runtime_loadable.get("openvino") == Some(&true)
    } else {
        audio_cpp_library.is_some()
    };
    let library_dir = if loadable {
        None
    } else {
        let directory_help = if runtime == Runtime::Openvino {
            "Enter an absolute OpenVINO installation directory containing libopenvino_c and plugins.xml. Leave empty to use a runtime already available through configured, package, or system paths."
        } else if runtime != Runtime::Default {
            "Enter an absolute complete accelerator-enabled audio.cpp installation directory. Omaspeak does not install vendor runtimes or assemble provider plugins."
        } else {
            "Enter an absolute complete audio.cpp installation directory. Leave empty to use the packaged CPU provider."
        };
        let Some(value) = selector.input("Native runtime directory", directory_help)? else {
            return Ok(None);
        };
        (!value.is_empty()).then(|| PathBuf::from(value))
    };
    Ok(Some((runtime, device, library_dir)))
}

fn setup_path_list(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        "none".to_owned()
    } else {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    }
}

fn runtime_item(label: &str, detail: &str, available: bool, remediation: &str) -> MenuItem {
    if available {
        MenuItem::available(label, format!("Detected · {detail}"))
    } else {
        MenuItem::available(label, format!("Needs setup · {detail}; {remediation}"))
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
        Runtime::Vulkan => &[
            (
                "auto",
                "Recommended · select the first available Vulkan GPU",
            ),
            ("gpu", "Use Vulkan GPU execution explicitly"),
        ],
        Runtime::Hip => &[
            ("auto", "Recommended · select the first available AMD GPU"),
            ("gpu", "Use HIP/ROCm GPU execution explicitly"),
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

#[cfg(test)]
fn save_runtime_with(
    config_path: &Path,
    runtime: Runtime,
    device: &str,
    library_dir: Option<&Path>,
    validate: impl FnOnce(&Config, &Path, bool) -> Result<()>,
) -> Result<()> {
    let config = runtime_configuration_candidate(config_path, runtime, device, library_dir)?;
    validate(&config, config_path, library_dir.is_some())?;
    config.save(config_path)
}

fn runtime_configuration_candidate(
    config_path: &Path,
    runtime: Runtime,
    device: &str,
    library_dir: Option<&Path>,
) -> Result<Config> {
    let mut config = app_setup::load_config(config_path)?;
    let runtime_changed = config.backend.runtime != runtime;
    config.backend.runtime = runtime;
    config.backend.device = device.into();
    config.backend.kind = if runtime == Runtime::Openvino {
        "supertonic".into()
    } else {
        "audiocpp".into()
    };
    if runtime_changed {
        config.backend.options.clear();
    }
    if !matches!(runtime, Runtime::Cuda | Runtime::Vulkan | Runtime::Hip) {
        config.backend.device_id = 0;
    }
    if let Some(directory) = library_dir {
        apply_runtime_directory(&mut config, config_path, directory)?;
    }
    config.backend.validate_shape()?;
    if config.backend.kind == "audiocpp" {
        return Ok(config);
    }
    config.backend = omaspeak::runtime_inventory::resolve(&config.backend, config_path);
    Ok(config)
}

#[cfg(test)]
fn configure_runtime_directory(config_path: &Path, directory: &Path) -> Result<()> {
    configure_runtime_directory_with(config_path, directory, validate_runtime_configuration)
}

#[cfg(test)]
fn configure_runtime_directory_with(
    config_path: &Path,
    directory: &Path,
    validate: impl FnOnce(&Config, &Path, bool) -> Result<()>,
) -> Result<()> {
    let mut config = app_setup::load_config(config_path)?;
    apply_runtime_directory(&mut config, config_path, directory)?;
    validate(&config, config_path, true)?;
    config.save(config_path)
}

fn validate_runtime_configuration(
    config: &Config,
    config_path: &Path,
    _explicit_directory: bool,
) -> Result<()> {
    if config.backend.kind == "audiocpp" {
        omaspeak::audio_cpp::discover_provider_library(config, config_path)?
            .context("audio.cpp provider is not configured")?;
        return Ok(());
    }
    omaspeak::runtime_inventory::apply_with(
        config,
        config_path,
        false,
        omaspeak::runtime_inventory::probe,
    )
    .map(|_| ())
}

#[cfg(test)]
fn validate_runtime_report(
    runtime: Runtime,
    report: &omaspeak::runtime::LibraryPathReport,
) -> Result<()> {
    let name = runtime_name(runtime);
    if report.runtime_loadable.get(name) != Some(&true) {
        let detail = report
            .runtime_probe_errors
            .get(name)
            .cloned()
            .or_else(|| report.remediation(runtime))
            .unwrap_or_else(|| "runtime probe failed".into());
        bail!("{name} runtime validation failed; configuration was not changed: {detail}");
    }
    if report.runtime_device_accessible.get(name) == Some(&false) {
        let detail = report
            .device_probe_errors
            .get(name)
            .cloned()
            .unwrap_or_else(|| "selected device is not accessible".into());
        bail!("{name} device validation failed; configuration was not changed: {detail}");
    }
    Ok(())
}

fn apply_runtime_directory(
    config: &mut Config,
    config_path: &Path,
    directory: &Path,
) -> Result<()> {
    let base = config_path.parent().unwrap_or_else(|| Path::new("."));
    let directory = if directory.is_absolute() {
        directory.to_owned()
    } else {
        base.join(directory)
    };
    let directory = directory
        .canonicalize()
        .with_context(|| format!("resolve native runtime directory {}", directory.display()))?;
    if !directory.is_dir() {
        bail!(
            "native runtime path is not a directory: {}",
            directory.display()
        );
    }
    let mut candidates = vec![
        directory.clone(),
        directory.join("lib"),
        directory.join("lib64"),
        directory.join("runtime/lib/intel64"),
        directory.join("runtime/lib/intel64/Release"),
    ];
    candidates = candidates
        .into_iter()
        .filter(|path| path.is_dir())
        .filter_map(|path| path.canonicalize().ok())
        .fold(Vec::new(), |mut paths, path| {
            if !paths.contains(&path) {
                paths.push(path);
            }
            paths
        });
    if config.backend.kind == "audiocpp" {
        config.backend.library = Some(find_runtime_file(
            &candidates,
            &directory,
            "libaudiocpp.so",
        )?);
        config.backend.library_dirs = candidates
            .into_iter()
            .filter(|directory| directory_contains_shared_libraries(directory))
            .collect();
        return Ok(());
    }
    if config.backend.runtime == Runtime::Openvino {
        config.backend.openvino_library = Some(find_runtime_file(
            &candidates,
            &directory,
            "libopenvino_c.so",
        )?);
        config.backend.openvino_plugins = Some(find_openvino_plugins(&candidates, &directory)?);
    } else {
        config.backend.library = Some(find_runtime_file(
            &candidates,
            &directory,
            "libaudiocpp.so",
        )?);
    }
    let mut selected_parents = [
        config.backend.library.as_ref(),
        config.backend.openvino_library.as_ref(),
        config.backend.openvino_plugins.as_ref(),
    ]
    .into_iter()
    .flatten()
    .filter_map(|path| path.parent().map(Path::to_owned))
    .collect::<Vec<_>>();
    selected_parents.extend(
        candidates
            .into_iter()
            .filter(|directory| directory_contains_shared_libraries(directory)),
    );
    // Keep user-supplied vendor overlays (for example CUDA and cuDNN) when an
    // explicit directory pins the runtime core and provider libraries.
    let mut library_dirs = config.backend.library_dirs.clone();
    for directory in selected_parents {
        if !library_dirs.contains(&directory) {
            library_dirs.push(directory);
        }
    }
    config.backend.library_dirs = library_dirs;
    Ok(())
}

fn find_openvino_plugins(candidates: &[PathBuf], root: &Path) -> Result<PathBuf> {
    candidates
        .iter()
        .flat_map(|directory| {
            [
                directory.join("plugins.xml"),
                directory.join("openvino/plugins.xml"),
            ]
        })
        .find(|path| path.is_file())
        .with_context(|| format!("plugins.xml was not found below {}", root.display()))?
        .canonicalize()
        .with_context(|| format!("resolve OpenVINO plugins.xml below {}", root.display()))
}

fn find_runtime_file(candidates: &[PathBuf], root: &Path, name: &str) -> Result<PathBuf> {
    for directory in candidates {
        let direct = directory.join(name);
        if direct.is_file() {
            return direct
                .canonicalize()
                .with_context(|| format!("resolve native library {}", direct.display()));
        }
    }
    let prefix = format!("{name}.");
    let mut matches = candidates
        .iter()
        .flat_map(|directory| fs::read_dir(directory).into_iter().flatten().flatten())
        .map(|entry| entry.path())
        .filter_map(|path| {
            let version = path
                .is_file()
                .then(|| path.file_name()?.to_str()?.strip_prefix(&prefix))
                .flatten()?
                .split('.')
                .map(str::parse::<u64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok()?;
            (!version.is_empty()).then_some((version, path))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|(left, _), (right, _)| left.cmp(right));
    matches
        .pop()
        .map(|(_, path)| path)
        .with_context(|| format!("{name} was not found below {}", root.display()))?
        .canonicalize()
        .with_context(|| format!("resolve {name} below {}", root.display()))
}

fn directory_contains_shared_libraries(directory: &Path) -> bool {
    fs::read_dir(directory).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry.path().is_file()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.contains(".so"))
        })
    })
}

fn runtime_name(runtime: Runtime) -> &'static str {
    runtime.name()
}

fn guided_model(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    selector: &mut impl SetupSelector,
) -> Result<Option<String>> {
    let current = app_setup::load_config(config_path)?;
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
    let Some(voice) = choose_voice(config_path, paths, spec, installed, selector)? else {
        println!("Model setup cancelled.");
        return Ok(None);
    };
    if !confirm_model_license(spec, installed, selector)? {
        println!("Model setup cancelled; the model license was not accepted.");
        return Ok(None);
    }
    let directory = if installed {
        app_setup::model::model_directory(paths, spec)
    } else {
        operations.install(
            paths,
            spec,
            None,
            ProgressFormat::Human,
            spec.requires_acceptance.then_some(spec.license),
        )?
    };
    let mut config = app_setup::ensure_config(config_path)?;
    spec.activate(&mut config);
    config.model.voice = voice.id;
    prepare_npu_for_setup(&mut config, paths, ProgressFormat::Human)?;
    operations.prove(&mut config, paths)?;
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

fn confirm_model_license(
    spec: &omaspeak::catalog::ModelSpec,
    installed: bool,
    selector: &mut impl SetupSelector,
) -> Result<bool> {
    if installed || !spec.requires_acceptance {
        return Ok(true);
    }
    let items = [
        MenuItem::available(
            format!("Accept {} and continue", spec.license),
            "I have reviewed the model license and accept its use restrictions and distribution terms.",
        ),
        MenuItem::available("Cancel", "Do not download or install this model."),
    ];
    Ok(selector.select(
        "Model license",
        &format!(
            "{} is licensed under {}. Review the full terms at {}. Omaspeak stores the exact license and this acceptance beside the model.",
            spec.name, spec.license, spec.license_url
        ),
        &items,
        1,
    )? == Some(0))
}

fn choose_voice(
    config_path: &Path,
    paths: &AppPaths,
    spec: &omaspeak::catalog::ModelSpec,
    installed: bool,
    selector: &mut impl SetupSelector,
) -> Result<Option<omaspeak::voices::Voice>> {
    let current = app_setup::load_config(config_path)?;
    let mut candidate = current.clone();
    spec.activate(&mut candidate);
    let voices = if installed {
        omaspeak::voices::installed(&candidate, paths)?
    } else {
        omaspeak::voices::from_catalog(spec)
    };
    if voices.is_empty() {
        bail!("model {} does not expose any voices", spec.id);
    }
    let preferred_id = if current.model.name == spec.name {
        current.model.voice
    } else {
        voices[0].id
    };
    let preferred = voices
        .iter()
        .position(|voice| voice.id == preferred_id)
        .unwrap_or_default();
    let items = voices
        .iter()
        .map(|voice| {
            MenuItem::available(
                &voice.name,
                format!("Speaker ID {} for {}", voice.id, spec.name),
            )
        })
        .collect::<Vec<_>>();
    let selected = selector.select(
        "Omaspeak voice",
        "Choose the default speaker. `omaspeak say --voice ID` can override it per request.",
        &items,
        preferred,
    )?;
    Ok(selected.map(|index| voices[index].clone()))
}

fn choose_model(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    runtime: Option<(Runtime, &str)>,
    selector: &mut impl SetupSelector,
) -> Result<Option<String>> {
    let config = app_setup::load_config(config_path)?;
    let models = operations.models();
    let items: Vec<_> = models
        .iter()
        .map(|model| {
            let installed = operations.verify(paths, model).is_ok();
            let runtime_compatible = runtime.is_none_or(|(runtime, device)| {
                if runtime == Runtime::Openvino {
                    model.backend == "supertonic"
                        && model.openvino_capable
                        && (!device.eq_ignore_ascii_case("npu") || model.npu_capable)
                } else {
                    model.backend == "audiocpp"
                }
            });
            let selectable = installed || model.downloadable;
            let active = model.id == config.model.name;
            let status = if active && installed {
                "● active"
            } else if active && !model.downloadable {
                "● active · user-supplied archive required"
            } else if active && model.requires_acceptance {
                "● active · license acceptance required"
            } else if active {
                "● active · download required"
            } else if installed {
                "○ installed"
            } else if !model.downloadable {
                "· user-supplied only"
            } else if model.requires_acceptance {
                "· license acceptance required"
            } else {
                "· download"
            };
            let bytes = model.single_file.map_or(model.archive_size, |file| file.size)
                + model
                    .supplemental_files
                    .iter()
                    .map(|file| file.size)
                    .sum::<u64>();
            let npu = if model.npu_capable {
                " · Intel NPU compatible"
            } else {
                ""
            };
            let license = if model.requires_acceptance {
                format!(" · {} acceptance required", model.license)
            } else if !model.downloadable {
                format!(" · {}", model.license)
            } else {
                String::new()
            };
            let detail = format!(
                "{} · {} / {} · ~{} MiB{}{}",
                model.description,
                model.backend,
                model.family,
                bytes.div_ceil(1024 * 1024),
                license,
                npu
            );
            if runtime_compatible && selectable {
                MenuItem::available(format!("{}  {status}", model.id), detail)
            } else {
                MenuItem::unavailable(
                    format!("{}  {status}", model.id),
                    if !selectable {
                        format!("Use `omaspeak setup model --download {} --archive PATH` with a model archive you are licensed to use · {detail}", model.id)
                    } else {
                        format!("Incompatible with the selected provider/runtime · {detail}")
                    },
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
        "Choose a catalog model. Downloads begin only after any required license acceptance.",
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
    voice: Option<i32>,
    archive: Option<&Path>,
    accepted_license: Option<&str>,
    progress_format: ProgressFormat,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
) -> Result<()> {
    setup_all_with_validator(
        config_path,
        paths,
        operations,
        model,
        voice,
        archive,
        accepted_license,
        progress_format,
        install_launcher,
        service_is_active,
        reload_service,
        validate_runtime_configuration,
    )
}

#[allow(clippy::too_many_arguments)]
fn setup_all_with_validator(
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    model: &str,
    voice: Option<i32>,
    archive: Option<&Path>,
    accepted_license: Option<&str>,
    progress_format: ProgressFormat,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
    validate_runtime: impl FnOnce(&Config, &Path, bool) -> Result<()>,
) -> Result<()> {
    let mut config = app_setup::load_config(config_path)?;
    let spec = operations.resolve(model)?;
    spec.activate(&mut config);
    if spec.backend == "audiocpp" {
        config.backend.runtime = if config.backend.runtime.uses_audiocpp() {
            config.backend.runtime
        } else {
            Runtime::Default
        };
        config.backend.device = if matches!(
            config.backend.runtime,
            Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
        ) {
            "gpu".into()
        } else {
            "cpu".into()
        };
    }
    validate_runtime(&config, config_path, false)?;
    setup_all_with_config(
        config,
        config_path,
        paths,
        operations,
        model,
        voice,
        archive,
        accepted_license,
        progress_format,
        install_launcher,
        service_is_active,
        reload_service,
        |config, paths| app_setup::print_checks(config, paths, false),
        app_setup::print_checks_event,
    )
}

#[allow(clippy::too_many_arguments)]
fn setup_all_with_config(
    config: Config,
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    model: &str,
    voice: Option<i32>,
    archive: Option<&Path>,
    accepted_license: Option<&str>,
    progress_format: ProgressFormat,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
    check_human: impl FnOnce(&Path, &AppPaths) -> Result<()>,
    check_json: impl FnOnce(&Path, &AppPaths) -> Result<()>,
) -> Result<()> {
    setup_all_with_config_and_preparer(
        config,
        config_path,
        paths,
        operations,
        model,
        voice,
        archive,
        accepted_license,
        progress_format,
        install_launcher,
        service_is_active,
        reload_service,
        check_human,
        check_json,
        prepare_npu_for_setup,
    )
}

#[allow(clippy::too_many_arguments)]
fn setup_all_with_config_and_preparer(
    mut config: Config,
    config_path: &Path,
    paths: &AppPaths,
    operations: &impl ModelSetupOperations,
    model: &str,
    voice: Option<i32>,
    archive: Option<&Path>,
    accepted_license: Option<&str>,
    progress_format: ProgressFormat,
    install_launcher: impl FnOnce(&AppPaths) -> Result<PathBuf>,
    service_is_active: impl FnOnce() -> bool,
    reload_service: impl FnOnce(bool) -> Result<bool>,
    check_human: impl FnOnce(&Path, &AppPaths) -> Result<()>,
    check_json: impl FnOnce(&Path, &AppPaths) -> Result<()>,
    prepare_npu: impl FnOnce(&mut Config, &AppPaths, ProgressFormat) -> Result<()>,
) -> Result<()> {
    let spec = operations.resolve(model)?;
    if let Some(voice) = voice
        && !spec.voices.iter().any(|candidate| candidate.id == voice)
    {
        bail!("voice {voice} is unavailable for model {}", spec.id);
    }
    let original = config_snapshot(config_path)?;
    let result = (|| {
        let service_was_active = service_is_active();
        let directory =
            operations.install(paths, spec, archive, progress_format, accepted_license)?;
        activate_model_for_setup(spec, &mut config)?;
        if let Some(voice) = voice {
            config.model.voice = voice;
        }
        prepare_npu(&mut config, paths, progress_format)?;
        operations.prove(&mut config, paths)?;
        config.save(config_path)?;
        let launcher = install_launcher(paths)?;
        match progress_format {
            ProgressFormat::Human => check_human(config_path, paths)?,
            ProgressFormat::Json => check_json(config_path, paths)?,
        }
        let service_restarted = reload_service(service_was_active)?;
        print_setup_complete(
            &directory,
            config_path,
            paths,
            &launcher,
            service_was_active,
            service_restarted,
            progress_format,
        )
    })();
    if let Err(error) = result {
        if let Err(restore_error) = restore_config_snapshot(config_path, original.as_deref()) {
            return Err(error.context(format!(
                "setup also failed to restore the prior config: {restore_error:#}"
            )));
        }
        return Err(error);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_setup_complete(
    directory: &Path,
    config_path: &Path,
    paths: &AppPaths,
    launcher: &Path,
    service_was_active: bool,
    service_restarted: bool,
    progress_format: ProgressFormat,
) -> Result<()> {
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
            Ok(())
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
            Ok(())
        }
    }
}

fn config_snapshot(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("read config snapshot {}", path.display()))
        }
    }
}

fn restore_config_snapshot(path: &Path, bytes: Option<&[u8]>) -> Result<()> {
    let temporary = path.with_extension("toml.tmp");
    match bytes {
        Some(bytes) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&temporary, bytes)?;
            fs::rename(&temporary, path)?;
        }
        None => {
            if path.exists() {
                fs::remove_file(path)?;
            }
            if temporary.exists() {
                fs::remove_file(temporary)?;
            }
        }
    }
    Ok(())
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
    accepted_license: Option<String>,
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
        activate_model_for_setup(spec, &mut config)?;
        prepare_npu_for_setup(&mut config, paths, progress_format)?;
        operations.prove(&mut config, paths)?;
        config.save(config_path)?;
        println!("active model: {}", spec.id);
        return Ok(());
    }
    let selected = match download {
        Some(id) => Some(id),
        None => {
            print_models_with(paths, operations);
            println!(
                "Run `omaspeak setup model --download supertonic-3-gguf --accept-license OpenRAIL-M` to install the default model."
            );
            None
        }
    };
    if let Some(id) = selected {
        let spec = operations.resolve(&id)?;
        let directory = operations.install(
            paths,
            spec,
            archive.as_deref(),
            progress_format,
            accepted_license.as_deref(),
        )?;
        if !no_activate {
            let activation = (|| -> Result<()> {
                let mut config = app_setup::ensure_config(config_path)?;
                activate_model_for_setup(spec, &mut config)?;
                prepare_npu_for_setup(&mut config, paths, progress_format)?;
                operations.prove(&mut config, paths)?;
                config.save(config_path)
            })();
            activation.with_context(|| {
                format!(
                    "model installation succeeded at {}, but activation was not saved because its provider proof failed; configure a complete provider with `omaspeak setup runtime`, then run `omaspeak setup model --set {}`",
                    directory.display(),
                    spec.id
                )
            })?;
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
        } else if !model.downloadable {
            "user-supplied"
        } else if model.requires_acceptance {
            "acceptance-required"
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

fn activate_model_for_setup(
    spec: &omaspeak::catalog::ModelSpec,
    config: &mut Config,
) -> Result<()> {
    spec.activate(config);
    if spec.backend == "audiocpp" {
        if config.backend.runtime == Runtime::Openvino {
            config.backend.runtime = Runtime::Default;
        }
        config.backend.device = if matches!(
            config.backend.runtime,
            Runtime::Cuda | Runtime::Vulkan | Runtime::Hip
        ) {
            "gpu".into()
        } else {
            "cpu".into()
        };
    }
    Ok(())
}

fn active_model_is_installed(config: &Config, paths: &AppPaths) -> bool {
    if let Some(spec) = omaspeak::catalog::model(&config.model.name) {
        return spec.backend == config.backend.kind
            && app_setup::model::verify(paths, spec).is_ok();
    }
    if config.backend.kind == "audiocpp" {
        return !config.model.file.is_empty()
            && config
                .model_directory(paths)
                .join(&config.model.file)
                .is_file();
    }
    false
}

fn pin_audio_cpp_library(config: &mut Config, config_path: &Path) -> Result<()> {
    if config.backend.kind == "audiocpp" {
        config.backend.library = Some(
            omaspeak::audio_cpp::discover_provider_library(config, config_path)?.with_context(
                || "packaged audio.cpp provider was not found; configure backend.library first",
            )?,
        );
    }
    Ok(())
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
