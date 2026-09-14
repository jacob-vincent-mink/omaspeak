use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use omaspeak::backend::{Fallback, Runtime, compiled_capabilities};
use omaspeak::config::Config;
use omaspeak::engine::{Engine, Synthesis};
use omaspeak::paths::AppPaths;
use omaspeak::protocol::{Command, Request, Response, ResultPayload};
use omaspeak::setup as app_setup;
use omaspeak::setup::model::ProgressFormat;
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
    Stop,
    Voices {
        #[arg(long)]
        json: bool,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
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
    Check {
        #[arg(long)]
        json: bool,
    },
    All {
        #[arg(long, default_value = "en_US-lessac-medium")]
        model: String,
        #[arg(long)]
        archive: Option<PathBuf>,
        #[arg(long)]
        no_start: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    Model {
        #[arg(long)]
        list: bool,
        #[arg(long)]
        json: bool,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["set", "verify"])]
        download: Option<String>,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["download", "verify"])]
        set: Option<String>,
        #[arg(long, value_name = "MODEL", conflicts_with_all = ["download", "set"])]
        verify: Option<String>,
        #[arg(long, requires = "download")]
        archive: Option<PathBuf>,
        #[arg(long, requires = "download")]
        no_activate: bool,
        #[arg(long, value_enum, default_value_t)]
        progress_format: ProgressFormat,
    },
    Runtime {
        #[arg(long)]
        json: bool,
    },
    Systemd {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
        #[arg(long, conflicts_with_all = ["uninstall", "status"])]
        no_start: bool,
    },
    Menu {
        #[arg(long, conflicts_with = "status")]
        uninstall: bool,
        #[arg(long)]
        status: bool,
    },
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
    let paths = AppPaths::discover();
    let config_path = cli.config.unwrap_or_else(|| paths.config_file.clone());
    match cli.command {
        TopCommand::Daemon => run_daemon(&config_path, &paths),
        TopCommand::Status { json } => print_status(&config_path, &paths, json),
        TopCommand::Say(args) => say(&config_path, &paths, args),
        TopCommand::Stop => stop(&paths),
        TopCommand::Voices { json } => voices(&config_path, json),
        TopCommand::Config { command } => config_command(command, &config_path, &paths),
        TopCommand::Setup { command } => setup(command, &config_path, &paths),
    }
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
    let response = if paths.socket().exists() {
        send_request(&paths.socket(), &request)?
    } else {
        let engine = Engine::load(&config, paths)?;
        handle_request(&engine, &config, paths, request)
    };
    print_response(response)
}

fn run_daemon(config_path: &Path, paths: &AppPaths) -> Result<()> {
    let config = Config::load(config_path)?;
    let engine = Engine::load(&config, paths)?;
    fs::create_dir_all(&paths.runtime_dir)?;
    fs::set_permissions(&paths.runtime_dir, fs::Permissions::from_mode(0o700))?;
    fs::create_dir_all(&paths.state_dir)?;
    let socket = paths.socket();
    if socket.exists() {
        match UnixStream::connect(&socket) {
            Ok(_) => bail!("daemon is already running at {}", socket.display()),
            Err(_) => fs::remove_file(&socket).context("remove stale daemon socket")?,
        }
    }
    let listener = UnixListener::bind(&socket)
        .with_context(|| format!("bind daemon socket {}", socket.display()))?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    eprintln!(
        "omaspeak: ready model={} sample_rate={} load_ms={} socket={}",
        engine.model_name,
        engine.sample_rate,
        engine.load_time.as_millis(),
        socket.display()
    );

    let mut should_stop = false;
    while !should_stop {
        let (mut stream, _) = listener.accept().context("accept daemon request")?;
        let response = match read_request(&mut stream, config.daemon.max_text_bytes + 16_384) {
            Ok(request) => {
                should_stop = matches!(request.command, Command::Shutdown);
                handle_request(&engine, &config, paths, request)
            }
            Err(error) => Response::error("unknown", "invalid_request", error),
        };
        match serde_json::to_vec(&response) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                if let Err(error) = stream.write_all(&bytes) {
                    eprintln!("omaspeak: client disconnected before response: {error}");
                }
            }
            Err(error) => eprintln!("omaspeak: failed to encode response: {error}"),
        }
    }
    drop(listener);
    let _ = fs::remove_file(&socket);
    Ok(())
}

fn read_request(stream: &mut UnixStream, limit: usize) -> Result<Request> {
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

fn handle_request(
    engine: &Engine,
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

fn synthesis_payload(engine: &Engine, synthesis: Synthesis) -> ResultPayload {
    ResultPayload::Synthesis {
        output: synthesis.output.to_string_lossy().into_owned(),
        sample_rate: synthesis.sample_rate,
        samples: synthesis.samples,
        audio_seconds: synthesis.samples as f64 / synthesis.sample_rate as f64,
        load_milliseconds: engine.load_time.as_millis() as u64,
        synthesis_milliseconds: synthesis.synthesis_time.as_millis() as u64,
    }
}

fn status_payload(engine: &Engine, config: &Config) -> ResultPayload {
    ResultPayload::Status {
        running: true,
        pid: std::process::id(),
        model: engine.model_name.clone(),
        sample_rate: engine.sample_rate,
        backend: json!({
            "kind": engine.backend_kind,
            "requested": {"runtime": config.backend.runtime, "device": config.backend.canonical_device().unwrap_or_else(|_| config.backend.device.clone())},
            "effective": {"runtime": engine.effective_runtime, "device": if engine.effective_runtime == Runtime::Default { "cpu" } else { config.backend.device.as_str() }, "provider": if engine.effective_runtime == Runtime::Default { "CPUExecutionProvider" } else { "unverified" }},
            "compiled_capabilities": compiled_capabilities(),
            "fallback_policy": config.backend.fallback,
            "fallback_used": engine.fallback_used,
            "placement_verified": engine.effective_runtime == Runtime::Default,
            "evidence": if engine.effective_runtime == Runtime::Default { vec!["static CPU build"] } else { Vec::<&str>::new() }
        }),
    }
}

fn send_request(socket: &Path, request: &Request) -> Result<Response> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connect to daemon {}", socket.display()))?;
    serde_json::to_writer(&mut stream, request)?;
    stream.write_all(b"\n")?;
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    serde_json::from_str(&line).context("decode daemon response")
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
    let status = if paths.socket().exists() {
        let response = send_request(
            &paths.socket(),
            &Request {
                protocol: 1,
                id: request_id(),
                command: Command::Status,
            },
        )?;
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
    if !paths.socket().exists() {
        bail!("daemon is not running");
    }
    print_response(send_request(
        &paths.socket(),
        &Request {
            protocol: 1,
            id: request_id(),
            command: Command::Shutdown,
        },
    )?)
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
                    "backend.runtime\tdefault|openvino|cuda\nbackend.device\truntime-dependent\nmodel.family\tpiper|vits"
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
            set_config(&mut config, &key, &value)?;
            config.backend.validate_shape()?;
            save_config(path, &config)?;
        }
        ConfigCommand::Unset { key } => {
            let mut config = Config::load(path)?;
            unset_config(&mut config, &key)?;
            save_config(path, &config)?;
        }
    }
    Ok(())
}

fn set_config(config: &mut Config, key: &str, value: &str) -> Result<()> {
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
        "model.voice" => config.model.voice = value.parse()?,
        "audio.device" => config.audio.device = value.into(),
        "audio.volume" => config.audio.volume = value.parse()?,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
}

fn unset_config(config: &mut Config, key: &str) -> Result<()> {
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
        "model.voice" => config.model.voice = defaults.model.voice,
        "audio.device" => config.audio.device = defaults.audio.device,
        "audio.volume" => config.audio.volume = defaults.audio.volume,
        _ => bail!("unknown or unsupported config key {key}"),
    }
    Ok(())
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
            {"key":"model.family","type":"enum","section":"Model","label":"Family","description":"sherpa TTS model family","value":config.model.family,"file_value":null,"compiled":true,"restart_required":true,"choices":["piper","vits"]},
            {"key":"model.directory","type":"path","section":"Model","label":"Directory","description":"Model asset directory","value":config.model_directory(paths),"file_value":config.model.directory,"compiled":true,"restart_required":true}],
        "collections":[],"constraints":[{"kind":"matrix","keys":["backend.runtime","backend.device"],"rows":[{"backend.runtime":"default","backend.device":["auto","cpu"]},{"backend.runtime":"cuda","backend.device":["auto","gpu"]},{"backend.runtime":"openvino","backend.device":["auto","npu","gpu","cpu","auto:<devices>","hetero:<2+ devices>","multi:<2+ devices>"]}]}]}),
    )
}

fn setup(command: Option<SetupCommand>, config_path: &Path, paths: &AppPaths) -> Result<()> {
    match command.unwrap_or(SetupCommand::Check { json: false }) {
        SetupCommand::Check { json } => app_setup::print_checks(config_path, paths, json),
        SetupCommand::Runtime { json } => app_setup::print_runtime(json),
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
            if list || json {
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(omaspeak::catalog::models())?
                    );
                } else {
                    print_models(paths);
                }
                if download.is_none() && set.is_none() && verify.is_none() {
                    return Ok(());
                }
            }
            if let Some(id) = verify {
                let spec = model_spec(&id)?;
                app_setup::model::verify(paths, spec)?;
                println!(
                    "verified: {}",
                    app_setup::model::model_directory(paths, spec).display()
                );
                return Ok(());
            }
            if let Some(id) = set {
                let spec = model_spec(&id)?;
                app_setup::model::verify(paths, spec)?;
                let mut config = app_setup::ensure_config(config_path)?;
                spec.activate(&mut config);
                config.save(config_path)?;
                println!("active model: {}", spec.id);
                return Ok(());
            }
            let selected = match download {
                Some(id) => Some(id),
                None if !list && !json && std::io::stdin().is_terminal() => {
                    print_models(paths);
                    eprint!("Install en_US-lessac-medium? [Y/n] ");
                    std::io::stderr().flush()?;
                    let mut answer = String::new();
                    std::io::stdin().read_line(&mut answer)?;
                    if answer.trim().is_empty() || answer.trim().eq_ignore_ascii_case("y") {
                        Some("en_US-lessac-medium".into())
                    } else {
                        None
                    }
                }
                None => {
                    print_models(paths);
                    println!(
                        "Run `omaspeak setup model --download en_US-lessac-medium` to install the default model."
                    );
                    None
                }
            };
            if let Some(id) = selected {
                let spec = model_spec(&id)?;
                let directory =
                    app_setup::model::install(paths, spec, archive.as_deref(), progress_format)?;
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
            no_start,
            progress_format,
        } => {
            let spec = model_spec(&model)?;
            let mut config = app_setup::ensure_config(config_path)?;
            let directory =
                app_setup::model::install(paths, spec, archive.as_deref(), progress_format)?;
            spec.activate(&mut config);
            config.save(config_path)?;
            let launcher = app_setup::menu::install(paths)?;
            let service = app_setup::systemd::install(paths, config_path, !no_start)?;
            match progress_format {
                ProgressFormat::Human => {
                    println!(
                        "model: {}\nconfig: {}\nlauncher: {}\nservice: {}",
                        directory.display(),
                        config_path.display(),
                        launcher.display(),
                        service.display()
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
                            "service": service,
                        }))?
                    );
                    app_setup::print_checks_event(config_path, paths)
                }
            }
        }
    }
}

fn model_spec(id: &str) -> Result<&'static omaspeak::catalog::ModelSpec> {
    omaspeak::catalog::model(id)
        .ok_or_else(|| anyhow!("unknown model {id}; run `omaspeak setup model --list`"))
}

fn print_models(paths: &AppPaths) {
    for model in omaspeak::catalog::models() {
        let status = if app_setup::model::verify(paths, model).is_ok() {
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

fn play(path: &Path) -> Result<()> {
    for program in ["pw-play", "aplay"] {
        if ProcessCommand::new(program)
            .arg(path)
            .status()
            .is_ok_and(|status| status.success())
        {
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
