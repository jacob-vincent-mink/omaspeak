//! Cloud setup and explicit qualification commands. Ordinary checks never make requests.
use crate::{cloud_http, config::Config, paths::AppPaths};
use anyhow::{Result, ensure};
use clap::{Args, Subcommand};
use serde_json::json;
use std::{
    fs,
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

#[derive(Args)]
pub struct CloudOptions {
    #[arg(long)]
    pub provider: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub base_url: Option<String>,
    /// Environment variable name; never supply the API key here.
    #[arg(long)]
    pub api_key_env: Option<String>,
    /// Provider voice ID; omit for guided entry or use `cloud voices` to discover IDs.
    #[arg(long)]
    pub voice: Option<String>,
}
#[derive(Subcommand)]
pub enum CloudCommand {
    /// Store a key from stdin in a private file, or check client/daemon availability.
    Credential {
        #[command(subcommand)]
        command: CredentialCommand,
    },
    /// Fetch provider voice inventory; optionally save a selected ID as a local alias.
    Voices {
        #[arg(long)]
        json: bool,
        #[arg(long)]
        save: Option<String>,
        #[arg(long, requires = "save")]
        alias: Option<String>,
    },
    /// Make one explicit provider request (may incur charges); never runs actions.
    Smoke {
        #[arg(long, default_value = "Hello. Your cloud voice is ready.")]
        text: String,
        #[arg(long)]
        out: PathBuf,
        /// Generate and measure without playing the result.
        #[arg(long)]
        no_play: bool,
    },
}
#[derive(Subcommand)]
pub enum CredentialCommand {
    /// Read the API key from stdin, save mode 0600, and configure its file reference.
    Install {
        #[arg(long, required = true)]
        stdin: bool,
    },
    /// Check availability without displaying the key or contacting the provider.
    Check {
        #[arg(long)]
        daemon: bool,
    },
}
fn key_env(config: &Config) -> &str {
    if !config.backend.cloud.api_key_env.is_empty() {
        return &config.backend.cloud.api_key_env;
    }
    match config.backend.kind.as_str() {
        "elevenlabs" => "ELEVENLABS_API_KEY",
        "cartesia" => "CARTESIA_API_KEY",
        "deepgram" => "DEEPGRAM_API_KEY",
        _ => "OPENAI_API_KEY",
    }
}
fn prompt(label: &str, default: &str) -> Result<String> {
    prompt_io(&mut io::stdin().lock(), &mut io::stderr(), label, default)
}
fn prompt_io(
    input: &mut dyn io::BufRead,
    output: &mut dyn Write,
    label: &str,
    default: &str,
) -> Result<String> {
    write!(output, "{label} [{default}]: ")?;
    output.flush()?;
    let mut value = String::new();
    ensure!(
        input.read_line(&mut value)? > 0,
        "cloud setup input closed; configuration unchanged"
    );
    let input = value;
    let value = input.trim();
    Ok(if value.is_empty() {
        default.to_owned()
    } else {
        value.to_owned()
    })
}
pub fn configure(options: CloudOptions, path: &Path) -> Result<()> {
    ensure!(
        options.provider.is_some() || io::stdin().is_terminal(),
        "use --provider for unattended cloud setup"
    );
    configure_with(options, path, &mut prompt)
}
fn configure_with(
    options: CloudOptions,
    path: &Path,
    prompt: &mut dyn FnMut(&str, &str) -> Result<String>,
) -> Result<()> {
    let guided = options.provider.is_none();
    let provider = if let Some(provider) = options.provider {
        provider
    } else {
        eprintln!("Cloud providers: elevenlabs, openai-compatible, cartesia, deepgram");
        prompt("Provider", "deepgram")?
    };
    ensure!(
        crate::cloud::is_cloud(&provider),
        "unsupported cloud provider"
    );
    let mut config = crate::setup::load_config(path)?;
    if config.backend.kind != provider {
        config.backend.cloud = Default::default();
    }
    config.backend.kind = provider;
    config.backend.runtime = crate::backend::Runtime::Default;
    config.backend.device = "remote".into();
    config.backend.device_id = 0;
    config.backend.fallback = crate::backend::Fallback::Error;
    config.backend.options.clear();
    if let Some(url) = options.base_url {
        config.backend.cloud.base_url = url;
    }
    if let Some(env) = options.api_key_env {
        config.backend.cloud.api_key_env = env;
    }
    if let Some(model) = options.model {
        config.backend.cloud.model = model;
    } else if guided {
        config.backend.cloud.model = prompt(
            "Model (blank uses provider default)",
            &config.backend.cloud.model,
        )?;
    }
    config.model.options.clear();
    config.model.voice = 0.into();
    if let Some(voice) = options.voice {
        config.backend.cloud.voice = voice;
        config.backend.cloud.voices.clear();
    } else if guided && matches!(config.backend.kind.as_str(), "elevenlabs" | "cartesia") {
        config.backend.cloud.voice = prompt(
            "Voice ID (discover with `omaspeak cloud voices`)",
            &config.backend.cloud.voice,
        )?;
    }
    if !matches!(config.backend.kind.as_str(), "elevenlabs" | "cartesia")
        || !config.backend.cloud.voice.is_empty()
        || !config.backend.cloud.voices.is_empty()
    {
        crate::cloud::voices(&config)?;
    }
    // Endpoint and placement validation is offline; keys and paid requests are optional here.
    cloud_http::base_url(&config.backend.cloud, "https://example.com")?;
    cloud_http::credential(&config.backend.cloud, key_env(&config), false)?;
    config.save(path)?;
    eprintln!(
        "Cloud configuration saved. Use `omaspeak cloud credential install --stdin` for a key file shared by the CLI and daemon. Use `omaspeak cloud voices --save ID --alias NAME` to select an account voice. Restart an already-running daemon to apply configuration changes."
    );
    Ok(())
}
fn credential(command: CredentialCommand, path: &Path) -> Result<()> {
    let mut config = Config::load(path)?;
    ensure!(
        crate::cloud::is_cloud(&config.backend.kind),
        "select a cloud provider first"
    );
    match command {
        CredentialCommand::Install { .. } => {
            ensure!(
                !io::stdin().is_terminal(),
                "pipe the key through stdin; do not put it in command arguments"
            );
            let mut key = String::new();
            io::stdin().take(4097).read_to_string(&mut key)?;
            ensure!(key.len() <= 4096, "cloud key exceeds limit");
            let key = key.trim_end_matches(['\r', '\n']);
            ensure!(
                !key.is_empty() && !key.contains(['\r', '\n', '\0']),
                "invalid cloud key"
            );
            let directory = path.parent().unwrap_or(Path::new("."));
            fs::create_dir_all(directory)?;
            let target = directory.join("omaspeak-cloud-api-key");
            let temporary = directory.join(format!(".omaspeak-key-{}", std::process::id()));
            use std::os::unix::fs::OpenOptionsExt;
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&temporary)?;
            let result = (|| {
                file.write_all(key.as_bytes())?;
                file.sync_all()?;
                fs::rename(&temporary, &target)?;
                Ok::<_, anyhow::Error>(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            result?;
            config.backend.cloud.api_key_file =
                target.canonicalize()?.to_string_lossy().into_owned();
            config.save(path)?;
            eprintln!(
                "Private key file configured. CLI and daemon read it directly; restart an already-running daemon to load the new configuration."
            );
        }
        CredentialCommand::Check { daemon } => {
            let available = if !config.backend.cloud.api_key_file.is_empty() {
                cloud_http::read_private_key(Path::new(&config.backend.cloud.api_key_file)).is_ok()
            } else {
                false
            };
            let mut pid = None;
            let available = if daemon {
                let output = Command::new("systemctl")
                    .args([
                        "--user",
                        "show",
                        "omaspeak.service",
                        "--property=MainPID",
                        "--value",
                    ])
                    .output()?;
                ensure!(output.status.success(), "cannot inspect daemon service");
                let value = String::from_utf8(output.stdout)?.trim().parse::<u32>()?;
                ensure!(value > 0, "daemon is not running");
                pid = Some(value);
                ensure!(
                    daemon_targets_config(value, path)?,
                    "daemon service uses a different configuration or is not running the daemon command"
                );
                let environment = fs::read(format!("/proc/{value}/environ"))?;
                let prefix = format!("{}=", key_env(&config));
                available
                    || environment
                        .split(|b| *b == 0)
                        .any(|v| v.starts_with(prefix.as_bytes()) && v.len() > prefix.len())
            } else {
                cloud_http::credential(&config.backend.cloud, key_env(&config), false)?.is_some()
            };
            println!(
                "{}",
                json!({"provider":config.backend.kind,"credential_available":available,"daemon":daemon,"pid":pid,"config_restart_required":daemon,"note":if daemon {"File check uses the saved config; restart the daemon after config changes."} else {"No provider request was made."}})
            );
            ensure!(available, "cloud credential unavailable");
        }
    }
    Ok(())
}
pub fn run(command: CloudCommand, path: &Path, _paths: &AppPaths) -> Result<Option<PathBuf>> {
    match command {
        CloudCommand::Credential { command } => {
            credential(command, path)?;
            Ok(None)
        }
        CloudCommand::Voices { json, save, alias } => {
            let mut config = Config::load(path)?;
            let voices = crate::cloud::discover_voices(&config)?;
            if let Some(id) = save {
                let selected = voices.iter().find(|v| v.id == id).ok_or_else(|| {
                    anyhow::anyhow!("selected voice ID was not in the provider inventory")
                })?;
                let name = alias.unwrap_or_else(|| selected.name.clone());
                ensure!(
                    !name.trim().is_empty() && name.len() <= 256,
                    "invalid voice alias"
                );
                config.backend.cloud.voice.clear();
                config.backend.cloud.voices.insert(name.clone(), id);
                config.model.voice = crate::voices::VoiceSelection::Name(name);
                crate::cloud::voices(&config)?;
                config.save(path)?;
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&voices)?);
            } else {
                for voice in voices {
                    println!(
                        "{}\t{}",
                        voice.id,
                        voice.name.replace(['\r', '\n', '\t'], " ")
                    );
                }
            }
            Ok(None)
        }
        CloudCommand::Smoke { text, out, no_play } => {
            use crate::engine::TtsBackend;
            let config = Config::load(path)?;
            ensure!(
                crate::cloud::is_cloud(&config.backend.kind),
                "smoke test requires a cloud provider"
            );
            ensure!(
                !out.exists(),
                "smoke output already exists; choose a new file"
            );
            ensure!(
                !text.trim().is_empty() && text.chars().count() <= 500,
                "smoke text must contain 1..500 characters"
            );
            let backend = crate::cloud::CloudBackend::create(&config)?;
            let inventory = crate::cloud::voices(&config)?;
            let voice = config.model.voice.resolve(&inventory)?;
            let staged = out.with_extension(format!("smoke-{}.tmp", std::process::id()));
            let file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staged)?;
            let result = (|| {
                let mut wav = hound::WavWriter::new(
                    file,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 24000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )?;
                let started = Instant::now();
                let mut first = None;
                let mut samples = 0u64;
                backend.generate_stream(&text, 1.0, voice, &mut |pcm| {
                    if !pcm.is_empty() {
                        first.get_or_insert(started.elapsed());
                    }
                    for sample in pcm {
                        wav.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                    }
                    samples += pcm.len() as u64;
                    Ok(())
                })?;
                ensure!(samples > 0, "cloud provider returned no audio");
                wav.finalize()?;
                // Hard-link provides no-clobber publication even if another process creates out.
                fs::hard_link(&staged, &out)?;
                println!(
                    "{}",
                    json!({"provider":config.backend.kind,"model":crate::cloud::model_name(&config),"first_audio_ms":first.map(|d|d.as_secs_f64()*1000.0),"total_ms":started.elapsed().as_secs_f64()*1000.0,"audio_seconds":samples as f64/24000.0,"output":out})
                );
                Ok::<_, anyhow::Error>(())
            })();
            let _ = fs::remove_file(staged);
            result?;
            Ok((!no_play).then_some(out))
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/cloud_cli.rs"]
mod tests;

fn daemon_targets_config(pid: u32, path: &Path) -> Result<bool> {
    let cmdline = fs::read(format!("/proc/{pid}/cmdline"))?;
    let args: Vec<_> = cmdline
        .split(|b| *b == 0)
        .filter(|arg| !arg.is_empty())
        .collect();
    if !args.contains(&b"daemon".as_slice()) {
        return Ok(false);
    }
    let selected = if let Some(pair) = args.windows(2).find(|pair| pair[0] == b"--config") {
        use std::os::unix::ffi::OsStrExt;
        let selected = PathBuf::from(std::ffi::OsStr::from_bytes(pair[1]));
        if selected.is_absolute() {
            selected
        } else {
            fs::read_link(format!("/proc/{pid}/cwd"))?.join(selected)
        }
    } else {
        AppPaths::discover().config_file
    };
    Ok(selected.canonicalize()? == path.canonicalize()?)
}
