//! Temporary, cancellable voice auditions. Never routes through the live daemon.
use super::*;
use omaspeak::setup::wizard::Preview;
use serde::Deserialize;

const SAMPLE: &str = "Hello! This is my voice. I can read your messages and notifications.";

#[derive(Serialize, Deserialize)]
struct Job {
    config: Config,
    paths: AppPaths,
    output: PathBuf,
}

pub(super) fn worker(request: &str) -> Result<()> {
    let job: Job = serde_json::from_str(request)?;
    #[cfg(not(test))]
    omaspeak::audio_cpp::disable_core_dumps()?;
    let engine = Engine::load(&job.config, &job.paths)?;
    engine.synthesize(SAMPLE, 1.0, job.config.model.voice, &job.output)?;
    Ok(())
}

pub(super) struct VoicePreview {
    config: Config,
    paths: AppPaths,
    voices: Vec<i32>,
    installed: bool,
    directory: Option<PathBuf>,
    child: Option<Child>,
    playing: bool,
    started: Instant,
}

impl VoicePreview {
    pub(super) fn new(config: Config, paths: AppPaths, voices: Vec<i32>, installed: bool) -> Self {
        Self {
            config,
            paths,
            voices,
            installed,
            directory: None,
            child: None,
            playing: false,
            started: Instant::now(),
        }
    }

    fn start(&mut self, index: usize) -> Result<()> {
        anyhow::ensure!(
            self.installed,
            "Install this model first to preview its voices."
        );
        let voice = *self.voices.get(index).context("unknown preview voice")?;
        let directory = self
            .paths
            .runtime_dir
            .join(format!("voice-preview-{}", request_id()));
        fs::create_dir_all(&self.paths.runtime_dir)?;
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(&directory)?;
        self.directory = Some(directory.clone());
        let mut config = self.config.clone();
        config.model.voice = voice;
        config.backend.fallback = Fallback::Error;
        let job = Job {
            config,
            paths: self.paths.clone(),
            output: directory.join("sample.wav"),
        };
        let mut command = ProcessCommand::new(std::env::current_exe()?);
        command
            .arg("__voice-preview")
            .arg(serde_json::to_string(&job)?);
        if job.config.backend.kind != "audiocpp" {
            let report = omaspeak::runtime::discover(&job.config.backend, &job.paths.config_file);
            if let Some(loader_path) = omaspeak::runtime::reexec_loader_path(&report)? {
                command.env("LD_LIBRARY_PATH", loader_path);
            }
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        use std::os::unix::process::CommandExt;
        command.process_group(0);
        configure_child_parent_death(&mut command);
        self.child = Some(command.spawn().context("start voice sample synthesis")?);
        self.playing = false;
        self.started = Instant::now();
        Ok(())
    }

    fn poll_inner(&mut self) -> Result<Option<String>> {
        self.poll_with(spawn_playback_worker)
    }

    fn poll_with(
        &mut self,
        mut spawn: impl FnMut(&Path) -> std::io::Result<Child>,
    ) -> Result<Option<String>> {
        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };
        if self.started.elapsed() > Duration::from_secs(120) {
            bail!("Voice sample timed out; check the selected runtime.");
        }
        let Some(status) = child.try_wait()? else {
            return Ok(None);
        };
        self.child = None;
        if !status.success() {
            bail!(
                "Voice sample failed; check the model, runtime and audio output with setup check."
            );
        }
        if self.playing {
            self.stop();
            return Ok(Some("Sample finished · Space to replay".into()));
        }
        let output = self
            .directory
            .as_ref()
            .context("missing sample directory")?
            .join("sample.wav");
        self.child = Some(spawn(&output).context("Start voice sample playback worker")?);
        self.playing = true;
        Ok(Some("Preparing playback · Space to stop".into()))
    }
}

fn spawn_playback_worker(output: &Path) -> std::io::Result<Child> {
    // Pause negotiation and playback stay outside the TUI. The existing preview
    // process-group cancellation also kills this worker and its player.
    let mut command = ProcessCommand::new(std::env::current_exe()?);
    command.arg("__voice-playback");
    command
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    use std::os::unix::process::CommandExt;
    command.process_group(0);
    configure_child_parent_death(&mut command);
    command.spawn()
}

impl Preview for VoicePreview {
    fn toggle(&mut self, index: usize) -> Result<String> {
        if self.child.is_some() {
            self.stop();
            return Ok("Sample stopped · Space to replay".into());
        }
        if let Err(error) = self.start(index) {
            self.stop();
            return Err(error);
        }
        Ok("Preparing sample · Space to stop".into())
    }

    fn poll(&mut self) -> Result<Option<String>> {
        let result = self.poll_inner();
        if result.is_err() {
            self.stop();
        }
        result
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // This process group belongs only to this audition, including any
            // native worker or player descendants. Never stop the live daemon.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(directory) = self.directory.take() {
            let _ = fs::remove_dir_all(directory);
        }
        self.playing = false;
    }
}

impl Drop for VoicePreview {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
#[path = "../tests/unit/voice_preview.rs"]
mod tests;
